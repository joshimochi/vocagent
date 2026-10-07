"""Local MiniMax H3 text-to-video with native audio and GPU-only weights."""

from dataclasses import dataclass
import math
import logging
from pathlib import Path
import secrets
import tempfile
import types
from typing import ClassVar

from .benchmark import current_step, model_phase
from .providers import VideoGenerator
from .gpu_boundaries import align_h3_residual_outputs


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class H3Settings:
    """Explicit landscape profiles with fixed duration and configurable sigma points."""

    seconds: ClassVar[float] = 15
    steps: int = 50
    profile: str = "quality"

    def __post_init__(self):
        if not isinstance(self.profile, str) or self.profile not in ("quality", "preview"):
            raise ValueError("H3 profile must be 'quality' or 'preview'.")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 2:
            raise ValueError("H3 requires an integer number of at least two sigma points.")

    @property
    def width(self):
        return 960 if self.profile == "preview" else 1344

    @property
    def height(self):
        return 544 if self.profile == "preview" else 768

    @property
    def num_frames(self):
        # The video VAE emits 17*n+5 frames. Stay inside its 15-second limit.
        return min(345, 17 * math.ceil((round(self.seconds * 24) - 5) / 17) + 5)


def _patch_pointwise_conv3d(module):
    """S4000 has no Conv3d kernel; 1x1x1 convolution is exactly a linear map."""
    import torch
    for child in module.modules():
        if not isinstance(child, torch.nn.Conv3d):
            continue
        if (child.kernel_size, child.stride, child.padding, child.dilation, child.groups) != (
            (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1
        ):
            continue

        def pointwise(self, tensor):
            return torch.nn.functional.linear(
                tensor.movedim(1, -1), self.weight[:, :, 0, 0, 0], self.bias
            ).movedim(-1, 1).contiguous()

        child.forward = types.MethodType(pointwise, child)


def _patch_audio_convolutions(module):
    """Express unsupported MUSA 1-D convolutions using resident GPU matmuls."""
    import torch

    def conv(self, tensor):
        stride, padding, dilation = self.stride[0], self.padding[0], self.dilation[0]
        kernel = self.kernel_size[0]
        if padding:
            tensor = torch.nn.functional.pad(tensor, (padding, padding), mode="constant")
        windows = tensor.unfold(-1, dilation * (kernel - 1) + 1, stride)[..., ::dilation]
        # The last two axes must flatten in channel-then-kernel order.
        windows = windows.permute(0, 2, 1, 3)
        batch, length = windows.shape[:2]
        windows = windows.reshape(batch, length, self.groups, -1)
        weights = self.weight.reshape(self.groups, self.out_channels // self.groups, -1)
        output = torch.einsum("blgi,goi->bgol", windows, weights).reshape(batch, self.out_channels, length)
        if self.bias is not None:
            output = output + self.bias[None, :, None]
        return output.contiguous()

    def transpose(self, tensor, output_size=None):
        if output_size is not None:
            raise ValueError("H3 audio convolution does not use output_size overrides.")
        stride, padding, dilation = self.stride[0], self.padding[0], self.dilation[0]
        kernel, length = self.kernel_size[0], tensor.shape[-1]
        target_length = (length - 1) * stride - 2 * padding + dilation * (kernel - 1) + self.output_padding[0] + 1
        output = tensor.new_zeros((tensor.shape[0], self.out_channels, target_length))
        grouped = tensor.reshape(tensor.shape[0], self.groups, self.in_channels // self.groups, length)
        weights = self.weight.reshape(self.groups, self.in_channels // self.groups, self.out_channels // self.groups, kernel)
        for index in range(kernel):
            offset = index * dilation - padding
            first = max(0, math.ceil(-offset / stride))
            last = min(length, math.ceil((target_length - offset) / stride))
            if last <= first:
                continue
            projected = torch.einsum("bgil,gio->bgol", grouped[..., first:last], weights[..., index])
            start = offset + first * stride
            stop = offset + last * stride
            output[:, :, start:stop:stride] += projected.reshape(tensor.shape[0], self.out_channels, last - first)
        if self.bias is not None:
            output = output + self.bias[None, :, None]
        return output

    def filter_config(module, channels):
        return types.SimpleNamespace(
            stride=(module.stride,), padding=(0,), dilation=(1,), output_padding=(0,),
            kernel_size=(module.filter.shape[-1],), groups=channels,
            in_channels=channels, out_channels=channels,
            weight=module.filter.expand(channels, -1, -1), bias=None,
        )

    def lowpass(self, tensor):
        tensor = torch.nn.functional.pad(tensor, (self.pad_left, self.pad_right), mode="replicate")
        return conv(filter_config(self, tensor.shape[1]), tensor)

    def upsample(self, tensor):
        tensor = torch.nn.functional.pad(tensor, (self.pad, self.pad), mode="replicate")
        output = self.ratio * transpose(filter_config(self, tensor.shape[1]), tensor)
        return output[..., self.pad_left:-self.pad_right]

    for child in module.modules():
        if isinstance(child, torch.nn.Conv1d):
            if child.padding_mode != "zeros" or isinstance(child.padding, str):
                raise ValueError("Unsupported audio convolution padding mode.")
            child.forward = types.MethodType(conv, child)
        elif isinstance(child, torch.nn.ConvTranspose1d):
            child.forward = types.MethodType(transpose, child)
        elif type(child).__name__ == "MiniMaxH3AudioLowPassFilter1d":
            child.forward = types.MethodType(lowpass, child)
        elif type(child).__name__ == "MiniMaxH3AudioUpSample1d":
            child.forward = types.MethodType(upsample, child)


def _chunk_h3_rotary_emb(hidden_states, rotary_emb, apply_rotary, chunk_size=8192):
    """Apply the released RoPE to token slices with GPU-only scratch space.

    H3's cosine/sine arrays have shape (tokens, rotary_dim), while query/key
    tensors have shape (batch, tokens, heads, head_dim). Each token rotates
    independently; attention still receives the complete rotated sequence.
    """
    if chunk_size < 1:
        raise ValueError("H3 rotary chunk size must be positive.")
    if hidden_states.shape[1] <= chunk_size:
        return apply_rotary(hidden_states, *rotary_emb)
    cosine, sine = rotary_emb
    output = hidden_states.new_empty(hidden_states.shape)
    for start in range(0, hidden_states.shape[1], chunk_size):
        stop = min(start + chunk_size, hidden_states.shape[1])
        rotated = apply_rotary(hidden_states[:, start:stop], cosine[start:stop], sine[start:stop])
        output[:, start:stop] = rotated
        del rotated
    return output


def _patch_musa_attention(model):
    """Use torch-musa's SDPA kernel with the released H3 attention math."""
    import torch
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Attention, MiniMaxH3AttnProcessor, _apply_rotary_emb,
    )

    class MUSAAttention(MiniMaxH3AttnProcessor):
        def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
            if attn.fused_projections:
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                query, key, value = (projection(hidden_states) for projection in (
                    attn.to_q, attn.to_k, attn.to_v
                ))
            query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
            key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
            value = value.unflatten(-1, (attn.heads, -1))
            if rotary_emb is not None:
                query = _chunk_h3_rotary_emb(query, rotary_emb, _apply_rotary_emb)
                key = _chunk_h3_rotary_emb(key, rotary_emb, _apply_rotary_emb)
            # Replacing each source releases its untransposed storage before
            # SDPA allocates scratch space for the full packed sequence.
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()
            output_dtype = query.dtype
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
                dropout_p=0.0, is_causal=False,
            )
            del query, key, value
            output = output.transpose(1, 2).flatten(2, 3).to(dtype=output_dtype)
            return attn.to_out[1](attn.to_out[0](output))

    for module in model.modules():
        if isinstance(module, MiniMaxH3Attention):
            module.set_processor(MUSAAttention())


def _chunk_h3_feed_forwards(model, chunk_size=8192):
    """Bound native-resolution SwiGLU scratch space without changing its math.

    H3 feed-forwards operate independently on each token. Keep all weights and
    activations on their assigned GPUs, but project a limited number of tokens
    at a time. Attention still sees the entire packed video/audio sequence.
    """
    if chunk_size < 1:
        raise ValueError("H3 feed-forward chunk size must be positive.")

    def wrap(original):
        def forward(self, hidden_states, *args, **kwargs):
            if hidden_states.shape[1] <= chunk_size:
                return original(hidden_states, *args, **kwargs)
            output = None
            offset = 0
            for chunk in hidden_states.split(chunk_size, dim=1):
                projected = original(chunk, *args, **kwargs)
                if output is None:
                    output = projected.new_empty((
                        projected.shape[0], hidden_states.shape[1], *projected.shape[2:],
                    ))
                length = projected.shape[1]
                output[:, offset:offset + length] = projected
                offset += length
                del projected
            return output
        return forward

    count = 0
    for block in model.modules():
        if type(block).__name__ not in ("MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock"):
            continue
        if getattr(block.ff, "_vocagent_chunk_size", None) == chunk_size:
            continue
        if getattr(block.ff, "_vocagent_chunk_size", None) is not None:
            raise ValueError("H3 feed-forward chunking is already configured.")
        block.ff.forward = types.MethodType(wrap(block.ff.forward), block.ff)
        block.ff._vocagent_chunk_size = chunk_size
        count += 1
    log.info("Chunked %s H3 feed-forwards at %s tokens per projection.", count, chunk_size)


def _chunk_h3_adaln_blocks(model, chunk_size=8192):
    """Bound packed AdaLN and residual scratch without changing block math.

    Scale, shift and gate tables are selected independently for each token.
    Normalize/modulate and gated-add bounded slices, then pass the complete
    normalized sequence to the original attention and feed-forward modules.
    """
    if chunk_size < 1:
        raise ValueError("H3 AdaLN chunk size must be positive.")

    def tokenwise(hidden_states, adaln_indices, operation):
        output = None
        for start in range(0, hidden_states.shape[1], chunk_size):
            stop = min(start + chunk_size, hidden_states.shape[1])
            projected = operation(start, stop, adaln_indices[start:stop])
            if output is None:
                output = projected.new_empty((
                    projected.shape[0], hidden_states.shape[1], *projected.shape[2:],
                ))
            output[:, start:stop] = projected
            del projected
        return output

    def modulated_norm(norm, hidden_states, scale, shift, adaln_indices):
        return tokenwise(hidden_states, adaln_indices, lambda start, stop, indices:
            norm(hidden_states[:, start:stop]) * (
                1.0 + scale.index_select(0, indices)
            ) + shift.index_select(0, indices))

    def gated_residual(hidden_states, branch, gate, adaln_indices):
        return tokenwise(hidden_states, adaln_indices, lambda start, stop, indices:
            hidden_states[:, start:stop] + gate.index_select(0, indices) * branch[:, start:stop])

    def wrap(original):
        def forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
            if hidden_states.shape[1] <= chunk_size:
                return original(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
            norm_hidden_states = modulated_norm(
                self.norm1, hidden_states, scale_msa, shift_msa, adaln_indices,
            )
            attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
            del norm_hidden_states
            hidden_states = gated_residual(hidden_states, attn_output, gate_msa, adaln_indices)
            del attn_output, shift_msa, scale_msa, gate_msa
            norm_hidden_states = modulated_norm(
                self.norm2, hidden_states, scale_mlp, shift_mlp, adaln_indices,
            )
            ff_output = self.ff(norm_hidden_states)
            del norm_hidden_states
            return gated_residual(hidden_states, ff_output, gate_mlp, adaln_indices)
        return forward

    count = 0
    for block in model.modules():
        if type(block).__name__ != "MiniMaxH3TransformerBlock":
            continue
        configured = getattr(block, "_vocagent_adaln_chunk_size", None)
        if configured == chunk_size:
            continue
        if configured is not None:
            raise ValueError("H3 AdaLN chunking is already configured.")
        block.forward = types.MethodType(wrap(block.forward), block)
        block._vocagent_adaln_chunk_size = chunk_size
        count += 1
    log.info("Chunked %s H3 AdaLN blocks at %s tokens per modulation.", count, chunk_size)


def _dispatch_resident(model, runtime):
    """Shard whole H3 blocks across GPUs; never add CPU or disk to the map."""
    from accelerate import dispatch_model, infer_auto_device_map
    accelerator = getattr(runtime.torch, runtime.backend)
    budgets = {}
    for index in range(accelerator.device_count()):
        free, total = accelerator.mem_get_info(index)
        # The encoder, transformer and VAEs coexist. Budget their combined
        # residency rather than filling each new model up to a fresh GPU cap.
        # Native-resolution Q/K normalization and MLPs need this extra room.
        resident = accelerator.memory_allocated(index)
        budgets[index] = max(0, min(int(total * .75) - resident, free - 8 * 1024**3))
    budgets["cpu"] = 0
    mapping = infer_auto_device_map(
        model, max_memory=budgets,
        no_split_module_classes=list(getattr(model, "_no_split_modules", []) or [])
        + ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefiner", "MiniMaxH3AdaLayerNorm"],
    )
    if any(str(device) in ("cpu", "disk", "meta") for device in mapping.values()):
        raise RuntimeError("The full H3 weights cannot fit on the available GPUs without offload.")
    mapping = {name: f"{runtime.backend}:{device}" if isinstance(device, int) else device
               for name, device in mapping.items()}
    model = dispatch_model(model, device_map=mapping, main_device=runtime.device)
    runtime.assert_on_accelerator(model)
    residency = {}
    for parameter in model.parameters():
        key = str(parameter.device)
        residency[key] = residency.get(key, 0) + parameter.numel() * parameter.element_size()
    log.info("%s GPU-resident weights (GiB): %s", type(model).__name__,
             {device: round(size / 1024**3, 2) for device, size in residency.items()})
    return model


def _gpu_decode_boundary(vae, device, output_device, *, autocast=False):
    """Move activations between GPU components; model weights never move."""
    original = vae.decode

    def decode(self, latents, *args, **kwargs):
        import torch
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=autocast):
            result = original(latents.to(device=device, dtype=self.dtype), *args, **kwargs)
        sample = result[0] if isinstance(result, tuple) and result else getattr(result, "sample", None)
        if not torch.is_tensor(sample) or sample.numel() == 0:
            raise RuntimeError("H3 decoder produced empty or nonfinite output.")
        if sample.device != torch.device(device):
            raise RuntimeError("H3 decoder output left its configured GPU.")
        if not torch.isfinite(sample).all().item():
            raise RuntimeError("H3 decoder produced empty or nonfinite output.")
        if isinstance(result, tuple):
            return (sample.to(output_device), *result[1:])
        result.sample = sample.to(output_device)
        return result

    vae.decode = types.MethodType(decode, vae)


def _patch_h3_video_decode_device(pipe, device):
    """Run the released video decode/postprocess step on the VAE's GPU.

    The original step creates latent and pixel constants on its execution
    device. Override that device only for this block and move its input there;
    later blocks retain the pipeline's execution device and original latents.
    """
    import torch
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep

    device = torch.device(device)

    class DecodeComponents:
        def __init__(self, components):
            self._source = components

        @property
        def _execution_device(self):
            return device

        def __getattr__(self, name):
            return getattr(self._source, name)

    def patch_block(block):
        configured = getattr(block, "_vocagent_decode_device", None)
        if configured is not None:
            if configured != str(device):
                raise ValueError("H3 video decode device is already configured.")
            return
        original_call = type(block).__call__

        class GPUVideoDecode(type(block)):
            _vocagent_decode_device = str(device)

            def __call__(self, components, state):
                original_latents = state.get("latents")
                state.set("latents", original_latents.to(device))
                try:
                    _, result = original_call(self, DecodeComponents(components), state)
                finally:
                    state.set("latents", original_latents)
                return components, result

        block.__class__ = GPUVideoDecode

    found = 0
    # The public blocks property returns a deepcopy. Patch the registry that
    # ModularPipeline actually executes, rather than its inspection snapshot.
    pending = [pipe._blocks]
    while pending:
        block = pending.pop()
        if isinstance(block, MiniMaxH3VideoDecodeStep):
            patch_block(block)
            found += 1
        pending.extend((getattr(block, "sub_blocks", None) or {}).values())
    if found != 1:
        raise RuntimeError("H3 requires exactly one video decode block for GPU placement.")


class MiniMaxH3VideoGenerator(VideoGenerator):
    def __init__(self, runtime, model_path, settings=None):
        self.runtime = runtime
        self.model_path = Path(model_path)
        self.settings = settings or H3Settings()

    def _load(self):
        from diffusers import (
            AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio,
            MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel, ModularPipeline,
        )
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
        from diffusers.video_processor import VideoProcessor
        runtime = self.runtime
        torch = runtime.torch
        root = self.model_path
        count = getattr(torch, runtime.backend).device_count()
        if count < 2:
            raise RuntimeError("Full precision GPU-only H3 requires multiple GPUs.")
        device = torch.device(runtime.device)
        vae_device = f"{runtime.backend}:{count - 1}"

        # Reserve the GPU decoder memory before the two larger model maps are planned.
        video_vae = AutoencoderKLMiniMaxH3.from_pretrained(
            str(root / "vae"), torch_dtype=torch.float32, local_files_only=True
        ).eval().requires_grad_(False).to(vae_device)
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            str(root / "audio_vae"), torch_dtype=torch.float32, local_files_only=True
        ).eval().requires_grad_(False).to(vae_device)
        if runtime.backend == "musa":
            _patch_pointwise_conv3d(video_vae)
            _patch_audio_convolutions(audio_vae)
        _gpu_decode_boundary(video_vae, vae_device, vae_device, autocast=True)
        _gpu_decode_boundary(audio_vae, vae_device, device)
        runtime.assert_on_accelerator(video_vae)
        runtime.assert_on_accelerator(audio_vae)

        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            str(root / "text_encoder"), torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        ).eval().requires_grad_(False)
        text_encoder = _dispatch_resident(text_encoder, runtime)
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            str(root / "transformer"), torch_dtype=torch.bfloat16, local_files_only=True,
        ).eval().requires_grad_(False)
        if runtime.backend == "musa":
            _patch_musa_attention(transformer)
        _chunk_h3_feed_forwards(transformer)
        _chunk_h3_adaln_blocks(transformer)
        transformer = _dispatch_resident(transformer, runtime)
        align_h3_residual_outputs(transformer, runtime.device)

        pipe = ModularPipeline.from_pretrained(str(root), workflow="t2va", local_files_only=True)
        pipe.update_components(
            text_encoder=text_encoder, transformer=transformer,
            vae=video_vae, audio_vae=audio_vae,
            video_processor=VideoProcessor(vae_scale_factor=16, do_normalize=False),
            tokenizer=AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True),
            processor=AutoProcessor.from_pretrained(str(root / "processor"), local_files_only=True),
            scheduler=MiniMaxH3Scheduler.from_pretrained(str(root / "scheduler"), local_files_only=True),
            audio_scheduler=MiniMaxH3Scheduler.from_pretrained(str(root / "audio_scheduler"), local_files_only=True),
        )

        class ExplicitDevicePipeline(type(pipe)):
            @property
            def _execution_device(self):
                return device

        pipe.__class__ = ExplicitDevicePipeline
        _patch_h3_video_decode_device(pipe, vae_device)
        if runtime.backend == "musa":
            # H3 creates positional metadata in FP64 on the host. S4000 has no
            # FP64 device arithmetic; the model's RoPE itself uses FP32.
            from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
            original = MiniMaxH3PrepareLayoutStep.build_packed_sequence
            if not getattr(original, "_vocagent_fp32", False):
                def fp32_layout(*args, **kwargs):
                    result = original(*args, **kwargs)
                    return (result[0].float(), *result[1:])
                fp32_layout._vocagent_fp32 = True
                MiniMaxH3PrepareLayoutStep.build_packed_sequence = staticmethod(fp32_layout)
        return pipe

    def generate(self, video_prompt: str, output_path: Path) -> None:
        if not isinstance(video_prompt, str) or not video_prompt.strip():
            raise ValueError("H3 requires a nonempty video prompt.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.runtime.use(("minimax-h3", str(self.model_path.resolve())), self._load) as pipe:
            torch = self.runtime.torch
            with model_phase("run", self.runtime, model=str(self.model_path)) as phase:
                with torch.inference_mode():
                    result = pipe(
                        prompt=video_prompt, width=self.settings.width, height=self.settings.height,
                        num_frames=self.settings.num_frames, num_inference_steps=self.settings.steps,
                        generator=torch.Generator(device="cpu").manual_seed(secrets.randbits(32)),
                        output=["videos", "audio", "sampling_rate"],
                    )
                if current_step() is not None:
                    frames = len(result["videos"][0])
                    # H3's configured sigma points delimit one fewer denoise updates.
                    phase.add_work(frames=frames, video_seconds=frames / 24,
                                   diffusion_steps=self.settings.steps - 1)
            audio = result["audio"][0]
            if audio.numel() == 0 or not torch.isfinite(audio).all().item():
                raise RuntimeError("H3 produced missing or nonfinite audio.")
            from diffusers.utils.export_utils import encode_video
            with tempfile.TemporaryDirectory(prefix=".h3-", dir=output_path.parent) as directory:
                candidate = Path(directory) / "video.mp4"
                encode_video(
                    result["videos"][0], fps=24, output_path=str(candidate),
                    audio=audio.detach().float().cpu(), audio_sample_rate=int(result["sampling_rate"]),
                )
                if not candidate.is_file() or not candidate.stat().st_size:
                    raise RuntimeError("H3 failed to write its video/audio output.")
                candidate.replace(output_path)
