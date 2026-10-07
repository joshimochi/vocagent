"""H3 capacity policy, frame geometry and GPU-compatible convolution checks."""

import importlib.util
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import MagicMock, Mock, patch

from vocagent.benchmark import Benchmark
from vocagent.minimax import (
    H3Settings, MiniMaxH3VideoGenerator, _dispatch_resident, _patch_pointwise_conv3d, _patch_audio_convolutions,
    _gpu_decode_boundary,
    _chunk_h3_feed_forwards,
    _chunk_h3_adaln_blocks,
    _chunk_h3_rotary_emb,
    _patch_musa_attention,
    _patch_h3_video_decode_device,
)


class H3SettingsTests(unittest.TestCase):
    def test_generation_benchmark_excludes_encoding_and_records_completed_output(self):
        clock = [0.0]
        audio = MagicMock()
        audio.numel.return_value = 12

        def pipe(**kwargs):
            clock[0] += 7
            return {"videos": [[object()] * 345], "audio": [audio], "sampling_rate": 48000}

        @contextmanager
        def use(key, loader):
            yield pipe

        def encode(videos, **kwargs):
            clock[0] += 3
            Path(kwargs["output_path"]).write_bytes(b"video and audio")

        torch = SimpleNamespace(inference_mode=nullcontext, Generator=Mock(), isfinite=Mock())
        torch.isfinite.return_value.all.return_value.item.return_value = True
        runtime = SimpleNamespace(use=use, torch=torch, backend="musa")
        with tempfile.TemporaryDirectory() as directory, patch(
            "vocagent.benchmark.perf_counter", side_effect=lambda: clock[0],
        ), patch.dict("sys.modules", {"diffusers": SimpleNamespace(), "diffusers.utils": SimpleNamespace(),
                                     "diffusers.utils.export_utils": SimpleNamespace(encode_video=encode)}):
            benchmark = Benchmark(Path(directory) / "benchmark.json")
            output = Path(directory) / "video.mp4"
            with benchmark.step("video.generation"):
                MiniMaxH3VideoGenerator(runtime, "/models/h3").generate("hello", output)
            total = benchmark.finish()["total"]
            self.assertTrue(output.is_file())
        self.assertEqual(total["model_run_seconds"], 7)
        self.assertEqual(total["wall_seconds"], 10)
        self.assertEqual(total["work"], {"frames": 345, "video_seconds": 14.375, "diffusion_steps": 49})

    def test_frame_count_obeys_vae_grid_and_duration_limits(self):
        frames = H3Settings().num_frames
        self.assertEqual((frames - 5) % 17, 0)
        self.assertLessEqual(frames / 24, 15)
        self.assertGreater((frames + 17) / 24, 15)

    def test_dimensions_and_duration_cannot_be_configured(self):
        for settings in ({"width": 1344}, {"height": 768}, {"seconds": 10}):
            with self.subTest(settings=settings), self.assertRaises(TypeError):
                H3Settings(**settings)
        with self.assertRaises(ValueError):
            H3Settings(steps=1)

    def test_explicit_profiles_preserve_full_duration_and_worker_round_trip(self):
        for profile, dimensions in (("quality", (1344, 768)), ("preview", (960, 544))):
            with self.subTest(profile=profile):
                settings = H3Settings(profile=profile, steps=20)
                self.assertEqual((settings.width, settings.height), dimensions)
                self.assertEqual(settings.seconds, 15)
                self.assertEqual(settings.num_frames, 345)
                self.assertEqual(H3Settings(**asdict(settings)), settings)
                self.assertEqual(asdict(settings), {"steps": 20, "profile": profile})
        self.assertEqual(H3Settings().profile, "quality")
        self.assertEqual(H3Settings().steps, 50)

    def test_invalid_profile_and_noninteger_steps_fail(self):
        for profile in ("", "fast", "PREVIEW", None, [], True):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "profile"):
                H3Settings(profile=profile)
        for steps in (True, False, 2.0, "20", None, float("nan"), float("inf"), 0, -1):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "integer"):
                H3Settings(steps=steps)

    def test_insufficient_gpu_capacity_never_dispatches_to_cpu(self):
        accelerate = SimpleNamespace(infer_auto_device_map=Mock(return_value={"": "cpu"}),
                                     dispatch_model=Mock())
        gpu = SimpleNamespace(device_count=lambda: 1, mem_get_info=lambda index: (8*1024**3, 8*1024**3),
                              memory_allocated=lambda index: 0)
        runtime = SimpleNamespace(torch=SimpleNamespace(musa=gpu), backend="musa", device="musa:0")
        with patch.dict("sys.modules", {"accelerate": accelerate}):
            with self.assertRaisesRegex(RuntimeError, "without offload"):
                _dispatch_resident(SimpleNamespace(_no_split_modules=[]), runtime)
        accelerate.dispatch_model.assert_not_called()

    def test_gpu_budget_accounts_for_models_already_resident(self):
        gib = 1024**3
        model = SimpleNamespace(_no_split_modules=[], parameters=lambda: ())
        accelerate = SimpleNamespace(infer_auto_device_map=Mock(return_value={"": 0}),
                                     dispatch_model=Mock(return_value=model))
        gpu = SimpleNamespace(device_count=lambda: 3,
                              mem_get_info=lambda index: ((36, 43, 5)[index] * gib, 48 * gib),
                              memory_allocated=lambda index: (10, 3, 40)[index] * gib)
        runtime = SimpleNamespace(torch=SimpleNamespace(musa=gpu), backend="musa", device="musa:0",
                                  assert_on_accelerator=Mock())
        with patch.dict("sys.modules", {"accelerate": accelerate}):
            self.assertIs(_dispatch_resident(model, runtime), model)
        self.assertEqual(accelerate.infer_auto_device_map.call_args.kwargs["max_memory"],
                         {0: 26 * gib, 1: 33 * gib, 2: 0, "cpu": 0})
        runtime.assert_on_accelerator.assert_called_once_with(model)


class H3VideoDecodePlacementTests(unittest.TestCase):
    def test_only_video_block_uses_vae_device_and_pipeline_state_is_restored(self):
        class Tensor:
            def __init__(self, device):
                self.device = device

            def to(self, device):
                return Tensor(device)

        class State:
            def __init__(self):
                self.values = {"latents": Tensor("musa:0"), "audio_latents": Tensor("musa:0")}

            def get(self, name):
                return self.values[name]

            def set(self, name, value):
                self.values[name] = value

        observations = []

        class MiniMaxH3VideoDecodeStep:
            def __call__(self, components, state):
                observations.append((components._execution_device, state.get("latents").device))
                self_test.assertIs(components.vae, vae)
                self_test.assertEqual(components.pixel_mean, (0.1, 0.2, 0.3))
                if getattr(self, "fail", False):
                    raise RuntimeError("decoder failed")
                state.set("videos", Tensor(components._execution_device))
                return components, state

        self_test = self
        vae = object()
        video = MiniMaxH3VideoDecodeStep()
        untouched_video = MiniMaxH3VideoDecodeStep()
        audio = object()

        class Pipeline(SimpleNamespace):
            @property
            def blocks(self):
                return deepcopy(self._blocks)

        pipe = Pipeline(
            _execution_device="musa:0", vae=vae, pixel_mean=(0.1, 0.2, 0.3),
            _blocks=SimpleNamespace(sub_blocks={"decoders": SimpleNamespace(
                sub_blocks={"video": video, "audio": audio},
            )}),
        )
        public_video = pipe.blocks.sub_blocks["decoders"].sub_blocks["video"]
        self.assertIsNot(public_video, video)
        original_class = type(video)
        dependency = SimpleNamespace(MiniMaxH3VideoDecodeStep=MiniMaxH3VideoDecodeStep)
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(device=lambda device: device),
            "diffusers.modular_pipelines.minimax_h3.decoders": dependency,
        }):
            _patch_h3_video_decode_device(pipe, "musa:3")
            configured_class = type(video)
            self.assertIsNot(configured_class, original_class)
            _patch_h3_video_decode_device(pipe, "musa:3")
            self.assertIs(type(video), configured_class)
            self.assertIs(type(untouched_video), original_class)
            self.assertIs(type(public_video), original_class)
            self.assertIs(pipe._blocks.sub_blocks["decoders"].sub_blocks["audio"], audio)
            for fail in (False, True):
                with self.subTest(fail=fail):
                    state = State()
                    latents, audio_latents = state.get("latents"), state.get("audio_latents")
                    video.fail = fail
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                            video(pipe, state)
                    else:
                        components, result = video(pipe, state)
                        self.assertIs(components, pipe)
                        self.assertIs(result, state)
                        self.assertEqual(state.get("videos").device, "musa:3")
                    self.assertIs(state.get("latents"), latents)
                    self.assertIs(state.get("audio_latents"), audio_latents)
                    self.assertEqual(pipe._execution_device, "musa:0")
            untouched_video(pipe, State())
        self.assertEqual(observations, [("musa:3", "musa:3"), ("musa:3", "musa:3"), ("musa:0", "musa:0")])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is an optional model dependency")
class H3FeedForwardTests(unittest.TestCase):
    def test_chunked_swiglu_preserves_tokens_weights_and_distinct_blocks(self):
        import torch

        class SwiGLU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(7, 22)
                self.out = torch.nn.Linear(11, 7)

            def forward(self, inputs):
                value, gate = self.proj(inputs).chunk(2, dim=-1)
                return self.out(value * torch.nn.functional.silu(gate))

        class MiniMaxH3TransformerBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ff = SwiGLU()

        torch.manual_seed(45)
        model = torch.nn.ModuleList([MiniMaxH3TransformerBlock(), MiniMaxH3TransformerBlock()])
        parameters = [(id(parameter), parameter.data_ptr(), parameter.device) for parameter in model.parameters()]
        inputs = torch.randn(2, 7, 19).transpose(1, 2)
        expected = [block.ff(inputs) for block in model]
        _chunk_h3_feed_forwards(model, chunk_size=4)
        for block, reference in zip(model, expected):
            torch.testing.assert_close(block.ff(inputs), reference)
        self.assertEqual(parameters, [(id(p), p.data_ptr(), p.device) for p in model.parameters()])

    def test_small_sequences_keep_fast_path_and_chunking_is_idempotent(self):
        import torch

        class MiniMaxH3TokenRefinerBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ff = torch.nn.Identity()

        block = MiniMaxH3TokenRefinerBlock()
        block.ff.forward = Mock(side_effect=lambda inputs: inputs)
        original = block.ff.forward
        _chunk_h3_feed_forwards(block, chunk_size=4)
        _chunk_h3_feed_forwards(block, chunk_size=4)
        inputs = torch.randn(2, 4, 7)
        self.assertIs(block.ff(inputs), inputs)
        original.assert_called_once_with(inputs)
        original.reset_mock()
        inputs = torch.randn(2, 9, 7)
        torch.testing.assert_close(block.ff(inputs), inputs)
        self.assertEqual([call.args[0].shape[1] for call in original.call_args_list], [4, 4, 1])

    def test_projected_chunks_are_released_before_the_next_projection(self):
        import torch
        references = []

        class Projection(torch.nn.Module):
            def forward(self, inputs):
                self_test.assertTrue(all(reference() is None for reference in references))
                output = inputs.square() + 0.5
                references.append(weakref.ref(output))
                return output

        class MiniMaxH3TransformerBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ff = Projection()

        self_test = self
        block = MiniMaxH3TransformerBlock()
        _chunk_h3_feed_forwards(block, chunk_size=4)
        inputs = torch.randn(2, 7, 11).transpose(1, 2)
        with torch.inference_mode():
            actual = block.ff(inputs)
        torch.testing.assert_close(actual, inputs.square() + 0.5)
        self.assertEqual(len(references), 3)
        self.assertTrue(all(reference() is None for reference in references))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is an optional model dependency")
class H3AdaLNTests(unittest.TestCase):
    @staticmethod
    def make_block(dtype):
        import torch

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 24, bias=False)
                self.sequence_lengths = []

            def forward(self, hidden_states, rotary_emb, attention_mask):
                self.sequence_lengths.append(hidden_states.shape[1])
                query, key, value = self.proj(hidden_states).chunk(3, -1)
                return torch.nn.functional.scaled_dot_product_attention(
                    query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1),
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                ).squeeze(1)

        class Modulation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(5, 48, bias=False)

            def forward(self, temb):
                return self.proj(temb).chunk(6, -1)

        class MiniMaxH3TransformerBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = torch.nn.RMSNorm(8)
                self.norm2 = torch.nn.RMSNorm(8)
                self.attn = Attention()
                self.ff = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 8))
                self.adaln_proj = Modulation()

            def forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
                # The released MiniMaxH3TransformerBlock operation order.
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
                residual = hidden_states
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (
                    1.0 + scale_msa.index_select(0, adaln_indices)
                ) + shift_msa.index_select(0, adaln_indices)
                attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
                hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attn_output
                residual = hidden_states
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (
                    1.0 + scale_mlp.index_select(0, adaln_indices)
                ) + shift_mlp.index_select(0, adaln_indices)
                ff_output = self.ff(norm_hidden_states)
                return residual + gate_mlp.index_select(0, adaln_indices) * ff_output

        return MiniMaxH3TransformerBlock().to(dtype=dtype).eval().requires_grad_(False)

    def test_chunked_modulation_matches_released_block_and_keeps_full_attention(self):
        import torch
        torch.manual_seed(51)
        for dtype in (torch.float32, torch.bfloat16):
            for mask in (None, torch.eye(11, dtype=torch.bool).logical_not()):
                with self.subTest(dtype=dtype, mask=mask is not None):
                    model = torch.nn.ModuleList([self.make_block(dtype), self.make_block(dtype)])
                    inputs = torch.randn(2, 8, 11, dtype=dtype).transpose(1, 2)
                    temb = torch.randn(4, 5, dtype=dtype)
                    indices = torch.tensor([3, 0, 1, 1, 2, 0, 3, 2, 1, 0, 2])
                    rotary = object()
                    parameters = [(id(p), p.data_ptr(), p.device) for p in model.parameters()]
                    with torch.inference_mode():
                        expected = [block(inputs, temb, indices, rotary, mask) for block in model]
                        _chunk_h3_adaln_blocks(model, chunk_size=4)
                        for block, reference in zip(model, expected):
                            actual = block(inputs, temb, indices, rotary, mask)
                            tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=0, atol=0)
                            torch.testing.assert_close(actual, reference, **tolerance)
                            self.assertEqual(block.attn.sequence_lengths, [11, 11])
                            self.assertEqual(actual.dtype, dtype)
                            self.assertEqual(actual.device, inputs.device)
                    self.assertEqual(parameters, [(id(p), p.data_ptr(), p.device) for p in model.parameters()])

    def test_norm_scratch_and_completed_attention_branch_are_released(self):
        import torch
        block = self.make_block(torch.float32)
        norm_outputs, attn_inputs, attn_outputs = [], [], []
        chunk_lengths = []

        def norm_before(module, args):
            self.assertTrue(all(reference() is None for reference in norm_outputs))
            chunk_lengths.append(args[0].shape[1])
            if module is block.norm2:
                self.assertTrue(all(reference() is None for reference in attn_inputs + attn_outputs))

        def norm_after(module, args, output):
            norm_outputs.append(weakref.ref(output))

        def attention_after(module, args, output):
            attn_inputs.append(weakref.ref(args[0]))
            attn_outputs.append(weakref.ref(output))

        for norm in (block.norm1, block.norm2):
            norm.register_forward_pre_hook(norm_before)
            norm.register_forward_hook(norm_after)
        block.attn.register_forward_hook(attention_after)
        _chunk_h3_adaln_blocks(block, chunk_size=4)
        with torch.inference_mode():
            output = block(torch.randn(2, 11, 8), torch.randn(4, 5), torch.arange(11) % 4, None)
        self.assertEqual(output.shape, (2, 11, 8))
        self.assertEqual(chunk_lengths, [4, 4, 3, 4, 4, 3])
        self.assertTrue(all(reference() is None for reference in norm_outputs + attn_inputs + attn_outputs))

    def test_short_sequences_use_original_forward_and_chunk_setting_is_guarded(self):
        import torch
        block = self.make_block(torch.float32)
        block.forward = Mock(return_value=object())
        original = block.forward
        _chunk_h3_adaln_blocks(block, chunk_size=4)
        _chunk_h3_adaln_blocks(block, chunk_size=4)
        args = (torch.randn(2, 4, 8), torch.randn(3, 5), torch.arange(4) % 3, object(), object())
        self.assertIs(block(*args), original.return_value)
        original.assert_called_once_with(*args)
        with self.assertRaisesRegex(ValueError, "already configured"):
            _chunk_h3_adaln_blocks(block, chunk_size=3)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _chunk_h3_adaln_blocks(block, chunk_size=0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is an optional model dependency")
class H3RotaryTests(unittest.TestCase):
    def test_chunked_rotary_matches_released_math_and_releases_temporary_results(self):
        import torch

        def released_rotary(inputs, cosine, sine):
            # H3 rotates the leading channels by rotate-half and passes the
            # remainder through unchanged, using frequencies shared by heads.
            width = cosine.shape[-1]
            active = inputs[..., :width]
            rotated = torch.cat((-active[..., width // 2:], active[..., :width // 2]), dim=-1)
            cosine = cosine.to(dtype=inputs.dtype).unsqueeze(0).unsqueeze(2)
            sine = sine.to(dtype=inputs.dtype).unsqueeze(0).unsqueeze(2)
            return torch.cat((active * cosine + rotated * sine, inputs[..., width:]), dim=-1).contiguous()

        torch.manual_seed(47)
        tokens = 2 * 8192 + 5
        frequencies = torch.randn(96, tokens)
        cosine, sine = frequencies.cos().transpose(0, 1), frequencies.sin().transpose(0, 1)
        self.assertFalse(cosine.is_contiguous())
        self.assertFalse(sine.is_contiguous())
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs = torch.randn(2, 3, tokens, 128, dtype=dtype).transpose(1, 2)
                self.assertFalse(inputs.is_contiguous())
                references, lengths = [], []

                def observed_rotary(chunk, chunk_cosine, chunk_sine):
                    self.assertTrue(all(reference() is None for reference in references))
                    self.assertEqual(chunk_cosine.shape, (chunk.shape[1], 96))
                    self.assertEqual(chunk_sine.shape, (chunk.shape[1], 96))
                    result = released_rotary(chunk, chunk_cosine, chunk_sine)
                    references.append(weakref.ref(result))
                    lengths.append(chunk.shape[1])
                    return result

                with torch.inference_mode():
                    expected = released_rotary(inputs, cosine, sine)
                    actual = _chunk_h3_rotary_emb(inputs, (cosine, sine), observed_rotary)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual[..., 96:], inputs[..., 96:], rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device, inputs.device)
                self.assertEqual(lengths, [8192, 8192, 5])
                self.assertTrue(all(reference() is None for reference in references))

    def test_short_rotary_sequences_keep_the_original_call(self):
        import torch
        inputs = torch.randn(2, 7, 3, 128)
        rotary_emb = (torch.randn(7, 96), torch.randn(7, 96))
        result = torch.randn_like(inputs)
        original = Mock(return_value=result)
        self.assertIs(_chunk_h3_rotary_emb(inputs, rotary_emb, original), result)
        self.assertEqual(original.call_count, 1)
        self.assertIs(original.call_args.args[0], inputs)
        self.assertIs(original.call_args.args[1], rotary_emb[0])
        self.assertIs(original.call_args.args[2], rotary_emb[1])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is an optional model dependency")
class H3AttentionTests(unittest.TestCase):
    def test_resident_attention_matches_original_math_and_releases_projection_storage(self):
        import torch
        functional = torch.nn.functional

        def rotary(inputs, cosine, sine):
            return inputs * cosine + torch.roll(inputs, 1, -1) * sine

        class MiniMaxH3Attention(torch.nn.Module):
            def __init__(self, fused):
                super().__init__()
                self.heads = 2
                self.fused_projections = fused
                self.to_q = torch.nn.Linear(8, 8)
                self.to_k = torch.nn.Linear(8, 8)
                self.to_v = torch.nn.Linear(8, 8)
                self.to_qkv = torch.nn.Linear(8, 24)
                self.norm_q = torch.nn.RMSNorm(4)
                self.norm_k = torch.nn.RMSNorm(4)
                self.to_out = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Identity()])

            def set_processor(self, processor):
                self.processor = processor

        dependency = SimpleNamespace(
            MiniMaxH3Attention=MiniMaxH3Attention,
            MiniMaxH3AttnProcessor=object, _apply_rotary_emb=rotary,
        )
        torch.manual_seed(46)
        inputs = torch.randn(2, 8, 7).transpose(1, 2)
        cosine, sine = torch.randn(1, 7, 1, 4), torch.randn(1, 7, 1, 4)
        mask = torch.ones(7, 7, dtype=torch.bool)
        mask[:, -1] = False
        for fused in (True, False):
            for use_rotary in (True, False):
                for attention_mask in (None, mask):
                    with self.subTest(fused=fused, rotary=use_rotary, mask=attention_mask is not None):
                        attention = MiniMaxH3Attention(fused)
                        with torch.inference_mode():
                            if fused:
                                query, key, value = attention.to_qkv(inputs).chunk(3, dim=-1)
                            else:
                                query, key, value = (projection(inputs) for projection in (
                                    attention.to_q, attention.to_k, attention.to_v,
                                ))
                            query = attention.norm_q(query.unflatten(-1, (2, -1)))
                            key = attention.norm_k(key.unflatten(-1, (2, -1)))
                            value = value.unflatten(-1, (2, -1))
                            if use_rotary:
                                query, key = rotary(query, cosine, sine), rotary(key, cosine, sine)
                            expected = functional.scaled_dot_product_attention(
                                query.transpose(1, 2).contiguous(), key.transpose(1, 2).contiguous(),
                                value.transpose(1, 2).contiguous(), attn_mask=attention_mask,
                                dropout_p=0.0, is_causal=False,
                            ).transpose(1, 2).flatten(2, 3).type_as(query)
                            expected = attention.to_out[1](attention.to_out[0](expected))
                        del query, key, value
                        projections = []

                        def retain_reference(module, args, output):
                            projections.append(weakref.ref(output))

                        sources = (attention.to_qkv,) if fused else (
                            attention.to_q, attention.to_k, attention.to_v,
                        )
                        hooks = [source.register_forward_hook(retain_reference) for source in sources]
                        original_sdpa = functional.scaled_dot_product_attention

                        def resident_sdpa(*args, **kwargs):
                            self.assertTrue(all(reference() is None for reference in projections))
                            self.assertTrue(all(tensor.is_contiguous() for tensor in args[:3]))
                            return original_sdpa(*args, **kwargs)

                        with patch.dict("sys.modules", {
                            "diffusers.models.transformers.transformer_minimax_h3": dependency,
                        }), patch.object(functional, "scaled_dot_product_attention", side_effect=resident_sdpa):
                            _patch_musa_attention(attention)
                            with torch.inference_mode():
                                actual = attention.processor(
                                    attention, inputs, (cosine, sine) if use_rotary else None,
                                    attention_mask,
                                )
                        for hook in hooks:
                            hook.remove()
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(actual.dtype, expected.dtype)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is an optional model dependency")
class H3DecodeTests(unittest.TestCase):
    def test_empty_and_nonfinite_decodes_are_rejected_before_export(self):
        import torch
        for sample in (torch.empty(0), torch.tensor([float("nan")]), torch.tensor([float("inf")])):
            for as_tuple in (True, False):
                with self.subTest(sample=sample, as_tuple=as_tuple):
                    output = (sample,) if as_tuple else SimpleNamespace(sample=sample)
                    vae = SimpleNamespace(dtype=torch.float32, decode=lambda latents: output)
                    _gpu_decode_boundary(vae, "cpu", "cpu")
                    with self.assertRaisesRegex(RuntimeError, "empty or nonfinite"):
                        vae.decode(torch.zeros(1))

    def test_finite_decodes_preserve_output_and_enforce_the_configured_device(self):
        import torch
        sample = torch.tensor([0.1, -0.2, 0.3])
        for as_tuple in (True, False):
            output = (sample,) if as_tuple else SimpleNamespace(sample=sample)
            vae = SimpleNamespace(dtype=torch.float32, decode=lambda latents: output)
            _gpu_decode_boundary(vae, "cpu", "cpu")
            decoded = vae.decode(torch.zeros(1))
            torch.testing.assert_close(decoded[0] if as_tuple else decoded.sample, sample)
        misplaced = torch.empty(1, device="meta")
        vae = SimpleNamespace(dtype=torch.float32, decode=lambda latents: (misplaced,))
        _gpu_decode_boundary(vae, "cpu", "cpu")
        with self.assertRaisesRegex(RuntimeError, "configured GPU"):
            vae.decode(torch.zeros(1))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is an optional model dependency")
class H3ConvolutionTests(unittest.TestCase):
    def test_depthwise_audio_resamplers_match_native_filters(self):
        import torch
        functional = torch.nn.functional

        class MiniMaxH3AudioLowPassFilter1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pad_left, self.pad_right, self.stride = 5, 6, 2
                self.register_buffer("filter", torch.randn(1, 1, 12))

            def forward(self, inputs):
                inputs = functional.pad(inputs, (self.pad_left, self.pad_right), mode="replicate")
                return functional.conv1d(inputs, self.filter.expand(inputs.shape[1], -1, -1),
                                         stride=self.stride, groups=inputs.shape[1])

        class MiniMaxH3AudioUpSample1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ratio, self.stride, self.pad = 2, 2, 5
                self.pad_left, self.pad_right = 15, 15
                self.register_buffer("filter", torch.randn(1, 1, 12))

            def forward(self, inputs):
                inputs = functional.pad(inputs, (self.pad, self.pad), mode="replicate")
                output = self.ratio * functional.conv_transpose1d(
                    inputs, self.filter.expand(inputs.shape[1], -1, -1), stride=self.stride,
                    groups=inputs.shape[1],
                )
                return output[..., self.pad_left:-self.pad_right]

        torch.manual_seed(44)
        for model_class in (MiniMaxH3AudioLowPassFilter1d, MiniMaxH3AudioUpSample1d):
            for channels in (1, 8):
                with self.subTest(kind=model_class, channels=channels):
                    model = model_class()
                    inputs = torch.randn(2, channels, 29)
                    expected = model(inputs)
                    _patch_audio_convolutions(model)
                    torch.testing.assert_close(model(inputs), expected)

    def test_audio_convolutions_match_native_for_groups_stride_padding_and_dilation(self):
        import torch
        torch.manual_seed(43)
        for kind in (torch.nn.Conv1d, torch.nn.ConvTranspose1d):
            for groups in (1, 2):
                for bias in (True, False):
                    for kernel, stride, padding, dilation, output_padding in (
                        (1, 1, 0, 1, 0), (3, 1, 1, 3, 0), (7, 1, 3, 1, 0),
                        (16, 8, 4, 1, 0), (5, 3, 3, 2, 0), (4, 2, 1, 1, 1),
                    ):
                        with self.subTest(kind=kind, groups=groups, bias=bias, kernel=kernel,
                                          stride=stride, dilation=dilation, output_padding=output_padding):
                            extra = {"output_padding": output_padding} if kind is torch.nn.ConvTranspose1d else {}
                            convolution = kind(4, 6, kernel, stride=stride, padding=padding,
                                               dilation=dilation, groups=groups, bias=bias, **extra)
                            inputs = torch.randn(2, 4, 13)
                            expected = convolution(inputs)
                            _patch_audio_convolutions(convolution)
                            torch.testing.assert_close(convolution(inputs), expected)

    def test_pointwise_replacement_matches_conv3d_with_bias_and_noncontiguous_input(self):
        import torch
        torch.manual_seed(42)
        for bias in (True, False):
            conv = torch.nn.Conv3d(3, 7, 1, bias=bias)
            inputs = torch.randn(2, 3, 4, 5, 6).transpose(3, 4)
            expected = conv(inputs)
            _patch_pointwise_conv3d(conv)
            torch.testing.assert_close(conv(inputs), expected)

    def test_non_pointwise_convolutions_are_preserved(self):
        import torch
        conv = torch.nn.Conv3d(3, 7, 3)
        original = conv.forward
        _patch_pointwise_conv3d(conv)
        self.assertEqual(conv.forward, original)


if __name__ == "__main__":
    unittest.main()
