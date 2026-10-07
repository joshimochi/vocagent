"""Transformers chat client with explicit GPU placement and on-demand residency."""

import io
import json
import math

from .benchmark import current_step, model_phase
from .qwen import QwenError


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_IMAGES = 64
VISION_MIN_PIXELS = 65536
VISION_MAX_PIXELS = 262144


class LocalChatClient:
    def __init__(self, runtime, model_path, *, vision=False, max_context=16384):
        if type(max_context) is not int or max_context < 1:
            raise ValueError("Local context limit must be a positive integer.")
        self.runtime = runtime
        self.model_path = str(model_path)
        self.vision = vision
        self.max_context = max_context

    def _load(self):
        from accelerate import dispatch_model
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
        from transformers import Qwen3VLForConditionalGeneration
        runtime = self.runtime
        model_class = Qwen3VLForConditionalGeneration if self.vision else AutoModelForCausalLM
        if self.vision:
            config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
            mapping = runtime.device_map(model_class, config)
            processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True,
                size={"shortest_edge": VISION_MIN_PIXELS, "longest_edge": VISION_MAX_PIXELS},
                min_pixels=VISION_MIN_PIXELS, max_pixels=VISION_MAX_PIXELS,
            )
        else:
            mapping = {"": runtime.device}
            processor = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        # safetensors' Rust device parser does not recognize MUSA. Deserialize
        # checkpoints on CPU, then place every module on its planned GPU before
        # inference. No CPU modules or offloading hooks remain after dispatch.
        model = model_class.from_pretrained(
            self.model_path, dtype=runtime.dtype,
            attn_implementation="sdpa" if self.vision else "eager", local_files_only=True,
        )
        model = dispatch_model(model, device_map=mapping, offload_buffers=False).eval()
        if self.vision and runtime.backend == "musa":
            from .gpu_ops import patch_musa_convolutions
            patch_musa_convolutions(model)
        runtime.assert_on_accelerator(model)
        return model, processor

    def _messages(self, messages):
        from PIL import Image
        converted = []
        image_count = 0
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                # Transformers 4.57 ProcessorMixin iterates content blocks even
                # for text-only/system messages when tokenize=True.
                content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if item["type"] == "image":
                        image_count += 1
                        if image_count > MAX_IMAGES:
                            raise ValueError(f"Local vision accepts at most {MAX_IMAGES} images.")
                        jpeg = item["image"]
                        if not isinstance(jpeg, bytes):
                            raise ValueError("Local vision frames must be JPEG bytes.")
                        if len(jpeg) > MAX_IMAGE_BYTES:
                            raise ValueError("Local vision frame exceeds the 8 MiB limit.")
                        with Image.open(io.BytesIO(jpeg)) as picture:
                            if picture.format != "JPEG":
                                raise ValueError("Local vision frame must contain JPEG pixels.")
                            if picture.width * picture.height > MAX_IMAGE_PIXELS:
                                raise ValueError("Local vision frame exceeds the decoded pixel limit.")
                            picture.load()
                            parts.append({"type": "image", "image": picture.convert("RGB")})
                    elif item["type"] == "text":
                        parts.append(item)
                    else:
                        raise ValueError("Unsupported local multimodal content type.")
                content = parts
            else:
                raise ValueError("Local chat message content must be text or multimodal blocks.")
            converted.append({"role": message["role"], "content": content})
        return converted

    def complete(self, messages, *, max_tokens, temperature, non_thinking=False):
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Local generation token limit must be a positive integer.")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature < 0):
            raise ValueError("Local generation temperature must be finite and nonnegative.")
        runtime = self.runtime
        messages = self._messages(messages) if self.vision else list(messages)
        with runtime.use(("chat", self.model_path, self.vision), self._load) as (model, processor):
            template_kwargs = {"enable_thinking": False} if non_thinking else {}
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", truncation=False, **template_kwargs,
            )
            if self.vision and "image_grid_thw" in inputs:
                patch_size = processor.image_processor.patch_size
                # Some Transformers versions retain max_pixels as an inert
                # attribute while sizing from size.longest_edge. Verify actual
                # processed geometry before allocating vision activations.
                for temporal, height, width in inputs["image_grid_thw"]:
                    if int(temporal) != 1 or int(height) * int(width) * patch_size**2 > VISION_MAX_PIXELS:
                        raise QwenError("Processed vision frame exceeds the configured pixel budget.")
            length = inputs["input_ids"].shape[-1]
            if length + max_tokens > self.max_context:
                raise QwenError("Prompt exceeds the configured local context; truncation is disabled.")
            inputs = inputs.to(model.device)
            # The deployed MUSA/Transformers stack repeats or corrupts cached
            # Qwen3-VL completions. Recomputing prefixes passed known-answer and
            # 32-image checks on hs; model weights remain fully GPU-resident.
            options = {"max_new_tokens": max_tokens, "do_sample": temperature > 0,
                       "use_cache": not (self.vision and runtime.backend == "musa")}
            if temperature > 0:
                options.update(temperature=temperature, top_p=.8)
            with model_phase("run", runtime, model=self.model_path) as phase:
                if current_step() is not None:
                    phase.add_work(input_tokens=length)
                    if self.vision:
                        phase.add_work(images=sum(
                            item["type"] == "image" for message in messages
                            for item in message["content"]
                        ))
                with runtime.torch.inference_mode():
                    ids = model.generate(**inputs, **options)
                tokens = ids[0, length:]
                if current_step() is not None:
                    phase.add_work(generated_tokens=len(tokens))
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else (eos or [])
            tokenizer = getattr(processor, "tokenizer", processor)
            if not len(tokens) or int(tokens[-1]) not in eos:
                # Keep failed generations observable without accepting a partial
                # verdict or dumping an unbounded model response into logs.
                diagnostic = {
                    "generated_tokens": len(tokens), "max_tokens": max_tokens,
                    "eos_token_ids": eos,
                    "tail_token_ids": [int(token) for token in tokens[-8:]],
                    "output_prefix": tokenizer.decode(tokens[:256], skip_special_tokens=False)[:1024],
                }
                raise QwenError(
                    "Local completion did not reach EOS; truncated output is rejected. "
                    + json.dumps(diagnostic, ensure_ascii=True)
                )
            result = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if not result:
                raise QwenError("Local model returned empty content.")
            return result
