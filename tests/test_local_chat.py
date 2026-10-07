"""Transformers chat contracts, native image decoding and truncation rejection."""

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from vocagent.benchmark import Benchmark
from vocagent.local_chat import LocalChatClient, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MAX_IMAGES
from vocagent.qwen import QwenError


class Batch(dict):
    def __init__(self, length):
        super().__init__(input_ids=SimpleNamespace(shape=(1, length)))
        self.to = MagicMock(return_value=self)


class Generated:
    def __init__(self, tokens):
        self.tokens = tokens

    def __getitem__(self, key):
        row, selection = key
        assert row == 0
        return self.tokens[selection]


class Runtime:
    backend = "musa"
    device = "musa:0"
    dtype = "bfloat16"

    def __init__(self, resource=None):
        self.resource = resource
        self.keys = []
        self.torch = SimpleNamespace(inference_mode=nullcontext)
        self.assert_on_accelerator = MagicMock()
        self.device_map = MagicMock(return_value={"model.visual": "musa:0", "model.language_model": "musa:1"})

    @contextmanager
    def use(self, key, loader):
        self.keys.append(key)
        yield self.resource if self.resource is not None else loader()


class LocalChatTests(unittest.TestCase):
    def setup_chat(self, *, length=3, tokens=None, text="answer", eos=9, vision=False, max_context=64):
        tokens = [7, 9] if tokens is None else tokens
        inputs = Batch(length)
        tokenizer = SimpleNamespace(decode=MagicMock(return_value=text))
        processor = SimpleNamespace(apply_chat_template=MagicMock(return_value=inputs), tokenizer=tokenizer)
        model = SimpleNamespace(
            device="musa:0", generation_config=SimpleNamespace(eos_token_id=eos),
            generate=MagicMock(return_value=Generated([0] * length + tokens)),
        )
        runtime = Runtime((model, processor))
        client = LocalChatClient(runtime, "local-model", vision=vision, max_context=max_context)
        return client, model, processor, inputs

    def image_module(self, *, image_format="JPEG", width=64, height=48):
        image = MagicMock()
        image.format = image_format
        image.width, image.height = width, height
        image.__enter__.return_value = image
        module = SimpleNamespace(Image=SimpleNamespace(open=MagicMock(return_value=image)))
        return module, image

    def test_benchmark_counts_actual_tokens_and_times_generation_without_preparation_or_decode(self):
        client, model, processor, inputs = self.setup_chat(length=5, tokens=[2, 3, 9])
        clock = [0.0]

        def prepare(*args, **kwargs):
            clock[0] += 3
            return inputs

        def generate(**kwargs):
            clock[0] += 7
            return Generated([0] * 5 + [2, 3, 9])

        def decode(*args, **kwargs):
            clock[0] += 2
            return "answer"

        processor.apply_chat_template.side_effect = prepare
        model.generate.side_effect = generate
        processor.tokenizer.decode.side_effect = decode
        with tempfile.TemporaryDirectory() as directory, patch(
            "vocagent.benchmark.perf_counter", side_effect=lambda: clock[0],
        ):
            benchmark = Benchmark(Path(directory) / "benchmark.json")
            with benchmark.step("chat"):
                client.complete([{"role": "user", "content": "hello"}], max_tokens=7, temperature=0)
            total = benchmark.finish()["total"]
        self.assertEqual(total["model_run_seconds"], 7)
        self.assertEqual(total["wall_seconds"], 12)
        self.assertEqual(total["work"], {"input_tokens": 5, "generated_tokens": 3})

    def test_failed_completion_retains_generated_token_and_image_work(self):
        client, *_ = self.setup_chat(tokens=[7, 8], vision=True)
        pil, _ = self.image_module()
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"PIL": pil}):
            benchmark = Benchmark(Path(directory) / "benchmark.json")
            with self.assertRaises(QwenError):
                with benchmark.step("vision"):
                    client.complete([{"role": "user", "content": [
                        {"type": "image", "image": b"jpeg"}, {"type": "text", "text": "describe"},
                    ]}], max_tokens=3, temperature=0)
            report = benchmark.finish(error=QwenError("truncated"))
        self.assertEqual(report["total"]["work"], {"input_tokens": 3, "generated_tokens": 2, "images": 1})
        self.assertEqual(report["calls"][0]["status"], "failed")
        self.assertEqual(report["total"]["model_run_count"], 1)

    def test_text_completion_uses_full_prompt_and_rejects_context_overflow_before_gpu_copy(self):
        client, model, processor, inputs = self.setup_chat(length=60, max_context=64)
        with self.assertRaisesRegex(QwenError, "truncation is disabled"):
            client.complete([{"role": "user", "content": "complete transcript"}], max_tokens=5, temperature=0)
        model.generate.assert_not_called()
        inputs.to.assert_not_called()
        options = processor.apply_chat_template.call_args.kwargs
        self.assertIs(options["truncation"], False)

    def test_eos_and_nonempty_content_are_required_for_a_valid_completion(self):
        for tokens, text, eos in (([], "answer", 9), ([7], "answer", 9), ([7, 8], "answer", [9, 10]),
                                  ([9], " ", 9), ([7, 9], "answer", None)):
            with self.subTest(tokens=tokens, text=text, eos=eos):
                client, *_ = self.setup_chat(tokens=tokens, text=text, eos=eos)
                with self.assertRaises(QwenError):
                    client.complete([{"role": "user", "content": "prompt"}], max_tokens=4, temperature=0)
        client, model, processor, inputs = self.setup_chat(tokens=[7, 10], text="  answer  ", eos=[9, 10])
        self.assertEqual(client.complete([{"role": "user", "content": "prompt"}], max_tokens=4, temperature=0), "answer")
        processor.tokenizer.decode.assert_called_once_with([7, 10], skip_special_tokens=True)
        inputs.to.assert_called_once_with("musa:0")
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])

    def test_sampling_options_and_non_thinking_template_are_separate(self):
        client, model, processor, _ = self.setup_chat()
        client.complete([{"role": "user", "content": "prompt"}], max_tokens=4, temperature=.7, non_thinking=True)
        self.assertEqual(model.generate.call_args.kwargs["temperature"], .7)
        self.assertEqual(model.generate.call_args.kwargs["top_p"], .8)
        self.assertTrue(model.generate.call_args.kwargs["do_sample"])
        self.assertFalse(processor.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_musa_vision_uses_verified_uncached_decoding(self):
        for backend, vision in (("musa", True), ("musa", False), ("cuda", True)):
            client, model, _, _ = self.setup_chat(vision=vision)
            client.runtime.backend = backend
            module, _ = self.image_module()
            with self.subTest(backend=backend, vision=vision), patch.dict("sys.modules", {"PIL": module}):
                client.complete([{"role": "user", "content": "prompt"}], max_tokens=4, temperature=0)
            self.assertEqual(model.generate.call_args.kwargs["use_cache"], not (backend == "musa" and vision))

    def test_actual_image_grid_enforces_pixel_budget_before_gpu_copy(self):
        for grid, allowed in (([1, 32, 32], True), ([1, 28, 48], False), ([2, 16, 16], False)):
            client, model, processor, inputs = self.setup_chat(vision=True)
            processor.image_processor = SimpleNamespace(patch_size=16)
            inputs["image_grid_thw"] = [grid]
            module, _ = self.image_module()
            with self.subTest(grid=grid), patch.dict("sys.modules", {"PIL": module}):
                if allowed:
                    client.complete([{"role": "user", "content": "picture"}], max_tokens=4, temperature=0)
                    model.generate.assert_called_once()
                else:
                    with self.assertRaisesRegex(QwenError, "pixel budget"):
                        client.complete([{"role": "user", "content": "picture"}], max_tokens=4, temperature=0)
                    inputs.to.assert_not_called()
                    model.generate.assert_not_called()

    def test_truncated_generation_retains_bounded_diagnostic_and_still_fails(self):
        client, model, processor, _ = self.setup_chat(tokens=[7] * 20, text="x" * 5000)
        with self.assertRaises(QwenError) as raised:
            client.complete([{"role": "user", "content": "prompt"}], max_tokens=20, temperature=0)
        diagnostic = json.loads(str(raised.exception).split("rejected. ", 1)[1])
        self.assertEqual(diagnostic["generated_tokens"], 20)
        self.assertEqual(diagnostic["eos_token_ids"], [9])
        self.assertEqual(diagnostic["tail_token_ids"], [7] * 8)
        self.assertEqual(len(diagnostic["output_prefix"]), 1024)
        processor.tokenizer.decode.assert_called_once_with([7] * 20, skip_special_tokens=False)

    def test_vision_wraps_system_and_text_messages_and_decodes_native_jpeg_bytes(self):
        client = LocalChatClient(Runtime(), "local-model", vision=True)
        module, image = self.image_module()
        source = [
            {"role": "system", "content": "Judge the candidates."},
            {"role": "user", "content": [
                {"type": "text", "text": "Candidate 0, time 0.000 seconds:"},
                {"type": "image", "image": b"jpeg-pixels"},
            ]},
            {"role": "user", "content": "Choose one."},
        ]
        with patch.dict("sys.modules", {"PIL": module}):
            converted = client._messages(source)
        self.assertEqual(converted[0]["content"], [{"type": "text", "text": "Judge the candidates."}])
        self.assertEqual(converted[2]["content"], [{"type": "text", "text": "Choose one."}])
        self.assertEqual(converted[1]["content"][1], {"type": "image", "image": image.convert.return_value})
        self.assertEqual(module.Image.open.call_args.args[0].getvalue(), b"jpeg-pixels")
        image.load.assert_called_once()
        image.convert.assert_called_once_with("RGB")
        self.assertIsInstance(source[0]["content"], str)
        self.assertEqual(source[1]["content"][1]["image"], b"jpeg-pixels")

    def test_text_only_qwen_vl_call_still_passes_processor_content_blocks(self):
        client, model, processor, _ = self.setup_chat(vision=True)
        module, _ = self.image_module()
        with patch.dict("sys.modules", {"PIL": module}):
            client.complete([{"role": "system", "content": "story"}, {"role": "user", "content": "words"}],
                            max_tokens=4, temperature=0)
        messages = processor.apply_chat_template.call_args.args[0]
        for message in messages:
            self.assertIsInstance(message["content"], list)
            for block in message["content"]:
                self.assertEqual(block["type"], "text")
        module.Image.open.assert_not_called()

    def test_image_urls_nonbytes_and_huge_payloads_fail_before_loading_weights(self):
        for block in (
            {"type": "image_url", "image_url": {"url": "https://example.test/image.jpg"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            {"type": "image", "image": "data:image/jpeg;base64,AAAA"},
            {"type": "image", "image": None},
            {"type": "image", "image": b"A" * (MAX_IMAGE_BYTES + 1)},
        ):
            runtime = Runtime()
            client = LocalChatClient(runtime, "local-model", vision=True)
            module, _ = self.image_module()
            with self.subTest(type=block["type"]), patch.dict("sys.modules", {"PIL": module}), self.assertRaises(ValueError):
                client.complete([{"role": "user", "content": [
                    block,
                ]}], max_tokens=4, temperature=0)
            self.assertEqual(runtime.keys, [])
            module.Image.open.assert_not_called()

    def test_wrong_image_format_and_huge_decoded_pixels_fail_before_loading(self):
        for module, image in (self.image_module(image_format="PNG"),
                              self.image_module(width=MAX_IMAGE_PIXELS + 1, height=1)):
            client = LocalChatClient(Runtime(), "local-model", vision=True)
            with patch.dict("sys.modules", {"PIL": module}), self.assertRaises(ValueError):
                client._messages([{"role": "user", "content": [
                    {"type": "image", "image": b"jpeg-pixels"},
                ]}])
            image.load.assert_not_called()

    def test_image_count_limit_bounds_vision_processing(self):
        client = LocalChatClient(Runtime(), "local-model", vision=True)
        module, _ = self.image_module()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": b"jpeg-pixels"},
        ] * (MAX_IMAGES + 1)}]
        with patch.dict("sys.modules", {"PIL": module}), self.assertRaisesRegex(ValueError, "at most"):
            client._messages(messages)
        self.assertEqual(module.Image.open.call_count, MAX_IMAGES)

    def test_vision_loader_uses_planned_mapping_and_guard_stays_on_gpu(self):
        for vision in (False, True):
            runtime = Runtime()
            model = MagicMock()
            model.eval.return_value = model
            factory = MagicMock(return_value=model)
            config = object()
            module = SimpleNamespace(
                AutoConfig=SimpleNamespace(from_pretrained=MagicMock(return_value=config)),
                AutoProcessor=SimpleNamespace(from_pretrained=MagicMock()),
                AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock()),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=factory),
                Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=factory),
            )
            dispatch = MagicMock(side_effect=lambda candidate, **kwargs: candidate)
            with self.subTest(vision=vision), patch.dict("sys.modules", {
                "transformers": module, "accelerate": SimpleNamespace(dispatch_model=dispatch),
            }), patch("vocagent.gpu_ops.patch_musa_convolutions") as patch_convolutions:
                LocalChatClient(runtime, "/models/test", vision=vision)._load()
            self.assertEqual(patch_convolutions.call_count, int(vision))
            options = factory.call_args.kwargs
            # Passing MUSA to from_pretrained reaches safe_open's unsupported
            # Rust device parser, including the single-device Guard fast path.
            self.assertNotIn("device_map", options)
            dispatch.assert_called_once_with(
                model, device_map=runtime.device_map.return_value if vision else {"": "musa:0"},
                offload_buffers=False,
            )
            self.assertEqual(options["dtype"], "bfloat16")
            self.assertEqual(options["attn_implementation"], "sdpa" if vision else "eager")
            self.assertTrue(options["local_files_only"])
            runtime.assert_on_accelerator.assert_called_once_with(model)
            if vision:
                self.assertEqual(module.AutoProcessor.from_pretrained.call_args.kwargs["max_pixels"], 262144)
                self.assertEqual(module.AutoProcessor.from_pretrained.call_args.kwargs["min_pixels"], 65536)
                self.assertEqual(module.AutoProcessor.from_pretrained.call_args.kwargs["size"],
                                 {"shortest_edge": 65536, "longest_edge": 262144})

    def test_dispatch_failure_never_returns_cpu_model_as_fallback(self):
        runtime = Runtime()
        model = MagicMock()
        factory = MagicMock(return_value=model)
        module = SimpleNamespace(
            AutoConfig=SimpleNamespace(), AutoProcessor=SimpleNamespace(),
            AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=factory),
            Qwen3VLForConditionalGeneration=SimpleNamespace(),
        )
        dispatch = MagicMock(side_effect=RuntimeError("MUSA allocation failed"))
        with patch.dict("sys.modules", {
            "transformers": module, "accelerate": SimpleNamespace(dispatch_model=dispatch),
        }), self.assertRaisesRegex(RuntimeError, "MUSA allocation failed"):
            LocalChatClient(runtime, "/models/guard")._load()
        runtime.assert_on_accelerator.assert_not_called()
        model.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
