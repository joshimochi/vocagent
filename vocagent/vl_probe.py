"""Opt-in live Qwen3-VL diagnostic; run only while its GPU allocation is free."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import time

from .diagnostics import ARITHMETIC_PROMPT
from .gpu_runtime import GPURuntime
from .local_chat import LocalChatClient
from .qwen import QwenQuality, QwenVideoQuality


class ProbeClient(LocalChatClient):
    """Observe real generation without changing truncation or parsing rules."""

    def __init__(self, *args, attention=None, cache=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = attention
        self.cache = cache
        self.generations = []
        self.metadata = {}

    def _load(self):
        model, processor = super()._load()
        if self.attention is not None:
            model.set_attn_implementation(self.attention)
        tokenizer = getattr(processor, "tokenizer", processor)
        image_processor = getattr(processor, "image_processor", None)
        self.metadata = {
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "generation_config": model.generation_config.to_dict(),
            "device_map": {key: str(value) for key, value in model.hf_device_map.items()},
            "tokenizer_eos_id": tokenizer.eos_token_id,
            "tokenizer_eos": tokenizer.eos_token,
            "attention_override": self.attention,
            "cache_override": self.cache,
            "image_processor": {
                name: getattr(image_processor, name, None)
                for name in ("size", "min_pixels", "max_pixels", "patch_size", "temporal_patch_size", "merge_size")
            },
        }
        original = model.generate

        def observed_generate(*args, **kwargs):
            if self.cache is not None:
                kwargs["use_cache"] = self.cache
            prompt_ids = kwargs["input_ids"][0]
            entry = {
                "input_tokens": len(prompt_ids),
                "input_tail": tokenizer.decode(prompt_ids[-96:], skip_special_tokens=False)[-1024:],
                "input_tail_token_ids": prompt_ids[-16:].detach().cpu().tolist(),
                "inputs": {name: list(value.shape) for name, value in kwargs.items()
                           if hasattr(value, "shape")},
                "max_new_tokens": kwargs.get("max_new_tokens"),
                "do_sample": kwargs.get("do_sample"), "use_cache": kwargs.get("use_cache"),
            }
            if "image_grid_thw" in kwargs:
                entry["image_grid_thw"] = kwargs["image_grid_thw"].detach().cpu().tolist()
            self.generations.append(entry)
            started = time.monotonic()
            try:
                result = original(*args, **kwargs)
                tokens = result[0, len(prompt_ids):]
                entry.update(
                    generated_tokens=len(tokens),
                    output_prefix=tokenizer.decode(tokens[:256], skip_special_tokens=False)[:1024],
                    output_tail=tokenizer.decode(tokens[-64:], skip_special_tokens=False)[-512:],
                    output_head_token_ids=tokens[:16].detach().cpu().tolist(),
                    output_tail_token_ids=tokens[-16:].detach().cpu().tolist(),
                )
                return result
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"[:2048]
                raise
            finally:
                entry["seconds"] = round(time.monotonic() - started, 3)

        model.generate = observed_generate
        return model, processor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("/data/group6/vocagent/models/Qwen3-VL-32B-Instruct"))
    parser.add_argument("--video", type=Path, required=True,
                        help="Existing real video; sampled twice to exercise two candidate groups")
    parser.add_argument("--second-video", type=Path,
                        help="Optional distinct candidate; defaults to the first video")
    parser.add_argument("--video-prompt", default="A curious traveler safely crosses a bridge.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", choices=("musa", "cuda"), default="musa")
    parser.add_argument("--max-frames", type=int, default=4,
                        help="Frames per candidate (default 4; use 16 for the production frame budget)")
    parser.add_argument("--attention", choices=("sdpa", "eager"),
                        help="Optional diagnostic override after loading; default is the production SDPA path")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--no-cache", action="store_false", dest="cache",
                       help="Disable KV caching in this diagnostic only")
    cache.add_argument("--use-cache", action="store_true", dest="cache",
                       help="Enable KV caching in this diagnostic only, to investigate the known hs failure")
    parser.set_defaults(cache=None)
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Report must be a new path to preserve earlier diagnostic evidence.")
    videos = (args.video.resolve(), (args.second_video or args.video).resolve())
    for video in videos:
        if not video.is_file():
            parser.error(f"Video does not exist: {video}")
    if not 2 <= args.max_frames <= 16:
        parser.error("max-frames must be between 2 and 16 for the 32-frame candidate budget")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s VL probe %(message)s")
    runtime = GPURuntime(args.backend)
    client = ProbeClient(runtime, args.model, vision=True,
                         attention=args.attention, cache=args.cache)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model), "backend": args.backend,
        "videos": [str(video) for video in videos], "frames_per_candidate": args.max_frames,
        "same_video_candidates": videos[0] == videos[1],
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "accelerate")},
        "cases": [],
    }

    def save():
        report["metadata"] = client.metadata
        report["generations"] = client.generations
        temporary = args.report.with_name(args.report.name + ".partial")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        temporary.replace(args.report)

    cases = [
        ("short_text", "4", lambda: client.complete(
            [{"role": "user", "content": ARITHMETIC_PROMPT}],
            max_tokens=32, temperature=0, non_thinking=True)),
        ("text_selection", 0, lambda: QwenQuality(client).choose(
            "outline", ("curious", "bridge"), (
                "A curious child asks how a bridge is built, then safely crosses it with a parent.",
                "A person eats lunch alone. Nothing else happens.",
            ))),
        ("video_selection", None, lambda: QwenVideoQuality(client, max_frames=args.max_frames).choose(
            args.video_prompt, videos)),
    ]
    try:
        for name, expected, run in cases:
            entry = {"name": name, "generation_start": len(client.generations)}
            started = time.monotonic()
            try:
                result = run()
                entry.update(status="passed", result=result)
                if expected is not None:
                    entry["expected_result"] = expected
                    if result != expected:
                        entry.update(status="failed", error="Completion was valid but did not answer the known test correctly.")
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:4096])
            entry["seconds"] = round(time.monotonic() - started, 3)
            report["cases"].append(entry)
            save()
            print(json.dumps(entry, ensure_ascii=True), flush=True)
    finally:
        runtime.close()
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        save()
    return int(any(case["status"] != "passed" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
