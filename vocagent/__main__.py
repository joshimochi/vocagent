"""Command-line entry point for the complete GPU-only local workflow."""

import argparse
import importlib
import logging
import sys
from pathlib import Path

from .models import VideoPromptResult
from .pipeline import Workflow
from .providers import Providers
from .selection import HumanInterventionRequired
from .vocabulary import normalize_target_word


DEFAULT_PROVIDER_FACTORY = "vocagent.local:create_providers"


def positive_integer(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def english_word(value: str) -> str:
    try:
        return normalize_target_word(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def load_providers(spec: str) -> Providers:
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("Provider factory must have the form module_name:function_name.")
    factory = getattr(importlib.import_module(module_name), factory_name)
    providers = factory()
    if not isinstance(providers, Providers):
        raise TypeError("Provider factory must return a Providers instance.")
    return providers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vocagent", description="Generate a video prompt or subtitled video teaching one English word.",
    )
    parser.add_argument("word", type=english_word, help="One English word to teach")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                        help="New or empty output directory (default: output)")
    parser.add_argument("--providers", metavar="MODULE:FACTORY",
                        default=DEFAULT_PROVIDER_FACTORY,
                        help="Provider factory (default: vocagent.local:create_providers)")
    parser.add_argument("--stop-after", choices=("video-prompt", "video"), default="video",
                        help="Last stage to run (default: video)")
    parser.add_argument("--outline-candidate-count", type=positive_integer, default=2,
                        help="Number of outline candidate slots (default: 2)")
    parser.add_argument("--video-prompt-candidate-count", type=positive_integer, default=2,
                        help="Number of video prompt candidate slots (default: 2)")
    parser.add_argument("--video-candidate-count", type=positive_integer, default=2,
                        help="Number of video candidate slots (default: 2)")
    parser.add_argument("--concurrent-candidates", action="store_true",
                        help="Generate candidate slots concurrently (default: sequential)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    providers = None
    try:
        providers = load_providers(args.providers)
        result = Workflow(
            providers,
            outline_candidate_count=args.outline_candidate_count,
            video_prompt_candidate_count=args.video_prompt_candidate_count,
            video_candidate_count=args.video_candidate_count,
            concurrent_candidates=args.concurrent_candidates,
        ).run(args.word, args.output_dir, stop_after=args.stop_after.replace("-", "_"))
    except HumanInterventionRequired as exc:
        print(f"Human intervention required: {exc}", file=sys.stderr)
        return 2
    except (NotImplementedError, OSError, RuntimeError, ValueError, TypeError,
            ImportError, AttributeError) as exc:
        print(f"Workflow stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if providers is not None:
            providers.close()
    print(result.video_prompt if isinstance(result, VideoPromptResult) else result.final_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
