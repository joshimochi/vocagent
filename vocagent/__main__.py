"""Command-line entry point with Qwen text providers and optional custom providers."""

import argparse
import importlib
import sys
from pathlib import Path

from .models import StoryboardResult
from .pipeline import Workflow
from .providers import Providers
from .selection import HumanInterventionRequired


DEFAULT_PROVIDER_FACTORY = "vocagent.qwen:create_providers"


def positive_integer(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


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
        prog="vocagent", description="Generate a storyboard or subtitled video from vocabulary.",
    )
    parser.add_argument("vocabulary", nargs="+", help="Words or quoted phrases to include")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                        help="New or empty output directory (default: output)")
    parser.add_argument("--providers", metavar="MODULE:FACTORY",
                        default=DEFAULT_PROVIDER_FACTORY,
                        help="Provider factory (default: vocagent.qwen:create_providers)")
    parser.add_argument("--stop-after", choices=("storyboard", "video"),
                        help="Last stage to run (default: storyboard for built-in Qwen providers; "
                             "video for custom providers)")
    parser.add_argument("--outline-candidate-count", type=positive_integer, default=2,
                        help="Number of outline candidate slots (default: 2)")
    parser.add_argument("--storyboard-candidate-count", type=positive_integer, default=2,
                        help="Number of storyboard candidate slots (default: 2)")
    parser.add_argument("--video-candidate-count", type=positive_integer, default=2,
                        help="Number of video candidate slots (default: 2)")
    parser.add_argument("--concurrent-candidates", action="store_true",
                        help="Generate candidate slots concurrently (default: sequential)")
    args = parser.parse_args(argv)
    stop_after = args.stop_after or (
        "storyboard" if args.providers == DEFAULT_PROVIDER_FACTORY else "video"
    )
    try:
        providers = load_providers(args.providers)
        result = Workflow(
            providers,
            outline_candidate_count=args.outline_candidate_count,
            storyboard_candidate_count=args.storyboard_candidate_count,
            video_candidate_count=args.video_candidate_count,
            concurrent_candidates=args.concurrent_candidates,
        ).run(args.vocabulary, args.output_dir, stop_after=stop_after)
    except HumanInterventionRequired as exc:
        print(f"Human intervention required: {exc}", file=sys.stderr)
        return 2
    except (NotImplementedError, OSError, RuntimeError, ValueError, TypeError,
            ImportError, AttributeError) as exc:
        print(f"Workflow stopped: {exc}", file=sys.stderr)
        return 1
    print(result.storyboard if isinstance(result, StoryboardResult) else result.final_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
