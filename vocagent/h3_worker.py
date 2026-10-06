"""H3 worker isolated from Qwen-ASR's incompatible Transformers dependency."""

import argparse
import json
import logging
from pathlib import Path

from .gpu_runtime import GPURuntime
from .minimax import H3Settings, MiniMaxH3VideoGenerator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s H3 %(message)s")
    runtime = GPURuntime(request["backend"])
    try:
        generator = MiniMaxH3VideoGenerator(
            runtime, request["model"], H3Settings(**request["settings"]),
        )
        generator.generate(request["storyboard"], Path(request["output"]))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
