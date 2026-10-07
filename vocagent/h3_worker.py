"""H3 worker isolated from Qwen-ASR's incompatible Transformers dependency."""

import argparse
from contextlib import nullcontext
import json
import logging
from pathlib import Path

from .benchmark import Benchmark
from .gpu_runtime import GPURuntime
from .minimax import H3Settings, MiniMaxH3VideoGenerator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s H3 %(message)s")
    benchmark = Benchmark(
        Path(request["benchmark_output"]), metadata={"worker": "h3", "backend": request["backend"]},
    ) if request.get("benchmark_output") else None
    runtime = GPURuntime(request["backend"])
    error = None
    try:
        with benchmark.step("video.generation") if benchmark else nullcontext():
            generator = MiniMaxH3VideoGenerator(
                runtime, request["model"], H3Settings(**request["settings"]),
            )
            generator.generate(request["storyboard"], Path(request["output"]))
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if benchmark:
                benchmark.finish(error=error)
        finally:
            runtime.close()


if __name__ == "__main__":
    main()
