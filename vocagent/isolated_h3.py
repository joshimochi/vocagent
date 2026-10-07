"""Run H3 in its own dependency environment while retaining the GPU residency lock."""

from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import subprocess
import tempfile

from .benchmark import current_step, merge_model_metrics
from .minimax import MiniMaxH3VideoGenerator


log = logging.getLogger(__name__)


class IsolatedMiniMaxH3VideoGenerator(MiniMaxH3VideoGenerator):
    def __init__(self, runtime, model_path, *, python_executable, settings=None, timeout=14400):
        super().__init__(runtime, model_path, settings)
        if not str(python_executable).strip():
            raise ValueError("H3 worker Python executable must be nonempty.")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("H3 worker timeout must be finite and positive.")
        self.python_executable = str(python_executable)
        self.timeout = timeout

    def generate(self, video_prompt, output_path):
        if not isinstance(video_prompt, str) or not video_prompt.strip():
            raise ValueError("H3 requires a nonempty video prompt.")
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # The sentinel owns no weights. Entering evicts the previous family;
        # the lock covers the subprocess until its GPU context has exited.
        with self.runtime.use(("isolated-h3", str(self.model_path)), object, measure_load=False):
            with tempfile.TemporaryDirectory(prefix=".h3-request-", dir=output_path.parent) as temporary:
                request = Path(temporary) / "request.json"
                candidate = Path(temporary) / "video.mp4"
                payload = {
                    "backend": self.runtime.backend,
                    "model": str(self.model_path.resolve()),
                    "video_prompt": video_prompt,
                    "settings": asdict(self.settings),
                    "output": str(candidate),
                }
                step = current_step()
                report_path = Path(temporary) / "benchmark.json" if step is not None else None
                if report_path is not None:
                    payload["benchmark_output"] = str(report_path)
                    step.metadata["worker_benchmark_status"] = "unavailable"
                request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                worker_error = None
                try:
                    try:
                        subprocess.run(
                            [self.python_executable, "-m", "vocagent.h3_worker", str(request)],
                            check=True, timeout=self.timeout,
                        )
                    except (OSError, subprocess.SubprocessError) as exc:
                        raise RuntimeError(f"GPU-only H3 worker failed: {exc}") from exc
                except BaseException as exc:
                    worker_error = exc
                    raise
                finally:
                    # Import the worker's real load/inference phases before its
                    # temporary directory disappears, including partial failures.
                    if report_path is not None:
                        if report_path.is_file():
                            try:
                                report = json.loads(report_path.read_text(encoding="utf-8"))
                                if worker_error is None and report.get("status") != "completed":
                                    raise ValueError("Successful H3 worker benchmark must be completed.")
                                total = report["total"]
                                if not isinstance(total, dict):
                                    raise ValueError("Worker benchmark total must be an object.")
                                merge_model_metrics(total)
                                step.metadata["worker_benchmark_status"] = (
                                    "completed" if report.get("status") == "completed" else "partial"
                                )
                            except Exception as exc:
                                if worker_error is None:
                                    raise RuntimeError("H3 worker returned an invalid benchmark report.") from exc
                                log.warning("Could not read failed H3 worker benchmark: %s", exc)
                        elif worker_error is None:
                            raise RuntimeError("H3 worker returned without writing its benchmark report.")
                if not candidate.is_file() or not candidate.stat().st_size:
                    raise RuntimeError("H3 worker returned without writing a nonempty video.")
                candidate.replace(output_path)
