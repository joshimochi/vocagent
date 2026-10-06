"""Run H3 in its own dependency environment while retaining the GPU residency lock."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import tempfile

from .minimax import MiniMaxH3VideoGenerator


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

    def generate(self, storyboard, output_path):
        if not isinstance(storyboard, str) or not storyboard.strip():
            raise ValueError("H3 requires a nonempty storyboard.")
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # The sentinel owns no weights. Entering evicts the previous family;
        # the lock covers the subprocess until its GPU context has exited.
        with self.runtime.use(("isolated-h3", str(self.model_path)), object):
            with tempfile.TemporaryDirectory(prefix=".h3-request-", dir=output_path.parent) as temporary:
                request = Path(temporary) / "request.json"
                candidate = Path(temporary) / "video.mp4"
                request.write_text(json.dumps({
                    "backend": self.runtime.backend,
                    "model": str(self.model_path.resolve()),
                    "storyboard": storyboard,
                    "settings": asdict(self.settings),
                    "output": str(candidate),
                }, ensure_ascii=False), encoding="utf-8")
                try:
                    subprocess.run(
                        [self.python_executable, "-m", "vocagent.h3_worker", str(request)],
                        check=True, timeout=self.timeout,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise RuntimeError(f"GPU-only H3 worker failed: {exc}") from exc
                if not candidate.is_file() or not candidate.stat().st_size:
                    raise RuntimeError("H3 worker returned without writing a nonempty video.")
                candidate.replace(output_path)
