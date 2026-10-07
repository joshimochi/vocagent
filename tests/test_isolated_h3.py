"""H3 subprocess failures must not bypass GPU serialization or artifact checks."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vocagent.benchmark import Benchmark
from vocagent.isolated_h3 import IsolatedMiniMaxH3VideoGenerator
from vocagent.h3_worker import main as worker_main
from vocagent.minimax import H3Settings


class IsolatedH3Tests(unittest.TestCase):
    def runtime(self):
        state = {"active": False}

        @contextmanager
        def use(key, loader, *, measure_load=True):
            self.assertFalse(measure_load, "the parent sentinel must not count as a model load")
            state["active"] = True
            try:
                yield loader()
            finally:
                state["active"] = False

        return SimpleNamespace(backend="musa", use=use), state

    def worker_total(self):
        return {
            "model_load_seconds": 2.0, "model_run_seconds": 3.0,
            "model_load_count": 1, "model_run_count": 1,
            "process_cpu_seconds": 4.0, "work": {"frames": 8, "video_seconds": 8 / 24},
            "gpu_peak_allocated_bytes": {"musa:0": 100},
            "gpu_peak_reserved_bytes": {"musa:0": 200},
        }

    def test_worker_metrics_are_merged_without_counting_startup_wait_or_encoding_as_inference(self):
        runtime, _ = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python")
        clock = [0.0]
        worker_reports = []

        def run(command, **kwargs):
            request = json.loads(Path(command[3]).read_text())
            clock[0] += 9
            worker_report = Path(request["benchmark_output"])
            worker_reports.append(worker_report)
            worker_report.write_text(json.dumps({"status": "completed", "total": self.worker_total()}))
            Path(request["output"]).write_bytes(b"video")

        with tempfile.TemporaryDirectory() as directory, patch(
            "vocagent.benchmark.perf_counter", side_effect=lambda: clock[0],
        ), patch("vocagent.isolated_h3.subprocess.run", side_effect=run):
            benchmark = Benchmark(Path(directory) / "benchmark.json")
            with benchmark.step("video.generation"):
                provider.generate("hello", Path(directory) / "clip.mp4")
            report = benchmark.finish()
        total = report["total"]
        self.assertEqual(total["wall_seconds"], 9)
        self.assertEqual(total["model_load_seconds"], 2)
        self.assertEqual(total["model_run_seconds"], 3)
        self.assertEqual(total["model_load_count"], 1)
        self.assertEqual(total["model_run_count"], 1)
        self.assertEqual(total["worker_cpu_seconds"], 4)
        self.assertEqual(total["gpu_peak_allocated_bytes"], {"musa:0": 100})
        self.assertEqual(total["work"]["frames"], 8)
        self.assertEqual(report["calls"][0]["other_seconds"], 4)
        self.assertEqual(report["calls"][0]["metadata"]["worker_benchmark_status"], "completed")
        self.assertFalse(worker_reports[0].exists())

    def test_worker_failure_and_timeout_preserve_partial_benchmarks(self):
        runtime, state = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python")
        for error in (subprocess.CalledProcessError(1, ["python"]), subprocess.TimeoutExpired(["python"], 1)):
            with tempfile.TemporaryDirectory() as directory, self.subTest(error=error):
                def run(command, **kwargs):
                    request = json.loads(Path(command[3]).read_text())
                    Path(request["benchmark_output"]).write_text(json.dumps({"status": "failed", "total": self.worker_total()}))
                    raise error

                benchmark = Benchmark(Path(directory) / "benchmark.json")
                with patch("vocagent.isolated_h3.subprocess.run", side_effect=run), self.assertRaisesRegex(
                    RuntimeError, "GPU-only H3 worker failed",
                ):
                    with benchmark.step("video.generation"):
                        provider.generate("hello", Path(directory) / "clip.mp4")
                report = benchmark.finish(error=error)
                self.assertEqual(report["total"]["model_load_seconds"], 2)
                self.assertEqual(report["total"]["model_run_seconds"], 3)
                self.assertEqual(report["calls"][0]["status"], "failed")
                self.assertEqual(report["calls"][0]["metadata"]["worker_benchmark_status"], "partial")
                self.assertFalse(state["active"])

    def test_timeout_without_report_marks_measurements_unavailable_and_preserves_failure(self):
        runtime, state = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python")
        error = subprocess.TimeoutExpired(["python"], 1)
        with tempfile.TemporaryDirectory() as directory:
            benchmark = Benchmark(Path(directory) / "benchmark.json")
            with patch("vocagent.isolated_h3.subprocess.run", side_effect=error), self.assertRaisesRegex(
                RuntimeError, "GPU-only H3 worker failed",
            ):
                with benchmark.step("video.generation"):
                    provider.generate("hello", Path(directory) / "clip.mp4")
            report = benchmark.finish(error=error)
        self.assertEqual(report["calls"][0]["metadata"]["worker_benchmark_status"], "unavailable")
        self.assertEqual(report["total"]["model_run_count"], 0)
        self.assertFalse(state["active"])

    def test_successful_worker_requires_report_when_benchmarking(self):
        runtime, _ = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python")
        for malformed in (
            None, {"status": "completed", "total": {}},
            {"status": "failed", "total": self.worker_total()},
            {"status": "running", "total": self.worker_total()},
        ):
            with tempfile.TemporaryDirectory() as directory, self.subTest(malformed=malformed):
                def run(command, **kwargs):
                    request = json.loads(Path(command[3]).read_text())
                    Path(request["output"]).write_bytes(b"video")
                    if malformed is not None:
                        Path(request["benchmark_output"]).write_text(json.dumps(malformed))

                benchmark = Benchmark(Path(directory) / "benchmark.json")
                with patch("vocagent.isolated_h3.subprocess.run", side_effect=run), self.assertRaisesRegex(
                    RuntimeError, "benchmark report",
                ):
                    with benchmark.step("video.generation"):
                        provider.generate("hello", Path(directory) / "clip.mp4")
                self.assertFalse((Path(directory) / "clip.mp4").exists())

    def test_worker_finishes_failure_report_before_releasing_its_runtime(self):
        for error in (None, RuntimeError("GPU allocation failed")):
            with tempfile.TemporaryDirectory() as directory, self.subTest(error=error):
                request_path = Path(directory) / "request.json"
                report_path = Path(directory) / "benchmark.json"
                request_path.write_text(json.dumps({
                    "backend": "musa", "model": "/models/H3", "settings": {},
                    "storyboard": "hello", "output": str(Path(directory) / "video.mp4"),
                    "benchmark_output": str(report_path),
                }))
                runtime = Mock()

                def close():
                    report = json.loads(report_path.read_text())
                    self.assertEqual(report["status"], "failed" if error else "completed")
                    self.assertEqual(report["calls"][0]["status"], "failed" if error else "completed")

                runtime.close.side_effect = close
                generator = Mock()
                generator.generate.side_effect = error
                with patch("vocagent.h3_worker.GPURuntime", return_value=runtime), patch(
                    "vocagent.h3_worker.MiniMaxH3VideoGenerator", return_value=generator,
                ):
                    if error is None:
                        worker_main([str(request_path)])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "GPU allocation failed"):
                            worker_main([str(request_path)])
                runtime.close.assert_called_once_with()

    def test_native_worker_runs_inside_lock_with_literal_unicode_request(self):
        runtime, state = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(
            runtime, "/models/H3", python_executable="/env/bin/python", settings=H3Settings(steps=12),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'clip.mp4'

            def run(command, **kwargs):
                self.assertTrue(state["active"])
                self.assertEqual(command[:3], ["/env/bin/python", "-m", "vocagent.h3_worker"])
                request = json.loads(Path(command[3]).read_text())
                self.assertEqual(request['storyboard'], '说 hello; $(not-a-command)')
                self.assertEqual(request['backend'], 'musa')
                self.assertEqual(request['settings'], {'steps': 12, 'profile': 'quality'})
                self.assertEqual(H3Settings(**request['settings']), provider.settings)
                self.assertNotIn('benchmark_output', request)
                self.assertNotIn('shell', kwargs)
                self.assertNotEqual(Path(request['output']), output)
                Path(request['output']).write_bytes(b'video')

            with patch('vocagent.isolated_h3.subprocess.run', side_effect=run):
                provider.generate('说 hello; $(not-a-command)', output)
            self.assertFalse(state['active'])
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['clip.mp4'])

    def test_preview_profile_reaches_native_worker_with_full_duration(self):
        runtime, state = self.runtime()
        settings = H3Settings(profile='preview', steps=20)
        provider = IsolatedMiniMaxH3VideoGenerator(
            runtime, '/models/H3', python_executable='/env/bin/python', settings=settings,
        )
        worker_runtime = Mock()
        native_generator = Mock()
        native_generator.generate.side_effect = lambda storyboard, output: output.write_bytes(b'preview')

        def run(command, **kwargs):
            self.assertTrue(state['active'])
            request = json.loads(Path(command[3]).read_text())
            self.assertEqual(request['settings'], {'steps': 20, 'profile': 'preview'})
            worker_main([command[3]])

        with tempfile.TemporaryDirectory() as directory, patch(
            'vocagent.isolated_h3.subprocess.run', side_effect=run,
        ), patch('vocagent.h3_worker.GPURuntime', return_value=worker_runtime), patch(
            'vocagent.h3_worker.MiniMaxH3VideoGenerator', return_value=native_generator,
        ) as factory:
            output = Path(directory) / 'clip.mp4'
            provider.generate('An English teaching scene.', output)
            self.assertEqual(output.read_bytes(), b'preview')
            worker_settings = factory.call_args.args[2]
            self.assertEqual(worker_settings, settings)
            self.assertEqual((worker_settings.width, worker_settings.height), (960, 544))
            self.assertEqual(worker_settings.num_frames, 345)
        self.assertFalse(state['active'])
        worker_runtime.close.assert_called_once_with()

    def test_worker_failure_or_absent_output_never_passes(self):
        runtime, state = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python")
        for error in (subprocess.CalledProcessError(1, ['python']), subprocess.TimeoutExpired(['python'], 1), None):
            with tempfile.TemporaryDirectory() as directory, self.subTest(error=error):
                with patch('vocagent.isolated_h3.subprocess.run', side_effect=error), self.assertRaises(RuntimeError):
                    provider.generate('hello', Path(directory) / 'clip.mp4')
                self.assertFalse(state['active'])
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_stale_output_cannot_mask_missing_or_failed_generation(self):
        runtime, state = self.runtime()
        provider = IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python")
        for error in (None, subprocess.TimeoutExpired(['python'], 1)):
            with tempfile.TemporaryDirectory() as directory, self.subTest(error=error):
                output = Path(directory) / "clip.mp4"
                output.write_bytes(b"previous video")

                def run(command, **kwargs):
                    self.assertEqual(output.read_bytes(), b"previous video")
                    if error:
                        request = json.loads(Path(command[3]).read_text())
                        Path(request['output']).write_bytes(b"partial video")
                        raise error

                with patch('vocagent.isolated_h3.subprocess.run', side_effect=run), self.assertRaises(RuntimeError):
                    provider.generate('hello', output)
                self.assertEqual(output.read_bytes(), b"previous video")
                self.assertEqual([path.name for path in Path(directory).iterdir()], ["clip.mp4"])
                self.assertFalse(state['active'])

    def test_worker_closes_runtime_on_success_and_model_failure(self):
        for error in (None, RuntimeError("GPU allocation failed")):
            with tempfile.TemporaryDirectory() as directory, self.subTest(error=error):
                path = Path(directory) / "request.json"
                path.write_text(json.dumps({
                    "backend": "musa", "model": "/models/H3", "settings": {},
                    "storyboard": "hello", "output": str(Path(directory) / "video.mp4"),
                }))
                runtime = Mock()
                generator = Mock()
                generator.generate.side_effect = error
                with patch('vocagent.h3_worker.GPURuntime', return_value=runtime), \
                        patch('vocagent.h3_worker.MiniMaxH3VideoGenerator', return_value=generator):
                    if error is None:
                        worker_main([str(path)])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "GPU allocation failed"):
                            worker_main([str(path)])
                runtime.close.assert_called_once_with()

    def test_invalid_worker_configuration_fails_before_launch(self):
        runtime, _ = self.runtime()
        for timeout in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable="python", timeout=timeout)
        with self.assertRaises(ValueError):
            IsolatedMiniMaxH3VideoGenerator(runtime, "/models/H3", python_executable=" ")
