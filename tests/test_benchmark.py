"""Benchmark accounting without loading models or requiring GPU libraries."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vocagent.benchmark import Benchmark, add_work, current_step, merge_model_metrics, model_phase


class Clock:
    def __init__(self):
        self.wall = 0.0
        self.cpu = 0.0

    def advance(self, seconds, cpu=0.0):
        self.wall += seconds
        self.cpu += cpu


class Accelerator:
    def __init__(self, initialized=True):
        self.initialized = initialized
        self.synchronized = []
        self.reset = []
        self.allocated = [100, 200]
        self.reserved = [300, 400]

    def is_available(self):
        return True

    def is_initialized(self):
        return self.initialized

    def device_count(self):
        return 2

    def synchronize(self, device):
        self.synchronized.append(device)

    def reset_peak_memory_stats(self, device):
        self.reset.append(device)

    def max_memory_allocated(self, device):
        return self.allocated[device]

    def max_memory_reserved(self, device):
        return self.reserved[device]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "nested" / "benchmark.json"
        self.clock = Clock()
        for name, getter in (
            ("perf_counter", lambda: self.clock.wall),
            ("thread_time", lambda: self.clock.cpu),
            ("process_time", lambda: self.clock.cpu),
        ):
            replacement = patch(f"vocagent.benchmark.{name}", getter)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.runtime = SimpleNamespace(backend="cuda", _torch=None)

    def read_report(self):
        return json.loads(self.output.read_text(encoding="utf-8"))

    def test_model_split_cache_reuse_latencies_and_unit_specific_total_throughput(self):
        benchmark = Benchmark(self.output, metadata={"backend": "cuda"})
        self.clock.advance(2, .2)
        with benchmark.step("text", candidate=1) as step:
            with model_phase("load", self.runtime, "chat"):
                self.clock.advance(2, .1)
            with model_phase("run", self.runtime, "chat") as phase:
                self.clock.advance(3, .4)
                phase.add_work(input_tokens=10, generated_tokens=30)
            self.clock.advance(1, .1)
            step.outcome("passed")
        partial = self.read_report()
        self.assertEqual(partial["status"], "running")
        self.assertEqual(partial["total"]["calls"], 1)
        with benchmark.step("text", candidate=2) as step:
            # Reusing resident weights records only a run phase.
            with model_phase("run", self.runtime, "chat"):
                self.clock.advance(2, .3)
            add_work(input_tokens=10, generated_tokens=20)
            self.clock.advance(2, .1)
            step.outcome("rejected")
        with benchmark.step("video"):
            with model_phase("load", self.runtime, "video"):
                self.clock.advance(1, .1)
            with model_phase("run", self.runtime, "video") as phase:
                self.clock.advance(10, .5)
                phase.add_work(frames=120, video_seconds=5, diffusion_steps=30, images=1)
        with benchmark.step("speech"):
            with model_phase("run", self.runtime, "asr") as phase:
                self.clock.advance(4, .4)
                phase.add_work(audio_seconds=20)
            self.clock.advance(1, .1)
        self.clock.advance(3, .2)
        report = benchmark.finish()
        total = report["total"]
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["metadata"]["backend"], "cuda")
        self.assertIn("python", report["metadata"])
        self.assertEqual([row["name"] for row in report["steps"]], ["text", "video", "speech"])
        self.assertEqual(total["wall_seconds"], 31)
        self.assertEqual(total["cumulative_step_wall_seconds"], 26)
        self.assertEqual(total["model_load_seconds"], 3)
        self.assertEqual(total["model_run_seconds"], 19)
        self.assertEqual(total["other_seconds"], 9)
        self.assertEqual(total["cumulative_other_seconds"], 4)
        self.assertEqual(total["model_load_count"], 2)
        self.assertEqual(total["model_run_count"], 4)
        self.assertAlmostEqual(total["process_cpu_seconds"], 2.5)
        self.assertAlmostEqual(total["thread_cpu_seconds"], 2.1)
        rates = total["throughput"]
        self.assertEqual(rates["generated_tokens_per_second"], 10)
        self.assertEqual(rates["input_tokens_per_second"], 4)
        self.assertEqual(rates["frames_per_second"], 12)
        self.assertEqual(rates["video_real_time_factor"], 2)
        self.assertEqual(rates["audio_real_time_factor"], .2)
        self.assertEqual(rates["images_per_second"], .1)
        self.assertEqual(rates["diffusion_steps_per_second"], 3)
        self.assertAlmostEqual(rates["calls_per_second"], 4 / 31)
        text = report["steps"][0]
        self.assertEqual(text["outcomes"], {"passed": 1, "rejected": 1})
        self.assertEqual(text["cumulative_wall_seconds"], 10)
        self.assertEqual(text["min_wall_seconds"], 4)
        self.assertEqual(text["mean_wall_seconds"], 5)
        self.assertEqual(text["p50_wall_seconds"], 5)
        self.assertAlmostEqual(text["p95_wall_seconds"], 5.9)
        self.assertEqual(text["max_wall_seconds"], 6)
        cached = report["calls"][1]
        self.assertEqual(cached["model_load_seconds"], 0)
        self.assertEqual(cached["model_load_count"], 0)
        self.assertEqual(cached["metadata"], {"candidate": 2})
        self.assertEqual(self.read_report(), report)
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])
        self.clock.advance(100)
        self.assertEqual(benchmark.finish()["total"]["wall_seconds"], 31)

    def test_parallel_calls_keep_context_and_distinguish_elapsed_from_cumulative_wall(self):
        benchmark = Benchmark(self.output)
        entered = Barrier(3)
        finish = Event()

        def run(candidate, tokens):
            with benchmark.step("chat", candidate=candidate) as step:
                with model_phase("run", self.runtime):
                    self.assertIs(current_step(), step)
                    entered.wait(timeout=5)
                    self.assertTrue(finish.wait(5))
                    add_work(generated_tokens=tokens)
                step.outcome(f"candidate-{candidate}")
            self.assertIsNone(current_step())

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(run, 1, 10)
            second = executor.submit(run, 2, 20)
            try:
                entered.wait(timeout=5)
                self.assertIsNone(current_step())
                self.clock.advance(5)
            finally:
                finish.set()
            first.result(timeout=5)
            second.result(timeout=5)
        report = benchmark.finish()
        total = report["total"]
        self.assertEqual(total["wall_seconds"], 5)
        self.assertEqual(total["cumulative_step_wall_seconds"], 10)
        self.assertEqual(total["model_run_seconds"], 10)
        self.assertEqual(total["work"]["generated_tokens"], 30)
        self.assertEqual(total["throughput"]["calls_per_second"], .4)
        self.assertEqual(report["steps"][0]["throughput"]["calls_per_second"], .2)
        self.assertEqual({call["metadata"]["candidate"]: call["work"]["generated_tokens"]
                          for call in report["calls"]}, {1: 10, 2: 20})

    def test_initial_loading_syncs_after_torch_initialization_and_merges_device_peaks(self):
        benchmark = Benchmark(self.output)
        accelerator = Accelerator()
        with benchmark.step("video"):
            with model_phase("load", self.runtime):
                self.runtime._torch = SimpleNamespace(cuda=accelerator)
                self.clock.advance(2)
            self.assertEqual(accelerator.synchronized, [0, 1])
            self.assertEqual(accelerator.reset, [])
            with model_phase("run", self.runtime):
                accelerator.allocated = [150, 180]
                accelerator.reserved = [250, 500]
                self.clock.advance(3)
        total = benchmark.finish()["total"]
        self.assertEqual(accelerator.synchronized, [0, 1, 0, 1, 0, 1])
        self.assertEqual(accelerator.reset, [0, 1])
        self.assertEqual(total["gpu_peak_allocated_bytes"], {"cuda:0": 150, "cuda:1": 200})
        self.assertEqual(total["gpu_peak_reserved_bytes"], {"cuda:0": 300, "cuda:1": 500})

    def test_uninitialized_allocator_and_unsupported_stats_do_not_skip_other_devices(self):
        accelerator = Accelerator(initialized=False)
        self.runtime._torch = SimpleNamespace(cuda=accelerator)
        benchmark = Benchmark(self.output)
        with benchmark.step("load"):
            with model_phase("load", self.runtime):
                self.clock.advance(1)
        self.assertEqual(accelerator.reset, [])
        accelerator.initialized = True

        def unsupported_reset(device):
            accelerator.reset.append(device)
            if device == 0:
                raise RuntimeError("allocator unavailable")

        accelerator.reset_peak_memory_stats = unsupported_reset
        with benchmark.step("run"):
            with model_phase("run", self.runtime):
                self.clock.advance(1)
        self.assertEqual(accelerator.reset, [0, 1])
        self.assertEqual(accelerator.synchronized, [0, 1] * 4)
        self.assertEqual(benchmark.finish()["total"]["model_run_seconds"], 1)

    def test_gpu_sync_failure_stops_successful_call_and_preserves_model_body_error(self):
        accelerator = Accelerator()
        self.runtime._torch = SimpleNamespace(cuda=accelerator)
        benchmark = Benchmark(self.output)
        synchronize = accelerator.synchronize

        def fail_after_enter(device):
            synchronize(device)
            if len(accelerator.synchronized) > 2:
                raise RuntimeError("GPU launch failure")

        accelerator.synchronize = fail_after_enter
        with self.assertRaisesRegex(RuntimeError, "GPU launch failure"):
            with benchmark.step("chat"):
                with model_phase("run", self.runtime):
                    self.clock.advance(3)
        self.assertEqual(accelerator.synchronized, [0, 1, 0, 1])
        partial = self.read_report()
        self.assertEqual(partial["total"]["failed_calls"], 1)
        self.assertEqual(partial["calls"][0]["phases"][0]["status"], "failed")

        accelerator.synchronized.clear()
        original = ValueError("model failed first")
        try:
            with benchmark.step("chat"):
                with model_phase("run", self.runtime):
                    self.clock.advance(2)
                    raise original
        except ValueError as exc:
            self.assertIs(exc, original)
        else:
            self.fail("Model exception was swallowed")
        report = benchmark.finish(error=original)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], {"type": "ValueError", "message": "model failed first"})
        self.assertEqual(report["total"]["failed_calls"], 2)
        self.assertEqual(report["total"]["model_run_seconds"], 5)
        self.assertEqual(report["calls"][1]["phases"][0]["error"]["message"], "model failed first")

    def test_worker_merge_counts_only_model_metrics_and_worker_cpu(self):
        benchmark = Benchmark(self.output)
        worker = {
            "wall_seconds": 999, "calls": 999, "failed_calls": 999,
            "thread_cpu_seconds": 99, "process_cpu_seconds": 1.5,
            "model_load_seconds": 2, "model_run_seconds": 3,
            "model_load_count": 1, "model_run_count": 1,
            "work": {"frames": 60, "video_seconds": 2},
            "gpu_peak_allocated_bytes": {"musa:0": 1000},
            "gpu_peak_reserved_bytes": {"musa:0": 1500},
        }
        with benchmark.step("video"):
            merge_model_metrics(worker)
            self.clock.advance(7, .25)
        report = benchmark.finish()
        total = report["total"]
        self.assertEqual(total["calls"], 1)
        self.assertEqual(total["failed_calls"], 0)
        self.assertEqual(total["wall_seconds"], 7)
        self.assertEqual(total["other_seconds"], 2)
        self.assertEqual(total["thread_cpu_seconds"], .25)
        self.assertEqual(total["process_cpu_seconds"], .25)
        self.assertEqual(total["worker_cpu_seconds"], 1.5)
        self.assertEqual(total["model_run_seconds"], 3)
        self.assertEqual(total["work"], worker["work"])
        self.assertEqual(total["gpu_peak_allocated_bytes"], {"musa:0": 1000})
        self.assertEqual(report["calls"][0]["phases"], [])

    def test_invalid_worker_metrics_are_rejected_before_partial_merge(self):
        benchmark = Benchmark(self.output)
        valid = {"model_load_seconds": 0, "model_run_seconds": 2,
                 "model_load_count": 0, "model_run_count": 1}
        with benchmark.step("worker"):
            for invalid in ({}, {**valid, "model_run_seconds": -1},
                            {**valid, "model_run_count": 1.5},
                            {**valid, "work": {"frames": float("nan")}}):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    merge_model_metrics(invalid)
        self.assertEqual(benchmark.finish()["total"]["model_run_seconds"], 0)

    def test_outside_step_does_no_gpu_work_and_empty_report_is_zero_safe(self):
        class UnusableRuntime:
            @property
            def _torch(self):
                raise AssertionError("Unbenchmarked operation accessed torch")

        with model_phase("run", UnusableRuntime()) as phase:
            phase.add_work(generated_tokens=4)
            add_work(generated_tokens=4)
            merge_model_metrics({})
        benchmark = Benchmark(self.output)
        report = benchmark.finish()
        self.assertEqual(report["steps"], [])
        self.assertEqual(report["calls"], [])
        self.assertEqual(report["total"]["wall_seconds"], 0)
        self.assertIsNone(report["total"]["throughput"]["calls_per_second"])
        self.assertIsNone(report["total"]["p95_wall_seconds"])

    def test_partial_report_write_failure_preserves_original_model_error(self):
        benchmark = Benchmark(self.output)
        original = RuntimeError("model failure")
        with patch.object(benchmark, "_write", side_effect=OSError("disk failure")):
            with self.assertLogs("vocagent.benchmark", level="WARNING"):
                with self.assertRaisesRegex(RuntimeError, "model failure"):
                    with benchmark.step("chat"):
                        raise original
                report = benchmark.finish(error=original)
        self.assertEqual(report["total"]["failed_calls"], 1)


if __name__ == "__main__":
    unittest.main()
