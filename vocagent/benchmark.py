"""Workflow benchmarks with separate model loading and inference measurements.

Model hooks do no accelerator work unless a workflow step is active in the
current execution context. Step durations are cumulative
when calls overlap; the report's total wall time is actual elapsed time.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import math
from numbers import Real
import os
from pathlib import Path
import platform
import tempfile
from threading import RLock
from time import perf_counter, process_time, thread_time


log = logging.getLogger(__name__)
_CURRENT_STEP = ContextVar("vocagent_benchmark_step", default=None)
_MODEL_FIELDS = (
    "model_load_seconds", "model_run_seconds", "model_load_count", "model_run_count",
)
_GPU_FIELDS = ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")
_THROUGHPUT_COUNTERS = {
    "input_tokens": "input_tokens_per_second",
    "generated_tokens": "generated_tokens_per_second",
    "frames": "frames_per_second",
    "images": "images_per_second",
    "diffusion_steps": "diffusion_steps_per_second",
    "audio_seconds": "audio_seconds_per_second",
    "video_seconds": "video_seconds_per_second",
}


def current_step():
    """Return the benchmark step active in this thread/context, if any."""
    return _CURRENT_STEP.get()


def _counts(counts):
    for name, value in counts.items():
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Benchmark counter {name!r} must be a finite, nonnegative number.")
    return counts


def add_work(**counts):
    """Add quantities such as tokens, frames or media seconds to the active step."""
    step = current_step()
    if step is not None:
        step.add_work(**counts)


def _empty_metrics():
    return {
        "model_load_seconds": 0.0,
        "model_run_seconds": 0.0,
        "model_load_count": 0,
        "model_run_count": 0,
        "worker_cpu_seconds": 0.0,
        "work": {},
        "gpu_peak_allocated_bytes": {},
        "gpu_peak_reserved_bytes": {},
    }


def _merge_peaks(target, source):
    for device, amount in source.items():
        target[device] = max(target.get(device, 0), amount)


def _merge_metrics(target, source):
    for key in (*_MODEL_FIELDS, "worker_cpu_seconds"):
        target[key] += source.get(key, 0)
    for key, amount in source.get("work", {}).items():
        target["work"][key] = target["work"].get(key, 0) + amount
    for key in _GPU_FIELDS:
        _merge_peaks(target[key], source.get(key, {}))


def _throughput(metrics, calls=None, wall_seconds=None, call_count=None):
    rates = {}
    for name, label in _THROUGHPUT_COUNTERS.items():
        if name not in metrics["work"]:
            continue
        # A workflow can run several unrelated models. Token throughput must
        # use token-producing calls, rather than also dividing by video/ASR time.
        seconds = metrics["model_run_seconds"] if calls is None else sum(
            call["model_run_seconds"] for call in calls if call["work"].get(name, 0) > 0
        )
        amount = metrics["work"][name]
        rates[label] = amount / seconds if seconds > 0 else None
        if name in ("audio_seconds", "video_seconds"):
            rates[f"{name.removesuffix('_seconds')}_real_time_factor"] = seconds / amount if amount > 0 and seconds > 0 else None
    if wall_seconds is not None and call_count is not None:
        rates["calls_per_second"] = call_count / wall_seconds if wall_seconds > 0 else None
    return rates


def _latencies(calls):
    values = sorted(call["wall_seconds"] for call in calls)
    if not values:
        return {f"{name}_wall_seconds": None for name in ("min", "mean", "p50", "p95", "max")}

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "min_wall_seconds": values[0],
        "mean_wall_seconds": sum(values) / len(values),
        "p50_wall_seconds": percentile(.5),
        "p95_wall_seconds": percentile(.95),
        "max_wall_seconds": values[-1],
    }


def _error(error):
    return {"type": type(error).__name__, "message": str(error)}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class Step:
    """One workflow operation; repeated names become one aggregate report row."""

    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata
        self._lock = RLock()
        self._metrics = _empty_metrics()
        self._phases = []
        self._outcome = "completed"

    def add_work(self, **counts):
        with self._lock:
            for name, value in _counts(counts).items():
                work = self._metrics["work"]
                work[name] = work.get(name, 0) + value

    def outcome(self, outcome):
        """Record a domain outcome, such as ``passed`` or ``rejected``."""
        with self._lock:
            self._outcome = str(outcome)

    def _add_phase(self, phase):
        with self._lock:
            self._phases.append(phase)
            kind = phase["kind"]
            self._metrics[f"model_{kind}_seconds"] += phase["wall_seconds"]
            self._metrics[f"model_{kind}_count"] += 1
            for key in _GPU_FIELDS:
                _merge_peaks(self._metrics[key], phase[key])

    def _record(self, identifier, wall_seconds, cpu_seconds, error):
        with self._lock:
            metrics = json.loads(json.dumps(self._metrics))
            return {
                "id": identifier,
                "name": self.name,
                "metadata": self.metadata,
                "status": "failed" if error is not None else "completed",
                "outcome": "failed" if error is not None else self._outcome,
                "error": _error(error) if error is not None else None,
                "wall_seconds": wall_seconds,
                "thread_cpu_seconds": cpu_seconds,
                "other_seconds": max(0.0, wall_seconds - metrics["model_load_seconds"] - metrics["model_run_seconds"]),
                **metrics,
                "throughput": _throughput(metrics, wall_seconds=wall_seconds, call_count=1),
                "phases": list(self._phases),
            }


class Benchmark:
    """Collect operation calls and atomically persist ``benchmark.json`` reports."""

    def __init__(self, output_path: Path, metadata: dict | None = None):
        self.output_path = Path(output_path)
        self.metadata = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **(metadata or {}),
        }
        self.started_at = _utc_now()
        self._started = perf_counter()
        self._cpu_started = process_time()
        self._lock = RLock()
        self._next_id = 1
        self._calls = []
        self._finished_at = None
        self._elapsed = None
        self._cpu_elapsed = None
        self._error = None

    @contextmanager
    def step(self, name: str, **metadata):
        step = Step(name, metadata)
        with self._lock:
            identifier = self._next_id
            self._next_id += 1
        started, cpu_started = perf_counter(), thread_time()
        token = _CURRENT_STEP.set(step)
        error = None
        try:
            yield step
        except BaseException as exc:
            error = exc
            raise
        finally:
            elapsed = max(0.0, perf_counter() - started)
            cpu_elapsed = max(0.0, thread_time() - cpu_started)
            _CURRENT_STEP.reset(token)
            try:
                record = step._record(identifier, elapsed, cpu_elapsed, error)
                with self._lock:
                    self._calls.append(record)
                    self._write(self._report())
            except Exception:
                if error is None:
                    raise
                log.warning("Could not save partial benchmark after failed %s", name, exc_info=True)

    def _report(self):
        calls = sorted(self._calls, key=lambda call: call["id"])
        grouped = {}
        total = {
            **_empty_metrics(),
            "calls": len(calls),
            "failed_calls": 0,
            "outcomes": {},
            "wall_seconds": self._elapsed if self._elapsed is not None else max(0.0, perf_counter() - self._started),
            "process_cpu_seconds": self._cpu_elapsed if self._cpu_elapsed is not None else max(0.0, process_time() - self._cpu_started),
            "thread_cpu_seconds": 0.0,
            "cumulative_step_wall_seconds": 0.0,
            "cumulative_other_seconds": 0.0,
        }
        for call in calls:
            name = call["name"]
            if name not in grouped:
                grouped[name] = {
                    "name": name,
                    **_empty_metrics(),
                    "calls": 0,
                    "failed_calls": 0,
                    "outcomes": {},
                    "cumulative_wall_seconds": 0.0,
                    "cumulative_other_seconds": 0.0,
                    "thread_cpu_seconds": 0.0,
                    "min_wall_seconds": call["wall_seconds"],
                    "max_wall_seconds": call["wall_seconds"],
                }
            aggregate = grouped[name]
            aggregate["calls"] += 1
            for target in (aggregate, total):
                _merge_metrics(target, call)
                target["failed_calls"] += call["status"] == "failed"
                target["outcomes"][call["outcome"]] = target["outcomes"].get(call["outcome"], 0) + 1
                target["thread_cpu_seconds"] += call["thread_cpu_seconds"]
                target["cumulative_other_seconds"] += call["other_seconds"]
            aggregate["cumulative_wall_seconds"] += call["wall_seconds"]
            total["cumulative_step_wall_seconds"] += call["wall_seconds"]
            aggregate["min_wall_seconds"] = min(aggregate["min_wall_seconds"], call["wall_seconds"])
            aggregate["max_wall_seconds"] = max(aggregate["max_wall_seconds"], call["wall_seconds"])
        for aggregate in grouped.values():
            related_calls = [call for call in calls if call["name"] == aggregate["name"]]
            aggregate.update(_latencies(related_calls))
            aggregate["throughput"] = _throughput(
                aggregate, related_calls, aggregate["cumulative_wall_seconds"], aggregate["calls"],
            )
        total.update(_latencies(calls))
        total["other_seconds"] = max(0.0, total["wall_seconds"] - total["model_load_seconds"] - total["model_run_seconds"])
        total["throughput"] = _throughput(total, calls, total["wall_seconds"], total["calls"])
        return {
            "schema_version": 1,
            "started_at": self.started_at,
            "finished_at": self._finished_at,
            "status": "running" if self._finished_at is None else ("failed" if self._error is not None else "completed"),
            "error": _error(self._error) if self._error is not None else None,
            "metadata": self.metadata,
            "timing_semantics": {
                "wall_seconds": "Calls use elapsed time; total uses actual workflow elapsed time.",
                "cumulative_seconds": "Step and model durations sum calls and may exceed total wall time when calls overlap.",
                "other_seconds": "Wall time minus model load/run time, clamped at zero; total is a residual estimate when model phases overlap.",
                "thread_cpu_seconds": "CPU time in the thread executing each call; summed across calls.",
                "process_cpu_seconds": "CPU time consumed by all threads in the workflow process.",
                "worker_cpu_seconds": "Additional CPU time reported by isolated model worker processes.",
                "throughput": "Work divided by inference time of calls contributing that work counter; call rates use cumulative step wall time or actual total elapsed time. Null when no applicable time was measured.",
                "gpu_peaks": "Maximum allocated/reserved allocator bytes per device observed at model phase boundaries.",
            },
            "steps": list(grouped.values()),
            "calls": calls,
            "total": total,
        }

    def _write(self, report):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.output_path.parent,
                                             prefix=f".{self.output_path.name}.", suffix=".tmp", delete=False) as handle:
                temporary = handle.name
                json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            os.replace(temporary, self.output_path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def finish(self, error: BaseException | None = None):
        """Save the final report, log a timing table and return its dictionary."""
        with self._lock:
            if self._finished_at is None:
                self._finished_at = _utc_now()
                self._elapsed = max(0.0, perf_counter() - self._started)
                self._cpu_elapsed = max(0.0, process_time() - self._cpu_started)
            if error is not None:
                self._error = error
            report = self._report()
            try:
                self._write(report)
            except Exception:
                if error is None:
                    raise
                log.warning("Could not save benchmark after workflow failure", exc_info=True)
        rows = ["Benchmark seconds: step | cumulative wall | model load | model run | calls"]
        for step in report["steps"]:
            rows.append(f"  {step['name']} | {step['cumulative_wall_seconds']:.3f} | "
                        f"{step['model_load_seconds']:.3f} | {step['model_run_seconds']:.3f} | {step['calls']}")
        total = report["total"]
        rows.append(f"  TOTAL (elapsed wall) | {total['wall_seconds']:.3f} | "
                    f"{total['model_load_seconds']:.3f} | {total['model_run_seconds']:.3f} | {total['calls']}")
        log.info("%s\nSaved benchmark to %s", "\n".join(rows), self.output_path)
        return report


class _Phase:
    def __init__(self, step):
        self.step = step
        self.work = {}

    def add_work(self, **counts):
        if self.step is not None:
            self.step.add_work(**counts)
            for name, value in counts.items():
                self.work[name] = self.work.get(name, 0) + value


def _accelerator(runtime):
    # Do not access runtime.torch: that would import/initialize torch merely to
    # collect a benchmark, including for remote providers and fake runtimes.
    torch = getattr(runtime, "_torch", None)
    backend = getattr(runtime, "backend", None)
    return getattr(torch, backend, None) if torch is not None and isinstance(backend, str) else None


def _gpu_boundary(runtime, reset=False):
    """Best-effort synchronization and allocator samples for every device."""
    allocated, reserved, synchronization_error = {}, {}, None
    accelerator = _accelerator(runtime)
    if accelerator is None:
        return allocated, reserved, synchronization_error
    try:
        available = getattr(accelerator, "is_available", None)
        if callable(available) and not available():
            return allocated, reserved, synchronization_error
        initialized = getattr(accelerator, "is_initialized", None)
        can_reset = callable(initialized) and initialized()
        count = accelerator.device_count()
    except Exception:
        # Unsupported allocator APIs should never make model execution fail.
        log.debug("Could not collect accelerator benchmark metrics", exc_info=True)
        return allocated, reserved, synchronization_error
    for index in range(count):
        # Explicit device indices also cover model shards on other GPUs.
        try:
            accelerator.synchronize(index)
        except Exception as exc:
            # A failed synchronization reports an actual GPU execution error.
            # Keep visiting every device, then let model_phase raise it unless
            # the model body has already raised a more useful original error.
            synchronization_error = synchronization_error or exc
        if reset and can_reset:
            try:
                reset_stats = getattr(accelerator, "reset_peak_memory_stats", None)
                if callable(reset_stats):
                    reset_stats(index)
            except Exception:
                log.debug("Could not reset benchmark allocator peaks", exc_info=True)
        if not reset:
            device = f"{runtime.backend}:{index}"
            for name, peaks in (("max_memory_allocated", allocated), ("max_memory_reserved", reserved)):
                try:
                    sample = getattr(accelerator, name, None)
                    if callable(sample):
                        peaks[device] = max(0, int(sample(index)))
                except Exception:
                    log.debug("Could not sample benchmark allocator peaks", exc_info=True)
    return allocated, reserved, synchronization_error


@contextmanager
def model_phase(kind, runtime, model=None):
    """Measure loading or running a model in the active benchmark step."""
    if kind not in ("load", "run"):
        raise ValueError("Model benchmark phase must be 'load' or 'run'.")
    step = current_step()
    phase = _Phase(step)
    if step is None:
        yield phase
        return
    _, _, synchronization_error = _gpu_boundary(runtime, reset=True)
    if synchronization_error is not None:
        raise synchronization_error
    started, cpu_started = perf_counter(), thread_time()
    error = None
    try:
        yield phase
    except BaseException as exc:
        error = exc
        raise
    finally:
        synchronization_error = None
        try:
            allocated, reserved, synchronization_error = _gpu_boundary(runtime)
            phase_error = error if error is not None else synchronization_error
            record = {
                "kind": kind,
                "model": str(model) if model is not None else None,
                "backend": getattr(runtime, "backend", None),
                "wall_seconds": max(0.0, perf_counter() - started),
                "thread_cpu_seconds": max(0.0, thread_time() - cpu_started),
                "status": "failed" if phase_error is not None else "completed",
                "error": _error(phase_error) if phase_error is not None else None,
                "work": phase.work,
                "gpu_peak_allocated_bytes": allocated,
                "gpu_peak_reserved_bytes": reserved,
            }
            step._add_phase(record)
        except Exception:
            log.debug("Could not record model benchmark phase", exc_info=True)
        if error is None and synchronization_error is not None:
            raise synchronization_error


def merge_model_metrics(total: dict):
    """Merge isolated-worker model metrics without counting its enclosing call.

    Parent call timing already includes the worker's elapsed time. Only model
    phase durations/counts, work and device peaks cross the process boundary.
    """
    step = current_step()
    if step is None:
        return
    if not isinstance(total, dict) or any(key not in total for key in _MODEL_FIELDS):
        raise ValueError("Worker benchmark is missing model timing/count metrics.")
    _counts({key: total[key] for key in _MODEL_FIELDS})
    for key in ("model_load_count", "model_run_count"):
        if not isinstance(total[key], int) or isinstance(total[key], bool):
            raise ValueError("Worker benchmark model phase counts must be integers.")
    work = total.get("work", {})
    if not isinstance(work, dict):
        raise ValueError("Worker benchmark work must be an object.")
    _counts(work)
    for key in _GPU_FIELDS:
        peaks = total.get(key, {})
        if not isinstance(peaks, dict):
            raise ValueError("Worker benchmark GPU peaks must be an object.")
        _counts(peaks)
    _counts({key: total.get(key, 0) for key in ("process_cpu_seconds", "worker_cpu_seconds")})
    metrics = {
        **{key: total.get(key, 0) for key in _MODEL_FIELDS},
        "work": total.get("work", {}),
        **{key: total.get(key, {}) for key in _GPU_FIELDS},
        "worker_cpu_seconds": total.get("process_cpu_seconds", 0) + total.get("worker_cpu_seconds", 0),
    }
    with step._lock:
        _merge_metrics(step._metrics, metrics)
