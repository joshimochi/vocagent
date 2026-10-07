"""Serialized, GPU-only model residency for the local hs providers."""

from contextlib import contextmanager, nullcontext
import gc
import logging
from threading import RLock

from .benchmark import model_phase


log = logging.getLogger(__name__)


class GPURuntime:
    """Keep one model family resident; discard its weights before loading another.

    CPU tokenization, decoding and checkpoint deserialization are permitted.
    Model parameters and inference caches must remain on the selected GPU backend.
    """

    def __init__(self, backend="musa"):
        if backend not in ("musa", "cuda"):
            raise ValueError("Inference backend must be musa or cuda; CPU fallback is disabled.")
        self.backend = backend
        self._lock = RLock()
        self._resource = None
        self._key = None
        self._torch = None

    @property
    def torch(self):
        if self._torch is None:
            import torch
            if self.backend == "musa":
                import torch_musa  # noqa: F401: registers the MUSA backend
            accelerator = getattr(torch, self.backend, None)
            if accelerator is None or not accelerator.is_available():
                raise RuntimeError(f"{self.backend} GPU is unavailable; CPU fallback is disabled.")
            self._torch = torch
        return self._torch

    @property
    def device(self):
        return f"{self.backend}:0"

    @property
    def dtype(self):
        return self.torch.bfloat16

    def assert_on_accelerator(self, model):
        bad = [(name, str(parameter.device)) for name, parameter in model.named_parameters()
               if parameter.device.type != self.backend]
        if bad:
            raise RuntimeError(f"CPU/disk/meta model placement is forbidden: {bad[:5]}")
        for placement in getattr(model, "hf_device_map", {}).values():
            if str(placement) in ("cpu", "disk", "meta"):
                raise RuntimeError("CPU/disk model offloading is forbidden.")
        # Accelerate attaches placement hooks to leaf modules, and may wrap
        # several hooks in SequentialHook. Checking only the root misses those.
        modules = model.modules() if hasattr(model, "modules") else (model,)
        seen_hooks = set()
        for module in modules:
            hooks = [getattr(module, "_hf_hook", None)]
            while hooks:
                hook = hooks.pop()
                if hook is None or id(hook) in seen_hooks:
                    continue
                seen_hooks.add(id(hook))
                if getattr(hook, "offload", False):
                    raise RuntimeError("Model weight offloading is forbidden.")
                hooks.extend(getattr(hook, "hooks", ()))

    def _release(self):
        if self._resource is None:
            return
        log.info("Unloading %s", self._key)
        self._resource = None
        self._key = None
        gc.collect()
        accelerator = getattr(self.torch, self.backend)
        for index in range(accelerator.device_count()):
            with accelerator.device(index):
                accelerator.synchronize()
                accelerator.empty_cache()

    @contextmanager
    def use(self, key, loader, *, measure_load=True):
        # Hold the lock throughout inference. Concurrent candidate threads may
        # preprocess in parallel but cannot evict a model another thread is using.
        with self._lock:
            if key != self._key:
                self._release()
                log.info("Loading %s on GPU", key)
                try:
                    phase = model_phase("load", self, model=str(key)) if measure_load else nullcontext()
                    with phase:
                        self._resource = loader()
                except BaseException:
                    self._resource = None
                    gc.collect()
                    if self._torch is not None:
                        getattr(self._torch, self.backend).empty_cache()
                    raise
                self._key = key
            yield self._resource

    def close(self):
        with self._lock:
            self._release()

    def device_map(self, model_class, config):
        """Plan from empty weights; reject offload before reading real weights."""
        from accelerate import init_empty_weights, infer_auto_device_map
        torch = self.torch
        accelerator = getattr(torch, self.backend)
        budget = {}
        for index in range(accelerator.device_count()):
            free, total = accelerator.mem_get_info(index)
            # Leave activation, vision encoder and KV-cache headroom on every GPU.
            budget[index] = max(0, min(int(total * .82), free - 6 * 1024**3))
        budget["cpu"] = 0
        with init_empty_weights():
            empty = model_class(config)
            empty.tie_weights()
        mapping = infer_auto_device_map(
            empty, max_memory=budget, dtype=self.dtype,
            no_split_module_classes=getattr(empty, "_no_split_modules", []),
        )
        del empty
        if any(str(device) in ("cpu", "disk", "meta") for device in mapping.values()):
            raise RuntimeError("Model cannot fit in available GPU memory without offloading.")
        return {name: f"{self.backend}:{device}" if isinstance(device, int) else device
                for name, device in mapping.items()}
