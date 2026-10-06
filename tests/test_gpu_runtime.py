"""GPU placement and serialized model residency, without accelerator dependencies."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import weakref

from vocagent.gpu_runtime import GPURuntime


GIB = 1024**3


def runtime_with_accelerator():
    accelerator = MagicMock()
    accelerator.device_count.return_value = 2
    accelerator.device.side_effect = lambda _: nullcontext()
    accelerator.mem_get_info.side_effect = [(45 * GIB, 48 * GIB), (10 * GIB, 48 * GIB)]
    runtime = GPURuntime("musa")
    runtime._torch = SimpleNamespace(musa=accelerator, bfloat16="bfloat16")
    return runtime, accelerator


class GPURuntimeTests(unittest.TestCase):
    def test_unsupported_backend_and_unavailable_gpu_never_use_cpu(self):
        with self.assertRaises(ValueError):
            GPURuntime("cpu")
        for backend in ("cuda", "musa"):
            torch = SimpleNamespace(**{backend: SimpleNamespace(is_available=lambda: False)})
            with self.subTest(backend=backend), patch.dict(
                "sys.modules", {"torch": torch, "torch_musa": SimpleNamespace()},
            ), self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                _ = GPURuntime(backend).torch

    def test_repeated_calls_reuse_weights_and_switching_destroys_previous_before_loading(self):
        runtime, accelerator = runtime_with_accelerator()
        references = []
        loads = []

        class Resource:
            pass

        def loader(name):
            if references:
                self.assertIsNone(references[-1](), "old model still resident while replacement loads")
            resource = Resource()
            references.append(weakref.ref(resource))
            loads.append(name)
            return resource

        with runtime.use("vl", lambda: loader("vl")):
            self.assertIsNotNone(references[0]())
        with runtime.use("vl", lambda: self.fail("same model loaded twice")):
            self.assertIsNotNone(references[0]())
        accelerator.empty_cache.assert_not_called()
        with runtime.use("guard", lambda: loader("guard")):
            self.assertIsNone(references[0]())
            self.assertIsNotNone(references[1]())
        self.assertEqual(loads, ["vl", "guard"])
        self.assertEqual(accelerator.synchronize.call_count, 2)
        self.assertEqual(accelerator.empty_cache.call_count, 2)
        runtime.close()
        self.assertIsNone(references[1]())
        self.assertEqual(accelerator.empty_cache.call_count, 4)
        runtime.close()
        self.assertEqual(accelerator.empty_cache.call_count, 4)

    def test_another_candidate_cannot_evict_weights_during_inference(self):
        runtime, _ = runtime_with_accelerator()
        first_active = Event()
        second_attempting = Event()
        allow_first_exit = Event()
        second_loaded = Event()

        def first():
            with runtime.use("vl", object):
                first_active.set()
                self.assertTrue(allow_first_exit.wait(5))
                self.assertFalse(second_loaded.is_set())

        def second():
            self.assertTrue(first_active.wait(5))
            second_attempting.set()

            def loader():
                second_loaded.set()
                return object()

            with runtime.use("guard", loader):
                pass

        with ThreadPoolExecutor(max_workers=2) as executor:
            one = executor.submit(first)
            two = executor.submit(second)
            try:
                self.assertTrue(second_attempting.wait(5))
                self.assertFalse(second_loaded.is_set())
            finally:
                allow_first_exit.set()
            one.result(timeout=5)
            two.result(timeout=5)
        self.assertTrue(second_loaded.is_set())

    def test_failed_loader_is_not_cached_and_can_be_retried(self):
        runtime, _ = runtime_with_accelerator()
        loader = MagicMock(side_effect=[RuntimeError("out of memory"), object()])
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            with runtime.use("vl", loader):
                self.fail("failed loader yielded")
        with runtime.use("vl", loader) as resource:
            self.assertIsNotNone(resource)
        self.assertEqual(loader.call_count, 2)

    def test_parameter_placement_device_map_and_offload_hooks_are_checked(self):
        runtime, _ = runtime_with_accelerator()
        for device in ("cpu", "meta", "cuda"):
            model = SimpleNamespace(named_parameters=lambda: [
                ("weight", SimpleNamespace(device=SimpleNamespace(type=device))),
            ])
            with self.subTest(device=device), self.assertRaises(RuntimeError):
                runtime.assert_on_accelerator(model)
        for placement in ("cpu", "disk", "meta"):
            model = SimpleNamespace(named_parameters=lambda: [], hf_device_map={"layer": placement})
            with self.subTest(placement=placement), self.assertRaises(RuntimeError):
                runtime.assert_on_accelerator(model)
        model = SimpleNamespace(named_parameters=lambda: [], _hf_hook=SimpleNamespace(offload=True))
        with self.assertRaisesRegex(RuntimeError, "offloading"):
            runtime.assert_on_accelerator(model)
        runtime.assert_on_accelerator(SimpleNamespace(named_parameters=lambda: [
            ("weight", SimpleNamespace(device=SimpleNamespace(type="musa"))),
        ], hf_device_map={"layer": "musa:1"}))

    def test_device_planning_reserves_headroom_and_maps_to_selected_gpu_backend(self):
        runtime, _ = runtime_with_accelerator()
        empty = SimpleNamespace(tie_weights=MagicMock(), _no_split_modules=["DecoderLayer"])
        model_class = MagicMock(return_value=empty)
        infer = MagicMock(return_value={"visual": 0, "language": 1})
        config = object()
        with patch.dict("sys.modules", {"accelerate": SimpleNamespace(
            init_empty_weights=nullcontext, infer_auto_device_map=infer,
        )}):
            mapping = runtime.device_map(model_class, config)
        self.assertEqual(mapping, {"visual": "musa:0", "language": "musa:1"})
        model_class.assert_called_once_with(config)
        empty.tie_weights.assert_called_once()
        budget = infer.call_args.kwargs["max_memory"]
        self.assertEqual(budget["cpu"], 0)
        self.assertLessEqual(budget[0], 39 * GIB)
        self.assertEqual(budget[1], 4 * GIB)
        self.assertEqual(infer.call_args.kwargs["no_split_module_classes"], ["DecoderLayer"])
        self.assertEqual(infer.call_args.kwargs["dtype"], "bfloat16")

    def test_nested_accelerate_hooks_cannot_enable_cpu_weight_offloading(self):
        runtime, _ = runtime_with_accelerator()
        offload = SimpleNamespace(offload=True)
        for hook in (offload, SimpleNamespace(hooks=(SimpleNamespace(offload=False), offload))):
            child = SimpleNamespace(_hf_hook=hook)
            model = SimpleNamespace(named_parameters=lambda: [], modules=lambda: [child])
            with self.subTest(hook=hook), self.assertRaisesRegex(RuntimeError, "offloading"):
                runtime.assert_on_accelerator(model)
        # Multi-GPU device alignment itself is allowed; only offload hooks fail.
        child = SimpleNamespace(_hf_hook=SimpleNamespace(hooks=(SimpleNamespace(offload=False),)))
        runtime.assert_on_accelerator(SimpleNamespace(named_parameters=lambda: [], modules=lambda: [child]))

    def test_offload_device_map_is_rejected_before_loading_real_weights(self):
        for device in ("cpu", "disk", "meta"):
            runtime, _ = runtime_with_accelerator()
            model_class = MagicMock(return_value=SimpleNamespace(tie_weights=lambda: None))
            infer = MagicMock(return_value={"visual": 0, "language": device})
            with self.subTest(device=device), patch.dict("sys.modules", {"accelerate": SimpleNamespace(
                init_empty_weights=nullcontext, infer_auto_device_map=infer,
            )}), self.assertRaisesRegex(RuntimeError, "cannot fit"):
                runtime.device_map(model_class, object())
            model_class.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
