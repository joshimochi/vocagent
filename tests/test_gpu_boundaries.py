"""H3 top-level tensor joins remain on the execution device when branches shard."""

from types import SimpleNamespace
import unittest

from vocagent.gpu_boundaries import align_h3_residual_outputs


class Tensor:
    """Small exact device-aware tensor stand-in; GPU hardware is not required."""

    def __init__(self, values, device):
        self.values = tuple(values)
        self.device = device

    def to(self, device):
        return Tensor(self.values, device)

    def index_copy(self, indices, source):
        if self.device != indices.device or self.device != source.device:
            raise RuntimeError("mixed-device scatter")
        values = list(self.values)
        for index, value in zip(indices.values, source.values):
            values[index] = value
        return Tensor(values, self.device)

    def index_select(self, indices):
        if self.device != indices.device:
            raise RuntimeError("mixed-device gather")
        return Tensor([self.values[index] for index in indices.values], self.device)


class ShardedModule:
    def __init__(self, device, scale=1):
        self.device, self.scale = device, scale
        self.hooks = []
        self.weight = object()

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return hook

    def __call__(self, tensor):
        # Simulate Accelerate's input alignment and resident module arithmetic.
        output = Tensor([value * self.scale for value in tensor.values], self.device)
        for hook in self.hooks:
            output = hook(self, (tensor,), output)
        return output


class H3BoundaryTests(unittest.TestCase):
    def test_sharded_branches_scatter_and_gather_without_moving_weights(self):
        model = SimpleNamespace(
            proj_in=ShardedModule("musa:1", 2),
            audio_proj_in=ShardedModule("musa:2", 3),
            token_refiner=ShardedModule("musa:3", 4),
            proj_out=ShardedModule("musa:2", 5),
            audio_proj_out=ShardedModule("musa:3", 6),
        )
        modules = tuple(vars(model).values())
        weights = tuple(module.weight for module in modules)
        indices = [Tensor([index], "musa:0") for index in range(3)]

        def forward():
            video = model.proj_in(Tensor([2], "musa:0"))
            audio = model.audio_proj_in(Tensor([3], "musa:0"))
            text = model.token_refiner(Tensor([4], "musa:0"))
            packed = Tensor([0, 0, 0], text.device)
            for index, source in zip(indices, (video, audio, text)):
                packed = packed.index_copy(index, source)
            # Block outputs may end on another device. Both output heads must
            # return to the root's device before its unhooked index_select.
            hidden = packed.to("musa:3")
            return (model.proj_out(hidden).index_select(indices[0]),
                    model.audio_proj_out(hidden).index_select(indices[1]))

        with self.assertRaisesRegex(RuntimeError, "mixed-device scatter"):
            forward()
        handles = align_h3_residual_outputs(model, "musa:0")
        video, audio = forward()
        self.assertEqual((video.values, audio.values), ((20,), (54,)))
        self.assertEqual((video.device, audio.device), ("musa:0", "musa:0"))
        self.assertEqual(weights, tuple(module.weight for module in modules))
        self.assertEqual([module.device for module in modules],
                         ["musa:1", "musa:2", "musa:3", "musa:2", "musa:3"])
        self.assertEqual(len(handles), 5)

    def test_wrong_model_fails_before_any_hooks_are_installed(self):
        module = ShardedModule("musa:0")
        with self.assertRaises(AttributeError):
            align_h3_residual_outputs(SimpleNamespace(proj_in=module), "musa:0")
        self.assertEqual(module.hooks, [])


if __name__ == "__main__":
    unittest.main()
