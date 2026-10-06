"""H3 capacity policy, frame geometry and GPU-compatible convolution checks."""

import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vocagent.minimax import (
    H3Settings, _dispatch_resident, _patch_pointwise_conv3d, _patch_audio_convolutions,
    _gpu_decode_boundary,
    _chunk_h3_feed_forwards,
)


class H3SettingsTests(unittest.TestCase):
    def test_frame_count_obeys_vae_grid_and_duration_limits(self):
        for duration in (5, 5.1, 8, 10, 14.9, 15):
            frames = H3Settings(seconds=duration).num_frames
            self.assertEqual((frames - 5) % 17, 0)
            self.assertGreaterEqual(frames / 24, 5)
            self.assertLessEqual(frames / 24, 15)

    def test_invalid_dimensions_and_duration_fail_before_loading(self):
        for settings in ({"width": 1350}, {"height": 0}, {"seconds": float("nan")},
                         {"seconds": 16}, {"steps": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                H3Settings(**settings)

    def test_insufficient_gpu_capacity_never_dispatches_to_cpu(self):
        accelerate = SimpleNamespace(infer_auto_device_map=Mock(return_value={"": "cpu"}),
                                     dispatch_model=Mock())
        gpu = SimpleNamespace(device_count=lambda: 1, mem_get_info=lambda index: (8*1024**3, 8*1024**3),
                              memory_allocated=lambda index: 0)
        runtime = SimpleNamespace(torch=SimpleNamespace(musa=gpu), backend="musa", device="musa:0")
        with patch.dict("sys.modules", {"accelerate": accelerate}):
            with self.assertRaisesRegex(RuntimeError, "without offload"):
                _dispatch_resident(SimpleNamespace(_no_split_modules=[]), runtime)
        accelerate.dispatch_model.assert_not_called()

    def test_gpu_budget_accounts_for_models_already_resident(self):
        gib = 1024**3
        model = SimpleNamespace(_no_split_modules=[], parameters=lambda: ())
        accelerate = SimpleNamespace(infer_auto_device_map=Mock(return_value={"": 0}),
                                     dispatch_model=Mock(return_value=model))
        gpu = SimpleNamespace(device_count=lambda: 3,
                              mem_get_info=lambda index: ((36, 43, 5)[index] * gib, 48 * gib),
                              memory_allocated=lambda index: (10, 3, 40)[index] * gib)
        runtime = SimpleNamespace(torch=SimpleNamespace(musa=gpu), backend="musa", device="musa:0",
                                  assert_on_accelerator=Mock())
        with patch.dict("sys.modules", {"accelerate": accelerate}):
            self.assertIs(_dispatch_resident(model, runtime), model)
        self.assertEqual(accelerate.infer_auto_device_map.call_args.kwargs["max_memory"],
                         {0: 26 * gib, 1: 33 * gib, 2: 0, "cpu": 0})
        runtime.assert_on_accelerator.assert_called_once_with(model)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is an optional model dependency")
class H3FeedForwardTests(unittest.TestCase):
    def test_chunked_swiglu_preserves_tokens_weights_and_distinct_blocks(self):
        import torch

        class SwiGLU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(7, 22)
                self.out = torch.nn.Linear(11, 7)

            def forward(self, inputs):
                value, gate = self.proj(inputs).chunk(2, dim=-1)
                return self.out(value * torch.nn.functional.silu(gate))

        class MiniMaxH3TransformerBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ff = SwiGLU()

        torch.manual_seed(45)
        model = torch.nn.ModuleList([MiniMaxH3TransformerBlock(), MiniMaxH3TransformerBlock()])
        parameters = [(id(parameter), parameter.data_ptr(), parameter.device) for parameter in model.parameters()]
        inputs = torch.randn(2, 7, 19).transpose(1, 2)
        expected = [block.ff(inputs) for block in model]
        _chunk_h3_feed_forwards(model, chunk_size=4)
        for block, reference in zip(model, expected):
            torch.testing.assert_close(block.ff(inputs), reference)
        self.assertEqual(parameters, [(id(p), p.data_ptr(), p.device) for p in model.parameters()])

    def test_small_sequences_keep_fast_path_and_chunking_is_idempotent(self):
        import torch

        class MiniMaxH3TokenRefinerBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ff = torch.nn.Identity()

        block = MiniMaxH3TokenRefinerBlock()
        block.ff.forward = Mock(side_effect=lambda inputs: inputs)
        original = block.ff.forward
        _chunk_h3_feed_forwards(block, chunk_size=4)
        _chunk_h3_feed_forwards(block, chunk_size=4)
        inputs = torch.randn(2, 4, 7)
        self.assertIs(block.ff(inputs), inputs)
        original.assert_called_once_with(inputs)
        original.reset_mock()
        inputs = torch.randn(2, 9, 7)
        torch.testing.assert_close(block.ff(inputs), inputs)
        self.assertEqual([call.args[0].shape[1] for call in original.call_args_list], [4, 4, 1])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is an optional model dependency")
class H3DecodeTests(unittest.TestCase):
    def test_empty_and_nonfinite_decodes_are_rejected_before_export(self):
        import torch
        for sample in (torch.empty(0), torch.tensor([float("nan")]), torch.tensor([float("inf")])):
            for as_tuple in (True, False):
                with self.subTest(sample=sample, as_tuple=as_tuple):
                    output = (sample,) if as_tuple else SimpleNamespace(sample=sample)
                    vae = SimpleNamespace(dtype=torch.float32, decode=lambda latents: output)
                    _gpu_decode_boundary(vae, "cpu", "cpu")
                    with self.assertRaisesRegex(RuntimeError, "empty or nonfinite"):
                        vae.decode(torch.zeros(1))

    def test_finite_decodes_preserve_output_and_enforce_the_configured_device(self):
        import torch
        sample = torch.tensor([0.1, -0.2, 0.3])
        for as_tuple in (True, False):
            output = (sample,) if as_tuple else SimpleNamespace(sample=sample)
            vae = SimpleNamespace(dtype=torch.float32, decode=lambda latents: output)
            _gpu_decode_boundary(vae, "cpu", "cpu")
            decoded = vae.decode(torch.zeros(1))
            torch.testing.assert_close(decoded[0] if as_tuple else decoded.sample, sample)
        misplaced = torch.empty(1, device="meta")
        vae = SimpleNamespace(dtype=torch.float32, decode=lambda latents: (misplaced,))
        _gpu_decode_boundary(vae, "cpu", "cpu")
        with self.assertRaisesRegex(RuntimeError, "configured GPU"):
            vae.decode(torch.zeros(1))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is an optional model dependency")
class H3ConvolutionTests(unittest.TestCase):
    def test_depthwise_audio_resamplers_match_native_filters(self):
        import torch
        functional = torch.nn.functional

        class MiniMaxH3AudioLowPassFilter1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pad_left, self.pad_right, self.stride = 5, 6, 2
                self.register_buffer("filter", torch.randn(1, 1, 12))

            def forward(self, inputs):
                inputs = functional.pad(inputs, (self.pad_left, self.pad_right), mode="replicate")
                return functional.conv1d(inputs, self.filter.expand(inputs.shape[1], -1, -1),
                                         stride=self.stride, groups=inputs.shape[1])

        class MiniMaxH3AudioUpSample1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ratio, self.stride, self.pad = 2, 2, 5
                self.pad_left, self.pad_right = 15, 15
                self.register_buffer("filter", torch.randn(1, 1, 12))

            def forward(self, inputs):
                inputs = functional.pad(inputs, (self.pad, self.pad), mode="replicate")
                output = self.ratio * functional.conv_transpose1d(
                    inputs, self.filter.expand(inputs.shape[1], -1, -1), stride=self.stride,
                    groups=inputs.shape[1],
                )
                return output[..., self.pad_left:-self.pad_right]

        torch.manual_seed(44)
        for model_class in (MiniMaxH3AudioLowPassFilter1d, MiniMaxH3AudioUpSample1d):
            for channels in (1, 8):
                with self.subTest(kind=model_class, channels=channels):
                    model = model_class()
                    inputs = torch.randn(2, channels, 29)
                    expected = model(inputs)
                    _patch_audio_convolutions(model)
                    torch.testing.assert_close(model(inputs), expected)

    def test_audio_convolutions_match_native_for_groups_stride_padding_and_dilation(self):
        import torch
        torch.manual_seed(43)
        for kind in (torch.nn.Conv1d, torch.nn.ConvTranspose1d):
            for groups in (1, 2):
                for bias in (True, False):
                    for kernel, stride, padding, dilation, output_padding in (
                        (1, 1, 0, 1, 0), (3, 1, 1, 3, 0), (7, 1, 3, 1, 0),
                        (16, 8, 4, 1, 0), (5, 3, 3, 2, 0), (4, 2, 1, 1, 1),
                    ):
                        with self.subTest(kind=kind, groups=groups, bias=bias, kernel=kernel,
                                          stride=stride, dilation=dilation, output_padding=output_padding):
                            extra = {"output_padding": output_padding} if kind is torch.nn.ConvTranspose1d else {}
                            convolution = kind(4, 6, kernel, stride=stride, padding=padding,
                                               dilation=dilation, groups=groups, bias=bias, **extra)
                            inputs = torch.randn(2, 4, 13)
                            expected = convolution(inputs)
                            _patch_audio_convolutions(convolution)
                            torch.testing.assert_close(convolution(inputs), expected)

    def test_pointwise_replacement_matches_conv3d_with_bias_and_noncontiguous_input(self):
        import torch
        torch.manual_seed(42)
        for bias in (True, False):
            conv = torch.nn.Conv3d(3, 7, 1, bias=bias)
            inputs = torch.randn(2, 3, 4, 5, 6).transpose(3, 4)
            expected = conv(inputs)
            _patch_pointwise_conv3d(conv)
            torch.testing.assert_close(conv(inputs), expected)

    def test_non_pointwise_convolutions_are_preserved(self):
        import torch
        conv = torch.nn.Conv3d(3, 7, 3)
        original = conv.forward
        _patch_pointwise_conv3d(conv)
        self.assertEqual(conv.forward, original)


if __name__ == "__main__":
    unittest.main()
