"""Numerical equivalence for GPU-compatible vision/audio convolution math."""

import importlib.util
import unittest

from vocagent.gpu_ops import patch_musa_convolutions


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class ConvolutionTests(unittest.TestCase):
    def test_conv2d_matches_native_for_group_stride_dilation_and_padding(self):
        import torch
        torch.manual_seed(100)
        for groups in (1, 2):
            for kernel in ((1, 1), (3, 2)):
                for stride in (1, 2):
                    for dilation in (1, 2):
                        for padding in (0, 1):
                            with self.subTest(groups=groups, kernel=kernel, stride=stride,
                                              dilation=dilation, padding=padding):
                                model = torch.nn.Conv2d(4, 6, kernel, stride=stride,
                                                        dilation=dilation, padding=padding, groups=groups)
                                inputs = torch.randn(2, 4, 9, 10)
                                expected = model(inputs)
                                patch_musa_convolutions(model)
                                torch.testing.assert_close(model(inputs), expected)

    def test_conv2d_matches_unbatched_same_padding_and_nonzero_modes(self):
        import torch
        for padding, mode in (("same", "zeros"), (1, "reflect"), (1, "replicate"), (1, "circular")):
            with self.subTest(padding=padding, mode=mode):
                model = torch.nn.Conv2d(3, 5, 3, padding=padding, padding_mode=mode, bias=False)
                inputs = torch.randn(3, 8, 8)
                expected = model(inputs)
                patch_musa_convolutions(model)
                torch.testing.assert_close(model(inputs), expected)

    def test_qwen_vision_patch_conv3d_matches_native(self):
        import torch
        model = torch.nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4), bias=False)
        inputs = torch.randn(7, 3, 2, 4, 4)
        expected = model(inputs)
        patch_musa_convolutions(model)
        torch.testing.assert_close(model(inputs), expected)
        with self.assertRaisesRegex(RuntimeError, "isolated patch"):
            model(torch.randn(7, 3, 4, 4, 4))


if __name__ == "__main__":
    unittest.main()
