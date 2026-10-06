"""Exact convolution decompositions for MUSA builds without native kernels."""

import types


def patch_musa_convolutions(model):
    """Keep weights/activations resident and use GPU views and matrix products.

    Tensor.unfold is a strided view; F.unfold/im2col has its own unsupported MUSA
    kernel. Conv3d support here is deliberately limited to isolated vision patches.
    """
    import torch

    def conv2d(self, tensor):
        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.dilation
        stride_h, stride_w = self.stride
        if isinstance(self.padding, str):
            if self.padding == "valid":
                padding = (0, 0, 0, 0)
            elif self.padding == "same":
                total_h, total_w = dilation_h * (kernel_h - 1), dilation_w * (kernel_w - 1)
                padding = (total_w // 2, total_w - total_w // 2, total_h // 2, total_h - total_h // 2)
            else:
                raise ValueError(f"Unsupported convolution padding: {self.padding}")
        else:
            padding_h, padding_w = self.padding
            padding = (padding_w, padding_w, padding_h, padding_h)
        unbatched = tensor.ndim == 3
        if unbatched:
            tensor = tensor.unsqueeze(0)
        if any(padding):
            mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
            tensor = torch.nn.functional.pad(tensor, padding, mode=mode)
        windows = tensor.unfold(2, dilation_h * (kernel_h - 1) + 1, stride_h)
        windows = windows.unfold(3, dilation_w * (kernel_w - 1) + 1, stride_w)
        windows = windows[..., ::dilation_h, ::dilation_w].permute(0, 2, 3, 1, 4, 5)
        batch, height, width = windows.shape[:3]
        if self.groups == 1:
            output = torch.nn.functional.linear(
                windows.reshape(batch, height, width, -1), self.weight.flatten(1), self.bias,
            ).permute(0, 3, 1, 2)
        else:
            windows = windows.reshape(batch, height, width, self.groups, -1)
            weights = self.weight.reshape(self.groups, self.out_channels // self.groups, -1)
            output = torch.einsum("bhwgi,goi->bgohw", windows, weights)
            output = output.reshape(batch, self.out_channels, height, width)
            if self.bias is not None:
                output = output + self.bias[None, :, None, None]
        output = output.contiguous()
        return output.squeeze(0) if unbatched else output

    def vision_patch(self, tensor):
        if tuple(tensor.shape[-3:]) != self.kernel_size:
            raise RuntimeError("The MUSA vision Conv3d adapter requires one isolated patch per sample.")
        output = torch.nn.functional.linear(tensor.flatten(1), self.weight.flatten(1), self.bias)
        return output[:, :, None, None, None]

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.forward = types.MethodType(conv2d, module)
        elif isinstance(module, torch.nn.Conv3d) and (
            module.kernel_size == module.stride and module.padding == (0, 0, 0)
            and module.dilation == (1, 1, 1) and module.groups == 1
        ):
            module.forward = types.MethodType(vision_patch, module)
