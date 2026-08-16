"""Pure-PyTorch spatially adaptive depthwise sampling."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatiallyAdaptiveDepthwiseSampler(nn.Module):
    """Apply two depthwise kernels to one vectorized adaptive sample tensor.

    All nine 3x3 kernel locations are packed along the output-width dimension,
    so each forward performs exactly one ``grid_sample`` call.
    """

    def __init__(
        self,
        padding_mode: str = "border",
        align_corners: bool = True,
        internal_sampling_fp32: bool = True,
    ):
        super().__init__()
        if padding_mode != "border":
            raise ValueError("FA-MPR fixes grid_sample padding_mode to 'border'")
        if align_corners is not True:
            raise ValueError("FA-MPR fixes align_corners to True")
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.internal_sampling_fp32 = bool(internal_sampling_fp32)

    @staticmethod
    def _normalize(coordinate: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(coordinate)
        return 2.0 * coordinate / float(size - 1) - 1.0

    def forward(
        self,
        x: torch.Tensor,
        dilation: torch.Tensor,
        kernel_low: torch.Tensor,
        kernel_high: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"x must be BCHW, got shape {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if dilation.shape != (batch, 1, height, width):
            raise ValueError(
                "dilation must have shape "
                f"{(batch, 1, height, width)}, got {tuple(dilation.shape)}"
            )
        expected_kernel_shape = (channels, 1, 3, 3)
        if kernel_low.shape != expected_kernel_shape:
            raise ValueError(
                f"kernel_low must be {expected_kernel_shape}, "
                f"got {tuple(kernel_low.shape)}"
            )
        if kernel_high.shape != expected_kernel_shape:
            raise ValueError(
                f"kernel_high must be {expected_kernel_shape}, "
                f"got {tuple(kernel_high.shape)}"
            )

        output_dtype = x.dtype
        sampling_dtype = torch.float32 if self.internal_sampling_fp32 else x.dtype
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            sample_input = x.to(dtype=sampling_dtype)
            sample_dilation = dilation.to(dtype=sampling_dtype).squeeze(1)
            low = kernel_low.to(dtype=sampling_dtype)
            high = kernel_high.to(dtype=sampling_dtype)

            y_coordinates = torch.arange(
                height, device=x.device, dtype=sampling_dtype
            )
            x_coordinates = torch.arange(
                width, device=x.device, dtype=sampling_dtype
            )
            base_y, base_x = torch.meshgrid(
                y_coordinates, x_coordinates, indexing="ij"
            )
            base_y = base_y.view(1, height, width, 1)
            base_x = base_x.view(1, height, width, 1)
            offsets = torch.tensor(
                [
                    (-1.0, -1.0),
                    (-1.0, 0.0),
                    (-1.0, 1.0),
                    (0.0, -1.0),
                    (0.0, 0.0),
                    (0.0, 1.0),
                    (1.0, -1.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                ],
                device=x.device,
                dtype=sampling_dtype,
            )
            sample_y = base_y + sample_dilation.unsqueeze(-1) * offsets[:, 0]
            sample_x = base_x + sample_dilation.unsqueeze(-1) * offsets[:, 1]
            grid = torch.stack(
                (
                    self._normalize(sample_x, width),
                    self._normalize(sample_y, height),
                ),
                dim=-1,
            ).reshape(batch, height, width * 9, 2)

            sampled = F.grid_sample(
                sample_input,
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            ).reshape(batch, channels, height, width, 9)
            low_flat = low.reshape(channels, 9)
            high_flat = high.reshape(channels, 9)
            y_low = torch.einsum("bchwk,ck->bchw", sampled, low_flat)
            y_high = torch.einsum("bchwk,ck->bchw", sampled, high_flat)

        return y_low.to(dtype=output_dtype), y_high.to(dtype=output_dtype)
