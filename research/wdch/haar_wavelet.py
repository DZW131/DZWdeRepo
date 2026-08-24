"""Fixed, orthonormal, channel-wise one-level Haar DWT/IDWT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedHaarDWT2D(nn.Module):
    """Matched one-level 2-D Haar analysis and synthesis.

    Filters are registered buffers, never parameters.  The ordering is
    ``LL, LH, HL, HH`` and each input channel is transformed independently.
    """

    def __init__(self) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5], [0.5, -0.5]],
                [[0.5, 0.5], [-0.5, -0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("analysis_filters", filters, persistent=True)
        self.register_buffer("synthesis_filters", filters.clone(), persistent=True)

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, received shape {tuple(x.shape)}")
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "Haar DWT requires even spatial dimensions; interpolation is forbidden"
            )

    def dwt(self, x: torch.Tensor):
        self._validate_input(x)
        batch, channels, height, width = x.shape
        filters = self.analysis_filters.to(device=x.device, dtype=x.dtype).repeat(
            channels, 1, 1, 1
        )
        coefficients = F.conv2d(x, filters, stride=2, groups=channels)
        coefficients = coefficients.reshape(
            batch, channels, 4, height // 2, width // 2
        )
        return tuple(coefficients.unbind(dim=2))

    def idwt(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        bands = (ll, lh, hl, hh)
        if any(band.ndim != 4 for band in bands):
            raise ValueError("Every Haar band must be BCHW")
        if any(band.shape != ll.shape for band in bands[1:]):
            raise ValueError("All Haar bands must have identical shapes")
        batch, channels, height, width = ll.shape
        coefficients = torch.stack(bands, dim=2).reshape(
            batch, 4 * channels, height, width
        )
        filters = self.synthesis_filters.to(
            device=ll.device, dtype=ll.dtype
        ).repeat(channels, 1, 1, 1)
        return F.conv_transpose2d(
            coefficients, filters, stride=2, groups=channels
        )

    def forward(self, x: torch.Tensor):
        return self.dwt(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.idwt(*self.dwt(x))
