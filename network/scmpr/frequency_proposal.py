"""Fixed, parameter-free frequency proposals used by SC-MPR."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_lowpass(feature: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Unit-sum replicate-padded average filtering with unchanged shape."""
    if feature.ndim != 4:
        raise ValueError(f"Expected BCHW feature tensor, got {feature.shape}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )
    padding = kernel_size // 2
    padded = F.pad(
        feature, (padding, padding, padding, padding), mode="replicate"
    )
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


class FixedFrequencyProposal(nn.Module):
    """Produce fixed fine- and morphology-frequency residual proposals."""

    def __init__(self, eps: float = 1e-6, quality_clamp: float = 5.0):
        super().__init__()
        self.eps = float(eps)
        self.quality_clamp = float(quality_clamp)
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.quality_clamp <= 0.0:
            raise ValueError("quality_clamp must be positive")

    def _normalize_quality(self, residual: torch.Tensor) -> torch.Tensor:
        quality = residual.abs().mean(dim=1, keepdim=True)
        spatial_mean = quality.mean(dim=(-2, -1), keepdim=True)
        return (quality / (spatial_mean + self.eps)).clamp(
            min=0.0, max=self.quality_clamp
        )

    def forward(
        self, feature: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        lowpass3 = fixed_lowpass(feature, 3)
        lowpass15 = fixed_lowpass(feature, 15)
        residual_fine = feature - lowpass3
        residual_morphology = lowpass3 - lowpass15
        residuals = {
            "fine": residual_fine,
            "morphology": residual_morphology,
        }
        qualities = {
            "fine": self._normalize_quality(residual_fine),
            "morphology": self._normalize_quality(residual_morphology),
        }
        return residuals, qualities
