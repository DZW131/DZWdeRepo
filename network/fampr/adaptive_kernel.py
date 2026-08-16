"""Morphology-preserving low/high adaptive depthwise kernel spectrum."""

from typing import Tuple

import torch
import torch.nn as nn


class AdaptiveKernelSpectrum(nn.Module):
    """Decompose a learnable depthwise kernel and predict neutral channel gates."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if reduction != 16:
            raise ValueError("FA-MPR fixes AdaKern reduction to 16")
        self.channels = int(channels)
        self.reduction = int(reduction)

        gaussian = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
        ) / 16.0
        self.base_kernel = nn.Parameter(
            gaussian.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        )

        hidden = max(channels // reduction, 16)
        self.gate_network = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )
        final_linear = self.gate_network[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def decompose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_low = self.base_kernel.mean(dim=(-2, -1), keepdim=True).expand_as(
            self.base_kernel
        )
        kernel_high = self.base_kernel - kernel_low
        return kernel_low, kernel_high

    def forward(
        self, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"feature must be [B,{self.channels},H,W], "
                f"got {tuple(feature.shape)}"
            )
        pooled = feature.mean(dim=(-2, -1))
        gate_logits = self.gate_network(pooled)
        gate_low, gate_high = (2.0 * torch.sigmoid(gate_logits)).chunk(2, dim=1)
        kernel_low, kernel_high = self.decompose()
        return (
            kernel_low,
            kernel_high,
            gate_low[:, :, None, None],
            gate_high[:, :, None, None],
        )
