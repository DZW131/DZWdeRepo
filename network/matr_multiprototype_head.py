"""OT-MTR two-mode CAM head for MATR-v1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPrototypeCAMHead(nn.Module):
    """Four class anchors with exactly two centered morphology modes each."""

    def __init__(self, in_channels: int = 512, classes: int = 4, modes: int = 2):
        super().__init__()
        if modes != 2:
            raise ValueError("Frozen MATR-v1 requires exactly two modes per class")
        self.in_channels = in_channels
        self.classes = classes
        self.modes = modes
        self.base = nn.Conv2d(in_channels, classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.base.weight)
        self.d_raw = nn.Parameter(torch.empty(classes, modes, in_channels))
        nn.init.normal_(self.d_raw, mean=0.0, std=1.0e-3)
        with torch.no_grad():
            self.d_raw.sub_(self.d_raw.mean(dim=1, keepdim=True))

    def mode_weights(self) -> torch.Tensor:
        centered = self.d_raw - self.d_raw.mean(dim=1, keepdim=True)
        return self.base.weight[:, None, :, 0, 0] + centered

    def mode_cosine(self) -> torch.Tensor:
        normalized = F.normalize(self.mode_weights().float(), dim=-1)
        return (normalized[:, 0] * normalized[:, 1]).sum(dim=-1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.mode_weights().reshape(
            self.classes * self.modes, self.in_channels, 1, 1
        )
        bias = self.base.bias[:, None].expand(self.classes, self.modes).reshape(-1)
        logits = F.conv2d(features, weights, bias=bias)
        logits = logits.view(
            features.shape[0], self.classes, self.modes,
            features.shape[2], features.shape[3],
        )
        # For two modes, log-mean-exp is mean + log(cosh(half-difference)).
        # This equivalent form gives an exact zero correction when both modes
        # coincide, satisfying the frozen single-mode degeneration contract.
        midpoint = 0.5 * (logits[:, :, 0] + logits[:, :, 1])
        half_difference = 0.5 * (logits[:, :, 0] - logits[:, :, 1])
        aggregated = midpoint + torch.log(torch.cosh(half_difference))
        return aggregated, logits
