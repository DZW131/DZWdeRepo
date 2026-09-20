"""Reliability-Aware Class Responsibility Control (RACC-v1)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ReliabilityArbitrator(nn.Module):
    """Shared query-position controller for the HQMR deep/local residual."""

    def __init__(self, hidden_dim: int = 8, alpha_max: float = 4.0):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.controller = nn.Sequential(nn.Conv2d(4, hidden_dim, 1), nn.GELU(), nn.Conv2d(hidden_dim, 1, 1))
        nn.init.xavier_uniform_(self.controller[0].weight)
        nn.init.zeros_(self.controller[0].bias)
        nn.init.zeros_(self.controller[2].weight)
        nn.init.constant_(self.controller[2].bias, math.log((1.0 / self.alpha_max) / (1.0 - 1.0 / self.alpha_max)))

    def forward(self, coarse: torch.Tensor, direct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        up = F.interpolate(coarse, size=direct.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.stack((up.sigmoid(), direct.sigmoid(), (up.sigmoid() - direct.sigmoid()).abs(),
                                up.sigmoid() * direct.sigmoid()), dim=2)
        batch, queries, _, height, width = features.shape
        alpha = self.alpha_max * self.controller(features.reshape(batch * queries, 4, height, width)).sigmoid()
        alpha = alpha.reshape(batch, queries, height, width)
        return up + alpha * direct, alpha


class LocalPresenceRescue(nn.Module):
    """Shared class-wise local-presence predictor from C4/C3 statistics."""

    def __init__(self, hidden_dim: int = 8, topk_ratio: float = 0.05):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.head = nn.Sequential(nn.Linear(5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.xavier_uniform_(self.head[0].weight)
        nn.init.zeros_(self.head[0].bias)
        nn.init.zeros_(self.head[2].weight)
        nn.init.constant_(self.head[2].bias, -4.0)

    def _summary(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = value.float().flatten(2)
        count = max(1, int(math.ceil(flat.shape[-1] * self.topk_ratio)))
        return flat.topk(count, dim=-1).values.mean(-1), flat.max(-1).values

    def forward(self, c4: torch.Tensor, c3: torch.Tensor) -> dict:
        t4, m4 = self._summary(c4)
        t3, m3 = self._summary(c3)
        features = torch.stack((t4, t3, m4, m3, (t4 - t3).abs()), dim=-1)
        logits = self.head(features).squeeze(-1)
        return {"features": features, "logits": logits, "probability": logits.sigmoid()}


class RACCController(nn.Module):
    def __init__(self, hidden_dim: int = 8, alpha_max: float = 4.0, topk_ratio: float = 0.05):
        super().__init__()
        self.arbitration = ReliabilityArbitrator(hidden_dim, alpha_max)
        self.presence = LocalPresenceRescue(hidden_dim, topk_ratio)

    @staticmethod
    def rescued_gate(deep_gate: torch.Tensor, local_probability: torch.Tensor,
                     deep_threshold: torch.Tensor, local_threshold: float = 0.5) -> torch.Tensor:
        return (deep_gate > deep_threshold) | (local_probability > local_threshold)


__all__ = ["ReliabilityArbitrator", "LocalPresenceRescue", "RACCController"]
