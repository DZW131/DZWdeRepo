"""Hierarchical Query-conditioned Mask Refinement (HQMR)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class KeyValueProjection(nn.Module):
    """Independent 1x1 key/value projections followed by GroupNorm."""

    def __init__(self, in_channels: int, dim: int = 256):
        super().__init__()
        groups = 32 if dim % 32 == 0 else 1
        self.key = nn.Sequential(nn.Conv2d(in_channels, dim, 1, bias=False), nn.GroupNorm(groups, dim))
        self.value = nn.Sequential(nn.Conv2d(in_channels, dim, 1, bias=False), nn.GroupNorm(groups, dim))
        for branch in (self.key, self.value):
            nn.init.xavier_uniform_(branch[0].weight)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key(feature), self.value(feature)


class QueryRegionUpdate(nn.Module):
    """FP32 soft region pooling followed by the frozen residual update/FFN."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.region_projection = nn.Linear(dim, dim, bias=False)
        self.update_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(0.0), nn.Linear(2 * dim, dim))
        self.final_norm = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.region_projection.weight)
        nn.init.xavier_uniform_(self.ffn[0].weight); nn.init.zeros_(self.ffn[0].bias)
        nn.init.xavier_uniform_(self.ffn[3].weight); nn.init.zeros_(self.ffn[3].bias)

    @staticmethod
    def pool(mask_logits: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=mask_logits.device.type, enabled=False):
            probability = normalized_region_weights(mask_logits)
            pooled = torch.einsum("bqn,bdn->bqd", probability, value.float().flatten(2))
        return pooled.to(mask_logits.dtype)

    def forward(self, query: torch.Tensor, mask_logits: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query = self.update_norm(query + self.region_projection(self.pool(mask_logits, value)))
        return self.final_norm(query + self.ffn(query))


def direct_affinity(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bqd,bdhw->bqhw", query, key) / math.sqrt(query.shape[-1])


def normalized_region_weights(mask_logits: torch.Tensor) -> torch.Tensor:
    probability = mask_logits.float().sigmoid().flatten(2)
    return probability / probability.sum(-1, keepdim=True).clamp_min(1.0e-8)


def residual_logits(coarse: torch.Tensor, direct: torch.Tensor) -> torch.Tensor:
    return F.interpolate(coarse, size=direct.shape[-2:], mode="bilinear", align_corners=False) + direct


class HQMR(nn.Module):
    """Shared F5->F4 hierarchy with an optional Stage3-only F3 refinement."""

    MODES = {"full", "fine_only", "coarse_fine", "no_query_update", "coarse_only", "mid_final"}

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.query_norm = nn.LayerNorm(dim)
        self.scale5 = KeyValueProjection(256, dim)
        self.scale4 = KeyValueProjection(128, dim)
        self.scale3 = KeyValueProjection(256, dim)
        self.update5 = QueryRegionUpdate(dim)
        self.update4 = QueryRegionUpdate(dim)

    def forward(self, query: torch.Tensor, h5: torch.Tensor, h4: torch.Tensor,
                h3: torch.Tensor | None = None, mode: str = "full") -> dict:
        if mode not in self.MODES:
            raise ValueError(f"Unknown HQMR mode: {mode}")
        q0 = self.query_norm(query)
        k5, v5 = self.scale5(h5)
        k4, v4 = self.scale4(h4)
        k3 = v3 = None
        if h3 is not None:
            k3, v3 = self.scale3(h3)

        if mode == "fine_only":
            final_logits = direct_affinity(q0, k3 if k3 is not None else k4)
            return {"basis_logits": final_logits, "basis": final_logits.float().sigmoid(),
                    "logits5": None, "logits4": None, "logits3": final_logits if k3 is not None else None,
                    "direct4": None, "direct3": final_logits if k3 is not None else None,
                    "query0": q0, "query5": q0, "query4": q0, "value3": v3, "mode": mode}

        logits5 = direct_affinity(q0, k5)
        q5 = q0 if mode == "no_query_update" else self.update5(q0, logits5, v5)
        if mode == "coarse_only":
            target = h3 if h3 is not None else h4
            final_logits = F.interpolate(logits5, size=target.shape[-2:], mode="bilinear", align_corners=False)
            return {"basis_logits": final_logits, "basis": final_logits.float().sigmoid(),
                    "logits5": logits5, "logits4": None, "logits3": None,
                    "direct4": None, "direct3": None,
                    "query0": q0, "query5": q5, "query4": q5, "value3": v3, "mode": mode}

        if mode == "coarse_fine" and k3 is not None:
            direct3 = direct_affinity(q5, k3)
            logits3 = residual_logits(logits5, direct3)
            return {"basis_logits": logits3, "basis": logits3.float().sigmoid(),
                    "logits5": logits5, "logits4": None, "logits3": logits3,
                    "direct4": None, "direct3": direct3,
                    "query0": q0, "query5": q5, "query4": q5, "value3": v3, "mode": mode}

        direct4 = direct_affinity(q5, k4)
        logits4 = residual_logits(logits5, direct4)
        q4 = q5 if mode == "no_query_update" else self.update4(q5, logits4, v4)
        if h3 is None:
            final_logits, logits3 = logits4, None
        elif mode == "mid_final":
            final_logits = F.interpolate(logits4, size=h3.shape[-2:], mode="bilinear", align_corners=False)
            logits3 = None
        else:
            direct3 = direct_affinity(q4, k3)
            logits3 = residual_logits(logits4, direct3)
            final_logits = logits3
        return {"basis_logits": final_logits, "basis": final_logits.float().sigmoid(),
                "logits5": logits5, "logits4": logits4, "logits3": logits3,
                "direct4": direct4, "direct3": direct3 if h3 is not None and mode not in {"mid_final"} else None,
                "query0": q0, "query5": q5, "query4": q4, "value3": v3, "mode": mode}


def class_mixture(basis: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Apply the original detached GCQM global class weights to HQMR bases."""
    detached = weights.detach().float()
    if detached.ndim != 3 or basis.shape[:2] != detached.shape[:2]:
        raise ValueError("HQMR basis/GCQM weight shape mismatch")
    mixture = torch.einsum("bqc,bqhw->bchw", detached, basis.float()).clamp(0.0, 1.0)
    if not torch.isfinite(mixture).all():
        raise FloatingPointError("Non-finite HQMR mixture")
    return mixture


__all__ = ["HQMR", "KeyValueProjection", "QueryRegionUpdate", "class_mixture", "direct_affinity",
           "normalized_region_weights", "residual_logits"]
