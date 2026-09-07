"""CCRA query primitives for the CQRF Phase-0 responsibility gate."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from network.fqrf_query import CrossFirstFocusDecoder, FocusPatchQueries, sine_position_2d
from network.hqrf_query import CHPF, DualConfidenceAllocator, MaskEmbedding, Projection


class CQRFPixelDecoder(nn.Module):
    """One coherent F4 CHPF transform with separate pixel/memory projections."""

    def __init__(self):
        super().__init__()
        self.f4_projection = Projection(512, 128)
        self.f4_chpf = CHPF(128)
        self.f3_projection = Projection(256, 128)
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256), nn.GELU(), nn.Conv2d(256, 256, 3, padding=1),
        )

    def forward(self, f4, f3):
        raw4 = self.f4_projection(f4)
        context4 = self.f4_chpf(raw4)
        raw3 = self.f3_projection(f3)
        up4 = F.interpolate(context4, raw3.shape[-2:], mode="bilinear", align_corners=False)
        pixel = self.fusion(torch.cat((up4, raw3), dim=1))
        return pixel, {"F4_raw": raw4, "F4_context": context4, "F3_raw": raw3}


class CCRALayer(nn.Module):
    """Class-conditioned competition, normalized over queries for every pixel/class."""

    def __init__(self, dimension=256, hidden=1024, dropout=.1, classes=4, eps=1.e-8):
        super().__init__()
        self.dimension = dimension
        self.classes = classes
        self.eps = eps
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.q_projection = nn.Linear(dimension, dimension, bias=False)
        self.k_projection = nn.Linear(dimension, dimension, bias=False)
        self.v_projection = nn.Linear(dimension, dimension, bias=False)
        self.output_projection = nn.Linear(dimension, dimension, bias=False)
        self.update_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dimension), nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(dimension)

    def class_prior(self, p_class, deep_gate):
        """Detached semantic prior with an image-level all-gates-tiny fallback."""
        with torch.autocast(device_type=p_class.device.type, enabled=False):
            base = p_class.detach().float()
            base = base / base.sum(-1, keepdim=True).clamp_min(self.eps)
            gate = deep_gate.detach().float()
            weighted = base * gate[:, None, :]
            normalized = weighted / weighted.sum(-1, keepdim=True).clamp_min(self.eps)
            tiny = gate.sum(-1, keepdim=True) <= self.eps
            return torch.where(tiny[:, None, :], base, normalized)

    def forward(self, query, base_position, memory, memory_position, p_class, deep_gate):
        if p_class.shape != (query.shape[0], query.shape[1], self.classes):
            raise ValueError("CCRA class-prior shape mismatch")
        if memory.shape != memory_position.shape:
            raise ValueError("CCRA memory and Kp must have identical shapes")
        qbar = self.query_norm(query + base_position)
        xbar = self.memory_norm(memory + memory_position)
        q = self.q_projection(qbar)
        k = self.k_projection(xbar)
        v = self.v_projection(xbar)
        with torch.autocast(device_type=query.device.type, enabled=False):
            affinity = torch.einsum("bnd,bsd->bns", q.float(), k.float()) / (self.dimension ** .5)
            prior = self.class_prior(p_class, deep_gate)
            logits = affinity[..., None] + torch.log(prior[:, :, None, :] + self.eps)
            responsibility_class = torch.softmax(logits, dim=1)
            integrity = (responsibility_class.sum(1) - 1).abs()
            integrity_max = float(integrity.max().detach())
            if not torch.isfinite(responsibility_class).all() or integrity_max > 1.e-5:
                raise FloatingPointError(f"CCRA responsibility normalization failed: {integrity_max}")
            responsibility = (prior[:, :, None, :] * responsibility_class).sum(-1)
            normalized = (responsibility + self.eps) / (responsibility + self.eps).sum(-1, keepdim=True)
            pooled = torch.bmm(normalized, v.float())
        projected_update = self.output_projection(pooled.to(query.dtype))
        updated = self.update_norm(query + projected_update)
        new_query = self.final_norm(updated + self.ffn(updated))
        return new_query, {
            "class_prior": prior.detach(),
            "responsibility_class": responsibility_class.detach(),
            "responsibility": responsibility.detach(),
            "responsibility_normalized": normalized.detach(),
            "integrity_max": integrity_max,
            "integrity_mean": float(integrity.mean().detach()),
            "projected_update": projected_update.detach(),
            "query_delta": (new_query - query).detach(),
        }


__all__ = [
    "CCRALayer", "CHPF", "CQRFPixelDecoder", "CrossFirstFocusDecoder",
    "DualConfidenceAllocator", "FocusPatchQueries", "MaskEmbedding", "Projection",
    "sine_position_2d",
]
