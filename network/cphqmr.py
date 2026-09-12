"""Coverage-Preserving Hierarchical Query-to-Mask Reconstruction."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from network.hqmr import KeyValueProjection, QueryRegionUpdate, direct_affinity


def coverage_preserving_fusion(coverage: torch.Tensor, discriminative: torch.Tensor) -> dict:
    """Bounded residual rescue: D <= M <= C where C>D, otherwise M=D."""
    rescue = F.relu(coverage - discriminative)
    rescue_gate = 1.0 - discriminative.sigmoid()
    fused = discriminative + rescue_gate * rescue
    return {"fused": fused, "rescue": rescue, "rescue_gate": rescue_gate}


class DetailGuidedSpatialRestoration(nn.Module):
    """Query/class-agnostic 3x3 dynamic spatial restoration from raw F3."""

    def __init__(self):
        super().__init__()
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False), nn.GroupNorm(32, 64), nn.GELU())
        self.kernel_head = nn.Conv2d(64, 9, 1)
        self.gate_head = nn.Conv2d(64, 1, 1)
        for module in (self.detail_encoder[0], self.kernel_head, self.gate_head):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)

    @staticmethod
    def dynamic_restore(logits: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        padded = F.pad(logits, (1, 1, 1, 1), mode="replicate")
        height, width = logits.shape[-2:]; restored = torch.zeros_like(logits)
        edge = 0
        for dy in range(3):
            for dx in range(3):
                restored = restored + kernel[:, edge:edge + 1] * padded[..., dy:dy + height, dx:dx + width]
                edge += 1
        return restored

    def forward(self, semantic_logits: torch.Tensor, h3: torch.Tensor, enabled: bool = True) -> dict:
        upsampled = F.interpolate(semantic_logits, size=h3.shape[-2:], mode="bilinear", align_corners=False)
        detail = self.detail_encoder(h3); kernel = self.kernel_head(detail).softmax(dim=1); gate = self.gate_head(detail).sigmoid()
        restored = self.dynamic_restore(upsampled, kernel)
        final = (1.0 - gate) * upsampled + gate * restored if enabled else upsampled
        return {"logits": final, "upsampled": upsampled, "restored": restored, "kernel": kernel,
                "gate": gate, "detail": detail, "enabled": enabled}


class CPHQMR(nn.Module):
    """Dual-state H5/H4 semantic reconstruction and Stage3-only DGSR."""

    MODES = {"full", "discriminative_only", "coverage_only", "simple_average", "bilinear_only", "old_f3_semantic"}

    def __init__(self, dim: int = 256):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.scale5 = KeyValueProjection(256, dim)
        self.scale4 = KeyValueProjection(128, dim)
        self.update5 = QueryRegionUpdate(dim)
        self.dgsr = DetailGuidedSpatialRestoration()

    def forward(self, query: torch.Tensor, h5: torch.Tensor, h4: torch.Tensor,
                h3: torch.Tensor | None = None, mode: str = "full") -> dict:
        if mode not in self.MODES: raise ValueError(f"Unknown CP-HQMR mode: {mode}")
        q_cov = self.query_norm(query)
        k5, v5 = self.scale5(h5); k4, _ = self.scale4(h4)
        c5 = direct_affinity(q_cov, k5)
        q_disc = self.update5(q_cov, c5, v5)
        a_cov4, a_disc4 = direct_affinity(q_cov, k4), direct_affinity(q_disc, k4)
        c4, d4 = c5 + a_cov4, c5 + a_disc4
        fusion = coverage_preserving_fusion(c4, d4)
        semantic = {"full": fusion["fused"], "bilinear_only": fusion["fused"],
                    "old_f3_semantic": fusion["fused"], "discriminative_only": d4,
                    "coverage_only": c4, "simple_average": .5 * (c4 + d4)}[mode]
        dgsr = None
        if h3 is None:
            final = semantic
        elif mode == "old_f3_semantic":
            upsampled = F.interpolate(semantic, size=h3.shape[-2:], mode="bilinear", align_corners=False)
            final = upsampled + direct_affinity(q_disc, h3)
        else:
            dgsr = self.dgsr(semantic, h3, enabled=mode != "bilinear_only")
            final = dgsr["logits"]
        return {"basis_logits": final, "basis": final.sigmoid(), "semantic_logits": semantic,
                "C5": c5, "C4": c4, "D4": d4, "M4": fusion["fused"],
                "q_cov": q_cov, "q_disc": q_disc, "A_cov4": a_cov4, "A_disc4": a_disc4,
                "rescue": fusion["rescue"], "rescue_gate": fusion["rescue_gate"],
                "dgsr": dgsr, "mode": mode, "h4_semantic_endpoint": True}


__all__ = ["CPHQMR", "DetailGuidedSpatialRestoration", "coverage_preserving_fusion"]
