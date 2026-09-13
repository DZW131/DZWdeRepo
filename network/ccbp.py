"""Class-Competitive Basis Purification (CCBP)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


EPS = 1.0e-8


def balanced_competitive_loss(logits: torch.Tensor, positive: torch.Tensor,
                              present: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Class-balanced CE over reliable foreground pixels only."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        value = F.interpolate(logits.float(), positive.shape[-2:], mode="bilinear", align_corners=False)
        reliable = positive.bool() & present[:, :, None, None].bool()
        losses, counts = [], []
        for cls in range(value.shape[1]):
            mask = reliable[:, cls]
            counts.append(int(mask.sum()))
            if mask.any():
                target = torch.full((int(mask.sum()),), cls, device=value.device, dtype=torch.long)
                losses.append(F.cross_entropy(value.permute(0, 2, 3, 1)[mask], target))
        loss = torch.stack(losses).mean() if losses else value.sum() * 0.0
    return loss, {"class_counts": counts, "classes_used": len(losses),
                  "reliable_foreground_only": True, "background_ignored": True,
                  "uncertain_ignored": True, "class_balanced": True}


class CCBP(nn.Module):
    """Suppress-only strongest-rival competition in the existing H4 space."""

    MODES = {"full", "off", "raw_space", "no_g_prior", "mean_rival", "hard_gate"}

    def __init__(self, dim: int = 256):
        super().__init__()
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.semantic_projection = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.eye_(self.query_projection.weight)
        nn.init.eye_(self.semantic_projection.weight[:, :, 0, 0])
        initial = 4.0 / 19.0
        self.theta_gamma = nn.Parameter(torch.tensor(math.log(initial / (1.0 - initial))))

    @property
    def gamma(self) -> torch.Tensor:
        return 1.0 + 19.0 * self.theta_gamma.sigmoid()

    @staticmethod
    def class_prototype(query: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bqc,bqd->bcd", weights.detach().float(), query.detach().float())

    @staticmethod
    def rival(logits: torch.Tensor, present: torch.Tensor, mean: bool = False) -> torch.Tensor:
        batch, classes, height, width = logits.shape
        output = torch.empty_like(logits)
        active = present.bool()
        for cls in range(classes):
            valid = active.clone(); valid[:, cls] = False
            candidates = logits.masked_fill(~valid[:, :, None, None], float("-inf"))
            if mean:
                count = valid.sum(1).clamp_min(1)[:, None, None]
                candidates = torch.where(valid[:, :, None, None], logits, torch.zeros_like(logits))
                score = candidates.sum(1) / count
            else:
                score = candidates.max(1).values
            output[:, cls] = score
        return output

    def forward(self, basis: torch.Tensor, weights: torch.Tensor, query: torch.Tensor,
                key4: torch.Tensor, class_prior: torch.Tensor, present: torch.Tensor,
                mode: str = "full") -> dict:
        if mode not in self.MODES: raise ValueError(f"Unknown CCBP mode: {mode}")
        prototype = self.class_prototype(query, weights)
        with torch.autocast(device_type=basis.device.type, enabled=False):
            q = prototype.float() if mode == "raw_space" else self.query_projection(prototype.float())
            z = key4.detach().float() if mode == "raw_space" else self.semantic_projection(key4.detach().float())
            q = F.normalize(q.float(), dim=-1); z = F.normalize(z.float(), dim=1)
            cosine = torch.einsum("bcd,bdhw->bchw", q, z)
            logits = self.gamma.float() * cosine
            if mode != "no_g_prior": logits = logits + class_prior.detach().float().clamp_min(EPS).log()[:, :, None, None]
            rival = self.rival(logits, present, mean=mode == "mean_rival")
            active_count = present.bool().sum(1)
            if mode == "off": gate4 = torch.ones_like(logits)
            elif mode == "hard_gate": gate4 = (logits >= rival).float()
            else: gate4 = torch.exp(-F.relu(rival - logits))
            gate4 = torch.where((active_count <= 1)[:, None, None, None], torch.ones_like(gate4), gate4)
            gate4 = torch.where(present.bool()[:, :, None, None], gate4, torch.ones_like(gate4))
            gate = F.interpolate(gate4, basis.shape[-2:], mode="bilinear", align_corners=False)
            purified_basis = basis.float()[:, :, None] * gate.detach().float()[:, None]
            if mode == "off":
                mixture = torch.einsum("bqc,bqhw->bchw", weights.detach().float(), basis.float()).clamp(0.0, 1.0)
            else:
                mixture = torch.einsum("bqc,bqchw->bchw", weights.detach().float(), purified_basis).clamp(0.0, 1.0)
        if not all(torch.isfinite(value).all() for value in (logits, gate, purified_basis, mixture)):
            raise FloatingPointError("Non-finite CCBP output")
        return {"prototype": prototype, "projected_prototype": q, "semantic": z,
                "cosine": cosine, "logits": logits, "rival": rival, "gate_h4": gate4,
                "gate": gate, "purified_basis": purified_basis, "mixture": mixture,
                "gamma": self.gamma, "mode": mode, "suppress_only": True,
                "gate_detached_for_base": True}


__all__ = ["CCBP", "EPS", "balanced_competitive_loss"]
