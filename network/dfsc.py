"""Dual-frequency selective relation affinity and monotone consensus completion."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


EPS = 1.0e-6


def _valid_neighbors(batch: int, height: int, width: int, device) -> torch.Tensor:
    return F.unfold(torch.ones(batch, 1, height, width, device=device), 3, padding=1).reshape(
        batch, 9, height * width
    ).bool()


def relation_labels(target_detail: dict, output_hw: tuple[int, int]) -> torch.Tensor:
    """Map frozen reliable foreground/background targets to one local relation label map."""
    positive = F.interpolate(target_detail["positive"].float(), output_hw, mode="nearest").bool()
    background = F.interpolate(target_detail["reliable_background"][:, None].float(), output_hw,
                               mode="nearest")[:, 0].bool()
    label = torch.full((positive.shape[0], *output_hw), -1, dtype=torch.long, device=positive.device)
    label[background] = positive.shape[1]
    any_positive = positive.any(1)
    label[any_positive] = positive.float().argmax(1)[any_positive]
    return label.detach()


def pair_masks(label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive/negative masks for valid non-self 3x3 pairs."""
    batch, height, width = label.shape
    neighbor = F.unfold((label + 1).float()[:, None], 3, padding=1).reshape(batch, 9, height * width).long() - 1
    center = label.flatten(1)[:, None]
    valid = _valid_neighbors(batch, height, width, label.device) & (center >= 0) & (neighbor >= 0)
    valid[:, 4] = False
    return (valid & (neighbor == center)).detach(), (valid & (neighbor != center)).detach()


def balanced_relation_loss(raw_affinity: torch.Tensor, positive: torch.Tensor,
                           negative: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Balanced BCE over reliable positive and negative neighbor pairs."""
    affinity = raw_affinity.float().clamp(EPS, 1.0 - EPS)
    terms = []
    if positive.any(): terms.append(-affinity[positive].log().mean())
    if negative.any(): terms.append(-(1.0 - affinity[negative]).log().mean())
    loss = sum(terms) / len(terms) if terms else affinity.sum() * 0.0
    return loss, {"positive_pairs": int(positive.sum()), "negative_pairs": int(negative.sum()),
                  "positive_term_available": bool(positive.any()), "negative_term_available": bool(negative.any()),
                  "balanced_available_terms": len(terms)}


class DFRA(nn.Module):
    """Tiny learned dual-frequency relation estimator (exactly 16,386 parameters)."""
    def __init__(self, channels: int = 256, relation_dim: int = 32):
        super().__init__()
        self.low_projection = nn.Conv2d(channels, relation_dim, 1, bias=False)
        self.high_projection = nn.Conv2d(channels, relation_dim, 1, bias=False)
        initial = math.log(math.expm1(1.0))
        self.theta_low = nn.Parameter(torch.tensor(initial))
        self.theta_high = nn.Parameter(torch.tensor(initial))
        nn.init.xavier_uniform_(self.low_projection.weight)
        nn.init.xavier_uniform_(self.high_projection.weight)

    @staticmethod
    def frequency_components(pixel_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        relation_feature = pixel_feature.detach().float()
        low = F.avg_pool2d(relation_feature, 5, stride=1, padding=2)
        high = relation_feature - F.avg_pool2d(relation_feature, 3, stride=1, padding=1)
        return low, high

    @staticmethod
    def _distance(embedding: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = embedding.shape
        patches = F.unfold(embedding, 3, padding=1).reshape(batch, channels, 9, height * width)
        center = embedding.flatten(2).unsqueeze(2)
        return (1.0 - (patches * center).sum(1)).clamp(0.0, 2.0)

    def forward(self, pixel_feature: torch.Tensor, target_detail: dict | None = None,
                mode: str = "full") -> dict:
        if mode not in {"full", "uniform", "raw_cosine", "low_only", "high_only"}:
            raise ValueError(f"Unknown DFRA mode: {mode}")
        with torch.autocast(device_type=pixel_feature.device.type, enabled=False):
            low, high = self.frequency_components(pixel_feature)
            z_low = F.normalize(self.low_projection(low), dim=1, eps=EPS)
            z_high = F.normalize(self.high_projection(high), dim=1, eps=EPS)
            d_low, d_high = self._distance(z_low), self._distance(z_high)
            beta_low, beta_high = F.softplus(self.theta_low), F.softplus(self.theta_high)
            batch, _, height, width = pixel_feature.shape
            valid = _valid_neighbors(batch, height, width, pixel_feature.device)
            if mode == "uniform":
                raw = valid.float()
            elif mode == "raw_cosine":
                normalized = F.normalize(pixel_feature.detach().float(), dim=1, eps=EPS)
                raw = (1.0 - self._distance(normalized)).relu() * valid
            else:
                exponent = (beta_low * d_low if mode != "high_only" else 0.0) + (beta_high * d_high if mode != "low_only" else 0.0)
                raw = torch.exp(-exponent) * valid
            raw[:, 4] = 1.0
            affinity = raw / raw.sum(1, keepdim=True).clamp_min(EPS)
            positive = negative = None
            if target_detail is not None:
                labels = relation_labels(target_detail, (height, width))
                positive, negative = pair_masks(labels)
                relation_loss, loss_detail = balanced_relation_loss(raw, positive, negative)
            else:
                labels = None; relation_loss = raw.sum() * 0.0
                loss_detail = {"positive_pairs": 0, "negative_pairs": 0, "positive_term_available": False,
                               "negative_term_available": False, "balanced_available_terms": 0}
        return {"low_frequency": low, "high_frequency": high, "embedding_low": z_low,
                "embedding_high": z_high, "distance_low": d_low, "distance_high": d_high,
                "raw_affinity": raw, "affinity": affinity, "affinity_comp": affinity.detach(),
                "beta_low": beta_low, "beta_high": beta_high, "labels": labels,
                "positive_pairs": positive, "negative_pairs": negative,
                "relation_loss": relation_loss.float(), "relation_loss_detail": loss_detail,
                "relation_feature_detached": True, "completion_affinity_detached": True, "mode": mode}


def mcc_complete(mask: torch.Tensor, affinity: torch.Tensor, iterations: int = 2,
                 update: str = "mcc") -> dict:
    """Apply fixed two-hop monotone consensus, with old weak update only for ablation."""
    if update not in {"mcc", "old_weak"}: raise ValueError(f"Unknown completion update: {update}")
    with torch.autocast(device_type=mask.device.type, enabled=False):
        base = mask.float().clamp(0.0, 1.0); current = base; deltas = []
        for _ in range(iterations):
            batch, classes, height, width = current.shape
            neighbors = F.unfold(current, 3, padding=1).reshape(batch, classes, 9, height * width)
            consensus = (neighbors * affinity.detach()[:, None]).sum(2).reshape_as(current)
            delta = (consensus - current).relu()
            if update == "old_weak": delta = (1.0 - current) * delta
            current = (current + delta).clamp(0.0, 1.0); deltas.append(delta)
    return {"base": base, "restored": current, "primary_output": current, "iteration_deltas": deltas,
            "iterations": iterations, "update": update, "completion_affinity_detached": True}


__all__ = ["DFRA", "relation_labels", "pair_masks", "balanced_relation_loss", "mcc_complete"]
