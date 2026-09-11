"""Zero-parameter class-conditioned affinity completion for GCQM masks."""
from __future__ import annotations

import torch
from torch.nn import functional as F


EPS = 1.0e-8


def _local_affinity(pixel_feature: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached row-normalized 3x3 affinities and their valid-neighbor mask."""
    if mode not in {"feature", "uniform"}:
        raise ValueError(f"Unknown CCAC affinity mode: {mode}")
    with torch.no_grad(), torch.autocast(device_type=pixel_feature.device.type, enabled=False):
        feature = F.normalize(pixel_feature.detach().float(), dim=1, eps=EPS)
        batch, channels, height, width = feature.shape
        patches = F.unfold(feature, kernel_size=3, padding=1).reshape(batch, channels, 9, height * width)
        valid = F.unfold(torch.ones(batch, 1, height, width, device=feature.device),
                         kernel_size=3, padding=1).reshape(batch, 9, height * width).bool()
        if mode == "feature":
            center = feature.flatten(2).unsqueeze(2)
            affinity = (patches * center).sum(1).relu() * valid
            affinity[:, 4] = 1.0
        else:
            affinity = valid.float()
        affinity = affinity / affinity.sum(1, keepdim=True).clamp_min(EPS)
    return affinity.detach(), valid.detach()


def _rival_gate(mask: torch.Tensor) -> torch.Tensor:
    """Compute 1-max rival probability per class with a stopped rival side path."""
    classes = mask.shape[1]
    if classes == 1:
        return torch.ones_like(mask)
    rival = mask.detach().unsqueeze(1).expand(-1, classes, -1, -1, -1)
    diagonal = torch.eye(classes, dtype=torch.bool, device=mask.device)[None, :, :, None, None]
    rival = rival.masked_fill(diagonal, -1.0).amax(2)
    return (1.0 - rival).clamp(0.0, 1.0)


def ccac_complete(mask: torch.Tensor, pixel_feature: torch.Tensor, iterations: int = 2,
                  affinity_mode: str = "feature", use_rival: bool = True) -> dict:
    """Fill spatially supported holes without suppressing existing GCQM evidence."""
    if mask.ndim != 4 or pixel_feature.ndim != 4 or mask.shape[0] != pixel_feature.shape[0]:
        raise ValueError("CCAC expects BCHW masks and pixel features with a shared batch")
    if mask.shape[-2:] != pixel_feature.shape[-2:]:
        pixel_feature = F.interpolate(pixel_feature, mask.shape[-2:], mode="bilinear", align_corners=False)
    if iterations < 0:
        raise ValueError("CCAC iterations must be non-negative")
    with torch.autocast(device_type=mask.device.type, enabled=False):
        base = mask.float().clamp(0.0, 1.0)
        affinity, valid = _local_affinity(pixel_feature, affinity_mode)
        current = base
        deltas = []
        protected = []
        for _ in range(iterations):
            batch, classes, height, width = current.shape
            neighbors = F.unfold(current, kernel_size=3, padding=1).reshape(
                batch, classes, 9, height * width
            )
            consensus = (neighbors * affinity[:, None]).sum(2).reshape(batch, classes, height, width)
            gate = _rival_gate(current) if use_rival else torch.ones_like(current)
            potential = (1.0 - current) * (consensus - current).relu()
            delta = gate * potential
            current = (current + delta).clamp(0.0, 1.0)
            deltas.append(delta)
            protected.append(potential - delta)
        total_delta = current - base
        entropy = -(affinity.clamp_min(EPS) * affinity.clamp_min(EPS).log()).sum(1)
        off_center = valid.clone(); off_center[:, 4] = False
        positive_neighbor = ((affinity > 0) & off_center).float().sum(1) / off_center.float().sum(1).clamp_min(1)
        diagnostics = {
            "completion_mass_mean": float(total_delta.detach().mean()),
            "changed_fraction": float((total_delta.detach() > 1.0e-7).float().mean()),
            "rival_protected_mass_mean": float(torch.stack(protected).detach().mean()) if protected else 0.0,
            "max_negative_change": float((-total_delta.detach()).clamp_min(0).max()),
            "affinity_entropy_mean": float(entropy.mean()),
            "positive_neighbor_affinity_fraction": float(positive_neighbor.mean()),
            "iterations": int(iterations),
            "affinity_mode": affinity_mode,
            "use_rival": bool(use_rival),
        }
    return {
        "base": base, "restored": current, "primary_output": current,
        "affinity": affinity, "iteration_deltas": deltas, "diagnostics": diagnostics,
        "pixel_feature_detached": True, "rival_sidepath_detached": bool(use_rival),
    }


__all__ = ["ccac_complete"]
