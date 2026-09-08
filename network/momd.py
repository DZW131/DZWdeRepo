"""Zero-parameter Mass-Preserving Ownership Mixture Decoding."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from network.hqrf_targets import IGNORE


EPS = 1.0e-8
LOGIT_EPS = 1.0e-6


def resize_responsibility(responsibility_class: torch.Tensor, memory_hw: tuple[int, int],
                          output_hw: tuple[int, int], cls: int) -> torch.Tensor:
    """Detach and bilinearly resize one class of R to the mask grid in FP32."""
    batch, queries, _, _ = responsibility_class.shape
    height, width = memory_hw
    value = responsibility_class[..., cls].detach().float().reshape(batch * queries, 1, height, width)
    return F.interpolate(value, size=output_hw, mode="bilinear", align_corners=False).reshape(
        batch, queries, *output_hw
    )


def route_one_class(responsibility_class: torch.Tensor, memory_hw: tuple[int, int],
                    output_hw: tuple[int, int], locality: torch.Tensor, cls: int) -> dict:
    """Build locality-aware A, falling back to globally normalized R per pixel."""
    with torch.autocast(device_type=responsibility_class.device.type, enabled=False):
        resized = resize_responsibility(responsibility_class, memory_hw, output_hw, cls)
        global_route = resized / resized.sum(1, keepdim=True).clamp_min(EPS)
        local_mass = resized * locality[None].float()
        denominator = local_mass.sum(1, keepdim=True)
        local_route = local_mass / denominator.clamp_min(EPS)
        fallback = denominator <= EPS
        routing = torch.where(fallback, global_route, local_route)
        error = (routing.sum(1) - 1.0).abs()
        if not torch.isfinite(routing).all() or float(error.max()) > 1.0e-5:
            raise FloatingPointError(f"MOMD routing normalization failed: {float(error.max())}")
    return {
        "responsibility": resized.detach(), "routing": routing.detach(),
        "fallback": fallback[:, 0].detach(), "sum_error_max": float(error.max()),
        "sum_error_mean": float(error.mean()),
    }


def mixture_decode(base_logits: torch.Tensor, responsibility_class: torch.Tensor,
                   memory_hw: tuple[int, int], locality: torch.Tensor,
                   materialize: bool = False) -> dict:
    """Decode class masks F=sum_i A_i*sigmoid(Z_i); optionally retain R/A/C/Q."""
    with torch.autocast(device_type=base_logits.device.type, enabled=False):
        base = base_logits.float().sigmoid()
        mixtures, routes = [], []
        for cls in range(responsibility_class.shape[-1]):
            route = route_one_class(responsibility_class, memory_hw, base.shape[-2:], locality, cls)
            mixture = (route["routing"] * base).sum(1)
            mixtures.append(mixture)
            routes.append(route)
        final = torch.stack(mixtures, dim=1).clamp(0.0, 1.0)
        payload = {
            "base_probability": base,
            "mixture": final,
            "primary_output": final,
            "routing_sum_error_max": max(x["sum_error_max"] for x in routes),
            "routing_sum_error_mean": sum(x["sum_error_mean"] for x in routes) / len(routes),
            "fallback_fraction": float(torch.stack([x["fallback"] for x in routes], 1).float().mean()),
            "responsibility_sidepath_detached": True,
        }
        if materialize:
            resized = torch.stack([x["responsibility"] for x in routes], dim=2)
            routing = torch.stack([x["routing"] for x in routes], dim=2)
            contribution = routing * base[:, :, None]
            posterior = contribution / final[:, None].clamp_min(EPS)
            valid = final > EPS
            c_error = (contribution.sum(1) - final).abs()
            q_error = (posterior.sum(1) - 1.0).abs()[valid]
            fallback = torch.stack([x["fallback"] for x in routes], dim=1)
            payload.update({
                "responsibility_resized": resized, "routing": routing,
                "contribution": contribution, "posterior_share": posterior,
                "fallback": fallback, "contribution_sum_error_max": float(c_error.max().detach()),
                "posterior_sum_error_max": float(q_error.max().detach()) if q_error.numel() else 0.0,
            })
    return payload


def mixture_class_bce(mixture: torch.Tensor, target: torch.Tensor,
                      present: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Class-level tri-state BCE for Stage2/3, with no query assignment or R reweighting."""
    with torch.autocast(device_type=mixture.device.type, enabled=False):
        probability = mixture.float().clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)
        logits = torch.logit(probability)
        resized = F.interpolate(target.float(), probability.shape[-2:], mode="nearest").to(torch.int8)
        valid = (resized != IGNORE) & present[:, :, None, None].bool()
        counts = valid.flatten(2).sum(-1)
        eligible = present.bool() & (counts >= 4)
        safe = resized.clamp_min(0).float()
        element = F.binary_cross_entropy_with_logits(logits, safe, reduction="none") * valid
        per_class = element.flatten(2).sum(-1) / counts.clamp_min(1)
        loss = per_class[eligible].mean() if eligible.any() else logits.sum() * 0.0
    return loss, {
        "eligible": eligible, "valid_labeled_pixels": counts, "valid_classes": int(eligible.sum()),
        "class_level_mixture_bce": True, "query_assignment": False,
        "ownership_weighted_bce": False, "second_responsibility_weighting": False,
    }


def pca_reference_envelope(base_probability: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
    """Detached same-model non-ownership reference E_ref=sum soft PCA weights * B."""
    with torch.no_grad():
        alpha = joint.detach().float() / joint.detach().float().sum(1, keepdim=True).clamp_min(EPS)
        return (alpha[..., None, None] * base_probability.detach().float()[:, :, None]).sum(1)


__all__ = ["EPS", "LOGIT_EPS", "resize_responsibility", "route_one_class", "mixture_decode",
           "mixture_class_bce", "pca_reference_envelope"]
