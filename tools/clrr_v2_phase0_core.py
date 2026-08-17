"""Frozen analytical CLRR-v2 operators used only by the Phase-0 audit."""

import math

import torch
import torch.nn.functional as F


EPS = 1e-8
RELIABILITY_FALLBACK_EPS = 1e-6
DIRECTION_ZERO_EPS = 1e-8


def normalized_entropy_reliability(probability, eps=EPS):
    """Return 1-H(P)/log(C) for a detached foreground probability map."""
    if probability.ndim != 4 or probability.shape[1] < 2:
        raise ValueError("probability must have shape [B,C,H,W] with C>=2")
    probability = probability.detach().float()
    entropy = -(
        probability * probability.clamp_min(eps).log()
    ).sum(dim=1, keepdim=True)
    reliability = 1.0 - entropy / math.log(probability.shape[1])
    return reliability.clamp(0.0, 1.0).detach()


def leave_one_out_consensus(
    probabilities,
    target_name,
    target_size,
    fallback_eps=RELIABILITY_FALLBACK_EPS,
):
    """Reliability-weighted consensus from every hierarchy except target."""
    if target_name not in probabilities:
        raise KeyError(f"Unknown target hierarchy: {target_name}")
    other_names = [name for name in probabilities if name != target_name]
    if not other_names:
        raise ValueError("Leave-one-out consensus requires another hierarchy")

    resized = {}
    reliabilities = {}
    for name in other_names:
        probability = probabilities[name].detach().float()
        if probability.shape[-2:] != tuple(target_size):
            probability = F.interpolate(
                probability,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        probability = probability / probability.sum(
            dim=1, keepdim=True
        ).clamp_min(EPS)
        resized[name] = probability.detach()
        reliabilities[name] = normalized_entropy_reliability(probability)

    reliability_sum = torch.stack(
        list(reliabilities.values()), dim=0
    ).sum(dim=0)
    weighted_sum = sum(
        reliabilities[name] * resized[name] for name in other_names
    )
    weighted = weighted_sum / reliability_sum.clamp_min(EPS)
    simple_mean = torch.stack(
        [resized[name] for name in other_names], dim=0
    ).mean(dim=0)
    use_fallback = reliability_sum < fallback_eps
    consensus = torch.where(use_fallback, simple_mean, weighted)
    consensus = consensus / consensus.sum(
        dim=1, keepdim=True
    ).clamp_min(EPS)
    return {
        "source_names": tuple(other_names),
        "resized_probabilities": {
            name: value.detach() for name, value in resized.items()
        },
        "view_reliabilities": {
            name: value.detach() for name, value in reliabilities.items()
        },
        "reliability_sum": reliability_sum.detach(),
        "fallback_mask": use_fallback.detach(),
        "consensus": consensus.detach(),
        "consensus_reliability": normalized_entropy_reliability(consensus),
    }


def classifier_backprojection(semantic_error, classifier_weight):
    """Compute W^T E for a 1x1 Conv2d classifier without learning a projector."""
    if classifier_weight.ndim != 4 or classifier_weight.shape[-2:] != (1, 1):
        raise ValueError("classifier weight must have shape [C_cls,C_feat,1,1]")
    semantic_error = semantic_error.detach().float()
    weight = classifier_weight.detach().float()[:, :, 0, 0]
    if semantic_error.shape[1] != weight.shape[0]:
        raise ValueError("semantic error classes do not match classifier output")
    return torch.einsum("oc,bohw->bchw", weight, semantic_error).detach()


def analytical_virtual_correction(
    feature,
    probability,
    consensus_state,
    classifier_weight,
    gamma_sem,
    gamma_ctx,
    eta=0.05,
):
    """Build the frozen bounded CLRR-v2 virtual update in FP32."""
    if eta != 0.05:
        raise ValueError("Phase-0 eta is frozen at 0.05")
    feature = feature.detach().float()
    probability = probability.detach().float()
    consensus = consensus_state["consensus"].detach().float()
    rho = consensus_state["consensus_reliability"].detach().float()
    semantic_error = (consensus - probability).detach()
    mismatch = 0.5 * semantic_error.abs().sum(dim=1, keepdim=True)

    backprojection = classifier_backprojection(
        semantic_error, classifier_weight
    )
    direction_rms = backprojection.square().mean(
        dim=1, keepdim=True
    ).sqrt()
    normalized_direction = torch.where(
        direction_rms < DIRECTION_ZERO_EPS,
        torch.zeros_like(backprojection),
        backprojection / direction_rms.clamp_min(EPS),
    ).detach()
    feature_scale = feature.square().mean(
        dim=1, keepdim=True
    ).add(EPS).sqrt().detach()
    maturity = (
        gamma_sem.detach().float().abs()
        + gamma_ctx.detach().float().abs()
    ).clamp(0.0, 1.0).view(1, 1, 1, 1).detach()
    delta = (
        maturity
        * rho
        * mismatch
        * feature_scale
        * normalized_direction
    ).detach()
    update = (eta * delta).detach()
    updated_feature = (feature + update).detach()
    update_rms = update.square().mean(dim=1, keepdim=True).sqrt()
    update_ratio = update_rms / feature_scale.clamp_min(EPS)
    return {
        "consensus": consensus,
        "consensus_reliability": rho,
        "semantic_error": semantic_error,
        "mismatch": mismatch.detach(),
        "backprojection": backprojection,
        "direction_rms": direction_rms.detach(),
        "normalized_direction": normalized_direction,
        "feature_scale": feature_scale,
        "maturity": maturity,
        "delta": delta,
        "update": update,
        "updated_feature": updated_feature,
        "update_ratio": update_ratio.detach(),
    }
