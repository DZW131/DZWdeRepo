"""Pure diagnostic functions for OSMF-v1.2 Phase-0M."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from tools.osmf_v12_audit.decision import percentile


NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def local_affinity_map(
    feature: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 8-neighbor cosine affinities and a fixed valid-neighbor mask."""
    if feature.ndim != 4:
        raise ValueError("feature must have shape [B,C,H,W]")
    normalized = F.normalize(feature.float(), p=2, dim=1, eps=eps)
    _, _, height, width = normalized.shape
    padded = F.pad(normalized, (1, 1, 1, 1))
    affinities, masks = [], []
    for dy, dx in NEIGHBOR_OFFSETS:
        neighbor = padded[
            :, :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width
        ]
        affinities.append((normalized * neighbor).sum(dim=1))
        mask = torch.ones(
            (1, height, width), dtype=torch.bool, device=feature.device
        )
        if dy < 0:
            mask[:, : -dy, :] = False
        elif dy > 0:
            mask[:, height - dy :, :] = False
        if dx < 0:
            mask[:, :, : -dx] = False
        elif dx > 0:
            mask[:, :, width - dx :] = False
        masks.append(mask)
    return torch.stack(affinities, dim=1), torch.stack(masks, dim=1)


def inverse_align_affinity(
    affinity: torch.Tensor, mask: torch.Tensor, flip_dimension: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-align spatial coordinates and directional affinity channels."""
    if flip_dimension not in (2, 3):
        raise ValueError("flip_dimension must be vertical (2) or horizontal (3)")
    spatial_affinity = torch.flip(affinity, dims=(flip_dimension,))
    spatial_mask = torch.flip(mask, dims=(flip_dimension,))
    offset_to_index = {offset: index for index, offset in enumerate(NEIGHBOR_OFFSETS)}
    permutation = []
    for dy, dx in NEIGHBOR_OFFSETS:
        transformed = (-dy, dx) if flip_dimension == 2 else (dy, -dx)
        permutation.append(offset_to_index[transformed])
    index = torch.tensor(permutation, dtype=torch.long, device=affinity.device)
    return spatial_affinity.index_select(1, index), spatial_mask.index_select(1, index)


def affinity_equivariance_error(
    feature_a: torch.Tensor,
    feature_b: torch.Tensor,
    flip_dimension: int,
) -> torch.Tensor:
    affinity_a, mask_a = local_affinity_map(feature_a)
    affinity_b, mask_b = local_affinity_map(feature_b)
    aligned_b, aligned_mask_b = inverse_align_affinity(
        affinity_b, mask_b, flip_dimension
    )
    valid = mask_a & aligned_mask_b
    difference = (affinity_a - aligned_b).abs()
    return difference.masked_select(valid.expand_as(difference)).mean()


def flatten_gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flattened = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            flattened.append(torch.zeros_like(parameter, memory_format=torch.preserve_format).flatten())
        else:
            flattened.append(gradient.detach().float().flatten())
    return torch.cat(flattened)


def safe_cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    left = left.float()
    right = right.float()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= eps:
        return 0.0
    return float(torch.dot(left, right).div(denominator).cpu())


def morphology_gradient_competition(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> dict:
    names = ("eq", "base", "sem_pres", "orth", "rec")
    vectors = {
        name: flatten_gradients(
            losses[name], parameters, retain_graph=index < len(names) - 1
        )
        for index, name in enumerate(names)
    }
    eq = vectors["eq"]
    return {
        "eq_grad_norm": float(torch.linalg.vector_norm(eq).cpu()),
        "base_grad_norm": float(torch.linalg.vector_norm(vectors["base"]).cpu()),
        "sem_grad_norm": float(torch.linalg.vector_norm(vectors["sem_pres"]).cpu()),
        "orth_grad_norm": float(torch.linalg.vector_norm(vectors["orth"]).cpu()),
        "rec_grad_norm": float(torch.linalg.vector_norm(vectors["rec"]).cpu()),
        "cos_eq_base": safe_cosine(eq, vectors["base"]),
        "cos_eq_sem": safe_cosine(eq, vectors["sem_pres"]),
        "cos_eq_orth": safe_cosine(eq, vectors["orth"]),
        "cos_eq_rec": safe_cosine(eq, vectors["rec"]),
        "finite": all(bool(torch.isfinite(vector).all()) for vector in vectors.values()),
    }


def causal_statistics(rows: Sequence[Mapping]) -> dict:
    deltas = [float(row["delta"]) for row in rows]
    if not deltas:
        raise ValueError("same-pair causal rows cannot be empty")
    neutral_flags = [abs(delta) < 1e-6 for delta in deltas]
    improved = sum(delta < 0 and not neutral for delta, neutral in zip(deltas, neutral_flags))
    harmed = sum(delta > 0 and not neutral for delta, neutral in zip(deltas, neutral_flags))
    neutral = sum(neutral_flags)
    ordered = sorted(deltas)
    return {
        "num_eq_steps": len(deltas),
        "num_improved": improved,
        "num_harmed": harmed,
        "num_neutral": neutral,
        "improved_fraction": improved / len(deltas),
        "harmed_fraction": harmed / len(deltas),
        "neutral_fraction": neutral / len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "median_delta": percentile(ordered, 0.50),
        "p25_delta": percentile(ordered, 0.25),
        "p75_delta": percentile(ordered, 0.75),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
    }


def replication_deviations(
    observed: Mapping[str, float], reference: Mapping[str, float]
) -> tuple[dict[str, float], bool]:
    deviations = {
        name: abs(float(observed[name]) - float(value)) / (abs(float(value)) + 1e-12)
        for name, value in reference.items()
    }
    return deviations, any(value > 0.25 for value in deviations.values())


def decide_phase0m(
    *,
    causal: Mapping[str, float],
    fixed_raw_delta: float,
    fixed_affinity_delta: float,
    healthy: bool,
    mean_eq_base_cosine: float,
    replication_instability: bool,
) -> tuple[str, list[str], list[str]]:
    flags, reasons = [], []
    improved_fraction = float(causal["improved_fraction"])
    mean_delta = float(causal["mean_delta"])
    if improved_fraction >= 0.75 and mean_delta < 0:
        flags.append("SAME_PAIR_CAUSAL_VALID")
    elif improved_fraction >= 0.50 and mean_delta < 0:
        flags.append("SAME_PAIR_CAUSAL_AMBIGUOUS")
    else:
        flags.append("SAME_PAIR_CAUSAL_INVALID")
    if mean_eq_base_cosine < -0.50:
        flags.append("STRONG_MORPHOLOGY_TASK_CONFLICT")
    elif mean_eq_base_cosine < -0.30:
        flags.append("MORPHOLOGY_TASK_CONFLICT")
    if fixed_raw_delta >= 0 and fixed_affinity_delta < -0.005:
        flags.append("LOCAL_GEOMETRY_IMPROVES_DESPITE_RAW_FEATURE_EQ_FAILURE")
    if replication_instability:
        flags.append("REPLICATION_INSTABILITY")

    if improved_fraction < 0.50 or mean_delta >= 0:
        decision = "MORPH_EQ_OBJECTIVE_INVALID"
        reasons.append("SAME_PAIR_CAUSAL_INVALID")
    elif fixed_raw_delta >= 0 and fixed_affinity_delta < -0.005:
        decision = "MORPH_EQ_METRIC_MISMATCH_REVIEW"
        reasons.append("RAW_EQ_FAILS_WHILE_AFFINITY_IMPROVES")
    elif improved_fraction >= 0.75 and fixed_raw_delta < 0 and healthy:
        decision = "MORPH_EQ_OBJECTIVE_VALID"
    elif fixed_raw_delta >= 0:
        decision = "MORPH_EQ_GENERALIZATION_FAILURE"
        reasons.append("SAME_PAIR_EFFECT_DOES_NOT_GENERALIZE_TO_FIXED_PROBE")
    else:
        decision = "MORPH_EQ_METRIC_MISMATCH_REVIEW"
        reasons.append("CAUSAL_EVIDENCE_AMBIGUOUS_FOR_VALID_DECISION")

    if not healthy:
        reasons.append("REPRESENTATION_SAFETY_FAILURE")
        if decision == "MORPH_EQ_OBJECTIVE_VALID":
            decision = "MORPH_EQ_METRIC_MISMATCH_REVIEW"
    if replication_instability:
        reasons.append("REPLICATION_INSTABILITY")
        if decision == "MORPH_EQ_OBJECTIVE_VALID":
            decision = "MORPH_EQ_METRIC_MISMATCH_REVIEW"
    return decision, sorted(set(flags)), sorted(set(reasons))


def all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)
