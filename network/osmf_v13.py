"""OSMF-v1.3 local structural morphology learning.

The factorizer and all non-morphology objectives are frozen from OSMF-v1.2.
Only the morphology supervision is replaced by a direction-aware local
8-neighbour affinity objective.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from network.osmf_v12 import (
    OSMFV12Factorizer,
    cross_subspace_covariance,
    inverse_align_morphology,
    orthogonality_loss,
    reconstruction_cosine,
    reconstruction_loss,
    semantic_preservation_agreement,
    semantic_preservation_loss,
)


OSMF_STRUCTURAL_INTERVAL = 4
OSMF_LAMBDA_SEM = 0.05
OSMF_LAMBDA_STRUCT = 0.05
OSMF_LAMBDA_ORTH = 0.05
OSMF_LAMBDA_REC = 0.10

NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)


class OSMFV13Factorizer(OSMFV12Factorizer):
    """Versioned name for the exactly frozen v1.2 factorizer."""


def local_affinity_map(
    feature: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute channel-normalized 8-neighbour cosine affinities and masks."""

    if feature.ndim != 4:
        raise ValueError("feature must have shape [B,C,H,W]")
    normalized = F.normalize(feature.float(), p=2, dim=1, eps=eps)
    _, _, height, width = normalized.shape
    padded = F.pad(normalized, (1, 1, 1, 1))
    affinities, masks = [], []
    for dy, dx in NEIGHBOR_OFFSETS:
        neighbour = padded[
            :, :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width
        ]
        affinities.append((normalized * neighbour).sum(dim=1))
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Undo a spatial flip and permute directional neighbour channels."""

    if affinity.ndim != 4 or affinity.shape[1] != 8:
        raise ValueError("affinity must have shape [B,8,H,W]")
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


def structural_affinity_loss(
    morphology_a: torch.Tensor,
    morphology_b: torch.Tensor,
    flip_dimension: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Masked SmoothL1 between direction-aware inverse-aligned affinities."""

    affinity_a, mask_a = local_affinity_map(morphology_a)
    affinity_b, mask_b = local_affinity_map(morphology_b)
    affinity_b, mask_b = inverse_align_affinity(affinity_b, mask_b, flip_dimension)
    valid = (mask_a & mask_b).expand_as(affinity_a)
    pointwise = F.smooth_l1_loss(
        affinity_a, affinity_b, beta=beta, reduction="none"
    )
    return pointwise.masked_select(valid).mean()


def affinity_equivariance_error(
    feature_a: torch.Tensor,
    feature_b: torch.Tensor,
    flip_dimension: int,
) -> torch.Tensor:
    """Diagnostic masked mean absolute local-affinity error."""

    affinity_a, mask_a = local_affinity_map(feature_a)
    affinity_b, mask_b = local_affinity_map(feature_b)
    affinity_b, mask_b = inverse_align_affinity(affinity_b, mask_b, flip_dimension)
    valid = (mask_a & mask_b).expand_as(affinity_a)
    return (affinity_a - affinity_b).abs().masked_select(valid).mean()


__all__ = [
    "OSMFV13Factorizer",
    "OSMF_STRUCTURAL_INTERVAL",
    "OSMF_LAMBDA_SEM",
    "OSMF_LAMBDA_STRUCT",
    "OSMF_LAMBDA_ORTH",
    "OSMF_LAMBDA_REC",
    "NEIGHBOR_OFFSETS",
    "local_affinity_map",
    "inverse_align_affinity",
    "structural_affinity_loss",
    "affinity_equivariance_error",
    "cross_subspace_covariance",
    "inverse_align_morphology",
    "orthogonality_loss",
    "reconstruction_cosine",
    "reconstruction_loss",
    "semantic_preservation_agreement",
    "semantic_preservation_loss",
]
