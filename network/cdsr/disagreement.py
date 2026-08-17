"""Frozen analytical rectification-need signal used by CDSR."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-8


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Return class entropy normalized by log(C), with shape [B, 1, H, W]."""
    class_count = probabilities.shape[1]
    entropy = -(
        probabilities * probabilities.clamp_min(EPSILON).log()
    ).sum(dim=1, keepdim=True)
    return (entropy / math.log(class_count)).clamp(0.0, 1.0)


def normalized_jsd(
    first_probabilities: torch.Tensor,
    second_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return Jensen-Shannon divergence / log(2), shape [B, 1, H, W]."""
    mixture = 0.5 * (first_probabilities + second_probabilities)
    first_kl = (
        first_probabilities
        * (
            first_probabilities.clamp_min(EPSILON).log()
            - mixture.clamp_min(EPSILON).log()
        )
    ).sum(dim=1, keepdim=True)
    second_kl = (
        second_probabilities
        * (
            second_probabilities.clamp_min(EPSILON).log()
            - mixture.clamp_min(EPSILON).log()
        )
    ).sum(dim=1, keepdim=True)
    return (0.5 * (first_kl + second_kl) / math.log(2.0)).clamp(
        0.0, 1.0
    )


class AnalyticalRectificationNeed(nn.Module):
    """Compute the frozen, detached CDSR need map without trainable state."""

    def forward(
        self,
        raw_cam_stage: torch.Tensor,
        raw_cam_deep: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        stage_probabilities = torch.softmax(
            raw_cam_stage.detach().float(), dim=1
        )
        deep_probabilities = torch.softmax(
            raw_cam_deep.detach().float(), dim=1
        )
        deep_probabilities = F.interpolate(
            deep_probabilities,
            size=stage_probabilities.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        disagreement = normalized_jsd(
            stage_probabilities, deep_probabilities
        )
        stage_uncertainty = normalized_entropy(stage_probabilities)
        deep_reliability = (
            1.0 - normalized_entropy(deep_probabilities)
        ).clamp(0.0, 1.0)
        ambiguity = 1.0 - (1.0 - disagreement) * (
            1.0 - stage_uncertainty
        )
        need_map = (deep_reliability * ambiguity).clamp(0.0, 1.0)

        return {
            "disagreement": disagreement,
            "stage_uncertainty": stage_uncertainty,
            "deep_reliability": deep_reliability,
            "need_map": need_map,
        }
