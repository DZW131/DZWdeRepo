"""Frozen OT-MTR seed selection and balanced optimal-transport loss."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tool.sinkhorn import marginal_errors, sinkhorn_plan


def epoch_alpha(epoch: int) -> float:
    if epoch < 1:
        raise ValueError("epoch is one-indexed")
    return min(1.0, 0.25 * (epoch - 1))


def ot_mtr_loss(
    features: torch.Tensor,
    aggregated_cam: torch.Tensor,
    mode_logits: torch.Tensor,
    mode_weights: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute detached-plan OT over top winner seeds of GT-present classes."""

    batch, channels, height, width = features.shape
    if aggregated_cam.shape != (batch, 4, height, width):
        raise ValueError("aggregated CAM shape mismatch")
    if mode_logits.shape != (batch, 4, 2, height, width):
        raise ValueError("mode-logit shape mismatch")
    if mode_weights.shape != (4, 2, channels):
        raise ValueError("mode-weight shape mismatch")

    present = labels > 0.5
    masked_cam = aggregated_cam.detach().float().masked_fill(
        ~present[:, :, None, None], -1.0e4
    )
    winner = masked_cam.argmax(dim=1).reshape(batch, -1)
    flat_scores = masked_cam.reshape(batch, 4, -1)
    flat_features = features.float().permute(0, 2, 3, 1).reshape(batch, -1, channels)

    pair_losses = []
    valid_by_class = torch.zeros(4, device=features.device, dtype=torch.float32)
    seed_sum_by_class = torch.zeros_like(valid_by_class)
    transport_mass_sum = torch.zeros(4, 2, device=features.device, dtype=torch.float32)
    max_row_error = torch.zeros((), device=features.device, dtype=torch.float32)
    max_col_error = torch.zeros_like(max_row_error)

    with torch.autocast(device_type=features.device.type, enabled=False):
        normalized_modes = F.normalize(mode_weights.float(), dim=-1)
        for sample_index in range(batch):
            for class_index in torch.where(present[sample_index])[0].tolist():
                candidates = torch.where(winner[sample_index] == class_index)[0]
                candidate_count = int(candidates.numel())
                if candidate_count < 4:
                    continue
                seeds_requested = min(128, max(4, math.ceil(0.30 * candidate_count)))
                candidate_scores = flat_scores[sample_index, class_index, candidates]
                selected = candidates[torch.topk(candidate_scores, seeds_requested).indices]
                seed_features = F.normalize(flat_features[sample_index, selected], dim=-1)
                similarity = seed_features @ normalized_modes[class_index].t()
                plan = sinkhorn_plan(1.0 - similarity, epsilon=0.1, iterations=20)
                row_error, col_error = marginal_errors(plan)
                max_row_error = torch.maximum(max_row_error, row_error)
                max_col_error = torch.maximum(max_col_error, col_error)
                pair_losses.append((plan * (1.0 - similarity)).sum())
                valid_by_class[class_index] += 1.0
                seed_sum_by_class[class_index] += float(seeds_requested)
                transport_mass_sum[class_index] += plan.sum(dim=0)

    loss = torch.stack(pair_losses).mean() if pair_losses else (
        features.sum() * 0.0 + mode_weights.sum() * 0.0
    )
    activation_ratio = (
        mode_logits.detach()[:, :, 0] > mode_logits.detach()[:, :, 1]
    ).float().mean(dim=(0, 2, 3))
    normalized_for_cosine = F.normalize(mode_weights.detach().float(), dim=-1)
    mode_cosine = (normalized_for_cosine[:, 0] * normalized_for_cosine[:, 1]).sum(dim=-1)
    valid_pairs = valid_by_class.sum()
    mean_seeds = seed_sum_by_class.sum() / valid_pairs.clamp_min(1.0)
    return {
        "loss": loss,
        "valid_pairs": valid_pairs.detach(),
        "valid_pairs_by_class": valid_by_class.detach(),
        "seed_sum_by_class": seed_sum_by_class.detach(),
        "mean_seeds": mean_seeds.detach(),
        "transport_mass_sum": transport_mass_sum.detach(),
        "mode_activation_ratio": activation_ratio.detach(),
        "mode_cosine": mode_cosine.detach(),
        "max_row_marginal_error": max_row_error.detach(),
        "max_col_marginal_error": max_col_error.detach(),
    }
