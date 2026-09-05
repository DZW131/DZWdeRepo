"""Tri-state CAM supervision, scaled circular locality and masked logit BCE."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


FULL25_STEPS = 29_275
IGNORE = -1


def radius_at_step(step: int, denominator: int = FULL25_STEPS) -> int:
    if denominator != FULL25_STEPS:
        raise ValueError("Phase-0 locality must use the frozen Full25 denominator")
    if not 0 <= step <= denominator:
        raise ValueError("Invalid optimizer step")
    progress = step / denominator
    if progress <= 0.10:
        return 1
    if progress >= 0.40:
        return 5
    return round(1 + 4 * (1 - math.cos(math.pi * (progress - 0.10) / 0.30)) / 2)


def normalize_cam(raw_cam: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=raw_cam.device.type, enabled=False):
        cam = F.relu(raw_cam.float())
        flat = cam.flatten(2)
        lower = flat.min(-1, keepdim=True).values[..., None]
        upper = flat.max(-1, keepdim=True).values[..., None]
        return (cam - lower) / (upper - lower).clamp_min(1.0e-6)


@torch.no_grad()
def tri_state_targets(normalized_cam: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Return int8 targets in {-1 ignore, 0 negative, 1 positive}."""
    cam = normalized_cam.float()
    batch, classes, height, width = cam.shape
    count = height * width
    take = int(math.ceil(0.15 * count))
    order = torch.argsort(cam.flatten(2), dim=-1, descending=True, stable=True)[..., :take]
    top = torch.zeros_like(cam, dtype=torch.bool).flatten(2)
    top.scatter_(2, order, True)
    top = top.reshape_as(cam)
    other_max = torch.empty_like(cam)
    for cls in range(classes):
        alternatives = [index for index in range(classes) if index != cls]
        other_max[:, cls] = cam[:, alternatives].max(dim=1).values
    positive = (cam >= 0.60) & top & ((cam - other_max) >= 0.10) & present[:, :, None, None].bool()
    any_positive = positive.any(dim=1, keepdim=True)
    positive_dilation = F.max_pool2d(any_positive.float(), 3, stride=1, padding=1).bool()
    reliable_background = (cam.max(dim=1, keepdim=True).values <= 0.10) & ~positive_dilation
    target = torch.full_like(cam, IGNORE, dtype=torch.int8)
    for cls in range(classes):
        rival = positive[:, [index for index in range(classes) if index != cls]].any(dim=1)
        active = present[:, cls, None, None].bool()
        negative = active & (rival | reliable_background[:, 0])
        target[:, cls][negative] = 0
        target[:, cls][positive[:, cls]] = 1
    return target, {
        "positive": positive,
        "reliable_background": reliable_background[:, 0],
        "any_positive_dilation": positive_dilation[:, 0],
    }


def circular_locality(query_grid: int, mask_hw: tuple[int, int], radius: int, device=None) -> torch.Tensor:
    """Map a query-grid radius into mask pixels while preserving spatial scale."""
    if query_grid != 14 or not 1 <= radius <= 5:
        raise ValueError("Frozen HQRF locality expects a 14x14 query grid and radius 1..5")
    height, width = mask_hw
    if height % query_grid or width % query_grid:
        raise ValueError("Mask resolution must be an integer multiple of query resolution")
    scale_y, scale_x = height / query_grid, width / query_grid
    qy, qx = torch.meshgrid(torch.arange(query_grid, device=device), torch.arange(query_grid, device=device), indexing="ij")
    py, px = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    center_y = (qy.flatten().float() + 0.5) * scale_y - 0.5
    center_x = (qx.flatten().float() + 0.5) * scale_x - 0.5
    distance = ((py[None] - center_y[:, None, None]) / scale_y).square()
    distance += ((px[None] - center_x[:, None, None]) / scale_x).square()
    return distance <= float(radius * radius)


def masked_query_bce(mask_logits: torch.Tensor, target: torch.Tensor, present: torch.Tensor,
                     p_class: torch.Tensor, joint: torch.Tensor, locality: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Sparse implementation of the frozen dense local masked BCE."""
    with torch.autocast(device_type=mask_logits.device.type, enabled=False):
        logits = mask_logits.float()
        resized = F.interpolate(target.float(), logits.shape[-2:], mode="nearest").to(torch.int8)
        assigned = p_class.argmax(dim=-1)
        query_index, pixel_index = locality.flatten(1).nonzero(as_tuple=True)
        flat_target = resized.flatten(2)
        batch_index = torch.arange(logits.shape[0], device=logits.device)[:, None]
        selected_target = flat_target[batch_index, assigned[:, query_index], pixel_index]
        selected_logits = logits.flatten(2)[:, query_index, pixel_index]
        assigned_present = present.gather(1, assigned)[:, query_index].bool()
        valid = (selected_target != IGNORE) & assigned_present
        valid_count = torch.zeros((logits.shape[0], logits.shape[1]), device=logits.device)
        valid_count.scatter_add_(1, query_index[None].expand(logits.shape[0], -1), valid.float())
        eligible = valid_count >= 4
        safe_target = selected_target.clamp_min(0).float()
        element = F.binary_cross_entropy_with_logits(selected_logits, safe_target, reduction="none") * valid
        loss_sum = torch.zeros_like(valid_count)
        loss_sum.scatter_add_(1, query_index[None].expand(logits.shape[0], -1), element)
        query_loss = loss_sum / valid_count.clamp_min(1)
        assigned_confidence = joint.gather(2, assigned[..., None]).squeeze(-1).detach()
        weights = assigned_confidence * eligible
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        image_valid = eligible.any(dim=1)
        per_image = (query_loss * weights).sum(dim=1)
        loss = per_image[image_valid].mean() if image_valid.any() else logits.sum() * 0
    return loss, {
        "assigned": assigned,
        "eligible": eligible,
        "valid_labeled_pixels": valid_count,
        "query_weights": weights,
        "valid_images": int(image_valid.sum()),
    }
