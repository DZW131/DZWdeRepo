"""Deterministic Whole/Core/Rim construction and frozen token pooling."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tools.crra_v0 import (
    INTERNAL_AMBIGUITY_FRACTION,
    MIN_REGION_AREA,
    N_LABELS,
    PURITY_THRESHOLD,
)


CONNECTIVITY = 8


def slide_id_from_image(image_id: str) -> str:
    slide = image_id.split("_xmin", 1)[0]
    if not slide.startswith("TCGA-"):
        raise ValueError(f"Cannot parse BCSS slide id from {image_id}")
    return slide


def component_records(label_map: np.ndarray):
    records = []
    for predicted_class in range(4):
        count, components = cv2.connectedComponents(
            np.asarray(label_map == predicted_class, dtype=np.uint8),
            connectivity=CONNECTIVITY,
        )
        for component_id in range(1, count):
            indices = np.flatnonzero(components.reshape(-1) == component_id)
            records.append((predicted_class, component_id, indices))
    return records


def deterministic_top_fraction(indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    count = max(1, int(np.ceil(len(indices) * INTERNAL_AMBIGUITY_FRACTION)))
    order = np.lexsort((indices, -values))
    return indices[order[:count]]


def core_rim_indices(
    indices: np.ndarray,
    feature_pixels: torch.Tensor,
    whole_token: torch.Tensor,
    height: int,
    width: int,
):
    normalized_pixels = F.normalize(feature_pixels.float(), dim=0, eps=1e-8)
    normalized_token = F.normalize(whole_token.float(), dim=0, eps=1e-8)
    deviation = 1.0 - (normalized_pixels * normalized_token[:, None]).sum(dim=0)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask.reshape(-1)[indices] = 1
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    outer = np.flatnonzero((mask - eroded).reshape(-1) > 0)
    internal = deterministic_top_fraction(
        indices, deviation.detach().cpu().numpy().astype(np.float64)
    )
    rim = np.union1d(outer, internal).astype(np.int64)
    core = np.setdiff1d(indices, rim, assume_unique=True)
    return core, rim


def resize_gt_nearest(ground_truth: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(ground_truth, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=target_size, mode="nearest")
    return resized[0, 0].to(torch.int64).numpy().astype(np.uint8)


def _token(pixels: torch.Tensor) -> torch.Tensor:
    return pixels.float().mean(dim=1)


def _dispersion(pixels: torch.Tensor, token: torch.Tensor) -> float:
    if pixels.shape[1] == 0:
        return float("nan")
    normalized_pixels = F.normalize(pixels.float(), dim=0, eps=1e-8)
    normalized_token = F.normalize(token.float(), dim=0, eps=1e-8)
    return float((1.0 - (normalized_pixels * normalized_token[:, None]).sum(0)).mean().item())


def _cosine_discrepancy(left: torch.Tensor, right: torch.Tensor) -> float:
    cosine = F.cosine_similarity(left.float()[None], right.float()[None], dim=1, eps=1e-8)
    return float((1.0 - cosine[0]).item())


def extract_image_regions(
    proposal: np.ndarray,
    ground_truth: np.ndarray,
    feature: torch.Tensor,
    image_id: str,
    image_index: int,
):
    """Return metadata and aligned token arrays for all area>=2 proposals."""

    height, width = proposal.shape
    if tuple(feature.shape[-2:]) != (height, width):
        raise ValueError("Proposal and H28_1 spatial sizes differ")
    gt_grid = resize_gt_nearest(ground_truth, (height, width))
    if not set(np.unique(gt_grid).tolist()).issubset(set(range(N_LABELS))):
        raise ValueError("BCSS GT contains a label outside the released 0-4 mapping")
    feature_flat = feature.float().reshape(feature.shape[0], -1)
    slide_id = slide_id_from_image(image_id)
    rows, whole_tokens, core_tokens, rim_tokens = [], [], [], []
    raw_components = tiny_components = 0

    for predicted_class, component_id, indices_np in component_records(proposal):
        raw_components += 1
        if len(indices_np) < MIN_REGION_AREA:
            tiny_components += 1
            continue
        indices = torch.as_tensor(indices_np, dtype=torch.long)
        pixels = feature_flat.index_select(1, indices)
        whole = _token(pixels)
        core_np, rim_np = core_rim_indices(indices_np, pixels, whole, height, width)
        core_pixels = feature_flat.index_select(
            1, torch.as_tensor(core_np, dtype=torch.long)
        )
        rim_pixels = feature_flat.index_select(
            1, torch.as_tensor(rim_np, dtype=torch.long)
        )
        core = _token(core_pixels) if len(core_np) else torch.full_like(whole, float("nan"))
        rim = _token(rim_pixels) if len(rim_np) else torch.full_like(whole, float("nan"))

        counts = np.bincount(gt_grid.reshape(-1)[indices_np].astype(np.int64), minlength=N_LABELS)
        majority = int(np.argmax(counts))
        purity = float(counts[majority] / len(indices_np))
        if purity < PURITY_THRESHOLD:
            taxonomy = "Mixed"
        elif majority == predicted_class:
            taxonomy = "Type-A"
        else:
            taxonomy = "Type-B"
        common = len(core_np) >= 1 and len(rim_np) >= 1
        row = {
            "image_index": int(image_index),
            "image_id": image_id,
            "slide_id": slide_id,
            "region_id": f"{image_id}:{predicted_class}:{component_id}",
            "component_id": int(component_id),
            "base_predicted_class": int(predicted_class),
            "gt_majority_class": majority,
            "purity": purity,
            "taxonomy": taxonomy,
            "region_area": int(len(indices_np)),
            "core_area": int(len(core_np)),
            "rim_area": int(len(rim_np)),
            "empty_core": bool(len(core_np) == 0),
            "empty_rim": bool(len(rim_np) == 0),
            "common_support": bool(common),
            "whole_dispersion": _dispersion(pixels, whole),
            "core_dispersion": _dispersion(core_pixels, core),
            "rim_dispersion": _dispersion(rim_pixels, rim),
            "core_rim_discrepancy": (
                _cosine_discrepancy(core, rim) if common else float("nan")
            ),
        }
        for class_index in range(N_LABELS):
            row[f"gt_pixels_{class_index}"] = int(counts[class_index])
        rows.append(row)
        whole_tokens.append(whole.numpy().astype(np.float32))
        core_tokens.append(core.numpy().astype(np.float32))
        rim_tokens.append(rim.numpy().astype(np.float32))

    channels = int(feature.shape[0])
    empty = np.empty((0, channels), dtype=np.float32)
    arrays = {
        "z_whole": np.stack(whole_tokens) if whole_tokens else empty,
        "z_core": np.stack(core_tokens) if core_tokens else empty.copy(),
        "z_rim": np.stack(rim_tokens) if rim_tokens else empty.copy(),
    }
    return rows, arrays, {
        "raw_components": raw_components,
        "tiny_components": tiny_components,
        "proposed_regions": len(rows),
    }
