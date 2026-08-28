"""Frozen evaluation utilities for the RDDR Phase-1 utility gate."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


N_CLASS = 4
BACKGROUND = 4
EXPECTED_VAL = 3418
BCSS_THRESHOLDS = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
CAM_WEIGHTS = (0.6, 0.2, 0.2)
TTA_TRANSFORMS = (((), ()), ((3,), (2,)), ((2,), (1,)))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(item):
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return float(item)
        return item

    path.write_text(
        json.dumps(convert(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class SortedValidationDataset(Dataset):
    def __init__(self, root, image_size=224, max_images=0):
        root = Path(root)
        lowered = str(root).lower()
        if "test" in lowered or "luad" in lowered:
            raise AssertionError("RDDR Phase-1 analysis is BCSS validation-only")
        images = sorted((root / "img").glob("*.png"))
        masks = sorted((root / "mask").glob("*.png"))
        if [path.stem for path in images] != [path.stem for path in masks]:
            raise AssertionError("BCSS validation image/mask names differ")
        if max_images > 0:
            images = images[:max_images]
            masks = masks[:max_images]
        elif len(images) != EXPECTED_VAL:
            raise AssertionError(f"Expected {EXPECTED_VAL} validation images, got {len(images)}")
        self.images = images
        self.masks = masks
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        original = np.asarray(image, dtype=np.uint8).copy()
        truth = np.asarray(Image.open(self.masks[index]), dtype=np.uint8).copy()
        if image.size != (self.image_size, self.image_size):
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
            )
        tensor = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return image_path.stem, tensor, original, truth


def minmax_normalize(cam):
    cam = np.asarray(cam, dtype=np.float32)
    minimum = cam.min(axis=(1, 2), keepdims=True)
    maximum = cam.max(axis=(1, 2), keepdims=True)
    return (cam - minimum) / (maximum - minimum + 1.0e-8)


def presence_from_probability(probability):
    presence = (
        np.asarray(probability, dtype=np.float32) > BCSS_THRESHOLDS
    ).astype(np.float32)
    if presence.sum() == 0:
        presence[int(np.argmax(probability))] = 1.0
    return presence


def official_histogram(truth, prediction):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64).copy()
    prediction[truth == BACKGROUND] = BACKGROUND
    valid = (truth >= 0) & (truth <= BACKGROUND)
    encoded = 5 * truth[valid] + prediction[valid]
    return np.bincount(encoded, minlength=25).reshape(5, 5).astype(np.int64)


def scores_from_histogram(histogram):
    hist = np.asarray(histogram, dtype=np.float64).copy()
    hist[4, 4] = 0.0
    diagonal = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - diagonal
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = diagonal[:4] / union[:4]
    dice = []
    for index in range(4):
        tp = diagonal[index]
        fp = hist[:, index].sum() - tp
        fn = hist[index, :].sum() - tp
        denominator = 2 * tp + fp + fn
        dice.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return {
        "mIoU": float(np.nanmean(iou)),
        "mDice": float(np.mean(dice)),
        "class_iou": {str(index): float(iou[index]) for index in range(4)},
        "class_dice": {str(index): float(dice[index]) for index in range(4)},
        "histogram": hist.astype(np.int64).tolist(),
    }


def foreground_boundary_distance(truth):
    truth = np.asarray(truth, dtype=np.uint8)
    foreground = truth < BACKGROUND
    boundary = np.zeros_like(foreground, dtype=bool)
    height, width = truth.shape
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        left = truth[y0:y1, x0:x1]
        right = truth[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        transition = (left < BACKGROUND) & (right < BACKGROUND) & (left != right)
        boundary[y0:y1, x0:x1] |= transition
        boundary[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= transition
    distance = (
        ndimage.distance_transform_edt(~boundary)
        if boundary.any()
        else np.full(truth.shape, np.inf)
    )
    return {
        "boundary_le_7": foreground & (distance <= 7.0),
        "interior_gt_7": foreground & (distance > 7.0),
    }


def restricted_histogram(truth, prediction, mask):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    valid = np.asarray(mask, dtype=bool) & (truth >= 0) & (truth <= BACKGROUND)
    encoded = 5 * truth[valid] + prediction[valid]
    return np.bincount(encoded, minlength=25).reshape(5, 5).astype(np.int64)


class ZoneMetricAccumulator:
    def __init__(self):
        self.data = {
            name: {
                "pixels": 0,
                "correct": 0,
                "histogram": np.zeros((5, 5), dtype=np.int64),
            }
            for name in ("boundary_le_7", "interior_gt_7")
        }

    def update(self, truth, prediction):
        for name, mask in foreground_boundary_distance(truth).items():
            row = self.data[name]
            row["pixels"] += int(mask.sum())
            row["correct"] += int((mask & (prediction == truth)).sum())
            row["histogram"] += restricted_histogram(truth, prediction, mask)

    def result(self):
        output = {}
        for name, row in self.data.items():
            score = scores_from_histogram(row["histogram"])
            output[name] = {
                "pixels": int(row["pixels"]),
                "accuracy": row["correct"] / max(row["pixels"], 1),
                "restricted_mIoU": score["mIoU"],
                "class_iou": score["class_iou"],
            }
        return output


def component_thresholds(val_root):
    areas = {class_id: [] for class_id in range(4)}
    structure = np.ones((3, 3), dtype=np.uint8)
    for mask_path in sorted((Path(val_root) / "mask").glob("*.png")):
        truth = np.asarray(Image.open(mask_path), dtype=np.uint8)
        for class_id in range(4):
            labels, count = ndimage.label(truth == class_id, structure=structure)
            if count:
                areas[class_id].extend(np.bincount(labels.ravel())[1:].tolist())
    return {
        class_id: {
            "component_count": len(values),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
        for class_id, values in areas.items()
    }


class ComponentMetricAccumulator:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.structure = np.ones((3, 3), dtype=np.uint8)
        self.rows = {}
        self.histograms = {
            size: np.zeros((5, 5), dtype=np.int64)
            for size in ("small", "medium", "large")
        }

    def update(self, truth, prediction):
        size_masks = {
            size: np.zeros_like(truth, dtype=bool)
            for size in ("small", "medium", "large")
        }
        for class_id in range(4):
            labels, count = ndimage.label(truth == class_id, structure=self.structure)
            threshold = self.thresholds[class_id]
            for component_id in range(1, count + 1):
                mask = labels == component_id
                area = int(mask.sum())
                size = (
                    "small"
                    if area <= threshold["q25"]
                    else "medium"
                    if area <= threshold["q75"]
                    else "large"
                )
                row = self.rows.setdefault(
                    (class_id, size),
                    {"components": 0, "pixels": 0, "correct": 0},
                )
                row["components"] += 1
                row["pixels"] += area
                row["correct"] += int((prediction[mask] == class_id).sum())
                size_masks[size] |= mask
        for size, mask in size_masks.items():
            self.histograms[size] += restricted_histogram(truth, prediction, mask)

    def result(self):
        aggregate = {}
        for size in ("small", "medium", "large"):
            rows = [
                self.rows.get(
                    (class_id, size),
                    {"components": 0, "pixels": 0, "correct": 0},
                )
                for class_id in range(4)
            ]
            pixels = sum(row["pixels"] for row in rows)
            correct = sum(row["correct"] for row in rows)
            score = scores_from_histogram(self.histograms[size])
            aggregate[size] = {
                "components": int(sum(row["components"] for row in rows)),
                "pixels": int(pixels),
                "historical_component_recall": correct / max(pixels, 1),
                "diagnostic_size_restricted_mIoU": score["mIoU"],
            }
        return aggregate


def vectorized_miou(histograms):
    hist = np.asarray(histograms, dtype=np.float64).copy()
    hist[:, 4, 4] = 0.0
    diagonal = np.diagonal(hist, axis1=1, axis2=2)
    union = hist.sum(1) + hist.sum(2) - diagonal
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = diagonal[:, :4] / union[:, :4]
    return np.nanmean(iou, axis=1)


def paired_bootstrap_miou(base, candidate, resamples=10000, seed=42):
    base = np.asarray(base, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    if base.shape != candidate.shape or base.shape[1:] != (5, 5):
        raise ValueError("Paired bootstrap expects matching [N,5,5] histograms")
    observed = (
        scores_from_histogram(candidate.sum(0))["mIoU"]
        - scores_from_histogram(base.sum(0))["mIoU"]
    )
    generator = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    chunk_size = 64
    for start in range(0, resamples, chunk_size):
        count = min(chunk_size, resamples - start)
        indices = generator.integers(0, len(base), size=(count, len(base)))
        base_score = vectorized_miou(base[indices].sum(axis=1))
        candidate_score = vectorized_miou(candidate[indices].sum(axis=1))
        values[start:start + count] = candidate_score - base_score
    return {
        "observed_delta_mIoU": float(observed),
        "bootstrap_mean": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "resamples": int(resamples),
        "seed": int(seed),
    }, values
