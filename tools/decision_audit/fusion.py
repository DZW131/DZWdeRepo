"""Official-metric fusion utilities and the frozen static simplex audit."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from tools.decision_audit import BRANCH_NAMES, OFFICIAL_FUSION


NUM_CLASSES = 4
METRIC_CLASSES = 5


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Apply the released per-image, per-class spatial min-max normalization."""
    cam = np.asarray(cam, dtype=np.float32)
    minimum = cam.min(axis=(-2, -1), keepdims=True)
    maximum = cam.max(axis=(-2, -1), keepdims=True)
    return (cam - minimum) / (maximum - minimum + 1e-8)


def prediction_from_scores(scores: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """Reproduce the released foreground mask conversion.

    The released implementation computes a background mask but does not append
    it to the tensor passed to argmax. Therefore its prediction contains only
    classes 0..3, while official metric code later overwrites GT-background
    pixels with label 4.
    """
    scores = np.asarray(scores, dtype=np.float32)
    presence = np.asarray(presence, dtype=np.float32)
    if scores.shape[-3] != NUM_CLASSES or presence.shape[-1] != NUM_CLASSES:
        raise ValueError("BCSS scores/presence must contain four classes")
    gated = scores * presence[..., :, None, None]
    return np.argmax(gated, axis=-3).astype(np.uint8)


def fast_hist(label_true: np.ndarray, label_pred: np.ndarray) -> np.ndarray:
    mask = (label_true >= 0) & (label_true < METRIC_CLASSES)
    return np.bincount(
        METRIC_CLASSES * label_true[mask].astype(np.int64)
        + label_pred[mask].astype(np.int64),
        minlength=METRIC_CLASSES**2,
    ).reshape(METRIC_CLASSES, METRIC_CLASSES)


def official_hist_update(
    histogram: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    prediction = np.array(prediction, copy=True, dtype=np.uint8)
    prediction[np.asarray(ground_truth) == 4] = 4
    histogram += fast_hist(np.asarray(ground_truth).reshape(-1), prediction.reshape(-1))


def official_score_from_hist(histogram: np.ndarray) -> dict:
    """Exact algebra used by ``tool.iouutils.scores`` after accumulation."""
    hist = np.asarray(histogram, dtype=np.float64).copy()
    hist[4, 4] = 0
    total = hist.sum()
    pixel_accuracy = np.diag(hist).sum() / total
    class_accuracy = np.diag(hist)[:4] / hist.sum(axis=1)[:4]
    mean_accuracy = np.nanmean(class_accuracy)
    denominator = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))[:4]
    iou = np.diag(hist)[:4] / denominator
    mean_iou = np.nanmean(iou)
    frequency = hist.sum(axis=1)[:4] / total
    frequency_weighted = (frequency[frequency > 0] * iou[frequency > 0]).sum()
    dice = {}
    for class_id in range(4):
        true_positive = np.diag(hist)[class_id]
        false_positive = hist[:, class_id].sum() - true_positive
        false_negative = hist[class_id, :].sum() - true_positive
        dice_denominator = 2 * true_positive + false_positive + false_negative
        dice[class_id] = (
            2 * true_positive / dice_denominator
            if dice_denominator > 0
            else 0.0
        )
    return {
        "Pixel Accuracy": float(pixel_accuracy),
        "Mean Accuracy": float(mean_accuracy),
        "Frequency Weighted IoU": float(frequency_weighted),
        "Mean IoU": float(mean_iou),
        "Class IoU": {class_id: float(value) for class_id, value in enumerate(iou)},
        "Dice Coefficients": dice,
        "Mean Dice": float(np.mean(list(dice.values()))),
    }


def score_predictions(ground_truth: np.ndarray, predictions: np.ndarray) -> dict:
    histogram = np.zeros((METRIC_CLASSES, METRIC_CLASSES), dtype=np.float64)
    for truth, prediction in zip(ground_truth, predictions):
        official_hist_update(histogram, truth, prediction)
    return official_score_from_hist(histogram)


def simplex_weights(step: float = 0.05) -> np.ndarray:
    """Return every nonnegative four-way simplex point on the frozen grid."""
    reciprocal = round(1.0 / step)
    if not math.isclose(reciprocal * step, 1.0, abs_tol=1e-12):
        raise ValueError("step must divide one exactly")
    weights = []
    for first in range(reciprocal + 1):
        for second in range(reciprocal - first + 1):
            for third in range(reciprocal - first - second + 1):
                fourth = reciprocal - first - second - third
                weights.append((first, second, third, fourth))
    return np.asarray(weights, dtype=np.float64) / reciprocal


def _cache_arrays(cache_dir: Path):
    return [
        np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in BRANCH_NAMES
    ]


def evaluate_static_grid(
    cache_dir: Path,
    weights: np.ndarray,
    device: str = "cuda",
    image_batch_size: int = 16,
    weight_batch_size: int = 64,
) -> list[dict]:
    """Evaluate a frozen weight grid with GPU-side streaming confusion matrices."""
    cache_dir = Path(cache_dir)
    cam_arrays = _cache_arrays(cache_dir)
    ground_truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    presence = np.load(cache_dir / "class_presence.npy", mmap_mode="r")
    official_predictions = np.load(
        cache_dir / "official_predictions.npy", mmap_mode="r"
    )
    sample_count = ground_truth.shape[0]
    weights = np.asarray(weights, dtype=np.float32)
    all_histograms = np.zeros((len(weights), 5, 5), dtype=np.int64)

    for weight_start in range(0, len(weights), weight_batch_size):
        weight_stop = min(weight_start + weight_batch_size, len(weights))
        weight_tensor = torch.from_numpy(weights[weight_start:weight_stop]).to(device)
        histogram = torch.zeros(
            (weight_stop - weight_start, 5, 5), dtype=torch.int64, device=device
        )
        for image_start in range(0, sample_count, image_batch_size):
            image_stop = min(image_start + image_batch_size, sample_count)
            cams_np = np.stack(
                [array[image_start:image_stop] for array in cam_arrays], axis=1
            )
            cams = torch.from_numpy(np.asarray(cams_np, dtype=np.float32)).to(device)
            fused = torch.einsum("qs,bschw->bqchw", weight_tensor, cams)
            presence_tensor = torch.from_numpy(
                np.asarray(presence[image_start:image_stop], dtype=np.float32)
            ).to(device)
            fused = fused * presence_tensor[:, None, :, None, None]
            prediction = fused.argmax(dim=2).to(torch.int64)
            truth = torch.from_numpy(
                np.asarray(ground_truth[image_start:image_stop], dtype=np.int64)
            ).to(device)
            truth_expanded = truth[:, None].expand_as(prediction)
            prediction = torch.where(
                truth_expanded == 4,
                torch.full_like(prediction, 4),
                prediction,
            )
            candidate_offsets = (
                torch.arange(weight_stop - weight_start, device=device)
                .view(1, -1, 1, 1)
                * 25
            )
            encoded = candidate_offsets + truth_expanded * 5 + prediction
            batch_histogram = torch.bincount(
                encoded.reshape(-1), minlength=(weight_stop - weight_start) * 25
            ).reshape(weight_stop - weight_start, 5, 5)
            histogram += batch_histogram
        all_histograms[weight_start:weight_stop] = histogram.cpu().numpy()

    rows = []
    official_index = np.flatnonzero(
        np.all(np.isclose(weights, np.asarray(OFFICIAL_FUSION)), axis=1)
    )
    if official_index.size != 1:
        raise RuntimeError("Frozen grid must contain official fusion exactly once")
    official_score = score_predictions(ground_truth, official_predictions)
    for candidate_index, (candidate, histogram) in enumerate(zip(weights, all_histograms)):
        score = (
            official_score
            if candidate_index == int(official_index[0])
            else official_score_from_hist(histogram)
        )
        rows.append(
            {
                "w56": float(candidate[0]),
                "w28_1": float(candidate[1]),
                "w28_2": float(candidate[2]),
                "wdeep": float(candidate[3]),
                "mIoU": 100 * score["Mean IoU"],
                "mDice": 100 * score["Mean Dice"],
                **{
                    f"class{class_id}_iou": 100 * score["Class IoU"][class_id]
                    for class_id in range(4)
                },
                "delta_vs_official": 100
                * (score["Mean IoU"] - official_score["Mean IoU"]),
            }
        )
    return rows


def official_weight_index(weights: np.ndarray) -> int:
    matching = np.flatnonzero(
        np.all(np.isclose(weights, np.asarray(OFFICIAL_FUSION)), axis=1)
    )
    if matching.size != 1:
        raise ValueError("Official weight vector is missing or duplicated")
    return int(matching[0])
