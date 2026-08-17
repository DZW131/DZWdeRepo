"""Observation-only cross-stage calibration and confidence diagnostics."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from tools.decision_audit import BRANCH_NAMES


HISTOGRAM_BINS = 10000
CONFIDENCE_BINS = 10


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(shifted, dtype=np.float32)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    cumulative = np.cumsum(histogram)
    target = quantile * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return min(index, HISTOGRAM_BINS - 1) / HISTOGRAM_BINS


def calibration_audit(cache_dir: Path, image_batch_size: int = 16) -> dict:
    cache_dir = Path(cache_dir)
    ground_truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    predictions = np.load(cache_dir / "branch_predictions.npy", mmap_mode="r")
    foreground = ground_truth < 4
    foreground_count = int(foreground.sum())
    summary_rows = []
    confidence_rows = []

    for branch_index, branch_name in enumerate(BRANCH_NAMES):
        cams = np.load(cache_dir / f"{branch_name}.npy", mmap_mode="r")
        confidence_values = np.empty(foreground_count, dtype=np.float32)
        correct_values = np.empty(foreground_count, dtype=np.uint8)
        position = 0
        entropy_sum = 0.0
        confidence_sum = 0.0
        probability_pixels = 0
        activation_sum = np.zeros(4, dtype=np.float64)
        activation_count = np.zeros(4, dtype=np.int64)
        activation_histograms = np.zeros(
            (4, HISTOGRAM_BINS), dtype=np.int64
        )

        for start in range(0, len(ground_truth), image_batch_size):
            stop = min(start + image_batch_size, len(ground_truth))
            cam_batch = np.asarray(cams[start:stop], dtype=np.float32)
            probabilities = _softmax(cam_batch)
            entropy = -(
                probabilities * np.log(np.clip(probabilities, 1e-8, None))
            ).sum(axis=1)
            confidence = probabilities.max(axis=1)
            entropy_sum += float(entropy.sum(dtype=np.float64))
            confidence_sum += float(confidence.sum(dtype=np.float64))
            probability_pixels += entropy.size
            for class_id in range(4):
                values = cam_batch[:, class_id].reshape(-1)
                activation_sum[class_id] += values.sum(dtype=np.float64)
                activation_count[class_id] += values.size
                activation_histograms[class_id] += np.histogram(
                    values,
                    bins=HISTOGRAM_BINS,
                    range=(0.0, 1.0),
                )[0]
            foreground_batch = foreground[start:stop]
            count = int(foreground_batch.sum())
            confidence_values[position : position + count] = confidence[
                foreground_batch
            ]
            correct_values[position : position + count] = (
                predictions[start:stop, branch_index][foreground_batch]
                == ground_truth[start:stop][foreground_batch]
            )
            position += count
        if position != foreground_count:
            raise RuntimeError("Foreground confidence cache is incomplete")

        order = np.argsort(confidence_values, kind="stable")
        for bin_id, selected in enumerate(np.array_split(order, CONFIDENCE_BINS)):
            confidence_rows.append(
                {
                    "branch": branch_name,
                    "bin": bin_id,
                    "count": int(len(selected)),
                    "mean_confidence": float(confidence_values[selected].mean()),
                    "pixel_accuracy": float(correct_values[selected].mean()),
                    "minimum_confidence": float(confidence_values[selected].min()),
                    "maximum_confidence": float(confidence_values[selected].max()),
                }
            )
        for class_id in range(4):
            activation_mean = activation_sum[class_id] / activation_count[class_id]
            p95 = _histogram_quantile(
                activation_histograms[class_id], 0.95
            )
            summary_rows.append(
                {
                    "branch": branch_name,
                    "class_id": class_id,
                    "mean_entropy": entropy_sum / probability_pixels,
                    "mean_max_confidence": confidence_sum / probability_pixels,
                    "activation_mass": activation_mean,
                    "activation_p95": p95,
                    "peakiness": p95 / (activation_mean + 1e-8),
                    "foreground_coverage": float(
                        np.mean(predictions[:, branch_index] < 4)
                    ),
                }
            )
        del confidence_values, correct_values, order
        gc.collect()
    return {
        "summary_rows": summary_rows,
        "confidence_rows": confidence_rows,
    }

