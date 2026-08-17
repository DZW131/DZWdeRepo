"""Paired grouped bootstrap over the 22 source-slide units."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.decision_audit.fusion import (
    fast_hist,
    official_score_from_hist,
)
from tools.routing_signal_audit import BOOTSTRAP_REPLICATES, FOLD_SEED


def _slide_histograms(
    truth: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    unique_groups: list[str],
) -> np.ndarray:
    histograms = np.zeros((len(unique_groups), 5, 5), dtype=np.int64)
    group_index = {group: index for index, group in enumerate(unique_groups)}
    for image_index, group in enumerate(groups):
        prediction = np.array(predictions[image_index], copy=True, dtype=np.uint8)
        target = np.asarray(truth[image_index], dtype=np.uint8)
        prediction[target == 4] = 4
        histograms[group_index[group]] += fast_hist(target.reshape(-1), prediction.reshape(-1))
    return histograms


def grouped_slide_bootstrap(
    cache_dir: Path,
    router_prediction_path: Path,
    source_groups: list[str],
) -> dict:
    cache_dir = Path(cache_dir)
    truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    router = np.load(router_prediction_path, mmap_mode="r")
    groups = np.asarray(source_groups)
    unique_groups = list(dict.fromkeys(source_groups))
    official_histograms = _slide_histograms(
        truth, official, groups, unique_groups
    )
    router_histograms = _slide_histograms(truth, router, groups, unique_groups)
    rng = np.random.default_rng(FOLD_SEED)
    rows = []
    deltas = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        official_score = official_score_from_hist(official_histograms[sampled].sum(axis=0))
        router_score = official_score_from_hist(router_histograms[sampled].sum(axis=0))
        delta = 100 * (router_score["Mean IoU"] - official_score["Mean IoU"])
        deltas[replicate] = delta
        rows.append(
            {
                "replicate": replicate,
                "sampling_unit": "source_slide",
                "sampled_slides": len(sampled),
                "delta_mIoU": float(delta),
            }
        )
    summary = {
        "replicates": BOOTSTRAP_REPLICATES,
        "sampling_unit": "source_slide",
        "num_source_slides": len(unique_groups),
        "seed": FOLD_SEED,
        "mean_delta_mIoU": float(np.mean(deltas)),
        "ci_2_5": float(np.quantile(deltas, 0.025)),
        "median_delta_mIoU": float(np.quantile(deltas, 0.50)),
        "ci_97_5": float(np.quantile(deltas, 0.975)),
    }
    return {"rows": rows, "summary": summary}
