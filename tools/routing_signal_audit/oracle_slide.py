"""Slide-level safe candidate oracle."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.decision_audit.fusion import score_predictions
from tools.routing_signal_audit import SAFE_CANDIDATES
from tools.routing_signal_audit.metrics import score_row


def run_slide_oracle(
    cache_dir: Path,
    source_groups: list[str],
    safe_image_summary: dict,
    output_prediction_path: Path,
) -> dict:
    cache_dir = Path(cache_dir)
    groups = np.asarray(source_groups)
    truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    branches = np.load(cache_dir / "branch_predictions.npy", mmap_mode="r")
    predictions = np.lib.format.open_memmap(
        output_prediction_path, mode="w+", dtype=np.uint8, shape=truth.shape
    )
    predictions[:] = official
    rows = []
    choice_by_group = {}
    for group in dict.fromkeys(source_groups):
        indices = np.flatnonzero(groups == group)
        candidate_predictions = [
            official[indices],
            *(branches[indices, branch] for branch in range(4)),
        ]
        scores = [
            score_predictions(truth[indices], candidate)
            for candidate in candidate_predictions
        ]
        miou_values = np.asarray(
            [score["Mean IoU"] for score in scores], dtype=np.float64
        )
        choice = int(np.argmax(miou_values))
        choice_by_group[group] = choice
        if choice > 0:
            predictions[indices] = branches[indices, choice - 1]
        for candidate_index, (candidate, score) in enumerate(
            zip(SAFE_CANDIDATES, scores)
        ):
            rows.append(
                {
                    "source_group": group,
                    "patches": len(indices),
                    "candidate": candidate,
                    "candidate_index": candidate_index,
                    "mIoU": 100 * score["Mean IoU"],
                    "mDice": 100 * score["Mean Dice"],
                    "selected": candidate_index == choice,
                }
            )
    predictions.flush()
    official_score = score_predictions(truth, official)
    oracle_score = score_predictions(truth, predictions)
    summary = score_row("slide_safe_oracle", oracle_score, official_score)
    safe_delta = float(safe_image_summary["delta_mIoU"])
    summary["slide_recovery_ratio"] = (
        float(summary["delta_mIoU"] / safe_delta) if safe_delta != 0 else 0.0
    )
    summary["phenotype_flag"] = bool(
        summary["slide_recovery_ratio"] >= 0.70
        and summary["delta_mIoU"] >= 0.5
    )
    summary["selected_official_slides"] = sum(
        choice == 0 for choice in choice_by_group.values()
    )
    return {
        "summary": summary,
        "rows": rows,
        "choice_by_group": choice_by_group,
        "predictions": predictions,
    }
