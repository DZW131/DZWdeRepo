"""Individual quality, complementarity, unique evidence, and error geometry."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from tools.decision_audit import BRANCH_NAMES
from tools.decision_audit.fusion import prediction_from_scores, score_predictions


def build_branch_predictions(cache_dir: Path) -> np.ndarray:
    cache_dir = Path(cache_dir)
    ground_truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    presence = np.load(cache_dir / "class_presence.npy", mmap_mode="r")
    cams = [
        np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in BRANCH_NAMES
    ]
    output_path = cache_dir / "branch_predictions.npy"
    predictions = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(ground_truth), len(BRANCH_NAMES), *ground_truth.shape[1:]),
    )
    for index in range(len(ground_truth)):
        for branch_index, cam in enumerate(cams):
            predictions[index, branch_index] = prediction_from_scores(
                cam[index], presence[index]
            )
    predictions.flush()
    return predictions


def individual_metrics(cache_dir: Path) -> list[dict]:
    cache_dir = Path(cache_dir)
    ground_truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    branch_predictions = np.load(
        cache_dir / "branch_predictions.npy", mmap_mode="r"
    )
    official_predictions = np.load(
        cache_dir / "official_predictions.npy", mmap_mode="r"
    )
    rows = []
    for branch_index, branch_name in enumerate(BRANCH_NAMES):
        score = score_predictions(ground_truth, branch_predictions[:, branch_index])
        rows.append(_score_row(branch_name, score))
    official_score = score_predictions(ground_truth, official_predictions)
    rows.append(_score_row("official_fusion", official_score))
    return rows


def _score_row(name: str, score: dict) -> dict:
    row = {
        "prediction": name,
        "mIoU": 100 * score["Mean IoU"],
        "mDice": 100 * score["Mean Dice"],
    }
    for class_id in range(4):
        row[f"class{class_id}_iou"] = 100 * score["Class IoU"][class_id]
        row[f"class{class_id}_dice"] = 100 * score["Dice Coefficients"][class_id]
    return row


def complementarity_tables(cache_dir: Path) -> dict[str, list[dict]]:
    cache_dir = Path(cache_dir)
    ground_truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    predictions = np.load(cache_dir / "branch_predictions.npy", mmap_mode="r")
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    foreground = ground_truth < 4
    total_foreground = int(foreground.sum())

    pairwise_rows = []
    for first, second in combinations(range(len(BRANCH_NAMES)), 2):
        for class_id in (-1, 0, 1, 2, 3):
            mask = foreground if class_id < 0 else ground_truth == class_id
            truth = ground_truth[mask]
            first_correct = predictions[:, first][mask] == truth
            second_correct = predictions[:, second][mask] == truth
            pairwise_rows.append(
                {
                    "branch_i": BRANCH_NAMES[first],
                    "branch_j": BRANCH_NAMES[second],
                    "class_id": class_id,
                    "pixels": int(mask.sum()),
                    "both_correct": int((first_correct & second_correct).sum()),
                    "both_wrong": int((~first_correct & ~second_correct).sum()),
                    "i_wrong_j_correct": int((~first_correct & second_correct).sum()),
                    "i_correct_j_wrong": int((first_correct & ~second_correct).sum()),
                }
            )

    correct = np.stack(
        [predictions[:, index] == ground_truth for index in range(len(BRANCH_NAMES))],
        axis=1,
    )
    unique_rows = []
    for branch_index, branch_name in enumerate(BRANCH_NAMES):
        other_correct = np.any(
            np.delete(correct, branch_index, axis=1), axis=1
        )
        unique = correct[:, branch_index] & ~other_correct & foreground
        unique_rows.append(
            {
                "branch": branch_name,
                "class_id": -1,
                "foreground_pixels": total_foreground,
                "unique_correct": int(unique.sum()),
                "unique_rate": float(unique.sum() / max(total_foreground, 1)),
            }
        )
        for class_id in range(4):
            class_mask = ground_truth == class_id
            class_unique = unique & class_mask
            class_pixels = int(class_mask.sum())
            unique_rows.append(
                {
                    "branch": branch_name,
                    "class_id": class_id,
                    "foreground_pixels": class_pixels,
                    "unique_correct": int(class_unique.sum()),
                    "unique_rate": float(class_unique.sum() / max(class_pixels, 1)),
                }
            )

    recoverability_rows = []
    official_correct = official == ground_truth
    for branch_index, branch_name in enumerate(BRANCH_NAMES):
        branch_correct = correct[:, branch_index]
        for class_id in (-1, 0, 1, 2, 3):
            mask = foreground if class_id < 0 else ground_truth == class_id
            recoverable = mask & ~official_correct & branch_correct
            harmful = mask & official_correct & ~branch_correct
            pixels = int(mask.sum())
            recoverability_rows.append(
                {
                    "branch": branch_name,
                    "class_id": class_id,
                    "foreground_pixels": pixels,
                    "recoverable": int(recoverable.sum()),
                    "harmful": int(harmful.sum()),
                    "net": int(recoverable.sum() - harmful.sum()),
                    "recoverable_rate": float(recoverable.sum() / max(pixels, 1)),
                    "harmful_rate": float(harmful.sum() / max(pixels, 1)),
                }
            )

    error_overlap_rows = []
    for class_id in (-1, 0, 1, 2, 3):
        mask = foreground if class_id < 0 else ground_truth == class_id
        truth = ground_truth[mask]
        errors = [predictions[:, index][mask] != truth for index in range(4)]
        for first in range(4):
            for second in range(4):
                intersection = int((errors[first] & errors[second]).sum())
                union = int((errors[first] | errors[second]).sum())
                error_overlap_rows.append(
                    {
                        "class_id": class_id,
                        "branch_i": BRANCH_NAMES[first],
                        "branch_j": BRANCH_NAMES[second],
                        "intersection": intersection,
                        "union": union,
                        "jaccard": float(intersection / union) if union else 1.0,
                    }
                )

    return {
        "pairwise": pairwise_rows,
        "unique": unique_rows,
        "recoverability": recoverability_rows,
        "error_overlap": error_overlap_rows,
    }


def class_preference(individual_rows: list[dict]) -> list[dict]:
    branch_rows = {
        row["prediction"]: row
        for row in individual_rows
        if row["prediction"] in BRANCH_NAMES
    }
    output = []
    for class_id in range(4):
        values = {
            name: branch_rows[name][f"class{class_id}_iou"]
            for name in BRANCH_NAMES
        }
        best = max(BRANCH_NAMES, key=lambda name: values[name])
        output.append(
            {
                "class_id": class_id,
                **values,
                "best_branch": best,
                "preference_gap_vs_cam28_1": values[best] - values["cam28_1"],
            }
        )
    return output
