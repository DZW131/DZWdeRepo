"""Frozen diagnostic image, image-class, and pixel oracle ceilings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.decision_audit import BRANCH_NAMES
from tools.decision_audit.fusion import prediction_from_scores, score_predictions


def _present_class_mean_iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    values = []
    for class_id in range(4):
        target = truth == class_id
        if not target.any():
            continue
        predicted = prediction == class_id
        union = np.count_nonzero(target | predicted)
        values.append(np.count_nonzero(target & predicted) / max(union, 1))
    return float(np.mean(values)) if values else 0.0


def _binary_class_iou(
    prediction: np.ndarray, truth: np.ndarray, class_id: int
) -> float:
    target = truth == class_id
    if not target.any():
        return 0.0
    predicted = prediction == class_id
    union = np.count_nonzero(target | predicted)
    return float(np.count_nonzero(target & predicted) / max(union, 1))


def run_oracles(cache_dir: Path) -> dict:
    cache_dir = Path(cache_dir)
    ground_truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    presence = np.load(cache_dir / "class_presence.npy", mmap_mode="r")
    branch_predictions = np.load(
        cache_dir / "branch_predictions.npy", mmap_mode="r"
    )
    official_predictions = np.load(
        cache_dir / "official_predictions.npy", mmap_mode="r"
    )
    cams = [
        np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in BRANCH_NAMES
    ]
    sample_count = len(ground_truth)
    image_choices = np.zeros(sample_count, dtype=np.uint8)
    image_class_choices = np.zeros((sample_count, 4), dtype=np.uint8)
    image_oracle = np.lib.format.open_memmap(
        cache_dir / "image_oracle_predictions.npy",
        mode="w+",
        dtype=np.uint8,
        shape=ground_truth.shape,
    )
    image_class_oracle = np.lib.format.open_memmap(
        cache_dir / "image_class_oracle_predictions.npy",
        mode="w+",
        dtype=np.uint8,
        shape=ground_truth.shape,
    )

    for index in range(sample_count):
        truth = ground_truth[index]
        image_scores = [
            _present_class_mean_iou(branch_predictions[index, branch], truth)
            for branch in range(4)
        ]
        image_choice = int(np.argmax(image_scores))
        image_choices[index] = image_choice
        image_oracle[index] = branch_predictions[index, image_choice]

        mixed_scores = np.zeros((4, *truth.shape), dtype=np.float32)
        for class_id in range(4):
            if not np.any(truth == class_id):
                choice = 0
            else:
                class_scores = [
                    _binary_class_iou(
                        branch_predictions[index, branch], truth, class_id
                    )
                    for branch in range(4)
                ]
                choice = int(np.argmax(class_scores))
            image_class_choices[index, class_id] = choice
            mixed_scores[class_id] = cams[choice][index, class_id]
        image_class_oracle[index] = prediction_from_scores(
            mixed_scores, presence[index]
        )
    image_oracle.flush()
    image_class_oracle.flush()

    foreground = ground_truth < 4
    branch_correct = np.stack(
        [
            branch_predictions[:, branch] == ground_truth
            for branch in range(len(BRANCH_NAMES))
        ],
        axis=1,
    )
    any_branch_correct = np.any(branch_correct, axis=1) & foreground
    pixel_oracle = np.array(official_predictions, copy=True)
    pixel_oracle[any_branch_correct] = ground_truth[any_branch_correct]
    np.save(cache_dir / "pixel_oracle_predictions.npy", pixel_oracle)

    official_score = score_predictions(ground_truth, official_predictions)
    oracle_scores = {
        "official_fusion": official_score,
        "image_oracle": score_predictions(ground_truth, image_oracle),
        "image_class_oracle": score_predictions(ground_truth, image_class_oracle),
        "pixel_oracle": score_predictions(ground_truth, pixel_oracle),
    }
    rows = []
    for name, score in oracle_scores.items():
        row = {
            "method": name,
            "mIoU": 100 * score["Mean IoU"],
            "mDice": 100 * score["Mean Dice"],
            "delta_vs_official": 100
            * (score["Mean IoU"] - official_score["Mean IoU"]),
        }
        row.update(
            {
                f"class{class_id}_iou": 100 * score["Class IoU"][class_id]
                for class_id in range(4)
            }
        )
        rows.append(row)

    foreground_pixels = int(foreground.sum())
    pixel_summary = {
        "foreground_pixels": foreground_pixels,
        "recoverable_pixels": int(any_branch_correct.sum()),
        "coverage": float(any_branch_correct.sum() / max(foreground_pixels, 1)),
        "unrecoverable_pixels": int((foreground & ~any_branch_correct).sum()),
        "unrecoverable_rate": float(
            (foreground & ~any_branch_correct).sum() / max(foreground_pixels, 1)
        ),
    }

    preference_rows = []
    for branch_index, branch_name in enumerate(BRANCH_NAMES):
        count = int((image_choices == branch_index).sum())
        preference_rows.append(
            {
                "level": "image",
                "class_id": -1,
                "branch": branch_name,
                "count": count,
                "fraction": count / sample_count,
            }
        )
    for class_id in range(4):
        present = np.asarray(
            [np.any(truth == class_id) for truth in ground_truth], dtype=bool
        )
        for branch_index, branch_name in enumerate(BRANCH_NAMES):
            count = int(
                ((image_class_choices[:, class_id] == branch_index) & present).sum()
            )
            preference_rows.append(
                {
                    "level": "image_class",
                    "class_id": class_id,
                    "branch": branch_name,
                    "count": count,
                    "fraction": count / max(int(present.sum()), 1),
                }
            )

    np.save(cache_dir / "image_oracle_choices.npy", image_choices)
    np.save(cache_dir / "image_class_oracle_choices.npy", image_class_choices)
    return {
        "rows": rows,
        "pixel_summary": pixel_summary,
        "preference_rows": preference_rows,
    }

