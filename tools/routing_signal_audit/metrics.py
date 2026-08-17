"""Shared utility targets, routing diagnostics, and table helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from tools.decision_audit.fusion import score_predictions
from tools.routing_signal_audit import BRANCH_NAMES, SAFE_CANDIDATES


def write_json(path: Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as output:
        json.dump(json_ready(value), output, indent=2, sort_keys=True)
        output.write("\n")


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty audit table: {path}")
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def present_class_mean_iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    values = []
    for class_id in range(4):
        target = truth == class_id
        if not np.any(target):
            continue
        predicted = prediction == class_id
        union = np.count_nonzero(target | predicted)
        values.append(np.count_nonzero(target & predicted) / max(union, 1))
    return float(np.mean(values)) if values else 0.0


def candidate_utilities(
    truth: np.ndarray,
    official: np.ndarray,
    branches: np.ndarray,
) -> np.ndarray:
    predictions = [official, *(branches[index] for index in range(4))]
    return np.asarray(
        [present_class_mean_iou(prediction, truth) for prediction in predictions],
        dtype=np.float32,
    )


def score_row(method: str, score: dict, official_score: dict) -> dict:
    row = {
        "method": method,
        "mIoU": 100 * score["Mean IoU"],
        "mDice": 100 * score["Mean Dice"],
        "delta_mIoU": 100 * (score["Mean IoU"] - official_score["Mean IoU"]),
        "delta_mDice": 100 * (
            score["Mean Dice"] - official_score["Mean Dice"]
        ),
    }
    for class_id in range(4):
        row[f"class{class_id}_iou"] = 100 * score["Class IoU"][class_id]
        row[f"class{class_id}_dice"] = (
            100 * score["Dice Coefficients"][class_id]
        )
    return row


def build_utility_targets(cache_dir: Path, output_path: Path) -> np.ndarray:
    cache_dir = Path(cache_dir)
    truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    branches = np.load(cache_dir / "branch_predictions.npy", mmap_mode="r")
    utilities = np.zeros((len(truth), len(SAFE_CANDIDATES)), dtype=np.float32)
    for index in range(len(truth)):
        utilities[index] = candidate_utilities(
            truth[index], official[index], branches[index]
        )
    if not np.isfinite(utilities).all():
        raise RuntimeError("Image utility targets contain non-finite values")
    np.save(output_path, utilities)
    return utilities


def route_from_relative_predictions(predicted_relative: np.ndarray) -> np.ndarray:
    predicted_relative = np.asarray(predicted_relative)
    branch_choice = np.argmax(predicted_relative, axis=1).astype(np.int8)
    fallback = np.max(predicted_relative, axis=1) <= 0.0
    branch_choice[fallback] = -1
    return branch_choice


def build_router_predictions(
    cache_dir: Path,
    choices: np.ndarray,
    output_path: Path | None = None,
) -> np.ndarray:
    cache_dir = Path(cache_dir)
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    branches = np.load(cache_dir / "branch_predictions.npy", mmap_mode="r")
    if output_path is None:
        predictions = np.array(official, copy=True)
    else:
        predictions = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.uint8, shape=official.shape
        )
        predictions[:] = official
    for branch_index in range(4):
        selected = np.flatnonzero(choices == branch_index)
        predictions[selected] = branches[selected, branch_index]
    if hasattr(predictions, "flush"):
        predictions.flush()
    return predictions


def routing_diagnostics(
    predicted_relative: np.ndarray,
    true_relative: np.ndarray,
    choices: np.ndarray,
) -> dict:
    predicted_relative = np.asarray(predicted_relative, dtype=np.float64)
    true_relative = np.asarray(true_relative, dtype=np.float64)
    overridden = choices >= 0
    chosen_gain = np.zeros(len(choices), dtype=np.float64)
    selected_indices = np.flatnonzero(overridden)
    if len(selected_indices):
        chosen_gain[selected_indices] = true_relative[
            selected_indices, choices[selected_indices]
        ]
    correct_override = overridden & (chosen_gain > 0)
    harmful_override = overridden & (chosen_gain < 0)
    true_best = np.argmax(true_relative, axis=1)
    predicted_order = np.argsort(-predicted_relative, axis=1, kind="stable")
    top1 = np.mean(predicted_order[:, 0] == true_best)
    top2 = np.mean(
        np.any(predicted_order[:, :2] == true_best[:, None], axis=1)
    )
    pair_correct = []
    for first in range(4):
        for second in range(first + 1, 4):
            true_difference = true_relative[:, first] - true_relative[:, second]
            predicted_difference = (
                predicted_relative[:, first] - predicted_relative[:, second]
            )
            pair_correct.append(
                np.where(
                    true_difference == 0,
                    predicted_difference == 0,
                    np.sign(true_difference) == np.sign(predicted_difference),
                )
            )
    correlation = spearmanr(
        predicted_relative.reshape(-1), true_relative.reshape(-1)
    ).statistic
    return {
        "override_rate": float(np.mean(overridden)),
        "oracle_override_opportunity": float(
            np.mean(np.max(true_relative, axis=1) > 0)
        ),
        "override_precision": float(
            np.mean(correct_override[overridden]) if np.any(overridden) else 0.0
        ),
        "harmful_override_rate": float(np.mean(harmful_override)),
        "mean_positive_override_gain": float(
            np.mean(chosen_gain[correct_override]) if np.any(correct_override) else 0.0
        ),
        "mean_harmful_override_loss": float(
            np.mean(np.abs(chosen_gain[harmful_override]))
            if np.any(harmful_override)
            else 0.0
        ),
        "best_branch_top1_accuracy": float(top1),
        "best_branch_top2_accuracy": float(top2),
        "pairwise_ranking_accuracy": float(np.mean(np.stack(pair_correct))),
        "relative_utility_mae": float(
            np.mean(np.abs(predicted_relative - true_relative))
        ),
        "predicted_true_spearman": float(correlation),
    }


def evaluate_probe_predictions(
    name: str,
    predicted_relative: np.ndarray,
    utilities: np.ndarray,
    cache_dir: Path,
    fold_by_index: np.ndarray,
    primary_prediction_path: Path | None = None,
) -> dict:
    true_relative = utilities[:, 1:] - utilities[:, [0]]
    choices = route_from_relative_predictions(predicted_relative)
    router_predictions = build_router_predictions(
        cache_dir, choices, primary_prediction_path
    )
    cache_dir = Path(cache_dir)
    truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    official_predictions = np.load(
        cache_dir / "official_predictions.npy", mmap_mode="r"
    )
    official_score = score_predictions(truth, official_predictions)
    router_score = score_predictions(truth, router_predictions)
    aggregate = score_row(name, router_score, official_score)
    diagnostics = routing_diagnostics(
        predicted_relative, true_relative, choices
    )
    aggregate.update(diagnostics)
    fold_rows = []
    for fold in range(5):
        heldout = np.flatnonzero(fold_by_index == fold)
        fold_official = score_predictions(
            truth[heldout], official_predictions[heldout]
        )
        fold_router = score_predictions(truth[heldout], router_predictions[heldout])
        fold_diag = routing_diagnostics(
            predicted_relative[heldout], true_relative[heldout], choices[heldout]
        )
        fold_rows.append(
            {
                "probe": name,
                "fold": fold,
                "images": len(heldout),
                "official_mIoU": 100 * fold_official["Mean IoU"],
                "router_mIoU": 100 * fold_router["Mean IoU"],
                "delta_mIoU": 100
                * (fold_router["Mean IoU"] - fold_official["Mean IoU"]),
                "override_rate": fold_diag["override_rate"],
                "override_precision": fold_diag["override_precision"],
            }
        )
    return {
        "aggregate": aggregate,
        "fold_rows": fold_rows,
        "choices": choices,
        "predicted_relative": predicted_relative,
        "router_predictions": router_predictions,
    }


def signal_target_audit(
    signal_sets: dict[str, tuple[np.ndarray, list[str]]],
    utilities: np.ndarray,
    fold_by_index: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    true_q = np.asarray(utilities[:, 1:], dtype=np.float64)
    true_relative = true_q - np.asarray(utilities[:, [0]], dtype=np.float64)
    correlation_rows = []
    auroc_rows = []
    scopes = [("all", -1, np.arange(len(utilities)))]
    scopes.extend(
        ("fold", fold, np.flatnonzero(fold_by_index == fold))
        for fold in range(5)
    )
    for signal_set, (features, names) in signal_sets.items():
        if features.shape[:2] != true_relative.shape:
            raise RuntimeError(f"Signal Set {signal_set} candidate shape mismatch")
        for feature_index, feature_name in enumerate(names):
            for scope, scope_id, image_indices in scopes:
                values = features[image_indices, :, feature_index].reshape(-1)
                relative = true_relative[image_indices].reshape(-1)
                quality = true_q[image_indices].reshape(-1)
                relative_corr = spearmanr(values, relative).statistic
                quality_corr = spearmanr(values, quality).statistic
                correlation_rows.append(
                    {
                        "signal_set": signal_set,
                        "signal": feature_name,
                        "scope": scope,
                        "scope_id": scope_id,
                        "branch": "all",
                        "samples": len(values),
                        "spearman_relative_utility": float(
                            0.0 if np.isnan(relative_corr) else relative_corr
                        ),
                        "spearman_absolute_utility": float(
                            0.0 if np.isnan(quality_corr) else quality_corr
                        ),
                    }
                )
                labels = relative > 0
                auc = (
                    roc_auc_score(labels, values)
                    if np.unique(labels).size == 2 and np.std(values) > 0
                    else 0.5
                )
                auroc_rows.append(
                    {
                        "signal_set": signal_set,
                        "signal": feature_name,
                        "scope": scope,
                        "scope_id": scope_id,
                        "branch": "all",
                        "samples": len(values),
                        "positive_fraction": float(labels.mean()),
                        "auroc_relative_gain_positive": float(auc),
                    }
                )
            for branch_index, branch_name in enumerate(BRANCH_NAMES):
                values = features[:, branch_index, feature_index]
                relative = true_relative[:, branch_index]
                quality = true_q[:, branch_index]
                relative_corr = spearmanr(values, relative).statistic
                quality_corr = spearmanr(values, quality).statistic
                correlation_rows.append(
                    {
                        "signal_set": signal_set,
                        "signal": feature_name,
                        "scope": "branch",
                        "scope_id": branch_index,
                        "branch": branch_name,
                        "samples": len(values),
                        "spearman_relative_utility": float(
                            0.0 if np.isnan(relative_corr) else relative_corr
                        ),
                        "spearman_absolute_utility": float(
                            0.0 if np.isnan(quality_corr) else quality_corr
                        ),
                    }
                )
                labels = relative > 0
                auc = (
                    roc_auc_score(labels, values)
                    if np.unique(labels).size == 2 and np.std(values) > 0
                    else 0.5
                )
                auroc_rows.append(
                    {
                        "signal_set": signal_set,
                        "signal": feature_name,
                        "scope": "branch",
                        "scope_id": branch_index,
                        "branch": branch_name,
                        "samples": len(values),
                        "positive_fraction": float(labels.mean()),
                        "auroc_relative_gain_positive": float(auc),
                    }
                )
    return correlation_rows, auroc_rows


def frozen_primary_decision(
    delta_miou: float,
    recovery_ratio: float,
    positive_folds: int,
    bootstrap_lower: float,
) -> tuple[str, str]:
    if (
        delta_miou >= 0.50
        and recovery_ratio >= 0.25
        and positive_folds >= 4
        and bootstrap_lower > 0
    ):
        return (
            "ROUTING_SIGNAL_STRONG_GO",
            "MLP-C meets aggregate gain, oracle recovery, fold stability, and positive bootstrap-CI requirements.",
        )
    if delta_miou >= 0.30 and recovery_ratio >= 0.15 and positive_folds >= 3:
        return (
            "ROUTING_SIGNAL_GO",
            "MLP-C meets the preregistered gain, recovery, and fold requirements.",
        )
    if delta_miou >= 0.10:
        return (
            "ROUTING_SIGNAL_WEAK_REVIEW",
            "MLP-C shows some aggregate learnability but does not satisfy the full GO stability contract.",
        )
    return (
        "ROUTING_SIGNAL_NOGO",
        "MLP-C gains less than +0.10 mIoU under the frozen OOF protocol.",
    )
