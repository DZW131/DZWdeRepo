"""Exact local Image x Class oracle over 5^4 frozen combinations."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from tools.decision_audit.fusion import prediction_from_scores, score_predictions
from tools.routing_signal_audit import BRANCH_NAMES, OFFICIAL_FUSION, SAFE_CANDIDATES
from tools.routing_signal_audit.metrics import present_class_mean_iou, score_row


def local_class_combinations() -> np.ndarray:
    combinations = list(itertools.product(range(5), repeat=4))
    combinations.sort(
        key=lambda value: (
            0 if value == (0, 0, 0, 0) else 1,
            sum(item != 0 for item in value),
            value,
        )
    )
    result = np.asarray(combinations, dtype=np.int8)
    if result.shape != (625, 4) or not np.all(result[0] == 0):
        raise RuntimeError("Exact local Image x Class enumeration contract failed")
    return result


def _candidate_utilities_torch(predictions: torch.Tensor, truth: torch.Tensor):
    values = torch.zeros(predictions.shape[0], dtype=torch.float64, device=predictions.device)
    present_count = 0
    for class_id in range(4):
        target = truth == class_id
        if not bool(target.any()):
            continue
        predicted = predictions == class_id
        intersection = (predicted & target[None]).sum(dim=(1, 2)).to(torch.float64)
        union = (predicted | target[None]).sum(dim=(1, 2)).clamp_min(1).to(torch.float64)
        values += intersection / union
        present_count += 1
    return values / max(present_count, 1)


def run_exact_local_imageclass_oracle(
    cache_dir: Path,
    safe_image_summary: dict,
    output_prediction_path: Path,
    device: str = "cuda",
) -> dict:
    cache_dir = Path(cache_dir)
    combinations = local_class_combinations()
    cams = [np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in BRANCH_NAMES]
    truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    presence = np.load(cache_dir / "class_presence.npy", mmap_mode="r")
    official_predictions = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    predictions = np.lib.format.open_memmap(
        output_prediction_path, mode="w+", dtype=np.uint8, shape=truth.shape
    )
    choices = np.zeros((len(truth), 4), dtype=np.int8)
    best_local_q = np.zeros(len(truth), dtype=np.float32)
    official_local_q = np.zeros(len(truth), dtype=np.float32)
    combo_tensor = torch.from_numpy(combinations.astype(np.int64)).to(device)
    with torch.no_grad():
        for index in range(len(truth)):
            official_scores = sum(
                weight * cam[index] for weight, cam in zip(OFFICIAL_FUSION, cams)
            )
            candidates = np.stack(
                [official_scores, *(cam[index] for cam in cams)], axis=0
            ).astype(np.float32, copy=False)
            candidate_tensor = torch.from_numpy(candidates).to(device)
            truth_tensor = torch.from_numpy(np.asarray(truth[index], dtype=np.int64)).to(device)
            presence_tensor = torch.from_numpy(np.asarray(presence[index], dtype=np.float32)).to(device)
            best_q = -1.0
            best_index = 0
            for start in range(0, len(combinations), 64):
                stop = min(start + 64, len(combinations))
                chunk = combo_tensor[start:stop]
                scores = torch.stack(
                    [candidate_tensor[chunk[:, class_id], class_id] for class_id in range(4)],
                    dim=1,
                )
                candidate_predictions = (scores * presence_tensor[None, :, None, None]).argmax(dim=1)
                utilities = _candidate_utilities_torch(candidate_predictions, truth_tensor)
                chunk_value, chunk_offset = torch.max(utilities, dim=0)
                value = float(chunk_value.item())
                candidate_index = start + int(chunk_offset.item())
                if value > best_q + 1e-12:
                    best_q = value
                    best_index = candidate_index
            selected = combinations[best_index]
            mixed_scores = np.stack(
                [candidates[selected[class_id], class_id] for class_id in range(4)],
                axis=0,
            )
            prediction = prediction_from_scores(mixed_scores, presence[index])
            predictions[index] = prediction
            choices[index] = selected
            best_local_q[index] = best_q
            official_local_q[index] = present_class_mean_iou(
                official_predictions[index], truth[index]
            )
    predictions.flush()
    official_score = score_predictions(truth, official_predictions)
    oracle_score = score_predictions(truth, predictions)
    summary = score_row("exact_local_imageclass_oracle", oracle_score, official_score)
    summary["enumerated_combinations"] = len(combinations)
    summary["mean_local_q"] = float(best_local_q.mean())
    summary["median_local_q"] = float(np.median(best_local_q))
    summary["official_mean_local_q"] = float(official_local_q.mean())
    summary["official_median_local_q"] = float(np.median(official_local_q))
    summary["aggregate_gain_beyond_safe_image"] = float(
        summary["delta_mIoU"] - safe_image_summary["delta_mIoU"]
    )
    mean_local_gain = float((best_local_q - official_local_q).mean())
    summary["mean_local_q_gain"] = mean_local_gain
    if summary["aggregate_gain_beyond_safe_image"] >= 0.5 and mean_local_gain > 0:
        summary["class_conditional_flag"] = "CLASS_CONDITIONAL_SIGNAL"
    elif summary["aggregate_gain_beyond_safe_image"] < 0.2:
        summary["class_conditional_flag"] = "CLASS_CONDITIONAL_LOW_PRIORITY"
    else:
        summary["class_conditional_flag"] = "CLASS_CONDITIONAL_REVIEW"
    rows = []
    for index in range(len(truth)):
        row = {
            "index": index,
            "official_local_q": float(official_local_q[index]),
            "exact_local_q": float(best_local_q[index]),
            "local_q_gain": float(best_local_q[index] - official_local_q[index]),
        }
        for class_id in range(4):
            choice = int(choices[index, class_id])
            row[f"class{class_id}_choice_index"] = choice
            row[f"class{class_id}_choice"] = SAFE_CANDIDATES[choice]
        rows.append(row)
    return {
        "summary": summary,
        "rows": rows,
        "choices": choices,
        "predictions": predictions,
    }
