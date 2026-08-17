"""Per-image frozen 0.1-simplex fusion oracle."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from tools.decision_audit.fusion import prediction_from_scores, score_predictions
from tools.routing_signal_audit import BRANCH_NAMES, OFFICIAL_FUSION
from tools.routing_signal_audit.metrics import score_row


def image_fusion_grid() -> np.ndarray:
    integer_weights = []
    for first in range(11):
        for second in range(11 - first):
            for third in range(11 - first - second):
                fourth = 10 - first - second - third
                integer_weights.append((first, second, third, fourth))
    official = np.asarray(OFFICIAL_FUSION) * 10
    integer_weights.sort(
        key=lambda value: (
            0 if np.array_equal(np.asarray(value), official) else 1,
            float(np.abs(np.asarray(value) - official).sum()),
            value,
        )
    )
    weights = np.asarray(integer_weights, dtype=np.float32) / 10.0
    if len(weights) != 286:
        raise RuntimeError(f"Frozen image-fusion grid must have 286 points, got {len(weights)}")
    if not np.array_equal(weights[0], np.asarray(OFFICIAL_FUSION, dtype=np.float32)):
        raise RuntimeError("Official fusion must be first in the frozen tie order")
    return weights


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


def run_image_fusion_oracle(
    cache_dir: Path,
    safe_image_summary: dict,
    output_prediction_path: Path,
    device: str = "cuda",
) -> dict:
    cache_dir = Path(cache_dir)
    weights = image_fusion_grid()
    cams = [np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in BRANCH_NAMES]
    truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    presence = np.load(cache_dir / "class_presence.npy", mmap_mode="r")
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    predictions = np.lib.format.open_memmap(
        output_prediction_path, mode="w+", dtype=np.uint8, shape=truth.shape
    )
    choices = np.zeros(len(truth), dtype=np.int16)
    local_q = np.zeros(len(truth), dtype=np.float32)
    weight_tensor = torch.from_numpy(weights).to(device)
    with torch.no_grad():
        for index in range(len(truth)):
            cam_tensor = torch.from_numpy(
                np.stack([cam[index] for cam in cams], axis=0)
            ).to(device)
            truth_tensor = torch.from_numpy(np.asarray(truth[index], dtype=np.int64)).to(device)
            presence_tensor = torch.from_numpy(
                np.asarray(presence[index], dtype=np.float32)
            ).to(device)
            best_q = -1.0
            best_index = 0
            for start in range(0, len(weights), 64):
                stop = min(start + 64, len(weights))
                scores = torch.einsum("qs,schw->qchw", weight_tensor[start:stop], cam_tensor)
                candidate_predictions = (scores * presence_tensor[None, :, None, None]).argmax(dim=1)
                utilities = _candidate_utilities_torch(candidate_predictions, truth_tensor)
                chunk_value, chunk_offset = torch.max(utilities, dim=0)
                value = float(chunk_value.item())
                candidate_index = start + int(chunk_offset.item())
                if value > best_q + 1e-12:
                    best_q = value
                    best_index = candidate_index
            choices[index] = best_index
            local_q[index] = best_q
            fused = sum(
                float(weight) * cam[index]
                for weight, cam in zip(weights[best_index], cams)
            )
            predictions[index] = prediction_from_scores(fused, presence[index])
    predictions.flush()
    official_score = score_predictions(truth, official)
    oracle_score = score_predictions(truth, predictions)
    summary = score_row("image_fusion_oracle", oracle_score, official_score)
    summary["grid_candidates"] = len(weights)
    summary["mean_local_q"] = float(local_q.mean())
    summary["median_local_q"] = float(np.median(local_q))
    summary["soft_gain_beyond_safe_hard"] = float(
        summary["delta_mIoU"] - safe_image_summary["delta_mIoU"]
    )
    if summary["soft_gain_beyond_safe_hard"] < 0.25:
        summary["mixture_flag"] = "HARD_SELECTION_FAVORED"
    elif summary["soft_gain_beyond_safe_hard"] >= 0.50:
        summary["mixture_flag"] = "SOFT_MIXTURE_FAVORED"
    else:
        summary["mixture_flag"] = "MIXTURE_FORM_REVIEW"
    rows = []
    for index, choice in enumerate(choices):
        row = {
            "index": index,
            "choice_index": int(choice),
            "local_q": float(local_q[index]),
        }
        row.update(
            {name: float(value) for name, value in zip(("w56", "w28_1", "w28_2", "wdeep"), weights[choice])}
        )
        rows.append(row)
    return {
        "summary": summary,
        "rows": rows,
        "choices": choices,
        "weights": weights,
        "predictions": predictions,
    }
