"""Safe image candidate oracle with official-fusion fallback candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.decision_audit.fusion import score_predictions
from tools.routing_signal_audit import SAFE_CANDIDATES
from tools.routing_signal_audit.metrics import score_row


def run_safe_image_oracle(
    cache_dir: Path,
    utilities: np.ndarray,
    output_prediction_path: Path,
) -> dict:
    cache_dir = Path(cache_dir)
    truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    branches = np.load(cache_dir / "branch_predictions.npy", mmap_mode="r")
    choices = np.argmax(utilities, axis=1).astype(np.uint8)
    predictions = np.lib.format.open_memmap(
        output_prediction_path, mode="w+", dtype=np.uint8, shape=truth.shape
    )
    predictions[:] = official
    for candidate_index in range(1, len(SAFE_CANDIDATES)):
        selected = np.flatnonzero(choices == candidate_index)
        predictions[selected] = branches[selected, candidate_index - 1]
    predictions.flush()
    official_score = score_predictions(truth, official)
    oracle_score = score_predictions(truth, predictions)
    summary = score_row("safe_image_candidate_oracle", oracle_score, official_score)
    summary.update(
        {
            f"choice_{candidate}": int(np.sum(choices == index))
            for index, candidate in enumerate(SAFE_CANDIDATES)
        }
    )
    image_rows = [
        {
            "index": index,
            "choice_index": int(choices[index]),
            "choice": SAFE_CANDIDATES[int(choices[index])],
            "official_q": float(utilities[index, 0]),
            "selected_q": float(utilities[index, choices[index]]),
            "local_gain": float(
                utilities[index, choices[index]] - utilities[index, 0]
            ),
        }
        for index in range(len(choices))
    ]
    return {
        "summary": summary,
        "image_rows": image_rows,
        "choices": choices,
        "predictions": predictions,
    }

