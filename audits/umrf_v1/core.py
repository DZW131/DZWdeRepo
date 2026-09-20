"""Pure helpers shared by the UMRF-v1 freeze and evaluation stages."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CHAINS = ("common", "sequential")
STAGES = ("5", "4", "3")
RULES = ("r1", "r2", "r3")
EXPECTED_CHECKPOINT_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax_class(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float32, copy=False) - values.max(0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(0, keepdims=True)


def rules_from_probabilities(p5: np.ndarray, p4: np.ndarray, p3: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the three prespecified GT-free rules to class-first probability maps."""
    h5, h4, h3 = (p.argmax(0).astype(np.uint8) for p in (p5, p4, p3))
    r1 = np.where(h4 == h3, h4, h5).astype(np.uint8)
    r2 = np.where(h5 == h4, h5, np.where(h5 == h3, h5, np.where(h4 == h3, h4, h5))).astype(np.uint8)
    r3 = ((p5 + p4 + p3) / 3.0).argmax(0).astype(np.uint8)
    return {"5": h5, "4": h4, "3": h3, "r1": r1, "r2": r2, "r3": r3}


def metric_block(h5: np.ndarray, h4: np.ndarray, h3: np.ndarray, truth: np.ndarray,
                 weights: np.ndarray | None = None) -> dict[str, float]:
    """Compute the frozen consensus metrics for aligned 1-D observations."""
    h5 = np.asarray(h5); h4 = np.asarray(h4); h3 = np.asarray(h3); truth = np.asarray(truth)
    w = np.ones(len(truth), np.float64) if weights is None else np.asarray(weights, np.float64)
    trigger = (h4 == h3) & (h4 != h5)
    h5_wrong, h5_right = h5 != truth, h5 == truth
    corrected = trigger & h5_wrong & (h4 == truth)
    harmed = trigger & h5_right & (h4 != truth)
    correct_trigger = trigger & (h4 == truth)
    total = float(w.sum())
    def mass(mask): return float(w[mask].sum())
    harm_mass = mass(harmed)
    return {
        "count_or_area": total,
        "trigger": mass(trigger),
        "trigger_rate": mass(trigger) / total if total else 0.0,
        "precision": mass(correct_trigger) / mass(trigger) if mass(trigger) else 0.0,
        "coverage": mass(corrected) / mass(h5_wrong) if mass(h5_wrong) else 0.0,
        "h5_correct_harm_rate": harm_mass / mass(h5_right) if mass(h5_right) else 0.0,
        "corrected": mass(corrected),
        "harmed": harm_mass,
        "nce": mass(corrected) / harm_mass if harm_mass else (float("inf") if mass(corrected) else 0.0),
        "h5_accuracy": mass(h5_right) / total if total else 0.0,
        "h4_given_h5_wrong_correct": mass(h5_wrong & (h4 == truth)) / mass(h5_wrong) if mass(h5_wrong) else 0.0,
        "h3_given_h5_wrong_correct": mass(h5_wrong & (h3 == truth)) / mass(h5_wrong) if mass(h5_wrong) else 0.0,
        "h54_agreement": mass(h5 == h4) / total if total else 0.0,
        "h43_agreement": mass(h4 == h3) / total if total else 0.0,
        "h53_agreement": mass(h5 == h3) / total if total else 0.0,
        "all_same": mass((h5 == h4) & (h4 == h3)) / total if total else 0.0,
    }


def decision(metrics: dict[str, float]) -> str:
    p, c, n, h = (metrics[k] for k in ("precision", "coverage", "nce", "h5_correct_harm_rate"))
    if p < .65 or c < .10 or n <= 1.0:
        return "NOGO"
    if p >= .85 and c >= .30 and n >= 2.0 and h < .05:
        return "STRONG_GO"
    if p >= .75 and c >= .20 and n >= 1.5 and h < .05:
        return "GO"
    return "WEAK"

