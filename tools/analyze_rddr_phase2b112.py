#!/usr/bin/env python3
"""Offline, numpy-only Phase2B1.12 audit of immutable paired snapshots.

No model is imported, updated, or selected by this module. All percentages in
CSV/JSON are fractions; columns ending ``_pp`` are percentage-point changes.
The official evaluator semantics are copied from tool/iouutils.py: foreground
IoU uses nanmean, absent-class Dice is zero, and background [4,4] is discarded.
Bootstrap units are images, but estimands are pooled pixels/confusion matrices,
never means of per-image IoU. RandomState(42) is reset for every paired endpoint
so every comparison uses the same resampled image counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PREFIX = "rddr_phase2b112_"
STEPS = (0, 50, 100, 250, 500)
ARMS = ("B", "A", "R")
COMPARISONS = (("A", "B"), ("A", "R"), ("R", "B"))
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
Q_EDGES = np.array([0.020935675129294395, 0.072734534740448,
                    0.163648784160614, 0.3369627296924591])
POP_METRICS = ("raw_accuracy", "deep_accuracy", "rect_accuracy",
               "raw_gt_probability", "deep_gt_probability",
               "rect_gt_probability", "raw_gt_margin", "deep_gt_margin")
CHECKPOINT_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
BASE_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"


def _divide(numerator: np.ndarray | float, denominator: np.ndarray | float,
            fill: float = np.nan) -> np.ndarray:
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator,
                     out=np.full(numerator.shape, fill, dtype=np.float64),
                     where=denominator != 0)


def _nanmean(values: np.ndarray, axis: int = -1) -> np.ndarray:
    valid = np.isfinite(values)
    return _divide(np.where(valid, values, 0).sum(axis=axis), valid.sum(axis=axis))


def confusion_metrics(cm: np.ndarray, official: bool = True) -> dict[str, np.ndarray]:
    """Compute pooled/batched metrics, with exact official absent-class rules."""
    cm = np.array(cm, dtype=np.float64, copy=True)
    size = 5 if official else 4
    if cm.shape[-2:] != (size, size):
        raise ValueError(f"Expected (...,{size},{size}) confusion matrix: {cm.shape}")
    if np.any(cm < 0) or not np.isfinite(cm).all():
        raise ValueError("Confusion matrices must be finite and nonnegative")
    if official:
        cm[..., 4, 4] = 0
    tp = np.diagonal(cm, axis1=-2, axis2=-1)[..., :4]
    rows, cols = cm.sum(axis=-1)[..., :4], cm.sum(axis=-2)[..., :4]
    iou = _divide(tp, rows + cols - tp)
    dice = _divide(2 * tp, rows + cols, fill=0.0)
    return {"accuracy": _divide(tp.sum(axis=-1), cm.sum(axis=(-2, -1))),
            "miou": _nanmean(iou), "mdice": dice.mean(axis=-1),
            "iou": iou, "dice": dice}


def _bootstrap_counts(n_images: int, replicates: int = BOOTSTRAP_REPLICATES,
                      seed: int = BOOTSTRAP_SEED, chunk_size: int = 64):
    """Bounded-memory paired image bootstrap; sampling does not use pixel GT."""
    if n_images < 1 or replicates < 1:
        raise ValueError("Bootstrap needs images and positive replicates")
    rng = np.random.RandomState(seed)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        indices = rng.randint(0, n_images, size=(size, n_images))
        flat = indices + np.arange(size, dtype=np.int64)[:, None] * n_images
        counts = np.bincount(flat.ravel(), minlength=size * n_images)
        yield counts.reshape(size, n_images).astype(np.float64)


def bootstrap_confusions(cm_left: np.ndarray, cm_right: np.ndarray,
                         replicates: int = BOOTSTRAP_REPLICATES,
                         seed: int = BOOTSTRAP_SEED) -> dict[str, dict[str, Any]]:
    """Paired percentile CIs of dataset-level official mIoU and mDice."""
    left, right = np.asarray(cm_left), np.asarray(cm_right)
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (5, 5):
        raise ValueError("Paired official confusion arrays must both be N x 5 x 5")
    point_l, point_r = confusion_metrics(left.sum(axis=0)), confusion_metrics(right.sum(axis=0))
    samples = {"miou": [], "mdice": []}
    flat_l, flat_r = left.reshape(len(left), -1), right.reshape(len(right), -1)
    for counts in _bootstrap_counts(len(left), replicates, seed):
        ml = confusion_metrics((counts @ flat_l).reshape(-1, 5, 5))
        mr = confusion_metrics((counts @ flat_r).reshape(-1, 5, 5))
        for key in samples:
            samples[key].append(ml[key] - mr[key])
    return {key: _interval(float(point_l[key] - point_r[key]), np.concatenate(vals), replicates, seed)
            for key, vals in samples.items()}


def _interval(point: float, samples: np.ndarray, requested: int, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    valid = np.asarray(samples)[np.isfinite(samples)]
    low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
    return {"delta": point, "ci_low": float(low), "ci_high": float(high),
            "bootstrap_replicates": requested, "valid_replicates": len(valid),
            "bootstrap_seed": seed, "ci_method": "paired_image_percentile"}


def bootstrap_ratios(left_sums: np.ndarray, right_sums: np.ndarray,
                     counts_per_image: np.ndarray,
                     replicates: int = BOOTSTRAP_REPLICATES,
                     seed: int = BOOTSTRAP_SEED) -> list[dict[str, Any]]:
    """Joint bootstrap of pooled frozen-population means (shared denominators)."""
    left, right = np.asarray(left_sums, dtype=np.float64), np.asarray(right_sums, dtype=np.float64)
    denominator = np.asarray(counts_per_image, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or denominator.shape != left.shape:
        raise ValueError("Ratio arrays must share N x endpoints shape")
    if np.any(denominator < 0):
        raise ValueError("Population counts cannot be negative")
    point = _divide((left - right).sum(axis=0), denominator.sum(axis=0))
    samples = []
    for counts in _bootstrap_counts(len(left), replicates, seed):
        samples.append(_divide(counts @ (left - right), counts @ denominator))
    sample = np.concatenate(samples)
    return [_interval(float(point[i]), sample[:, i], replicates, seed) for i in range(left.shape[1])]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _json_safe(value) for key, value in row.items()} for row in rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "PENDING", "missing_file": path.name}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_snapshot(path: Path) -> dict[str, np.ndarray]:
    required = ("names", "truth", "ps", "pd", "rect", "raw_logits", "deep_logits",
                "q", "delta", "official_cm", "boundary", "top20")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"{path.name}: missing snapshot arrays {sorted(missing)}")
        data = {key: archive[key] for key in required}
    n = len(data["names"])
    if n < 1 or data["names"].ndim != 1:
        raise ValueError(f"{path.name}: invalid names")
    names = data["names"].astype(str)
    if len(set(names)) != n or names.tolist() != sorted(names.tolist()):
        raise ValueError(f"{path.name}: names must be unique and sorted")
    shapes = {"truth": (n, 784), "q": (n, 784), "delta": (n, 784),
              "boundary": (n, 784), "top20": (n, 784), "official_cm": (n, 5, 5)}
    shapes.update({key: (n, 4, 784) for key in ("ps", "pd", "rect", "raw_logits", "deep_logits")})
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"{path.name}: invalid shape/nonfinite {key}: {data[key].shape}")
    if not np.isin(data["truth"], (0, 1, 2, 3, 4, 255)).all():
        raise ValueError(f"{path.name}: unexpected truth labels")
    for key in ("ps", "pd", "rect"):
        values = data[key]
        if np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(axis=1), 1, atol=2e-5):
            raise ValueError(f"{path.name}: {key} is not normalized 4-class probability")
    if np.any(data["q"] < -1e-6) or np.any(data["q"] > 1 + 1e-6):
        raise ValueError(f"{path.name}: q outside normalized JS range")
    cm = data["official_cm"]
    if np.any(cm < 0) or np.any(cm[:, 4, :4] != 0):
        raise ValueError(f"{path.name}: official background handling violated")
    for key in ("boundary", "top20"):
        if not np.isin(data[key], (False, True)).all():
            raise ValueError(f"{path.name}: {key} must be a binary mask")
        data[key] = data[key].astype(bool)
    data["names"] = names
    return data


def frozen_populations(step0: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    truth = step0["truth"]
    valid = truth < 4
    raw_ok, deep_ok = step0["ps"].argmax(axis=1) == truth, step0["pd"].argmax(axis=1) == truth
    # Frozen legacy tie convention: an exact threshold belongs to the lower bin.
    qbin = np.searchsorted(Q_EDGES, step0["q"], side="left")
    populations = {"All_FG": valid, "Deep-Win_0": valid & ~raw_ok & deep_ok,
                   "Shallow-Win_0": valid & raw_ok & ~deep_ok,
                   "Both-Wrong_0": valid & ~raw_ok & ~deep_ok,
                   "Stable-Correct_0": valid & raw_ok & deep_ok,
                   "Raw-Correct_0": valid & raw_ok, "Raw-Wrong_0": valid & ~raw_ok,
                   "Exactly-One-Correct_0": valid & (raw_ok != deep_ok),
                   "Top20_q0": valid & step0["top20"],
                   "boundary": valid & step0["boundary"], "interior": valid & ~step0["boundary"]}
    populations.update({f"Q{i + 1}_q0": valid & (qbin == i) for i in range(5)})
    return populations


def _pixel_metrics(snapshot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    truth = snapshot["truth"]
    safe_truth = np.minimum(truth, 3).astype(np.int64)
    metrics = {}
    for name, key in (("raw", "ps"), ("deep", "pd"), ("rect", "rect")):
        prob = snapshot[key]
        metrics[name + "_accuracy"] = prob.argmax(axis=1) == truth
        metrics[name + "_gt_probability"] = np.take_along_axis(prob, safe_truth[:, None], axis=1)[:, 0]
        if name != "rect":
            logits = snapshot[name + "_logits"]
            gt = np.take_along_axis(logits, safe_truth[:, None], axis=1)[:, 0]
            others = np.where(np.arange(4)[None, :, None] == safe_truth[:, None], -np.inf, logits)
            metrics[name + "_gt_margin"] = gt - others.max(axis=1)
    return metrics


def population_statistics(snapshot: dict[str, np.ndarray], populations: dict[str, np.ndarray]):
    metrics = _pixel_metrics(snapshot)
    names = list(populations)
    image_counts = np.stack([populations[name].sum(axis=1) for name in names], axis=1).astype(np.float64)
    image_sums = np.empty((len(snapshot["names"]), len(names), len(POP_METRICS)), dtype=np.float64)
    for i, name in enumerate(names):
        for j, metric in enumerate(POP_METRICS):
            image_sums[:, i, j] = np.where(populations[name], metrics[metric], 0).sum(axis=1, dtype=np.float64)
    means = _divide(image_sums.sum(axis=0), image_counts.sum(axis=0)[:, None])
    return image_counts, image_sums, means


def native_metrics(snapshot: dict[str, np.ndarray], arm: str, step: int) -> list[dict[str, Any]]:
    truth = snapshot["truth"]
    valid = truth < 4
    fg_truth = truth[valid].astype(np.int64)
    rows = []
    for head, key in (("raw", "ps"), ("deep", "pd"), ("rect", "rect")):
        prob = snapshot[key].transpose(0, 2, 1)[valid].astype(np.float64)
        prediction = prob.argmax(axis=1)
        cm = np.bincount(4 * fg_truth + prediction, minlength=16).reshape(4, 4)
        metric = confusion_metrics(cm, official=False)
        gt_probability = prob[np.arange(len(prob)), fg_truth]
        # Brier is the sum over four classes, not its class-average variant.
        brier = (np.square(prob).sum(axis=1) - 2 * gt_probability + 1).mean() if len(prob) else np.nan
        # Match the frozen Phase2B1.9 diagnostic, including additive EPS.
        nll = -np.log(gt_probability + 1e-8).mean() if len(prob) else np.nan
        row = {"arm": arm, "step": step, "head": head, "n_pixels": len(prob),
               **{key: float(metric[key]) for key in ("accuracy", "miou", "mdice")},
               "nll": float(nll), "brier": float(brier), "mask": "foreground_truth_0_to_3"}
        for cls in range(4):
            row[f"iou_class{cls}"] = float(metric["iou"][cls])
            row[f"dice_class{cls}"] = float(metric["dice"][cls])
        rows.append(row)
    return rows


def _fraction(condition: np.ndarray, population: np.ndarray) -> float:
    return float(_divide(np.count_nonzero(condition & population), np.count_nonzero(population)))


def gate_statistics(snapshot: dict[str, np.ndarray], populations: dict[str, np.ndarray],
                    initial_gate: np.ndarray, arm: str, step: int):
    gate, q, delta = snapshot["delta"] > 0, snapshot["q"], snapshot["delta"]
    valid = populations["All_FG"]
    dw, sw, exactly = [populations[key] for key in ("Deep-Win_0", "Shallow-Win_0", "Exactly-One-Correct_0")]
    row = {"arm": arm, "step": step, "n_all_pixels": gate.size, "n_foreground_pixels": int(valid.sum()),
           "mean_q_all": float(q.mean(dtype=np.float64)), "mean_q_foreground": float(q[valid].mean(dtype=np.float64)) if valid.any() else np.nan,
           "active_fraction_all": float(gate.mean()), "active_fraction_foreground": _fraction(gate, valid),
           "delta_mean_all": float(delta.mean(dtype=np.float64)), "delta_median_all": float(np.median(delta)),
           "deep_capture_frozen": _fraction(gate, dw), "shallow_protection_frozen": _fraction(~gate, sw),
           "deep_selection_precision_frozen": _fraction(dw, gate & exactly),
           "frozen_exactly_one_correct_count": int(exactly.sum())}
    for label, values in (("all", q.ravel()), ("foreground", q[valid])):
        for quantile in (0, 20, 40, 60, 80, 100):
            row[f"q_p{quantile}_{label}"] = float(np.percentile(values, quantile)) if values.size else np.nan
    raw_ok = snapshot["ps"].argmax(axis=1) == snapshot["truth"]
    deep_ok = snapshot["pd"].argmax(axis=1) == snapshot["truth"]
    dynamic = {"deep_win": valid & ~raw_ok & deep_ok, "shallow_win": valid & raw_ok & ~deep_ok,
               "both_wrong": valid & ~raw_ok & ~deep_ok, "stable_correct": valid & raw_ok & deep_ok}
    for name, population in dynamic.items():
        row[f"current_{name}_count"] = int(population.sum())
        row[f"current_{name}_gate_fraction"] = _fraction(gate, population)
    dynamic_exactly = dynamic["deep_win"] | dynamic["shallow_win"]
    row["deep_capture_dynamic"] = _fraction(gate, dynamic["deep_win"])
    row["shallow_protection_dynamic"] = _fraction(~gate, dynamic["shallow_win"])
    row["deep_selection_precision_dynamic"] = _fraction(dynamic["deep_win"], gate & dynamic_exactly)
    drift = []
    drift_populations = {"all_grid": np.ones_like(gate, dtype=bool), "All_FG": valid,
                         **{key: populations[key] for key in ("Top20_q0", "Deep-Win_0", "Shallow-Win_0")}}
    for name, population in drift_populations.items():
        drift.append({"arm": arm, "step": step, "population": name, "n_pixels": int(population.sum()),
                      "gate_flip_rate": _fraction(gate != initial_gate, population)})
    return row, drift


def _truthy(value: Any) -> bool | None:
    if value is True or (isinstance(value, str) and value.lower() in ("true", "1", "pass", "passed")):
        return True
    if value is False or (isinstance(value, str) and value.lower() in ("false", "0", "fail", "failed")):
        return False
    return None


def training_statistics(rows: list[dict[str, Any]], calibration: dict[str, Any]):
    """Audit all 1..500 steps, not snapshots or final minibatch only."""
    required = ("main_loss", "aux_loss", "weighted_aux_loss", "total_loss", "main_grad_norm",
                "aux_grad_norm", "weighted_gradient_ratio", "gradient_cosine", "total_grad_norm",
                "lr0", "lr1", "lr2", "lr3", "active_fraction", "seconds")
    evidence: dict[str, Any] = {"present": bool(rows), "errors": [], "arms": {}}
    summaries = []
    if not rows:
        return evidence, summaries
    parsed = {arm: [] for arm in ARMS}
    for row in rows:
        arm = row.get("arm")
        if arm not in ARMS:
            evidence["errors"].append(f"Unknown training arm: {arm}")
            continue
        try:
            item = {key: float(row[key]) for key in required}
            raw_step = float(row["step"])
            if not raw_step.is_integer():
                raise ValueError("noninteger step")
            item["step"] = int(raw_step)
            item["finite"] = _truthy(row.get("finite"))
            if not all(np.isfinite(item[key]) for key in required):
                evidence["errors"].append(f"{arm}/{item['step']}: nonfinite numeric training value")
            if item["finite"] is not True:
                evidence["errors"].append(f"{arm}/{item['step']}: finite flag not true")
            parsed[arm].append(item)
        except (KeyError, TypeError, ValueError) as exc:
            evidence["errors"].append(f"Malformed training row for {arm}: {exc}")
    lambda_value = calibration.get("lambda_value")
    for arm in ARMS:
        records = sorted(parsed[arm], key=lambda item: item["step"])
        actual_steps = [item["step"] for item in records]
        sequence_exact = actual_steps == list(range(1, 501))
        if not sequence_exact:
            evidence["errors"].append(f"{arm}: training step sequence is not exactly 1..500")
        ratios = np.array([item["weighted_gradient_ratio"] for item in records])
        arm_evidence = {"rows": len(records), "step_sequence_exact": sequence_exact,
                        "weighted_gradient_ratio_median_all500": float(np.median(ratios)) if len(ratios) else np.nan,
                        "weighted_gradient_ratio_max": float(np.max(ratios)) if len(ratios) else np.nan}
        if lambda_value is not None and records:
            aux = np.array([item["aux_loss"] for item in records])
            weighted = np.array([item["weighted_aux_loss"] for item in records])
            expected = aux * (float(lambda_value) if arm != "B" else 0.0)
            consistent = bool(np.allclose(weighted, expected, rtol=1e-5, atol=1e-10))
            arm_evidence["lambda_weight_consistent_all500"] = consistent
            arm_evidence["lambda_max_absolute_weight_error"] = float(np.max(np.abs(weighted - expected)))
            if not consistent:
                evidence["errors"].append(f"{arm}: weighted auxiliary loss inconsistent with the one calibrated lambda")
        else:
            arm_evidence["lambda_weight_consistent_all500"] = None
        if records:
            main = np.array([item["main_loss"] for item in records])
            weighted = np.array([item["weighted_aux_loss"] for item in records])
            total = np.array([item["total_loss"] for item in records])
            if not np.allclose(total, main + weighted, rtol=2e-5, atol=1e-7):
                evidence["errors"].append(f"{arm}: total loss not main + weighted auxiliary loss")
            for item in records:
                if item["weighted_gradient_ratio"] < 0 or not 0 <= item["active_fraction"] <= 1:
                    evidence["errors"].append(f"{arm}/{item['step']}: invalid ratio/active fraction")
        evidence["arms"][arm] = arm_evidence
        for block_start in range(1, 501, 50):
            block = [item for item in records if block_start <= item["step"] < block_start + 50]
            if not block:
                continue
            for metric in required:
                values = np.array([item[metric] for item in block])
                summaries.append({"arm": arm, "start_step": block_start, "end_step": block_start + 49,
                                  "metric": metric, "n": len(values), "mean": float(np.mean(values)),
                                  "std": float(np.std(values, ddof=0)), "median": float(np.median(values)),
                                  "min": float(np.min(values)), "max": float(np.max(values)), "std_ddof": 0})
    return evidence, summaries


def _finite(*values: Any) -> bool:
    return all(isinstance(value, (int, float, np.number)) and np.isfinite(value) for value in values)


def _gate(status: str, facts: dict[str, Any], rule: str) -> dict[str, Any]:
    return {"status": status, "facts": facts, "rule": rule}


def _status(check: bool, *values: Any) -> str:
    return ("PASS" if check else "FAIL") if _finite(*values) else "PENDING"


def evaluate_gates(summary: dict[str, Any], evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply approved decision priority. Missing evidence can never become GO."""
    bootstrap = summary["bootstrap"]
    official = summary["official_metrics"]
    population_rows = summary["population_metrics"]

    def boot(step: int, comparison: str, metric: str, population: str = "official"):
        return next(row for row in bootstrap if row["step"] == step and row["comparison"] == comparison
                    and row["metric"] == metric and row["population"] == population)

    def pop(arm: str, step: int, name: str):
        return next(row for row in population_rows if row["arm"] == arm and row["step"] == step and row["population"] == name)

    a = boot(500, "A-B", "miou")
    a_delta, a_low = a["delta"], a["ci_low"]
    a_pass = _finite(a_delta, a_low) and a_delta >= .001 and a_low > 0
    a_status = _status(a_pass, a_delta, a_low)
    if _finite(a_delta, a_low) and 0 < a_delta < .001:
        a_status = "WEAK_POSITIVE"
    gates = {"A": _gate(a_status, {"delta_pp": a_delta * 100 if a_delta is not None else None,
                                   "ci_low_pp": a_low * 100 if a_low is not None else None},
                                "step500 A-B official mIoU >= +0.10 pp and paired 95% CI lower > 0; weak-positive is not PASS")}
    b100, b250 = boot(100, "A-B", "miou")["delta"], boot(250, "A-B", "miou")["delta"]
    b_pass = _finite(a_delta, b100, b250) and b250 > 0 and a_delta > 0 and a_delta >= b100 - .001
    gates["B"] = _gate(_status(b_pass, a_delta, b100, b250),
                        {"delta100_pp": b100 * 100 if b100 is not None else None,
                         "delta250_pp": b250 * 100 if b250 is not None else None,
                         "delta500_pp": a_delta * 100 if a_delta is not None else None},
                        "A-B mIoU > 0 at 250 and 500; delta500 >= delta100 - 0.10 pp")
    c = boot(500, "A-R", "miou")
    c_delta, c_low = c["delta"], c["ci_low"]
    c_pass = _finite(c_delta, c_low) and c_delta >= .0005 and c_low > 0
    gates["C"] = _gate(_status(c_pass, c_delta, c_low),
                        {"delta_pp": c_delta * 100 if c_delta is not None else None,
                         "ci_low_pp": c_low * 100 if c_low is not None else None},
                        "step500 A-R official mIoU >= +0.05 pp and paired 95% CI lower > 0")
    d = boot(500, "A-B", "raw_accuracy", "Deep-Win_0")
    dm = boot(500, "A-B", "raw_gt_margin", "Deep-Win_0")
    d_pass = _finite(d["delta"], d["ci_low"], dm["delta"]) and d["delta"] > 0 and d["ci_low"] > 0 and dm["delta"] > 0
    gates["D"] = _gate(_status(d_pass, d["delta"], d["ci_low"], dm["delta"]),
                        {"raw_accuracy_delta_pp": d["delta"] * 100 if d["delta"] is not None else None,
                         "ci_low_pp": d["ci_low"] * 100 if d["ci_low"] is not None else None,
                         "raw_logit_gt_margin_delta": dm["delta"]},
                        "Deep-Win_0 raw accuracy A-B > 0 with CI lower > 0 and raw logit GT margin A-B > 0")
    e = boot(500, "A-B", "raw_accuracy", "Shallow-Win_0")
    em = boot(500, "A-B", "raw_gt_margin", "Shallow-Win_0")
    margin0 = pop("B", 0, "Shallow-Win_0")["raw_gt_margin"]
    tolerance = .05 * abs(margin0) if _finite(margin0) else np.nan
    e_pass = _finite(e["delta"], e["ci_low"], em["delta"], tolerance) and e["delta"] >= -.002 and e["ci_low"] > -.005 and em["delta"] >= -tolerance
    gates["E"] = _gate(_status(e_pass, e["delta"], e["ci_low"], em["delta"], tolerance),
                        {"raw_accuracy_delta_pp": e["delta"] * 100 if e["delta"] is not None else None,
                         "ci_low_pp": e["ci_low"] * 100 if e["ci_low"] is not None else None,
                         "raw_logit_gt_margin_delta": em["delta"], "step0_mean_logit_margin": margin0,
                         "fixed_margin_tolerance": tolerance},
                        "Shallow-Win_0 accuracy A-B >= -0.20 pp; CI lower > -0.50 pp; margin_A-margin_B >= -0.05*abs(step0 mean margin)")
    fa, fb = pop("A", 500, "Stable-Correct_0"), pop("B", 500, "Stable-Correct_0")
    fraw = fa["raw_accuracy"] - fb["raw_accuracy"] if _finite(fa["raw_accuracy"], fb["raw_accuracy"]) else np.nan
    frect = fa["rect_accuracy"] - fb["rect_accuracy"] if _finite(fa["rect_accuracy"], fb["rect_accuracy"]) else np.nan
    gates["F"] = _gate(_status(fraw >= -.001 and frect >= -.001, fraw, frect),
                        {"raw_accuracy_delta_pp": fraw * 100, "rect_accuracy_delta_pp": frect * 100,
                         "automatic_fail_below_minus030pp": bool(fraw < -.003 or frect < -.003)},
                        "Stable-Correct_0 raw and rect accuracy A-B each >= -0.10 pp; either < -0.30 pp is automatic FAIL")
    oa = next(row for row in official if row["arm"] == "A" and row["step"] == 500)
    ob = next(row for row in official if row["arm"] == "B" and row["step"] == 500)
    classes = [oa[f"iou_class{i}"] - ob[f"iou_class{i}"] if _finite(oa[f"iou_class{i}"], ob[f"iou_class{i}"]) else np.nan for i in range(4)]
    macro = float(np.mean(classes))
    gates["G"] = _gate(_status(min(classes) >= -.005 and macro > 0, *classes, macro),
                        {"per_class_iou_delta_pp": [value * 100 for value in classes], "macro_delta_pp": macro * 100,
                         "classes_checked": [0, 1, 2, 3]},
                        "Conservative all4-class check: each IoU A-B >= -0.50 pp and macro mean class delta > 0")
    provenance = evidence["optimizer_provenance"]
    p_state = _truthy(provenance.get("resolved"))
    provenance_errors = []
    if p_state is True:
        expected = {"checkpoint_sha256": CHECKPOINT_SHA256, "global_step": 29275, "max_step": 29275,
                    "state_policy": "fresh_optimizer_state"}
        for key, value in expected.items():
            if provenance.get(key) != value:
                provenance_errors.append(f"{key}: expected {value!r}, got {provenance.get(key)!r}")
    provenance_status = "BLOCKED" if p_state is False or provenance_errors else ("PASS" if p_state is True else "PENDING")
    runtime, identity, verification, calibration = [evidence[key] for key in ("runtime", "identity_step0", "verification", "lambda_calibration")]
    required_checks: dict[str, bool | None] = {}
    for key in ("all_finite", "no_amp_skipped_step", "no_unexpected_gradient_path", "no_state_corruption",
                "bn_statistics_frozen", "no_test_access", "no_luad_access"):
        required_checks["runtime." + key] = _truthy(runtime.get("checks", {}).get(key))
    required_checks["runtime.completed"] = _truthy(runtime.get("completed"))
    if "steps_per_arm" in runtime:
        required_checks["runtime.steps_per_arm"] = runtime["steps_per_arm"] == {arm: 500 for arm in ARMS}
    else:
        required_checks["runtime.steps_per_arm"] = None
    for key in ("three_arms_bitwise_equal", "strict_load", "main_forward_parity", "original_sources_unchanged"):
        required_checks["identity." + key] = _truthy(identity.get(key))
    required_checks["snapshot.step0_bitwise_equal"] = summary.get("snapshot_step0_bitwise_equal")
    required_checks["verification.passed"] = _truthy(verification.get("passed"))
    if verification.get("failures"):
        required_checks["verification.no_failures"] = False
    if calibration.get("missing_file"):
        required_checks["calibration.valid"] = None
    else:
        ratios = calibration.get("ratios", [])
        lambda_value, median = calibration.get("lambda_value"), calibration.get("r_median")
        numeric_ratios = len(ratios) == 32 and all(_finite(ratio) and ratio >= 0 for ratio in ratios)
        required_checks["calibration.valid"] = bool(
            calibration.get("batches") == 32 and calibration.get("seed") == 42
            and calibration.get("no_optimizer_step") is True and calibration.get("state_unchanged") is True
            and numeric_ratios and _finite(lambda_value, median)
            and np.isclose(median, np.median(ratios), rtol=1e-12, atol=1e-12)
            and np.isclose(lambda_value, .1 * median, rtol=1e-12, atol=1e-12))
    train = summary["training_audit"]
    required_checks["training.complete_finite_consistent"] = not bool(train["errors"]) if train["present"] else None
    ratio_a = train.get("arms", {}).get("A", {}).get("weighted_gradient_ratio_median_all500")
    ratio_r = train.get("arms", {}).get("R", {}).get("weighted_gradient_ratio_median_all500")
    gate500 = next(row for row in summary["gate_dynamics"] if row["arm"] == "A" and row["step"] == 500)
    active = gate500["active_fraction_all"]
    required_checks["step500.active_fraction_all_grid"] = .05 <= active <= .60 if _finite(active) else None
    required_checks["training.A_median_weighted_gradient_ratio_all500"] = ratio_a <= .30 if _finite(ratio_a) else None
    h_status = "FAIL" if False in required_checks.values() else ("PENDING" if None in required_checks.values() else "PASS")
    gates["H"] = _gate(h_status, {"checks": required_checks, "active_fraction_all_grid_step500_A": active,
                                   "weighted_gradient_ratio_median_all500_A": ratio_a,
                                   "weighted_gradient_ratio_median_all500_R": ratio_r, "training_errors": train["errors"]},
                        "All numerical/path/state/paired-run evidence passes; step500 A all-grid active fraction in [0.05,0.60]; median weighted gradient ratio over all500 A steps <= 0.30")
    decision = choose_decision(provenance_status, gates)
    strong = bool(all(gate["status"] == "PASS" for gate in gates.values()) and a_delta >= .003 and c_delta >= .0015
                  and d["ci_low"] > 0 and min(classes) >= -.0025 and provenance_status == "PASS")
    return {"gates": gates, "provenance_status": provenance_status, "provenance_errors": provenance_errors,
            "decision": decision, "report_status": "FINAL" if decision else "DRAFT_PENDING_EVIDENCE",
            "strong_short_horizon_adt_signal": strong if decision else None}


def choose_decision(provenance_status: str, gates: dict[str, dict[str, Any]]) -> str | None:
    """Confirmed contract priority; unresolved higher-priority evidence blocks finality."""
    if provenance_status == "BLOCKED":
        return "SHORT_HORIZON_OPTIMIZER_PROVENANCE_BLOCKED"
    if provenance_status != "PASS":
        return None
    status = {key: gates[key]["status"] for key in "ABCDEFGH"}
    if status["H"] == "FAIL":
        return "ADT_SHORT_HORIZON_ENGINEERING_NOGO"
    if status["H"] != "PASS":
        return None
    if any(status[key] == "PENDING" for key in "AB"):
        return None
    if any(status[key] != "PASS" for key in "AB"):
        return "ADT_LOCAL_SIGNAL_NOT_TRANSLATING_TO_OPTIMIZATION"
    if any(status[key] == "PENDING" for key in "DEFG"):
        return None
    if any(status[key] != "PASS" for key in "DEFG"):
        return "ADT_OPTIMIZATION_GAIN_WITH_SEMANTIC_SAFETY_REGRESSION"
    if status["C"] == "PENDING":
        return None
    if status["C"] != "PASS":
        return "SHORT_HORIZON_GAIN_NOT_CONTEXT_SPECIFIC"
    return "RDDR_ADT_SHORT_HORIZON_DYNAMICS_GO"


def _metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {name: _read_json(run_dir / f"{PREFIX}{name}.json") for name in
            ("optimizer_provenance", "lambda_calibration", "batch_manifest", "runtime", "identity_step0", "verification")}


def analyze_snapshots(run_dir: Path) -> dict[str, Any]:
    """Read each snapshot once, retaining only pooled and per-image statistics."""
    base = load_snapshot(run_dir / "snapshot_0000_B.npz")
    populations = frozen_populations(base)
    population_names = list(populations)
    initial_gate = base["delta"] > 0
    _, _, initial_means = population_statistics(base, populations)
    official_rows, native_rows, population_rows = [], [], []
    gate_rows, drift_rows, per_class_rows, bootstrap_rows, random_rows = [], [], [], [], []
    bitwise_step0 = True
    frozen_ci_names = ("Deep-Win_0", "Shallow-Win_0", "Stable-Correct_0", "Raw-Wrong_0")
    frozen_ci_indices = [population_names.index(name) for name in frozen_ci_names]
    for step in STEPS:
        print(f"Analyzing paired snapshot step {step}", flush=True)
        paired = {}
        step_official = {}
        for arm in ARMS:
            snapshot = base if step == 0 and arm == "B" else load_snapshot(run_dir / f"snapshot_{step:04d}_{arm}.npz")
            for key in ("names", "truth", "boundary", "top20"):
                if not np.array_equal(snapshot[key], base[key]):
                    raise ValueError(f"{arm}/{step}: frozen image identity or label array {key} differs from B step0")
            if step == 0:
                for key in base:
                    bitwise_step0 &= snapshot[key].dtype == base[key].dtype and snapshot[key].tobytes() == base[key].tobytes()
            counts, sums, means = population_statistics(snapshot, populations)
            cm = snapshot["official_cm"]
            official = confusion_metrics(cm.sum(axis=0))
            row = {"arm": arm, "step": step, "n_images": len(snapshot["names"]),
                   **{key: float(official[key]) for key in ("accuracy", "miou", "mdice")},
                   "primary_endpoint": step == 500, "estimator": "pooled_official_confusion_matrix"}
            for cls in range(4):
                row[f"iou_class{cls}"] = float(official["iou"][cls])
                row[f"dice_class{cls}"] = float(official["dice"][cls])
                per_class_rows.append({"arm": arm, "step": step, "class_id": cls,
                                       "iou": float(official["iou"][cls]), "dice": float(official["dice"][cls]),
                                       "gt_pixels": float(cm[:, cls, :].sum()),
                                       "predicted_pixels": float(cm[:, :, cls].sum()),
                                       "all_four_classes_checked": True})
            official_rows.append(row)
            step_official[arm] = row
            native_rows.extend(native_metrics(snapshot, arm, step))
            for i, name in enumerate(population_names):
                pop_row = {"arm": arm, "step": step, "population": name,
                           "n_pixels": int(counts[:, i].sum()), "n_images_with_population": int((counts[:, i] > 0).sum()),
                           "frozen_at_step": 0, "denominator": "pooled_foreground_pixels"}
                for j, metric in enumerate(POP_METRICS):
                    pop_row[metric] = float(means[i, j])
                    pop_row[metric + "_gain_vs_step0"] = float(means[i, j] - initial_means[i, j])
                population_rows.append(pop_row)
            gate, drift = gate_statistics(snapshot, populations, initial_gate, arm, step)
            gate_rows.append(gate)
            drift_rows.extend(drift)
            paired[arm] = {"cm": cm, "counts": counts[:, frozen_ci_indices], "sums": sums[:, frozen_ci_indices]}
            if snapshot is not base:
                del snapshot
        for left, right in COMPARISONS:
            comparison = f"{left}-{right}"
            estimates = bootstrap_confusions(paired[left]["cm"], paired[right]["cm"])
            for metric, interval in estimates.items():
                bootstrap_rows.append({"step": step, "comparison": comparison, "population": "official",
                                       "metric": metric, "n_images": len(base["names"]), "unit": "fraction", **interval,
                                       "delta_pp": 100 * interval["delta"], "ci_low_pp": 100 * interval["ci_low"],
                                       "ci_high_pp": 100 * interval["ci_high"]})
            if not np.array_equal(paired[left]["counts"], paired[right]["counts"]):
                raise ValueError("Frozen-population bootstrap denominators differ across arms")
            image_denominators = np.repeat(paired[left]["counts"], len(POP_METRICS), axis=1)
            estimates = bootstrap_ratios(paired[left]["sums"].reshape(len(base["names"]), -1),
                                         paired[right]["sums"].reshape(len(base["names"]), -1), image_denominators)
            for i, name in enumerate(frozen_ci_names):
                for j, metric in enumerate(POP_METRICS):
                    interval = estimates[i * len(POP_METRICS) + j]
                    is_margin = metric.endswith("margin")
                    bootstrap_rows.append({"step": step, "comparison": comparison, "population": name,
                                           "metric": metric, "n_images": len(base["names"]),
                                           "unit": "logit" if is_margin else "fraction", **interval,
                                           "delta_pp": None if is_margin else 100 * interval["delta"],
                                           "ci_low_pp": None if is_margin else 100 * interval["ci_low"],
                                           "ci_high_pp": None if is_margin else 100 * interval["ci_high"]})
        for row in bootstrap_rows:
            if row["step"] == step and row["comparison"] in ("A-R", "R-B"):
                random_rows.append(dict(row))
        print(f"Completed step {step}: official mIoU B/A/R = "
              + "/".join(f"{step_official[arm]['miou']:.8f}" for arm in ARMS), flush=True)
    for row in per_class_rows:
        baseline = next(item for item in per_class_rows if item["arm"] == "B" and item["step"] == row["step"] and item["class_id"] == row["class_id"])
        random = next(item for item in per_class_rows if item["arm"] == "R" and item["step"] == row["step"] and item["class_id"] == row["class_id"])
        row["iou_delta_vs_B_pp"] = 100 * (row["iou"] - baseline["iou"])
        row["dice_delta_vs_B_pp"] = 100 * (row["dice"] - baseline["dice"])
        row["iou_delta_vs_R_pp"] = 100 * (row["iou"] - random["iou"])
    output = {"official_metrics": official_rows, "native28_metrics": native_rows,
              "population_metrics": population_rows, "gate_dynamics": gate_rows, "gate_drift": drift_rows,
              "per_class": per_class_rows, "bootstrap": bootstrap_rows, "random_control": random_rows}
    for name, rows in output.items():
        _write_csv(run_dir / f"{PREFIX}{name}.csv", rows)
    for name, population in (("deepwin", "Deep-Win_0"), ("shallowwin", "Shallow-Win_0"),
                              ("stablecorrect", "Stable-Correct_0"), ("rawwrong", "Raw-Wrong_0")):
        _write_csv(run_dir / f"{PREFIX}{name}.csv", [row for row in population_rows if row["population"] == population])
    return {"schema_version": 1, "phase": "Phase2B1.12", "n_validation_images": len(base["names"]),
            "snapshot_steps": list(STEPS), "arms": list(ARMS), "primary_step": 500,
            "snapshot_step0_bitwise_equal": bool(bitwise_step0),
            "frozen_populations": {name: {"n_pixels": int(mask.sum()), "n_images": int(mask.any(axis=1).sum())}
                                   for name, mask in populations.items()},
            "settings": {"bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                         "bootstrap_estimator": "pooled_confusion_or_population_sum_over_pooled_count",
                         "q_quintile_edges": Q_EDGES.tolist(), "q_edge_tie_convention": "searchsorted_side_left",
                         "native_mask": "truth<4", "gt_margin": "GT_raw_logit_minus_max_other_logit",
                         "native_nll_additive_epsilon": 1e-8, "native_nll_formula": "-mean(log(p_GT+1e-8))",
                         "brier": "sum_over_four_classes",
                         "official_iou_absent_class": "nanmean", "official_dice_absent_class": 0,
                         "gate_h_active_fraction_mask": "all_native28_grid_positions",
                         "gate_h_gradient_ratio": "median_all500_A_training_steps",
                         "gate_g_classes": [0, 1, 2, 3]}, **output}


def _format(value: Any) -> str:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.8g}"
    if isinstance(value, (dict, list)):
        return json.dumps(_json_safe(value), ensure_ascii=False).replace("|", "\\|")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "PENDING: no supporting rows supplied.\n"
    text = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    text += "\n".join("| " + " | ".join(_format(row.get(key)) for key in columns) + " |" for row in rows)
    return text + "\n"


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n```\n"


def render_report(summary: dict[str, Any], evidence: dict[str, dict[str, Any]], run_dir: Path, report_path: Path) -> None:
    """Render exactly 29 sections from evidence; pending reports have no fake decision."""
    sections = []

    def add(title: str, body: str):
        sections.append(f"## {len(sections) + 1}. {title}\n\n{body.rstrip()}\n")

    def artifact(name: str, suffix: str = "csv") -> str:
        return f"[{PREFIX}{name}.{suffix}]({(run_dir / (PREFIX + name + '.' + suffix)).as_posix()})"

    def boot_rows(comparison: str, population: str = "official", step: int | None = None):
        return [row for row in summary["bootstrap"] if row["comparison"] == comparison and row["population"] == population
                and (step is None or row["step"] == step)]

    def population_section(name: str, artifact_name: str) -> str:
        rows = [row for row in summary["population_metrics"] if row["population"] == name]
        counts = summary["frozen_populations"][name]
        body = f"Frozen B-step0 membership: {counts['n_pixels']} foreground pixels in {counts['n_images']} images. "
        body += "All means use pooled foreground pixels. GT margin is a logit difference; probability is reported separately.\n\n"
        body += _table(rows, ["arm", "step", "raw_accuracy", "rect_accuracy", "raw_gt_probability", "raw_gt_margin",
                              "raw_accuracy_gain_vs_step0", "rect_accuracy_gain_vs_step0", "raw_gt_probability_gain_vs_step0"])
        body += "\nStep500 paired A-B image bootstrap:\n\n"
        body += _table(boot_rows("A-B", name, 500), ["metric", "delta", "ci_low", "ci_high", "unit", "valid_replicates"])
        return body + "\n" + artifact(artifact_name)

    add("Provenance", f"Status: {summary['report_status']}. Approved pure A0 base: `{BASE_COMMIT}`. "
        f"Locked C0 checkpoint SHA256: `{CHECKPOINT_SHA256}`. These identifiers are the preregistered references, "
        "not a substitute for the supplied provenance/verification evidence. No training or model selection is performed by this analysis.\n\n"
        + _json_block({"provenance_status": summary["provenance_status"], "provenance_errors": summary["provenance_errors"],
                       "source_hashes": evidence["optimizer_provenance"].get("source_hashes"),
                       "verification_passed": evidence["verification"].get("passed")})
        + "\n" + artifact("verification", "json"))
    add("Optimizer provenance", _json_block(evidence["optimizer_provenance"]) + "\n" + artifact("optimizer_provenance", "json"))
    add("Exact starting state", _json_block(evidence["identity_step0"]) + "\nSnapshot-array bitwise step0 equality: "
        + str(summary["snapshot_step0_bitwise_equal"]) + ". Checkpoint identity is separately attested by the supplied identity artifact.")
    add("Three-arm design", "Registered arms: B = official SSHR continuation; A = official loss + calibrated ADT; "
        "R = official loss + rate-matched random gate. All start at C0, target exactly 500 steps, "
        "and use BCSS seed42 / batch20 / BF16. Auxiliary updates include the approved b4..b4_5/bn45 affine parameters; "
        "all BN running statistics remain frozen. BN-affine and the corresponding original optimizer weight-decay effects "
        "are part of the A/R treatment. Actual execution checks:\n\n" + _json_block(evidence["runtime"].get("checks", {"status": "PENDING"})))
    add("ADT implementation", "Registered formula: `sum(q*m_D*KL(stopgrad(p_d)||p_s_aux))/(sum(q*m_D)+eps)`, "
        "with `q=JS(p_s,p_d)/ln(2)` and `m_D=(Delta_sym>0)`. q indicates need, not correctness or direction. "
        "The 15x15 exclude-self support is frozen. feat56, ic1, deep target, q, Delta and the gate are detached. "
        "No third evidence or threshold/loss redesign is authorized. Path correctness is an engineering verification claim, "
        "not something snapshot-only statistics can prove.\n\n" + artifact("verification", "json"))
    add("Random control", "The registered random gate matches the current A-arm per-image active count, uses independent "
        "seed42 randomness, and weights R's own predictions/q. This is a three-arm comparison, not a gate seed search. "
        "The following are actual paired snapshot differences; implementation/rate matching requires the independent checks.\n\n"
        + _table([row for row in summary["random_control"] if row["population"] == "official" and row["step"] == 500],
                 ["comparison", "metric", "delta_pp", "ci_low_pp", "ci_high_pp"])
        + "\n" + artifact("random_control"))
    add("Lambda calibration", _json_block(evidence["lambda_calibration"]) + "\nAll-step weighted-loss consistency audit:\n\n"
        + _json_block({arm: summary["training_audit"].get("arms", {}).get(arm, {}) for arm in ARMS})
        + "\nThis is one train-only no-step 32-batch calibration, not a validation-selected strength. "
        "Logged loss-weight equality checks consistency; detachment and absence of validation use require implementation verification.")
    manifest = evidence["batch_manifest"]
    manifest_array_keys = ("batches", "records", "images", "entries", "calibration", "training")
    manifest_summary = {key: value for key, value in manifest.items() if key not in manifest_array_keys}
    for key in manifest_array_keys:
        if isinstance(manifest.get(key), list):
            manifest_summary[key + "_count"] = len(manifest[key])
    manifest_path = run_dir / f"{PREFIX}batch_manifest.json"
    manifest_summary["artifact_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None
    add("Train stream synchronization", "Same transformed batches and main-network RNG across arms are preregistered. "
        "Batch manifest evidence (large entry arrays remain in the linked artifact):\n\n"
        + _json_block(manifest_summary) + "\n" + artifact("batch_manifest", "json")
        + "\n\nMatching names alone does not prove augmentation/RNG equality; those require verification.")
    add("Snapshot schedule", _json_block({"evaluated_steps": summary["snapshot_steps"], "primary_step": 500,
                                          "registered_checkpoint_steps": [0, 250, 500],
                                          "actual_training_steps": evidence["runtime"].get("steps_per_arm"),
                                          "validation_images": summary["n_validation_images"]})
        + "\nNo best-checkpoint selection, horizon extension, test/LUAD evaluation or other seed is authorized.")
    add("Official validation curves", "Same FINAL-style canonical evaluator/resolution/background handling must be supplied by "
        "the runner and verified independently. The analysis pools per-image 5x5 matrices, zeros [4,4], computes foreground "
        "nanmean IoU, and assigns absent-class Dice zero, matching `tool/iouutils.py`. Values below are fractions.\n\n"
        + _table(summary["official_metrics"], ["arm", "step", "miou", "mdice", "accuracy", "n_images"])
        + "\n" + artifact("official_metrics"))
    add("Primary mIoU and mDice", "Only step500 is the primary endpoint. Native28 metrics cannot replace official metrics.\n\n"
        + _table([row for row in summary["official_metrics"] if row["step"] == 500], ["arm", "miou", "mdice"])
        + "\n" + _table([row for row in summary["bootstrap"] if row["population"] == "official" and row["step"] == 500],
                          ["comparison", "metric", "delta_pp", "ci_low_pp", "ci_high_pp"]))
    add("ADT-Baseline bootstrap", "10,000 paired image-level percentile bootstrap replicates, seed42. Each draw recomputes "
        "the metric from pooled image confusion matrices. It does not average image IoU. CI differences are percentage points.\n\n"
        + _table(boot_rows("A-B"), ["step", "metric", "delta_pp", "ci_low_pp", "ci_high_pp", "valid_replicates"])
        + "\n" + artifact("bootstrap"))
    add("ADT-Random bootstrap", _table(boot_rows("A-R"), ["step", "metric", "delta_pp", "ci_low_pp", "ci_high_pp", "valid_replicates"])
        + "\nRandom-Baseline uses the same paired resamples and is retained in " + artifact("random_control") + ".")
    add("Native28 curves", "Mechanism diagnostic only. All metrics mask foreground truth 0..3, ignoring background/255. "
        "NLL retains the prior diagnostic `-mean(log(p_GT+1e-8))` (additive EPS, not a probability floor); "
        "Brier is the sum of squared errors over four classes.\n\n"
        + _table(summary["native28_metrics"], ["arm", "step", "head", "accuracy", "miou", "mdice", "nll", "brier"])
        + "\n" + artifact("native28_metrics"))
    add("Frozen Deep-Win", population_section("Deep-Win_0", "deepwin"))
    add("Frozen Shallow-Win", population_section("Shallow-Win_0", "shallowwin"))
    add("Stable-Correct", population_section("Stable-Correct_0", "stablecorrect"))
    add("Raw-Wrong", population_section("Raw-Wrong_0", "rawwrong") + "\n\nThe old Phase2B1.9 >=40% local BenefitRate gate "
        "is not re-used or reinterpreted. Its `ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE` decision remains frozen.")
    add("Per-class", "All four official foreground classes are reported and conservatively checked; no post-hoc powered-class exclusion. "
        "The full horizon is retained in the CSV.\n\n"
        + _table([row for row in summary["per_class"] if row["step"] == 500],
                 ["arm", "class_id", "iou", "dice", "gt_pixels", "iou_delta_vs_B_pp", "iou_delta_vs_R_pp"])
        + "\n" + artifact("per_class"))
    add("Gate dynamics", "The transfer gate is always Delta>0. Active fraction includes an all-grid denominator (used by Gate H) "
        "and a foreground-only diagnostic. DeepCapture/ShallowProtection/SelectionPrecision use frozen exactly-one-correct labels; "
        "current populations are separately recorded as drift diagnostics. All q quantiles, Delta summaries, current counts and "
        "fractions are in the linked CSV.\n\n"
        + _table(summary["gate_dynamics"], ["arm", "step", "mean_q_all", "active_fraction_all", "active_fraction_foreground",
                                                 "deep_capture_frozen", "shallow_protection_frozen", "deep_selection_precision_frozen"])
        + "\n" + artifact("gate_dynamics") + "\n\nFrozen population definitions/counts:\n\n"
        + _json_block(summary["frozen_populations"]) + "\nQ-bin edges are inherited, with exact ties assigned to the lower bin. "
        "Top20 and boundary labels are inherited unchanged. All frozen-population metrics: " + artifact("population_metrics"))
    add("Gate drift", "Flip rates compare each current arm's Delta>0 with the identical B-step0 gate; frozen masks are never reselected.\n\n"
        + _table([row for row in summary["gate_drift"] if row["step"] == 500], ["arm", "population", "n_pixels", "gate_flip_rate"])
        + "\n" + artifact("gate_drift"))
    representation = _read_csv(run_dir / f"{PREFIX}representation_drift.csv")
    add("Representation drift", "Runner-supplied diagnostic on the registered fixed 160 validation images. Feature drift cannot "
        "be reconstructed from probability-only snapshots. The table below reproduces supplied rows, without inventing missing features.\n\n"
        + _table(representation[:30], list(representation[0]) if representation else [])
        + (f"\nShowing first 30 of {len(representation)} rows; full per-image evidence is linked.\n" if len(representation) > 30 else "")
        + "\n" + artifact("representation_drift"))
    interaction = _read_csv(run_dir / f"{PREFIX}gradient_interaction.csv")
    add("Gradient interaction", "Runner-supplied no-step diagnostics on the same fixed training minibatch at 0/50/250/500. "
        "These diagnostic rows are distinct from Gate H's median over all 500 A training steps.\n\n"
        + _table(interaction, list(interaction[0]) if interaction else []) + "\n" + artifact("gradient_interaction"))
    add("Loss/gradient stability", "Every-step loss, norms, LR, active fraction, finite flag and timing are retained. "
        "Each 50-step block is summarized by mean/population-std/median/min/max. No clipping is introduced by this analysis.\n\n"
        + _json_block(summary["training_audit"]) + "\n" + artifact("training_curve") + "\n\n" + artifact("loss_gradient_dynamics"))
    add("Runtime/memory", _json_block(evidence["runtime"]) + "\n" + artifact("runtime", "json"))
    add("Gate A-H", _table([{"gate": key, "status": gate["status"], "rule": gate["rule"]} for key, gate in summary["gates"].items()],
                            ["gate", "status", "rule"]) + "\nExact facts (fraction-to-pp conversion only, no rounded-threshold decision):\n\n"
        + _json_block({key: value["facts"] for key, value in summary["gates"].items()}))
    add("STRONG_SHORT_HORIZON_ADT_SIGNAL", _json_block({"STRONG_SHORT_HORIZON_ADT_SIGNAL": summary["strong_short_horizon_adt_signal"]})
        + "\nRequires all A-H PASS, A-B >=+0.30 pp, A-R >=+0.15 pp, positive Deep-Win accuracy CI, "
        "Shallow-Win protection, and every class IoU delta >=-0.25 pp. Missing evidence is not a false/true scientific result.")
    decision = summary["decision"]
    if decision is None:
        interpretation = "Evidence is incomplete, so this is a draft and no scientific/final decision is asserted. "
        interpretation += "Complete the linked pending checks and regenerate with `--report-only`; this does not retune the experiment."
    elif decision == "RDDR_ADT_SHORT_HORIZON_DYNAMICS_GO":
        interpretation = "All preregistered short-horizon gates pass in this run. This supports translation of frozen-point ADT "
        interpretation += "mechanism evidence into the tested 500-step optimization setting, not a Full25/multi-seed generalization claim. "
        interpretation += "A separately approved Phase2B2 protocol could be designed; it is not automatically launched."
    elif decision == "ADT_SHORT_HORIZON_ENGINEERING_NOGO":
        interpretation = "The engineering gate fails; this run cannot establish safe optimization translation. The failed checks "
        interpretation += "are enumerated above. The result must not be rescued with altered LR, lambda, horizon or gates."
    elif decision == "SHORT_HORIZON_OPTIMIZER_PROVENANCE_BLOCKED":
        interpretation = "Optimizer provenance is not uniquely established under the approved contract. Scientific gain claims "
        interpretation += "are not licensed by this run. Any supplied metric tables are audit evidence only."
    elif decision == "ADT_OPTIMIZATION_GAIN_WITH_SEMANTIC_SAFETY_REGRESSION":
        interpretation = "Real validation translation/persistence pass, but at least one frozen-population/class safety gate fails. "
        interpretation += "The optimization gain therefore does not establish safe hierarchical semantic transfer."
    elif decision == "SHORT_HORIZON_GAIN_NOT_CONTEXT_SPECIFIC":
        interpretation = "Validation translation and semantic safety pass, but superiority to the rate-matched random gate "
        interpretation += "does not meet the preregistered contextual-attribution requirement."
    else:
        interpretation = "The official translation or persistence criterion is not met; a positive sub-threshold point estimate "
        interpretation += "is insufficient. Frozen-point local mechanism evidence has not met this run's preregistered "
        interpretation += "standard for reliable short-horizon optimization translation."
    interpretation += "\n\nPhase2B1.9 remains `ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE`; Phase2B1.11 remains "
    interpretation += "`THIRD_EVIDENCE_OPERATIONAL_HEADROOM_INSUFFICIENT`. Neither historical decision is changed. "
    interpretation += "No hyperparameter rescue, threshold search, third-evidence route, or automatic next phase is authorized."
    add("Scientific interpretation", interpretation)
    add("Exact final decision", ("Approved priority: provenance blocked; H fail; A/B nonpass; D/E/F/G fail; C fail; all pass GO.\n\n"
                                 + (f"DECISION = {decision}" if decision else "Final decision pending evidence; this draft is not a final audit report.")))
    assert len(sections) == 29
    header = "# Phase2B1.12 Short-Horizon ADT Optimization Dynamics Audit\n\n"
    header += "All metric values are fractions unless a column explicitly says pp (percentage points). "
    header += "Validation supports only the preregistered step500 endpoint; no checkpoint selection is performed.\n\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(header + "\n".join(sections).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing run directory with immutable snapshot NPZ files")
    parser.add_argument("--report", type=Path, required=True, help="Output Markdown report path")
    parser.add_argument("--report-only", action="store_true", help="Refresh evidence/gates/report using existing computed summary; no new bootstrap")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"Run directory does not exist: {run_dir}")
    summary_path = run_dir / f"{PREFIX}summary.json"
    if args.report_only:
        summary = _read_json(summary_path)
        if summary.get("schema_version") != 1:
            parser.error("--report-only requires a completed schema_version=1 analysis summary")
    else:
        summary = analyze_snapshots(run_dir)
    evidence = _metadata(run_dir)
    training_rows = _read_csv(run_dir / f"{PREFIX}training_curve.csv")
    summary["training_audit"], loss_rows = training_statistics(training_rows, evidence["lambda_calibration"])
    _write_csv(run_dir / f"{PREFIX}loss_gradient_dynamics.csv", loss_rows)
    summary["evidence_artifacts"] = {name: {"file": f"{PREFIX}{name}.json", "present": "missing_file" not in value}
                                     for name, value in evidence.items()}
    summary.update(evaluate_gates(summary, evidence))
    _write_json(summary_path, summary)
    render_report(summary, evidence, run_dir, args.report.resolve())
    print(json.dumps(_json_safe({"report": str(args.report.resolve()), "summary": str(summary_path),
                                 "status": summary["report_status"], "decision": summary["decision"]}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
