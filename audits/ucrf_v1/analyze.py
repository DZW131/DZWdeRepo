"""Pre-registered UCRF event taxonomy, matched controls, bootstrap, decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-8
RESAMPLES = 2000


def weighted_rate(mask: np.ndarray, area: np.ndarray) -> float:
    return float(np.sum(mask.astype(np.float64) * area) / max(float(area.sum()), EPS))


def match_tp(frame: pd.DataFrame, m1_indices: np.ndarray) -> tuple[pd.DataFrame, dict]:
    tp = frame[frame.cohort == "TP_candidate"]
    matches = []
    for index in m1_indices:
        row = frame.iloc[index]
        candidates = tp[(tp.true_class == row.true_class) & (tp.area_quartile == row.area_quartile)]
        if candidates.empty:
            candidates = tp[tp.true_class == row.true_class]
        if candidates.empty:
            continue
        same_purity = candidates[(candidates.purity >= .70) == (row.purity >= .70)]
        purity_match = not same_purity.empty
        if purity_match:
            candidates = same_purity
        same_image = candidates[candidates.image_id == row.image_id]
        if not same_image.empty:
            candidates = same_image
        logdiff = np.abs(np.log1p(candidates.area.to_numpy()) - np.log1p(row.area))
        puritydiff = np.abs(candidates.purity.to_numpy() - row.purity)
        closest = np.argmin(logdiff + .25 * puritydiff)
        control = candidates.iloc[closest]
        matches.append({"m1_index": int(index), "tp_index": int(control.name),
                        "same_image": bool(control.image_id == row.image_id),
                        "same_area_quartile": bool(control.area_quartile == row.area_quartile),
                        "same_purity_stratum": bool(purity_match),
                        "log_area_difference": float(logdiff[closest]),
                        "purity_difference": float(puritydiff[closest])})
    result = pd.DataFrame(matches)
    coverage = {"M1": int(len(m1_indices)), "matched": int(result.m1_index.nunique()),
                "TP_candidates": int(len(tp)), "unique_TP_used": int(result.tp_index.nunique()),
                "same_image_fraction": float(result.same_image.mean()),
                "same_quartile_fraction": float(result.same_area_quartile.mean()),
                "same_purity_stratum_fraction": float(result.same_purity_stratum.mean()),
                "median_log_area_difference": float(result.log_area_difference.median())}
    return result, coverage


def flags_for(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    m5 = frame.margin_logits5.to_numpy()
    mu5 = frame.margin_upsampled5.to_numpy()
    m4 = frame.margin_logits4.to_numpy()
    mu4 = frame.margin_upsampled4.to_numpy()
    m3 = frame.margin_logits3.to_numpy()
    delta_d = frame.direct4_effect_margin.to_numpy()
    proxy_d = frame.margin_direct4_standalone.to_numpy()
    confidence = frame.confidence_logits5.to_numpy()
    high_edge = frame.attrs["confidence_high_edge"]
    return {
        "deep_wrong": m5 < 0,
        "F1": (mu5 < 0) & (m4 < 0),
        "F2": (mu5 > 0) & (m4 < 0),
        "F3": (mu5 < 0) & (m4 > 0),
        "F4": (mu5 > 0) & (m4 > 0),
        "F1_native": (m5 < 0) & (m4 < 0),
        "F2_native": (m5 > 0) & (m4 < 0),
        "Q1": (m5 > 0) & (proxy_d > 0),
        "Q2": (m5 > 0) & (proxy_d < 0),
        "Q3": (m5 < 0) & (proxy_d > 0),
        "Q4": (m5 < 0) & (proxy_d < 0),
        "E1": (m5 > 0) & (delta_d > 0),
        "E2": (m5 > 0) & (delta_d < 0),
        "E3": (m5 < 0) & (delta_d > 0),
        "E4": (m5 < 0) & (delta_d < 0),
        "corrective_available": (m5 < 0) & (delta_d > 0),
        "corrective_suppressed": (m5 < 0) & (delta_d > 0) & (m4 < 0),
        "direct4_proxy_true": frame.margin_direct4_standalone.to_numpy() > 0,
        "direct4_proxy_available": (m5 < 0) & (delta_d > 0) &
                                   (frame.margin_direct4_standalone.to_numpy() > 0),
        "direct4_proxy_suppressed": (m5 < 0) & (delta_d > 0) & (m4 < 0) &
                                    (frame.margin_direct4_standalone.to_numpy() > 0),
        "stage3_amp": (mu4 < 0) & (m3 < mu4),
        "stage3_repair": (mu4 < 0) & (m3 > 0),
        "stage3_flip": (mu4 > 0) & (m3 < 0),
        "high_confidence_wrong": (m5 < 0) & (confidence >= high_edge),
        "rival_persistence": frame.dynamic_rival_logits5.to_numpy() == frame.predicted_class.to_numpy(),
        "direct4_rival_persistence": frame.dynamic_rival_direct4_standalone.to_numpy() == frame.predicted_class.to_numpy(),
        "region_wide_deep_error": frame.rival_pixel_fraction_logits5.to_numpy() >= .70,
        "gate_missing_true": ~frame.true_label_present.to_numpy(bool),
        "calibration_flip": frame.calibration_flip.to_numpy(bool),
        "final_wrong_competition": frame.margin_final_cam.to_numpy() < 0,
    }


def summarize(frame: pd.DataFrame, flags: dict[str, np.ndarray],
              index: np.ndarray) -> dict:
    if not len(index):
        return {"components": 0, "area_pixels": 0}
    area = frame.iloc[index].area.to_numpy(np.float64)
    result = {"components": int(len(index)), "area_pixels": int(area.sum())}
    for name, mask in flags.items():
        chosen = mask[index]
        result[f"{name}_component_rate"] = float(chosen.mean())
        result[f"{name}_area_rate"] = weighted_rate(chosen, area)
    available = flags["corrective_available"][index]
    suppressed = flags["corrective_suppressed"][index]
    result["CSR_component"] = float(suppressed.sum()/max(available.sum(), 1))
    result["CSR_area"] = float(area[suppressed].sum()/max(area[available].sum(), EPS))
    result["correction_available_count"] = int(available.sum())
    proxy_available = flags["direct4_proxy_available"][index]
    proxy_suppressed = flags["direct4_proxy_suppressed"][index]
    result["direct4_proxy_CSR_area"] = float(area[proxy_suppressed].sum()/
                                              max(area[proxy_available].sum(), EPS))
    wrong_persist = (frame.iloc[index].margin_upsampled4.to_numpy() < 0) & (
        frame.iloc[index].margin_logits3.to_numpy() < 0)
    ear = (np.abs(frame.iloc[index].normalized_margin_logits3.to_numpy()) /
           np.maximum(np.abs(frame.iloc[index].normalized_margin_upsampled4.to_numpy()), EPS))
    result["EAR_median_persisting_wrong"] = float(np.median(ear[wrong_persist])) if wrong_persist.any() else None
    for field in ("margin_logits5", "margin_upsampled5", "direct4_effect_margin",
                  "margin_logits4", "query4_isolated_effect_margin", "margin_logits3",
                  "rival_pixel_fraction_logits5", "rival_pixel_fraction_logits4",
                  "rival_pixel_fraction_logits3", "direct4_contribution_ratio"):
        result[f"median_{field}"] = float(np.nanmedian(frame.iloc[index][field]))
    return result


def bootstrap(frame: pd.DataFrame, flags: dict[str, np.ndarray], groups: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(42)
    names = ("deep_wrong", "F1", "F2", "F3", "F4", "corrective_available",
             "direct4_proxy_true", "direct4_proxy_available", "direct4_proxy_suppressed",
             "stage3_amp", "stage3_repair", "stage3_flip", "high_confidence_wrong",
             "rival_persistence", "region_wide_deep_error", "gate_missing_true",
             "calibration_flip", "final_wrong_competition")
    result = {}
    for group_name, indices in groups.items():
        if not len(indices):
            continue
        area = frame.iloc[indices].area.to_numpy(np.float64)
        values = {name: flags[name][indices].astype(np.float64) for name in names}
        available = flags["corrective_available"][indices].astype(np.float64)
        suppressed = flags["corrective_suppressed"][indices].astype(np.float64)
        records = {name: [] for name in (*names, "CSR")}
        for _ in range(RESAMPLES):
            sampled = rng.integers(0, len(indices), len(indices))
            weights = area[sampled]
            denominator = weights.sum()
            for name in names:
                records[name].append(float(np.dot(weights, values[name][sampled])/denominator))
            available_area = np.dot(weights, available[sampled])
            records["CSR"].append(float(np.dot(weights, suppressed[sampled]) /
                                        max(available_area, EPS)))
        result[group_name] = {name: [float(np.quantile(array, .025)),
                                     float(np.quantile(array, .975))]
                              for name, array in records.items()}
    return result


def paired_controls(frame: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in ("logits5", "logits4", "logits3", "final_cam"):
        for pair in matches.itertuples():
            m1, tp = frame.iloc[pair.m1_index], frame.iloc[pair.tp_index]
            true, rival = int(m1.true_class), int(m1.predicted_class)
            m1_margin = float(m1[f"score_{stage}_C{true}"] - m1[f"score_{stage}_C{rival}"])
            tp_margin = float(tp[f"score_{stage}_C{true}"] - tp[f"score_{stage}_C{rival}"])
            rows.append({"stage": stage, "m1_index": pair.m1_index, "tp_index": pair.tp_index,
                         "m1_margin": m1_margin, "tp_margin": tp_margin,
                         "m1_wrong": bool(m1_margin < 0), "tp_wrong": bool(tp_margin < 0),
                         "same_image": pair.same_image, "same_area_quartile": pair.same_area_quartile,
                         "same_purity_stratum": pair.same_purity_stratum})
    return pd.DataFrame(rows)


def decide(overall: dict, high_large: dict, intervals: dict) -> dict:
    d1 = (overall["deep_wrong_area_rate"] >= .60 and
          high_large["deep_wrong_area_rate"] >= .60 and
          overall["F1_area_rate"] >= .60 and overall["F2_area_rate"] < .15)
    d2 = overall["F2_area_rate"] >= .30
    d3 = (overall["corrective_available_area_rate"] >= .30 and overall["CSR_area"] >= .60)
    d4 = (overall["stage3_flip_area_rate"] >= .30 or
          (overall["stage3_amp_area_rate"] >= .60 and
           overall["EAR_median_persisting_wrong"] is not None and
           overall["EAR_median_persisting_wrong"] > 1 and
           overall["F1_area_rate"] < .60))
    if d3:
        primary = "DEEP_DOMINANCE_SUPPRESSES_CORRECTION"
        keys = ("corrective_available", "CSR")
        cutoff = (.30, .60)
    elif d2:
        primary = "DIRECT4_INDUCED_FLIP"
        keys, cutoff = ("F2",), (.30,)
    elif d1:
        primary = "DEEP_SEMANTIC_MISASSIGNMENT"
        keys, cutoff = ("deep_wrong", "F1"), (.60, .60)
    elif d4:
        primary = "QUERY_STAGE_AMPLIFICATION"
        keys, cutoff = ("stage3_flip",), (.30,)
    else:
        primary = "MIXED_RESPONSIBILITY_FAILURE"
        keys, cutoff = (), ()
    if primary == "MIXED_RESPONSIBILITY_FAILURE":
        confidence = "LOW"
    else:
        clears = all(intervals["overall"][key][0] > threshold
                     for key, threshold in zip(keys, cutoff))
        if primary == "DEEP_SEMANTIC_MISASSIGNMENT":
            clears &= intervals["high_purity_large"]["deep_wrong"][0] > .60
            clears &= intervals["overall"]["F2"][1] < .15
        confidence = "HIGH" if clears else "MEDIUM"
    secondary = []
    if overall["high_confidence_wrong_area_rate"] >= .30:
        secondary.append("EARLY_HIGH_CONFIDENCE_WRONG")
    if overall["corrective_available_area_rate"] >= .30:
        secondary.append("LOCAL_CORRECTIVE_EVIDENCE_EXISTS")
    if overall["rival_persistence_area_rate"] >= .60:
        secondary.append("RIVAL_PERSISTENCE")
    if overall["region_wide_deep_error_area_rate"] >= .60:
        secondary.append("REGION_WIDE_ERROR")
    return {"decision": primary, "confidence": confidence, "secondary_labels": secondary,
            "criteria": {"D1": d1, "D2": d2, "D3": d3, "D4": d4},
            "priority": "D3 > D2 > D1 > D4 > D5 when criteria overlap",
            "parameter_updates": 0,
            "note": "Class competition uses frozen W*sigmoid(query logits); direct4 effect is nonlinear counterfactual difference, not direct4 as a tissue logit."}


def run(output: Path) -> None:
    gate = json.loads((output / "00_reproduction_gate.json").read_text())
    if not gate.get("pass"):
        raise AssertionError("UCRF reproduction gate did not pass")
    frame = pd.read_parquet(output / "metrics/component_stage_margins.parquet").reset_index(drop=True)
    edges = json.loads((output / "metrics/confidence_tertiles.json").read_text())["thresholds"]
    frame.attrs["confidence_high_edge"] = edges[1]
    m1_indices = np.flatnonzero(frame.cohort.to_numpy() == "M1")
    if len(m1_indices) != 4440:
        raise AssertionError("M1 count mismatch")
    q = np.quantile(frame.iloc[m1_indices].area, [.25, .50, .75])
    frame["area_quartile"] = np.searchsorted(q, frame.area.to_numpy(), side="right")+1
    flags = flags_for(frame)
    frame["flip_event"] = np.select([flags[key] for key in ("F1", "F2", "F3", "F4")],
                                    ["F1_already_wrong", "F2_correct_to_wrong",
                                     "F3_wrong_to_correct", "F4_correct_persists"],
                                    default="zero_margin")
    frame["quadrant"] = np.select([flags[key] for key in ("Q1", "Q2", "Q3", "Q4")],
                                  ["Q1_deep_true_direct_true", "Q2_deep_true_direct_rival",
                                   "Q3_deep_rival_direct_true", "Q4_deep_rival_direct_rival"],
                                  default="zero_margin")
    frame["effect_quadrant"] = np.select([flags[key] for key in ("E1", "E2", "E3", "E4")],
        ["E1_deep_true_effect_true", "E2_deep_true_effect_rival",
         "E3_deep_rival_effect_true", "E4_deep_rival_effect_rival"], default="zero_margin")
    frame["stage3_event"] = np.select([flags[key] for key in ("stage3_repair", "stage3_flip", "stage3_amp")],
                                      ["Q_REPAIR", "Q_FLIP", "Q_AMP"], default="Q_STABLE_OR_OTHER")
    frame["confidence_tertile"] = np.searchsorted(edges, frame.confidence_logits5.to_numpy(), side="right")+1
    frame["state5"] = np.where(frame.normalized_margin_logits5 > .05, "TRUE_DOMINANT",
                                np.where(frame.normalized_margin_logits5 < -.05, "RIVAL_DOMINANT", "AMBIGUOUS"))
    frame["state4"] = np.where(frame.normalized_margin_logits4 > .05, "TRUE_DOMINANT",
                                np.where(frame.normalized_margin_logits4 < -.05, "RIVAL_DOMINANT", "AMBIGUOUS"))
    frame["state3"] = np.where(frame.normalized_margin_logits3 > .05, "TRUE_DOMINANT",
                                np.where(frame.normalized_margin_logits3 < -.05, "RIVAL_DOMINANT", "AMBIGUOUS"))
    frame.to_parquet(output / "metrics/component_event_table.parquet", index=False, compression="zstd")
    frame[["image_id", "component_id", "cohort", "true_class", "predicted_class", "area",
           "flip_event", "quadrant", "stage3_event", "state5", "state4", "state3"]].to_csv(
        output / "metrics/flip_event_table.csv", index=False)
    frame[["image_id", "component_id", "cohort", "area", "quadrant", "effect_quadrant",
           "margin_logits5", "direct4_effect_margin", "margin_logits4",
           "direct4_contribution_ratio"]].to_csv(output / "metrics/direct4_quadrants.csv", index=False)
    frame[["image_id", "component_id", "cohort", "area", "confidence_logits5",
           "confidence_tertile", "state5", "margin_logits5"]].to_csv(
        output / "metrics/confidence_analysis.csv", index=False)
    spatial_columns = ["image_id", "component_id", "cohort", "area"] + [
        f"rival_pixel_fraction_{name}" for name in
        ("logits5", "upsampled5", "logits4", "upsampled4", "logits3", "final_cam")]
    frame["error_expansion_5_to_4"] = (frame.rival_pixel_fraction_logits4 -
                                        frame.rival_pixel_fraction_upsampled5)
    frame["error_expansion_4_to_3"] = (frame.rival_pixel_fraction_logits3 -
                                        frame.rival_pixel_fraction_upsampled4)
    frame[spatial_columns+["error_expansion_5_to_4", "error_expansion_4_to_3"]].to_csv(
        output / "metrics/spatial_error_coverage.csv", index=False)
    frame[["image_id", "component_id", "cohort", "area", "margin_logits5",
           "margin_upsampled5", "margin_logits4", "margin_upsampled4",
           "margin_logits3", "normalized_margin_logits5", "normalized_margin_logits4",
           "normalized_margin_logits3", "margin_logits5_core", "margin_logits5_boundary",
           "margin_logits4_core", "margin_logits4_boundary", "margin_logits3_core",
           "margin_logits3_boundary"]].to_csv(output / "metrics/pixel_stage_margins.csv", index=False)
    m1 = frame.iloc[m1_indices]
    groups = {"overall": m1_indices,
              "high_purity": m1_indices[m1.purity.to_numpy() >= .70],
              "low_purity": m1_indices[m1.purity.to_numpy() < .70],
              "large_Q4": m1_indices[m1.area_quartile.to_numpy() == 4],
              "high_purity_large": m1_indices[(m1.purity.to_numpy() >= .70) &
                                               (m1.area_quartile.to_numpy() == 4)],
              "true_gate_present": m1_indices[m1.true_label_present.to_numpy(bool)],
              "true_gate_missing": m1_indices[~m1.true_label_present.to_numpy(bool)]}
    for cls in range(4):
        groups[f"class_{cls}"] = m1_indices[m1.true_class.to_numpy() == cls]
    for quartile in range(1, 5):
        groups[f"size_Q{quartile}"] = m1_indices[m1.area_quartile.to_numpy() == quartile]
    summary = {name: summarize(frame, flags, ids) for name, ids in groups.items()}
    intervals = bootstrap(frame, flags, groups)
    (output / "metrics/bootstrap_ci.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    (output / "metrics/cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{ "cohort": key, **value} for key, value in summary.items()
                  if key.startswith("class_")]).to_csv(output / "metrics/per_class_metrics.csv", index=False)
    pd.DataFrame([{ "cohort": key, **summary[key]} for key in ("high_purity", "low_purity",
        "high_purity_large")]).to_csv(output / "metrics/purity_stratified.csv", index=False)
    pd.DataFrame([{ "cohort": key, **summary[key]} for key in
                  ("size_Q1", "size_Q2", "size_Q3", "size_Q4")]).to_csv(
        output / "metrics/size_stratified.csv", index=False)
    matches, coverage = match_tp(frame, m1_indices)
    matches.to_csv(output / "metrics/matched_tp_pairs.csv", index=False)
    paired = paired_controls(frame, matches)
    paired.to_csv(output / "metrics/matched_tp_controls.csv", index=False)
    control_rows = []
    for stage, subset in paired.groupby("stage"):
        control_rows.append({"stage": stage, "pairs": int(len(subset)),
            "m1_wrong_rate": float(subset.m1_wrong.mean()), "tp_wrong_rate": float(subset.tp_wrong.mean()),
            "m1_margin_median": float(subset.m1_margin.median()),
            "tp_margin_median": float(subset.tp_margin.median()),
            "m1_vs_tp_margin_auroc": float(roc_auc_score(
                np.r_[np.ones(len(subset)), np.zeros(len(subset))],
                np.r_[-subset.m1_margin.to_numpy(), -subset.tp_margin.to_numpy()]))})
    pd.DataFrame(control_rows).to_csv(output / "metrics/matched_tp_summary.csv", index=False)
    (output / "metrics/matching_manifest.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    availability = {"overall": summary["overall"]["corrective_available_area_rate"],
                    "high_purity_large": summary["high_purity_large"]["corrective_available_area_rate"]}
    suppression = {"overall_CSR_area": summary["overall"]["CSR_area"],
                   "overall_CSR_component": summary["overall"]["CSR_component"]}
    stage3 = {"amplification_area": summary["overall"]["stage3_amp_area_rate"],
              "repair_area": summary["overall"]["stage3_repair_area_rate"],
              "flip_area": summary["overall"]["stage3_flip_area_rate"],
              "median_EAR_persisting_wrong": summary["overall"]["EAR_median_persisting_wrong"]}
    for name, value in (("correction_availability", availability),
                        ("correction_suppression", suppression),
                        ("stage3_amplification", stage3)):
        (output / "metrics" / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    decision = decide(summary["overall"], summary["high_purity_large"], intervals)
    decision["matched_control_coverage"] = coverage
    (output / "metrics/decision_matrix.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"event": "analysis_done", "decision": decision["decision"],
                      "confidence": decision["confidence"],
                      "deep_wrong_area": summary["overall"]["deep_wrong_area_rate"],
                      "F1_area": summary["overall"]["F1_area_rate"],
                      "F2_area": summary["overall"]["F2_area_rate"],
                      "correction_available_area": availability["overall"],
                      "CSR_area": suppression["overall_CSR_area"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
