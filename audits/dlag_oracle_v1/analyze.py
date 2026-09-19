"""Evaluate DLAG oracle headroom, safety, overlap, bootstrap, and decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.eval_gcqm_full25_bcss_seed42 import scores_from_confusion

ALPHAS = np.asarray([0., .25, .5, 1., 1.5, 2., 3., 4.])
BASELINE_REFERENCE = 0.6557244403737567
SSHR_REFERENCE = 0.666967
RESAMPLES = 2000
EPS = 1e-12


def metrics(histograms: np.ndarray) -> dict:
    return scores_from_confusion(histograms.sum(axis=0))


def paired_bootstrap(confusions: dict[str, np.ndarray], alpha: np.ndarray) -> dict:
    rng = np.random.default_rng(42)
    base = confusions["baseline"]
    names = ("oracle_A1", "oracle_A2", "oracle_B1", "oracle_B2", "oracle_AB")
    values = {name: [] for name in names}
    alpha_best = []
    count = len(base)
    for _ in range(RESAMPLES):
        sample = rng.integers(0, count, count)
        base_miou = metrics(base[sample])["mIoU"]
        for name in names:
            values[name].append((metrics(confusions[name][sample])["mIoU"]-base_miou)*100)
        # Secondary image-level oracle: per sampled image choose alpha with maximum correct pixels.
        per_image_correct = np.diagonal(alpha[:, sample], axis1=2, axis2=3).sum(-1)
        choice = np.argmax(per_image_correct, axis=0)
        selected = alpha[choice, sample]
        alpha_best.append((metrics(selected)["mIoU"]-base_miou)*100)
    values["image_level_A"] = alpha_best
    return {name: {"delta_pp_95ci": [float(np.quantile(value, .025)),
                                      float(np.quantile(value, .975))],
                   "delta_pp_bootstrap_median": float(np.median(value))}
            for name, value in values.items()}


def component_bootstrap(frame: pd.DataFrame) -> dict:
    rng = np.random.default_rng(42)
    valid = frame.valid_area.to_numpy(np.float64)
    correct = {name: frame[f"{name}_correct_pixels"].to_numpy(np.float64)
               for name in ("A", "B", "AB")}
    values = {name: [] for name in correct}
    unrecoverable = []
    for _ in range(RESAMPLES):
        sample = rng.integers(0, len(frame), len(frame))
        denominator = max(valid[sample].sum(), EPS)
        for name in correct:
            values[name].append(float(correct[name][sample].sum()/denominator))
        unrecoverable.append(float((valid[sample]-correct["AB"][sample]).sum()/denominator))
    values["unrecoverable"] = unrecoverable
    return {name: {"fraction_95ci": [float(np.quantile(value, .025)),
                                      float(np.quantile(value, .975))],
                   "fraction_median": float(np.median(value))}
            for name, value in values.items()}


def subgroup(frame: pd.DataFrame) -> dict:
    valid = max(float(frame.valid_area.sum()), EPS)
    area = max(float(frame.area.sum()), EPS)
    return {"components": int(len(frame)), "support_area": int(frame.area.sum()),
        "valid_area": int(frame.valid_area.sum()),
        "A_recovered_fraction": float(frame.A_correct_pixels.sum()/valid),
        "B_recovered_fraction": float(frame.B_correct_pixels.sum()/valid),
        "AB_recovered_fraction": float(frame.AB_correct_pixels.sum()/valid),
        "AB_unrecoverable_fraction": float((frame.valid_area-frame.AB_correct_pixels).sum()/valid),
        "A_majority_component_rate": float(frame.A_majority_recovered.mean()) if len(frame) else None,
        "B_majority_component_rate": float(frame.B_majority_recovered.mean()) if len(frame) else None,
        "AB_majority_component_rate": float(frame.AB_majority_recovered.mean()) if len(frame) else None,
        "deep_local_failure_area_fraction": float(frame.loc[frame.deep_local_failure, "area"].sum()/area),
        "gate_excluded_area_fraction": float(frame.loc[frame.gate_excluded, "area"].sum()/area)}


def run(output: Path) -> None:
    gate = json.loads((output / "00_reproduction_gate.json").read_text())
    integrity = json.loads((output / "gate_integrity.json").read_text())
    if not gate.get("pass") or not integrity.get("pass"):
        raise AssertionError("DLAG gates must pass before analysis")
    archive = np.load(output / "metrics/confusions.npz")
    alpha_confusions = archive["alpha_confusions"]
    method_names = ("baseline", "oracle_A1", "oracle_A2", "oracle_B1", "oracle_B2", "oracle_AB")
    confusions = {name: archive[name] for name in method_names}
    score = {name: metrics(value) for name, value in confusions.items()}
    if abs(score["baseline"]["mIoU"]-BASELINE_REFERENCE) > 1e-12:
        raise AssertionError("Oracle bank baseline does not reproduce frozen HQMR")
    alpha_scores = [metrics(alpha_confusions[index]) for index in range(len(ALPHAS))]
    safety = pd.read_csv(output / "metrics/oracle_safety.csv")
    alpha_safety = pd.read_csv(output / "metrics/tp_safety_by_alpha.csv")
    curve = []
    for index, (alpha, alpha_score) in enumerate(zip(ALPHAS, alpha_scores)):
        subset = alpha_safety[alpha_safety.alpha == alpha]
        curve.append({"alpha": alpha, "mIoU": alpha_score["mIoU"], "mDice": alpha_score["mDice"],
            "delta_vs_HQMR_pp": (alpha_score["mIoU"]-score["baseline"]["mIoU"])*100,
            "harmed_correct_area": int(subset.harmed_correct_area.sum()),
            "newly_correct_area": int(subset.newly_correct_area.sum()),
            **{f"IoU_C{c}": alpha_score["class_iou"][str(c)] for c in range(4)}})
    pd.DataFrame(curve).to_csv(output / "metrics/global_alpha_curve.csv", index=False)

    # Secondary image-level alpha oracle without pixel/component-wise alpha selection.
    per_image_correct = np.diagonal(alpha_confusions, axis1=2, axis2=3).sum(-1)
    image_choice = np.argmax(per_image_correct, axis=0)
    image_selected = alpha_confusions[image_choice, np.arange(alpha_confusions.shape[1])]
    score["image_level_A"] = metrics(image_selected)
    pd.DataFrame({"image_id": archive["image_ids"],
                  "best_alpha": ALPHAS[image_choice]}).to_csv(
        output / "arbitration_oracle/image_best_alpha.csv", index=False)

    frame = pd.read_csv(output / "joint_oracle/recoverability_table.csv")
    if len(frame) != 4440:
        raise AssertionError("Expected 4440 frozen M1 components")
    area_edges = np.quantile(frame.area, [.25, .50, .75])
    frame["area_quartile"] = np.searchsorted(area_edges, frame.area, side="right")+1
    frame.to_csv(output / "joint_oracle/recoverability_table.csv", index=False)
    groups = {"overall": frame, "high_purity": frame[frame.purity >= .70],
              "large_Q4": frame[frame.area_quartile == 4],
              "high_purity_large": frame[(frame.purity >= .70) & (frame.area_quartile == 4)],
              "gate_preserved": frame[~frame.gate_excluded],
              "gate_excluded": frame[frame.gate_excluded],
              "E3_corrective_subset": frame[frame.deep_local_failure]}
    group_summary = {name: subgroup(value) for name, value in groups.items()}
    (output / "metrics/purity_size_oracle.json").write_text(
        json.dumps(group_summary, indent=2), encoding="utf-8")

    overlap_rows = []
    total_area = max(float(frame.area.sum()), EPS)
    for name, subset in frame.groupby("failure_overlap"):
        value = subgroup(subset)
        overlap_rows.append({"failure_overlap": name,
            "area_fraction": float(subset.area.sum()/total_area), **value})
    pd.DataFrame(overlap_rows).to_csv(output / "metrics/failure_overlap.csv", index=False)

    total_valid = max(float(frame.valid_area.sum()), EPS)
    pixel_attribution = {
        "A_only_fraction": float(frame.A_only_correct_pixels.sum()/total_valid),
        "B_only_fraction": float(frame.B_only_correct_pixels.sum()/total_valid),
        "both_individual_fraction": float(frame.both_individual_correct_pixels.sum()/total_valid),
        "joint_only_fraction": float(frame.joint_only_correct_pixels.sum()/total_valid),
        "AB_recovered_fraction": float(frame.AB_correct_pixels.sum()/total_valid),
        "AB_unrecoverable_fraction": float(frame.AB_unrecoverable_pixels.sum()/total_valid)}
    category_rows = []
    for name, subset in frame.groupby("recoverability"):
        category_rows.append({"category": name, "components": len(subset),
            "component_fraction": len(subset)/len(frame), "support_area": int(subset.area.sum()),
            "support_area_fraction": float(subset.area.sum()/total_area)})
    pd.DataFrame(category_rows).to_csv(output / "metrics/recoverability_categories.csv", index=False)

    best_alpha = frame.groupby("best_alpha_A").agg(components=("component_id", "size"),
        support_area=("area", "sum"), valid_area=("valid_area", "sum"),
        recovered_pixels=("A_correct_pixels", "sum")).reset_index()
    best_alpha["support_area_fraction"] = best_alpha.support_area / total_area
    best_alpha.to_csv(output / "metrics/best_alpha_histogram.csv", index=False)
    response = frame.groupby("response_category").agg(components=("component_id", "size"),
        support_area=("area", "sum")).reset_index()
    response["support_area_fraction"] = response.support_area/total_area
    response.to_csv(output / "metrics/margin_response_categories.csv", index=False)

    per_class = []
    for cls in range(4):
        row = {"class": cls}
        for name in ("baseline", "oracle_A1", "oracle_B1", "oracle_AB"):
            row[f"{name}_IoU"] = score[name]["class_iou"][str(cls)]
        row["A_gain_pp"] = (row["oracle_A1_IoU"]-row["baseline_IoU"])*100
        row["B_gain_pp"] = (row["oracle_B1_IoU"]-row["baseline_IoU"])*100
        row["AB_gain_pp"] = (row["oracle_AB_IoU"]-row["baseline_IoU"])*100
        per_class.append(row)
    per_class_frame = pd.DataFrame(per_class)
    per_class_frame.to_csv(output / "metrics/per_class_oracle.csv", index=False)
    positive = per_class_frame.AB_gain_pp.clip(lower=0)
    concentration = float(positive.max()/max(positive.sum(), EPS))

    delta = {"A": (score["oracle_A1"]["mIoU"]-score["baseline"]["mIoU"])*100,
             "A2": (score["oracle_A2"]["mIoU"]-score["baseline"]["mIoU"])*100,
             "B": (score["oracle_B1"]["mIoU"]-score["baseline"]["mIoU"])*100,
             "B2": (score["oracle_B2"]["mIoU"]-score["baseline"]["mIoU"])*100,
             "AB": (score["oracle_AB"]["mIoU"]-score["baseline"]["mIoU"])*100,
             "image_A": (score["image_level_A"]["mIoU"]-score["baseline"]["mIoU"])*100}
    synergy = delta["AB"]-delta["A"]-delta["B"]
    relation = "complementary_synergy" if synergy > .05 else (
        "recovery_overlap" if synergy < -.05 else "approximately_independent")
    bootstrap = paired_bootstrap(confusions, alpha_confusions)
    bootstrap["component_recovery"] = component_bootstrap(frame)
    (output / "metrics/bootstrap_ci.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    if delta["AB"] < 1:
        decision = "NOGO"; boundary = 1.; confident = bootstrap["oracle_AB"]["delta_pp_95ci"][1] < 1
    elif delta["AB"] < 1.5:
        decision = "WEAK_GO"; boundary = 1.; confident = bootstrap["oracle_AB"]["delta_pp_95ci"][0] >= 1
    elif delta["AB"] < 2:
        decision = "ARCHITECTURE_GO"; boundary = 1.5; confident = bootstrap["oracle_AB"]["delta_pp_95ci"][0] >= 1.5
    elif (score["oracle_AB"]["mIoU"]-SSHR_REFERENCE)*100 >= .5:
        decision = "STRONG_ARCHITECTURE_GO"; boundary = 2.; confident = bootstrap["oracle_AB"]["delta_pp_95ci"][0] >= 2
    else:
        decision = "ARCHITECTURE_GO"; boundary = 1.5; confident = bootstrap["oracle_AB"]["delta_pp_95ci"][0] >= 1.5
    confidence = "HIGH" if confident and concentration <= .80 else "MEDIUM"
    safety_summary = {}
    for name, subset in safety.groupby("method"):
        safety_summary[name] = {key: int(subset[key].sum()) for key in
            ("baseline_correct_area", "preserved_correct_area", "harmed_correct_area", "newly_correct_area")}
    oracle_metrics = {name: score[name] for name in score}
    oracle_metrics["delta_pp"] = delta
    oracle_metrics["safety"] = safety_summary
    (output / "metrics/oracle_metrics.json").write_text(json.dumps(oracle_metrics, indent=2), encoding="utf-8")
    for name, payload in (("arbitration_oracle/oracle_A_metrics.json",
                           {"A1": score["oracle_A1"], "A2": score["oracle_A2"],
                            "image_level_A": score["image_level_A"], "delta_pp": {"A1": delta["A"], "A2": delta["A2"], "image": delta["image_A"]}}),
                          ("gate_oracle/oracle_B_metrics.json",
                           {"B1": score["oracle_B1"], "B2": score["oracle_B2"],
                            "delta_pp": {"B1": delta["B"], "B2": delta["B2"]}}),
                          ("joint_oracle/oracle_AB_metrics.json",
                           {"AB": score["oracle_AB"], "delta_pp": delta["AB"],
                            "pixel_recovery": pixel_attribution})):
        (output / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    synergy_payload = {"delta_A_pp": delta["A"], "delta_B_pp": delta["B"],
        "delta_AB_pp": delta["AB"], "synergy_pp": synergy, "relation": relation}
    (output / "metrics/synergy.json").write_text(json.dumps(synergy_payload, indent=2), encoding="utf-8")
    decision_payload = {"decision": decision, "confidence": confidence,
        "HQMR": score["baseline"]["mIoU"], "SSHR": SSHR_REFERENCE,
        "oracle_A": score["oracle_A1"]["mIoU"], "oracle_B": score["oracle_B1"]["mIoU"],
        "oracle_AB": score["oracle_AB"]["mIoU"], "delta_pp": delta,
        "AB_vs_SSHR_pp": (score["oracle_AB"]["mIoU"]-SSHR_REFERENCE)*100,
        "synergy_pp": synergy, "relation": relation, "class_gain_concentration": concentration,
        "class_specific": concentration > .80, "pixel_attribution": pixel_attribution,
        "decision_boundary_pp": boundary, "parameter_updates": 0}
    (output / "metrics/decision_matrix.json").write_text(
        json.dumps(decision_payload, indent=2), encoding="utf-8")
    print(json.dumps({"event": "analysis_done", **decision_payload}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
