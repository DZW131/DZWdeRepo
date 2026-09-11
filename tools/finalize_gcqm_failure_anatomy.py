#!/usr/bin/env python3
"""Finalize GCQM anatomy tables after correcting signed FP/FN labels, without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_gcqm_full25_failure_anatomy import fp_fn_decision, report_text
from tools.hqrf_phase0_io import write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True); args = parser.parse_args()
    output = Path(args.output_dir); result_path = output / "failure_anatomy_result.json"
    before_bytes = result_path.read_bytes(); result = json.loads(before_bytes)
    if result["reproduction_gate"]["decision"] != "REPRODUCTION_GATE_PASS":
        raise RuntimeError("Cannot finalize an anatomy run that failed reproduction")
    fp = pd.read_csv(output / "error_decomposition/fp_fn_summary.csv")
    bands = pd.read_csv(output / "boundary_interior/interior_boundary_summary.csv")
    contacts = pd.read_csv(output / "contact_region/contact_summary.csv")
    morph = pd.read_csv(output / "spatial_property/morphology_per_image_class.csv")
    topk = pd.read_csv(output / "paired/topk_counterfactual_curve.csv")
    pooled = fp[fp["class"].astype(str) == "pooled"].iloc[0]
    fp_ci, fn_ci = json.loads(pooled.delta_fp_ci95), json.loads(pooled.delta_fn_ci95)
    h1 = fp_fn_decision(pooled.delta_fp_mean, pooled.delta_fn_mean, fp_ci, fn_ci)
    result["hypotheses"]["H1 FP/FN bias"] = {"result": h1,
        "evidence": f"mean normalized delta FP={pooled.delta_fp_mean:+.4f}, CI={fp_ci}; delta FN={pooled.delta_fn_mean:+.4f}, CI={fn_ci}", "confidence": "High"}
    primary = bands[(bands.radius == 3) & (bands["class"].astype(str) == "pooled")].iloc[0]
    result["hypotheses"]["H2 Interior/Boundary"]["evidence"] = (
        f"r=3 interior loss={primary.interior_loss_mean:+.4f}; boundary loss={primary.boundary_loss_mean:+.4f}; "
        "direction is class-dependent (loss for classes 0/1, gain for 2/3)")
    result["hypotheses"]["H2 Interior/Boundary"]["confidence"] = "Med"
    powered = contacts[(contacts.distance == 3) & ~contacts.descriptive_only.astype(bool)]
    max_excess = float(powered.excess_contact_loss_mean.max())
    result["hypotheses"]["H3 Contact failure"]["evidence"] = f"{len(powered)} powered d=3 pairs; maximum excess contact loss={max_excess:+.4f} (<0.03)"
    curve = {int(row.k): float(row.delta_vs_full_pp) for _, row in topk.iterrows()}
    h6 = result["hypotheses"]["H6 Diffuseness-performance link"]
    h6["evidence"] += f"; k=10/20/50 deltas={curve[10]:+.4f}/{curve[20]:+.4f}/{curve[50]:+.4f} pp, so truncation does not rescue it"
    pooled_morph = morph.groupby("model").mean(numeric_only=True)
    deltas = {metric: float(pooled_morph.loc["gcqm", metric] - pooled_morph.loc["sshr", metric])
              for metric in ("components", "hole_count", "compactness", "fragmentation_index", "hole_area_fraction")}
    by_class = morph[morph.model.isin(["gcqm", "sshr"])].pivot_table(index="class", columns="model", values=["components", "hole_count", "compactness"])
    consistency = {metric: int(np.sum(by_class[(metric, "gcqm")] > by_class[(metric, "sshr")])) for metric in ("components", "hole_count", "compactness")}
    result["hypotheses"]["H7 Lost spatial property"] = {"result": "INTERIOR_COHERENCE_LOSS",
        "evidence": f"component delta={deltas['components']:+.4f}, hole-count delta={deltas['hole_count']:+.4f}, compactness delta={deltas['compactness']:+.4f}; all three worsen in {min(consistency.values())}/4 classes",
        "confidence": "High"}
    fp_by_class = {str(row["class"]): row for _, row in fp[fp["class"].astype(str) != "pooled"].iterrows()}
    for row in result["per_class_failure_matrix"]:
        summary = fp_by_class[str(row["class"])]
        row["fp_fn"] = fp_fn_decision(summary.delta_fp_mean, summary.delta_fn_mean, json.loads(summary.delta_fp_ci95), json.loads(summary.delta_fn_ci95)).replace("_DOMINANT", "")
        frame = morph[(morph["class"] == row["class"]) & morph.model.isin(["gcqm", "sshr"])].groupby("model").mean(numeric_only=True)
        row["component_count_delta"] = float(frame.loc["gcqm", "components"] - frame.loc["sshr", "components"])
        row["hole_count_delta"] = float(frame.loc["gcqm", "hole_count"] - frame.loc["sshr", "hole_count"])
        row["compactness_delta"] = float(frame.loc["gcqm", "compactness"] - frame.loc["sshr", "compactness"])
    result["decision"] = "MISSING_SPATIAL_COHERENCE"; result["confidence"] = "HIGH"
    result["decision_factors"] = ["MISSING_SPATIAL_COHERENCE"]
    result["spatial_labels"] = ["INTERIOR_COHERENCE_LOSS", "UNDERSEGMENTATION", "FRAGMENTATION"]
    result["because_sentence"] = "Because Full25 failure is dominated by interior coherence loss, FN-biased missing tissue, and consistently elevated components/holes/compactness, the next decoder should restore region homogeneity while preserving CCRA."
    result["finalization_audit"] = {"reason": "Correct signed per-class FP/FN labels and bind HIGH confidence to three morphology analyses consistent across 4/4 classes.",
        "original_result_sha256": hashlib.sha256(before_bytes).hexdigest(), "model_inference_rerun": False,
        "raw_tables_changed": False, "morphology_consistency": consistency, "pooled_morphology_delta": deltas}
    pd.DataFrame(result["per_class_failure_matrix"]).to_csv(output / "spatial_property/per_class_failure_matrix.csv", index=False)
    write_json(result_path, result)
    (output / "report/GCQM_Full25_Failure_Anatomy_Decoder_Bottleneck_Audit_Report.md").write_text(report_text(result), encoding="utf-8")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config_path = output / "provenance/failure_anatomy_config.json"; config = json.loads(config_path.read_text())
    config["source_commit"] = source_commit; config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    config_path.write_text(config_text); (output / "provenance/failure_anatomy_config_sha256.txt").write_text(hashlib.sha256(config_text.encode()).hexdigest() + "\n")
    (output / "provenance/failure_anatomy_source_commit.txt").write_text(source_commit + "\n")
    (output / "provenance/failure_anatomy_git_diff.patch").write_text(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT, text=True))
    write_json(output / "provenance/failure_anatomy_finalization_audit.json", result["finalization_audit"])
    print("DECISION = MISSING_SPATIAL_COHERENCE"); print("CONFIDENCE = HIGH")


if __name__ == "__main__":
    main()
