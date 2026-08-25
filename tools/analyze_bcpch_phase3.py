#!/usr/bin/env python3
"""Final matched validation analysis for EXP-BCPCH-003."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_bcch_phase1 import (
    ZoneIoUAccumulator,
    fmt,
    load_predictions,
    read_json,
)
from tools.wdch_common import PairedZoneAccumulator, sha256_file, write_json


VARIANTS = ("C0", "BC-CH", "CBCCH", "BCP-CH")
STAGES = ("56", "28_1", "28_2", "deep", "final")


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    directories = {
        "C0": Path(args.c0_dir),
        "BC-CH": Path(args.bcch_dir),
        "CBCCH": Path(args.cbcch_dir),
        "BCP-CH": Path(args.bcpch_dir),
    }
    completions = {
        name: read_json(directory / "complete.json")
        for name, directory in directories.items()
    }
    if completions["C0"].get("status") != "WDCH_MATCHED_BRANCH_COMPLETE":
        raise AssertionError("Invalid locked C0")
    if completions["BC-CH"].get("status") != "BCCH_MATCHED_COMPLETE":
        raise AssertionError("Invalid locked BC-CH")
    if completions["CBCCH"].get("status") != "CBCCH_MATCHED_COMPLETE":
        raise AssertionError("Invalid locked CBCCH")
    if completions["BCP-CH"].get("status") != "BCPCH_MATCHED_COMPLETE":
        raise AssertionError("BCP-CH run is incomplete")
    for name, completion in completions.items():
        final = completion["final_validation"]
        if completion.get("epochs") != [21, 22, 23, 24, 25]:
            raise AssertionError(f"{name}: wrong epoch contract")
        if completion.get("test_used") or final.get("test_used"):
            raise AssertionError(f"{name}: test use detected")
        if final.get("epoch") != 25:
            raise AssertionError(f"{name}: not Epoch25 FINAL")

    preflight = read_json(args.preflight)
    if preflight.get("status") != "BCPCH_PREFLIGHT_PASS":
        raise AssertionError("BCP-CH preflight did not pass")
    provenance = read_json(directories["BCP-CH"] / "provenance.json")
    common_sha = sha256_file(args.common_checkpoint)
    schedule_sha = sha256_file(args.schedule)
    if provenance["common_checkpoint_sha256"] != common_sha:
        raise AssertionError("Common checkpoint provenance differs")
    if provenance["schedule_sha256"] != schedule_sha:
        raise AssertionError("Schedule provenance differs")

    predictions = {
        name: load_predictions(directory / "predictions" / "epoch25_validation.npz")
        for name, directory in directories.items()
    }
    reference = predictions["C0"]
    for name, value in predictions.items():
        if not np.array_equal(reference["image_ids"], value["image_ids"]):
            raise AssertionError(f"{name}: validation image order differs")
        if not np.array_equal(reference["truths"], value["truths"]):
            raise AssertionError(f"{name}: validation truths differ")

    spatial = {}
    for name in VARIANTS[1:]:
        accumulator = PairedZoneAccumulator()
        for truth, base, candidate in zip(
            reference["truths"],
            reference["predictions"],
            predictions[name]["predictions"],
        ):
            accumulator.update(truth, base, candidate)
        spatial[name] = accumulator.result()
    baseline_boundary = spatial["CBCCH"]["boundary_le_7"]["base_accuracy"]
    baseline_interior = spatial["CBCCH"]["interior_ge_8"]["base_accuracy"]

    zone_iou = {}
    for name in VARIANTS:
        accumulator = ZoneIoUAccumulator()
        for truth, prediction in zip(reference["truths"], predictions[name]["predictions"]):
            accumulator.update(truth, prediction)
        zone_iou[name] = accumulator.result()

    evaluations = {
        name: completion["final_validation"] for name, completion in completions.items()
    }
    c0_final = evaluations["C0"]["scores"]["final"]
    main_rows = []
    for name in VARIANTS:
        final = evaluations[name]["scores"]["final"]
        if name == "C0":
            boundary_accuracy = baseline_boundary
            interior_accuracy = baseline_interior
        else:
            boundary_accuracy = spatial[name]["boundary_le_7"]["candidate_accuracy"]
            interior_accuracy = spatial[name]["interior_ge_8"]["candidate_accuracy"]
        main_rows.append(
            {
                "variant": name,
                "mIoU": final["mIoU"],
                "mDice": final["mDice"],
                "delta_mIoU_vs_C0_pp": 100.0 * (final["mIoU"] - c0_final["mIoU"]),
                "boundary_accuracy": boundary_accuracy,
                "boundary_accuracy_delta_vs_C0_pp": 100.0 * (
                    boundary_accuracy - baseline_boundary
                ),
                "boundary_mIoU": zone_iou[name]["boundary_le_7"]["mIoU"],
                "interior_accuracy": interior_accuracy,
                "interior_mIoU": zone_iou[name]["interior_ge_8"]["mIoU"],
            }
        )

    cam_rows = []
    for name in VARIANTS:
        for stage in STAGES:
            value = evaluations[name]["scores"][stage]
            base = evaluations["C0"]["scores"][stage]
            cam_rows.append(
                {
                    "variant": name,
                    "stage": stage,
                    "mIoU": value["mIoU"],
                    "mDice": value["mDice"],
                    "delta_mIoU_vs_C0_pp": 100.0 * (value["mIoU"] - base["mIoU"]),
                }
            )

    bcp_main = next(row for row in main_rows if row["variant"] == "BCP-CH")
    bcp_cam28 = evaluations["BCP-CH"]["scores"]["28_1"]["mIoU"]
    if bcp_cam28 >= 0.664431:
        semantic_status = "COMPLETE"
    elif bcp_cam28 >= 0.659035:
        semantic_status = "PARTIAL"
    else:
        semantic_status = "FAIL"
    gates = {
        "A_boundary_accuracy": {
            "pass": bcp_main["boundary_accuracy"] > 0.522525,
            "observed": bcp_main["boundary_accuracy"],
            "threshold": 0.522525,
            "margin_pp": 100.0 * (bcp_main["boundary_accuracy"] - 0.522525),
            "criterion": "> 52.2525% (CBCCH)",
        },
        "B_CAM28_1_semantic_recovery": {
            "status": semantic_status,
            "pass": semantic_status == "COMPLETE",
            "observed": bcp_cam28,
            "complete_threshold": 0.664431,
            "partial_threshold": 0.659035,
            "margin_to_complete_pp": 100.0 * (bcp_cam28 - 0.664431),
            "criterion": "COMPLETE >=66.4431%; PARTIAL >=65.9035%",
        },
        "C_final_mIoU": {
            "pass": bcp_main["mIoU"] > 0.668555,
            "observed": bcp_main["mIoU"],
            "threshold": 0.668555,
            "margin_pp": 100.0 * (bcp_main["mIoU"] - 0.668555),
            "criterion": "> 66.8555% (C0)",
        },
    }
    if gates["A_boundary_accuracy"]["pass"] and gates["C_final_mIoU"]["pass"]:
        if semantic_status == "COMPLETE":
            decision = "SUCCESS"
        elif semantic_status == "PARTIAL":
            decision = "PARTIAL_RECOVERY"
        else:
            decision = "OVERALL_WITHOUT_CAM_RECOVERY"
    elif gates["A_boundary_accuracy"]["pass"]:
        decision = "BOUNDARY_ONLY"
    else:
        decision = "NO_GO"

    initial = read_json(directories["BCP-CH"] / "validation" / "epoch20_initial.json")
    initial_similarity = initial["feature_diagnostics"][
        "gt_boundary_prototype_similarity"
    ]
    final_similarity = evaluations["BCP-CH"]["feature_diagnostics"][
        "gt_boundary_prototype_similarity"
    ]
    prototype_diagnostic = {
        "epoch20": initial_similarity,
        "epoch25": final_similarity,
        "delta": final_similarity["mean"] - initial_similarity["mean"],
    }
    feature = evaluations["BCP-CH"]["feature_diagnostics"]
    history = read_json(directories["BCP-CH"] / "training_history.json")
    summary = {
        "experiment_id": "EXP-BCPCH-003",
        "decision": decision,
        "main_metrics": main_rows,
        "cam_metrics": cam_rows,
        "spatial": spatial,
        "zone_iou": zone_iou,
        "gates": gates,
        "prototype_diagnostic": prototype_diagnostic,
        "feature_diagnostics": feature,
        "training_history": history,
        "common_checkpoint_sha256": common_sha,
        "schedule_sha256": schedule_sha,
        "checkpoints": {
            name: {
                "path": completion["checkpoint"],
                "sha256": completion["checkpoint_sha256"],
            }
            for name, completion in completions.items()
        },
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "bcpch_phase3_summary.json", summary)

    lines = [
        "# BCP-CH Phase-3 Wavelet Low-frequency Semantic Prototype Recovery",
        "",
        "## 1. Frozen protocol",
        "",
        "- Experiment: `EXP-BCPCH-003`; BCSS seed42; locked public Epoch20 → Epoch21–25 matched continuation.",
        "- Only BCP-CH is newly trained. C0, BC-CH and CBCCH are reused SHA-locked Epoch25 FINAL references.",
        "- LL embedding is exactly `z=L2(IDWT(LL,0,0,0))`; CAM selection is per-class spatial min-max `ReLU(ic1(F))>0.70` with detached mask.",
        "- Existing `fc8(feat_deep)` and official BCSS thresholds determine image presence; no new classifier, projection or trainable parameter.",
        "- `Y=(1-B)(0.5P_affinity+0.5P_prototype)+BF`; no-valid-prototype fallback is exactly `P_prototype=P_affinity`.",
        "- CBCCH local15, top-20% boundary anchors, τ=0.07 and `L=L_official+0.1L_con` remain unchanged.",
        "- Same optimizer, LR schedule, batch/augmentation/model seeds, BF16 and Epoch25 FINAL selection. No test, LUAD, other seed or tuning.",
        "",
        "## 2. Overall and boundary validation results",
        "",
        "| Variant | mIoU | Δ vs C0 pp | mDice | Boundary acc. | Δ vs C0 pp | Boundary mIoU | Interior acc. | Interior mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['mIoU'])} | {row['delta_mIoU_vs_C0_pp']:+.4f} | "
            f"{fmt(row['mDice'])} | {fmt(row['boundary_accuracy'])} | "
            f"{row['boundary_accuracy_delta_vs_C0_pp']:+.4f} | {fmt(row['boundary_mIoU'])} | "
            f"{fmt(row['interior_accuracy'])} | {fmt(row['interior_mIoU'])} |"
        )
    lines += [
        "",
        "## 3. CAM hierarchy",
        "",
        "| Variant | CAM56 | CAM28_1 | Δ CAM28_1 vs C0 pp | CAM28_2 | Deep | Final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        selected = {row["stage"]: row for row in cam_rows if row["variant"] == name}
        lines.append(
            f"| {name} | {fmt(selected['56']['mIoU'])} | {fmt(selected['28_1']['mIoU'])} | "
            f"{selected['28_1']['delta_mIoU_vs_C0_pp']:+.4f} | {fmt(selected['28_2']['mIoU'])} | "
            f"{fmt(selected['deep']['mIoU'])} | {fmt(selected['final']['mIoU'])} |"
        )
    lines += [
        "",
        "## 4. Prototype and propagation diagnostics",
        "",
        f"- GT-category boundary-to-prototype cosine: Epoch20={initial_similarity['mean']:.6f} over {initial_similarity['pixels']} pixels; Epoch25={final_similarity['mean']:.6f} over {final_similarity['pixels']} pixels; Δ={prototype_diagnostic['delta']:+.6f}.",
        f"- CAM confidence fraction={feature['cam_confidence_fraction']['mean']:.6f}; predicted presence/image={feature['predicted_presence_per_image']['mean']:.4f}; valid prototypes/image={feature['valid_prototypes_per_image']['mean']:.4f}; fallback fraction={feature['fallback_fraction']['mean']:.6f}.",
        f"- LL reconstruction/input RMS={feature['ll_reconstruction_rms']['mean']:.6f}; affinity output/input RMS={feature['affinity_output_rms']['mean']:.6f}; prototype output/input RMS={feature['prototype_output_rms']['mean']:.6f}.",
        f"- Boundary/interior context residual RMS={feature['boundary_context_residual_rms']['mean']:.6f}/{feature['interior_context_residual_rms']['mean']:.6f}.",
        f"- Final gamma_context/gamma_veto={feature['gamma_context']:.8f}/{feature['gamma_veto']:.8f}.",
        "",
        "## 5. Preregistered gates",
        "",
        "| Gate | Observed | Margin/status | Criterion | Result |",
        "|---|---:|---:|---|:---:|",
        f"| A Boundary accuracy | {fmt(gates['A_boundary_accuracy']['observed'])} | {gates['A_boundary_accuracy']['margin_pp']:+.4f} pp | {gates['A_boundary_accuracy']['criterion']} | {'PASS' if gates['A_boundary_accuracy']['pass'] else 'FAIL'} |",
        f"| B CAM28_1 recovery | {fmt(gates['B_CAM28_1_semantic_recovery']['observed'])} | {semantic_status} | {gates['B_CAM28_1_semantic_recovery']['criterion']} | {semantic_status} |",
        f"| C Final mIoU | {fmt(gates['C_final_mIoU']['observed'])} | {gates['C_final_mIoU']['margin_pp']:+.4f} pp | {gates['C_final_mIoU']['criterion']} | {'PASS' if gates['C_final_mIoU']['pass'] else 'FAIL'} |",
        "",
        "## 6. Training and resource evidence",
        "",
        f"- Preflight: `{preflight['status']}`; real batch20 BF16; official+contrastive loss; no optimizer step.",
        f"- Epoch21→25 official loss: {history[0]['official_classification_loss']:.6f} → {history[-1]['official_classification_loss']:.6f}; contrastive loss: {history[0]['contrastive_loss']:.6f} → {history[-1]['contrastive_loss']:.6f}.",
        f"- Continuation runtime={completions['BCP-CH']['runtime_seconds']/3600:.3f} h; peak CUDA memory={completions['BCP-CH']['peak_cuda_memory_bytes']/2**30:.3f} GiB.",
        f"- Common Epoch20 SHA256: `{common_sha}`.",
        f"- Schedule SHA256: `{schedule_sha}`.",
    ]
    for name in VARIANTS:
        lines.append(f"- {name} Epoch25 SHA256: `{completions[name]['checkpoint_sha256']}`.")
    lines += [
        "- Validation image order and GT masks are byte-equal across all four variants.",
        "- BCP-CH trainable parameter count equals C0; the legacy CH15 parameter restores for parity but is dormant under the frozen Phase-3 equation.",
        "",
        "## 7. Scientific interpretation",
        "",
    ]
    if decision == "SUCCESS":
        lines.append("LL prototype pull complements contrastive push: boundary, CAM28_1 and overall utility all satisfy their frozen gates.")
    elif decision == "PARTIAL_RECOVERY":
        lines.append("LL prototype pull yields overall and boundary utility with partial, but not complete, CAM28_1 recovery.")
    elif decision == "BOUNDARY_ONLY":
        lines.append("Boundary separation remains, but LL prototype anchoring does not translate into positive overall utility.")
    else:
        lines.append("The frozen LL prototype construction does not validate the proposed semantic-recovery mechanism.")
    lines += ["", f"DECISION = {decision}", "", "STOP."]
    report = output / "bcpch_phase3_wavelet_ll_prototype_final_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print(f"DECISION = {decision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--c0-dir", required=True)
    parser.add_argument("--bcch-dir", required=True)
    parser.add_argument("--cbcch-dir", required=True)
    parser.add_argument("--bcpch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
