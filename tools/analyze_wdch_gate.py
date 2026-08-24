#!/usr/bin/env python3
"""Compute preregistered WD-CH gates and the final four-way decision."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_wdch import Net_CAM as WDCHNetCAM
from tools.wdch_common import (
    PairedComponentAccumulator,
    PairedZoneAccumulator,
    component_thresholds,
    read_state,
    sha256_file,
    verify_validation_root,
    write_json,
)
from tools.wdch_evaluation import evaluate_bcss


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_component_delta(rows):
    output = {}
    for size in ("small", "medium", "large"):
        selected = [row for row in rows if row["size"] == size]
        pixels = sum(row["pixels"] for row in selected)
        base = sum(row["base_correct"] for row in selected) / max(pixels, 1)
        candidate = sum(row["candidate_correct"] for row in selected) / max(pixels, 1)
        output[size] = {
            "pixels": pixels,
            "base_recall": base,
            "candidate_recall": candidate,
            "delta_pp": 100.0 * (candidate - base),
        }
    return output


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def run_band_interventions(args, kernel, w1_checkpoint, output):
    model = WDCHNetCAM(4, wdch_kernel_size=kernel)
    incompat = model.load_state_dict(read_state(w1_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    model = model.cuda()
    results = {}
    interventions = {
        "LH_zero": ("LH",),
        "HL_zero": ("HL",),
        "HH_zero": ("HH",),
        "all_HF_zero": ("LH", "HL", "HH"),
    }
    for name, bands in interventions.items():
        model.hfrm_28_1.wdch.set_ablation(bands)
        results[name] = evaluate_bcss(
            model, args.val_root, num_workers=args.num_workers
        )
        print(
            f"WDCH_BAND_INTERVENTION name={name} "
            f"mIoU={100*results[name]['scores']['final']['mIoU']:.4f}",
            flush=True,
        )
    model.hfrm_28_1.wdch.set_ablation(())
    write_json(output / "wdch_band_interventions.json", results)
    return results


def run(args):
    verify_validation_root(args.val_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "wdch_utility_gate_summary.json").exists():
        raise FileExistsError("WD-CH final gate already analyzed")
    phase0 = read_json(args.phase0_summary)
    phase1 = read_json(args.phase1_summary)
    if phase0["phase0_status"] != "PASS" or phase1["phase1_status"] != "PASS":
        raise AssertionError("Phase 0 and Phase 1 must pass before utility analysis")
    kernel = int(phase0["selected_kernel"])
    c0_complete = read_json(Path(args.c0_dir) / "complete.json")
    w1_complete = read_json(Path(args.w1_dir) / "complete.json")
    if c0_complete["final_validation"]["epoch"] != 25 or w1_complete["final_validation"]["epoch"] != 25:
        raise AssertionError("Primary comparison must be epoch25 FINAL")
    c0_data = load_predictions(
        Path(args.c0_dir) / "predictions" / "epoch25_validation.npz"
    )
    w1_data = load_predictions(
        Path(args.w1_dir) / "predictions" / "epoch25_validation.npz"
    )
    if not np.array_equal(c0_data["image_ids"], w1_data["image_ids"]):
        raise AssertionError("C0/W1 validation order differs")
    if not np.array_equal(c0_data["truths"], w1_data["truths"]):
        raise AssertionError("C0/W1 validation truths differ")
    thresholds = component_thresholds(args.val_root)
    zones = PairedZoneAccumulator()
    components = PairedComponentAccumulator(thresholds)
    for truth, c0_prediction, w1_prediction in zip(
        c0_data["truths"], c0_data["predictions"], w1_data["predictions"]
    ):
        zones.update(truth, c0_prediction, w1_prediction)
        components.update(truth, c0_prediction, w1_prediction)
    zone_result = zones.result()
    component_rows = components.result()
    component_aggregate = aggregate_component_delta(component_rows)
    write_csv(output / "wdch_component_size_final.csv", component_rows)

    c0_eval = c0_complete["final_validation"]
    w1_eval = w1_complete["final_validation"]
    c0_final = c0_eval["scores"]["final"]
    w1_final = w1_eval["scores"]["final"]
    delta_miou = 100.0 * (w1_final["mIoU"] - c0_final["mIoU"])
    delta_mdice = 100.0 * (w1_final["mDice"] - c0_final["mDice"])
    per_class = {
        str(index): 100.0 * (
            w1_final["class_iou"][str(index)] - c0_final["class_iou"][str(index)]
        )
        for index in range(4)
    }
    boundary_delta = zone_result["boundary_le_7"]["delta_pp"]
    interior_delta = zone_result["interior_ge_8"]["delta_pp"]
    class_collapse = {key: value for key, value in per_class.items() if value < -0.50}
    gate_a = boundary_delta >= 0.25 and interior_delta >= -0.15
    gate_b = delta_miou >= 0.10 and not class_collapse
    if gate_a and gate_b:
        decision = "ROUTE_WAVELET_GO"
    elif gate_a:
        decision = "MECHANISM_VALID_MODEL_INCOMPLETE"
    elif gate_b:
        decision = "PERFORMANCE_GAIN_WITHOUT_MECHANISM_SUPPORT"
    else:
        decision = "ROUTE_WAVELET_CLOSE"

    band_results = None
    if gate_a or gate_b:
        band_results = run_band_interventions(
            args, kernel, w1_complete["checkpoint"], output
        )
    multiscale_rows = []
    for stage in ("56", "28_1", "28_2", "deep", "final"):
        c0_score = c0_eval["scores"][stage]
        w1_score = w1_eval["scores"][stage]
        multiscale_rows.append(
            {
                "stage": stage,
                "C0_mIoU": c0_score["mIoU"],
                "W1_mIoU": w1_score["mIoU"],
                "delta_mIoU_pp": 100.0 * (w1_score["mIoU"] - c0_score["mIoU"]),
                "C0_mDice": c0_score["mDice"],
                "W1_mDice": w1_score["mDice"],
                "delta_mDice_pp": 100.0 * (w1_score["mDice"] - c0_score["mDice"]),
            }
        )
    write_csv(output / "wdch_multiscale_final.csv", multiscale_rows)
    summary = {
        "decision": decision,
        "kernel": kernel,
        "gate_a": {
            "pass": gate_a,
            "boundary_delta_pp": boundary_delta,
            "required_boundary_delta_pp": 0.25,
            "interior_delta_pp": interior_delta,
            "required_interior_floor_pp": -0.15,
        },
        "gate_b": {
            "pass": gate_b,
            "mIoU_delta_pp": delta_miou,
            "required_mIoU_delta_pp": 0.10,
            "class_collapse_warning": class_collapse,
        },
        "mDice_delta_pp": delta_mdice,
        "per_class_iou_delta_pp": per_class,
        "zone_metrics": zone_result,
        "component_metrics": component_aggregate,
        "component_rows": component_rows,
        "multiscale": multiscale_rows,
        "feature_diagnostics": {
            "C0": c0_eval["feature_diagnostics"],
            "W1": w1_eval["feature_diagnostics"],
        },
        "checkpoints": {
            "C0": {
                "path": c0_complete["checkpoint"],
                "sha256": c0_complete["checkpoint_sha256"],
            },
            "W1": {
                "path": w1_complete["checkpoint"],
                "sha256": w1_complete["checkpoint_sha256"],
            },
        },
        "band_interventions": band_results,
        "checkpoint_selection": "none; epoch25 FINAL only",
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "wdch_utility_gate_summary.json", summary)
    c0_zone = {
        "boundary": zone_result["boundary_le_7"]["base_accuracy"],
        "interior": zone_result["interior_ge_8"]["base_accuracy"],
    }
    w1_zone = {
        "boundary": zone_result["boundary_le_7"]["candidate_accuracy"],
        "interior": zone_result["interior_ge_8"]["candidate_accuracy"],
    }
    lines = [
        "# WD-CH Utility Gate Final Report",
        "",
        "## 1. Frozen protocol",
        "",
        f"- A0 source: `{phase0.get('a0_commit')}`; k*={kernel} selected only by Phase-0 impulse response.",
        "- BCSS seed42; common fresh training to epoch20; matched C0/W1 continuation through epoch25.",
        "- Batch20, 224×224, BF16, released loss/optimizer/poly schedule and official validation inference.",
        "- Epoch25 FINAL only; no test, LUAD, other seed or best-epoch selection.",
        "",
        "## 2. Main result",
        "",
        "| Variant | mIoU | mDice | Boundary | Interior | Small | Medium | Large |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| C0 | {100*c0_final['mIoU']:.4f} | {100*c0_final['mDice']:.4f} | {100*c0_zone['boundary']:.4f} | {100*c0_zone['interior']:.4f} | {100*component_aggregate['small']['base_recall']:.4f} | {100*component_aggregate['medium']['base_recall']:.4f} | {100*component_aggregate['large']['base_recall']:.4f} |",
        f"| W1 | {100*w1_final['mIoU']:.4f} | {100*w1_final['mDice']:.4f} | {100*w1_zone['boundary']:.4f} | {100*w1_zone['interior']:.4f} | {100*component_aggregate['small']['candidate_recall']:.4f} | {100*component_aggregate['medium']['candidate_recall']:.4f} | {100*component_aggregate['large']['candidate_recall']:.4f} |",
        f"| Δ (pp) | {delta_miou:+.4f} | {delta_mdice:+.4f} | {boundary_delta:+.4f} | {interior_delta:+.4f} | {component_aggregate['small']['delta_pp']:+.4f} | {component_aggregate['medium']['delta_pp']:+.4f} | {component_aggregate['large']['delta_pp']:+.4f} |",
        "",
        "## 3. Per-class IoU delta",
        "",
        "- " + ", ".join(f"C{key}: {value:+.4f} pp" for key, value in per_class.items()),
        f"- CLASS_COLLAPSE_WARNING: {class_collapse if class_collapse else 'none'}",
        "",
        "## 4. Multi-scale CAM",
        "",
        "| Stage | C0 mIoU | W1 mIoU | Δ pp |",
        "|---|---:|---:|---:|",
    ]
    for row in multiscale_rows:
        lines.append(
            f"| {row['stage']} | {100*row['C0_mIoU']:.4f} | "
            f"{100*row['W1_mIoU']:.4f} | {row['delta_mIoU_pp']:+.4f} |"
        )
    lines += [
        "",
        "## 5. Feature diagnostics",
        "",
        f"- C0 Output/Input RMS: {c0_eval['feature_diagnostics']['output_input_rms']['mean']:.6f} ± {c0_eval['feature_diagnostics']['output_input_rms']['std']:.6f}",
        f"- W1 Output/Input RMS: {w1_eval['feature_diagnostics']['output_input_rms']['mean']:.6f} ± {w1_eval['feature_diagnostics']['output_input_rms']['std']:.6f}",
        f"- C0 rectification RMS: {c0_eval['feature_diagnostics']['rectification_rms']['mean']:.6f}",
        f"- W1 rectification RMS: {w1_eval['feature_diagnostics']['rectification_rms']['mean']:.6f}",
        "",
        "## 6. Preregistered gates",
        "",
        f"- MECHANISM_GATE: {'PASS' if gate_a else 'FAIL'} (Boundary {boundary_delta:+.4f} pp; Interior {interior_delta:+.4f} pp).",
        f"- MODEL_UTILITY_GATE: {'PASS' if gate_b else 'FAIL'} (mIoU {delta_miou:+.4f} pp; collapse={bool(class_collapse)}).",
        "",
        "## 7. Required questions",
        "",
        f"- Q1 Overall improvement: {'yes' if delta_miou > 0 else 'no'} ({delta_miou:+.4f} pp).",
        f"- Q2 Boundary improvement: {'yes' if boundary_delta > 0 else 'no'} ({boundary_delta:+.4f} pp).",
        f"- Q3 Interior cost: {interior_delta:+.4f} pp; retained under Gate A = {interior_delta >= -0.15}.",
        f"- Q4 FA-MPR-like small-up/large-down failure: {component_aggregate['small']['delta_pp'] > 0 and component_aggregate['large']['delta_pp'] < 0}.",
        f"- Q5 Frequency-blind CH limitation supported at the preregistered level: {gate_a}.",
        "",
        "## 8. Band intervention",
        "",
        "Run only because at least one gate passed." if band_results else "Not run because neither gate passed.",
        "",
        "## 9. Provenance",
        "",
        f"- C0 checkpoint: `{c0_complete['checkpoint_sha256']}`",
        f"- W1 checkpoint: `{w1_complete['checkpoint_sha256']}`",
        "- Machine-readable outputs: `wdch_utility_gate_summary.json`, `wdch_multiscale_final.csv`, `wdch_component_size_final.csv`.",
        "",
        f"DECISION = {decision}",
        "",
        "STOP.",
    ]
    (output / "wdch_utility_gate_final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "decision": decision,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "delta_mIoU_pp": delta_miou,
        "boundary_delta_pp": boundary_delta,
        "interior_delta_pp": interior_delta,
    }, indent=2), flush=True)
    print(f"DECISION = {decision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--phase0-summary", required=True)
    parser.add_argument("--phase1-summary", required=True)
    parser.add_argument("--c0-dir", required=True)
    parser.add_argument("--w1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
