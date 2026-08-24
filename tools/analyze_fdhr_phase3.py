#!/usr/bin/env python3
"""Analyze EXP-FDHR-003 final checkpoints against locked C0/W1 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.wdch_common import (
    PairedComponentAccumulator,
    PairedZoneAccumulator,
    component_thresholds,
    sha256_file,
    write_json,
)


VARIANTS = ("C0", "W1", "A", "B", "C")
STAGES = ("56", "28_1", "28_2", "deep", "final")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


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


def validate_completion(name, completion):
    expected = (
        "WDCH_MATCHED_BRANCH_COMPLETE"
        if name in ("C0", "W1")
        else "FDHR_MATCHED_VARIANT_COMPLETE"
    )
    if completion.get("status") != expected:
        raise AssertionError(f"{name}: {completion.get('status')} != {expected}")
    if completion.get("epochs") != [21, 22, 23, 24, 25]:
        raise AssertionError(f"{name}: continuation epochs differ")
    if completion["final_validation"].get("epoch") != 25:
        raise AssertionError(f"{name}: primary evaluation is not epoch25 FINAL")
    if completion.get("test_used") or completion["final_validation"].get("test_used"):
        raise AssertionError(f"{name}: test set was used")


def fmt(value):
    return f"{100.0 * value:.4f}"


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    directories = {
        "C0": Path(args.c0_dir),
        "W1": Path(args.w1_dir),
        "A": Path(args.a_dir),
        "B": Path(args.b_dir),
        "C": Path(args.c_dir),
    }
    completions = {name: read_json(path / "complete.json") for name, path in directories.items()}
    for name, completion in completions.items():
        validate_completion(name, completion)
    predictions = {
        name: load_predictions(path / "predictions" / "epoch25_validation.npz")
        for name, path in directories.items()
    }
    reference = predictions["C0"]
    for name, values in predictions.items():
        if not np.array_equal(reference["image_ids"], values["image_ids"]):
            raise AssertionError(f"{name}: validation image order differs")
        if not np.array_equal(reference["truths"], values["truths"]):
            raise AssertionError(f"{name}: validation truths differ")

    thresholds = component_thresholds(args.val_root)
    spatial = {}
    objects = {}
    for name in VARIANTS[1:]:
        zones = PairedZoneAccumulator()
        components = PairedComponentAccumulator(thresholds)
        for truth, base_prediction, candidate_prediction in zip(
            reference["truths"],
            reference["predictions"],
            predictions[name]["predictions"],
        ):
            zones.update(truth, base_prediction, candidate_prediction)
            components.update(truth, base_prediction, candidate_prediction)
        spatial[name] = zones.result()
        objects[name] = aggregate_component_delta(components.result())

    evaluations = {
        name: completion["final_validation"] for name, completion in completions.items()
    }
    c0_final = evaluations["C0"]["scores"]["final"]
    main_rows = []
    cam_rows = []
    class_rows = []
    for name in VARIANTS:
        final = evaluations[name]["scores"]["final"]
        if name == "C0":
            boundary = spatial["W1"]["boundary_le_7"]["base_accuracy"]
            interior = spatial["W1"]["interior_ge_8"]["base_accuracy"]
            small = objects["W1"]["small"]["base_recall"]
            medium = objects["W1"]["medium"]["base_recall"]
            large = objects["W1"]["large"]["base_recall"]
            boundary_delta = interior_delta = 0.0
        else:
            boundary = spatial[name]["boundary_le_7"]["candidate_accuracy"]
            interior = spatial[name]["interior_ge_8"]["candidate_accuracy"]
            small = objects[name]["small"]["candidate_recall"]
            medium = objects[name]["medium"]["candidate_recall"]
            large = objects[name]["large"]["candidate_recall"]
            boundary_delta = spatial[name]["boundary_le_7"]["delta_pp"]
            interior_delta = spatial[name]["interior_ge_8"]["delta_pp"]
        main_rows.append(
            {
                "variant": name,
                "mIoU": final["mIoU"],
                "mDice": final["mDice"],
                "delta_mIoU_pp": 100.0 * (final["mIoU"] - c0_final["mIoU"]),
                "delta_mDice_pp": 100.0 * (final["mDice"] - c0_final["mDice"]),
                "boundary": boundary,
                "boundary_delta_pp": boundary_delta,
                "interior": interior,
                "interior_delta_pp": interior_delta,
                "small": small,
                "medium": medium,
                "large": large,
            }
        )
        for stage in STAGES:
            value = evaluations[name]["scores"][stage]
            c0_value = evaluations["C0"]["scores"][stage]
            cam_rows.append(
                {
                    "variant": name,
                    "stage": stage,
                    "mIoU": value["mIoU"],
                    "mDice": value["mDice"],
                    "delta_mIoU_pp": 100.0 * (value["mIoU"] - c0_value["mIoU"]),
                }
            )
        for class_index in range(4):
            key = str(class_index)
            class_rows.append(
                {
                    "variant": name,
                    "class": class_index,
                    "IoU": final["class_iou"][key],
                    "delta_IoU_pp": 100.0
                    * (final["class_iou"][key] - c0_final["class_iou"][key]),
                }
            )

    gates = {}
    for name in ("A", "B", "C"):
        cam28 = next(row for row in cam_rows if row["variant"] == name and row["stage"] == "28_1")
        main = next(row for row in main_rows if row["variant"] == name)
        semantic = cam28["delta_mIoU_pp"] > -0.28
        structural = main["boundary_delta_pp"] > 0.2
        overall = main["delta_mIoU_pp"] > 0.1
        gates[name] = {
            "semantic_recovery": semantic,
            "CAM28_1_delta_pp": cam28["delta_mIoU_pp"],
            "semantic_floor_pp": -0.28,
            "structural_preservation": structural,
            "boundary_delta_pp": main["boundary_delta_pp"],
            "boundary_target_pp": 0.2,
            "overall_improvement": overall,
            "mIoU_delta_pp": main["delta_mIoU_pp"],
            "mIoU_target_pp": 0.1,
            "success": semantic and structural and overall,
        }

    if gates["B"]["success"]:
        decision = "GO"
        next_step = "Develop the frozen evidence route into the FDHR final model: HF-guided semantic rectification."
        conclusion = "Minimal HF-to-semantic guidance solved the preregistered semantic, structural, and overall utility gates."
    elif gates["C"]["success"]:
        decision = "GO"
        next_step = "Develop a dedicated HF residual structural-correction module."
        conclusion = "Minimal HF residual correction solved all three preregistered gates."
    elif gates["A"]["success"]:
        decision = "NEXT_STEP"
        next_step = "Study controlled HF preservation; do not yet claim a final cross-band guidance mechanism."
        conclusion = "Fixed HF amplification recovered utility, but it does not establish HF-guided LL rectification."
    elif any(gate["semantic_recovery"] or gate["structural_preservation"] or gate["overall_improvement"] for gate in gates.values()):
        decision = "NEXT_STEP"
        next_step = "Reformulate frequency interaction from the partial gate evidence; do not tune these fixed variants."
        conclusion = "No variant solved all three gates, although at least one preregistered signal remained positive."
    else:
        decision = "STOP"
        next_step = "Close these minimal interaction forms and reformulate the frequency-interaction hypothesis."
        conclusion = "All minimal cross-band variants failed the preregistered utility evidence."

    frequency = {
        name: evaluations[name]["feature_diagnostics"]
        for name in ("A", "B", "C")
    }
    summary = {
        "experiment_id": "EXP-FDHR-003",
        "decision": decision,
        "next_step": next_step,
        "conclusion": conclusion,
        "main_metrics": main_rows,
        "cam_metrics": cam_rows,
        "per_class": class_rows,
        "spatial": spatial,
        "object_size": objects,
        "frequency": frequency,
        "gates": gates,
        "checkpoints": {
            name: {
                "path": completions[name]["checkpoint"],
                "sha256": completions[name]["checkpoint_sha256"],
            }
            for name in VARIANTS
        },
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "schedule_sha256": sha256_file(args.schedule),
        "checkpoint_rule": "epoch25 FINAL only",
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "fdhr_phase3_cross_band_summary.json", summary)

    lines = [
        "# FDHR Phase-3 Cross-Band Interaction Utility Gate",
        "",
        "## 1. Frozen protocol",
        "",
        "- Experiment: `EXP-FDHR-003`.",
        "- BCSS seed42; every model starts from the identical locked SSHR Epoch 20 state and follows the identical Epoch 21–25 schedule.",
        "- Batch20, 224×224, BF16, official loss/optimizer/poly schedule, augmentation, inference and metric.",
        "- C0 and W1 are reused SHA-locked matched artifacts; only A/B/C were newly continued.",
        "- Primary result: Epoch 25 FINAL only. No test, LUAD, other seed, best-checkpoint selection or tuning.",
        "- Only HFRM28_1 changes. Fixed strengths are α=β=γ=0.1 and are non-trainable buffers.",
        "- Variant C resolves `Pool(HF)` as arithmetic mean over the LH/HL/HH band axis. The Haar bands already have LL spatial resolution, so no second spatial downsampling is applied.",
        "",
        "## 2. Overall, spatial and object-size metrics",
        "",
        "| Variant | mIoU | Δ mIoU pp | mDice | Boundary | Δ Boundary pp | Interior | Δ Interior pp | Small | Medium | Large |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['mIoU'])} | {row['delta_mIoU_pp']:+.4f} | "
            f"{fmt(row['mDice'])} | {fmt(row['boundary'])} | {row['boundary_delta_pp']:+.4f} | "
            f"{fmt(row['interior'])} | {row['interior_delta_pp']:+.4f} | "
            f"{fmt(row['small'])} | {fmt(row['medium'])} | {fmt(row['large'])} |"
        )
    lines += [
        "",
        "## 3. CAM hierarchy",
        "",
        "| Variant | CAM56 | CAM28_1 | Δ CAM28_1 pp | CAM28_2 | Deep CAM | Final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        by_stage = {row["stage"]: row for row in cam_rows if row["variant"] == name}
        lines.append(
            f"| {name} | {fmt(by_stage['56']['mIoU'])} | {fmt(by_stage['28_1']['mIoU'])} | "
            f"{by_stage['28_1']['delta_mIoU_pp']:+.4f} | {fmt(by_stage['28_2']['mIoU'])} | "
            f"{fmt(by_stage['deep']['mIoU'])} | {fmt(by_stage['final']['mIoU'])} |"
        )
    lines += [
        "",
        "## 4. Per-class final IoU",
        "",
        "| Variant | Class 0 | Class 1 | Class 2 | Class 3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        selected = [row for row in class_rows if row["variant"] == name]
        lines.append("| " + name + " | " + " | ".join(fmt(row["IoU"]) for row in selected) + " |")
    lines += [
        "",
        "## 5. Frequency and interaction statistics",
        "",
        "Definitions: `E_LL = mean(LL²)`, `E_HF = mean(LH²+HL²+HH²)`, and interaction magnitude is the reconstructed feature-domain RMS of the added cross-band term.",
        "",
        "| Variant | E_LL mean±std | E_HF mean±std | Interaction RMS mean±std | Interaction/Input RMS mean±std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("A", "B", "C"):
        row = frequency[name]
        lines.append(
            f"| {name} | {row['E_LL']['mean']:.6f}±{row['E_LL']['std']:.6f} | "
            f"{row['E_HF']['mean']:.6f}±{row['E_HF']['std']:.6f} | "
            f"{row['interaction_rms']['mean']:.6f}±{row['interaction_rms']['std']:.6f} | "
            f"{row['interaction_input_rms']['mean']:.6f}±{row['interaction_input_rms']['std']:.6f} |"
        )
    lines += [
        "",
        "## 6. Preregistered utility gates",
        "",
        "Strict criteria: CAM28_1 Δ > −0.28 pp; Boundary Δ > +0.20 pp; final mIoU Δ > +0.10 pp. A variant succeeds only if all three pass.",
        "",
        "| Variant | Semantic Δ | Gate A | Boundary Δ | Gate B | mIoU Δ | Gate C | Variant success |",
        "|---|---:|:---:|---:|:---:|---:|:---:|:---:|",
    ]
    for name in ("A", "B", "C"):
        gate = gates[name]
        lines.append(
            f"| {name} | {gate['CAM28_1_delta_pp']:+.4f} | {'PASS' if gate['semantic_recovery'] else 'FAIL'} | "
            f"{gate['boundary_delta_pp']:+.4f} | {'PASS' if gate['structural_preservation'] else 'FAIL'} | "
            f"{gate['mIoU_delta_pp']:+.4f} | {'PASS' if gate['overall_improvement'] else 'FAIL'} | "
            f"{'PASS' if gate['success'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 7. Reproducibility",
        "",
        f"- Common Epoch 20 checkpoint SHA256: `{summary['common_checkpoint_sha256']}`.",
        f"- Schedule SHA256: `{summary['schedule_sha256']}`.",
    ]
    for name in VARIANTS:
        lines.append(f"- {name} Epoch 25 checkpoint SHA256: `{completions[name]['checkpoint_sha256']}`.")
    lines += [
        "- Validation prediction order and GT masks were byte-equal across all five variants.",
        "",
        "## 8. Scientific decision",
        "",
        conclusion,
        "",
        f"NEXT_STEP: {next_step}",
        "",
        f"DECISION = {decision}",
        "",
        "STOP.",
    ]
    (output / "fdhr_phase3_cross_band_final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print(f"DECISION = {decision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--c0-dir", required=True)
    parser.add_argument("--w1-dir", required=True)
    parser.add_argument("--a-dir", required=True)
    parser.add_argument("--b-dir", required=True)
    parser.add_argument("--c-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
