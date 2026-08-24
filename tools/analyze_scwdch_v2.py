#!/usr/bin/env python3
"""Analyze C0/W1/W2 and apply EXP-WDCH-002 Gates A/B/C."""

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

from network import resnet38_cls, resnet38_scwdch, resnet38_wdch
from tools.wdch_common import (
    PairedComponentAccumulator,
    component_thresholds,
    foreground_boundary_distance,
    read_state,
    sha256_file,
    verify_validation_root,
    write_json,
)
from tools.wdch_evaluation import summarize_feature_magnitude
from tools.scwdch_constants import (
    EXPECTED_C0_FINAL_SHA256,
    EXPECTED_COMMON_EPOCH20_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    EXPECTED_W1_FINAL_SHA256,
    EXPERIMENT_ID,
)


VARIANTS = ("C0", "W1", "W2")
STAGES = ("56", "28_1", "28_2", "deep", "final")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def aggregate_components(rows):
    output = {}
    for size in ("small", "medium", "large"):
        selected = [row for row in rows if row["size"] == size]
        pixels = sum(row["pixels"] for row in selected)
        base_correct = sum(row["base_correct"] for row in selected)
        candidate_correct = sum(row["candidate_correct"] for row in selected)
        base = base_correct / max(pixels, 1)
        candidate = candidate_correct / max(pixels, 1)
        output[size] = {
            "pixels": pixels,
            "base_recall": base,
            "candidate_recall": candidate,
            "delta_pp": 100.0 * (candidate - base),
        }
    return output


def pair_component_metrics(truths, base, candidate, thresholds):
    accumulator = PairedComponentAccumulator(thresholds)
    for truth, base_prediction, candidate_prediction in zip(truths, base, candidate):
        accumulator.update(truth, base_prediction, candidate_prediction)
    rows = accumulator.result()
    return rows, aggregate_components(rows)


def zone_metrics(truths, predictions):
    counts = {
        variant: {
            "boundary_correct": 0,
            "boundary_total": 0,
            "interior_correct": 0,
            "interior_total": 0,
        }
        for variant in VARIANTS
    }
    for index, truth in enumerate(truths):
        masks = foreground_boundary_distance(truth)
        for variant in VARIANTS:
            prediction = predictions[variant][index]
            boundary = masks["boundary_le_7"]
            interior = masks["interior_ge_8"]
            counts[variant]["boundary_correct"] += int(
                np.count_nonzero((prediction == truth) & boundary)
            )
            counts[variant]["boundary_total"] += int(np.count_nonzero(boundary))
            counts[variant]["interior_correct"] += int(
                np.count_nonzero((prediction == truth) & interior)
            )
            counts[variant]["interior_total"] += int(np.count_nonzero(interior))
    return {
        variant: {
            **value,
            "boundary_accuracy": value["boundary_correct"]
            / max(value["boundary_total"], 1),
            "interior_accuracy": value["interior_correct"]
            / max(value["interior_total"], 1),
        }
        for variant, value in counts.items()
    }


def load_model(variant, checkpoint, kernel, scale):
    if variant == "C0":
        model = resnet38_cls.Net(4)
    elif variant == "W1":
        model = resnet38_wdch.Net(4, wdch_kernel_size=kernel)
    else:
        model = resnet38_scwdch.Net(
            4, wdch_kernel_size=kernel, scwdch_scale=scale
        )
    incompat = model.load_state_dict(read_state(checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    return model


def run(args):
    verify_validation_root(args.val_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "scwdch_v2_strength_calibration_final_report.md"
    if report_path.exists():
        raise FileExistsError(report_path)

    calibration = read_json(args.calibration)
    if calibration.get("experiment_id") != EXPERIMENT_ID:
        raise AssertionError("Unexpected calibration experiment")
    if calibration.get("validation_used") or calibration.get("test_used"):
        raise AssertionError("Calibration provenance is contaminated")
    common_sha = sha256_file(args.common_checkpoint)
    schedule_sha = sha256_file(args.schedule)
    if common_sha != EXPECTED_COMMON_EPOCH20_SHA256:
        raise AssertionError("Frozen common epoch20 SHA256 mismatch")
    if schedule_sha != EXPECTED_SCHEDULE_SHA256:
        raise AssertionError("Frozen schedule SHA256 mismatch")
    if calibration["checkpoint_sha256"] != common_sha:
        raise AssertionError("Calibration/common checkpoint mismatch")
    kernel = int(calibration["kernel"])
    scale = float(calibration["scale"])

    dirs = {
        "C0": Path(args.c0_dir),
        "W1": Path(args.w1_dir),
        "W2": Path(args.w2_dir),
    }
    completions = {variant: read_json(path / "complete.json") for variant, path in dirs.items()}
    provenances = {variant: read_json(path / "provenance.json") for variant, path in dirs.items()}
    for variant in VARIANTS:
        completion = completions[variant]
        if completion["branch"] != variant or completion["epochs"] != [21, 22, 23, 24, 25]:
            raise AssertionError(f"{variant} is not an epoch25 matched continuation")
        if completion["final_validation"]["epoch"] != 25:
            raise AssertionError(f"{variant} final checkpoint rule violated")
        if sha256_file(completion["checkpoint"]) != completion["checkpoint_sha256"]:
            raise AssertionError(f"{variant} checkpoint SHA mismatch")
        provenance = provenances[variant]
        if provenance["common_checkpoint_sha256"] != common_sha:
            raise AssertionError(f"{variant} common checkpoint differs")
        if provenance["schedule_sha256"] != schedule_sha:
            raise AssertionError(f"{variant} schedule differs")
    if completions["C0"]["checkpoint_sha256"] != EXPECTED_C0_FINAL_SHA256:
        raise AssertionError("Frozen C0 FINAL checkpoint mismatch")
    if completions["W1"]["checkpoint_sha256"] != EXPECTED_W1_FINAL_SHA256:
        raise AssertionError("Frozen W1 FINAL checkpoint mismatch")

    prediction_data = {
        variant: load_predictions(path / "predictions" / "epoch25_validation.npz")
        for variant, path in dirs.items()
    }
    reference_ids = prediction_data["C0"]["image_ids"]
    truths = prediction_data["C0"]["truths"]
    for variant in ("W1", "W2"):
        if not np.array_equal(reference_ids, prediction_data[variant]["image_ids"]):
            raise AssertionError(f"{variant} validation order differs")
        if not np.array_equal(truths, prediction_data[variant]["truths"]):
            raise AssertionError(f"{variant} validation truths differ")
    predictions = {
        variant: prediction_data[variant]["predictions"] for variant in VARIANTS
    }

    zones = zone_metrics(truths, predictions)
    thresholds = component_thresholds(args.val_root)
    component_pair_rows = []
    component_pairs = {}
    for base, candidate in (("C0", "W1"), ("C0", "W2"), ("W1", "W2")):
        rows, aggregate = pair_component_metrics(
            truths, predictions[base], predictions[candidate], thresholds
        )
        name = f"{candidate}-{base}"
        component_pairs[name] = aggregate
        component_pair_rows.extend(
            {"comparison": name, **row} for row in rows
        )
    write_csv(output / "scwdch_component_comparison.csv", component_pair_rows)

    scores = {
        variant: completions[variant]["final_validation"]["scores"]
        for variant in VARIANTS
    }
    performance_rows = []
    for variant in VARIANTS:
        final = scores[variant]["final"]
        performance_rows.append(
            {
                "variant": variant,
                "mIoU": final["mIoU"],
                "mDice": final["mDice"],
                **{
                    f"class_{index}_IoU": final["class_iou"][str(index)]
                    for index in range(4)
                },
            }
        )
    write_csv(output / "scwdch_performance.csv", performance_rows)

    multiscale_rows = []
    for stage in STAGES:
        row = {"stage": stage}
        for variant in VARIANTS:
            row[f"{variant}_mIoU"] = scores[variant][stage]["mIoU"]
            row[f"{variant}_mDice"] = scores[variant][stage]["mDice"]
        row["W2_minus_C0_mIoU_pp"] = 100.0 * (
            row["W2_mIoU"] - row["C0_mIoU"]
        )
        row["W2_minus_W1_mIoU_pp"] = 100.0 * (
            row["W2_mIoU"] - row["W1_mIoU"]
        )
        multiscale_rows.append(row)
    write_csv(output / "scwdch_multiscale.csv", multiscale_rows)

    feature_audit = {}
    for variant in VARIANTS:
        model = load_model(
            variant, completions[variant]["checkpoint"], kernel, scale
        ).cuda()
        feature_audit[variant] = summarize_feature_magnitude(
            model, args.val_root, num_workers=args.num_workers
        )
        del model
        torch.cuda.empty_cache()
    write_json(output / "scwdch_feature_magnitude.json", feature_audit)

    final = {variant: scores[variant]["final"] for variant in VARIANTS}
    delta_miou_w2_c0 = 100.0 * (final["W2"]["mIoU"] - final["C0"]["mIoU"])
    delta_miou_w2_w1 = 100.0 * (final["W2"]["mIoU"] - final["W1"]["mIoU"])
    delta_mdice_w2_c0 = 100.0 * (final["W2"]["mDice"] - final["C0"]["mDice"])
    boundary_w2_c0 = 100.0 * (
        zones["W2"]["boundary_accuracy"] - zones["C0"]["boundary_accuracy"]
    )
    interior_w2_c0 = 100.0 * (
        zones["W2"]["interior_accuracy"] - zones["C0"]["interior_accuracy"]
    )
    boundary_w2_w1 = 100.0 * (
        zones["W2"]["boundary_accuracy"] - zones["W1"]["boundary_accuracy"]
    )
    interior_w2_w1 = 100.0 * (
        zones["W2"]["interior_accuracy"] - zones["W1"]["interior_accuracy"]
    )
    final_strength_ratio = (
        feature_audit["W2"]["rectification_rms_absolute"]["mean"]
        / feature_audit["C0"]["rectification_rms_absolute"]["mean"]
    )
    cam28_w2_w1 = next(
        row["W2_minus_W1_mIoU_pp"]
        for row in multiscale_rows
        if row["stage"] == "28_1"
    )
    cam28_w2_c0 = next(
        row["W2_minus_C0_mIoU_pp"]
        for row in multiscale_rows
        if row["stage"] == "28_1"
    )
    per_class_delta = {
        str(index): 100.0 * (
            final["W2"]["class_iou"][str(index)]
            - final["C0"]["class_iou"][str(index)]
        )
        for index in range(4)
    }
    class_collapse = {
        key: value for key, value in per_class_delta.items() if value < -0.50
    }

    gate_a = 0.9 <= final_strength_ratio <= 1.1
    gate_b = boundary_w2_c0 >= 0.25 and interior_w2_c0 >= -0.15
    gate_c = delta_miou_w2_c0 >= 0.10
    cam28_recovered = cam28_w2_w1 > 0.0
    if gate_a and gate_b and gate_c and cam28_recovered:
        decision = "GO"
        interpretation = "SC-WDCH succeeds; strength recovery restores semantic utility while preserving the boundary mechanism."
    elif gate_a and gate_b:
        decision = "NEXT_STEP"
        interpretation = "Strength is recovered and the boundary mechanism is preserved, but semantic/model utility remains incomplete; proceed to Cross-Band Semantic Interaction."
    elif gate_a:
        decision = "NEXT_STEP"
        interpretation = "Strength is recovered but the boundary mechanism is not preserved; proceed to Selective Frequency Strength Allocation."
    else:
        decision = "STOP"
        interpretation = "Final strength recovery failed; close the fixed strength-calibration hypothesis."

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "interpretation": interpretation,
        "calibration": calibration,
        "gate_a_strength_recovery": {
            "pass": gate_a,
            "initial_ratio": calibration["initial_strength_ratio"],
            "final_ratio": final_strength_ratio,
            "allowed_range": [0.9, 1.1],
        },
        "gate_b_mechanism_preservation": {
            "pass": gate_b,
            "boundary_W2_minus_C0_pp": boundary_w2_c0,
            "interior_W2_minus_C0_pp": interior_w2_c0,
            "boundary_requirement_pp": 0.25,
            "interior_floor_pp": -0.15,
        },
        "gate_c_model_improvement": {
            "pass": gate_c,
            "mIoU_W2_minus_C0_pp": delta_miou_w2_c0,
            "requirement_pp": 0.10,
            "class_collapse_warning": class_collapse,
        },
        "W2_minus_W1": {
            "mIoU_pp": delta_miou_w2_w1,
            "boundary_pp": boundary_w2_w1,
            "interior_pp": interior_w2_w1,
            "CAM28_1_mIoU_pp": cam28_w2_w1,
        },
        "mDice_W2_minus_C0_pp": delta_mdice_w2_c0,
        "CAM28_1_W2_minus_C0_pp": cam28_w2_c0,
        "CAM28_1_recovered_vs_W1": cam28_recovered,
        "per_class_IoU_W2_minus_C0_pp": per_class_delta,
        "performance": performance_rows,
        "zones": zones,
        "components": component_pairs,
        "multiscale": multiscale_rows,
        "feature_magnitude": feature_audit,
        "common_checkpoint_sha256": common_sha,
        "schedule_sha256": schedule_sha,
        "checkpoints": {
            variant: completions[variant]["checkpoint_sha256"]
            for variant in VARIANTS
        },
        "checkpoint_selection": "epoch25 FINAL only",
        "validation_used_for_calibration": False,
        "test_used": False,
    }
    write_json(output / "scwdch_v2_summary.json", summary)

    component_absolute = {
        "C0": {
            size: component_pairs["W1-C0"][size]["base_recall"]
            for size in ("small", "medium", "large")
        },
        "W1": {
            size: component_pairs["W1-C0"][size]["candidate_recall"]
            for size in ("small", "medium", "large")
        },
        "W2": {
            size: component_pairs["W2-C0"][size]["candidate_recall"]
            for size in ("small", "medium", "large")
        },
    }
    lines = [
        "# SC-WDCH Strength Calibration Final Report",
        "",
        "## 1. Frozen protocol and provenance",
        "",
        "- Experiment: `EXP-WDCH-002`; dataset: BCSS; seed: 42.",
        "- Calibration used the full training split and common epoch20 only; validation/test were forbidden.",
        "- C0/W1 are the hash-verified matched v1 continuations; only W2 was newly trained from the same common state.",
        "- Epoch21-25, batch20, 224x224, BF16, identical optimizer/poly schedule/augmentations/batch order.",
        "- Epoch25 FINAL only; validation was observation-only and test was not run.",
        f"- Common checkpoint SHA256: `{common_sha}`",
        f"- Schedule SHA256: `{schedule_sha}`",
        "",
        "## 2. Training-only strength calibration",
        "",
        f"- R_CH: {calibration['R_CH']:.8f}",
        f"- R_WD: {calibration['R_WD']:.8f}",
        f"- Fixed scale s=R_CH/R_WD: {scale:.8f}",
        f"- Direct initial SC-WD rectification RMS: {calibration['R_SC_WD']:.8f}",
        f"- Initial strength ratio: {calibration['initial_strength_ratio']:.6f}",
        "",
        "## 3. Final validation performance",
        "",
        "| Variant | mIoU | mDice | C0 IoU | C1 IoU | C2 IoU | C3 IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        value = final[variant]
        lines.append(
            f"| {variant} | {100*value['mIoU']:.4f} | {100*value['mDice']:.4f} | "
            + " | ".join(f"{100*value['class_iou'][str(i)]:.4f}" for i in range(4))
            + " |"
        )
    lines += [
        "",
        f"- W2-C0: mIoU {delta_miou_w2_c0:+.4f} pp; mDice {delta_mdice_w2_c0:+.4f} pp.",
        f"- W2-W1: mIoU {delta_miou_w2_w1:+.4f} pp.",
        f"- Per-class W2-C0 IoU: " + ", ".join(
            f"C{key} {value:+.4f} pp" for key, value in per_class_delta.items()
        ),
        "",
        "## 4. Boundary and interior",
        "",
        "| Variant | Boundary accuracy | Interior accuracy |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        lines.append(
            f"| {variant} | {100*zones[variant]['boundary_accuracy']:.4f} | "
            f"{100*zones[variant]['interior_accuracy']:.4f} |"
        )
    lines += [
        "",
        f"- W2-C0: Boundary {boundary_w2_c0:+.4f} pp; Interior {interior_w2_c0:+.4f} pp.",
        f"- W2-W1: Boundary {boundary_w2_w1:+.4f} pp; Interior {interior_w2_w1:+.4f} pp.",
        "",
        "## 5. Component-size analysis",
        "",
        "| Variant | Small | Medium | Large |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        lines.append(
            f"| {variant} | {100*component_absolute[variant]['small']:.4f} | "
            f"{100*component_absolute[variant]['medium']:.4f} | "
            f"{100*component_absolute[variant]['large']:.4f} |"
        )
    lines += [
        "",
        "## 6. CAM hierarchy",
        "",
        "| Stage | C0 mIoU | W1 mIoU | W2 mIoU | W2-C0 | W2-W1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in multiscale_rows:
        lines.append(
            f"| {row['stage']} | {100*row['C0_mIoU']:.4f} | "
            f"{100*row['W1_mIoU']:.4f} | {100*row['W2_mIoU']:.4f} | "
            f"{row['W2_minus_C0_mIoU_pp']:+.4f} | "
            f"{row['W2_minus_W1_mIoU_pp']:+.4f} |"
        )
    lines += [
        "",
        "## 7. Feature magnitude audit",
        "",
        "| Variant/operator | Input RMS | Context output RMS | Rectification RMS | Output/Input |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        value = feature_audit[variant]
        lines.append(
            f"| {variant}/{value['operator']} | {value['input_rms']['mean']:.6f} | "
            f"{value['context_output_rms']['mean']:.6f} | "
            f"{value['rectification_rms_absolute']['mean']:.6f} | "
            f"{value['output_input_rms']['mean']:.6f} |"
        )
    lines += [
        "",
        f"- Final SC-WD/CH rectification-strength ratio: {final_strength_ratio:.6f}.",
        "",
        "## 8. Preregistered gates",
        "",
        f"- Gate A — Strength Recovery: {'PASS' if gate_a else 'FAIL'} ({final_strength_ratio:.6f}; required 0.9-1.1).",
        f"- Gate B — Mechanism Preservation: {'PASS' if gate_b else 'FAIL'} (Boundary {boundary_w2_c0:+.4f} pp; Interior {interior_w2_c0:+.4f} pp).",
        f"- Gate C — Model Improvement: {'PASS' if gate_c else 'FAIL'} (mIoU {delta_miou_w2_c0:+.4f} pp; required +0.10 pp).",
        f"- CAM28_1 recovery versus W1: {'YES' if cam28_recovered else 'NO'} ({cam28_w2_w1:+.4f} pp).",
        "",
        "## 9. Interpretation",
        "",
        interpretation,
        "",
        "## 10. Checkpoints",
        "",
    ]
    for variant in VARIANTS:
        lines.append(
            f"- {variant}: `{completions[variant]['checkpoint_sha256']}`"
        )
    lines += [
        "",
        "No test, LUAD, other seed, checkpoint selection, scalar tuning or additional model variant was used.",
        "",
        f"DECISION = {decision}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "scale": scale,
        "strength_ratio": final_strength_ratio,
        "mIoU_W2_minus_C0_pp": delta_miou_w2_c0,
        "boundary_W2_minus_C0_pp": boundary_w2_c0,
        "interior_W2_minus_C0_pp": interior_w2_c0,
        "CAM28_1_W2_minus_W1_pp": cam28_w2_w1,
    }, indent=2), flush=True)
    print(f"DECISION = {decision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--c0-dir", required=True)
    parser.add_argument("--w1-dir", required=True)
    parser.add_argument("--w2-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
