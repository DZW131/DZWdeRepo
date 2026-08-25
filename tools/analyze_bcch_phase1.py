#!/usr/bin/env python3
"""Final matched validation analysis for EXP-BCCH-001."""

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
    foreground_boundary_distance,
    sha256_file,
    write_json,
)


VARIANTS = ("C0", "W1", "BC-CH")
STAGES = ("56", "28_1", "28_2", "deep", "final")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def aggregate_component(rows):
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


class ZoneIoUAccumulator:
    def __init__(self):
        self.hist = {
            "boundary_le_7": np.zeros((4, 4), dtype=np.float64),
            "interior_ge_8": np.zeros((4, 4), dtype=np.float64),
        }

    def update(self, truth, prediction):
        zones = foreground_boundary_distance(truth)
        for name, mask in zones.items():
            encoded = 4 * truth[mask].astype(np.int64) + prediction[mask].astype(np.int64)
            self.hist[name] += np.bincount(encoded, minlength=16).reshape(4, 4)

    def result(self):
        output = {}
        for name, hist in self.hist.items():
            diagonal = np.diag(hist)
            union = hist.sum(1) + hist.sum(0) - diagonal
            iou = np.divide(
                diagonal,
                union,
                out=np.full_like(diagonal, np.nan),
                where=union > 0,
            )
            output[name] = {
                "mIoU": float(np.nanmean(iou)),
                "class_iou": {str(index): float(iou[index]) for index in range(4)},
                "pixels": int(hist.sum()),
            }
        return output


def fmt(value):
    return f"{100.0 * value:.4f}"


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    directories = {
        "C0": Path(args.c0_dir),
        "W1": Path(args.w1_dir),
        "BC-CH": Path(args.bcch_dir),
    }
    completions = {name: read_json(path / "complete.json") for name, path in directories.items()}
    for name in ("C0", "W1"):
        value = completions[name]
        if value.get("status") != "WDCH_MATCHED_BRANCH_COMPLETE" or value.get("branch") != name:
            raise AssertionError(f"Invalid locked {name} completion")
    bcch = completions["BC-CH"]
    if bcch.get("status") != "BCCH_MATCHED_COMPLETE":
        raise AssertionError("BCCH run is incomplete")
    for name, value in completions.items():
        if value.get("epochs") != [21, 22, 23, 24, 25]:
            raise AssertionError(f"{name}: wrong epochs")
        if value.get("test_used") or value["final_validation"].get("test_used"):
            raise AssertionError(f"{name}: test use detected")
        if value["final_validation"].get("epoch") != 25:
            raise AssertionError(f"{name}: not Epoch25 FINAL")

    provenance = read_json(directories["BC-CH"] / "provenance.json")
    preflight = read_json(args.preflight)
    if preflight.get("status") != "BCCH_PREFLIGHT_PASS":
        raise AssertionError("BCCH preflight did not pass")
    common_sha = sha256_file(args.common_checkpoint)
    schedule_sha = sha256_file(args.schedule)
    if provenance["common_checkpoint_sha256"] != common_sha:
        raise AssertionError("Common checkpoint provenance differs")
    if provenance["schedule_sha256"] != schedule_sha:
        raise AssertionError("Schedule provenance differs")

    predictions = {
        name: load_predictions(path / "predictions" / "epoch25_validation.npz")
        for name, path in directories.items()
    }
    reference = predictions["C0"]
    for name, values in predictions.items():
        if not np.array_equal(reference["image_ids"], values["image_ids"]):
            raise AssertionError(f"{name}: validation order differs")
        if not np.array_equal(reference["truths"], values["truths"]):
            raise AssertionError(f"{name}: validation truths differ")

    thresholds = component_thresholds(args.val_root)
    spatial = {}
    objects = {}
    for name in ("W1", "BC-CH"):
        zone = PairedZoneAccumulator()
        components = PairedComponentAccumulator(thresholds)
        for truth, base_prediction, candidate_prediction in zip(
            reference["truths"],
            reference["predictions"],
            predictions[name]["predictions"],
        ):
            zone.update(truth, base_prediction, candidate_prediction)
            components.update(truth, base_prediction, candidate_prediction)
        spatial[name] = zone.result()
        objects[name] = aggregate_component(components.result())

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
                "boundary_accuracy": boundary,
                "boundary_accuracy_delta_pp": boundary_delta,
                "interior_accuracy": interior,
                "interior_accuracy_delta_pp": interior_delta,
                "boundary_mIoU": zone_iou[name]["boundary_le_7"]["mIoU"],
                "interior_mIoU": zone_iou[name]["interior_ge_8"]["mIoU"],
                "small": small,
                "medium": medium,
                "large": large,
            }
        )

    cam_rows = []
    class_rows = []
    for name in VARIANTS:
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
        final = evaluations[name]["scores"]["final"]
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

    bcch_main = next(row for row in main_rows if row["variant"] == "BC-CH")
    gates = {
        "A_boundary": {
            "pass": bcch_main["boundary_accuracy_delta_pp"] > 0.0,
            "delta_pp": bcch_main["boundary_accuracy_delta_pp"],
            "criterion": "> 0.0 pp",
        },
        "B_interior": {
            "pass": bcch_main["interior_accuracy_delta_pp"] > -0.2,
            "delta_pp": bcch_main["interior_accuracy_delta_pp"],
            "criterion": "> -0.2 pp",
        },
        "C_overall": {
            "pass": bcch_main["delta_mIoU_pp"] > 0.0,
            "delta_pp": bcch_main["delta_mIoU_pp"],
            "criterion": "> 0.0 pp",
        },
    }
    if all(value["pass"] for value in gates.values()):
        decision = "GO"
        interpretation = "Wavelet-derived boundary-aware selective CH passes all three mechanism gates."
        next_step = "Proceed to a separately preregistered contrastive boundary-aware CH experiment."
    elif gates["A_boundary"]["pass"] and gates["B_interior"]["pass"]:
        decision = "NEXT_STEP"
        interpretation = "BC-CH preserves a boundary mechanism signal without positive overall utility."
        next_step = "Semantic discrimination is still required; design contrastive affinity learning as a new experiment."
    else:
        decision = "STOP"
        interpretation = "Detached HF energy alone does not provide a valid selective-CH routing mechanism."
        next_step = "Do not add contrastive learning on this router; reformulate learned boundary semantics first."

    feature = evaluations["BC-CH"]["feature_diagnostics"]
    summary = {
        "experiment_id": "EXP-BCCH-001",
        "decision": decision,
        "interpretation": interpretation,
        "next_step": next_step,
        "main_metrics": main_rows,
        "cam_metrics": cam_rows,
        "per_class": class_rows,
        "spatial": spatial,
        "zone_iou": zone_iou,
        "object_size": objects,
        "feature_diagnostics": feature,
        "gates": gates,
        "training_implementation_commit": provenance["source_commit"],
        "common_checkpoint_sha256": common_sha,
        "schedule_sha256": schedule_sha,
        "checkpoints": {
            name: {
                "path": completions[name]["checkpoint"],
                "sha256": completions[name]["checkpoint_sha256"],
            }
            for name in VARIANTS
        },
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "bcch_phase1_summary.json", summary)

    lines = [
        "# BCCH Phase-1 Boundary-Aware CH Mechanism Validation",
        "",
        "## 1. Frozen protocol",
        "",
        "- Experiment: `EXP-BCCH-001`; BCSS seed42; Epoch20 common state → Epoch21–25 matched continuation.",
        "- C0/W1 are reused SHA-locked matched artifacts; only BC-CH is newly trained.",
        "- Only HFRM28_1 CH changes; original CH15 parameter and optimizer state restore exactly.",
        "- `E_HF=sqrt(LH²+HL²+HH²)` → channel mean → per-image spatial min-max → bilinear upsample; `B` is detached and `alpha=1-B`.",
        "- No new trainable parameter, classifier, GSR change, loss, contrastive objective, inference change or metric change.",
        "- Batch20, 224×224, BF16, same batch/augmentation/model seeds, official optimizer/poly schedule and Epoch25 FINAL only.",
        "- No test, LUAD, other seed, best-checkpoint selection or validation tuning.",
        "",
        "## 2. Overall, spatial and object-size results",
        "",
        "The preregistered gates use the exact earlier WD-CH boundary/interior pixel-accuracy definition (`≤7 px` / `≥8 px`). Zone-restricted mIoU is additionally reported so it is not mislabeled as accuracy.",
        "",
        "| Variant | mIoU | Δ pp | mDice | Boundary acc. | Δ pp | Interior acc. | Δ pp | Boundary mIoU | Interior mIoU | Small | Medium | Large |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['mIoU'])} | {row['delta_mIoU_pp']:+.4f} | {fmt(row['mDice'])} | "
            f"{fmt(row['boundary_accuracy'])} | {row['boundary_accuracy_delta_pp']:+.4f} | "
            f"{fmt(row['interior_accuracy'])} | {row['interior_accuracy_delta_pp']:+.4f} | "
            f"{fmt(row['boundary_mIoU'])} | {fmt(row['interior_mIoU'])} | "
            f"{fmt(row['small'])} | {fmt(row['medium'])} | {fmt(row['large'])} |"
        )
    lines += [
        "",
        "## 3. CAM hierarchy",
        "",
        "| Variant | CAM56 | CAM28_1 | Δ CAM28_1 pp | CAM28_2 | Deep | Final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        stage = {row["stage"]: row for row in cam_rows if row["variant"] == name}
        lines.append(
            f"| {name} | {fmt(stage['56']['mIoU'])} | {fmt(stage['28_1']['mIoU'])} | "
            f"{stage['28_1']['delta_mIoU_pp']:+.4f} | {fmt(stage['28_2']['mIoU'])} | "
            f"{fmt(stage['deep']['mIoU'])} | {fmt(stage['final']['mIoU'])} |"
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
        "## 5. Feature mechanism statistics",
        "",
        "Residual RMS values are normalized by input-feature RMS. Boundary/interior residuals use continuous `B` / `1-B` weighting and therefore require no post-hoc threshold.",
        "",
        f"- Raw CH residual RMS: {feature['raw_ch_residual_rms']['mean']:.6f} ± {feature['raw_ch_residual_rms']['std']:.6f}.",
        f"- Selected BC-CH residual RMS: {feature['selected_ch_residual_rms']['mean']:.6f} ± {feature['selected_ch_residual_rms']['std']:.6f}.",
        f"- Boundary selected residual RMS: {feature['boundary_selected_residual_rms']['mean']:.6f} ± {feature['boundary_selected_residual_rms']['std']:.6f}; retention={feature['boundary_residual_retention']['mean']:.6f}.",
        f"- Interior selected residual RMS: {feature['interior_selected_residual_rms']['mean']:.6f} ± {feature['interior_selected_residual_rms']['std']:.6f}; retention={feature['interior_residual_retention']['mean']:.6f}.",
        f"- Boundary map mean/std: {feature['boundary_map_mean']['mean']:.6f} / {feature['boundary_map_std']['mean']:.6f}.",
        f"- Alpha mean/std: {feature['alpha_mean']['mean']:.6f} / {feature['alpha_std']['mean']:.6f}.",
        f"- Final gamma_context / gamma_veto: {feature['gamma_context']:.8f} / {feature['gamma_veto']:.8f}.",
        "",
        "## 6. Preregistered gates",
        "",
        "| Gate | Observed Δ | Criterion | Result |",
        "|---|---:|---:|:---:|",
    ]
    for label, value in gates.items():
        lines.append(
            f"| {label} | {value['delta_pp']:+.4f} pp | {value['criterion']} | {'PASS' if value['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 7. Engineering and reproducibility evidence",
        "",
        f"- Preflight: `{preflight['status']}`; real batch={preflight['batch_size']}, BF16, official loss, no optimizer step.",
        f"- Training implementation commit: `{provenance['source_commit']}`.",
        f"- Common Epoch20 SHA256: `{common_sha}`.",
        f"- Schedule SHA256: `{schedule_sha}`.",
    ]
    for name in VARIANTS:
        lines.append(f"- {name} Epoch25 SHA256: `{completions[name]['checkpoint_sha256']}`.")
    lines += [
        "- Prediction order and GT masks are byte-equal across C0/W1/BC-CH.",
        "",
        "## 8. Interpretation and decision",
        "",
        interpretation,
        "",
        f"NEXT_STEP: {next_step}",
        "",
        f"DECISION = {decision}",
        "",
        "STOP.",
    ]
    (output / "bcch_phase1_boundary_aware_final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print(f"DECISION = {decision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--c0-dir", required=True)
    parser.add_argument("--w1-dir", required=True)
    parser.add_argument("--bcch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
