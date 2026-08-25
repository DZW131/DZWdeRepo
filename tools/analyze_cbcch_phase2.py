#!/usr/bin/env python3
"""Final matched validation analysis for EXP-CBCCH-002."""

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
    aggregate_component,
    fmt,
    load_predictions,
    read_json,
)
from tools.wdch_common import (
    PairedComponentAccumulator,
    PairedZoneAccumulator,
    component_thresholds,
    sha256_file,
    write_json,
)


VARIANTS = ("C0", "W1", "BC-CH", "A2", "A3")
STAGES = ("56", "28_1", "28_2", "deep", "final")


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    directories = {
        "C0": Path(args.c0_dir),
        "W1": Path(args.w1_dir),
        "BC-CH": Path(args.bcch_dir),
        "A2": Path(args.a2_dir),
        "A3": Path(args.a3_dir),
    }
    completions = {
        name: read_json(directory / "complete.json")
        for name, directory in directories.items()
    }
    for name in ("C0", "W1"):
        value = completions[name]
        if value.get("status") != "WDCH_MATCHED_BRANCH_COMPLETE" or value.get("branch") != name:
            raise AssertionError(f"Invalid locked {name}")
    if completions["BC-CH"].get("status") != "BCCH_MATCHED_COMPLETE":
        raise AssertionError("Invalid locked BC-CH")
    for name in ("A2", "A3"):
        value = completions[name]
        if value.get("status") != "CBCCH_MATCHED_COMPLETE" or value.get("variant") != name:
            raise AssertionError(f"Invalid {name}")
    for name, value in completions.items():
        if value.get("epochs") != [21, 22, 23, 24, 25]:
            raise AssertionError(f"{name}: wrong epochs")
        final = value["final_validation"]
        if value.get("test_used") or final.get("test_used"):
            raise AssertionError(f"{name}: test use detected")
        if final.get("epoch") != 25:
            raise AssertionError(f"{name}: final checkpoint rule violated")

    preflight = read_json(args.preflight)
    if preflight.get("status") != "CBCCH_PREFLIGHT_PASS":
        raise AssertionError("CBCCH preflight did not pass")
    common_sha = sha256_file(args.common_checkpoint)
    schedule_sha = sha256_file(args.schedule)
    for name in ("A2", "A3"):
        provenance = read_json(directories[name] / "provenance.json")
        if provenance["common_checkpoint_sha256"] != common_sha:
            raise AssertionError(f"{name}: common checkpoint differs")
        if provenance["schedule_sha256"] != schedule_sha:
            raise AssertionError(f"{name}: schedule differs")

    predictions = {
        name: load_predictions(directory / "predictions" / "epoch25_validation.npz")
        for name, directory in directories.items()
    }
    reference = predictions["C0"]
    for name, values in predictions.items():
        if not np.array_equal(reference["image_ids"], values["image_ids"]):
            raise AssertionError(f"{name}: image order differs")
        if not np.array_equal(reference["truths"], values["truths"]):
            raise AssertionError(f"{name}: truths differ")

    thresholds = component_thresholds(args.val_root)
    spatial = {}
    objects = {}
    for name in VARIANTS[1:]:
        zones = PairedZoneAccumulator()
        components = PairedComponentAccumulator(thresholds)
        for truth, base, candidate in zip(
            reference["truths"],
            reference["predictions"],
            predictions[name]["predictions"],
        ):
            zones.update(truth, base, candidate)
            components.update(truth, base, candidate)
        spatial[name] = zones.result()
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
    baseline_boundary = spatial["W1"]["boundary_le_7"]["base_accuracy"]
    baseline_interior = spatial["W1"]["interior_ge_8"]["base_accuracy"]
    baseline_sizes = {
        size: objects["W1"][size]["base_recall"]
        for size in ("small", "medium", "large")
    }
    main_rows = []
    for name in VARIANTS:
        final = evaluations[name]["scores"]["final"]
        if name == "C0":
            boundary = baseline_boundary
            interior = baseline_interior
            sizes = baseline_sizes
        else:
            boundary = spatial[name]["boundary_le_7"]["candidate_accuracy"]
            interior = spatial[name]["interior_ge_8"]["candidate_accuracy"]
            sizes = {
                size: objects[name][size]["candidate_recall"]
                for size in ("small", "medium", "large")
            }
        main_rows.append(
            {
                "variant": name,
                "mIoU": final["mIoU"],
                "mDice": final["mDice"],
                "delta_mIoU_pp": 100.0 * (final["mIoU"] - c0_final["mIoU"]),
                "boundary_accuracy": boundary,
                "boundary_accuracy_delta_pp": 100.0 * (boundary - baseline_boundary),
                "interior_accuracy": interior,
                "interior_accuracy_delta_pp": 100.0 * (interior - baseline_interior),
                "boundary_mIoU": zone_iou[name]["boundary_le_7"]["mIoU"],
                "interior_mIoU": zone_iou[name]["interior_ge_8"]["mIoU"],
                **sizes,
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
                    "delta_mIoU_pp": 100.0 * (value["mIoU"] - base["mIoU"]),
                }
            )

    a3_main = next(row for row in main_rows if row["variant"] == "A3")
    c0_cam28 = evaluations["C0"]["scores"]["28_1"]["mIoU"]
    a3_cam28 = evaluations["A3"]["scores"]["28_1"]["mIoU"]
    gates = {
        "A_CAM28_1_recovery": {
            "pass": a3_cam28 > c0_cam28 - 0.001,
            "observed": a3_cam28,
            "threshold": c0_cam28 - 0.001,
            "margin_pp": 100.0 * (a3_cam28 - (c0_cam28 - 0.001)),
            "criterion": "A3 CAM28_1 > C0 CAM28_1 - 0.1 pp",
        },
        "B_boundary_accuracy": {
            "pass": a3_main["boundary_accuracy"] > baseline_boundary,
            "observed": a3_main["boundary_accuracy"],
            "threshold": baseline_boundary,
            "margin_pp": a3_main["boundary_accuracy_delta_pp"],
            "criterion": "A3 boundary accuracy > C0",
        },
        "C_final_mIoU": {
            "pass": a3_main["mIoU"] > c0_final["mIoU"],
            "observed": a3_main["mIoU"],
            "threshold": c0_final["mIoU"],
            "margin_pp": a3_main["delta_mIoU_pp"],
            "criterion": "A3 final mIoU > C0",
        },
    }
    if all(value["pass"] for value in gates.values()):
        decision = "SUCCESS"
        interpretation = "CBCCH passes semantic recovery, boundary preservation and overall utility gates."
    elif gates["B_boundary_accuracy"]["pass"] and not gates["C_final_mIoU"]["pass"]:
        decision = "BOUNDARY_ONLY"
        interpretation = "Boundary behavior improves, but contrastive affinity does not deliver positive overall utility."
    else:
        decision = "NO_IMPROVEMENT"
        interpretation = "The frozen contrastive pair construction does not validate CBCCH utility."

    histories = {
        name: read_json(directories[name] / "training_history.json")
        for name in ("A2", "A3")
    }
    mechanisms = {
        name: evaluations[name]["feature_diagnostics"] for name in ("A2", "A3")
    }
    summary = {
        "experiment_id": "EXP-CBCCH-002",
        "decision": decision,
        "interpretation": interpretation,
        "main_metrics": main_rows,
        "cam_metrics": cam_rows,
        "spatial": spatial,
        "zone_iou": zone_iou,
        "object_size": objects,
        "mechanism": mechanisms,
        "training_history": histories,
        "gates": gates,
        "common_checkpoint_sha256": common_sha,
        "schedule_sha256": schedule_sha,
        "checkpoints": {
            name: {
                "path": value["checkpoint"],
                "sha256": value["checkpoint_sha256"],
            }
            for name, value in completions.items()
        },
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "cbcch_phase2_summary.json", summary)

    lines = [
        "# CBCCH Phase-2 Contrastive Boundary Affinity Learning",
        "",
        "## 1. Frozen protocol",
        "",
        "- Experiment: `EXP-CBCCH-002`; BCSS seed42; locked common Epoch20 → Epoch21–25 matched continuation.",
        "- C0, W1 and Phase-1 BC-CH are reused SHA-locked artifacts; A2 and A3 are newly trained.",
        "- A2: local semantic-affinity propagation at every pixel; every valid pixel participates in contrastive learning.",
        "- A3: `Y=(1-B)P_affinity+B*F`; only exact top-20% detached-B anchors participate in contrastive learning.",
        "- Local neighborhood=15×15; `z_s` reuses `ic1`; `z_h` is mean(|LH|), mean(|HL|), mean(|HH|); one deterministic positive/negative; τ=0.07.",
        "- `L=L_official+0.1 L_con`; no new trainable parameters. Same schedule, batches, augmentation/model seeds, optimizer, BF16 and Epoch25 FINAL rule.",
        "- No test, LUAD, alternate seed, best-checkpoint selection or validation tuning.",
        "",
        "## 2. Overall and spatial validation results",
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
        selected = {row["stage"]: row for row in cam_rows if row["variant"] == name}
        lines.append(
            f"| {name} | {fmt(selected['56']['mIoU'])} | {fmt(selected['28_1']['mIoU'])} | "
            f"{selected['28_1']['delta_mIoU_pp']:+.4f} | {fmt(selected['28_2']['mIoU'])} | "
            f"{fmt(selected['deep']['mIoU'])} | {fmt(selected['final']['mIoU'])} |"
        )
    lines += ["", "## 4. Contrastive and affinity mechanism", ""]
    for name in ("A2", "A3"):
        feature = mechanisms[name]
        final_train = histories[name][-1]
        lines += [
            f"### {name}",
            "",
            f"- Epoch25 contrastive loss={final_train['contrastive_loss']:.6f}; valid-anchor fraction={final_train['valid_anchor_fraction']:.6f}.",
            f"- Positive/negative semantic similarity={final_train['positive_similarity']:.6f}/{final_train['negative_similarity']:.6f}; margin={final_train['similarity_margin']:.6f}.",
            f"- Affinity entropy={feature['affinity_entropy']['mean']:.6f}±{feature['affinity_entropy']['std']:.6f}; max={feature['affinity_max']['mean']:.6f}; self={feature['affinity_self']['mean']:.6f}; effective neighbors={feature['affinity_effective_neighbors']['mean']:.3f}.",
            f"- Boundary/interior propagation RMS={feature['boundary_propagation_rms']['mean']:.6f}/{feature['interior_propagation_rms']['mean']:.6f}; global residual={feature['propagation_residual_rms']['mean']:.6f}.",
            f"- Final gamma_context/gamma_veto={feature['gamma_context']:.8f}/{feature['gamma_veto']:.8f}.",
            "",
        ]
    lines += [
        "## 5. Preregistered gates",
        "",
        "| Gate | Margin | Criterion | Result |",
        "|---|---:|---|:---:|",
    ]
    for name, value in gates.items():
        lines.append(
            f"| {name} | {value['margin_pp']:+.4f} pp | {value['criterion']} | {'PASS' if value['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 6. Reproducibility and resource evidence",
        "",
        f"- Preflight: `{preflight['status']}`; real batch20 BF16; official+contrastive loss; no optimizer step.",
        f"- Common Epoch20 SHA256: `{common_sha}`.",
        f"- Schedule SHA256: `{schedule_sha}`.",
    ]
    for name in VARIANTS:
        lines.append(f"- {name} Epoch25 SHA256: `{completions[name]['checkpoint_sha256']}`.")
    for name in ("A2", "A3"):
        lines.append(
            f"- {name}: continuation runtime={completions[name]['runtime_seconds']/3600:.3f} h; peak CUDA memory={completions[name]['peak_cuda_memory_bytes']/2**30:.3f} GiB."
        )
    lines += [
        "- Prediction order and validation ground truth are byte-equal across all five variants.",
        "- Trainable parameter count is exactly C0 for A2/A3; legacy CH15 state is restored for parity but intentionally dormant under the frozen Phase-2 aggregation equation.",
        "",
        "## 7. Decision",
        "",
        interpretation,
        "",
        f"DECISION = {decision}",
        "",
        "STOP.",
    ]
    report = output / "cbcch_phase2_contrastive_affinity_final_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    parser.add_argument("--a2-dir", required=True)
    parser.add_argument("--a3-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
