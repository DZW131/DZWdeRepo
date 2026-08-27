#!/usr/bin/env python3
"""Analyze matched LW-SHR continuations and render the preregistered report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lw_shr_common import (
    C0_FINAL_SHA256,
    ComponentMetricAccumulator,
    ZoneMetricAccumulator,
    component_thresholds,
    official_histogram,
    paired_image_bootstrap_miou,
    read_json,
    sha256_file,
    write_json,
)

STAGES = ("56", "28_1", "28_2", "deep", "final")
VARIANTS = ("A1", "A2", "A3")


def percentage(value):
    return 100.0 * float(value)


def delta_pp(left, right):
    return percentage(left) - percentage(right)


def load_prediction_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    required = {"image_ids", "predictions", "truths"}
    if not required.issubset(result):
        raise AssertionError(f"Prediction archive missing {required.difference(result)}")
    return result


def align_to_reference(reference, candidate):
    reference_ids = [str(value) for value in reference["image_ids"]]
    candidate_ids = [str(value) for value in candidate["image_ids"]]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise AssertionError("Duplicate candidate image IDs")
    lookup = {image_id: index for index, image_id in enumerate(candidate_ids)}
    if set(reference_ids) != set(candidate_ids):
        raise AssertionError("C0/candidate validation image IDs differ")
    indices = np.asarray([lookup[image_id] for image_id in reference_ids])
    return {
        **candidate,
        "image_ids": candidate["image_ids"][indices],
        "predictions": candidate["predictions"][indices],
        "truths": candidate["truths"][indices],
        "histograms": candidate.get("histograms", None)
        if "histograms" not in candidate
        else candidate["histograms"][indices],
    }


def structural_from_archive(archive, thresholds):
    zones = ZoneMetricAccumulator()
    components = ComponentMetricAccumulator(thresholds)
    histograms = []
    for truth, prediction in zip(archive["truths"], archive["predictions"]):
        zones.update(truth, prediction)
        components.update(truth, prediction)
        histograms.append(official_histogram(truth, prediction))
    return {
        "zones": zones.result(),
        "components": components.result(),
        "histograms": np.stack(histograms),
    }


def c0_record(c0_dir, val_root, cache_dir):
    c0_dir = Path(c0_dir)
    complete = read_json(c0_dir / "complete.json")
    checkpoint = Path(complete["checkpoint"])
    if sha256_file(checkpoint) != C0_FINAL_SHA256:
        raise AssertionError("Locked C0 final checkpoint SHA256 mismatch")
    archive = load_prediction_archive(
        c0_dir / "predictions" / "epoch25_validation.npz"
    )
    cache = Path(cache_dir) / "c0_structural.json"
    histogram_cache = Path(cache_dir) / "c0_image_histograms.npy"
    if cache.exists() and histogram_cache.exists():
        structural = read_json(cache)
        histograms = np.load(histogram_cache, allow_pickle=False)
    else:
        computed = structural_from_archive(archive, component_thresholds(val_root))
        structural = {
            "zones": computed["zones"],
            "components": computed["components"],
        }
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        write_json(cache, structural)
        np.save(histogram_cache, computed["histograms"])
        histograms = computed["histograms"]
    return {
        "variant": "C0",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": C0_FINAL_SHA256,
        "scores": complete["final_validation"]["scores"],
        "runtime": complete["final_validation"]["runtime"],
        "structural": structural,
        "mechanism": None,
        "archive": archive,
        "histograms": histograms,
        "complete": complete,
    }


def candidate_record(root, variant, reference, thresholds):
    root = Path(root) / variant
    completion = read_json(root / "completion.json")
    if completion.get("status") != "COMPLETE" or completion.get("variant") != variant:
        raise AssertionError(f"Incomplete {variant} artifact")
    if completion.get("test_used") or completion.get("luad_used"):
        raise AssertionError("Forbidden dataset evaluation is recorded")
    if sha256_file(completion["checkpoint"]) != completion["checkpoint_sha256"]:
        raise AssertionError(f"{variant} checkpoint SHA256 mismatch")
    archive = align_to_reference(
        reference, load_prediction_archive(completion["predictions"])
    )
    if not np.array_equal(reference["truths"], archive["truths"]):
        raise AssertionError(f"{variant} validation truths differ from C0")
    computed = structural_from_archive(archive, thresholds)
    # Recompute structural metrics from the frozen prediction artifact; they must
    # agree with the online evaluator and provide paired per-image histograms.
    online = completion["history"][-1]["validation"]["structural"]
    for zone in ("boundary_le_7", "interior_ge_8"):
        if abs(
            computed["zones"][zone]["accuracy"]
            - online["zones"][zone]["accuracy"]
        ) > 1.0e-12:
            raise AssertionError(f"{variant} cached/online zone mismatch")
    validation = completion["history"][-1]["validation"]
    return {
        "variant": variant,
        "checkpoint": completion["checkpoint"],
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "scores": validation["scores"],
        "runtime": validation["runtime"],
        "structural": {
            "zones": computed["zones"],
            "components": computed["components"],
        },
        "mechanism": validation["mechanism"],
        "archive": archive,
        "histograms": computed["histograms"],
        "complete": completion,
    }


def preregistered_gates(c0, candidate):
    overall_delta = delta_pp(
        candidate["scores"]["final"]["mIoU"], c0["scores"]["final"]["mIoU"]
    )
    bootstrap = paired_image_bootstrap_miou(
        c0["histograms"], candidate["histograms"], resamples=10000, seed=42
    )
    cam_delta = delta_pp(
        candidate["scores"]["28_1"]["mIoU"], c0["scores"]["28_1"]["mIoU"]
    )
    interior_delta = delta_pp(
        candidate["structural"]["zones"]["interior_ge_8"]["accuracy"],
        c0["structural"]["zones"]["interior_ge_8"]["accuracy"],
    )
    boundary_delta = delta_pp(
        candidate["structural"]["zones"]["boundary_le_7"]["accuracy"],
        c0["structural"]["zones"]["boundary_le_7"]["accuracy"],
    )
    small_delta = delta_pp(
        candidate["structural"]["components"]["aggregate"]["small"][
            "historical_component_recall"
        ],
        c0["structural"]["components"]["aggregate"]["small"][
            "historical_component_recall"
        ],
    )
    mechanism = candidate["mechanism"]
    spatial_std = mechanism["spatial_std"]["mean"]
    channel_std = mechanism["channel_std"]["mean"]
    gates = {
        "A_overall_utility": {
            "pass": overall_delta > 0.0 and bootstrap["ci95_low_pp"] >= 0.0,
            "delta_mIoU_pp": overall_delta,
            "bootstrap": bootstrap,
        },
        "B_semantic_safety": {
            "pass": cam_delta >= -0.10 and interior_delta >= -0.10,
            "cam28_1_delta_pp": cam_delta,
            "interior_accuracy_delta_pp": interior_delta,
        },
        "C_structural_mechanism": {
            "pass": boundary_delta > 0.0 or small_delta > 0.0,
            "boundary_accuracy_delta_pp": boundary_delta,
            "historical_small_component_recall_delta_pp": small_delta,
            "protocol_note": (
                "The historical object-size statistic is pixel-weighted component "
                "recall, not component mIoU; size-restricted mIoU is diagnostic only."
            ),
        },
        "D_non_degenerate_gate": {
            "pass": spatial_std > 0.0 and channel_std > 0.0,
            "spatial_std": spatial_std,
            "channel_std": channel_std,
        },
    }
    gates["all_pass"] = all(value["pass"] for value in gates.values())
    gates["raw_positive_signal_for_A3"] = (
        overall_delta > 0.0 or boundary_delta > 0.0 or small_delta > 0.0
    )
    return gates


def analyze(args):
    cache_dir = Path(args.output_dir) / "analysis"
    c0 = c0_record(args.c0_dir, args.val_root, cache_dir)
    thresholds = component_thresholds(args.val_root)
    candidates = {}
    for variant in VARIANTS:
        if (Path(args.experiment_root) / variant / "completion.json").exists():
            candidates[variant] = candidate_record(
                args.experiment_root, variant, c0["archive"], thresholds
            )
    gates = {
        variant: preregistered_gates(c0, record)
        for variant, record in candidates.items()
    }
    a3_unlocked = any(
        gates[variant]["raw_positive_signal_for_A3"]
        for variant in ("A1", "A2")
        if variant in gates
    )
    result = {
        "c0": {key: value for key, value in c0.items() if key not in ("archive", "histograms")},
        "variants": {
            variant: {
                key: value
                for key, value in record.items()
                if key not in ("archive", "histograms", "complete")
            }
            for variant, record in candidates.items()
        },
        "gates": gates,
        "a3_unlocked": a3_unlocked,
        "a3_unlock_rule": (
            "Run A3 only if A1 or A2 has delta mIoU > 0, boundary accuracy "
            "delta > 0, or historical small-component recall delta > 0."
        ),
        "final_decision": (
            "GO_PHASE2" if any(row["all_pass"] for row in gates.values()) else "NO_GO"
        ),
        "test_used": False,
        "luad_used": False,
    }
    write_json(Path(args.output_dir) / "lw_shr_phase1_analysis.json", result)
    return result, c0, candidates


def metric_row(name, record, c0):
    score = record["scores"]["final"]
    baseline = c0["scores"]["final"]
    return (
        f"| {name} | {percentage(score['mIoU']):.4f} | "
        f"{delta_pp(score['mIoU'], baseline['mIoU']):+.4f} | "
        f"{percentage(score['mDice']):.4f} |"
    )


def render_report(args, result, c0, candidates):
    if not all(variant in candidates for variant in ("A1", "A2")):
        raise AssertionError("Final report requires completed A1 and A2")
    if result["a3_unlocked"] and "A3" not in candidates:
        raise AssertionError("A3 was unlocked but has not completed")

    lines = [
        "# LW-SHR Phase-1 LWTformer Transfer Utility Report",
        "",
        "## 1. Commit hash",
        "",
        f"- Implementation commit: `{next(iter(candidates.values()))['complete']['source_commit']}`",
        f"- Pure official A0 base: `{next(iter(candidates.values()))['complete']['a0_commit']}`",
        "",
        "## 2. Exact commands",
        "",
        "```bash",
        "python tools/run_lw_shr_phase0.py --common-checkpoint <COMMON> --schedule <SCHEDULE> --train-root <BCSS_TRAIN> --output-dir <OUTPUT>/phase0",
        "python tools/train_lw_shr_matched.py --variant A1 --common-checkpoint <COMMON> --schedule <SCHEDULE> --phase0-summary <OUTPUT>/phase0/lw_shr_phase0_summary.json --train-root <BCSS_TRAIN> --val-root <BCSS_VAL> --output-dir <OUTPUT>/matched",
        "python tools/train_lw_shr_matched.py --variant A2 --common-checkpoint <COMMON> --schedule <SCHEDULE> --phase0-summary <OUTPUT>/phase0/lw_shr_phase0_summary.json --train-root <BCSS_TRAIN> --val-root <BCSS_VAL> --output-dir <OUTPUT>/matched",
    ]
    if "A3" in candidates:
        lines.append(
            "python tools/train_lw_shr_matched.py --variant A3 --common-checkpoint <COMMON> --schedule <SCHEDULE> --phase0-summary <OUTPUT>/phase0/lw_shr_phase0_summary.json --train-root <BCSS_TRAIN> --val-root <BCSS_VAL> --output-dir <OUTPUT>/matched"
        )
    lines += [
        "```",
        "",
        "All continuations independently start from the same frozen Epoch-20 state; no test or LUAD data were used.",
        "",
        "## 3. Checkpoint SHA256",
        "",
        f"- C0: `{c0['checkpoint_sha256']}`",
    ]
    for variant in VARIANTS:
        if variant in candidates:
            lines.append(f"- {variant}: `{candidates[variant]['checkpoint_sha256']}`")
    lines += [
        "",
        "## 4. Baseline-equivalence audit",
        "",
        "Phase-0 verified FP32 maximum absolute output difference below `1e-5` for A1/A2/A3 at identity initialization. The original `wavelet_hfrm_mode=none` path remains bitwise identical to A0. A2/A3 filter gradients are expected to be zero at step 1 because the final gate projection is zero-initialized, and were required to become positive by step 2.",
        "",
        "## 5. C0/A1/A2/A3 overall comparison",
        "",
        "| Variant | mIoU (%) | Delta vs C0 (pp) | mDice (%) |",
        "|---|---:|---:|---:|",
        metric_row("C0", c0, c0),
    ]
    for variant in VARIANTS:
        if variant in candidates:
            lines.append(metric_row(variant, candidates[variant], c0))
        elif variant == "A3":
            lines.append("| A3 | not run (unlock rule not met) | — | — |")

    lines += ["", "## 6. CAM hierarchy", ""]
    header = "| Variant | CAM56 | CAM28_1 | CAM28_2 | CAMdeep | Final |"
    lines += [header, "|---|---:|---:|---:|---:|---:|"]
    for name, record in [("C0", c0), *candidates.items()]:
        values = [percentage(record["scores"][stage]["mIoU"]) for stage in STAGES]
        lines.append(
            f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )

    lines += ["", "## 7. Boundary/interior", ""]
    lines += [
        "| Variant | Boundary accuracy (%) | Boundary restricted mIoU (%) | Interior accuracy (%) | Interior restricted mIoU (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, record in [("C0", c0), *candidates.items()]:
        zones = record["structural"]["zones"]
        boundary = zones["boundary_le_7"]
        interior = zones["interior_ge_8"]
        lines.append(
            f"| {name} | {percentage(boundary['accuracy']):.4f} | "
            f"{percentage(boundary['restricted_mIoU']):.4f} | "
            f"{percentage(interior['accuracy']):.4f} | "
            f"{percentage(interior['restricted_mIoU']):.4f} |"
        )

    lines += [
        "",
        "## 8. Object size",
        "",
        "The frozen historical object-size statistic is pixel-weighted component recall. It is reported under that accurate name; size-restricted mIoU is an additional diagnostic and is not substituted into the preregistered Gate C.",
        "",
        "| Variant | Small recall | Small diagnostic mIoU | Medium recall | Medium diagnostic mIoU | Large recall | Large diagnostic mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in [("C0", c0), *candidates.items()]:
        aggregate = record["structural"]["components"]["aggregate"]
        values = []
        for size in ("small", "medium", "large"):
            values.extend(
                [
                    percentage(aggregate[size]["historical_component_recall"]),
                    percentage(aggregate[size]["diagnostic_size_restricted_mIoU"]),
                ]
            )
        lines.append(
            f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )

    lines += ["", "## 9. Filter drift", ""]
    lines += [
        "| Variant | dec_lo | dec_hi | low drift L2 | high drift L2 | low cosine | high cosine |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for variant, record in candidates.items():
        filters = record["mechanism"]["filters"]
        dec_lo = filters["dec_lo"]
        dec_hi = filters["dec_hi"]
        lines.append(
            f"| {variant} | `{dec_lo['values']}` | `{dec_hi['values']}` | "
            f"{dec_lo['l2_drift']:.8f} | {dec_hi['l2_drift']:.8f} | "
            f"{dec_lo['cosine_to_haar']:.8f} | {dec_hi['cosine_to_haar']:.8f} |"
        )

    lines += ["", "## 10. Gate statistics", ""]
    lines += [
        "| Variant | mean | std | p05 | p25 | p50 | p75 | p95 | min | max | spatial std | channel std | boundary mean | interior mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, record in candidates.items():
        mechanism = record["mechanism"]
        gate = mechanism["gate"]
        vals = [gate[key] for key in ("mean", "std", "p05", "p25", "p50", "p75", "p95", "min", "max")]
        vals += [
            mechanism["spatial_std"]["mean"],
            mechanism["channel_std"]["mean"],
            mechanism["boundary_gate_mean"]["mean"],
            mechanism["interior_gate_mean"]["mean"],
        ]
        lines.append(
            f"| {variant} | " + " | ".join(f"{value:.6f}" for value in vals) + " |"
        )

    lines += ["", "## 11. Context residual statistics", ""]
    lines += [
        "| Variant | Raw RMS | Gated RMS | Gated/raw | Boundary ratio | Interior ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, record in candidates.items():
        mechanism = record["mechanism"]
        values = [
            mechanism["raw_context_rms"]["mean"],
            mechanism["gated_context_rms"]["mean"],
            mechanism["context_rms_ratio"]["mean"],
            mechanism["boundary_context_ratio"]["mean"],
            mechanism["interior_context_ratio"]["mean"],
        ]
        lines.append(
            f"| {variant} | " + " | ".join(f"{value:.6f}" for value in values) + " |"
        )

    lines += ["", "## 12. Per-class IoU", ""]
    lines += [
        "| Variant | Class 0 | Class 1 | Class 2 | Class 3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, record in [("C0", c0), *candidates.items()]:
        class_iou = record["scores"]["final"]["class_iou"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{percentage(class_iou[str(index)]):.4f}" for index in range(4))
            + " |"
        )

    lines += ["", "## 13. Paired image bootstrap", ""]
    lines += [
        "10,000 paired image-level resamples (seed 42); each resample sums image confusion matrices and recomputes the official global mIoU.",
        "",
        "| Variant | Observed delta (pp) | Bootstrap mean delta (pp) | 95% CI (pp) |",
        "|---|---:|---:|---:|",
    ]
    for variant, gate in result["gates"].items():
        overall = gate["A_overall_utility"]
        bootstrap = overall["bootstrap"]
        lines.append(
            f"| {variant} | {overall['delta_mIoU_pp']:+.4f} | "
            f"{bootstrap['mean_delta_pp']:+.4f} | "
            f"[{bootstrap['ci95_low_pp']:+.4f}, {bootstrap['ci95_high_pp']:+.4f}] |"
        )

    lines += ["", "## 14. Failure analysis", ""]
    for variant, gates in result["gates"].items():
        failed = [name for name, row in gates.items() if isinstance(row, dict) and "pass" in row and not row["pass"]]
        lines.append(
            f"- {variant}: " + ("all preregistered gates passed." if not failed else "failed " + ", ".join(failed) + ".")
        )
    if "A3" not in candidates:
        lines.append("- A3 was not run because neither A1 nor A2 met the frozen raw-positive-signal unlock rule.")

    lines += ["", "## 15. GO/NO-GO", ""]
    for variant, gates in result["gates"].items():
        lines.append(
            f"- {variant}: A={gates['A_overall_utility']['pass']}, "
            f"B={gates['B_semantic_safety']['pass']}, "
            f"C={gates['C_structural_mechanism']['pass']}, "
            f"D={gates['D_non_degenerate_gate']['pass']}; "
            f"overall={'GO' if gates['all_pass'] else 'NO-GO'}."
        )
    lines += [
        "",
        "Phase-2 is authorized only when at least one executed variant passes all four preregistered gates. No model or threshold was selected using test data.",
        "",
        f"DECISION = {result['final_decision']}",
    ]
    report = "\n".join(lines) + "\n"
    path = Path(args.output_dir) / "lw_shr_phase1_lwt_transfer_utility_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("decision", "final"), default="final")
    parser.add_argument("--c0-dir", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result, c0, candidates = analyze(args)
    if args.mode == "decision":
        if not all(variant in candidates for variant in ("A1", "A2")):
            raise AssertionError("A3 decision requires A1 and A2")
        print(
            "LW_SHR_A3_UNLOCK=" + ("YES" if result["a3_unlocked"] else "NO"),
            flush=True,
        )
    else:
        path = render_report(args, result, c0, candidates)
        print(f"LW_SHR_REPORT={path}", flush=True)
        print(f"DECISION = {result['final_decision']}", flush=True)


if __name__ == "__main__":
    main()
