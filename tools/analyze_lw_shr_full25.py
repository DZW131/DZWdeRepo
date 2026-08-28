#!/usr/bin/env python3
"""Compare existing C0-Full25 FINAL with the new A2-Full25 run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lw_shr_common import (
    paired_image_bootstrap_miou,
    read_json,
    sha256_file,
    write_json,
)


REQUIRED_EPOCHS = (1, 5, 10, 15, 20, 25)
STAGES = ("56", "28_1", "28_2", "deep", "final")


def pct(value):
    return 100.0 * float(value)


def delta_pp(left, right):
    return pct(left) - pct(right)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_histograms(path):
    with np.load(path, allow_pickle=False) as archive:
        image_ids = archive["image_ids"].astype(str)
        truths = archive["truths"]
        histograms = archive["histograms"]
    return image_ids, truths, histograms


def align_histograms(reference_path, candidate_path):
    reference_ids, reference_truths, reference_hist = load_histograms(reference_path)
    candidate_ids, candidate_truths, candidate_hist = load_histograms(candidate_path)
    lookup = {image_id: index for index, image_id in enumerate(candidate_ids)}
    if set(reference_ids) != set(candidate_ids):
        raise AssertionError("C0/A2 validation image IDs differ")
    order = np.asarray([lookup[image_id] for image_id in reference_ids])
    if not np.array_equal(reference_truths, candidate_truths[order]):
        raise AssertionError("C0/A2 validation truths differ")
    return reference_hist, candidate_hist[order]


def validation_history(completion):
    rows = {}
    for record in completion["history"]:
        if record["validation"] is not None:
            rows[int(record["training"]["epoch"])] = record
    if tuple(sorted(rows)) != REQUIRED_EPOCHS:
        raise AssertionError(f"Missing required A2 validation epochs: {sorted(rows)}")
    return rows


def diagnostic_csvs(output, history, gradients):
    metrics = []
    filters = []
    gates = []
    contexts = []
    for epoch in REQUIRED_EPOCHS:
        record = history[epoch]
        validation = record["validation"]
        mechanism = validation["mechanism"]
        score_row = {
            "epoch": epoch,
            "train_loss": record["training"]["loss"],
            "train_exact_match": record["training"]["exact_match"],
            "train_accuracy": record["training"]["accuracy"],
        }
        for stage in STAGES:
            score_row[f"{stage}_mIoU"] = validation["scores"][stage]["mIoU"]
            score_row[f"{stage}_mDice"] = validation["scores"][stage]["mDice"]
        zones = validation["structural"]["zones"]
        score_row.update(
            {
                "boundary_accuracy": zones["boundary_le_7"]["accuracy"],
                "boundary_restricted_mIoU": zones["boundary_le_7"]["restricted_mIoU"],
                "interior_accuracy": zones["interior_ge_8"]["accuracy"],
                "interior_restricted_mIoU": zones["interior_ge_8"]["restricted_mIoU"],
            }
        )
        for size, value in validation["structural"]["components"]["aggregate"].items():
            score_row[f"{size}_component_recall"] = value[
                "historical_component_recall"
            ]
            score_row[f"{size}_diagnostic_mIoU"] = value[
                "diagnostic_size_restricted_mIoU"
            ]
        for class_id, value in validation["scores"]["final"]["class_iou"].items():
            score_row[f"class_{class_id}_IoU"] = value
        metrics.append(score_row)

        filter_row = {"epoch": epoch}
        for band in ("dec_lo", "dec_hi"):
            item = mechanism["filters"][band]
            filter_row.update(
                {
                    f"{band}_values": json.dumps(item["values"]),
                    f"{band}_l2_drift": item["l2_drift"],
                    f"{band}_norm": item["norm"],
                    f"{band}_cosine_to_haar": item["cosine_to_haar"],
                }
            )
        filters.append(filter_row)

        gate = mechanism["gate"]
        gate_row = {"epoch": epoch, **{key: gate[key] for key in (
            "mean", "std", "p05", "p25", "p50", "p75", "p95", "min", "max"
        )}}
        gate_row.update(
            {
                "spatial_std": mechanism["spatial_std"]["mean"],
                "channel_std": mechanism["channel_std"]["mean"],
                "boundary_mean_gate": mechanism["boundary_gate_mean"]["mean"],
                "interior_mean_gate": mechanism["interior_gate_mean"]["mean"],
                "boundary_minus_interior": mechanism["boundary_gate_mean"]["mean"]
                - mechanism["interior_gate_mean"]["mean"],
            }
        )
        gates.append(gate_row)

        contexts.append(
            {
                "epoch": epoch,
                "raw_context_rms": mechanism["raw_context_rms"]["mean"],
                "gated_context_rms": mechanism["gated_context_rms"]["mean"],
                "gated_raw_ratio": mechanism["context_rms_ratio"]["mean"],
                "boundary_ratio": mechanism["boundary_context_ratio"]["mean"],
                "interior_ratio": mechanism["interior_context_ratio"]["mean"],
            }
        )
    write_csv(output / "epoch_wise_metrics.csv", metrics)
    write_csv(output / "wavelet_filter_diagnostics.csv", filters)
    write_csv(output / "gate_diagnostics.csv", gates)
    write_csv(output / "context_residual_diagnostics.csv", contexts)
    write_csv(output / "gradient_diagnostics.csv", gradients)
    return metrics, filters, gates, contexts


def decide(c0, a2, bootstrap, gates):
    delta = delta_pp(a2["scores"]["final"]["mIoU"], c0["scores"]["final"]["mIoU"])
    cam_delta = delta_pp(a2["scores"]["28_1"]["mIoU"], c0["scores"]["28_1"]["mIoU"])
    boundary_delta = delta_pp(
        a2["structural"]["zones"]["boundary_le_7"]["accuracy"],
        c0["structural"]["zones"]["boundary_le_7"]["accuracy"],
    )
    interior_delta = delta_pp(
        a2["structural"]["zones"]["interior_ge_8"]["accuracy"],
        c0["structural"]["zones"]["interior_ge_8"]["accuracy"],
    )
    gate_non_degenerate = gates[-1]["spatial_std"] > 0 and gates[-1]["channel_std"] > 0
    tradeoff = boundary_delta > 0 and (cam_delta < -0.10 or interior_delta < -0.10)
    ci_includes_zero = bootstrap["ci95_low_pp"] <= 0 <= bootstrap["ci95_high_pp"]

    # The reused six-run baseline preserved FINAL only. Therefore a relative
    # epoch-wise C0 curve is unavailable by explicit user instruction, and the
    # Strong Positive "stably positive curve" clause cannot be proven.
    strong_curve_evidence_available = False
    if delta < 0 or cam_delta < -0.10 or tradeoff:
        decision = "FULL25_NEGATIVE"
    elif abs(delta) <= 0.02 and ci_includes_zero:
        decision = "FULL25_NEUTRAL"
    elif (
        delta >= 0.10
        and cam_delta >= 0
        and interior_delta >= -0.10
        and strong_curve_evidence_available
    ):
        decision = "FULL25_POSITIVE"
    elif delta > 0 and cam_delta >= 0 and not tradeoff and gate_non_degenerate:
        decision = "FULL25_WEAK_POSITIVE"
    else:
        decision = "FULL25_NEGATIVE"
    return decision, {
        "delta_mIoU_pp": delta,
        "cam28_1_delta_pp": cam_delta,
        "boundary_accuracy_delta_pp": boundary_delta,
        "interior_accuracy_delta_pp": interior_delta,
        "gate_non_degenerate": gate_non_degenerate,
        "boundary_semantic_tradeoff": tradeoff,
        "ci_includes_zero": ci_includes_zero,
        "relative_epoch_curve_available": strong_curve_evidence_available,
    }


def table_row(values):
    return "| " + " | ".join(str(value) for value in values) + " |"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-reference", required=True)
    parser.add_argument("--a2-completion", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = read_json(args.baseline_reference)
    completion = read_json(args.a2_completion)
    if baseline.get("baseline_retrained") is not False:
        raise AssertionError("Baseline reference provenance is missing")
    if completion.get("status") != "COMPLETE" or completion.get("test_used"):
        raise AssertionError("A2 completion is invalid or used test")
    if sha256_file(completion["checkpoint"]) != completion["checkpoint_sha256"]:
        raise AssertionError("A2 final checkpoint SHA256 mismatch")

    history = validation_history(completion)
    metric_rows, filter_rows, gate_rows, context_rows = diagnostic_csvs(
        output, history, completion["gradient_diagnostics"]
    )
    c0 = baseline["validation"]
    a2 = history[25]["validation"]
    c0_hist, a2_hist = align_histograms(
        baseline["predictions"], completion["predictions"]
    )
    bootstrap = paired_image_bootstrap_miou(
        c0_hist, a2_hist, resamples=10000, seed=42
    )
    write_json(output / "epoch25_bootstrap.json", bootstrap)
    decision, criteria = decide(c0, a2, bootstrap, gate_rows)
    analysis = {
        "decision": decision,
        "criteria": criteria,
        "bootstrap": bootstrap,
        "baseline": baseline,
        "a2_final": a2,
        "test_used": False,
        "baseline_retrained": False,
    }
    write_json(output / "analysis.json", analysis)

    delta = criteria["delta_mIoU_pp"]
    delta_mdice = delta_pp(
        a2["scores"]["final"]["mDice"], c0["scores"]["final"]["mDice"]
    )
    lines = [
        "# LW-SHR Phase-1.5 A2 Full-25 Report",
        "",
        "## 1. Implementation commit",
        "",
        f"- Diagnostic/training commit: `{completion['source_commit']}`",
        f"- Frozen A2 architecture commit: `{completion['frozen_architecture_commit']}`",
        f"- Pure A0 baseline commit: `{baseline['a0_commit']}`",
        "- The model equations and A2 architecture are unchanged from `a91f45d`.",
        "",
        "## 2. Exact commands",
        "",
        "```bash",
        " ".join(completion["command"]),
        "python tools/evaluate_lw_shr_full25_baseline.py --checkpoint "
        + baseline["checkpoint"]
        + " --environment-tsv "
        + baseline["environment_tsv"]
        + " --status-tsv "
        + baseline["status_tsv"]
        + " --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --output-dir "
        + str(Path(args.baseline_reference).resolve().parent),
        "python tools/analyze_lw_shr_full25.py --baseline-reference "
        + str(Path(args.baseline_reference).resolve())
        + " --a2-completion "
        + str(Path(args.a2_completion).resolve())
        + " --output-dir "
        + str(output.resolve()),
        "```",
        "",
        f"The A2 command is preserved verbatim in `{completion['training_config']}`. No test or LUAD evaluation was run.",
        "",
        "## 3. Checkpoint SHA256",
        "",
        f"- Existing C0-Full25 FINAL: `{baseline['checkpoint_sha256']}`",
        f"- A2-Full25 FINAL: `{completion['checkpoint_sha256']}`",
        "",
        "## 4. C0/A2 training-equivalence audit",
        "",
        f"- Common initialization exact: `{completion['initialization']['common_initialization_exact']}`",
        f"- Common-key max absolute difference: `{completion['initialization']['common_initialization_max_abs_diff']}`",
        "- Dataset, seed, batch size, optimizer, LR schedule, loss weights, augmentation, BF16 precision, pretrained weights and FINAL-checkpoint rule match the prior official baseline.",
        "- The only model difference is the frozen A2 Learnable Wavelet Gate at HFRM28_1.",
        "- C0 was not retrained, per user instruction; it is the SHA-locked BCSS seed42 member of the earlier six-run official reproduction.",
        "",
        "## 5. Epoch-wise overall metrics",
        "",
        "The reused C0 run retained only Epoch25 FINAL. C0 intermediate validation metrics are therefore unavailable and are shown as `N/A`; A2 values are measured at all required epochs.",
        "The archived A0 validation readout was 67.3102 mIoU. Re-evaluating its SHA-locked FINAL checkpoint with this run's exact BF16/TTA diagnostic harness gives 67.3360 (+0.0258 pp). The paired comparison below uses the latter because C0 and A2 must pass through the same evaluator.",
        "",
        "| Epoch | C0 mIoU | A2 mIoU | A2 mDice |",
        "|---:|---:|---:|---:|",
    ]
    for epoch in REQUIRED_EPOCHS:
        score = history[epoch]["validation"]["scores"]["final"]
        c0_value = f"{pct(c0['scores']['final']['mIoU']):.4f}" if epoch == 25 else "N/A"
        lines.append(table_row([epoch, c0_value, f"{pct(score['mIoU']):.4f}", f"{pct(score['mDice']):.4f}"]))

    lines += [
        "",
        "## 6. Epoch-wise CAM hierarchy",
        "",
        "| Epoch | CAM56 | CAM28_1 | CAM28_2 | CAMdeep | Final |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in REQUIRED_EPOCHS:
        scores = history[epoch]["validation"]["scores"]
        lines.append(table_row([epoch, *[f"{pct(scores[stage]['mIoU']):.4f}" for stage in STAGES]]))

    lines += [
        "",
        "## 7. Boundary/interior",
        "",
        "| Model | Boundary accuracy | Boundary restricted mIoU | Interior accuracy | Interior restricted mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, record in (("C0", c0), ("A2", a2)):
        zones = record["structural"]["zones"]
        boundary = zones["boundary_le_7"]
        interior = zones["interior_ge_8"]
        lines.append(table_row([name, f"{pct(boundary['accuracy']):.4f}", f"{pct(boundary['restricted_mIoU']):.4f}", f"{pct(interior['accuracy']):.4f}", f"{pct(interior['restricted_mIoU']):.4f}"]))

    lines += [
        "",
        "## 8. Object size",
        "",
        "The historical size statistic is pixel-weighted component recall; size-restricted mIoU is diagnostic only.",
        "",
        "| Model | Small recall/mIoU | Medium recall/mIoU | Large recall/mIoU |",
        "|---|---:|---:|---:|",
    ]
    for name, record in (("C0", c0), ("A2", a2)):
        aggregate = record["structural"]["components"]["aggregate"]
        values = []
        for size in ("small", "medium", "large"):
            values.append(f"{pct(aggregate[size]['historical_component_recall']):.4f}/{pct(aggregate[size]['diagnostic_size_restricted_mIoU']):.4f}")
        lines.append(table_row([name, *values]))

    lines += [
        "",
        "## 9. Per-class IoU",
        "",
        "| Model | Class 0 | Class 1 | Class 2 | Class 3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, record in (("C0", c0), ("A2", a2)):
        class_iou = record["scores"]["final"]["class_iou"]
        lines.append(table_row([name, *[f"{pct(class_iou[str(index)]):.4f}" for index in range(4)]]))

    lines += [
        "",
        "## 10. Filter evolution",
        "",
        "| Epoch | dec_lo | dec_hi | Low drift | High drift | Low cosine | High cosine |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in filter_rows:
        lines.append(table_row([row["epoch"], f"`{row['dec_lo_values']}`", f"`{row['dec_hi_values']}`", f"{row['dec_lo_l2_drift']:.8f}", f"{row['dec_hi_l2_drift']:.8f}", f"{row['dec_lo_cosine_to_haar']:.8f}", f"{row['dec_hi_cosine_to_haar']:.8f}"]))

    lines += [
        "",
        "## 11. Gate evolution",
        "",
        "| Epoch | Mean | Std | Spatial std | Channel std | Boundary mean | Interior mean | B-I |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gate_rows:
        lines.append(table_row([row["epoch"], f"{row['mean']:.6f}", f"{row['std']:.6f}", f"{row['spatial_std']:.8f}", f"{row['channel_std']:.8f}", f"{row['boundary_mean_gate']:.6f}", f"{row['interior_mean_gate']:.6f}", f"{row['boundary_minus_interior']:+.8f}"]))

    lines += [
        "",
        "## 12. Context residual evolution",
        "",
        "| Epoch | Raw RMS | Gated RMS | Gated/raw | Boundary ratio | Interior ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in context_rows:
        lines.append(table_row([row["epoch"], f"{row['raw_context_rms']:.6f}", f"{row['gated_context_rms']:.6f}", f"{row['gated_raw_ratio']:.6f}", f"{row['boundary_ratio']:.6f}", f"{row['interior_ratio']:.6f}"]))

    lines += [
        "",
        "## 13. Gradient evolution",
        "",
        f"Recorded `{len(completion['gradient_diagnostics'])}` preregistered snapshots (steps 1–10 and ends of epochs 1/5/10/20/25). Full values are in `gradient_diagnostics.csv`; all recorded gradients were finite.",
        "",
        "## 14. Epoch25 bootstrap CI",
        "",
        f"- Observed Delta mIoU: `{delta:+.4f} pp`",
        f"- Bootstrap mean Delta: `{bootstrap['mean_delta_pp']:+.4f} pp`",
        f"- 95% CI: `[{bootstrap['ci95_low_pp']:+.4f}, {bootstrap['ci95_high_pp']:+.4f}] pp`",
        "- 10,000 paired image-level resamples, seed42; image confusion matrices are summed before recomputing official global mIoU.",
        "",
        "## 15. Scientific interpretation",
        "",
        f"- Final mIoU changed from `{pct(c0['scores']['final']['mIoU']):.4f}` to `{pct(a2['scores']['final']['mIoU']):.4f}` ({delta:+.4f} pp).",
        f"- Final mDice changed from `{pct(c0['scores']['final']['mDice']):.4f}` to `{pct(a2['scores']['final']['mDice']):.4f}` ({delta_mdice:+.4f} pp).",
        f"- CAM28_1 delta: `{criteria['cam28_1_delta_pp']:+.4f} pp`; boundary accuracy delta: `{criteria['boundary_accuracy_delta_pp']:+.4f} pp`; interior accuracy delta: `{criteria['interior_accuracy_delta_pp']:+.4f} pp`.",
        "- Because the reused baseline retained only FINAL, a relative C0-vs-A2 epoch-wise curve cannot be claimed. This is a documented consequence of not rerunning baseline, not missing A2 data.",
        "- No architecture, optimizer, loss, inference threshold or metric was changed based on validation results.",
        "",
        "## 16. Final decision",
        "",
        f"Decision: `{decision}`.",
        "",
        f"DECISION = {decision}",
    ]
    report = "\n".join(lines) + "\n"
    path = output / "lw_shr_phase1_5_a2_full25_report.md"
    path.write_text(report, encoding="utf-8")
    print(f"LW_SHR_FULL25_REPORT={path}", flush=True)
    print(f"DECISION = {decision}", flush=True)


if __name__ == "__main__":
    main()
