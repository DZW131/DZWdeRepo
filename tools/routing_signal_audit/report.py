"""Generate the frozen Phase-0B scientific audit report."""

from __future__ import annotations

from pathlib import Path


def _metric_table(rows: list[dict]) -> str:
    lines = [
        "| Method | mIoU | mDice | Delta mIoU |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mIoU']:.4f} | {row['mDice']:.4f} | {row['delta_mIoU']:+.4f} |"
        )
    return "\n".join(lines)


def _fold_table(rows: list[dict]) -> str:
    lines = [
        "| Fold | Official | Router | Delta | Override rate | Override precision |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fold']} | {row['official_mIoU']:.4f} | {row['router_mIoU']:.4f} | "
            f"{row['delta_mIoU']:+.4f} | {row['override_rate']:.4f} | {row['override_precision']:.4f} |"
        )
    return "\n".join(lines)


def write_report(output_dir: Path, summary: dict, tables: dict) -> Path:
    output_dir = Path(output_dir)
    report_path = output_dir / "docs" / "phase0b_routing_signal_learnability_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    primary = summary["primary_probe"]
    safe = summary["oracles"]["safe_image"]
    slide = summary["oracles"]["slide"]
    fusion = summary["oracles"]["image_fusion"]
    local = summary["oracles"]["exact_local_imageclass"]
    bootstrap = summary["bootstrap"]
    positive_folds = summary["positive_folds"]
    correlation_top = sorted(
        [row for row in tables["correlations"] if row["scope"] == "all"],
        key=lambda row: abs(row["spearman_relative_utility"]),
        reverse=True,
    )[:15]
    correlation_lines = [
        "| Set | Signal | Spearman(relative utility) | Spearman(absolute utility) |",
        "|---|---|---:|---:|",
    ]
    for row in correlation_top:
        correlation_lines.append(
            f"| {row['signal_set']} | {row['signal']} | "
            f"{row['spearman_relative_utility']:+.4f} | {row['spearman_absolute_utility']:+.4f} |"
        )
    phenotype_lines = "\n".join(
        f"- `{flag}`" for flag in summary["phenotype_flags"]
    )
    diagnostic_keys = (
        "override_rate",
        "oracle_override_opportunity",
        "override_precision",
        "harmful_override_rate",
        "mean_positive_override_gain",
        "mean_harmful_override_loss",
        "best_branch_top1_accuracy",
        "best_branch_top2_accuracy",
        "pairwise_ranking_accuracy",
        "relative_utility_mae",
        "predicted_true_spearman",
    )
    diagnostic_lines = "\n".join(
        f"- {key}: {primary[key]:.6f}" for key in diagnostic_keys
    )
    sections = [
        "# SSHR Phase-0B Routing Signal Learnability Audit",
        "## 1. Executive conclusion\n\n"
        + summary["executive_conclusion"]
        + f"\n\nFinal frozen decision: **{summary['decision']}**.",
        "## 2. Frozen contract\n\n"
        f"- Phase-0B parent commit: `{summary['phase0b_parent_commit']}`.\n"
        f"- Baseline commit: `{summary['baseline_commit']}`.\n"
        f"- Phase-0B audit commit: `{summary['phase0b_commit']}`.\n"
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`.\n"
        "- Dataset/split: BCSS validation only (3418 images, 22 source slides).\n"
        "- SSHR training: false. Test evaluation: false. LUAD evaluation: false.\n"
        "- Network, released inference, thresholds, TTA, and metric are unchanged.\n\n"
        "Exact command:\n\n```bash\n"
        + summary["exact_command"]
        + "\n```",
        "## 3. Exact parity\n\n"
        f"- Released-vs-Phase-0B differing prediction pixels: {summary['parity']['released_vs_phase0b_differing_pixels']}.\n"
        f"- Phase-0 released differing pixels: {summary['parity']['phase0_released_differing_pixels']}.\n"
        f"- mIoU absolute difference: {summary['parity']['mIoU_absolute_difference']:.12g}.\n"
        f"- mDice absolute difference: {summary['parity']['mDice_absolute_difference']:.12g}.\n"
        f"- Parity gate: {summary['parity']['pass']}.",
        "## 4. Reproduction of Phase-0 references\n\n"
        f"Official fusion is {summary['official']['mIoU']:.4f} mIoU / {summary['official']['mDice']:.4f} mDice. "
        "The frozen Phase-0 global-fusion, branch-oracle, pixel-oracle, and OOF class-probe references remain unchanged in the parent audit artifacts.",
        "## 5. Safe Image Candidate Oracle\n\n"
        + _metric_table([summary["official"], safe])
        + "\n\nOfficial fusion is included as the first, tie-preferred safety candidate.",
        "## 6. Slide-Level Safe Oracle\n\n"
        + _metric_table([summary["official"], slide])
        + f"\n\nSlide recovery ratio: {slide['slide_recovery_ratio']:.4f}; slide-context flag: {slide['phenotype_flag']}.",
        "## 7. Image-Level Fusion Oracle\n\n"
        + _metric_table([summary["official"], fusion])
        + f"\n\nFrozen grid size: {fusion['grid_candidates']}; soft gain beyond safe hard selection: "
        f"{fusion['soft_gain_beyond_safe_hard']:+.4f} pp; phenotype: `{fusion['mixture_flag']}`.",
        "## 8. Exact Local Image×Class Oracle\n\n"
        + _metric_table([summary["official"], local])
        + f"\n\nThis is an exact per-image local diagnostic ceiling over {local['enumerated_combinations']} combinations, not a dataset-additive bound. "
        f"Mean/median local q: {local['mean_local_q']:.4f}/{local['median_local_q']:.4f}; phenotype: `{local['class_conditional_flag']}`.",
        "## 9. Routing signal definitions\n\n"
        f"Signal A contains {summary['signals']['a_features']} aggregated-CAM/confidence/morphology/disagreement scalars per candidate. "
        f"Signal B adds {summary['signals']['b_increment_features']} aligned three-view TTA reliability scalars. "
        f"Signal C adds 12 frozen feature statistics and four train-fold-only {summary['signals']['pca_dimensions']}-D PCA contexts. "
        "No GT, slide ID, filename, patient ID, error mask, or validation-fitted calibration enters any probe input.",
        "## 10. Signal-target correlations\n\n"
        + "\n".join(correlation_lines)
        + "\n\nThese correlations are diagnostic only; no signal was selected or removed from a probe after inspection.",
        "## 11. Linear-A/B/C\n\n"
        + _metric_table([row for row in tables["probe_summaries"] if row["method"].startswith("Linear")]),
        "## 12. MLP-A/B/C\n\n"
        + _metric_table([row for row in tables["probe_summaries"] if row["method"].startswith("MLP")]),
        "## 13. Formal MLP-C OOF segmentation result\n\n"
        + _metric_table([summary["official"], primary])
        + f"\n\nOracle recovery ratio: {summary['oracle_recovery_ratio']:.4f}. Only this preregistered primary probe determines GO/NOGO.",
        "## 14. Override diagnostics\n\n" + diagnostic_lines,
        "## 15. Fold stability\n\n"
        + _fold_table(tables["primary_folds"])
        + f"\n\nPositive held-out folds: {positive_folds}/5.",
        "## 16. Slide-level paired bootstrap\n\n"
        f"The paired grouped bootstrap uses {bootstrap['replicates']} replicates over {bootstrap['num_source_slides']} source slides "
        f"(seed {bootstrap['seed']}). Mean/median delta: {bootstrap['mean_delta_mIoU']:+.4f}/{bootstrap['median_delta_mIoU']:+.4f} pp; "
        f"95% CI [{bootstrap['ci_2_5']:+.4f}, {bootstrap['ci_97_5']:+.4f}].",
        "## 17. Oracle recovery ratio\n\n"
        f"MLP-C recovers {100 * summary['oracle_recovery_ratio']:.2f}% of the safe-image oracle gap. Negative recovery is retained rather than clipped.",
        "## 18. Routing phenotype flags\n\n" + phenotype_lines,
        "## 19. Qualitative routing cases\n\n"
        f"{len(tables['qualitative'])} cases were selected automatically: eight successful overrides, eight harmful overrides, eight missed opportunities, and eight correct fallbacks. "
        "See `figures/qualitative/` and `tables/qualitative_manifest.csv`.",
        "## 20. Scientific interpretation\n\n"
        + summary["decision_rationale"]
        + " The secondary probes, oracle phenotypes, and correlations are explanatory only and cannot replace MLP-C as the primary decision probe.",
        "## 21. Final frozen decision\n\n"
        + summary["decision_rationale"]
        + "\n\nThis audit now stops. It does not authorize a formal router, SSHR changes, test/LUAD/other-seed runs, feature additions, or hyperparameter/threshold tuning.",
        summary["decision"],
    ]
    report_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return report_path
