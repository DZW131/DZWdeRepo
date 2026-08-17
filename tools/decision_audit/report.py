"""Render the final, source-backed Decision Bottleneck Phase-0 report."""

from __future__ import annotations

from pathlib import Path

from tools.decision_audit import BRANCH_NAMES


def _table(headers, rows):
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(output)


def _individual_table(rows):
    return _table(
        ["Prediction", "mIoU", "mDice", "C0", "C1", "C2", "C3"],
        [
            [
                row["prediction"],
                f'{row["mIoU"]:.4f}',
                f'{row["mDice"]:.4f}',
                *[f'{row[f"class{class_id}_iou"]:.4f}' for class_id in range(4)],
            ]
            for row in rows
        ],
    )


def write_report(output_dir: Path, summary: dict, tables: dict) -> Path:
    output_dir = Path(output_dir)
    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "phase0_decision_bottleneck_audit.md"
    individual = tables["individual"]
    static_top10 = tables["static_top10"]
    class_preference = tables["class_preference"]
    unique = tables["unique"]
    recoverability = tables["recoverability"]
    error_overlap = tables["error_overlap"]
    oracle_rows = tables["oracle"]
    probe = summary["class_probe"]
    calibration = tables["calibration"]
    qualitative = tables["qualitative"]
    official = next(row for row in individual if row["prediction"] == "official_fusion")
    best_static = static_top10[0]
    oracle_by_name = {row["method"]: row for row in oracle_rows}
    cam56_unique = next(
        row for row in unique if row["branch"] == "cam56" and row["class_id"] == -1
    )
    cam56_recovery = next(
        row
        for row in recoverability
        if row["branch"] == "cam56" and row["class_id"] == -1
    )
    overall_overlap = [row for row in error_overlap if row["class_id"] == -1]
    off_diagonal = [
        row["jaccard"]
        for row in overall_overlap
        if row["branch_i"] != row["branch_j"]
    ]
    mean_overlap = sum(off_diagonal) / len(off_diagonal)
    overlap_description = (
        "high" if mean_overlap >= 0.8 else "moderate" if mean_overlap >= 0.6 else "low"
    )
    preference_branches = [row["best_branch"] for row in class_preference]
    class_dependent = len(set(preference_branches)) >= 2

    sections = [
        "# SSHR Phase-0 Decision Bottleneck Audit",
        "",
        "## 1. Executive conclusion",
        "",
        summary["executive_conclusion"],
        "",
        f'Final frozen decision: **{summary["decision"]}**.',
        "",
        "No SSHR training, test-set access, threshold tuning, or model change was performed. "
        "Validation GT was used only for diagnosis, oracle ceilings, and held-out-fold probe fitting.",
        "",
        "## 2. Frozen protocol and parity",
        "",
        f'- Base commit: `{summary["base_commit"]}`.',
        f'- Checkpoint SHA256: `{summary["checkpoint_sha256"]}`.',
        f'- BCSS validation images/masks: {summary["num_images"]}/{summary["num_masks"]}.',
        '- TTA: official three-way identity/horizontal/vertical.',
        '- Official fusion: `0 CAM56 + 0.6 CAM28_1 + 0.2 CAM28_2 + 0.2 CAMdeep`.',
        '- Class-presence thresholds: `0.8 / 0.9 / 0.8 / 0.6`.',
        '- Main metric: released `tool.iouutils.scores()`.',
        f'- Released/audit differing prediction pixels: {summary["parity"]["differing_prediction_pixels"]}.',
        f'- mIoU absolute difference: {summary["parity"]["miou_absolute_difference"]:.12g}.',
        f'- mDice absolute difference: {summary["parity"]["mdice_absolute_difference"]:.12g}.',
        "",
        "Exact command:",
        "",
        "```bash",
        summary["exact_command"],
        "```",
        "",
        "## 3. Individual hierarchy quality",
        "",
        _individual_table(individual),
        "",
        "## 4. Global static fusion",
        "",
        f'The best frozen grid point is `{best_static["w56"]:.2f}/'
        f'{best_static["w28_1"]:.2f}/{best_static["w28_2"]:.2f}/'
        f'{best_static["wdeep"]:.2f}` with mIoU {best_static["mIoU"]:.4f}, '
        f'a {best_static["delta_vs_official"]:+.4f} pp change from the same-run official baseline.',
        "",
        _table(
            ["Rank", "w56", "w28_1", "w28_2", "wdeep", "mIoU", "Delta"],
            [
                [
                    rank,
                    f'{row["w56"]:.2f}',
                    f'{row["w28_1"]:.2f}',
                    f'{row["w28_2"]:.2f}',
                    f'{row["wdeep"]:.2f}',
                    f'{row["mIoU"]:.4f}',
                    f'{row["delta_vs_official"]:+.4f}',
                ]
                for rank, row in enumerate(static_top10, start=1)
            ],
        ),
        "",
        "## 5. Class preference and unique evidence",
        "",
        _table(
            ["Class", *BRANCH_NAMES, "Best", "Gap vs CAM28_1"],
            [
                [
                    row["class_id"],
                    *[f'{row[name]:.4f}' for name in BRANCH_NAMES],
                    row["best_branch"],
                    f'{row["preference_gap_vs_cam28_1"]:+.4f}',
                ]
                for row in class_preference
            ],
        ),
        "",
        f'Different classes prefer different hierarchies: **{class_dependent}**. '
        f'CAM56 contributes {cam56_unique["unique_correct"]:,} unique-correct foreground pixels '
        f'({100 * cam56_unique["unique_rate"]:.4f}%). Against official fusion it has '
        f'{cam56_recovery["recoverable"]:,} recoverable and '
        f'{cam56_recovery["harmful"]:,} harmful pixels '
        f'(net {cam56_recovery["net"]:+,}).',
        "",
        "Full pairwise, per-class unique, and recoverability tables are stored under `tables/`.",
        "",
        "## 6. Error-set geometry",
        "",
        f'The mean off-diagonal foreground error-set Jaccard is {mean_overlap:.4f} '
        f'({overlap_description} overlap).',
        "",
        _table(
            ["Branch i", "Branch j", "Jaccard"],
            [
                [row["branch_i"], row["branch_j"], f'{row["jaccard"]:.4f}']
                for row in overall_overlap
                if BRANCH_NAMES.index(row["branch_i"])
                < BRANCH_NAMES.index(row["branch_j"])
            ],
        ),
        "",
        "## 7. Oracle ceilings",
        "",
        _table(
            ["Method", "mIoU", "mDice", "Delta"],
            [
                [
                    row["method"],
                    f'{row["mIoU"]:.4f}',
                    f'{row["mDice"]:.4f}',
                    f'{row["delta_vs_official"]:+.4f}',
                ]
                for row in oracle_rows
            ],
        ),
        "",
        f'Pixel-oracle coverage is {100 * summary["pixel_oracle"]["coverage"]:.4f}% '
        f'with {summary["pixel_oracle"]["unrecoverable_pixels"]:,} unrecoverable foreground pixels.',
        "",
        "## 8. Five-fold class-conditioned linear probe",
        "",
        f'The formal GroupKFold OOF probe uses {probe["trainable_scalars"]} scalars, '
        f'{probe["num_groups"]} source slides, Adam lr={probe["learning_rate"]}, '
        f'{probe["steps"]} fixed steps, and image batch={probe["batch_size"]}.',
        "",
        _table(
            ["Method", "mIoU", "mDice", "Delta mIoU"],
            [
                ["Official", f'{probe["official_mIoU"]:.4f}', f'{probe["official_mDice"]:.4f}', "—"],
                ["5-fold OOF probe", f'{probe["oof_mIoU"]:.4f}', f'{probe["oof_mDice"]:.4f}', f'{probe["delta_mIoU"]:+.4f}'],
            ],
        ),
        "",
        "Every image appears exactly once out-of-fold and no source group crosses a fold boundary.",
        "",
        "## 9. Calibration and evidence comparability",
        "",
        _table(
            ["Branch", "Mean entropy", "Mean max confidence", "Foreground coverage"],
            [
                [
                    branch,
                    f'{next(row for row in calibration if row["branch"] == branch)["mean_entropy"]:.4f}',
                    f'{next(row for row in calibration if row["branch"] == branch)["mean_max_confidence"]:.4f}',
                    f'{100 * next(row for row in calibration if row["branch"] == branch)["foreground_coverage"]:.2f}%',
                ]
                for branch in BRANCH_NAMES
            ],
        ),
        "",
        "No temperature, ECE, Platt, or isotonic calibration was fitted.",
        "",
        "## 10. Automatically selected qualitative evidence",
        "",
        f'{len(qualitative)} validation panels were selected solely by frozen recoverable-error counts '
        "across Types A–D; none were hand-picked. See `figures/qualitative/` and "
        "`tables/qualitative_manifest.csv`.",
        "",
        "## 11. Answers to the ten preregistered questions",
        "",
        f'1. Individual hierarchy mIoUs are listed in Section 3; official is {official["mIoU"]:.4f}%.',
        "2. Per-class best hierarchies are: "
        + ", ".join(
            f'C{row["class_id"]}={row["best_branch"]}' for row in class_preference
        )
        + ".",
        f'3. Different classes have different scale preferences: {class_dependent}.',
        f'4. CAM56 unique evidence: {cam56_unique["unique_correct"]:,} pixels; net vs official {cam56_recovery["net"]:+,}.',
        f'5. Branch error sets have {overlap_description} overlap (mean Jaccard {mean_overlap:.4f}).',
        f'6. Best global static fusion gain: {best_static["delta_vs_official"]:+.4f} pp.',
        f'7. Image oracle gain: {oracle_by_name["image_oracle"]["delta_vs_official"]:+.4f} pp.',
        f'8. Image-class oracle gain: {oracle_by_name["image_class_oracle"]["delta_vs_official"]:+.4f} pp.',
        f'9. Pixel oracle gain: {oracle_by_name["pixel_oracle"]["delta_vs_official"]:+.4f} pp.',
        f'10. Held-out 5-fold probe gain: {probe["delta_mIoU"]:+.4f} pp.',
        "",
        "## 12. Frozen decision",
        "",
        summary["decision_rationale"],
        "",
        "This audit stops here. It does not authorize UCER, a nonlinear router, test evaluation, LUAD, "
        "threshold changes, or any model implementation without human review.",
        "",
        summary["decision"],
    ]
    report_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return report_path

