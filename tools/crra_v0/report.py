"""Markdown reporting for CRRA-v0."""

from __future__ import annotations

from pathlib import Path


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(summary, output_path):
    reps = summary["representations"]
    decision = summary["decision"]
    coverage = summary["coverage"]
    diagnostics = summary["diagnostics"]
    best = decision["best_candidate"]
    best_label = "Core" if best == "core" else "Core+Rim"
    lines = [
        "# CRRA-v0 Core-aware Region Representation Audit",
        "",
        "## 1. Executive conclusion",
        "",
        f"**{decision['decision']}**",
        "",
        f"Representation flag: **{decision['representation_flag']}**.",
        "",
        f"The highest frozen, slide-held-out representation is {best_label}, with a "
        f"Macro-F1 delta of {decision['best_delta_macro_f1']:+.4f} and a Type-B "
        f"accuracy delta of {decision['best_delta_type_b_accuracy']:+.4f} versus WholeToken.",
        "",
        "This is a representation audit only. It does not establish a WSSS segmentation gain.",
        "",
        "## 2. Frozen protocol and provenance",
        "",
        f"- A0 source commit: `{summary['provenance']['a0_commit']}`",
        f"- Audit commit: `{summary['provenance']['audit_commit']}`",
        f"- A0 checkpoint: `{summary['provenance']['checkpoint']}`",
        f"- Checkpoint SHA256: `{summary['provenance']['checkpoint_sha256']}`",
        f"- Exact command: `{summary['provenance']['command']}`",
        f"- Precision: {summary['provenance']['amp_dtype']}",
        f"- Validation images/slides: {summary['dataset']['images']} / {summary['dataset']['slides']}",
        "- Training: none. Test/LUAD: not accessed.",
        "",
        "## 3. Common-support and exclusion audit",
        "",
        f"- Raw connected components: {coverage['raw_components']}",
        f"- Area-1 components rejected by the frozen minimum-area rule: {coverage['tiny_components']}",
        f"- Total proposed regions (area >= 2): {coverage['total_proposed_regions']}",
        f"- Empty-core regions: {coverage['empty_core_regions']}",
        f"- Empty-rim regions: {coverage['empty_rim_regions']}",
        f"- Excluded regions: {coverage['excluded_regions']}",
        f"- Common-support regions: {coverage['common_support_regions']}",
        f"- Common-support fraction: {coverage['common_support_fraction']:.4%}",
        f"- Coverage review: {coverage['coverage_review']}",
        "",
        "Per-class exclusions are recorded in `diagnostics/exclusion_by_predicted_class.csv` and "
        "`diagnostics/exclusion_by_gt_majority_class.csv`.",
        "",
        "## 4. Required executive table",
        "",
        "| Representation | Dim | OOF Macro-F1 | Delta vs Whole | Type-B Acc | DeltaB vs Whole | Type-A Acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("whole", "Whole"), ("core", "Core"), ("core_rim", "Core+Rim")):
        item = reps[name]
        delta = "—" if name == "whole" else f"{item['delta_macro_f1_vs_whole']:+.4f}"
        delta_b = "—" if name == "whole" else f"{item['delta_type_b_accuracy_vs_whole']:+.4f}"
        lines.append(
            f"| {label} | {item['dim']} | {item['macro_f1']:.4f} | {delta} | "
            f"{item['type_b_accuracy']:.4f} | {delta_b} | {item['type_a_accuracy']:.4f} |"
        )

    lines.extend([
        "",
        "OOF Macro-F1 is the mean over foreground classes C0-C3. The fixed multinomial probe "
        "is trained on every pure common-support region, including pure background-majority "
        "false-positive regions as class 4; accuracy and balanced accuracy therefore cover all observed labels.",
        "",
        "## 5. Required fold table",
        "",
        "| Fold | Whole F1 | Core F1 | Core-Whole | Core+Rim F1 | CR-Whole |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["fold_comparison"]:
        lines.append(
            f"| {row['fold'] + 1} | {row['whole_f1']:.4f} | {row['core_f1']:.4f} | "
            f"{row['core_delta']:+.4f} | {row['core_rim_f1']:.4f} | {row['core_rim_delta']:+.4f} |"
        )
    lines.extend([
        "",
        f"Best-candidate positive folds: {decision['positive_folds']}/5; mean/min/max fold delta: "
        f"{summary['fold_stability'][best]['mean_delta']:+.4f} / "
        f"{summary['fold_stability'][best]['min_delta']:+.4f} / "
        f"{summary['fold_stability'][best]['max_delta']:+.4f}.",
        "",
        "## 6. Required per-class table",
        "",
        "| Class | Whole F1 | Core F1 | DeltaCore | Core+Rim F1 | DeltaCR |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary["per_class_comparison"]:
        lines.append(
            f"| C{row['class']} | {row['whole_f1']:.4f} | {row['core_f1']:.4f} | "
            f"{row['core_delta']:+.4f} | {row['core_rim_f1']:.4f} | {row['core_rim_delta']:+.4f} |"
        )
    lines.extend([
        "",
        "Complete accuracy, balanced accuracy, confusion matrices, OOF predictions, and fold manifests "
        "are stored under `probes/` and `folds/`.",
        "",
        "## 7. Core/rim diagnostics",
        "",
        "The table reports mean dispersion and median Core-Rim discrepancy.",
        "",
        "| Group | Whole Dispersion | Core Dispersion | Rim Dispersion | Core-Rim Discrepancy |",
        "|---|---:|---:|---:|---:|",
    ])
    for group in ("Type-A", "Type-B", "Mixed"):
        item = diagnostics["groups"][group]
        lines.append(
            f"| {group} | {_fmt(item['whole_dispersion_mean'], 6)} | "
            f"{_fmt(item['core_dispersion_mean'], 6)} | {_fmt(item['rim_dispersion_mean'], 6)} | "
            f"{_fmt(item['core_rim_discrepancy_median'], 6)} |"
        )
    rank = diagnostics["type_b_vs_type_a_rank_test"]
    lines.extend([
        "",
        f"Type-B vs Type-A discrepancy Mann-Whitney U: {_fmt(rank.get('u'), 2)}; "
        f"two-sided p={_fmt(rank.get('p_two_sided'), 6)}; "
        f"rank-biserial={_fmt(rank.get('rank_biserial_type_b_vs_type_a'), 6)}.",
        "",
        "## 8. Slide bootstrap uncertainty",
        "",
    ])
    for name, label in (("core_minus_whole", "Core-Whole"), ("core_rim_minus_whole", "Core+Rim-Whole")):
        item = summary["bootstrap"][name]
        lines.append(
            f"- {label}: mean={item['mean']:+.4f}, 95% CI "
            f"[{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}] "
            f"({item['samples']} slide bootstrap samples, seed={item['seed']})."
        )

    whole = reps["whole"]
    core = reps["core"]
    dual = reps["core_rim"]
    c2_core = summary["per_class_comparison"][2]["core_delta"]
    c2_dual = summary["per_class_comparison"][2]["core_rim_delta"]
    core_disp_below = all(
        diagnostics["groups"][group]["core_dispersion_mean"]
        < diagnostics["groups"][group]["whole_dispersion_mean"]
        for group in ("Type-A", "Type-B", "Mixed")
    )
    discrepancy_higher = (
        diagnostics["groups"]["Type-B"]["core_rim_discrepancy_median"]
        > diagnostics["groups"]["Type-A"]["core_rim_discrepancy_median"]
    )
    lines.extend([
        "",
        "## 9. Answers to the preregistered questions",
        "",
        f"1. CoreToken vs WholeToken: delta Macro-F1={core['delta_macro_f1_vs_whole']:+.4f}; "
        f"this {'is' if core['delta_macro_f1_vs_whole'] >= 0.03 else 'is not'} a GO-scale improvement.",
        f"2. Core+Rim vs Core: delta Macro-F1={dual['macro_f1'] - core['macro_f1']:+.4f}; "
        f"Type-B additional accuracy={dual['type_b_accuracy'] - core['type_b_accuracy']:+.4f}.",
        f"3. Highest OOF Macro-F1: {best_label} ({reps[best]['macro_f1']:.4f}).",
        f"4. Fold stability: {decision['positive_folds']}/5 best-candidate folds are positive.",
        f"5. Type-B separability: best gain={decision['best_delta_type_b_accuracy']:+.4f}.",
        f"6. Type-A preservation: best-candidate drop={decision['best_type_a_drop']:+.4f}; "
        f"review threshold exceeded={decision['best_type_a_drop'] > 0.02}.",
        f"7. C2 benefit: Core={c2_core:+.4f}, Core+Rim={c2_dual:+.4f} versus Whole.",
        f"8. Core dispersion is lower than Whole in all three taxonomies: {core_disp_below}.",
        f"9. Type-B median Core-Rim discrepancy exceeds Type-A: {discrepancy_higher}.",
        f"10. Common-support coverage={coverage['common_support_fraction']:.4%}; sufficient={coverage['common_support_fraction'] >= 0.70}.",
        f"11. Representation recommendation: {decision['representation_flag']}.",
        f"12. Region-centric route decision: {decision['decision']}.",
        "",
        "## 10. Decision trace and stop boundary",
        "",
        f"- Best candidate: {best_label}",
        f"- Best delta Macro-F1: {decision['best_delta_macro_f1']:+.4f}",
        f"- Best Type-B accuracy delta: {decision['best_delta_type_b_accuracy']:+.4f}",
        f"- Positive folds: {decision['positive_folds']}/5",
        f"- Non-negative/positive foreground classes: {decision['nonnegative_classes']}/4 / {decision['positive_classes']}/4",
        f"- Type-A drop: {decision['best_type_a_drop']:+.4f}",
        f"- Hard NOGO conditions: `{decision['hard_nogo_conditions']}`",
        f"- REVIEW conditions: `{decision['review_conditions']}`",
        "",
        f"Final decision: **{decision['decision']}**",
        f"Final flag: **{decision['representation_flag']}**",
        "",
        "The audit stops here. CRSR training, segmentation training, test, LUAD, graph, prototype, "
        "attention pooling, and any fourth representation were not run.",
    ])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
