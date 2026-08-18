"""Versioned OSMF-v1.3-R1 report writer."""

from __future__ import annotations

from tools.osmf_v13_audit.report import write_csv


def write_report(output_dir, summary):
    name = (
        "osmf_v13r1_readiness_report.md"
        if summary["gate"] == "readiness"
        else "osmf_v13r1_phase0s_report.md"
    )
    path = output_dir / "docs" / name
    causal = summary["same_pair_causal"]
    lines = [
        f"# OSMF-v1.3-R1 {summary['gate'].upper()} Audit", "",
        "## Decision", "", f"**{summary['decision']}**", "",
        f"Reasons: `{summary['decision_reasons']}`",
        f"Processed batches: {summary['processed_batches']}/{summary['authorized_batches']}",
        "", "## Corrected graph contract", "",
        "- `grad(L_struct, p_morph) > 0`",
        "- `grad(L_struct, u_morph) = 0` (expected by graph)",
        "- `grad(L_total, p_morph) > 0`",
        "- `grad(L_total, u_morph) > 0` with measurable update",
        f"- Graph expectation satisfied: {summary['morphology_graph_expected']}",
        "", "## Causal evidence", "",
        f"- Improved/harmed/neutral: {causal['num_improved']} / {causal['num_harmed']} / {causal['num_neutral']}",
        f"- Improved fraction: {causal['improved_fraction']:.6f}",
        f"- Mean delta: {causal['mean_delta']:.8g}",
        "", "## Gradient budgets", "",
    ]
    for objective, values in summary["gradient_budget"].items():
        lines.append(f"- {objective}: mean={values['mean']:.6f}, max={values['max']:.6f}, p95={values['p95']:.6f}")
    if summary.get("fixed_probe"):
        fixed = summary["fixed_probe"]
        lines += [
            "", "## Fixed 64-image probe", "",
            f"- AffinityEqErr(M): {fixed['affinity_morphology_start']:.8g} -> {fixed['affinity_morphology_end']:.8g}",
            f"- AffinityEqErr(S): {fixed['affinity_semantic_start']:.8g} -> {fixed['affinity_semantic_end']:.8g}",
            f"- StructImprove(M): {fixed['struct_improve_m']:.6%}",
            f"- StructImprove(S): {fixed['struct_improve_s']:.6%}",
            f"- SpecificityGap: {fixed['specificity_gap']:.6%}",
        ]
    lines += [
        "", "## Boundary", "",
        "No checkpoint, test, LUAD, segmentation-GT, pilot, or full training was run.",
        "", summary["decision"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
