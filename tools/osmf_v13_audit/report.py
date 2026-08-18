"""Minimal deterministic report writer for OSMF-v1.3 gates."""

from __future__ import annotations

import csv
from pathlib import Path


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, summary: dict) -> Path:
    name = (
        "osmf_v13_readiness_report.md"
        if summary["gate"] == "readiness"
        else "osmf_v13_phase0s_report.md"
    )
    path = output_dir / "docs" / name
    causal = summary["same_pair_causal"]
    lines = [
        f"# OSMF-v1.3 {summary['gate'].upper()} Audit",
        "",
        "## Decision",
        "",
        f"**{summary['decision']}**",
        "",
        f"Reasons: `{summary['decision_reasons']}`",
        f"Processed batches: {summary['processed_batches']}/{summary['authorized_batches']}",
        "",
        "## Frozen contract",
        "",
        f"- Audit commit: `{summary['audit_commit']}`",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- Parity proof SHA256: `{summary['parity_summary_sha256']}`",
        f"- Readiness proof SHA256: `{summary.get('readiness_summary_sha256')}`",
        "- BCSS train only; seed 20260817; batch 20; 224x224; BF16.",
        "- Frozen weights sem/struct/orth/rec = 0.05/0.05/0.05/0.10.",
        "- Structural interval = 4; masked SmoothL1 beta = 1.0.",
        f"- Exact command: `{summary['exact_command']}`",
        "",
        "## Same-pair causal evidence",
        "",
        f"- Improved/harmed/neutral: {causal['num_improved']} / {causal['num_harmed']} / {causal['num_neutral']}",
        f"- Improved fraction: {causal['improved_fraction']:.6f}",
        f"- Mean delta: {causal['mean_delta']:.8g}",
        "",
        "## Gradient budgets and safety",
        "",
    ]
    for objective, values in summary["gradient_budget"].items():
        lines.append(
            f"- {objective}: mean={values['mean']:.6f}, max={values['max']:.6f}, p95={values['p95']:.6f}"
        )
    rep = summary["representation"]
    lines.extend([
        f"- SemAgree end: {rep['semantic_agreement_end']:.6f}",
        f"- Reconstruction cosine end: {rep['reconstruction_cosine_end']:.6f}",
        f"- CrossCov start/end: {rep['cross_covariance_start']:.6f} / {rep['cross_covariance_end']:.6f}",
    ])
    if summary.get("fixed_probe"):
        fixed = summary["fixed_probe"]
        lines.extend([
            "",
            "## Fixed 64-image probe",
            "",
            f"- AffinityEqErr(M) start/end: {fixed['affinity_morphology_start']:.8g} / {fixed['affinity_morphology_end']:.8g}",
            f"- AffinityEqErr(S) start/end: {fixed['affinity_semantic_start']:.8g} / {fixed['affinity_semantic_end']:.8g}",
            f"- StructImprove(M): {fixed['struct_improve_m']:.6%}",
            f"- StructImprove(S): {fixed['struct_improve_s']:.6%}",
            f"- SpecificityGap: {fixed['specificity_gap']:.6%}",
        ])
    lines.extend([
        "",
        "## Boundary",
        "",
        "No checkpoint was saved. Validation, test, LUAD, segmentation GT, three-epoch pilot, and full training were not run.",
        "",
        summary["decision"],
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
