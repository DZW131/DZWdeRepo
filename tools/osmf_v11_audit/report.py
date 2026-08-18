"""Deterministic tables, figures, and reports for OSMF-v1.1 audits."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, series, title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for label, rows, key in series:
        points = [
            (int(row["step"]), float(row[key]))
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.5,
                label=label,
            )
    axis.set_title(title)
    axis.set_xlabel("Real BCSS training batches")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def make_figures(
    output_dir: Path,
    loss_rows: list[dict],
    ratio_rows: list[dict],
    cosine_rows: list[dict],
    representation_rows: list[dict],
    update_rows: list[dict],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot(
        figures / "loss_curves.png",
        [
            ("L_SSHR", loss_rows, "loss_sshr"),
            ("L_sem_pres", loss_rows, "loss_sem_pres"),
            ("L_eq", loss_rows, "loss_eq"),
            ("L_orth", loss_rows, "loss_orth"),
            ("L_rec", loss_rows, "loss_rec"),
        ],
        "OSMF-v1.1 objective traces",
        "Loss",
    )
    objectives = ("sem_pres", "eq", "orth", "rec")
    _plot(
        figures / "gradient_ratio_curves.png",
        [
            (name, [row for row in ratio_rows if row["objective"] == name], "ratio")
            for name in objectives
        ],
        "Weighted auxiliary / SSHR gradient ratio at H28_1",
        "Weighted gradient ratio",
    )
    _plot(
        figures / "gradient_cosine_curves.png",
        [
            (name, [row for row in cosine_rows if row["objective"] == name], "cosine")
            for name in objectives
        ],
        "Auxiliary versus SSHR gradient direction",
        "Cosine similarity",
    )
    _plot(
        figures / "branch_rms.png",
        [
            ("H", representation_rows, "h_rms"),
            ("S", representation_rows, "semantic_rms"),
            ("M", representation_rows, "morphology_rms"),
            ("H_hat", representation_rows, "reconstruction_rms"),
        ],
        "Representation RMS",
        "RMS",
    )
    _plot(
        figures / "semantic_response.png",
        [
            ("Z_S", representation_rows, "semantic_student_response_rms"),
            ("Z_H", representation_rows, "semantic_teacher_response_rms"),
            ("R_Z", representation_rows, "semantic_response_rms_ratio"),
        ],
        "Semantic response preservation",
        "RMS / ratio",
    )
    _plot(
        figures / "semantic_agreement.png",
        [("SemAgree", representation_rows, "semantic_agreement")],
        "Pretrained semantic geometry agreement",
        "Class-channel cosine",
    )
    _plot(
        figures / "reconstruction_cosine.png",
        [("Cos(H,H_hat)", representation_rows, "reconstruction_cosine")],
        "Reconstruction stability",
        "Cosine",
    )
    _plot(
        figures / "cross_covariance.png",
        [("CrossCov", representation_rows, "cross_covariance")],
        "Cross-subspace covariance",
        "Mean squared standardized covariance",
    )
    _plot(
        figures / "equivariance_error.png",
        [
            ("EqErr(M)", representation_rows, "eq_error_morphology"),
            ("EqErr(S)", representation_rows, "eq_error_semantic"),
        ],
        "Inverse-aligned equivariance error",
        "Error",
    )
    _plot(
        figures / "parameter_update.png",
        [
            (
                name,
                [row for row in update_rows if row["parameter"] == name],
                "cumulative_update_norm",
            )
            for name in sorted({row["parameter"] for row in update_rows})
        ],
        "Cumulative OSMF-v1.1 parameter movement",
        "Absolute update norm",
    )


def _metric_stats(rows: list[dict], key: str) -> dict:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    if not values:
        return {name: float("nan") for name in ("start", "mean", "end", "min", "max")}
    return {
        "start": values[0],
        "mean": sum(values) / len(values),
        "end": values[-1],
        "min": min(values),
        "max": max(values),
    }


def build_main_table(
    loss_rows: list[dict],
    ratio_rows: list[dict],
    cosine_rows: list[dict],
    representation_rows: list[dict],
) -> list[dict]:
    definitions = [
        ("L_SSHR", loss_rows, "loss_sshr"),
        ("L_sem_pres", loss_rows, "loss_sem_pres"),
        ("L_eq", loss_rows, "loss_eq"),
        ("L_orth", loss_rows, "loss_orth"),
        ("L_rec", loss_rows, "loss_rec"),
    ]
    for objective in ("sem_pres", "eq", "orth", "rec"):
        definitions.append(
            (
                f"r_{objective}",
                [row for row in ratio_rows if row["objective"] == objective],
                "ratio",
            )
        )
        definitions.append(
            (
                f"cos(base,{objective})",
                [row for row in cosine_rows if row["objective"] == objective],
                "cosine",
            )
        )
    definitions.extend(
        [
            ("RMS(H)", representation_rows, "h_rms"),
            ("RMS(S)", representation_rows, "semantic_rms"),
            ("RMS(M)", representation_rows, "morphology_rms"),
            ("RMS(H_hat)", representation_rows, "reconstruction_rms"),
            ("RMS(S)/RMS(M)", representation_rows, "semantic_morphology_rms_ratio"),
            ("Cos(H,H_hat)", representation_rows, "reconstruction_cosine"),
            ("ResidualRatio", representation_rows, "residual_ratio"),
            ("CrossCov(S,M)", representation_rows, "cross_covariance"),
            ("EqErr(M)", representation_rows, "eq_error_morphology"),
            ("EqErr(S)", representation_rows, "eq_error_semantic"),
            ("RMS(Z_S)", representation_rows, "semantic_student_response_rms"),
            ("RMS(Z_H)", representation_rows, "semantic_teacher_response_rms"),
            ("R_Z", representation_rows, "semantic_response_rms_ratio"),
            ("SemAgree", representation_rows, "semantic_agreement"),
        ]
    )
    return [
        {"metric": metric, **_metric_stats(rows, key)}
        for metric, rows, key in definitions
    ]


def write_report(
    output_dir: Path,
    summary: dict,
    main_table: list[dict],
    parameter_summary: Mapping[str, Mapping],
) -> Path:
    report_name = (
        "osmf_v11_semantic_readiness_report.md"
        if summary["gate"] == "readiness"
        else "osmf_v11_phase0_report.md"
    )
    report = output_dir / "docs" / report_name
    lines = [
        f"# OSMF-v1.1 {summary['gate'].title()} Audit",
        "",
        "## 1. Decision",
        "",
        f"**{summary['decision']}**",
        "",
        f"Processed real BCSS batches: {summary['processed_batches']}/{summary['authorized_batches']}.",
        f"Decision reasons: `{summary['decision_reasons']}`.",
        f"Flags: `{summary['flags']}`.",
        "",
        "## 2. Frozen contract",
        "",
        f"- Audit commit: `{summary['audit_commit']}`",
        f"- A0 checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- Parity proof SHA256: `{summary['parity_summary_sha256']}`",
        "- BCSS train only; seed 20260817; batch 20; 224x224; BF16.",
        "- Fixed objective weights: 0.20/0.20/0.05/0.10.",
        "- Fresh A0 checkpoint and optimizer state; no continuation from another audit.",
        f"- Exact command: `{summary['exact_command']}`",
        "",
        "## 3. Main mechanism table",
        "",
        "| Metric | Start | Mean | End | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| {metric} | {start:.6g} | {mean:.6g} | {end:.6g} | {min:.6g} | {max:.6g} |".format(
            **row
        )
        for row in main_table
    )
    lines.extend(
        [
            "",
            "## 4. Parameter health",
            "",
            "| Parameter | Grad nonzero | Mean grad norm | Absolute update | Relative update |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, health in parameter_summary.items():
        lines.append(
            f"| `{name}` | {health['grad_nonzero']} | {health['mean_grad_norm']:.6g} | "
            f"{health['end_update_norm']:.6g} | {health['end_relative_update']:.6g} |"
        )
    sem = summary["semantic"]
    mechanism = summary["mechanism"]
    lines.extend(
        [
            "",
            "## 5. Semantic preservation answers",
            "",
            f"1. v1.0 r_sem range was 2.480527–4.106567; v1.1 r_sem_pres range is {sem['ratio_min']:.6g}–{sem['ratio_max']:.6g}, mean {sem['ratio_mean']:.6g}.",
            f"2. p_sem/u_sem active: {sem['semantic_parameters_active']}.",
            f"3. Semantic response non-degenerate: {sem['response_non_degenerate']}; end R_Z={sem['response_ratio_end']:.6g}, SemAgree={sem['agreement_end']:.6g}.",
            f"4. End reconstruction cosine: {mechanism['reconstruction_cosine_end']:.8f}.",
            f"5. Next gate authorized: {summary['next_gate_authorized']}.",
            "",
            "## 6. Morphology and reconstruction",
            "",
            f"- Morphology objective gradient active: {mechanism['morphology_eq_gradient_active']}.",
            f"- EqErr(M) start/end: {mechanism['eq_error_morphology_start']:.6g} / {mechanism['eq_error_morphology_end']:.6g}.",
            f"- S/M RMS ratio end: {mechanism['semantic_morphology_rms_ratio_end']:.6g}.",
            f"- CrossCov start/end: {mechanism['cross_covariance_start']:.6g} / {mechanism['cross_covariance_end']:.6g}.",
            "",
            "## 7. Safety and boundary",
            "",
            f"- All finite: {summary['finite']}.",
            f"- Auxiliary semantic ic1 gradient-free: {summary['ic1_aux_gradient_free']}.",
            f"- Original SSHR path updated ic1: {summary['ic1_base_gradient_active']}.",
            "- Validation performance evaluated: false.",
            "- Test/LUAD/segmentation GT accessed: false.",
            "- Checkpoint saved for continuation: false.",
            "- Phase 1 started: false.",
            "",
            "The run stops at this gate. Even a Phase-0 GO requires separate human authorization for a 3-epoch pilot.",
            "",
            summary["decision"],
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report

