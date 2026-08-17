"""Tables, deterministic figures, and report for OSMF Phase 0."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows: Iterable[Mapping], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            value = float(value)
            if math.isfinite(value):
                values.append(value)
    return values


def stats(rows: Iterable[Mapping], key: str) -> dict:
    values = finite_values(rows, key)
    if not values:
        return {name: float("nan") for name in ("start", "mean", "end", "min", "max")}
    return {
        "start": values[0],
        "mean": sum(values) / len(values),
        "end": values[-1],
        "min": min(values),
        "max": max(values),
    }


def _line_plot(
    path: Path,
    series: Sequence[tuple[str, Sequence[Mapping], str]],
    title: str,
    ylabel: str,
) -> None:
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
    _line_plot(
        figures / "loss_curves.png",
        [
            ("L_SSHR", loss_rows, "loss_sshr"),
            ("L_sem", loss_rows, "loss_sem"),
            ("L_eq", loss_rows, "loss_eq"),
            ("L_orth", loss_rows, "loss_orth"),
            ("L_rec", loss_rows, "loss_rec"),
        ],
        "OSMF Phase-0 loss traces",
        "Loss",
    )
    ratio_series = []
    cosine_series = []
    for objective in ("sem", "eq", "orth", "rec"):
        ratio_series.append(
            (
                objective,
                [row for row in ratio_rows if row["objective"] == objective],
                "ratio",
            )
        )
        cosine_series.append(
            (
                objective,
                [row for row in cosine_rows if row["objective"] == objective],
                "cosine",
            )
        )
    _line_plot(
        figures / "gradient_ratio_curves.png",
        ratio_series,
        "Weighted auxiliary / SSHR gradient ratio at H28_1",
        "Weighted gradient ratio",
    )
    _line_plot(
        figures / "gradient_cosine_curves.png",
        cosine_series,
        "Auxiliary versus SSHR gradient direction",
        "Cosine similarity",
    )
    _line_plot(
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
    _line_plot(
        figures / "reconstruction_cosine.png",
        [("Cos(H,H_hat)", representation_rows, "reconstruction_cosine")],
        "Reconstruction stability",
        "Cosine",
    )
    _line_plot(
        figures / "cross_covariance.png",
        [("CrossCov", representation_rows, "cross_covariance")],
        "Cross-subspace covariance",
        "Mean squared standardized covariance",
    )
    _line_plot(
        figures / "equivariance_error.png",
        [
            ("EqErr(M)", representation_rows, "eq_error_morphology"),
            ("EqErr(S), diagnostic", representation_rows, "eq_error_semantic"),
        ],
        "Inverse-aligned equivariance error",
        "Error",
    )
    parameter_series = []
    for name in sorted({row["parameter"] for row in update_rows}):
        parameter_series.append(
            (
                name,
                [row for row in update_rows if row["parameter"] == name],
                "cumulative_update_norm",
            )
        )
    _line_plot(
        figures / "parameter_update.png",
        parameter_series,
        "Cumulative OSMF parameter movement",
        "Absolute update norm",
    )


def _status(metric: str, values: dict) -> str:
    if not all(math.isfinite(float(value)) for value in values.values()):
        return "NONFINITE"
    if metric.startswith("r_"):
        if values["max"] > 0.50:
            return "HARD-STOP RANGE OBSERVED"
        if values["max"] > 0.30:
            return "REVIEW RANGE OBSERVED"
        if values["mean"] < 0.02:
            return "BELOW REFERENCE RANGE"
        return "HEALTHY REFERENCE RANGE"
    if metric == "Cos(H,H_hat)":
        if values["end"] < 0.90:
            return "NOGO"
        if values["end"] < 0.95:
            return "REVIEW"
        return "GO RANGE"
    if metric == "RMS(S)/RMS(M)":
        if values["min"] <= 0.05 or values["max"] >= 20:
            return "COLLAPSE RANGE OBSERVED"
        if values["min"] < 0.10 or values["max"] > 10:
            return "IMBALANCE RANGE OBSERVED"
        return "HEALTHY"
    return "FINITE"


def build_main_table(
    loss_rows: list[dict],
    ratio_rows: list[dict],
    cosine_rows: list[dict],
    representation_rows: list[dict],
) -> list[dict]:
    definitions = [
        ("L_SSHR", loss_rows, "loss_sshr"),
        ("L_sem", loss_rows, "loss_sem"),
        ("L_eq", loss_rows, "loss_eq"),
        ("L_orth", loss_rows, "loss_orth"),
        ("L_rec", loss_rows, "loss_rec"),
    ]
    for objective in ("sem", "eq", "orth", "rec"):
        definitions.append(
            (
                f"r_{objective}",
                [row for row in ratio_rows if row["objective"] == objective],
                "ratio",
            )
        )
    for objective in ("sem", "eq", "orth", "rec"):
        definitions.append(
            (
                f"cos(base,{objective})",
                [row for row in cosine_rows if row["objective"] == objective],
                "cosine",
            )
        )
    definitions.extend(
        [
            ("RMS(S)", representation_rows, "semantic_rms"),
            ("RMS(M)", representation_rows, "morphology_rms"),
            ("RMS(S)/RMS(M)", representation_rows, "semantic_morphology_rms_ratio"),
            ("Cos(H,H_hat)", representation_rows, "reconstruction_cosine"),
            ("Residual Ratio", representation_rows, "residual_ratio"),
            ("CrossCov(S,M)", representation_rows, "cross_covariance"),
            ("EqErr(M)", representation_rows, "eq_error_morphology"),
            ("EqErr(S)", representation_rows, "eq_error_semantic"),
        ]
    )
    table = []
    for metric, rows, key in definitions:
        values = stats(rows, key)
        table.append({"metric": metric, **values, "status": _status(metric, values)})
    return table


def write_report(
    output_dir: Path,
    summary: dict,
    main_table: list[dict],
    parameter_summary: Mapping[str, Mapping],
) -> Path:
    docs = output_dir / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    report = docs / "osmf_phase0_128batch_audit.md"
    lines = [
        "# OSMF-v1.0 Phase 0 — 128-Batch Structural & Gradient Audit",
        "",
        "## 1. Executive decision",
        "",
        f"Final decision: **{summary['decision']}**.",
        "",
        f"Processed real BCSS training batches: {summary['processed_batches']}/128.",
        "This is a mechanism-safety audit, not a segmentation-performance experiment.",
        "No validation mIoU was used and no test/LUAD data were accessed.",
        "",
        "## 2. Frozen contract",
        "",
        f"- Phase-0 parent/OSMF implementation commit: `{summary['phase0_parent_commit']}`.",
        f"- Phase-0 audit commit: `{summary['audit_commit']}`.",
        f"- Frozen A0 commit: `{summary['baseline_commit']}`.",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`.",
        "- Optimizer: freshly initialized released PolyOptimizer/SGD from the A0 checkpoint.",
        f"- Optimizer momentum: {summary['optimizer']['momentum']}.",
        f"- Poly schedule max steps: {summary['optimizer']['max_step']} (25-epoch formal scale).",
        "- Batch size: 20; image size: 224; seed: 20260817; BF16.",
        "- Fixed auxiliary weights: 0.20/0.20/0.05/0.10.",
        "- Equivariance interval: 4; transforms: horizontal/vertical flip only.",
        f"- Exact command: `{summary['exact_command']}`.",
        "",
        "## 3. Start-state safety",
        "",
        f"- max|H_hat-H|: {summary['start_state']['feature_max_abs']:.10g}.",
        f"- Reconstruction cosine: {summary['start_state']['reconstruction_cosine']:.10f}.",
        f"- All tensors/CAMs/losses finite: {summary['start_state']['finite']}.",
        "",
        "## 4. Required main table",
        "",
        "| Metric | Start | Mean | End | Min | Max | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in main_table:
        lines.append(
            "| {metric} | {start:.6g} | {mean:.6g} | {end:.6g} | {min:.6g} | {max:.6g} | {status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 5. Parameter health",
            "",
            "| Parameter | Grad nonzero? | Mean grad norm | End absolute update | End relative update | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name, health in parameter_summary.items():
        status = "ACTIVE" if health["grad_nonzero"] and health["measurable_update"] else "DEAD_PATH_WARNING"
        lines.append(
            f"| `{name}` | {health['grad_nonzero']} | {health['mean_grad_norm']:.6g} | "
            f"{health['end_update_norm']:.6g} | {health['end_relative_update']:.6g} | {status} |"
        )
    lines.extend(
        [
            "",
            "## 6. Gradient safety and direction",
            "",
            f"Flags: `{summary['flags']}`.",
            f"Decision reasons: `{summary['decision_reasons']}`.",
            "Independent objective gradients were measured with `torch.autograd.grad` "
            f"at completed audit steps {summary['gradient_audit_steps_completed']} "
            "and did not populate optimizer gradients. Later preregistered points "
            "were not reached after a hard stop.",
            "",
            "## 7. Representation health and early specialization",
            "",
            f"- End reconstruction cosine: {summary['mechanism']['end_reconstruction_cosine']:.8f}.",
            f"- End S/M RMS ratio: {summary['mechanism']['end_semantic_morphology_rms_ratio']:.6g}.",
            f"- EqErr(M) start/end: {summary['mechanism']['eq_error_morphology_start']:.6g} / {summary['mechanism']['eq_error_morphology_end']:.6g}.",
            f"- CrossCov start/end: {summary['mechanism']['cross_covariance_start']:.6g} / {summary['mechanism']['cross_covariance_end']:.6g}.",
            f"- Equivariance response detected: {summary['mechanism']['eq_responsive']}.",
            f"- Morphology equivariance gradient active: {summary['mechanism']['morphology_eq_gradient_active']}.",
            "",
            "## 8. Compute cost",
            "",
            f"- Mean iteration time: {summary['cost']['mean_iteration_seconds']:.6f} s.",
            f"- Mean non-equivariance iteration: {summary['cost']['mean_non_equivariance_seconds']:.6f} s.",
            f"- Peak training-step GPU memory: {summary['cost']['peak_memory_allocated_bytes'] / 2**30:.3f} GiB.",
            "",
            (
                "Equivariance-step runtime and overhead were not estimable because "
                "the preregistered hard stop occurred before step 4."
                if summary["cost"]["equivariance_samples"] == 0
                else "- Mean equivariance iteration: "
                f"{summary['cost']['mean_equivariance_seconds']:.6f} s; "
                "interval-averaged overhead proxy versus non-equivariance OSMF step: "
                f"{summary['cost']['overhead_percent']:.2f}%."
            ),
            "No additional A0 training batches were run for timing.",
            "",
            "## 9. Protocol safety",
            "",
            "- Training labels: image-level filename labels only.",
            "- Segmentation GT used in training: false.",
            "- Validation evaluated: false.",
            "- Test evaluated: false.",
            "- LUAD evaluated: false.",
            "- 3-epoch or 25-epoch training: false.",
            "- Lambda/LR/architecture/fusion/threshold/TTA changes: false.",
            "",
            "## 10. Phase boundary",
            "",
            "This audit now stops. Even a GO only permits human review before a "
            "separate 3-epoch mechanism pilot; it does not authorize Phase 1 automatically.",
            "",
            summary["decision"],
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
