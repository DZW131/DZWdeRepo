"""Tables, figures, and report generation for OSMF-v1.2 Phase-0M."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def make_figures(
    output_dir: Path,
    causal_rows: list[dict],
    fixed_rows: list[dict],
    gradient_rows: list[dict],
    representation_rows: list[dict],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.hist([row["delta"] for row in causal_rows], bins=12, edgecolor="black")
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set(title="Same-pair causal delta distribution", xlabel="EqAfter - EqBefore", ylabel="Eq steps")
    _save(figure, figures / "causal_delta_histogram.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot([row["step"] for row in causal_rows], [row["delta"] for row in causal_rows], marker="o")
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set(title="Same-pair causal effect over training", xlabel="Real BCSS batches", ylabel="EqAfter - EqBefore")
    axis.grid(alpha=0.25)
    _save(figure, figures / "causal_delta_over_steps.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    steps = [row["step"] for row in fixed_rows]
    axis.plot(steps, [row["eq_error_morphology"] for row in fixed_rows], marker="o", label="Morphology")
    axis.plot(steps, [row["eq_error_semantic"] for row in fixed_rows], marker="o", label="Semantic control")
    axis.set(title="Fixed-probe raw feature equivariance", xlabel="Real BCSS batches", ylabel="EqErr")
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, figures / "fixed_probe_eq_curve.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(steps, [row["affinity_eq_error_morphology"] for row in fixed_rows], marker="o", label="Morphology")
    axis.plot(steps, [row["affinity_eq_error_semantic"] for row in fixed_rows], marker="o", label="Semantic control")
    axis.set(title="Fixed-probe local-affinity equivariance", xlabel="Real BCSS batches", ylabel="AffinityEqErr")
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, figures / "fixed_probe_affinity_curve.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    gradient_steps = [row["step"] for row in gradient_rows]
    for key, label in (
        ("cos_eq_base", "eq vs SSHR"),
        ("cos_eq_sem", "eq vs semantic"),
        ("cos_eq_orth", "eq vs orth"),
        ("cos_eq_rec", "eq vs reconstruction"),
    ):
        axis.plot(gradient_steps, [row[key] for row in gradient_rows], marker="o", label=label)
    axis.axhline(-0.30, color="orange", linestyle="--", linewidth=1)
    axis.axhline(-0.50, color="red", linestyle="--", linewidth=1)
    axis.set(title="Morphology-parameter gradient competition", xlabel="Real BCSS batches", ylabel="Cosine")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    _save(figure, figures / "morphology_gradient_conflict.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    raw_start = fixed_rows[0]["eq_error_morphology"]
    affinity_start = fixed_rows[0]["affinity_eq_error_morphology"]
    axis.plot(steps, [row["eq_error_morphology"] - raw_start for row in fixed_rows], marker="o", label="Raw feature delta")
    axis.plot(steps, [row["affinity_eq_error_morphology"] - affinity_start for row in fixed_rows], marker="o", label="Local affinity delta")
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set(title="Raw vs local-affinity morphology trend", xlabel="Real BCSS batches", ylabel="Change from fixed-probe start")
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, figures / "raw_vs_affinity_equivariance.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    representation_steps = [row["step"] for row in representation_rows]
    for key, label in (
        ("semantic_agreement", "SemAgree"),
        ("reconstruction_cosine", "Cos(H,H_hat)"),
        ("semantic_morphology_rms_ratio", "RMS(S)/RMS(M)"),
    ):
        axis.plot(representation_steps, [row[key] for row in representation_rows], marker="o", label=label)
    axis.set(title="Representation safety replication", xlabel="Real BCSS batches", ylabel="Value")
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, figures / "representation_health.png")


def write_report(output_dir: Path, summary: dict) -> Path:
    report = output_dir / "docs" / "osmf_v12_phase0m_morphology_objective_causal_audit.md"
    causal = summary["same_pair_causal"]
    fixed = summary["fixed_probe"]
    competition = summary["morphology_gradient_competition"]
    representation = summary["representation"]
    replication = summary["replication"]
    budget = summary["gradient_budget_replication"]
    lines = [
        "# OSMF-v1.2 Phase-0M Morphology Objective Causal Audit",
        "",
        "## 1. Primary decision",
        "",
        f"**{summary['decision']}**",
        "",
        f"Decision reasons: `{summary['decision_reasons']}`.",
        f"Secondary flags: `{summary['flags']}`.",
        "",
        "## 2. Frozen contract",
        "",
        f"- Audit commit: `{summary['audit_commit']}`",
        f"- Frozen v1.2 executed commit: `{summary['frozen_v12_executed_commit']}`",
        f"- A0 checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- v1.2 Phase-0 proof SHA256: `{summary['v12_phase0_summary_sha256']}`",
        "- Fresh A0 restart; BCSS train only; seed 20260817; batch 20; 224x224; BF16.",
        "- Loss weights remain 0.05/0.05/0.05/0.10; architecture, objective, optimizer, schedule, and augmentation are unchanged.",
        f"- Exact command: `{summary['exact_command']}`",
        "",
        "## 3. Same-pair causal before/after",
        "",
        f"- Eq-active steps: {causal['num_eq_steps']}",
        f"- Improved/harmed/neutral: {causal['num_improved']} / {causal['num_harmed']} / {causal['num_neutral']}",
        f"- Improved fraction: {causal['improved_fraction']:.6f}",
        f"- Harmed fraction: {causal['harmed_fraction']:.6f}",
        f"- Mean/median delta: {causal['mean_delta']:.8g} / {causal['median_delta']:.8g}",
        f"- P25/P75 delta: {causal['p25_delta']:.8g} / {causal['p75_delta']:.8g}",
        f"- Min/max delta: {causal['min_delta']:.8g} / {causal['max_delta']:.8g}",
        "",
        "Each before/after comparison reused the exact realized input tensor and flip. The normal full joint v1.2 update was executed once; no eq-only or second optimizer step was used.",
        "",
        "## 4. Fixed 64-image probe",
        "",
        f"- Raw EqErr(M) start/end/delta: {fixed['raw_morphology_start']:.8g} / {fixed['raw_morphology_end']:.8g} / {fixed['raw_morphology_delta']:.8g}",
        f"- Raw EqErr(S) start/end: {fixed['raw_semantic_start']:.8g} / {fixed['raw_semantic_end']:.8g}",
        f"- AffinityEqErr(M) start/end/delta: {fixed['affinity_morphology_start']:.8g} / {fixed['affinity_morphology_end']:.8g} / {fixed['affinity_morphology_delta']:.8g}",
        f"- AffinityEqErr(S) start/end: {fixed['affinity_semantic_start']:.8g} / {fixed['affinity_semantic_end']:.8g}",
        "",
        "The manifest fixes image IDs, dataset flips, pair flip, normalization, selection seed, and exact tensor SHA256. Probe forwards used no gradients and no optimizer update.",
        "",
        "## 5. Morphology-parameter gradient competition",
        "",
        f"- Mean cos(eq, SSHR): {competition['mean_cos_eq_base']:.6f}",
        f"- Mean cos(eq, semantic): {competition['mean_cos_eq_sem']:.6f}",
        f"- Mean cos(eq, orth): {competition['mean_cos_eq_orth']:.6f}",
        f"- Mean cos(eq, reconstruction): {competition['mean_cos_eq_rec']:.6f}",
        "",
        "## 6. Safety replication",
        "",
        f"- Mean r_sem / r_eq: {budget['sem_pres']['mean']:.6f} / {budget['eq']['mean']:.6f}",
        f"- SemAgree start/end: {representation['semantic_agreement_start']:.6f} / {representation['semantic_agreement_end']:.6f}",
        f"- Semantic response RMS ratio end: {representation['semantic_response_rms_ratio_end']:.6f}",
        f"- Reconstruction cosine end: {representation['reconstruction_cosine_end']:.6f}",
        f"- S/M RMS ratio end: {representation['semantic_morphology_rms_ratio_end']:.6f}",
        f"- CrossCov start/end: {representation['cross_covariance_start']:.6f} / {representation['cross_covariance_end']:.6f}",
        f"- Representation healthy: {representation['healthy']}",
        f"- Replication instability: {replication['instability']}",
        "",
        "## 7. Boundary",
        "",
        f"- Processed batches: {summary['processed_batches']}/{summary['authorized_batches']}",
        f"- All finite: {summary['finite']}",
        "- Checkpoint saved: false",
        "- Validation/test/LUAD/segmentation GT used: false",
        "- Three-epoch pilot and 25-epoch training started: false",
        "- v1.3 implemented: false",
        "",
        "This audit stops after the causal decision and waits for human scientific review.",
        "",
        summary["decision"],
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
