"""Deterministic review figures for the frozen HMA-v0 audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.hma_v0 import STAGES


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_figures(summary, kernel_frame, gradient_frame, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = summary["validation"]

    gamma = summary["gamma"]
    x = np.arange(len(STAGES))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - 0.18, [gamma[s]["gamma_veto"] for s in STAGES], 0.36, label="gamma_veto")
    ax.bar(x + 0.18, [gamma[s]["gamma_context"] for s in STAGES], 0.36, label="gamma_context")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x, STAGES); ax.set_ylabel("trained scalar"); ax.legend()
    _save(fig, output_dir / "gamma_values.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for stage in STAGES:
        values = kernel_frame.loc[kernel_frame.stage == stage, "uniform_cosine"]
        ax.hist(values, bins=35, alpha=0.45, density=True, label=stage)
    ax.set_xlabel("cosine(kernel, uniform K=15)"); ax.set_ylabel("density"); ax.legend()
    _save(fig, output_dir / "kernel_uniform_cosine_hist.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.23
    for index, metric in enumerate(("low_frequency_energy", "mid_frequency_energy", "high_frequency_energy")):
        ax.bar(x + (index - 1) * width,
               [kernel_frame.loc[kernel_frame.stage == s, metric].median() for s in STAGES],
               width, label=metric.replace("_frequency_energy", ""))
    ax.set_xticks(x, STAGES); ax.set_ylabel("median normalized spectral energy"); ax.legend()
    _save(fig, output_dir / "kernel_frequency_response.png")

    response = validation["gsr_response"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    absent = [response["absent_primary"][s]["mean_delta_logit"] for s in STAGES]
    present = [np.mean([response["by_stage_class"][s][str(c)]["present_delta_logit_mean"] for c in range(4)]) for s in STAGES]
    ax.bar(x - 0.18, absent, 0.36, label="image-absent")
    ax.bar(x + 0.18, present, 0.36, label="image-present")
    ax.axhline(0, color="black", linewidth=0.7); ax.set_xticks(x, STAGES)
    ax.set_ylabel("GSR - Raw pooled logit"); ax.legend()
    _save(fig, output_dir / "gsr_absent_present_delta.png")

    bins = ("B0_le_2", "B1_3_7", "B2_ge_8")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    spatial = validation["ch_spatial_effect"]
    for index, transition in enumerate(("raw_to_ch", "gsr_to_full")):
        ax.bar(np.arange(3) + (index - 0.5) * 0.34,
               [spatial[transition][b]["accuracy_delta"] * 100 for b in bins],
               0.34, label=transition)
    ax.axhline(0, color="black", linewidth=0.7); ax.set_xticks(np.arange(3), bins)
    ax.set_ylabel("accuracy change (percentage points)"); ax.legend()
    _save(fig, output_dir / "ch_boundary_interior_recovery.png")

    finals = validation["final_variants"]
    names = list(finals)
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.bar(np.arange(len(names)), [finals[n]["mean_iou"] * 100 for n in names])
    ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    ax.set_ylabel("official validation mIoU (%)")
    _save(fig, output_dir / "branch_causal_miou.png")

    taxonomy = validation["error_taxonomy"]
    categories = ("absent_class", "present_confusion", "boundary", "interior")
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for index, candidate in enumerate(("gsr_only", "ch_only", "official_full")):
        ax.bar(np.arange(4) + (index - 1) * 0.25,
               [taxonomy[candidate][c]["net"] for c in categories], 0.25, label=candidate)
    ax.axhline(0, color="black", linewidth=0.7); ax.set_xticks(np.arange(4), categories)
    ax.set_ylabel("recovered - harmed pixels"); ax.legend()
    _save(fig, output_dir / "error_taxonomy_recovery.png")

    grouped = gradient_frame.groupby(["loss_branch", "parameter_group"], sort=False).gradient_norm.mean().unstack()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    grouped.plot(kind="bar", logy=True, ax=ax)
    ax.set_ylabel("mean gradient norm (log scale)"); ax.set_xlabel("weighted loss branch")
    _save(fig, output_dir / "loss_gradient_norms.png")

    branches = ("56", "28_1", "28_2", "deep")
    cosines = summary["gradient"]["shared_early_gradient_cosine"]
    matrix = np.eye(4)
    for i, left in enumerate(branches):
        for j, right in enumerate(branches):
            key = f"{left}__{right}" if i <= j else f"{right}__{left}"
            matrix[i, j] = cosines[key]["mean"]
    fig, ax = plt.subplots(figsize=(5.3, 4.5))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(4), branches); ax.set_yticks(range(4), branches)
    ax.set_title("shared-early gradient cosine"); fig.colorbar(image, ax=ax)
    _save(fig, output_dir / "gradient_cosine_matrix.png")

    return sorted(path.name for path in output_dir.glob("*.png"))
