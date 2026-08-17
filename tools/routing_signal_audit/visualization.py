"""Deterministic Phase-0B summary and automatically selected routing panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

from tools.routing_signal_audit import BRANCH_NAMES, SAFE_CANDIDATES


MASK_CMAP = ListedColormap(["#d73027", "#4575b4", "#1a9850", "#984ea3", "#ffffff"])


def generate_summary_figures(
    figures_dir: Path,
    oracle_summaries: list[dict],
    slide_rows: list[dict],
    correlation_rows: list[dict],
    primary_prediction: np.ndarray,
    true_relative: np.ndarray,
    choices: np.ndarray,
    fold_rows: list[dict],
    bootstrap_rows: list[dict],
) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["method"] for row in oracle_summaries]
    values = [row["mIoU"] for row in oracle_summaries]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, values, color="#4c78a8")
    axis.set_ylabel("Official mIoU (%)")
    axis.set_title("Frozen safe-routing oracle ladder")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(figures_dir / "oracle_ladder.png", dpi=180)
    plt.close(figure)

    selected = [row for row in slide_rows if str(row["selected"]).lower() == "true"]
    counts = {candidate: 0 for candidate in SAFE_CANDIDATES}
    for row in selected:
        counts[row["candidate"]] += 1
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(list(counts), list(counts.values()), color="#72b7b2")
    axis.set_ylabel("Selected source slides")
    axis.set_title("Slide-level safe-oracle preference")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(figures_dir / "slide_preference.png", dpi=180)
    plt.close(figure)

    all_correlations = [
        row for row in correlation_rows if row["scope"] == "all"
    ]
    ranked = sorted(
        all_correlations,
        key=lambda row: abs(row["spearman_relative_utility"]),
        reverse=True,
    )[:24]
    matrix = np.asarray(
        [[row["spearman_relative_utility"]] for row in ranked], dtype=float
    )
    figure, axis = plt.subplots(figsize=(7, 9))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-0.5, vmax=0.5, aspect="auto")
    axis.set_yticks(range(len(ranked)))
    axis.set_yticklabels(
        [f"{row['signal_set']}:{row['signal']}" for row in ranked], fontsize=7
    )
    axis.set_xticks([0], ["Spearman(r)"])
    axis.set_title("Strongest diagnostic signal-target correlations")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(figures_dir / "signal_correlation_heatmap.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        true_relative.reshape(-1),
        primary_prediction.reshape(-1),
        s=4,
        alpha=0.12,
        color="#4c78a8",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("True diagnostic relative utility")
    axis.set_ylabel("Predicted relative utility")
    axis.set_title("Primary MLP-C OOF gain prediction")
    figure.tight_layout()
    figure.savefig(figures_dir / "predicted_vs_true_gain.png", dpi=180)
    plt.close(figure)

    overridden = choices >= 0
    chosen_gain = np.zeros(len(choices), dtype=float)
    indices = np.flatnonzero(overridden)
    chosen_gain[indices] = true_relative[indices, choices[indices]]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(chosen_gain[overridden], bins=50, color="#f58518", alpha=0.8)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("True utility of MLP-C override")
    axis.set_ylabel("Images")
    axis.set_title("Override gain/loss distribution")
    figure.tight_layout()
    figure.savefig(figures_dir / "override_gain_distribution.png", dpi=180)
    plt.close(figure)

    primary_folds = [row for row in fold_rows if row["probe"] == "MLP-C"]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        [str(row["fold"]) for row in primary_folds],
        [row["delta_mIoU"] for row in primary_folds],
        color="#54a24b",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("GroupKFold fold")
    axis.set_ylabel("Delta mIoU (pp)")
    axis.set_title("Primary MLP-C held-out fold stability")
    figure.tight_layout()
    figure.savefig(figures_dir / "fold_delta.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(
        [row["delta_mIoU"] for row in bootstrap_rows],
        bins=50,
        color="#e45756",
        alpha=0.8,
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("Paired slide-bootstrap delta mIoU (pp)")
    axis.set_ylabel("Replicates")
    axis.set_title("MLP-C grouped bootstrap")
    figure.tight_layout()
    figure.savefig(figures_dir / "bootstrap_delta.png", dpi=180)
    plt.close(figure)


def generate_qualitative_routing_panels(
    validation_root: Path,
    phase0_cache_dir: Path,
    figures_dir: Path,
    image_names: list[str],
    predicted_relative: np.ndarray,
    true_relative: np.ndarray,
    router_choices: np.ndarray,
    safe_oracle_choices: np.ndarray,
    signal_a: np.ndarray,
    signal_a_names: list[str],
    tta_features: np.ndarray,
    tta_names: list[str],
) -> list[dict]:
    figures_dir = Path(figures_dir)
    qualitative_dir = figures_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(phase0_cache_dir)
    truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    branches = np.load(cache_dir / "branch_predictions.npy", mmap_mode="r")
    overridden = router_choices >= 0
    chosen_gain = np.zeros(len(router_choices), dtype=float)
    override_indices = np.flatnonzero(overridden)
    chosen_gain[override_indices] = true_relative[
        override_indices, router_choices[override_indices]
    ]
    best_available = np.max(true_relative, axis=1)
    categories = {
        "type_a_successful_override": (
            overridden & (chosen_gain > 0), np.abs(chosen_gain)
        ),
        "type_b_harmful_override": (
            overridden & (chosen_gain < 0), np.abs(chosen_gain)
        ),
        "type_c_missed_opportunity": (
            ~overridden & (best_available > 0), np.abs(best_available)
        ),
        "type_d_correct_fallback": (
            ~overridden & (best_available <= 0), np.abs(best_available)
        ),
    }
    signal_lookup = {name: index for index, name in enumerate(signal_a_names)}
    tta_lookup = {name: index for index, name in enumerate(tta_names)}
    rows = []
    for category, (eligible, magnitude) in categories.items():
        candidates = np.flatnonzero(eligible)
        candidates = candidates[
            np.argsort(-magnitude[candidates], kind="stable")
        ]
        if len(candidates) < 8:
            raise RuntimeError(
                f"Qualitative category {category} has only {len(candidates)} cases; expected at least 8"
            )
        for rank, index in enumerate(candidates[:8], start=1):
            name = image_names[index]
            image = np.asarray(
                Image.open(Path(validation_root) / "img" / f"{name}.png").convert("RGB")
            )
            panels = [
                ("Input", image, None),
                ("GT", truth[index], MASK_CMAP),
                ("Official", official[index], MASK_CMAP),
                *[
                    (branch_name, branches[index, branch_index], MASK_CMAP)
                    for branch_index, branch_name in enumerate(BRANCH_NAMES)
                ],
            ]
            figure, axes = plt.subplots(2, 4, figsize=(16, 8))
            for axis, (title, value, cmap) in zip(axes.reshape(-1)[:7], panels):
                axis.imshow(value, cmap=cmap, vmin=0 if cmap else None, vmax=4 if cmap else None)
                axis.set_title(title)
                axis.axis("off")
            branch_for_signal = (
                int(router_choices[index]) if router_choices[index] >= 0 else 0
            )
            text_axis = axes.reshape(-1)[7]
            text_axis.axis("off")
            text_lines = [
                "pred r: " + ", ".join(f"{value:+.3f}" for value in predicted_relative[index]),
                "true r: " + ", ".join(f"{value:+.3f}" for value in true_relative[index]),
                f"router: {SAFE_CANDIDATES[int(router_choices[index]) + 1] if router_choices[index] >= 0 else 'official_fusion'}",
                f"safe oracle: {SAFE_CANDIDATES[int(safe_oracle_choices[index])]}",
                f"TTA consistency: {tta_features[index, branch_for_signal, tta_lookup['tta_three_view_argmax_consistency']]:.3f}",
                f"TTA JSD: {tta_features[index, branch_for_signal, tta_lookup['tta_jsd_mean']]:.4f}",
                f"cross JSD: {signal_a[index, branch_for_signal, signal_lookup['to_others_jsd_mean_mean']]:.4f}",
                f"majority agree: {signal_a[index, branch_for_signal, signal_lookup['agreement_with_majority_branch']]:.3f}",
            ]
            text_axis.text(0.0, 1.0, "\n".join(text_lines), va="top", fontsize=10)
            figure.suptitle(
                f"{category} rank {rank}: {name} | |gain|={magnitude[index]:.4f}",
                fontsize=11,
            )
            figure.tight_layout()
            filename = f"{category}_{rank:02d}_{name}.png"
            figure.savefig(qualitative_dir / filename, dpi=140)
            plt.close(figure)
            rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "index": int(index),
                    "image_name": name,
                    "absolute_true_utility": float(magnitude[index]),
                    "router_choice": (
                        SAFE_CANDIDATES[int(router_choices[index]) + 1]
                        if router_choices[index] >= 0
                        else "official_fusion"
                    ),
                    "safe_oracle_choice": SAFE_CANDIDATES[
                        int(safe_oracle_choices[index])
                    ],
                    "figure": f"figures/qualitative/{filename}",
                }
            )
    return rows
