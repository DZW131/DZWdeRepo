"""Deterministic figures and automatically ranked qualitative examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

from tools.decision_audit import BRANCH_NAMES


MASK_CMAP = ListedColormap(
    ["#d73027", "#4575b4", "#1a9850", "#984ea3", "#f7f7f7"]
)


def _annotated_heatmap(values, xlabels, ylabels, title, path, fmt=".2f"):
    figure, axis = plt.subplots(figsize=(7, 4.8))
    image = axis.imshow(values, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(xlabels)), labels=xlabels, rotation=25, ha="right")
    axis.set_yticks(range(len(ylabels)), labels=ylabels)
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                format(values[row, column], fmt),
                ha="center",
                va="center",
                color="white" if values[row, column] < values.max() * 0.72 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_summary_figures(
    figures_dir: Path,
    individual_rows: list[dict],
    error_overlap_rows: list[dict],
    preference_rows: list[dict],
    oracle_rows: list[dict],
    confidence_rows: list[dict],
):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    branch_rows = {
        row["prediction"]: row
        for row in individual_rows
        if row["prediction"] in BRANCH_NAMES
    }
    class_matrix = np.asarray(
        [
            [branch_rows[name][f"class{class_id}_iou"] for name in BRANCH_NAMES]
            for class_id in range(4)
        ]
    )
    _annotated_heatmap(
        class_matrix,
        BRANCH_NAMES,
        [f"class{class_id}" for class_id in range(4)],
        "Class × hierarchy IoU (%)",
        figures_dir / "class_stage_heatmap.png",
    )

    overall_overlap = [row for row in error_overlap_rows if row["class_id"] == -1]
    overlap_matrix = np.zeros((4, 4), dtype=np.float64)
    for row in overall_overlap:
        overlap_matrix[
            BRANCH_NAMES.index(row["branch_i"]),
            BRANCH_NAMES.index(row["branch_j"]),
        ] = row["jaccard"]
    _annotated_heatmap(
        overlap_matrix,
        BRANCH_NAMES,
        BRANCH_NAMES,
        "Foreground error-set Jaccard",
        figures_dir / "error_overlap_heatmap.png",
        fmt=".3f",
    )

    image_preferences = [
        row for row in preference_rows if row["level"] == "image"
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        [row["branch"] for row in image_preferences],
        [row["fraction"] * 100 for row in image_preferences],
        color="#4c78a8",
    )
    axis.set_ylabel("Best-branch images (%)")
    axis.set_title("Image-level branch oracle preference")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(figures_dir / "best_branch_histogram.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        [row["method"] for row in oracle_rows],
        [row["mIoU"] for row in oracle_rows],
        color=["#777777", "#4c78a8", "#f58518", "#54a24b"],
    )
    axis.set_ylabel("Official mIoU (%)")
    axis.set_title("Frozen hierarchy oracle ceilings")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(figures_dir / "oracle_ceiling_chart.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for branch_name in BRANCH_NAMES:
        rows = sorted(
            [row for row in confidence_rows if row["branch"] == branch_name],
            key=lambda row: row["bin"],
        )
        axis.plot(
            [row["mean_confidence"] for row in rows],
            [row["pixel_accuracy"] for row in rows],
            marker="o",
            label=branch_name,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Foreground pixel accuracy")
    axis.set_title("Confidence–accuracy curves (10 equal-count bins)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "confidence_accuracy_curves.png", dpi=180)
    plt.close(figure)


def generate_qualitative_examples(
    validation_root: Path,
    cache_dir: Path,
    figures_dir: Path,
) -> list[dict]:
    validation_root = Path(validation_root)
    cache_dir = Path(cache_dir)
    qualitative_dir = Path(figures_dir) / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    names = (cache_dir / "image_paths.txt").read_text(encoding="utf-8").splitlines()
    ground_truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    branches = np.load(cache_dir / "branch_predictions.npy", mmap_mode="r")
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    image_class = np.load(
        cache_dir / "image_class_oracle_predictions.npy", mmap_mode="r"
    )
    foreground = ground_truth < 4
    official_wrong = official != ground_truth
    branch_correct = np.stack(
        [branches[:, branch] == ground_truth for branch in range(4)], axis=1
    )
    ranking = {
        "type_a_cam56": (
            foreground & official_wrong & branch_correct[:, 0]
        ).sum(axis=(1, 2)),
        "type_b_cam28_1": (
            foreground & official_wrong & branch_correct[:, 1]
        ).sum(axis=(1, 2)),
        "type_c_cam28_2_or_deep": (
            foreground
            & official_wrong
            & (branch_correct[:, 2] | branch_correct[:, 3])
        ).sum(axis=(1, 2)),
        "type_d_all_wrong": (
            foreground & ~np.any(branch_correct, axis=1)
        ).sum(axis=(1, 2)),
    }
    selected = set()
    rows = []
    for category, counts in ranking.items():
        candidates = np.argsort(-counts, kind="stable")
        category_indices = []
        for index in candidates:
            if counts[index] <= 0 or int(index) in selected:
                continue
            selected.add(int(index))
            category_indices.append(int(index))
            if len(category_indices) == 6:
                break
        for rank, index in enumerate(category_indices, start=1):
            name = names[index]
            image = np.asarray(
                Image.open(validation_root / "img" / f"{name}.png").convert("RGB")
            )
            panels = [
                ("Input", image, None),
                ("GT", ground_truth[index], MASK_CMAP),
                ("CAM56", branches[index, 0], MASK_CMAP),
                ("CAM28_1", branches[index, 1], MASK_CMAP),
                ("CAM28_2", branches[index, 2], MASK_CMAP),
                ("CAMdeep", branches[index, 3], MASK_CMAP),
                ("Official", official[index], MASK_CMAP),
                ("Image-Class Oracle", image_class[index], MASK_CMAP),
            ]
            figure, axes = plt.subplots(1, 8, figsize=(24, 3.2))
            for axis, (title, value, cmap) in zip(axes, panels):
                axis.imshow(value, cmap=cmap, vmin=0 if cmap else None, vmax=4 if cmap else None)
                axis.set_title(title, fontsize=9)
                axis.axis("off")
            figure.suptitle(
                f"{category} rank {rank}: {name} | recoverable={int(counts[index])}",
                fontsize=10,
            )
            figure.tight_layout()
            output_name = f"{category}_{rank:02d}_{name}.png"
            figure.savefig(qualitative_dir / output_name, dpi=150)
            plt.close(figure)
            rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "image_name": name,
                    "recoverable_pixels": int(counts[index]),
                    "figure": f"figures/qualitative/{output_name}",
                }
            )
    return rows
