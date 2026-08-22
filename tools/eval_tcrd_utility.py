#!/usr/bin/env python3
"""Finalize the four-branch TCRD-v0 utility gate and emit one route decision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tcrd_common import BRANCH_DIRS, write_json


BRANCHES = ("C0", "D", "R", "DR")
BRANCH_LABELS = {
    "C0": "C0 Control", "D": "D SPED", "R": "R TCER", "DR": "DR Full",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def delta_pp(candidate, control, key="mIoU"):
    return 100.0 * (candidate[key] - control[key])


def present_confusion(predictions, truths):
    total = 0
    by_true_class = np.zeros(4, dtype=np.int64)
    matrix = np.zeros((4, 4), dtype=np.int64)
    masks = []
    for prediction, truth in zip(predictions, truths):
        # Predictions follow the released metric and may already contain label 4
        # where GT background was overwritten. Keep that index explicitly false;
        # only classes 0--3 can be image-level present competitors.
        present = np.zeros(5, dtype=bool)
        values = np.unique(truth)
        present[values[values < 4]] = True
        foreground = truth < 4
        wrong = foreground & (prediction != truth)
        predicted_present = present[prediction]
        mask = wrong & predicted_present
        masks.append(mask)
        total += int(mask.sum())
        for true_class in range(4):
            class_mask = mask & (truth == true_class)
            by_true_class[true_class] += int(class_mask.sum())
            for predicted_class in range(4):
                matrix[true_class, predicted_class] += int(
                    (class_mask & (prediction == predicted_class)).sum()
                )
    return {
        "wrong_pixels": total,
        "wrong_pixels_by_true_class": by_true_class.tolist(),
        "pair_confusion_matrix": matrix.tolist(),
        "masks": np.stack(masks),
    }


def compare_present(base, candidate, truths):
    base_info = present_confusion(base, truths)
    candidate_info = present_confusion(candidate, truths)
    correct_base = base == truths
    correct_candidate = candidate == truths
    recovered = int((base_info["masks"] & correct_candidate).sum())
    harmed = int((correct_base & candidate_info["masks"]).sum())
    reduction = (
        (base_info["wrong_pixels"] - candidate_info["wrong_pixels"])
        / max(1, base_info["wrong_pixels"])
    )
    result = {
        "base_wrong_pixels": base_info["wrong_pixels"],
        "candidate_wrong_pixels": candidate_info["wrong_pixels"],
        "recovered": recovered, "harmed": harmed, "net": recovered - harmed,
        "relative_reduction": reduction,
        "base_by_true_class": base_info["wrong_pixels_by_true_class"],
        "candidate_by_true_class": candidate_info["wrong_pixels_by_true_class"],
        "nonworse_true_classes": sum(
            candidate <= base for base, candidate in zip(
                base_info["wrong_pixels_by_true_class"],
                candidate_info["wrong_pixels_by_true_class"],
            )
        ),
        "candidate_pair_confusion_matrix": candidate_info["pair_confusion_matrix"],
    }
    return result


def boundary_mask(mask):
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[1:] |= mask[1:] != mask[:-1]
    boundary[:-1] |= mask[:-1] != mask[1:]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def compare_boundary(base_predictions, candidate_predictions, truths):
    bins = {
        "B0_le_2": {"total": 0, "base_wrong": 0, "candidate_wrong": 0, "recovered": 0, "harmed": 0},
        "B1_3_to_7": {"total": 0, "base_wrong": 0, "candidate_wrong": 0, "recovered": 0, "harmed": 0},
        "B2_ge_8": {"total": 0, "base_wrong": 0, "candidate_wrong": 0, "recovered": 0, "harmed": 0},
    }
    for base, candidate, truth in zip(base_predictions, candidate_predictions, truths):
        truth28 = cv2.resize(truth, (28, 28), interpolation=cv2.INTER_NEAREST)
        base28 = cv2.resize(base, (28, 28), interpolation=cv2.INTER_NEAREST)
        candidate28 = cv2.resize(candidate, (28, 28), interpolation=cv2.INTER_NEAREST)
        boundary = boundary_mask(truth28)
        distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 5)
        foreground = truth28 < 4
        masks = {
            "B0_le_2": foreground & (distance <= 2.0),
            "B1_3_to_7": foreground & (distance > 2.0) & (distance < 8.0),
            "B2_ge_8": foreground & (distance >= 8.0),
        }
        base_correct = base28 == truth28
        candidate_correct = candidate28 == truth28
        for name, region in masks.items():
            bins[name]["total"] += int(region.sum())
            bins[name]["base_wrong"] += int((region & ~base_correct).sum())
            bins[name]["candidate_wrong"] += int((region & ~candidate_correct).sum())
            bins[name]["recovered"] += int((region & ~base_correct & candidate_correct).sum())
            bins[name]["harmed"] += int((region & base_correct & ~candidate_correct).sum())
    for values in bins.values():
        values["net"] = values["recovered"] - values["harmed"]
        values["accuracy_delta_pp"] = 100.0 * values["net"] / max(1, values["total"])
        values["wrong_relative_change"] = (
            values["candidate_wrong"] - values["base_wrong"]
        ) / max(1, values["base_wrong"])
    return bins


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def plot_results(histories, final_diagnostics, confusion_counts, output):
    output.mkdir(parents=True, exist_ok=True)
    x = np.arange(6)
    labels = ["step0", "e1", "e2", "e3", "e4", "e5"]
    for key, filename, title in (
        ("scores", "miou_curves.png", "Official fused validation mIoU"),
        ("standalone_cam28_1", "cam28_curves.png", "Standalone CAM28_1 validation mIoU"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for branch in BRANCHES:
            ax.plot(x, [100 * row[key]["mIoU"] for row in histories[branch]], marker="o", label=branch)
        ax.set_xticks(x, labels); ax.set_ylabel("mIoU (%)"); ax.set_title(title); ax.legend()
        fig.tight_layout(); fig.savefig(output / filename, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    names = ["D", "DR"]
    positions = np.arange(2)
    ax.bar(positions - 0.18, [final_diagnostics[b]["conductance_same_mean"] for b in names], 0.36, label="same tissue")
    ax.bar(positions + 0.18, [final_diagnostics[b]["conductance_cross_mean"] for b in names], 0.36, label="cross tissue")
    ax.set_xticks(positions, names); ax.set_ylabel("mean normalized conductance")
    ax.set_title("SPED conductance selectivity"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "conductance_same_cross.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(BRANCHES, [confusion_counts[b] for b in BRANCHES])
    ax.set_ylabel("present-class confusion wrong pixels")
    ax.set_title("Epoch5 present-class confusion")
    fig.tight_layout(); fig.savefig(output / "present_confusion.png", dpi=180); plt.close(fig)

    matrices = []
    for branch in ("R", "DR"):
        matrices.append(np.asarray(histories[branch][-1]["mechanism_parameters"]["competition_matrix"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for ax, branch, matrix in zip(axes, ("R", "DR"), matrices):
        image = ax.imshow(matrix, cmap="viridis")
        ax.set_title(branch); ax.set_xlabel("competitor"); ax.set_ylabel("target")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(output / "reaction_matrix.png", dpi=180); plt.close(fig)


def run(args):
    experiment = Path(args.experiment_dir)
    comparison = experiment / "comparison"
    figures = experiment / "figures"
    docs = experiment / "docs"
    for directory in (comparison, figures, docs):
        directory.mkdir(parents=True, exist_ok=True)
    if (comparison / "route_decision.json").exists():
        raise FileExistsError("TCRD utility decision already exists")

    histories = {}
    completions = {}
    prediction_data = {}
    for branch in BRANCHES:
        branch_dir = experiment / BRANCH_DIRS[branch]
        histories[branch] = read_json(branch_dir / "validation" / "history.json")
        completions[branch] = read_json(branch_dir / "complete.json")
        if len(histories[branch]) != 6 or histories[branch][-1]["epoch"] != 5:
            raise AssertionError(f"Incomplete validation history for {branch}")
        with np.load(branch_dir / "predictions" / "epoch5_validation.npz") as data:
            prediction_data[branch] = {
                "image_ids": data["image_ids"].copy(),
                "predictions": data["predictions"].copy(),
                "truths": data["truths"].copy(),
            }
    reference_ids = prediction_data["C0"]["image_ids"]
    truths = prediction_data["C0"]["truths"]
    for branch in BRANCHES[1:]:
        if not np.array_equal(reference_ids, prediction_data[branch]["image_ids"]):
            raise AssertionError("Validation order mismatch")
        if not np.array_equal(truths, prediction_data[branch]["truths"]):
            raise AssertionError("Validation truth mismatch")

    epoch_rows = []
    for branch in BRANCHES:
        for row in histories[branch]:
            epoch_rows.append({
                "branch": branch, "point": row["point"], "epoch": row["epoch"],
                "mIoU": row["scores"]["mIoU"], "mDice": row["scores"]["mDice"],
                "cam28_1_mIoU": row["standalone_cam28_1"]["mIoU"],
                "cam28_1_mDice": row["standalone_cam28_1"]["mDice"],
                **{f"class_{i}_iou": row["scores"]["class_iou"][str(i)] for i in range(4)},
            })
    write_csv(comparison / "epoch_metrics.csv", epoch_rows)

    final = {branch: histories[branch][-1] for branch in BRANCHES}
    final_diag = {branch: final[branch]["diagnostics"] for branch in BRANCHES}
    c0_scores = final["C0"]["scores"]
    deltas = {}
    for branch in BRANCHES[1:]:
        deltas[branch] = {
            "final_mIoU_pp": delta_pp(final[branch]["scores"], c0_scores),
            "final_mDice_pp": delta_pp(final[branch]["scores"], c0_scores, "mDice"),
            "cam28_1_mIoU_pp": delta_pp(
                final[branch]["standalone_cam28_1"], final["C0"]["standalone_cam28_1"]
            ),
            "class_iou_pp": {
                str(index): 100 * (
                    final[branch]["scores"]["class_iou"][str(index)]
                    - c0_scores["class_iou"][str(index)]
                ) for index in range(4)
            },
        }

    present_absolute = {
        branch: present_confusion(prediction_data[branch]["predictions"], truths)
        for branch in BRANCHES
    }
    confusion_counts = {branch: value["wrong_pixels"] for branch, value in present_absolute.items()}
    present_comparisons = {
        "R_vs_C0": compare_present(
            prediction_data["C0"]["predictions"], prediction_data["R"]["predictions"], truths
        ),
        "DR_vs_C0": compare_present(
            prediction_data["C0"]["predictions"], prediction_data["DR"]["predictions"], truths
        ),
        "DR_vs_D": compare_present(
            prediction_data["D"]["predictions"], prediction_data["DR"]["predictions"], truths
        ),
    }
    boundary_comparisons = {
        "D_vs_C0": compare_boundary(
            prediction_data["C0"]["predictions"], prediction_data["D"]["predictions"], truths
        ),
        "DR_vs_C0": compare_boundary(
            prediction_data["C0"]["predictions"], prediction_data["DR"]["predictions"], truths
        ),
        "DR_vs_D": compare_boundary(
            prediction_data["D"]["predictions"], prediction_data["DR"]["predictions"], truths
        ),
    }

    d_boundary = boundary_comparisons["D_vs_C0"]["B0_le_2"]
    d_boundary_nonincrease = d_boundary["candidate_wrong"] <= d_boundary["base_wrong"]
    d_boundary_significant_increase = d_boundary["wrong_relative_change"] >= 0.005
    d_positive_classes = sum(value > 0 for value in deltas["D"]["class_iou_pp"].values())
    d_nonnegative_classes = sum(value >= 0 for value in deltas["D"]["class_iou_pp"].values())
    d_ratio = final_diag["D"]["conductance_same_cross_ratio"]
    if (
        deltas["D"]["final_mIoU_pp"] < 0.05
        or deltas["D"]["cam28_1_mIoU_pp"] <= 0
        or d_boundary_significant_increase
    ):
        d_decision = "SPED_UTILITY_NOGO"
    elif (
        deltas["D"]["final_mIoU_pp"] >= 0.30
        and deltas["D"]["cam28_1_mIoU_pp"] >= 0.30
        and d_nonnegative_classes >= 3 and d_ratio >= 1.05
        and d_boundary_nonincrease
    ):
        d_decision = "SPED_UTILITY_STRONG_PASS"
    elif (
        deltas["D"]["final_mIoU_pp"] >= 0.15
        and deltas["D"]["cam28_1_mIoU_pp"] >= 0.20
        and d_positive_classes >= 2 and d_ratio >= 1.05
    ):
        d_decision = "SPED_UTILITY_PASS"
    else:
        d_decision = "SPED_UTILITY_REVIEW"

    r_confusion = present_comparisons["R_vs_C0"]
    reaction_overconfidence_review = bool(
        final_diag["R"]["present_entropy_zt"] < final_diag["R"]["present_entropy_z0"]
        and final_diag["R"]["present_top1_top2_margin_zt"]
        > final_diag["R"]["present_top1_top2_margin_z0"]
        and r_confusion["relative_reduction"] <= 0
    )
    r_positive_classes = sum(value > 0 for value in deltas["R"]["class_iou_pp"].values())
    r_nonnegative_classes = sum(value >= 0 for value in deltas["R"]["class_iou_pp"].values())
    if (
        deltas["R"]["final_mIoU_pp"] < 0.05
        or deltas["R"]["cam28_1_mIoU_pp"] <= 0
        or r_confusion["relative_reduction"] <= 0
    ):
        r_decision = "TCER_UTILITY_NOGO"
    elif (
        deltas["R"]["final_mIoU_pp"] >= 0.30
        and deltas["R"]["cam28_1_mIoU_pp"] >= 0.30
        and r_confusion["relative_reduction"] >= 0.01
        and r_nonnegative_classes >= 3
    ):
        r_decision = "TCER_UTILITY_STRONG_PASS"
    elif (
        deltas["R"]["final_mIoU_pp"] >= 0.15
        and deltas["R"]["cam28_1_mIoU_pp"] >= 0.20
        and r_confusion["relative_reduction"] >= 0.005
        and r_positive_classes >= 2
    ):
        r_decision = "TCER_UTILITY_PASS"
    else:
        r_decision = "TCER_UTILITY_REVIEW"

    dr_delta = deltas["DR"]["final_mIoU_pp"]
    best_single = max(deltas["D"]["final_mIoU_pp"], deltas["R"]["final_mIoU_pp"])
    dr_present_decrease = present_comparisons["DR_vs_C0"]["relative_reduction"] > 0
    dr_boundary_nonincrease = (
        boundary_comparisons["DR_vs_C0"]["B0_le_2"]["candidate_wrong"]
        <= boundary_comparisons["DR_vs_C0"]["B0_le_2"]["base_wrong"]
    )
    if dr_delta < 0.05 or dr_delta < min(
        deltas["D"]["final_mIoU_pp"], deltas["R"]["final_mIoU_pp"]
    ) - 0.02:
        dr_decision = "TCRD_FULL_NOGO"
    elif (
        dr_delta >= 0.20 and dr_delta >= best_single + 0.05
        and deltas["DR"]["cam28_1_mIoU_pp"] >= 0.25
        and dr_present_decrease and dr_boundary_nonincrease
    ):
        dr_decision = "TCRD_FULL_SYNERGY_PASS"
    elif dr_delta >= 0.15 and dr_delta >= best_single - 0.02:
        dr_decision = "TCRD_FULL_ADDITIVE_PASS"
    else:
        dr_decision = "TCRD_FULL_REVIEW"

    d_pass = d_decision in ("SPED_UTILITY_PASS", "SPED_UTILITY_STRONG_PASS")
    r_pass = r_decision in ("TCER_UTILITY_PASS", "TCER_UTILITY_STRONG_PASS")
    dr_pass = dr_decision in ("TCRD_FULL_SYNERGY_PASS", "TCRD_FULL_ADDITIVE_PASS")
    if d_pass and r_pass and dr_pass:
        route = "ROUTE_A_TCRD_FULL"
    elif d_pass and not r_pass and dr_delta <= deltas["D"]["final_mIoU_pp"] + 0.02:
        route = "ROUTE_B_DIFFUSION_ONLY"
    elif r_pass and not d_pass and dr_delta <= deltas["R"]["final_mIoU_pp"] + 0.02:
        route = "ROUTE_C_REACTION_ONLY"
    elif not d_pass and not r_pass and dr_decision == "TCRD_FULL_SYNERGY_PASS":
        route = "ROUTE_D_UNIFIED_ONLY_SYNERGY"
    else:
        route = "ROUTE_E_CLOSE"

    mechanism_rows = []
    for branch in BRANCHES:
        row = final_diag[branch]
        mechanism_rows.append({
            "branch": branch,
            "diffusion_update_ratio": row["diffusion_update_ratio"],
            "reaction_update_ratio": row["reaction_update_ratio"],
            "conductance_same_mean": row["conductance_same_mean"],
            "conductance_cross_mean": row["conductance_cross_mean"],
            "conductance_same_cross_ratio": row["conductance_same_cross_ratio"],
            "entropy_z0": row["present_entropy_z0"],
            "entropy_zt": row["present_entropy_zt"],
            "margin_z0": row["present_top1_top2_margin_z0"],
            "margin_zt": row["present_top1_top2_margin_zt"],
            "present_confusion_wrong_pixels": confusion_counts[branch],
        })
    write_csv(comparison / "mechanism_metrics.csv", mechanism_rows)

    error_taxonomy = {
        "boundary_grid": "28x28 nearest-neighbor diagnostic grid",
        "boundary_significant_increase_definition": ">=0.5% relative increase in B0 wrong pixels",
        "present_confusion": {
            "absolute": {
                branch: {key: value for key, value in info.items() if key != "masks"}
                for branch, info in present_absolute.items()
            },
            "comparisons": present_comparisons,
        },
        "boundary_comparisons": boundary_comparisons,
    }
    write_json(comparison / "error_taxonomy.json", error_taxonomy)
    taxonomy_rows = []
    for comparison_name, bins in boundary_comparisons.items():
        for region, values in bins.items():
            taxonomy_rows.append({"comparison": comparison_name, "region": region, **values})
    write_csv(comparison / "error_taxonomy.csv", taxonomy_rows)

    step0_review = {
        branch: 100 * (
            histories[branch][0]["scores"]["mIoU"] - histories["C0"][0]["scores"]["mIoU"]
        ) for branch in BRANCHES[1:]
    }
    decisions = {
        "route": route,
        "branch_decisions": {"D": d_decision, "R": r_decision, "DR": dr_decision},
        "deltas": deltas,
        "step0_delta_mIoU_pp": step0_review,
        "step0_catastrophic_drop": {
            branch: value < -3.0 for branch, value in step0_review.items()
        },
        "mechanism_checks": {
            "D_same_cross_ratio_ge_1_05": d_ratio >= 1.05,
            "D_update_ratio_ge_0_005": final_diag["D"]["diffusion_update_ratio"] >= 0.005,
            "D_boundary_nonincreasing": d_boundary_nonincrease,
            "R_update_ratio_ge_0_005": final_diag["R"]["reaction_update_ratio"] >= 0.005,
            "R_present_confusion_relative_reduction": r_confusion["relative_reduction"],
            "R_present_confusion_nonworse_classes": r_confusion["nonworse_true_classes"],
            "REACTION_OVERCONFIDENCE_REVIEW": reaction_overconfidence_review,
            "DR_present_confusion_decreased": dr_present_decrease,
            "DR_boundary_nonincreasing": dr_boundary_nonincrease,
        },
        "primary_comparison": "epoch5 candidate minus epoch5 C0",
        "best_epoch_not_used_for_gate": True,
        "test_used": False, "luad_used": False,
    }
    write_json(comparison / "route_decision.json", decisions)
    plot_results(histories, final_diag, confusion_counts, figures)

    lines = [
        "# TCRD-v0 Matched 5-Epoch Utility Gate", "",
        "## 1. Executive conclusion", "",
        f"- Final route: **{route}**",
        f"- SPED: **{d_decision}**",
        f"- TCER: **{r_decision}**",
        f"- Full TCRD: **{dr_decision}**",
        "- Primary comparisons use epoch5 candidate minus epoch5 matched C0; best epochs did not select the gate result.",
        "", "## 2. Matched experimental control", "",
        "- All four branches started from the same SHA-locked SSHR A0 seed42 epoch25 FINAL checkpoint.",
        "- All reused the same 5×1171×20 schedule of indices, per-sample augmentation seeds and per-step model seeds.",
        "- All original SSHR parameters remained trainable under the same derived epoch20/25 tail-replay PolyOptimizer schedule.",
        "- Effective batch20, 224×224, BF16 and the official four classification weights were frozen.",
        "", "## 3. Required executive table", "",
        "| Branch | Step0 mIoU | Epoch5 mIoU | Δ vs C0 | CAM28_1 Δ | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for branch in BRANCHES:
        delta = None if branch == "C0" else deltas[branch]
        decision = "Reference" if branch == "C0" else {"D": d_decision, "R": r_decision, "DR": dr_decision}[branch]
        final_delta_text = "—" if delta is None else f"{delta['final_mIoU_pp']:+.4f}"
        cam_delta_text = "—" if delta is None else f"{delta['cam28_1_mIoU_pp']:+.4f}"
        lines.append(
            f"| {BRANCH_LABELS[branch]} | {100*histories[branch][0]['scores']['mIoU']:.4f} | "
            f"{100*final[branch]['scores']['mIoU']:.4f} | "
            f"{final_delta_text} | {cam_delta_text} | {decision} |"
        )
    lines += ["", "## 4. Epoch-by-epoch validation", ""]
    lines += ["| Point | " + " | ".join(BRANCHES) + " |", "|---|" + "---:|" * 4]
    for index, point in enumerate(("step0", "epoch1", "epoch2", "epoch3", "epoch4", "epoch5")):
        lines.append("| " + point + " | " + " | ".join(
            f"{100*histories[branch][index]['scores']['mIoU']:.4f}" for branch in BRANCHES
        ) + " |")
    lines += [
        "", "## 5. Required mechanism table", "",
        "| Branch | Update RMS/Z0 RMS | Main mechanism metric | Boundary Δ | Present-confusion Δ |",
        "|---|---:|---:|---:|---:|",
        f"| D | {final_diag['D']['diffusion_update_ratio']:.6f} | same/cross={d_ratio:.6f} | {d_boundary['accuracy_delta_pp']:+.4f} pp | — |",
        f"| R | {final_diag['R']['reaction_update_ratio']:.6f} | ηR={final['R']['mechanism_parameters']['eta_r']:.6f} | — | {100*r_confusion['relative_reduction']:+.4f}% |",
        f"| DR | D={final_diag['DR']['diffusion_update_ratio']:.6f}; R={final_diag['DR']['reaction_update_ratio']:.6f} | same/cross={final_diag['DR']['conductance_same_cross_ratio']:.6f} | {boundary_comparisons['DR_vs_C0']['B0_le_2']['accuracy_delta_pp']:+.4f} pp | {100*present_comparisons['DR_vs_C0']['relative_reduction']:+.4f}% |",
        "", "## 6. SPED finding", "",
        f"- Same/cross conductance: {final_diag['D']['conductance_same_mean']:.8f} / {final_diag['D']['conductance_cross_mean']:.8f} (ratio {d_ratio:.6f}).",
        f"- Diffusion update RMS/Z0 RMS: {final_diag['D']['diffusion_update_ratio']:.6f}.",
        f"- B0 boundary recovered/harmed/net: {d_boundary['recovered']}/{d_boundary['harmed']}/{d_boundary['net']}.",
        "", "## 7. TCER finding", "",
        f"- Reaction update RMS/Z0 RMS: {final_diag['R']['reaction_update_ratio']:.6f}.",
        f"- Present-confusion wrong pixels C0/R: {r_confusion['base_wrong_pixels']}/{r_confusion['candidate_wrong_pixels']} ({100*r_confusion['relative_reduction']:+.4f}%).",
        f"- Present entropy Z0→ZT: {final_diag['R']['present_entropy_z0']:.6f}→{final_diag['R']['present_entropy_zt']:.6f}.",
        f"- Top1–top2 margin Z0→ZT: {final_diag['R']['present_top1_top2_margin_z0']:.6f}→{final_diag['R']['present_top1_top2_margin_zt']:.6f}.",
        f"- REACTION_OVERCONFIDENCE_REVIEW: {reaction_overconfidence_review}.",
        "", "## 8. Best epochs (observation only)", "",
    ]
    for branch in BRANCHES:
        best = max(histories[branch], key=lambda row: row["scores"]["mIoU"])
        lines.append(f"- {branch}: {best['point']} mIoU={100*best['scores']['mIoU']:.4f}; not used by the gate.")
    lines += [
        "", "## 9. Provenance and artifacts", "",
        "- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`",
        "- Validation pairs: 3,418; official TTA, predicted presence, hard gate, min-max, 0/0.6/0.2/0.2 fusion and released metric.",
        "- Machine-readable metrics are under `comparison/`; branch histories and final checkpoints remain in their branch directories.",
        "", "## 10. Interpretation limit", "",
        "This mature-checkpoint continuation gate evaluates whether a mechanism can improve a mature SSHR representation. It does not establish fresh-25-epoch performance, multi-seed stability, LUAD generalization or a publication claim.",
        "", "## 11. Stop boundary", "",
        "No test, LUAD, other seed, fresh 25-epoch training, hierarchy expansion, T/eta/formula change or auxiliary loss was run.",
        "", f"**{route}**", "", "STOP.",
    ]
    report = docs / "tcrd_v0_utility_gate_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "route": route,
        "branch_decisions": decisions["branch_decisions"],
        "deltas": deltas,
        "report": str(report),
    }, indent=2, sort_keys=True), flush=True)
    print(route, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
