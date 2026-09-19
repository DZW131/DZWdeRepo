"""Generate DLAG aggregate figures and 5x20 replay-validated case panels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audits.dlag_oracle_v1.oracle_bank import ALPHAS, infer_alpha_bank, unpack_mask
from audits.ucrf_v1.gate import load_model
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import prediction_from_cam, presence


def pick_top(frame: pd.DataFrame, criterion: pd.Series, score: pd.Series, count: int = 20) -> list[int]:
    chosen = frame.loc[criterion].assign(_score=score[criterion]).sort_values(
        ["_score", "area"], ascending=False).index.tolist()[:count]
    if len(chosen) < count:
        chosen.extend(index for index in frame.sort_values("area", ascending=False).index
                      if index not in chosen)
    return [int(value) for value in chosen[:count]]


def select_cases(output: Path, ucrf_output: Path, val_root: Path) -> tuple[pd.DataFrame, dict]:
    components = pd.read_csv(output / "joint_oracle/recoverability_table.csv")
    selections, manifest = [], {}
    rules = {
        "arbitration_only_recovered": ((components.A_correct_pixels > 0) &
                                        (components.B_correct_pixels == 0), components.A_gain_pixels),
        "gate_only_recovered": ((components.B_correct_pixels > 0) &
                                 (components.A_correct_pixels == 0), components.B_gain_pixels),
        "joint_only_recovered": (components.joint_only_correct_pixels > 0,
                                  components.joint_only_correct_pixels),
        "joint_unrecoverable": (components.AB_correct_pixels == 0, components.area),
    }
    for category, (criterion, score) in rules.items():
        indices = pick_top(components, criterion, score)
        manifest[category] = {"criterion_available": int(criterion.sum()),
                              "selected_meeting_criterion": int(sum(bool(criterion.iloc[i]) for i in indices)),
                              "selected": len(indices)}
        for index in indices:
            row = components.iloc[index]
            alpha = row.best_alpha_AB if category == "joint_only_recovered" else row.best_alpha_A
            selections.append({"category": category, "component_table_index": index,
                "frame_index": int(row.frame_index), "alpha": float(alpha), "source": "M1"})

    ucrf = pd.read_parquet(ucrf_output / "metrics/component_event_table.parquet")
    masks = np.load(ucrf_output / "masks/whole.npz")["packed"]
    base = np.load(output / "counterfactuals/alpha_100/predictions.npz")["predictions"]
    strong = np.load(output / "counterfactuals/alpha_400/predictions.npz")["predictions"]
    ids = np.load(output / "counterfactuals/alpha_100/predictions.npz")["image_ids"]
    lookup = {str(value): index for index, value in enumerate(ids)}
    harmed = []
    for frame_index, row in ucrf[ucrf.cohort == "TP_candidate"].iterrows():
        image_index = lookup[str(row.image_id)]
        truth = np.asarray(Image.open(val_root / "mask" / f"{row.image_id}.png"))
        mask = unpack_mask(masks[frame_index], truth.shape) & (truth < 4)
        harm = int(((base[image_index] == truth) & (strong[image_index] != truth) & mask).sum())
        if harm:
            harmed.append({"frame_index": int(frame_index), "harm": harm, "area": int(row.area)})
    harmed_available = len(harmed)
    harmed = sorted(harmed, key=lambda item: (item["harm"], item["area"]), reverse=True)[:20]
    if len(harmed) != 20:
        raise AssertionError("Fewer than 20 strong-alpha TP-harm cases")
    manifest["tp_harmed_strong_alpha"] = {"criterion_available": harmed_available,
        "selected_meeting_criterion": 20, "selected": 20}
    selections.extend({"category": "tp_harmed_strong_alpha", "component_table_index": -1,
        "frame_index": item["frame_index"], "alpha": 4., "source": "TP"} for item in harmed)
    result = pd.DataFrame(selections)
    if len(result) != 100:
        raise AssertionError("Expected exactly 100 cases")
    return result, manifest


def aggregate_figures(output: Path) -> None:
    visual = output / "visualizations"
    visual.mkdir(exist_ok=True)
    curve = pd.read_csv(output / "metrics/global_alpha_curve.csv")
    decision = json.loads((output / "metrics/decision_matrix.json").read_text())
    hist = pd.read_csv(output / "metrics/best_alpha_histogram.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve.alpha, curve.mIoU*100, marker="o", label="global alpha")
    ax.axhline(decision["HQMR"]*100, color="black", ls="--", label="HQMR")
    ax.axhline(decision["SSHR"]*100, color="red", ls=":", label="SSHR")
    ax.set(xlabel="alpha", ylabel="mIoU (%)", title="Global alpha counterfactual")
    ax.legend(); fig.tight_layout(); fig.savefig(visual / "figure1_global_alpha_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(hist.best_alpha_A.astype(str), hist.support_area_fraction*100)
    ax.set(xlabel="component best alpha", ylabel="M1 support area (%)", title="Area-weighted alpha oracle distribution")
    fig.tight_layout(); fig.savefig(visual / "figure2_best_alpha_histogram.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["HQMR", "Oracle A", "Oracle B", "Oracle A+B"]
    values = [decision["HQMR"], decision["oracle_A"], decision["oracle_B"], decision["oracle_AB"]]
    ax.bar(labels, np.asarray(values)*100, color=["#777777", "#377eb8", "#4daf4a", "#984ea3"])
    ax.axhline(decision["SSHR"]*100, color="red", ls=":", label="SSHR")
    ax.set(ylabel="mIoU (%)", title="Architecture-realizable oracle ceilings"); ax.legend()
    fig.tight_layout(); fig.savefig(visual / "figure3_oracle_headroom.png", dpi=180); plt.close(fig)
    attribution = decision["pixel_attribution"]
    keys = ["A_only_fraction", "B_only_fraction", "both_individual_fraction",
            "joint_only_fraction", "AB_unrecoverable_fraction"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(["A only", "B only", "A&B", "Joint only", "Unrecoverable"],
           [attribution[key]*100 for key in keys])
    ax.set(ylabel="valid M1 pixels (%)", title="M1 recovery attribution")
    fig.tight_layout(); fig.savefig(visual / "figure4_recovery_attribution.png", dpi=180); plt.close(fig)


def render_case(target: Path, image: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                base: np.ndarray, chosen: np.ndarray, gate_pred: np.ndarray,
                joint: np.ndarray, cams: list[np.ndarray], gate_score: np.ndarray,
                baseline_gate: np.ndarray, forced_gate: np.ndarray, row: pd.Series,
                category: str, alpha: float) -> None:
    true = int(row.true_class)
    baseline_scores = np.asarray([cam[mask].mean() for cam in cams[3]])
    rival = int(np.argmax(np.where(np.arange(4) == true, -np.inf, baseline_scores)))
    alpha_index = int(np.flatnonzero(np.isclose(ALPHAS, alpha))[0])
    coordinates = np.argwhere(mask)
    y0, x0 = np.maximum(coordinates.min(0)-10, 0)
    y1, x1 = np.minimum(coordinates.max(0)+11, truth.shape)
    crop = np.s_[y0:y1, x0:x1]
    maps = [(cams[3][true], "Baseline true CAM", "viridis"),
            (cams[3][rival], "Baseline rival CAM", "viridis"),
            (cams[3][true]-cams[3][rival], "Baseline margin", "coolwarm"),
            (cams[alpha_index][true], f"alpha={alpha:g} true CAM", "viridis"),
            (cams[alpha_index][rival], f"alpha={alpha:g} rival CAM", "viridis"),
            (cams[alpha_index][true]-cams[alpha_index][rival], "Best-alpha margin", "coolwarm")]
    fig, ax = plt.subplots(4, 4, figsize=(14, 14), constrained_layout=True)
    fig.suptitle(f"{category} | {row.image_id} #{row.component_id} | C{true} vs C{rival} | alpha*={alpha:g}")
    panels = [(image, "Original", None), (truth, "GT", "tab10"),
              (base, "Baseline", "tab10"), (chosen, "Best-alpha prediction", "tab10"),
              (gate_pred, "Gate oracle", "tab10"), (joint, "Joint oracle", "tab10")]
    for cell, (data, title, cmap) in zip(ax.flat[:6], panels):
        cell.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=3 if cmap else None); cell.set_title(title)
    for cell, (data, title, cmap) in zip(ax.flat[6:12], maps):
        cropped = data[crop]; vmax = max(float(np.quantile(np.abs(cropped), .99)), 1e-6) if cmap == "coolwarm" else None
        cell.imshow(cropped, cmap=cmap, vmin=-vmax if vmax else None, vmax=vmax); cell.set_title(title)
    ax[3, 0].imshow(image); ax[3, 0].imshow(np.ma.masked_where(~mask, mask), alpha=.5, cmap="autumn"); ax[3, 0].set_title("Audited component")
    ax[3, 1].axis("off"); ax[3, 1].text(0, 1, f"baseline gate={baseline_gate.astype(int).tolist()}\nforced gate={forced_gate.astype(int).tolist()}\ngate score={np.round(gate_score,3).tolist()}\ntrue C{true}: {baseline_gate[true]:.0f} -> {forced_gate[true]:.0f}", va="top")
    base_margin = float((cams[3][true][mask]-cams[3][rival][mask]).mean())
    best_margin = float((cams[alpha_index][true][mask]-cams[alpha_index][rival][mask]).mean())
    ax[3, 2].axis("off"); ax[3, 2].text(0, 1, f"baseline margin={base_margin:+.4f}\nbest-alpha margin={best_margin:+.4f}\narea={int(row.area)}\npurity={float(row.purity):.3f}\nsource={row.cohort}", va="top")
    ax[3, 3].axis("off"); ax[3, 3].text(0, 1, "GT is used only to select among\nfrozen alpha-bank outputs or\nto enable an existing class gate.\nNo GT label is written directly.", va="top")
    for cell in ax.flat: cell.set_xticks([]); cell.set_yticks([])
    fig.savefig(target, dpi=120); plt.close(fig)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    aggregate_figures(args.output)
    selection, manifest = select_cases(args.output, args.ucrf_output, args.val_root)
    selection.to_csv(args.output / "metrics/visualization_selection.csv", index=False)
    ucrf = pd.read_parquet(args.ucrf_output / "metrics/component_event_table.parquet")
    mask_data = np.load(args.ucrf_output / "masks/whole.npz")
    packed, shape = mask_data["packed"], tuple(int(v) for v in mask_data["shape"])
    ids = np.load(args.output / "counterfactuals/alpha_100/predictions.npz")["image_ids"]
    id_lookup = {str(value): index for index, value in enumerate(ids)}
    base = np.load(args.output / "counterfactuals/alpha_100/predictions.npz")["predictions"]
    oracle_a = np.load(args.output / "arbitration_oracle/oracle_A_predictions/predictions.npz")["predictions"]
    oracle_b = np.load(args.output / "gate_oracle/oracle_B_predictions/predictions.npz")["predictions"]
    oracle_ab = np.load(args.output / "joint_oracle/oracle_AB_predictions/predictions.npz")["predictions"]
    model = load_model(args.checkpoint)
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    data_lookup = {Path(path).stem: index for index, path in enumerate(dataset.object)}
    rows_by_image = {}
    for image_id, subset in ucrf[ucrf.cohort == "M1"].groupby("image_id"):
        rows_by_image[str(image_id)] = subset
    visual = args.output / "visualizations"
    replay_errors = 0
    for ordinal, case in enumerate(selection.itertuples(), 1):
        row = ucrf.iloc[int(case.frame_index)]
        image_id = str(row.image_id); image_index = id_lookup[image_id]
        _, tensor = dataset[data_lookup[image_id]]
        image = np.asarray(Image.open(args.val_root / "img" / f"{image_id}.png").convert("RGB"))
        truth = np.asarray(Image.open(args.val_root / "mask" / f"{image_id}.png"))
        mask = unpack_mask(packed[int(case.frame_index)], shape)
        inferred = infer_alpha_bank(model, tensor[None].cuda(), truth.shape)
        baseline_gate = presence(inferred["gate_score"])
        forced_gate = baseline_gate.copy()
        for _, m1 in rows_by_image.get(image_id, pd.DataFrame()).iterrows():
            forced_gate[int(m1.true_class)] = 1
        replay = prediction_from_cam(inferred["cams"][3], baseline_gate, truth)
        replay_errors += int(not np.array_equal(replay, base[image_index]))
        category_dir = visual / case.category; category_dir.mkdir(exist_ok=True)
        chosen = oracle_a[image_index] if case.source == "M1" else prediction_from_cam(
            inferred["cams"][-1], baseline_gate, truth)
        render_case(category_dir / f"{ordinal:03d}_{image_id}_C{int(row.component_id)}.png",
            image, truth, mask, base[image_index], chosen, oracle_b[image_index],
            oracle_ab[image_index], inferred["cams"], inferred["gate_score"],
            baseline_gate, forced_gate, row, case.category, float(case.alpha))
        if ordinal % 10 == 0:
            print(json.dumps({"event": "visualization_progress", "cases": ordinal}), flush=True)
    validation = {"cases": len(selection), "replay_prediction_errors": replay_errors,
                  "selection": manifest, "parameter_updates": 0}
    (args.output / "metrics/visualization_manifest.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8")
    if replay_errors:
        raise AssertionError(f"Visualization replay mismatch: {replay_errors}")
    print(json.dumps({"event": "visualization_done", **validation}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--ucrf-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
