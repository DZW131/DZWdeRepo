"""Re-run frozen inference for 6x20 region cases and preserve raw stage tensors."""
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
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audits.ucrf_v1.extract import infer_chain
from audits.ucrf_v1.gate import load_model
from tool.GenDataset import Stage1_InferDataset


def select_cases(frame: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    m1 = frame.cohort == "M1"
    predicates = {
        "deep_already_wrong": m1 & (frame.margin_logits5 < 0) & (frame.margin_logits4 < 0),
        "correct_to_wrong": m1 & (frame.flip_event == "F2_correct_to_wrong"),
        "wrong_to_correct": m1 & (frame.flip_event == "F3_wrong_to_correct"),
        "corrective_suppressed": m1 & (frame.margin_logits5 < 0) &
                                (frame.direct4_effect_margin > 0) & (frame.margin_logits4 < 0),
        "stage3_amplification": m1 & (frame.stage3_event == "Q_AMP"),
    }
    scores = {
        "deep_already_wrong": -frame.margin_logits5,
        "correct_to_wrong": -frame.margin_logits4,
        "wrong_to_correct": frame.margin_logits4,
        "corrective_suppressed": frame.direct4_effect_margin,
        "stage3_amplification": frame.margin_upsampled4 - frame.margin_logits3,
    }
    selected, used, manifest = [], set(), {}
    for category, criterion in predicates.items():
        ranked = frame.loc[criterion].assign(rank_score=scores[category][criterion]).sort_values(
            ["area", "rank_score"], ascending=False)
        primary = [int(i) for i in ranked.index if i not in used][:20]
        if len(primary) < 20:
            fallback = frame.loc[m1].sort_values("area", ascending=False)
            primary.extend(int(i) for i in fallback.index if i not in used and i not in primary)
            primary = primary[:20]
        used.update(primary)
        manifest[category] = {"criterion_available": int(criterion.sum()),
                              "selected_meeting_criterion": int(sum(bool(criterion.iloc[i]) for i in primary)),
                              "selected": len(primary)}
        selected.extend({"category": category, "frame_index": i} for i in primary)
    # The control category is traced to a GT-class-matched M1 counterpart.
    selected_m1 = [row["frame_index"] for row in selected]
    controls = matches[matches.m1_index.isin(selected_m1)].copy()
    controls["control_area"] = controls.tp_index.map(frame.area)
    controls = controls.sort_values("control_area", ascending=False)
    tp_used = set()
    tp_rows = []
    for row in controls.itertuples():
        if int(row.tp_index) not in tp_used:
            tp_used.add(int(row.tp_index))
            tp_rows.append({"category": "matched_tp_control", "frame_index": int(row.tp_index),
                            "matched_m1_index": int(row.m1_index)})
        if len(tp_rows) == 20:
            break
    if len(tp_rows) < 20:
        for row in matches.sort_values("tp_index").itertuples():
            if int(row.tp_index) not in tp_used:
                tp_used.add(int(row.tp_index))
                tp_rows.append({"category": "matched_tp_control", "frame_index": int(row.tp_index),
                                "matched_m1_index": int(row.m1_index)})
            if len(tp_rows) == 20:
                break
    if len(tp_rows) != 20 or len(selected) != 100:
        raise AssertionError("Could not select exactly 120 cases")
    selected.extend(tp_rows)
    manifest["matched_tp_control"] = {"criterion_available": int(matches.tp_index.nunique()),
                                       "selected_meeting_criterion": 20, "selected": 20}
    return pd.DataFrame(selected), manifest


def as_native_map(item: torch.Tensor, hw: tuple[int, int]) -> np.ndarray:
    return F.interpolate(item[None].float(), size=hw, mode="bilinear",
                         align_corners=False)[0].numpy()


def render(image: np.ndarray, truth: np.ndarray, pred: np.ndarray,
           mask: np.ndarray, maps: dict[str, np.ndarray], row: pd.Series,
           category: str, target: Path) -> None:
    true, rival = int(row.true_class), int(row.predicted_class)
    h, w = truth.shape
    y0, y1 = max(0, int(row.bbox_ymin)-12), min(h, int(row.bbox_ymax)+13)
    x0, x1 = max(0, int(row.bbox_xmin)-12), min(w, int(row.bbox_xmax)+13)
    box = np.s_[y0:y1, x0:x1]
    fig, ax = plt.subplots(4, 4, figsize=(14, 14), constrained_layout=True)
    fig.suptitle(f"{category} | {row.image_id} #{row.component_id} | GT C{true} / Pred C{rival} | area {row.area}")
    ax[0, 0].imshow(image); ax[0, 0].set_title("Original")
    ax[0, 1].imshow(truth, vmin=0, vmax=3, cmap="tab10"); ax[0, 1].set_title("GT")
    ax[0, 2].imshow(pred, vmin=0, vmax=3, cmap="tab10"); ax[0, 2].set_title("HQMR prediction")
    ax[0, 3].imshow(image); ax[0, 3].imshow(np.ma.masked_where(~mask, mask), alpha=.50,
                                              cmap="autumn"); ax[0, 3].set_title("Selected region")
    specifications = [
        (1, 0, "logits5", true, "logits5 true", "viridis"),
        (1, 1, "logits5", rival, "logits5 rival", "viridis"),
        (1, 2, "logits5", None, "logits5 true-rival", "coolwarm"),
        (1, 3, "direct4_standalone", true, "direct4 proxy true", "viridis"),
        (2, 0, "direct4_standalone", rival, "direct4 proxy rival", "viridis"),
        (2, 1, "direct4_standalone", None, "direct4 proxy margin", "coolwarm"),
        (2, 2, "logits4", None, "logits4 margin", "coolwarm"),
        (2, 3, "logits3", None, "logits3 margin", "coolwarm"),
        (3, 0, "upsampled5", None, "upsampled5 margin", "coolwarm"),
        (3, 1, "logits3_no_q4", None, "logits3 no-query4 CF", "coolwarm"),
        (3, 2, "final_cam", None, "final CAM margin", "coolwarm"),
    ]
    for i, j, stage, cls, title, cmap in specifications:
        data = maps[stage][cls] if cls is not None else maps[stage][true]-maps[stage][rival]
        data = data[box]
        vmax = max(float(np.quantile(np.abs(data), .99)), 1e-6) if cls is None else None
        ax[i, j].imshow(data, cmap=cmap, vmin=-vmax if cls is None else None,
                        vmax=vmax); ax[i, j].set_title(title, fontsize=10)
    ax[3, 3].axis("off")
    ax[3, 3].text(0, 1, "\n".join([
        f"purity={row.purity:.3f}  confidence5={row.confidence_logits5:.3f}",
        f"M5={row.margin_logits5:+.4f}  U5={row.margin_upsampled5:+.4f}",
        f"direct4 effect={row.direct4_effect_margin:+.4f}",
        f"M4={row.margin_logits4:+.4f}  M3={row.margin_logits3:+.4f}",
        f"query4-isolated={row.query4_isolated_effect_margin:+.4f}",
        f"E5={row.rival_pixel_fraction_logits5:.3f}",
        f"E4={row.rival_pixel_fraction_logits4:.3f}",
        f"E3={row.rival_pixel_fraction_logits3:.3f}",
        f"{row.flip_event} / {row.stage3_event}",
        f"true gate present={row.true_label_present}",
        "direct4 proxy != additive class logits",
    ]), va="top", fontsize=10)
    for cell in ax.flat:
        cell.set_xticks([]); cell.set_yticks([])
    fig.savefig(target, dpi=120)
    plt.close(fig)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    output = args.output
    gate = json.loads((output / "00_reproduction_gate.json").read_text())
    if not gate.get("pass"):
        raise AssertionError("Reproduction gate required")
    frame = pd.read_parquet(output / "metrics/component_event_table.parquet")
    matches = pd.read_csv(output / "metrics/matched_tp_pairs.csv")
    selected, selection_manifest = select_cases(frame, matches)
    selected.to_csv(output / "metrics/visualization_selection.csv", index=False)
    mask_archive = np.load(output / "masks/whole.npz")
    packed, mask_shape = mask_archive["packed"], tuple(int(v) for v in mask_archive["shape"])
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    image_lookup = {Path(path).stem: index for index, path in enumerate(dataset.object)}
    model = load_model(args.checkpoint)
    (output / "visualizations").mkdir(exist_ok=True)
    for stage in ("logits5", "direct4", "logits4", "query4", "logits3"):
        (output / "tensors" / stage).mkdir(parents=True, exist_ok=True)
    cache = {}
    validation = {"cases": len(selected), "prediction_disagreements": 0,
                  "unique_images": int(selected.frame_index.map(lambda i: frame.iloc[i].image_id).nunique()),
                  "selection": selection_manifest, "parameter_updates": 0}
    for ordinal, case in enumerate(selected.itertuples(), 1):
        row = frame.iloc[int(case.frame_index)]
        image_id = str(row.image_id)
        if image_id != cache.get("image_id"):
            _, image_tensor = dataset[image_lookup[image_id]]
            image = np.asarray(Image.open(args.val_root / "img" / f"{image_id}.png").convert("RGB"))
            truth = np.asarray(Image.open(args.val_root / "mask" / f"{image_id}.png"))
            inference = infer_chain(model, image_tensor[None].cuda(), truth.shape, dump_tensors=True)
            maps = {name: as_native_map(value, truth.shape) for name, value in
                    inference["class_maps"].items()}
            for stage in ("logits5", "direct4", "logits4", "query4", "logits3"):
                arrays = {key: value for key, value in inference["raw_tensors"].items()
                          if key.endswith(f"_{stage}")}
                destination = output / "tensors" / stage / f"{image_id}.npz"
                if not destination.exists():
                    np.savez_compressed(destination, **arrays)
            cache = {"image_id": image_id, "image": image, "truth": truth,
                     "inference": inference, "maps": maps}
        mask = np.unpackbits(packed[int(case.frame_index)])[:np.prod(mask_shape)].reshape(mask_shape).astype(bool)
        pred = cache["inference"]["prediction"]
        if not np.all(pred[mask] == int(row.predicted_class)):
            validation["prediction_disagreements"] += 1
        category_dir = output / "visualizations" / case.category
        category_dir.mkdir(exist_ok=True)
        target = category_dir / f"{ordinal:03d}_{image_id}_C{row.component_id}.png"
        render(cache["image"], cache["truth"], pred, mask, cache["maps"], row,
               case.category, target)
        if ordinal % 10 == 0:
            print(json.dumps({"event": "visualization_progress", "cases": ordinal}), flush=True)
    if validation["prediction_disagreements"]:
        raise AssertionError(f"Visualization replay disagreed in {validation['prediction_disagreements']} cases")
    (output / "metrics/visualization_manifest.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps({"event": "visualization_done", **validation}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
