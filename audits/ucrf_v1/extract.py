"""TTA-faithful frozen HQMR class-responsibility extraction (no updates)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audits.ucrf_v1.gate import EXPECTED_M1, EXPECTED_PIXELS, load_model
from network.cirv import extract_regions
from network.hqmr import direct_affinity
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import (
    TTA, normalize_cam, prediction_from_cam, presence, resize_unflip,
)

STAGES = ("logits5", "upsampled5", "direct4_standalone", "logits4",
          "upsampled4", "logits3_no_q4", "logits3", "final_cam")
PRIMARY = ("logits5", "upsampled5", "logits4", "upsampled4", "logits3", "final_cam")
EPS = 1e-8


def class_map(logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Actual frozen GCQM class-mixture rule, with Stage3 weights held fixed."""
    return torch.einsum("qc,qhw->chw", weights.float(), logits.float().sigmoid()).clamp(0, 1)


@torch.inference_mode()
def infer_chain(model, image: torch.Tensor, original_hw: tuple[int, int],
                dump_tensors: bool = False) -> dict:
    views = {name: [] for name in STAGES if name != "final_cam"}
    mixture_views, gates = [], []
    raw_tensors = {}
    max_formula_error = {"logits4": 0.0, "logits3": 0.0}
    dummy = torch.ones((1, 4), device=image.device)
    for view_index, (input_flip, cam_flip) in enumerate(TTA):
        captured = []
        handle = model.hqmr.scale3.key.register_forward_hook(
            lambda _module, _inputs, value: captured.append(value.detach()))
        try:
            value = torch.flip(image, dims=input_flip) if input_flip else image
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(value, dummy, step=29275, hqmr_mode="full")
                item = output["stages"][2]["hqmr"]
                if len(captured) != 1:
                    raise AssertionError("Expected exactly one Stage3 key3 hook call")
                l5, d4, l4, d3, l3 = (item[name] for name in
                    ("logits5", "direct4", "logits4", "direct3", "logits3"))
                u5 = F.interpolate(l5, size=d4.shape[-2:], mode="bilinear", align_corners=False)
                u4 = F.interpolate(l4, size=d3.shape[-2:], mode="bilinear", align_corners=False)
                d3_q5 = direct_affinity(item["query5"], captured[0])
                l3_no_q4 = u4 + d3_q5
            max_formula_error["logits4"] = max(max_formula_error["logits4"],
                float(((u5+d4).float()-l4.float()).abs().max()))
            max_formula_error["logits3"] = max(max_formula_error["logits3"],
                float(((u4+d3).float()-l3.float()).abs().max()))
            weights = item["weights"][0].detach().float()
            values = {"logits5": l5[0], "upsampled5": u5[0],
                      "direct4_standalone": d4[0], "logits4": l4[0],
                      "upsampled4": u4[0], "logits3_no_q4": l3_no_q4[0],
                      "logits3": l3[0]}
            for name, logits in values.items():
                mixture = class_map(logits, weights)
                if cam_flip:
                    mixture = torch.flip(mixture, dims=cam_flip)
                views[name].append(mixture.detach().float().cpu())
            mixture_views.append(resize_unflip(item["mixture"], original_hw, cam_flip).float().cpu())
            gates.append(output["deep_gate"].detach().float().cpu())
            if dump_tensors:
                raw_tensors[f"view{view_index}_logits5"] = l5[0].detach().float().cpu().numpy().astype(np.float16)
                raw_tensors[f"view{view_index}_direct4"] = d4[0].detach().float().cpu().numpy().astype(np.float16)
                raw_tensors[f"view{view_index}_logits4"] = l4[0].detach().float().cpu().numpy().astype(np.float16)
                raw_tensors[f"view{view_index}_query4"] = item["query4"][0].detach().float().cpu().numpy().astype(np.float16)
                raw_tensors[f"view{view_index}_logits3"] = l3[0].detach().float().cpu().numpy().astype(np.float16)
        finally:
            handle.remove()
    if any(error > 1e-6 for error in max_formula_error.values()):
        raise AssertionError(f"Frozen forward formula failed: {max_formula_error}")
    mean_mixture = torch.stack(mixture_views).mean(0).numpy()
    final_cam = normalize_cam(mean_mixture)
    gate = torch.stack(gates).mean(0).numpy()[0]
    label = presence(gate)
    prediction = prediction_from_cam(final_cam, label, np.empty(original_hw))
    class_maps = {name: torch.stack(values).mean(0) for name, values in views.items()}
    class_maps["final_cam"] = torch.from_numpy(final_cam)
    return {"prediction": prediction, "class_maps": class_maps,
            "mean_mixture": mean_mixture, "presence": label, "deep_gate": gate,
            "formula_max_abs_error": max_formula_error, "raw_tensors": raw_tensors}


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    values = np.exp(shifted)
    return values / values.sum()


def score_map(mapping: torch.Tensor, mask: np.ndarray,
              original_resized: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                           size=mapping.shape[-2:], mode="area")[0, 0]
    mass = float(weight.sum())
    if mass <= EPS:
        nan4 = np.full(4, np.nan, np.float32)
        return nan4, nan4.copy(), nan4.copy()
    scores = ((mapping.float() * weight).sum((1, 2)) / mass).numpy().astype(np.float32)
    if original_resized is None:
        original_resized = F.interpolate(mapping[None].float(), size=mask.shape,
                                         mode="bilinear", align_corners=False)[0]
    mean_scores = original_resized[:, mask].mean(1).numpy().astype(np.float32)
    return scores, mean_scores, weight.numpy()


def projected_region(mapping: torch.Tensor, mask: np.ndarray, true_cls: int,
                     rival_cls: int, resized: torch.Tensor | None = None) -> dict:
    scores, means, weight = score_map(mapping, mask, resized)
    probabilities = _softmax(scores)
    order = np.argsort(scores)
    dynamic = int(np.argmax(np.where(np.arange(4) == true_cls, -np.inf, scores)))
    raw_margin = float(scores[true_cls]-scores[rival_cls])
    norm_margin = float(probabilities[true_cls]-probabilities[rival_cls])
    pixel_margin = (mapping[true_cls]-mapping[rival_cls]).float().numpy()
    mass = max(float(weight.sum()), EPS)
    return {"scores": scores, "means": means, "raw_margin": raw_margin,
            "mean_margin": float(means[true_cls]-means[rival_cls]),
            "normalized_margin": norm_margin, "dynamic_rival": dynamic,
            "top_class": int(order[-1]),
            "confidence": float(probabilities[order[-1]]-probabilities[order[-2]]),
            "true_pixel_fraction": float((weight * (pixel_margin > 0)).sum()/mass),
            "rival_pixel_fraction": float((weight * (pixel_margin < 0)).sum()/mass),
            "tie_pixel_fraction": float((weight * (pixel_margin == 0)).sum()/mass)}


def region_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = ndimage.distance_transform_edt(mask)
    if distance.max() == 0:
        return np.zeros_like(mask), np.zeros_like(mask)
    q = distance / distance.max()
    return mask & (q >= .5), mask & (q < .25)


def row_for_region(inference: dict, image_id: str, region: dict, truth: np.ndarray,
                   cohort: str, true_cls: int, purity: float,
                   resized_maps: dict[str, torch.Tensor]) -> dict:
    mask = region["mask"]
    pred_cls = int(region["class_id"])
    core, boundary = region_masks(mask)
    coordinates = np.argwhere(mask)
    row = {"image_id": image_id, "component_id": int(region["component_id"]),
           "cohort": cohort, "predicted_class": pred_cls, "true_class": true_cls,
           "area": int(region["area"]), "purity": float(purity),
           "bbox_ymin": int(coordinates[:, 0].min()), "bbox_ymax": int(coordinates[:, 0].max()),
           "bbox_xmin": int(coordinates[:, 1].min()), "bbox_xmax": int(coordinates[:, 1].max()),
           "core_valid": bool(core.any()), "boundary_valid": bool(boundary.any()),
           "true_label_present": bool(inference["presence"][true_cls] > 0),
           "pred_label_present": bool(inference["presence"][pred_cls] > 0),
           "deep_gate_true": float(inference["deep_gate"][true_cls]),
           "deep_gate_pred": float(inference["deep_gate"][pred_cls])}
    maps = inference["class_maps"]
    for name, mapping in maps.items():
        resized = resized_maps[name]
        values = projected_region(mapping, mask, true_cls, pred_cls, resized)
        row[f"margin_{name}"] = values["raw_margin"]
        row[f"mean_margin_{name}"] = values["mean_margin"]
        row[f"normalized_margin_{name}"] = values["normalized_margin"]
        row[f"dynamic_rival_{name}"] = values["dynamic_rival"]
        row[f"top_class_{name}"] = values["top_class"]
        row[f"confidence_{name}"] = values["confidence"]
        row[f"true_pixel_fraction_{name}"] = values["true_pixel_fraction"]
        row[f"rival_pixel_fraction_{name}"] = values["rival_pixel_fraction"]
        row[f"tie_pixel_fraction_{name}"] = values["tie_pixel_fraction"]
        for cls in range(4):
            row[f"score_{name}_C{cls}"] = float(values["scores"][cls])
            row[f"mean_score_{name}_C{cls}"] = float(values["means"][cls])
        if name in ("logits5", "logits4", "logits3"):
            for part, selected in (("core", core), ("boundary", boundary)):
                if selected.any():
                    sub = projected_region(mapping, selected, true_cls, pred_cls, resized)
                    row[f"margin_{name}_{part}"] = sub["raw_margin"]
                    row[f"normalized_margin_{name}_{part}"] = sub["normalized_margin"]
                else:
                    row[f"margin_{name}_{part}"] = float("nan")
                    row[f"normalized_margin_{name}_{part}"] = float("nan")
    row["direct4_effect_margin"] = row["margin_logits4"] - row["margin_upsampled5"]
    row["direct4_effect_norm_margin"] = row["normalized_margin_logits4"] - row["normalized_margin_upsampled5"]
    row["direct4_contribution_ratio"] = abs(row["direct4_effect_margin"]) / (
        abs(row["margin_upsampled5"]) + abs(row["direct4_effect_margin"]) + EPS)
    row["stage3_total_effect_margin"] = row["margin_logits3"] - row["margin_upsampled4"]
    row["direct3_fixed_q5_effect_margin"] = row["margin_logits3_no_q4"] - row["margin_upsampled4"]
    row["query4_isolated_effect_margin"] = row["margin_logits3"] - row["margin_logits3_no_q4"]
    row["stage3_effect_identity_error"] = abs(row["stage3_total_effect_margin"] -
        row["direct3_fixed_q5_effect_margin"] - row["query4_isolated_effect_margin"])
    row["calibration_flip"] = bool(row["margin_logits3"] > 0 and row["margin_final_cam"] < 0)
    row["packed_mask"] = np.packbits(mask.ravel())
    return row


@torch.inference_mode()
def run(args) -> None:
    gate = json.loads((args.output / "00_reproduction_gate.json").read_text())
    if not gate.get("pass"):
        raise AssertionError("Fresh UCRF reproduction gate required")
    model = load_model(args.checkpoint)
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    rows, all_confidence = [], []
    max_formula_error = {"logits4": 0.0, "logits3": 0.0}
    started = time.perf_counter()
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]
        original = Image.open(args.val_root / "img" / f"{image_id}.png")
        truth = np.asarray(Image.open(args.val_root / "mask" / f"{image_id}.png"))
        inference = infer_chain(model, image.cuda(non_blocking=True), (original.height, original.width))
        resized_maps = {name: F.interpolate(mapping[None].float(), size=truth.shape,
                                            mode="bilinear", align_corners=False)[0]
                        for name, mapping in inference["class_maps"].items()}
        for key in max_formula_error:
            max_formula_error[key] = max(max_formula_error[key], inference["formula_max_abs_error"][key])
        for region in extract_regions(inference["prediction"]):
            mask = region["mask"]
            # Validation-wide confidence tertiles are set before using GT.
            stage5_score, _, _ = score_map(inference["class_maps"]["logits5"], mask,
                                           resized_maps["logits5"])
            probabilities = _softmax(stage5_score)
            two = np.sort(probabilities)[-2:]
            all_confidence.append(float(two[-1]-two[-2]))
            valid = truth[mask]
            valid = valid[valid < 4]
            if not len(valid):
                continue
            counts = np.bincount(valid, minlength=4)
            true_cls = int(np.argmax(counts))
            purity = float(counts[true_cls] / region["area"])
            pred_cls = int(region["class_id"])
            if counts[pred_cls] == 0:
                cohort = "M1"
            elif counts[pred_cls] / region["area"] >= .70:
                cohort = "TP_candidate"
                true_cls = pred_cls
                purity = float(counts[pred_cls] / region["area"])
            else:
                continue
            rows.append(row_for_region(inference, image_id, region, truth,
                                       cohort, true_cls, purity, resized_maps))
        if index % 100 == 0:
            print(json.dumps({"event": "extract_progress", "images": index,
                              "regions": len(rows), "elapsed_s": round(time.perf_counter()-started, 1)}), flush=True)
    frame = pd.DataFrame(rows)
    m1 = frame[frame.cohort == "M1"]
    if len(m1) != EXPECTED_M1 or int(m1.area.sum()) != EXPECTED_PIXELS:
        raise AssertionError("M1 cohort differs from frozen 4440/8750254")
    prior = pd.read_parquet(args.prior_target)
    old_keys = set(zip(prior.loc[prior.cohort == "M1", "image_id"],
                       prior.loc[prior.cohort == "M1", "component_id"],
                       prior.loc[prior.cohort == "M1", "predicted_class"],
                       prior.loc[prior.cohort == "M1", "area"]))
    new_keys = set(zip(m1.image_id, m1.component_id, m1.predicted_class, m1.area))
    if old_keys != new_keys:
        raise AssertionError("Frozen M1 component identity differs from previous audit")
    packed = np.stack(frame.pop("packed_mask").to_list()).astype(np.uint8)
    (args.output / "masks").mkdir(exist_ok=True)
    np.savez_compressed(args.output / "masks/whole.npz", packed=packed,
                        shape=np.array([224, 224], np.int16))
    (args.output / "metrics").mkdir(exist_ok=True)
    frame.to_parquet(args.output / "metrics/component_stage_margins.parquet", index=False, compression="zstd")
    frame.to_csv(args.output / "metrics/component_stage_margins.csv", index=False)
    confidence_edges = np.quantile(np.asarray(all_confidence), [.33, .66])
    (args.output / "metrics/confidence_tertiles.json").write_text(json.dumps({
        "source": "All validation predicted components before GT cohort selection",
        "thresholds": confidence_edges.tolist(), "population": len(all_confidence),
        "seed": 42}, indent=2), encoding="utf-8")
    (args.output / "metrics/extraction_manifest.json").write_text(json.dumps({
        "M1": len(m1), "M1_pixels": int(m1.area.sum()),
        "TP_candidates": int((frame.cohort == "TP_candidate").sum()),
        "total_regions": len(frame), "validation_components_for_confidence": len(all_confidence),
        "formula_max_abs_error": max_formula_error,
        "frozen_component_identity_match": True, "parameter_updates": 0,
        "seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    print(json.dumps({"event": "extract_done", "M1": len(m1),
                      "TP": int((frame.cohort == "TP_candidate").sum()),
                      "formula_error": max_formula_error}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--prior-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
