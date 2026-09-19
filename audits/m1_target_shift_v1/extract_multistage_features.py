"""Frozen TTA extraction of observable HQMR features and mask-logit stages.

The H4 reconstruction and query update are logits/query operations, not
spatial tissue-feature tensors. Their evidence is stored separately.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audits.m1_target_shift_v1.preflight import EXPECTED, digest
from network.cirv import STRUCTURE8, extract_regions
from network.hqmr_net import HQMRNet
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import (
    TTA, load_state, normalize_cam, prediction_from_cam, presence, resize_unflip,
)

STAGES = ("H5_pre", "H5_context", "H4_input", "K4")
REGIONS = ("whole", "core", "boundary", "ring", "intersection")


@torch.inference_mode()
def infer(model: HQMRNet, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    maps = {name: [] for name in STAGES}
    logits = {name: [] for name in ("H5_logits", "H4_recon_logits", "H3_final_logits")}
    class_maps = {name: [] for name in logits}
    mixtures, gates = [], []
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, hqmr_mode="full")
        decoded = output["stages"][2]["hqmr"]
        spatial = {
            "H5_pre": output["features"]["F5"][0],
            "H5_context": output["query_detail"]["context_feature"][0],
            "H4_input": output["pixel_detail"]["F4_context"][0],
            "K4": decoded["key4"][0],
        }
        for name, tensor in spatial.items():
            if cam_flip:
                tensor = torch.flip(tensor, dims=cam_flip)
            maps[name].append(tensor.detach().float().cpu())
        for name, key in (("H5_logits", "logits5"),
                          ("H4_recon_logits", "logits4"),
                          ("H3_final_logits", "logits3")):
            tensor = decoded[key][0]
            if cam_flip:
                tensor = torch.flip(tensor, dims=cam_flip)
            logits[name].append(tensor.detach().float().cpu())
            weights = decoded["weights"][0].detach().float().cpu()
            class_maps[name].append(torch.einsum("qc,qhw->chw", weights, tensor.float().cpu().sigmoid()))
        mixtures.append(resize_unflip(decoded["mixture"], original_hw, cam_flip).float().cpu())
        gates.append(output["deep_gate"].detach().float().cpu())
    scores = normalize_cam(torch.stack(mixtures).mean(0).numpy())
    gate = torch.stack(gates).mean(0).numpy()[0]
    label = presence(gate)
    prediction = prediction_from_cam(scores, label, np.empty(original_hw))
    return {"prediction": prediction,
            "maps": {name: torch.stack(value).mean(0) for name, value in maps.items()},
            "logits": {name: torch.stack(value).mean(0) for name, value in logits.items()},
            "class_maps": {name: torch.stack(value).mean(0) for name, value in class_maps.items()}}


def region_masks(mask: np.ndarray, truth: np.ndarray | None, true_class: int) -> dict[str, np.ndarray]:
    distance = ndimage.distance_transform_edt(mask)
    maximum = float(distance.max())
    normalized = distance / maximum if maximum else distance
    radius = math.sqrt(int(mask.sum()) / math.pi)
    ring_width = int(np.clip(round(.25 * radius), 2, 12))
    ring = ndimage.binary_dilation(mask, structure=STRUCTURE8, iterations=ring_width) & ~mask
    return {
        "whole": mask,
        "core": mask & (normalized >= .5),
        "boundary": mask & (normalized < .25),
        "ring": ring,
        "intersection": mask & (truth == true_class) if truth is not None else np.zeros_like(mask),
    }


@torch.inference_mode()
def pool(feature: torch.Tensor, masks: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    vectors, dispersions = {}, {}
    shape = feature.shape[-2:]
    feature = feature.float().cuda(non_blocking=True)
    flat_feature = F.normalize(feature.flatten(1).T, dim=1)
    for name, mask in masks.items():
        if not mask.any():
            vectors[name] = np.full(feature.shape[0], np.nan, np.float32)
            dispersions[name] = float("nan")
            continue
        values = torch.from_numpy(mask.astype(np.float32))[None, None].cuda(non_blocking=True)
        weight = F.interpolate(values, size=shape, mode="area")[0, 0]
        mass = weight.sum()
        if mass.item() <= 0:
            vectors[name] = np.full(feature.shape[0], np.nan, np.float32)
            dispersions[name] = float("nan")
            continue
        raw = (feature * weight).sum((1, 2)) / mass
        unit = F.normalize(raw, dim=0)
        vectors[name] = unit.cpu().numpy().astype(np.float32)
        cosine = flat_feature @ unit
        dispersions[name] = float((weight.flatten() * (1 - cosine)).sum() / mass)
    return vectors, dispersions


@torch.inference_mode()
def logit_evidence(logits: torch.Tensor, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    weight = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                           size=logits.shape[-2:], mode="area")[0, 0]
    if weight.sum() <= 0:
        return float("nan")
    # No class label is assigned to an individual HQMR query. The pooled
    # quantity is query-mask activation, not true-class evidence.
    probability = logits.float().sigmoid().amax(0)
    return float((weight * probability).sum() / weight.sum())


def class_evidence(class_map: torch.Tensor, mask: np.ndarray, cls: int) -> float:
    if not mask.any():
        return float("nan")
    weight = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                           size=class_map.shape[-2:], mode="area")[0, 0]
    if weight.sum() <= 0:
        return float("nan")
    return float((weight * class_map[cls]).sum() / weight.sum())


def fragment(mask: np.ndarray, desired: int, mode: str) -> np.ndarray:
    """Deterministic area-matched connected GT tissue fragment."""
    if mask.sum() < desired or desired < 1:
        return np.zeros_like(mask)
    distance = ndimage.distance_transform_edt(mask)
    if mode == "clean":
        anchor = np.unravel_index(int(np.argmax(distance)), mask.shape)
    else:
        boundary = mask & (distance <= 1)
        points = np.argwhere(boundary)
        if not len(points):
            return np.zeros_like(mask)
        anchor = tuple(points[len(points) // 2])
    yy, xx = np.indices(mask.shape)
    order = np.where(mask, (yy - anchor[0]) ** 2 + (xx - anchor[1]) ** 2, np.inf)
    chosen = np.argpartition(order.ravel(), desired - 1)[:desired]
    selected = np.zeros(mask.size, np.bool_)
    selected[chosen] = True
    selected = selected.reshape(mask.shape)
    labels, count = ndimage.label(selected, structure=STRUCTURE8)
    if count:
        sizes = np.bincount(labels.ravel())[1:]
        selected = labels == int(np.argmax(sizes) + 1)
    return selected


def save_rows(output: Path, prefix: str, rows: list[dict], embeds: dict,
              packed_masks: dict[str, list[np.ndarray]]) -> None:
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / f"{prefix}_cohorts.parquet", index=False, compression="zstd")
    (output / "features").mkdir(exist_ok=True)
    for stage in STAGES:
        payload = {kind: np.stack(embeds[stage][kind]).astype(np.float32) for kind in REGIONS}
        np.savez_compressed(output / "features" / f"{prefix}_{stage}.npz", **payload)
    if packed_masks:
        (output / "masks").mkdir(exist_ok=True)
        for kind, values in packed_masks.items():
            np.savez_compressed(output / "masks" / f"{prefix}_{kind}.npz",
                                packed=np.stack(values).astype(np.uint8), shape=np.array([224, 224]))


@torch.inference_mode()
def run_target(args, model: HQMRNet) -> None:
    gate = json.loads((args.output / "00_reproduction_gate.json").read_text(encoding="utf-8"))
    if not gate.get("pass"):
        raise AssertionError("Fresh reproduction gate must pass before target extraction")
    loader = DataLoader(Stage1_InferDataset(str(args.val_root / "img"), img_size=224),
                        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    rows: list[dict] = []
    embeds = {stage: {kind: [] for kind in REGIONS} for stage in STAGES}
    packed_masks = {kind: [] for kind in ("whole", "core", "boundary", "ring")}
    started = time.perf_counter()
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]
        original = Image.open(args.val_root / "img" / f"{image_id}.png")
        truth = np.asarray(Image.open(args.val_root / "mask" / f"{image_id}.png"))
        inference = infer(model, image.cuda(non_blocking=True), (original.height, original.width))
        for region in extract_regions(inference["prediction"]):
            mask = region["mask"]
            counts = np.bincount(truth[mask][truth[mask] < 4], minlength=4)
            if not counts.sum():
                continue
            true_class = int(np.argmax(counts))
            purity = float(counts[true_class] / region["area"])
            cls = int(region["class_id"])
            if counts[cls] == 0:
                cohort = "M1"
            elif counts[cls] / region["area"] >= .90:
                cohort = "TP_candidate"
            else:
                continue
            masks = region_masks(mask, truth, true_class)
            row = {"image_id": image_id, "component_id": region["component_id"],
                   "cohort": cohort, "predicted_class": cls, "true_class": true_class,
                   "area": region["area"], "purity": purity,
                   "core_valid": bool(masks["core"].any()),
                   "boundary_valid": bool(masks["boundary"].any()),
                   "ring_valid": bool(masks["ring"].any()),
                   "intersection_valid": bool(masks["intersection"].any())}
            for stage in STAGES:
                pooled, dispersion = pool(inference["maps"][stage], masks)
                for kind in REGIONS:
                    embeds[stage][kind].append(pooled[kind])
                    row[f"dispersion_{stage}_{kind}"] = dispersion[kind]
            for stage, tensor in inference["logits"].items():
                row[f"max_query_activation_{stage}"] = logit_evidence(tensor, mask)
                class_map = inference["class_maps"][stage]
                true_score = class_evidence(class_map, mask, true_class)
                pred_score = class_evidence(class_map, mask, cls)
                row[f"true_class_score_{stage}"] = true_score
                row[f"pred_class_score_{stage}"] = pred_score
                row[f"pred_minus_true_score_{stage}"] = pred_score - true_score
            for kind in packed_masks:
                packed_masks[kind].append(np.packbits(masks[kind].ravel()))
            rows.append(row)
        if index % 100 == 0:
            print(json.dumps({"event": "target_extract", "images": index,
                              "regions": len(rows), "elapsed_s": round(time.perf_counter()-started, 1)}), flush=True)
    m1 = [row for row in rows if row["cohort"] == "M1"]
    if len(m1) != EXPECTED["m1_components"] or sum(row["area"] for row in m1) != EXPECTED["m1_pixels"]:
        raise AssertionError("Target extraction diverges from frozen M1 cohort")
    save_rows(args.output, "target", rows, embeds, packed_masks)
    print(json.dumps({"event": "target_done", "M1": len(m1), "TP_candidates": len(rows)-len(m1)}), flush=True)


@torch.inference_mode()
def run_reference(args, model: HQMRNet) -> None:
    import pandas as pd
    target = pd.read_parquet(args.output / "target_cohorts.parquet")
    m1_area = target.loc[target.cohort == "M1", "area"].to_numpy()
    target_sizes = np.maximum(8, np.quantile(m1_area, [.1, .3, .5, .7, .9]).astype(int))
    dataset = Stage1_InferDataset(str(args.train_root), img_size=224)
    if len(dataset) != 23422:
        raise AssertionError("Expected 23422 training images")
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(dataset), size=min(args.reference_images, len(dataset)), replace=False))
    rows: list[dict] = []
    embeds = {stage: {kind: [] for kind in REGIONS} for stage in STAGES}
    started = time.perf_counter()
    for serial, image_index in enumerate(indices, 1):
        image_id, image = dataset[int(image_index)]
        truth = np.asarray(Image.open(args.train_gt_root / f"{image_id}.png"))
        original = Image.open(args.train_root / f"{image_id}.png")
        inference = infer(model, image[None].cuda(), (original.height, original.width))
        desired = int(target_sizes[(serial-1) % len(target_sizes)])
        for cls in range(4):
            labels, count = ndimage.label(truth == cls, structure=STRUCTURE8)
            if not count:
                continue
            sizes = np.bincount(labels.ravel())[1:]
            candidates = np.argsort(sizes)[-3:][::-1] + 1
            for component_id in candidates:
                tissue = labels == component_id
                if tissue.sum() < desired:
                    continue
                for cohort, mode in (("clean_GT", "clean"), ("boundary_fragment", "boundary")):
                    selected = fragment(tissue, desired, mode)
                    if selected.sum() < .8 * desired:
                        continue
                    masks = region_masks(selected, truth, cls)
                    row = {"image_id": image_id, "component_id": int(component_id),
                           "cohort": cohort, "predicted_class": cls, "true_class": cls,
                           "area": int(selected.sum()), "purity": 1.0,
                           "core_valid": bool(masks["core"].any()),
                           "boundary_valid": bool(masks["boundary"].any()),
                           "ring_valid": bool(masks["ring"].any()),
                           "intersection_valid": True}
                    for stage in STAGES:
                        pooled, dispersion = pool(inference["maps"][stage], masks)
                        for kind in REGIONS:
                            embeds[stage][kind].append(pooled[kind])
                            row[f"dispersion_{stage}_{kind}"] = dispersion[kind]
                    rows.append(row)
        if serial % 100 == 0:
            print(json.dumps({"event": "reference_extract", "images": serial,
                              "regions": len(rows), "elapsed_s": round(time.perf_counter()-started, 1)}), flush=True)
    if not rows or any(not any(row["cohort"] == "clean_GT" and row["true_class"] == cls for row in rows)
                       for cls in range(4)):
        raise AssertionError("Missing clean GT reference for at least one class")
    save_rows(args.output, "reference", rows, embeds, {})
    print(json.dumps({"event": "reference_done", "images": len(indices), "regions": len(rows)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("target", "reference"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--train-gt-root", type=Path)
    parser.add_argument("--reference-images", type=int, default=1500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.checkpoint) != EXPECTED["checkpoint_sha256"]:
        raise AssertionError("Checkpoint SHA256 mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    model = HQMRNet().cuda()
    model.load_state_dict(load_state(args.checkpoint), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.mode == "target":
        if args.val_root is None:
            parser.error("target mode requires --val-root")
        run_target(args, model)
    else:
        if args.train_root is None or args.train_gt_root is None:
            parser.error("reference mode requires --train-root and --train-gt-root")
        run_reference(args, model)


if __name__ == "__main__":
    main()
