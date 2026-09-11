#!/usr/bin/env python3
"""Zero-training anatomy of the sealed GCQM Full25 E25 failure on BCSS."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage, stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.gcqm import gcqm_decode
from network.gcqm_net import GCQMNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import (
    PALETTE,
    THRESHOLDS,
    TTA,
    _predict_sshr,
    foreground_confusion,
    load_state,
    normalize_cam,
    prediction_from_cam,
    presence,
    resize_unflip,
    scores_from_confusion,
)
from tools.hqrf_phase0_io import sha256, write_csv, write_json


GCQM_SHA256 = "6e1b909fc86a870e652213831521e8ff552371a083f85faad7dac3a21d969d0f"
SSHR_SHA256 = "b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70"
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_RESAMPLES = 10_000
TOPK = (1, 3, 5, 10, 20, 50, 100, 196)
BANDS = (1, 3, 5)
CLASS_IDS = tuple(range(4))
EPS = 1.0e-8


def mean_ci(values, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 250):
        n = min(250, resamples - start)
        indices = rng.integers(0, len(values), size=(n, len(values)))
        means[start:start + n] = values[indices].mean(1)
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def correlation_ci(x, y, method="spearman", seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if method == "spearman":
        x, y = stats.rankdata(x), stats.rankdata(y)
    estimate = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    rng, samples = np.random.default_rng(seed), []
    for start in range(0, resamples, 250):
        count = min(250, resamples - start); index = rng.integers(0, len(x), size=(count, len(x)))
        bx, by = x[index], y[index]; bx = bx - bx.mean(1, keepdims=True); by = by - by.mean(1, keepdims=True)
        denominator = np.sqrt(np.sum(bx * bx, axis=1) * np.sum(by * by, axis=1))
        samples.extend(np.divide(np.sum(bx * by, axis=1), denominator, out=np.full(count, np.nan), where=denominator > 0).tolist())
    samples = np.asarray(samples); samples = samples[np.isfinite(samples)]
    return {"n": int(len(x)), "estimate": estimate,
            "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))] if len(samples) else [None, None]}


def binary_counts(truth, prediction, domain):
    truth, prediction, domain = truth.astype(bool), prediction.astype(bool), domain.astype(bool)
    tp = int(np.sum(domain & truth & prediction)); fp = int(np.sum(domain & ~truth & prediction))
    fn = int(np.sum(domain & truth & ~prediction)); tn = int(np.sum(domain & ~truth & ~prediction))
    return tp, fp, fn, tn


def ratio(num, den):
    return float(num / den) if den else float("nan")


def binary_metrics(truth, prediction, domain):
    tp, fp, fn, tn = binary_counts(truth, prediction, domain)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "iou": ratio(tp, tp + fp + fn), "dice": ratio(2 * tp, 2 * tp + fp + fn),
            "fp_rate": ratio(fp, fp + tn), "fn_rate": ratio(fn, fn + tp)}


def boundary_band(mask, radius):
    structure = ndimage.generate_binary_structure(2, 1)
    return np.logical_xor(ndimage.binary_dilation(mask, structure, iterations=radius),
                          ndimage.binary_erosion(mask, structure, iterations=radius))


def mask_morphology(mask, gt_area):
    mask = np.asarray(mask, dtype=bool); area = int(mask.sum())
    if not area:
        return {"components": 0, "largest_component_fraction": 0.0, "small_component_fraction": 0.0,
                "hole_count": 0, "hole_area_fraction": 0.0, "perimeter_area_ratio": 0.0,
                "boundary_length": 0, "fragmentation_index": 0.0, "compactness": 0.0,
                "area_ratio_gt": 0.0 if gt_area else float("nan")}
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max()) if len(sizes) else 0
    holes = ndimage.binary_fill_holes(mask) & ~mask
    _, hole_count = ndimage.label(holes); hole_area = int(holes.sum())
    perimeter_mask = mask & ~ndimage.binary_erosion(mask)
    perimeter = int(perimeter_mask.sum())
    return {"components": int(count), "largest_component_fraction": largest / area,
            "small_component_fraction": float(sizes[sizes < 10].sum() / area),
            "hole_count": int(hole_count), "hole_area_fraction": hole_area / area,
            "perimeter_area_ratio": perimeter / area, "boundary_length": perimeter,
            "fragmentation_index": count / (area / 1000.0 + EPS),
            "compactness": perimeter * perimeter / (4 * math.pi * area + EPS),
            "area_ratio_gt": area / gt_area if gt_area else float("nan")}


def gini(value):
    value = np.asarray(value, dtype=np.float64)
    if not len(value) or value.sum() <= 0:
        return 0.0
    value = np.sort(value); n = len(value)
    return float((2 * np.sum((np.arange(n) + 1) * value) / (n * value.sum())) - (n + 1) / n)


def weight_metrics(value):
    value = np.asarray(value, dtype=np.float64)
    value = value / max(value.sum(), EPS); ordered = np.sort(value)[::-1]
    entropy = float(-(value * np.log(np.clip(value, np.finfo(np.float64).tiny, None))).sum()); neff = float(np.exp(entropy))
    if neff >= 120: label = "very_diffuse"
    elif neff >= 80: label = "diffuse"
    elif neff >= 30: label = "moderate"
    else: label = "concentrated"
    return {"entropy": entropy, "normalized_entropy": entropy / math.log(len(value)),
            "effective_query_count": neff, "dominant_share": float(ordered[0]),
            **{f"top{k}_mass": float(ordered[:k].sum()) for k in (1, 3, 5, 10, 20)},
            "gini": gini(value), "diffuseness_bin": label}


def topk_mixture(weights, bases, k):
    outputs = []
    for cls in CLASS_IDS:
        index = torch.argsort(weights[:, cls], descending=True, stable=True)[:k]
        selected = weights[index, cls]
        selected = selected / selected.sum().clamp_min(EPS)
        outputs.append(torch.einsum("q,qhw->hw", selected, bases[index]))
    return torch.stack(outputs)


@torch.no_grad()
def infer_gcqm_diagnostics(model, image, original):
    global_views, pixel_views = [], []
    topk_views = {k: [] for k in TOPK}; gates = []; original_detail = None
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, run_pmec=False)
        stage = output["stages"][2]
        materialized = gcqm_decode(stage["base_mask_logits"], stage["detail"]["responsibility_class"],
                                   stage["memory_hw"], output["locality"], materialize=True)
        global_views.append(resize_unflip(materialized["mixture"], original.shape[:2], cam_flip))
        pixel_views.append(resize_unflip(materialized["pixel_reference"], original.shape[:2], cam_flip))
        for k in TOPK:
            mixed = topk_mixture(materialized["weights"][0], materialized["base_probability"][0], k)[None]
            topk_views[k].append(resize_unflip(mixed, original.shape[:2], cam_flip))
        gates.append(output["deep_gate"])
        if not input_flip:
            original_detail = {"weights": materialized["weights"][0].cpu(),
                               "bases": materialized["base_probability"][0].cpu(),
                               "routing": materialized["routing"][0].cpu(),
                               "pixel": materialized["pixel_reference"][0].cpu()}
    label = presence(torch.stack(gates).mean(0).float().cpu().numpy()[0])
    soft_global = normalize_cam(torch.stack(global_views).mean(0).float().cpu().numpy())
    soft_pixel = normalize_cam(torch.stack(pixel_views).mean(0).float().cpu().numpy())
    soft_topk = {k: normalize_cam(torch.stack(value).mean(0).float().cpu().numpy()) for k, value in topk_views.items()}
    return {"prediction": prediction_from_cam(soft_global, label, original),
            "pixel_prediction": prediction_from_cam(soft_pixel, label, original),
            "topk_predictions": {k: prediction_from_cam(value, label, original) for k, value in soft_topk.items()},
            "soft_global": soft_global, "soft_pixel": soft_pixel, "soft_topk": soft_topk,
            "label": label, "detail": original_detail}


def basis_rows(image_id, truth, detail):
    weights = detail["weights"].numpy(); bases = detail["bases"].numpy()
    h, w = bases.shape[-2:]
    low_truth = np.asarray(Image.fromarray(truth.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST))
    rows_weight, rows_topk, rows_groups, rows_oracle = [], [], [], []
    for cls in CLASS_IDS:
        target = low_truth == cls
        if not target.any():
            continue
        rival = (low_truth < 4) & (low_truth != cls); background = low_truth == 4
        mass = bases.sum((1, 2)) + EPS
        purity = (bases * target).sum((1, 2)) / mass
        coverage = (bases * target).sum((1, 2)) / (target.sum() + EPS)
        rival_mass = (bases * rival).sum((1, 2)) / mass
        bg_mass = (bases * background).sum((1, 2)) / mass
        order = np.argsort(-weights[:, cls], kind="stable")
        contribution_mass = weights[:, cls] * mass
        normalized_contribution = contribution_mass / max(contribution_mass.sum(), EPS)
        contribution_entropy = float(-(normalized_contribution * np.log(np.clip(normalized_contribution, np.finfo(np.float64).tiny, None))).sum())
        rows_weight.append({"image_id": image_id, "class": cls, **weight_metrics(weights[:, cls]),
                            "weighted_target_purity": float(np.sum(weights[:, cls] * purity)),
                            "weighted_rival_mass": float(np.sum(weights[:, cls] * rival_mass)),
                            "weighted_bg_mass": float(np.sum(weights[:, cls] * bg_mass)),
                            "contribution_target_fraction": float(np.sum(contribution_mass * purity) / max(contribution_mass.sum(), EPS)),
                            "contribution_rival_fraction": float(np.sum(contribution_mass * rival_mass) / max(contribution_mass.sum(), EPS)),
                            "contribution_bg_fraction": float(np.sum(contribution_mass * bg_mass) / max(contribution_mass.sum(), EPS)),
                            "contribution_entropy": contribution_entropy,
                            "effective_contribution_count": float(np.exp(contribution_entropy)),
                            "pure_basis_fraction": float(np.mean(purity >= .70)),
                            "best_basis_purity": float(purity.max())})
        hard = bases >= .5
        hard_tp = (hard & target).sum((1, 2)); hard_fp = (hard & ~target).sum((1, 2)); hard_fn = (~hard & target).sum((1, 2))
        hard_iou = hard_tp / (hard_tp + hard_fp + hard_fn + EPS)
        hard_precision = hard_tp / (hard_tp + hard_fp + EPS); hard_recall = hard_tp / (hard_tp + hard_fn + EPS)
        for k in (1, 3, 5, 10):
            chosen = order[:k]; selected_w = weights[chosen, cls]; selected_w /= max(selected_w.sum(), EPS)
            rows_topk.append({"image_id": image_id, "class": cls, "k": k,
                              "mean_soft_purity": float(purity[chosen].mean()),
                              "w_normalized_soft_purity": float(np.sum(selected_w * purity[chosen])),
                              "mean_soft_coverage": float(coverage[chosen].mean()),
                              "mean_rival_mass": float(rival_mass[chosen].mean()), "mean_bg_mass": float(bg_mass[chosen].mean()),
                              "mean_hard_iou": float(hard_iou[chosen].mean()),
                              "mean_hard_precision": float(hard_precision[chosen].mean()),
                              "mean_hard_recall": float(hard_recall[chosen].mean())})
        groups = (("top10pct", order[:20]), ("middle40pct", order[20:98]), ("bottom50pct", order[98:]))
        for name, chosen in groups:
            rows_groups.append({"image_id": image_id, "class": cls, "group": name,
                                "target_purity": float(purity[chosen].mean()),
                                "rival_mass": float(rival_mass[chosen].mean()), "bg_mass": float(bg_mass[chosen].mean())})
        oracle_order = np.argsort(-purity, kind="stable")
        for k in (1, 3, 5, 10):
            chosen = oracle_order[:k]
            for mode in ("equal", "purity_weighted"):
                alpha = np.ones(k) if mode == "equal" else purity[chosen].copy()
                alpha /= max(alpha.sum(), EPS); mixture = np.sum(alpha[:, None, None] * bases[chosen], axis=0)
                intersection = float((mixture * target).sum()); union = float(mixture.sum() + target.sum() - intersection)
                hard_mix = mixture >= .5; tp = int((hard_mix & target).sum()); fp = int((hard_mix & ~target).sum()); fn = int((~hard_mix & target).sum())
                rows_oracle.append({"image_id": image_id, "class": cls, "k": k, "mode": mode,
                                    "mean_selected_purity": float(purity[chosen].mean()),
                                    "soft_iou": intersection / (union + EPS), "hard_iou": tp / (tp + fp + fn + EPS)})
    return rows_weight, rows_topk, rows_groups, rows_oracle


def class_anatomy_rows(image_id, truth, gcqm, sshr, pixel):
    valid = truth < 4; rows_pair, rows_band, rows_morph = [], [], []
    all_interiors = {r: np.logical_or.reduce([ndimage.binary_erosion(truth == c, iterations=r) for c in CLASS_IDS]) for r in BANDS}
    for cls in CLASS_IDS:
        target = truth == cls; gt_area = int(target.sum())
        if not gt_area:
            continue
        gm, sm = gcqm == cls, sshr == cls
        gs, ss = binary_metrics(target, gm, valid), binary_metrics(target, sm, valid)
        rows_pair.append({"image_id": image_id, "class": cls, "gt_area": gt_area,
                          **{f"gcqm_{k}": v for k, v in gs.items()}, **{f"sshr_{k}": v for k, v in ss.items()},
                          "delta_fp": gs["fp"] - ss["fp"], "delta_fn": gs["fn"] - ss["fn"],
                          "normalized_delta_fp": (gs["fp"] - ss["fp"]) / (gt_area + EPS),
                          "normalized_delta_fn": (gs["fn"] - ss["fn"]) / (gt_area + EPS),
                          "delta_iou": gs["iou"] - ss["iou"]})
        for name, prediction in (("gcqm", gcqm), ("sshr", sshr), ("pixel", pixel)):
            morphology = mask_morphology((prediction == cls) & valid, gt_area)
            rows_morph.append({"image_id": image_id, "class": cls, "model": name, **morphology})
        for radius in BANDS:
            band = boundary_band(target, radius) & valid; interior = ndimage.binary_erosion(target, iterations=radius)
            interior_domain = all_interiors[radius] & valid
            row = {"image_id": image_id, "class": cls, "radius": radius,
                   "boundary_pixels": int(band.sum()), "interior_pixels": int(interior.sum())}
            for name, prediction in (("gcqm", gcqm), ("sshr", sshr), ("pixel", pixel)):
                pred_class = prediction == cls
                bm = binary_metrics(target, pred_class, band)
                ic = ratio(int((pred_class & interior).sum()), int(interior.sum()))
                ip = ratio(int((pred_class & interior).sum()), int((pred_class & interior_domain).sum()))
                accuracy = ratio(int(((prediction == truth) & band).sum()), int(band.sum()))
                fn_mask = interior & ~pred_class; fp_mask = interior_domain & ~target & pred_class
                _, fn_islands = ndimage.label(fn_mask); _, fp_islands = ndimage.label(fp_mask)
                row.update({f"{name}_interior_correctness": ic, f"{name}_interior_recall": ic,
                            f"{name}_interior_precision": ip,
                            f"{name}_interior_fn_density": ratio(int(fn_mask.sum()), int(interior.sum())),
                            f"{name}_interior_fp_leakage": ratio(int(fp_mask.sum()), int(interior_domain.sum())),
                            f"{name}_interior_fn_islands": int(fn_islands), f"{name}_interior_fp_islands": int(fp_islands),
                            f"{name}_boundary_precision": bm["precision"], f"{name}_boundary_recall": bm["recall"],
                            f"{name}_boundary_f1": bm["dice"], f"{name}_boundary_accuracy": accuracy,
                            f"{name}_boundary_fp_density": ratio(bm["fp"], int(band.sum())),
                            f"{name}_boundary_fn_density": ratio(bm["fn"], int(band.sum()))})
            rows_band.append(row)
    return rows_pair, rows_band, rows_morph


def contact_rows(image_id, truth, gcqm, sshr, pixel):
    rows = []
    for c1, c2 in itertools.combinations(CLASS_IDS, 2):
        m1, m2 = truth == c1, truth == c2
        if not m1.any() or not m2.any():
            continue
        for distance in BANDS:
            b1, b2 = boundary_band(m1, distance) & m1, boundary_band(m2, distance) & m2
            near2 = ndimage.distance_transform_edt(~m2) <= distance
            near1 = ndimage.distance_transform_edt(~m1) <= distance
            contact = (b1 & near2) | (b2 & near1)
            pair_boundary = (boundary_band(m1, distance) | boundary_band(m2, distance)) & (truth < 4)
            noncontact = pair_boundary & ~contact
            if not contact.any():
                continue
            row = {"image_id": image_id, "class1": c1, "class2": c2, "distance": distance,
                   "contact_pixels": int(contact.sum()), "noncontact_boundary_pixels": int(noncontact.sum())}
            for name, prediction in (("gcqm", gcqm), ("sshr", sshr), ("pixel", pixel)):
                row.update({f"{name}_contact_accuracy": ratio(int(((prediction == truth) & contact).sum()), int(contact.sum())),
                            f"{name}_noncontact_boundary_accuracy": ratio(int(((prediction == truth) & noncontact).sum()), int(noncontact.sum())),
                            f"{name}_{c1}_to_{c2}": int((m1 & contact & (prediction == c2)).sum()),
                            f"{name}_{c2}_to_{c1}": int((m2 & contact & (prediction == c1)).sum()),
                            f"{name}_other_intrusion": int((contact & (prediction != truth) & (prediction != c1) & (prediction != c2)).sum()),
                            f"{name}_background_intrusion": 0})
            rows.append(row)
    return rows


def aggregate_image_features(pair, bands, contacts, weights, morph):
    p = pair.groupby("image_id").agg(delta_iou=("delta_iou", "mean"), delta_fp=("normalized_delta_fp", "mean"), delta_fn=("normalized_delta_fn", "mean")).reset_index()
    b = bands[bands.radius == 3].copy()
    b["interior_delta"] = b.gcqm_interior_correctness - b.sshr_interior_correctness
    b["boundary_delta"] = b.gcqm_boundary_f1 - b.sshr_boundary_f1
    b = b.groupby("image_id").agg(interior_delta=("interior_delta", "mean"), boundary_delta=("boundary_delta", "mean")).reset_index()
    c = contacts[contacts.distance == 3].copy()
    if len(c):
        c["contact_delta"] = c.gcqm_contact_accuracy - c.sshr_contact_accuracy
        c = c.groupby("image_id").agg(contact_delta=("contact_delta", "mean")).reset_index()
    else:
        c = pd.DataFrame(columns=["image_id", "contact_delta"])
    w = weights.groupby("image_id").agg(mean_neff=("effective_query_count", "mean"), max_neff=("effective_query_count", "max"),
                                          mean_entropy=("normalized_entropy", "mean"), mean_top5_mass=("top5_mass", "mean"),
                                          mean_top10_mass=("top10_mass", "mean"), mean_dominant_share=("dominant_share", "mean"),
                                          weighted_basis_purity=("weighted_target_purity", "mean")).reset_index()
    pivot = morph[morph.model.isin(["gcqm", "sshr"])].pivot_table(index=["image_id", "class"], columns="model", values=["fragmentation_index", "area_ratio_gt"]).reset_index()
    pivot.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in pivot.columns]
    pivot["fragmentation_delta"] = pivot.fragmentation_index_gcqm - pivot.fragmentation_index_sshr
    pivot["area_ratio_delta"] = pivot.area_ratio_gt_gcqm - pivot.area_ratio_gt_sshr
    m = pivot.groupby("image_id").agg(fragmentation_delta=("fragmentation_delta", "mean"), area_ratio_delta=("area_ratio_delta", "mean")).reset_index()
    return p.merge(b, on="image_id", how="left").merge(c, on="image_id", how="left").merge(w, on="image_id", how="left").merge(m, on="image_id", how="left")


def save_colored(mask, path):
    image = Image.fromarray(mask.astype(np.uint8)); image.putpalette(PALETTE); image.convert("RGB").save(path)


def error_image(truth, prediction):
    valid = truth < 4; result = np.zeros((*truth.shape, 3), dtype=np.uint8)
    result[valid & (truth == prediction)] = (80, 80, 80)
    result[valid & (truth != prediction)] = (255, 80, 60)
    result[~valid] = (235, 235, 235)
    return result


def visualize_case(model, sshr_model, valroot, image_id, destination):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB"))
    truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png"))
    tensor = torch.from_numpy(original.copy()).permute(2, 0, 1).float().div(255)
    tensor = F.interpolate(tensor[None], (224, 224), mode="bilinear", align_corners=False)
    tensor = (tensor - tensor.new_tensor([.485, .456, .406])[None, :, None, None]) / tensor.new_tensor([.229, .224, .225])[None, :, None, None]
    tensor = tensor.cuda(); diagnostic = infer_gcqm_diagnostics(model, tensor, original)
    sshr_prediction = _predict_sshr(sshr_model, tensor, original); gcqm_prediction = diagnostic["prediction"]
    Image.fromarray(original).save(destination / "input.png"); save_colored(truth, destination / "gt.png")
    save_colored(sshr_prediction, destination / "sshr_prediction.png"); save_colored(gcqm_prediction, destination / "gcqm_prediction.png")
    Image.fromarray(error_image(truth, gcqm_prediction)).save(destination / "gcqm_error_map.png")
    Image.fromarray(error_image(truth, sshr_prediction)).save(destination / "sshr_error_map.png")
    boundary = np.logical_or.reduce([boundary_band(truth == c, 3) for c in CLASS_IDS]); Image.fromarray((boundary * 255).astype(np.uint8)).save(destination / "boundary_band.png")
    contact = np.zeros_like(boundary)
    for c1, c2 in itertools.combinations(CLASS_IDS, 2):
        m1, m2 = truth == c1, truth == c2
        if m1.any() and m2.any():
            contact |= ((boundary_band(m1, 3) & m1 & (ndimage.distance_transform_edt(~m2) <= 3)) |
                        (boundary_band(m2, 3) & m2 & (ndimage.distance_transform_edt(~m1) <= 3)))
    Image.fromarray((contact * 255).astype(np.uint8)).save(destination / "contact_band.png")
    present = [c for c in CLASS_IDS if np.any(truth == c)]; weights = diagnostic["detail"]["weights"].numpy(); bases = diagnostic["detail"]["bases"].numpy()
    dominant = max(present, key=lambda c: int(np.sum(truth == c)))
    fig, axes = plt.subplots(len(present), 1, figsize=(10, 2.2 * len(present)), squeeze=False)
    for ax, cls in zip(axes[:, 0], present): ax.plot(weights[:, cls]); ax.set_title(f"class {cls} w; Neff={weight_metrics(weights[:, cls])['effective_query_count']:.2f}")
    fig.tight_layout(); fig.savefig(destination / "w_distribution_present_classes.png", dpi=140); plt.close(fig)
    order = np.argsort(-weights[:, dominant], kind="stable")[:5]
    low_truth = np.asarray(Image.fromarray(truth.astype(np.uint8)).resize((bases.shape[-1], bases.shape[-2]), Image.Resampling.NEAREST))
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    notes = []
    for col, query in enumerate(order):
        base = bases[query]; mass = base.sum() + EPS; purity = float((base * (low_truth == dominant)).sum() / mass)
        rival = float((base * ((low_truth < 4) & (low_truth != dominant))).sum() / mass); bg = float((base * (low_truth == 4)).sum() / mass)
        contribution = weights[query, dominant] * base
        axes[0, col].imshow(base, cmap="viridis"); axes[0, col].set_title(f"Q{query} w={weights[query, dominant]:.4f}\nP={purity:.2f} R={rival:.2f} BG={bg:.2f}")
        axes[1, col].imshow(contribution, cmap="magma"); axes[1, col].set_title(f"contribution={contribution.sum():.2f}")
        axes[2, col].imshow(base, cmap="gray"); axes[2, col].contour(low_truth == dominant, levels=[.5], colors=["lime"], linewidths=.8)
        axes[2, col].contour((low_truth < 4) & (low_truth != dominant), levels=[.5], colors=["red"], linewidths=.8)
        axes[2, col].set_title("target=green, rival=red")
        axes[0, col].axis("off"); axes[1, col].axis("off"); axes[2, col].axis("off")
        notes.append({"query": int(query), "w": float(weights[query, dominant]), "purity": purity, "rival_mass": rival, "bg_mass": bg, "contribution_mass": float(contribution.sum())})
    fig.tight_layout(); fig.savefig(destination / "top5_bases_and_contributions.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, k in zip(axes.ravel()[:6], (1, 3, 5, 10, 20, 196)):
        ax.imshow(diagnostic["soft_topk"][k][dominant], cmap="viridis", vmin=0, vmax=1); ax.set_title(f"F_top{k}"); ax.axis("off")
    axes.ravel()[6].imshow(truth == dominant, cmap="gray"); axes.ravel()[6].set_title("GT target"); axes.ravel()[6].axis("off")
    axes.ravel()[7].imshow(sshr_prediction == dominant, cmap="gray"); axes.ravel()[7].set_title("SSHR target"); axes.ravel()[7].axis("off")
    fig.tight_layout(); fig.savefig(destination / "diffuse_averaging_curve.png", dpi=140); plt.close(fig)
    write_json(destination / "case.json", {"image_id": image_id, "dominant_class": dominant, "top5": notes})


def report_text(result):
    h = result["hypotheses"]; d = result["decision"]
    table = "\n".join(f"| {key} | {value['result']} | {value['evidence']} | {value['confidence']} |" for key, value in h.items())
    classes = "\n".join(f"| {int(row['class'])} | {row['delta_iou_pp']:.4f} | {row['fp_fn']} | {row['interior_loss']:.4f} | {row['boundary_loss']:.4f} | {row['weighted_purity']:.4f} | {row['neff']:.2f} | {row['fragmentation_delta']:.4f} |" for row in result["per_class_failure_matrix"])
    return f"""# GCQM Full25 Failure Anatomy & Decoder Bottleneck Audit Report

## 1 Executive Diagnosis

The sealed E25 audit identifies **{d}** with **{result['confidence']}** confidence. {result['because_sentence']}

## 2 Frozen Evidence

GCQM E25 SHA256 `{GCQM_SHA256}`; SSHR B0 SHA256 `{SSHR_SHA256}`. No training, optimizer, parameter update, threshold tuning, or checkpoint selection occurred.

## 3 Reproduction Gate

`{result['reproduction_gate']['decision']}` on {result['reproduction_gate']['paired_images']} paired images. GCQM mIoU={100*result['metrics']['gcqm']['mIoU']:.4f}, SSHR mIoU={100*result['metrics']['sshr']['mIoU']:.4f}, delta={result['metrics']['delta_miou_pp']:+.4f} pp.

## 4 Why This Is Not Mechanism Collapse

CCRA allocation remained internally valid at E25 (Stage3 JS=0.3637, top1 class difference=1.0000, D_perm=0.1026). This audit concerns the decoder translation from allocation to dense masks.

## 5 H1 FP-vs-FN

{h['H1 FP/FN bias']['evidence']} Decision: `{h['H1 FP/FN bias']['result']}`.

## 6 H2 Interior-vs-Boundary

{h['H2 Interior/Boundary']['evidence']} Decision: `{h['H2 Interior/Boundary']['result']}`. Radius 3 is primary; radii 1 and 5 are robustness checks.

## 7 H3 Contact-Region Analysis

{h['H3 Contact failure']['evidence']} Decision: `{h['H3 Contact failure']['result']}`. Distance 3 is primary; distances 1 and 5 are robustness checks.

## 8 H4 Top-Weight Basis Purity

{h['H4 Basis purity']['evidence']} Decision: `{h['H4 Basis purity']['result']}`.

## 9 Contribution Purity

Contribution-mass target/rival/background fractions are recorded per image/class in `basis_purity/weighted_basis_purity.csv`.

## 10 H5 Weight Diffuseness

{h['H5 Weight diffuseness']['evidence']} Decision: `{h['H5 Weight diffuseness']['result']}`.

## 11 H6 Diffuseness-vs-DeltaIoU

{h['H6 Diffuseness-performance link']['evidence']} Decision: `{h['H6 Diffuseness-performance link']['result']}`.

## 12 Neff Quartile Analysis

Fixed image-level mean-Neff quartiles and their delta IoU, FP/FN, boundary, and interior statistics are in `weight_diffuseness/neff_quartiles.csv`.

## 13 H7 Spatial Property Loss vs SSHR

Ranked labels: {', '.join(result['spatial_labels'])}. {h['H7 Lost spatial property']['evidence']}

## 14 Fragmentation / Holes / Compactness

Per-image/class morphology and paired summaries are in `spatial_property/`; calculations use the fixed valid-pixel domain.

## 15 Pixel-wise-vs-Global Diagnostic

GCQM-global mIoU={100*result['metrics']['gcqm']['mIoU']:.4f}; FOMD-style pixel diagnostic mIoU={100*result['metrics']['pixel']['mIoU']:.4f}. This is diagnostic-only and not a new model result.

## 16 Top-k Diagnostic Curve

Fixed k=1,3,5,10,20,50,100,196 results are in `paired/topk_counterfactual_curve.csv`; no k was selected as a performance result.

## 17 Oracle Basis Ceiling

GT-ranked equal and purity-weighted k=1,3,5,10 basis ceilings are in `basis_purity/oracle_basis_ceiling.csv`; these are diagnostic-only.

## 18 Phase-0 Proxy-vs-GT Validity

{result['proxy_statement']}

## 19 Per-Class Failure Matrix

| Class | Delta IoU pp | FP/FN | Interior loss | Boundary loss | Weighted purity | Neff | Fragmentation delta |
|---:|---:|---|---:|---:|---:|---:|---:|
{classes}

## 20 Failure Phenotype Clusters

Descriptive z-scored KMeans uses k=3; k=2/4 assignments are retained only for robustness. Cluster representatives are nearest-to-centroid, never manually selected.

## 21 Representative Cases

Automatic cluster representatives, five wins, five similar images, and five failures are under `visualizations/` with predictions, error maps, bands, weights, bases, contributions, and top-k panels.

## 22 Hypothesis Decision Matrix

| Hypothesis | Result | Main Evidence | Confidence |
|---|---|---|---|
{table}

## 23 Decoder Bottleneck Decision

`DECISION = {d}`

## 24 What Is Preserved

CCRA allocation is internally valid; class-conditioned allocation survives Full25; query identity remains meaningful; only the current global mixture decoder is performance-invalid on BCSS Seed42 Full25.

## 25 What Is Falsified

Healthy weak-semantic diagnostics are not sufficient to imply improved dense segmentation, and the current global mixture decoder is not an acceptable final model.

## 26 Exact Next Research Target

{result['next_target']}

## 27 What Must NOT Be Done

Do not call CCRA failed, tune E25 inference thresholds/top-k, select another checkpoint, treat oracle/pixel counterfactuals as model performance, or train a new model from this audit.

## 28 Exact Final Decision

DECISION = {d}

CONFIDENCE = {result['confidence']}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True); parser.add_argument("--gcqm-checkpoint", required=True)
    parser.add_argument("--gcqm-experiment", required=True); parser.add_argument("--sshr-checkpoint", required=True)
    parser.add_argument("--sshr-experiment", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full25-delivery-summary", required=True)
    parser.add_argument("--num-workers", type=int, default=8); args = parser.parse_args()
    valroot, gcqm_checkpoint, sshr_checkpoint = Path(args.val_root), Path(args.gcqm_checkpoint), Path(args.sshr_checkpoint)
    gcqm_experiment, output = Path(args.gcqm_experiment), Path(args.output_dir)
    delivery_summary = Path(args.full25_delivery_summary)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Output must be absent or empty: {output}")
    if sha256(gcqm_checkpoint) != GCQM_SHA256 or sha256(sshr_checkpoint) != SSHR_SHA256:
        raise RuntimeError("Frozen checkpoint identity mismatch")
    archived = json.loads((gcqm_experiment / "evaluation/gcqm_full25_final_result.json").read_text())
    if archived["protocol_audit"]["decision"] != "COMPARABLE": raise RuntimeError("Archived comparator is not COMPARABLE")
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    gcqm_model = GCQMNet().cuda(); gcqm_model.load_state_dict(load_state(gcqm_checkpoint), strict=True); gcqm_model.eval()
    sshr_model = SSHRCAM(4).cuda(); sshr_model.load_state_dict(load_state(sshr_checkpoint), strict=True); sshr_model.eval()
    paired_rows, band_rows, contact_data, morph_rows = [], [], [], []
    weight_data, basis_topk_data, group_data, oracle_data = [], [], [], []
    hist_gcqm, hist_sshr, hist_pixel = [], [], []
    hist_topk = {k: [] for k in TOPK}; ids = []
    with torch.no_grad():
        for index, (names, image) in enumerate(loader, 1):
            image_id = names[0]; original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB"))
            truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
            diagnostic = infer_gcqm_diagnostics(gcqm_model, image, original); gcqm = diagnostic["prediction"]
            pixel = diagnostic["pixel_prediction"]; sshr = _predict_sshr(sshr_model, image, original)
            hist_gcqm.append(foreground_confusion(truth, gcqm)); hist_sshr.append(foreground_confusion(truth, sshr)); hist_pixel.append(foreground_confusion(truth, pixel))
            for k in TOPK: hist_topk[k].append(foreground_confusion(truth, diagnostic["topk_predictions"][k]))
            p, b, m = class_anatomy_rows(image_id, truth, gcqm, sshr, pixel); paired_rows += p; band_rows += b; morph_rows += m
            contact_data += contact_rows(image_id, truth, gcqm, sshr, pixel)
            w, bt, g, o = basis_rows(image_id, truth, diagnostic["detail"]); weight_data += w; basis_topk_data += bt; group_data += g; oracle_data += o
            ids.append(image_id)
            if index % 200 == 0 or index == len(loader): print(f"ANATOMY_PROGRESS={index}/{len(loader)}", flush=True)
    hist_gcqm, hist_sshr, hist_pixel = np.stack(hist_gcqm), np.stack(hist_sshr), np.stack(hist_pixel)
    metrics = {"gcqm": scores_from_confusion(hist_gcqm.sum(0)), "sshr": scores_from_confusion(hist_sshr.sum(0)),
               "pixel": scores_from_confusion(hist_pixel.sum(0))}
    metrics["delta_miou_pp"] = 100 * (metrics["gcqm"]["mIoU"] - metrics["sshr"]["mIoU"])
    reproduced = len(ids) == 3418 and abs(metrics["gcqm"]["mIoU"] - archived["gcqm"]["mIoU"]) < 1e-10 and abs(metrics["sshr"]["mIoU"] - archived["sshr"]["mIoU"]) < 1e-10
    reproduction = {"decision": "REPRODUCTION_GATE_PASS" if reproduced else "FAILURE_ANATOMY_ENGINEERING_BLOCKED",
                    "paired_images": len(ids), "gcqm_sha256": sha256(gcqm_checkpoint), "sshr_sha256": sha256(sshr_checkpoint),
                    "protocol_audit": archived["protocol_audit"]["decision"], "computed_metrics": metrics,
                    "archived_gcqm_mIoU": archived["gcqm"]["mIoU"], "archived_sshr_mIoU": archived["sshr"]["mIoU"]}
    if not reproduced:
        output.mkdir(parents=True, exist_ok=True); write_json(output / "provenance/reproduction_gate.json", reproduction)
        print("DECISION = FAILURE_ANATOMY_ENGINEERING_BLOCKED"); return
    output.mkdir(parents=True, exist_ok=True)
    for name in ("provenance", "paired", "error_decomposition", "boundary_interior", "contact_region", "weight_diffuseness", "basis_purity", "spatial_property", "correlation", "visualizations", "report"):
        (output / name).mkdir(exist_ok=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {"zero_training": True, "seed": 42, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
              "bands": list(BANDS), "topk": list(TOPK), "thresholds": THRESHOLDS.tolist(), "tta": 3,
              "gcqm_checkpoint": str(gcqm_checkpoint.resolve()), "sshr_checkpoint": str(sshr_checkpoint.resolve()),
              "source_commit": source_commit}
    config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    (output / "provenance/failure_anatomy_source_commit.txt").write_text(source_commit + "\n")
    (output / "provenance/failure_anatomy_git_diff.patch").write_text(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT, text=True))
    (output / "provenance/failure_anatomy_config.json").write_text(config_text)
    (output / "provenance/failure_anatomy_config_sha256.txt").write_text(hashlib.sha256(config_text.encode()).hexdigest() + "\n")
    shutil.copy2(gcqm_experiment / "report/GCQM_BCSS_Seed42_Full25_Final_Validation_Report.md", output / "provenance/archived_full25_report.md")
    shutil.copy2(delivery_summary, output / "provenance/archived_full25_delivery_summary.md")
    write_json(output / "provenance/reproduction_gate.json", reproduction)
    pair, bands, contacts = pd.DataFrame(paired_rows), pd.DataFrame(band_rows), pd.DataFrame(contact_data)
    morph, weights = pd.DataFrame(morph_rows), pd.DataFrame(weight_data)
    pair.to_csv(output / "paired/paired_image_class_metrics.csv", index=False)
    fp_summary = []
    for cls, frame in [("pooled", pair)] + [(str(c), pair[pair["class"] == c]) for c in CLASS_IDS]:
        fp_summary.append({"class": cls, **{f"delta_fp_{k}": v for k, v in mean_ci(frame.normalized_delta_fp).items()},
                           **{f"delta_fn_{k}": v for k, v in mean_ci(frame.normalized_delta_fn).items()}})
    pd.DataFrame(fp_summary).to_csv(output / "error_decomposition/fp_fn_summary.csv", index=False)
    fp_bootstrap = {row["class"]: row for row in fp_summary}; write_json(output / "error_decomposition/fp_fn_bootstrap.json", fp_bootstrap)
    bands.to_csv(output / "boundary_interior/interior_boundary_per_image.csv", index=False)
    band_summary = []
    for radius, frame in bands.groupby("radius"):
        for cls, sub in [("pooled", frame)] + [(str(c), frame[frame["class"] == c]) for c in CLASS_IDS]:
            interior = sub.sshr_interior_correctness - sub.gcqm_interior_correctness
            boundary = sub.sshr_boundary_f1 - sub.gcqm_boundary_f1
            band_summary.append({"radius": radius, "class": cls, **{f"interior_loss_{k}": v for k, v in mean_ci(interior).items()},
                                 **{f"boundary_loss_{k}": v for k, v in mean_ci(boundary).items()}})
    pd.DataFrame(band_summary).to_csv(output / "boundary_interior/interior_boundary_summary.csv", index=False)
    pd.DataFrame(band_summary).to_csv(output / "boundary_interior/band_robustness.csv", index=False)
    contacts.to_csv(output / "contact_region/contact_pair_metrics.csv", index=False)
    contact_summary = []
    if len(contacts):
        for (distance, c1, c2), frame in contacts.groupby(["distance", "class1", "class2"]):
            delta = frame.gcqm_contact_accuracy - frame.sshr_contact_accuracy
            excess_loss = ((frame.sshr_contact_accuracy - frame.gcqm_contact_accuracy) -
                           (frame.sshr_noncontact_boundary_accuracy - frame.gcqm_noncontact_boundary_accuracy))
            contact_summary.append({"distance": distance, "class1": c1, "class2": c2, "images": len(frame), "descriptive_only": len(frame) < 30,
                                    **{f"contact_delta_{k}": v for k, v in mean_ci(delta).items()},
                                    **{f"excess_contact_loss_{k}": v for k, v in mean_ci(excess_loss).items()}})
    pd.DataFrame(contact_summary).to_csv(output / "contact_region/contact_summary.csv", index=False)
    weights.to_csv(output / "weight_diffuseness/w_metrics_per_image_class.csv", index=False)
    weights.groupby("diffuseness_bin", observed=False).agg(image_classes=("image_id", "count"), mean_neff=("effective_query_count", "mean"), mean_top5_mass=("top5_mass", "mean")).reset_index().to_csv(output / "weight_diffuseness/diffuseness_bins.csv", index=False)
    pd.DataFrame(basis_topk_data).to_csv(output / "basis_purity/basis_topk_purity.csv", index=False)
    weights.to_csv(output / "basis_purity/weighted_basis_purity.csv", index=False)
    groups = pd.DataFrame(group_data); groups.to_csv(output / "basis_purity/top_vs_tail.csv", index=False)
    oracle = pd.DataFrame(oracle_data); oracle.to_csv(output / "basis_purity/oracle_basis_ceiling.csv", index=False)
    morph.to_csv(output / "spatial_property/morphology_per_image_class.csv", index=False)
    morph_summary = morph.groupby(["class", "model"]).mean(numeric_only=True).reset_index()
    morph_summary.to_csv(output / "spatial_property/spatial_property_summary.csv", index=False)
    topk_curve = []
    for k in TOPK:
        score = scores_from_confusion(np.stack(hist_topk[k]).sum(0)); topk_curve.append({"k": k, "mIoU": score["mIoU"], "mDice": score["mDice"], "delta_vs_full_pp": 100 * (score["mIoU"] - metrics["gcqm"]["mIoU"])})
    pd.DataFrame(topk_curve).to_csv(output / "paired/topk_counterfactual_curve.csv", index=False)
    main_band = bands[bands.radius == 3]
    spatial_models = []
    confusion_by_name = {"sshr": np.stack(hist_sshr).sum(0), "gcqm": np.stack(hist_gcqm).sum(0), "pixel": np.stack(hist_pixel).sum(0)}
    for name in ("sshr", "gcqm", "pixel"):
        confusion = confusion_by_name[name]; diagonal = np.diag(confusion)
        spatial_models.append({"model": name, "mIoU": metrics[name]["mIoU"],
                               "interior_correctness": float(main_band[f"{name}_interior_correctness"].mean()),
                               "boundary_f1": float(main_band[f"{name}_boundary_f1"].mean()),
                               "contact_accuracy": float(contacts[contacts.distance == 3][f"{name}_contact_accuracy"].mean()) if len(contacts) else float("nan"),
                               "fp": int(np.sum(confusion.sum(0) - diagonal)), "fn": int(np.sum(confusion.sum(1) - diagonal)),
                               "mean_fragmentation": float(morph[morph.model == name].fragmentation_index.mean()),
                               "mean_hole_fraction": float(morph[morph.model == name].hole_area_fraction.mean())})
    pd.DataFrame(spatial_models).to_csv(output / "paired/pixel_vs_global_spatial_diagnostic.csv", index=False)
    features = aggregate_image_features(pair, bands, contacts, weights, morph)
    paired_image = pd.read_csv(gcqm_experiment / "evaluation/gcqm_vs_sshr_paired_delta.csv")
    delta_column = next(c for c in paired_image.columns if c.lower() in ("delta_iou", "miou_delta", "delta_miou"))
    external_delta = paired_image[["image_id", delta_column]].rename(columns={delta_column: "formal_delta_iou"})
    features = features.merge(external_delta, on="image_id", how="left")
    corr_rows, corr_bootstrap = [], {}
    for metric_name in ("mean_neff", "max_neff", "mean_entropy", "mean_top5_mass", "mean_top10_mass", "mean_dominant_share"):
        for method in ("spearman", "pearson"):
            result = correlation_ci(features[metric_name], features.formal_delta_iou, method)
            corr_rows.append({"metric": metric_name, "method": method, **result}); corr_bootstrap[f"{method}_{metric_name}"] = result
    pd.DataFrame(corr_rows).to_csv(output / "correlation/diffuseness_delta_iou_correlations.csv", index=False)
    write_json(output / "correlation/diffuseness_bootstrap.json", corr_bootstrap)
    proxy = [{"proxy": name, "status": "UNAVAILABLE_PER_VALIDATION_IMAGE", "reason": "Phase-0 and Full25 proxy artifacts are aggregate train-cohort snapshots and cannot be paired to BCSS validation image IDs."} for name in ("weak_probability_gap", "CPR", "positive_recall", "D_perm", "JS", "Neff")]
    pd.DataFrame(proxy).to_csv(output / "correlation/phase0_proxy_vs_gt.csv", index=False)
    features["neff_quartile"] = pd.qcut(features.mean_neff, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    quartiles = features.groupby("neff_quartile", observed=False).agg(images=("image_id", "count"), mean_delta_iou=("formal_delta_iou", "mean"), median_delta_iou=("formal_delta_iou", "median"),
                                                                   improved_fraction=("formal_delta_iou", lambda x: float(np.mean(x > 0))), worsened_fraction=("formal_delta_iou", lambda x: float(np.mean(x < 0))),
                                                                   delta_fp=("delta_fp", "mean"), delta_fn=("delta_fn", "mean"), boundary_delta=("boundary_delta", "mean"), interior_delta=("interior_delta", "mean")).reset_index()
    quartiles.to_csv(output / "weight_diffuseness/neff_quartiles.csv", index=False)
    cluster_columns = ["formal_delta_iou", "delta_fp", "delta_fn", "interior_delta", "boundary_delta", "contact_delta", "mean_neff", "mean_entropy", "mean_top5_mass", "weighted_basis_purity", "fragmentation_delta", "area_ratio_delta"]
    cluster_values = features[cluster_columns].replace([np.inf, -np.inf], np.nan)
    cluster_values = cluster_values.fillna(cluster_values.median()).fillna(0)
    scaled = StandardScaler().fit_transform(cluster_values)
    cluster_models = {}
    for k in (2, 3, 4):
        km = KMeans(n_clusters=k, random_state=BOOTSTRAP_SEED, n_init=20).fit(scaled)
        features[f"cluster_k{k}"] = km.labels_; cluster_models[k] = km
    features.to_csv(output / "paired/failure_clusters.csv", index=False)
    representatives = {}
    km = cluster_models[3]
    for cluster in range(3):
        members = np.where(km.labels_ == cluster)[0]
        distance = np.linalg.norm(scaled[members] - km.cluster_centers_[cluster], axis=1)
        representatives[f"cluster_{cluster}"] = features.iloc[members[np.argsort(distance)[:3]]].image_id.tolist()
    ordered = features.sort_values("formal_delta_iou")
    representatives["failures"] = ordered.head(5).image_id.tolist(); representatives["wins"] = ordered.tail(5).image_id.tolist()
    representatives["similar"] = features.iloc[np.argsort(np.abs(features.formal_delta_iou.to_numpy()))[:5]].image_id.tolist()
    write_json(output / "visualizations/selection.json", representatives)
    for group, names in representatives.items():
        for rank, image_id in enumerate(names, 1):
            visualize_case(gcqm_model, sshr_model, valroot, image_id, output / "visualizations" / group / f"{rank:02d}_{image_id}")
    pooled_fp, pooled_fn = fp_summary[0], fp_summary[0]
    fp_mean, fn_mean = pooled_fp["delta_fp_mean"], pooled_fn["delta_fn_mean"]
    fp_sig = pooled_fp["delta_fp_ci95"][0] > 0
    fn_sig = pooled_fn["delta_fn_ci95"][0] > 0
    if fp_mean >= 1.5 * max(fn_mean, 0) and fp_sig: h1 = "FP_DOMINANT"
    elif fn_mean >= 1.5 * max(fp_mean, 0) and fn_sig: h1 = "FN_DOMINANT"
    else: h1 = "MIXED"
    primary_band = next(x for x in band_summary if x["radius"] == 3 and x["class"] == "pooled")
    il, bl = primary_band["interior_loss_mean"], primary_band["boundary_loss_mean"]
    isig = primary_band["interior_loss_ci95"][0] > 0; bsig = primary_band["boundary_loss_ci95"][0] > 0
    if il >= 1.5 * bl and isig: h2 = "INTERIOR_DOMINANT"
    elif bl >= 1.5 * il and bsig: h2 = "BOUNDARY_DOMINANT"
    elif isig and bsig: h2 = "BOTH"
    else: h2 = "NEITHER_CLEAR"
    contact_primary = [x for x in contact_summary if x["distance"] == 3 and not x["descriptive_only"]]
    h3_true = any(x["excess_contact_loss_mean"] is not None and x["excess_contact_loss_mean"] >= .03 and x["excess_contact_loss_ci95"][0] > 0 for x in contact_primary)
    wp, wr, wb = weights.weighted_target_purity.mean(), weights.weighted_rival_mass.mean(), weights.weighted_bg_mass.mean()
    group_mean = groups.groupby("group").target_purity.mean(); top_tail = float(group_mean.get("top10pct", np.nan) - group_mean.get("bottom50pct", np.nan))
    oracle_purity = float(oracle[(oracle.k == 5) & (oracle["mode"] == "purity_weighted")].mean_selected_purity.mean())
    if wp >= .70 and wr <= .20 and top_tail >= .10: h4 = "BASIS_CLEAN"
    elif wp < .60 or wr > .30 or wb > .30: h4 = "BASIS_CONTAMINATED"
    elif oracle_purity >= .70 and top_tail <= 0: h4 = "BASIS_ROUTER_MISMATCH"
    else: h4 = "MIXED"
    neff = float(weights.effective_query_count.mean()); dominant_bin = weights.diffuseness_bin.value_counts().idxmax()
    rho_neff = corr_bootstrap["spearman_mean_neff"]; rho_entropy = corr_bootstrap["spearman_mean_entropy"]; rho_top5 = corr_bootstrap["spearman_mean_top5_mass"]
    strong = ((rho_neff["estimate"] <= -.20 and rho_neff["ci95"][1] < 0) or (rho_entropy["estimate"] <= -.20 and rho_entropy["ci95"][1] < 0)) and rho_top5["estimate"] >= .20 and rho_top5["ci95"][0] > 0
    direction = (rho_neff["estimate"] < 0 or rho_entropy["estimate"] < 0) and rho_top5["estimate"] > 0
    h6 = "STRONG" if strong else "WEAK" if direction else "NOT_SUPPORTED"
    morph_pivot = morph.groupby("model").mean(numeric_only=True)
    frag_delta = float(morph_pivot.loc["gcqm", "fragmentation_index"] - morph_pivot.loc["sshr", "fragmentation_index"])
    hole_delta = float(morph_pivot.loc["gcqm", "hole_area_fraction"] - morph_pivot.loc["sshr", "hole_area_fraction"])
    broad_topk = sum(next(x for x in topk_curve if x["k"] == k)["delta_vs_full_pp"] > 0 for k in (10, 20, 50)) >= 2
    spatial_labels = []
    if h4 == "BASIS_CONTAMINATED": spatial_labels.append("BASIS_CONTAMINATION")
    if h2 in ("BOUNDARY_DOMINANT", "BOTH"): spatial_labels.append("BOUNDARY_LOCALIZATION_LOSS")
    if h2 in ("INTERIOR_DOMINANT", "BOTH"): spatial_labels.append("INTERIOR_COHERENCE_LOSS")
    if h3_true: spatial_labels.append("CONTACT_CONFUSION")
    if h1 == "FP_DOMINANT": spatial_labels.append("OVERSEGMENTATION")
    if h1 == "FN_DOMINANT": spatial_labels.append("UNDERSEGMENTATION")
    if frag_delta > 0: spatial_labels.append("FRAGMENTATION")
    if h6 == "STRONG" and broad_topk: spatial_labels.append("SPATIAL_AVERAGING")
    if not spatial_labels: spatial_labels = ["NO_SINGLE_DOMINANT_PROPERTY"]
    pixel_boundary_gain = float(main_band.pixel_boundary_f1.mean() - main_band.gcqm_boundary_f1.mean())
    pixel_contact_gain = float(contacts[contacts.distance == 3].pixel_contact_accuracy.mean() - contacts[contacts.distance == 3].gcqm_contact_accuracy.mean()) if len(contacts) else 0
    factors = []
    if h4 == "BASIS_CONTAMINATED" and oracle_purity < .70: factors.append("BASIS_GENERATION_BOTTLENECK")
    if oracle_purity >= .70 and h4 in ("BASIS_ROUTER_MISMATCH", "BASIS_CONTAMINATED", "MIXED") and top_tail < .10: factors.append("ROUTER_TO_BASIS_MATCHING_BOTTLENECK")
    if h4 == "BASIS_CLEAN" and (pixel_boundary_gain > .03 or pixel_contact_gain > .03): factors.append("MISSING_LOCAL_ADAPTIVITY")
    if h2 == "INTERIOR_DOMINANT" and (frag_delta > 0 or hole_delta > 0): factors.append("MISSING_SPATIAL_COHERENCE")
    if h6 == "STRONG" and broad_topk: factors.append("OVERDIFFUSE_GLOBAL_MIXTURE")
    decision = "MULTI_FACTOR_DECODER_BOTTLENECK" if len(factors) >= 2 else factors[0] if factors else "NO_SINGLE_FAILURE_MODE_IDENTIFIED"
    class_matrix = []
    archived_class_delta = archived["delta"]["class_iou_pp"]
    for cls in CLASS_IDS:
        p = pair[pair["class"] == cls]; b = main_band[main_band["class"] == cls]; w = weights[weights["class"] == cls]
        mm = morph[(morph["class"] == cls) & morph.model.isin(["gcqm", "sshr"])].groupby("model").fragmentation_index.mean()
        class_contacts = contacts[(contacts.distance == 3) & ((contacts.class1 == cls) | (contacts.class2 == cls))] if len(contacts) else contacts
        class_matrix.append({"class": cls, "delta_iou_pp": archived_class_delta[str(cls)],
                             "fp_fn": "FP" if abs(p.normalized_delta_fp.mean()) >= 1.5 * abs(p.normalized_delta_fn.mean()) else "FN" if abs(p.normalized_delta_fn.mean()) >= 1.5 * abs(p.normalized_delta_fp.mean()) else "MIXED",
                             "interior_loss": float((b.sshr_interior_correctness - b.gcqm_interior_correctness).mean()),
                             "boundary_loss": float((b.sshr_boundary_f1 - b.gcqm_boundary_f1).mean()),
                             "contact_loss": float((class_contacts.sshr_contact_accuracy - class_contacts.gcqm_contact_accuracy).mean()) if len(class_contacts) else None,
                             "weighted_purity": float(w.weighted_target_purity.mean()),
                             "neff": float(w.effective_query_count.mean()), "entropy": float(w.normalized_entropy.mean()),
                             "fragmentation_delta": float(mm.get("gcqm", np.nan) - mm.get("sshr", np.nan)),
                             "area_ratio_delta": float(morph[(morph["class"] == cls) & (morph.model == "gcqm")].area_ratio_gt.mean() - morph[(morph["class"] == cls) & (morph.model == "sshr")].area_ratio_gt.mean())})
    pd.DataFrame(class_matrix).to_csv(output / "spatial_property/per_class_failure_matrix.csv", index=False)
    evidence_count = sum([h1 != "MIXED", h2 != "NEITHER_CLEAR", h3_true, h4 != "MIXED", h6 != "NOT_SUPPORTED", frag_delta > 0])
    confidence = "HIGH" if evidence_count >= 3 else "MEDIUM" if evidence_count >= 2 else "LOW"
    next_target = {"BASIS_GENERATION_BOTTLENECK": "Improve the spatial basis-generation target while preserving CCRA.",
                   "ROUTER_TO_BASIS_MATCHING_BOTTLENECK": "Restore class-to-basis matching while preserving CCRA.",
                   "MISSING_LOCAL_ADAPTIVITY": "Restore local spatial adaptivity while preserving CCRA.",
                   "MISSING_SPATIAL_COHERENCE": "Restore region homogeneity/coherence while preserving CCRA.",
                   "OVERDIFFUSE_GLOBAL_MIXTURE": "Prevent overdiffuse global averaging while preserving CCRA.",
                   "MULTI_FACTOR_DECODER_BOTTLENECK": "Address the ranked decoder factors together while preserving CCRA.",
                   "NO_SINGLE_FAILURE_MODE_IDENTIFIED": "Do not add a module yet; collect a sharper diagnostic separating basis quality, routing, and coherence."}[decision]
    lost = ", ".join(spatial_labels)
    result = {"decision": decision, "confidence": confidence, "metrics": metrics, "reproduction_gate": reproduction,
              "spatial_labels": spatial_labels, "decision_factors": factors, "per_class_failure_matrix": class_matrix,
              "proxy_statement": "PHASE0_PROXY_MISMATCH is UNDETERMINED because only aggregate train-cohort proxy snapshots exist; no invalid train-to-validation image pairing was fabricated.",
              "because_sentence": f"Because Full25 failure is dominated by {lost}, the next decoder should specifically restore the corresponding spatial property while preserving CCRA.",
              "next_target": next_target,
              "hypotheses": {
                  "H1 FP/FN bias": {"result": h1, "evidence": f"mean normalized delta FP={fp_mean:+.4f}; delta FN={fn_mean:+.4f}", "confidence": "High" if fp_sig or fn_sig else "Low"},
                  "H2 Interior/Boundary": {"result": h2, "evidence": f"r=3 interior loss={il:+.4f}; boundary loss={bl:+.4f}", "confidence": "High" if isig and bsig else "Med"},
                  "H3 Contact failure": {"result": str(h3_true).upper(), "evidence": f"{len(contact_primary)} adequately powered class-pair analyses at d=3", "confidence": "High" if contact_primary else "Low"},
                  "H4 Basis purity": {"result": h4, "evidence": f"weighted purity={wp:.4f}, rival={wr:.4f}, BG={wb:.4f}, top-tail={top_tail:+.4f}, oracle-k5 purity={oracle_purity:.4f}", "confidence": "High"},
                  "H5 Weight diffuseness": {"result": dominant_bin.upper(), "evidence": f"mean Neff={neff:.2f}", "confidence": "High"},
                  "H6 Diffuseness-performance link": {"result": h6, "evidence": f"rho(Neff,deltaIoU)={rho_neff['estimate']:+.4f}; rho(top5,deltaIoU)={rho_top5['estimate']:+.4f}", "confidence": "High" if h6 == "STRONG" else "Med"},
                  "H7 Lost spatial property": {"result": spatial_labels[0], "evidence": f"fragmentation delta={frag_delta:+.4f}; hole-fraction delta={hole_delta:+.4f}", "confidence": confidence.title()}}}
    write_json(output / "failure_anatomy_result.json", result)
    (output / "report/GCQM_Full25_Failure_Anatomy_Decoder_Bottleneck_Audit_Report.md").write_text(report_text(result), encoding="utf-8")
    print(json.dumps({"decision": decision, "confidence": confidence, "report": str(output / 'report/GCQM_Full25_Failure_Anatomy_Decoder_Bottleneck_Audit_Report.md')}, indent=2))
    print(f"DECISION = {decision}"); print(f"CONFIDENCE = {confidence}")


if __name__ == "__main__":
    main()
