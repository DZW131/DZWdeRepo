#!/usr/bin/env python3
"""Frozen-E25, zero-training HQMR M1 false-component separability audit."""
from __future__ import annotations

import argparse
import inspect
import itertools
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage, stats
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.hqmr_net import HQMRNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import (
    FIXED_WEIGHTS, TTA, foreground_confusion, load_state, normalize_cam,
    prediction_from_cam, presence, resize_unflip, scores_from_confusion,
)
from tools.hqrf_phase0_io import sha256, write_csv, write_json


HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
SSHR_SHA256 = "b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70"
MORPH_RESULT_SHA256 = "8f4e7b9a3b17f5eabdf229495f239217fc130a59814e06d09f797f83dc165579"
EXPECTED_HQMR = 0.6557244403737567
EXPECTED_SSHR = 0.6669670591172749
M1_EXCESS_SHAPLEY_PP = 1.01674947160018
STRUCT8 = np.ones((3, 3), np.uint8)
EPS = 1.0e-8
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_RESAMPLES = 10_000
FAMILIES = ("A", "M", "Q")
GENERIC_FAMILIES = ("A", "M", "G")
FAMILY_FEATURES = {
    "A": ("anchor_density", "anchor_distance"),
    "M": ("mean_margin", "p10_margin", "rival_ratio"),
    "Q": ("weighted_query_support", "top5_normalized_support", "top5_support_min"),
    "G": ("log_area", "compactness", "negative_nearest_same_class_distance"),
}
# Pre-registered, semantic false direction. No component labels are consulted.
FALSE_DIRECTION = {
    "anchor_density": -1, "anchor_count": -1, "anchor_present": -1,
    "anchor_distance": 1, "anchor_missing_in_image": 1,
    "mean_target": -1, "mean_rival": 1, "mean_margin": -1,
    "median_margin": -1, "p10_margin": -1, "fraction_margin_lt_005": 1,
    "rival_ratio": -1, "normalized_margin": -1,
    "weighted_query_support": -1, "top5_weighted_support": -1,
    "top10_weighted_support": -1, "top5_normalized_support": -1,
    "top10_normalized_support": -1, "query_entropy": 1,
    "effective_query_count": 1, "top1_contribution_mass": -1,
    "top5_contribution_mass": -1, "top10_contribution_mass": -1,
    "top5_support_mean": -1, "top5_support_std": 1, "top5_support_min": -1,
    "component_area": -1, "component_area_fraction_of_image": -1,
    "log_area": -1, "perimeter": -1, "perimeter_area_ratio": 1,
    "compactness": -1, "bbox_fill_ratio": -1,
    "distance_to_largest_same_class_component": 1,
    "distance_to_nearest_same_class_component": 1,
    "negative_nearest_same_class_distance": -1,
    "component_area_rank_within_class": 1,
    "number_of_same_class_components_in_image": 1,
}
CONFIG = {
    "audit": "HQMR-v1 M1 False-Component Separability",
    "dataset": "BCSS validation", "images": 3418, "seed": 42,
    "zero_training": True, "parameter_updates": 0, "threshold_tuning": False,
    "checkpoint_selection": False, "learned_classifier": False,
    "foreground_connectivity": 8, "component_label": "overlap(R, GT_c) == 0",
    "continuous_alignment": "bilinear_align_corners_false",
    "discrete_alignment": "nearest", "tta_views": 3,
    "anchor_definition": "majority (>=2/3) of frozen tri-state reliable-positive maps",
    "anchor_presence_mask": "frozen TTA-averaged deep-gate presence",
    "sshr_anchor_definition": "same reliable-positive rule on frozen SSHR evidence",
    "percentile_scope": "all components within each predicted class; labels unused",
    "margin_diagnostic_threshold": 0.05,
    "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "m1_excess_shapley_pp": M1_EXCESS_SHAPLEY_PP,
    "oracle_is_model_performance": False,
}


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resize_nearest_unflip(value: torch.Tensor, hw: tuple[int, int], dims) -> torch.Tensor:
    resized = F.interpolate(value.float(), size=hw, mode="nearest")[0]
    return torch.flip(resized, dims=dims) if dims else resized


@torch.no_grad()
def infer_hqmr_observables(model: HQMRNet, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    full, bases, weights, gates, anchors = [], [], [], [], []
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, hqmr_mode="full")
        item = output["stages"][2]["hqmr"]
        full.append(resize_unflip(item["mixture"], original_hw, cam_flip))
        basis = item["basis"][0]
        if cam_flip:
            basis = torch.flip(basis, dims=cam_flip)
        bases.append(basis.float().cpu())
        weights.append(item["weights"][0].float().cpu())
        gates.append(output["deep_gate"].float().cpu())
        anchors.append(_resize_nearest_unflip(output["target_detail"]["positive"].float(), original_hw, cam_flip).bool().cpu())
    # Preserve exact frozen BF16 view reduction for the formal prediction.
    scores = normalize_cam(torch.stack(full).mean(0).float().cpu().numpy())
    label = presence(torch.stack(gates).mean(0).numpy()[0])
    anchor_map = (torch.stack(anchors).sum(0) >= 2).numpy()
    anchor_map &= label[:, None, None].astype(bool)
    return {
        "scores": scores, "label": label,
        "prediction": prediction_from_cam(scores, label, np.empty(original_hw)),
        "basis": torch.stack(bases).mean(0).numpy(),
        "weights": torch.stack(weights).mean(0).numpy(),
        "anchors": anchor_map,
    }


@torch.no_grad()
def infer_sshr_observables(model: SSHRCAM, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    views, probabilities = [[], [], []], []
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, c1, c2, deep, probability = model.forward_cam(value)
        for index, cam in enumerate((c1, c2, deep)):
            views[index].append(resize_unflip(cam, original_hw, cam_flip))
        probabilities.append(probability)
    normalized = [normalize_cam(torch.stack(item).mean(0).float().cpu().numpy()) for item in views]
    scores = sum(weight * item for weight, item in zip(FIXED_WEIGHTS, normalized))
    label = presence(torch.stack(probabilities).mean(0).float().cpu().numpy()[0])
    anchors = reliable_positive_from_scores(scores, label)
    return {"scores": scores, "label": label,
            "prediction": prediction_from_cam(scores, label, np.empty(original_hw)),
            "anchors": anchors}


def reliable_positive_from_scores(scores: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Apply the frozen tri-state positive contract to final normalized evidence."""
    classes, height, width = scores.shape
    take = int(math.ceil(.15 * height * width))
    result = np.zeros_like(scores, dtype=bool)
    flat = scores.reshape(classes, -1)
    order = np.argsort(-flat, axis=1, kind="stable")[:, :take]
    top = np.zeros_like(flat, dtype=bool)
    for cls in range(classes):
        top[cls, order[cls]] = True
    top = top.reshape(scores.shape)
    for cls in range(classes):
        rival = np.max(scores[[index for index in range(classes) if index != cls]], axis=0)
        result[cls] = (scores[cls] >= .60) & top[cls] & ((scores[cls] - rival) >= .10) & bool(label[cls])
    return result


def _geometry(labels: np.ndarray, component: int, image_diagonal: float) -> dict:
    region = labels == component
    area = int(region.sum())
    boundary = region & ~ndimage.binary_erosion(region, structure=STRUCT8, border_value=0)
    perimeter = int(boundary.sum())
    rows, cols = np.where(region)
    bbox_area = int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))
    other = (labels > 0) & ~region
    if other.any():
        nearest = float(ndimage.distance_transform_edt(~other)[region].min())
    else:
        nearest = image_diagonal + 1.0
    sizes = np.bincount(labels.ravel())[1:]
    largest_id = int(np.argmax(sizes) + 1) if len(sizes) else component
    if component == largest_id or len(sizes) == 1:
        distance_largest = 0.0 if component == largest_id else image_diagonal + 1.0
    else:
        distance_largest = float(ndimage.distance_transform_edt(labels != largest_id)[region].min())
    rank = int(1 + np.sum(sizes > area))
    compactness = float(4 * math.pi * area / max(perimeter * perimeter, 1))
    return {
        "component_area": area, "component_area_fraction_of_image": area / labels.size,
        "log_area": math.log1p(area), "perimeter": perimeter,
        "perimeter_area_ratio": perimeter / max(area, 1), "compactness": compactness,
        "bbox_fill_ratio": area / max(bbox_area, 1),
        "distance_to_largest_same_class_component": distance_largest,
        "distance_to_nearest_same_class_component": nearest,
        "negative_nearest_same_class_distance": -nearest,
        "component_area_rank_within_class": rank,
        "number_of_same_class_components_in_image": int(len(sizes)),
    }


def extract_component_features_no_gt(image_id: str, prediction: np.ndarray, scores: np.ndarray,
                                     anchors: np.ndarray, basis: np.ndarray | None = None,
                                     weights: np.ndarray | None = None, model: str = "hqmr") -> list[dict]:
    """Extract observable component features. This function has no GT input or access."""
    if prediction.shape != scores.shape[1:] or anchors.shape != scores.shape:
        raise ValueError("Observable map resolution mismatch")
    if (basis is None) != (weights is None):
        raise ValueError("basis and weights must be provided together")
    if basis is not None:
        if weights.shape != (basis.shape[0], scores.shape[0]):
            raise ValueError("Query basis/weight mismatch")
        aligned_basis = F.interpolate(torch.from_numpy(basis)[None].float(), size=prediction.shape,
                                      mode="bilinear", align_corners=False)[0].numpy()
    else:
        aligned_basis = None
    rows = []
    diagonal = math.hypot(*prediction.shape)
    for cls in range(scores.shape[0]):
        labels, count = ndimage.label(prediction == cls, structure=STRUCT8)
        rival = np.max(scores[[index for index in range(scores.shape[0]) if index != cls]], axis=0)
        margin = scores[cls] - rival
        anchor = anchors[cls].astype(bool)
        anchor_missing = not anchor.any()
        anchor_distance = (ndimage.distance_transform_edt(~anchor) if not anchor_missing else None)
        if aligned_basis is not None:
            order = np.argsort(-weights[:, cls], kind="stable")
        for component in range(1, count + 1):
            region = labels == component
            area = int(region.sum())
            anchor_count = int(np.sum(region & anchor))
            row = {
                "image_id": image_id, "model": model, "class_id": cls,
                "component_id": component, "component_uid": f"{image_id}_c{cls}_r{component:04d}",
                "anchor_density": anchor_count / max(area, 1), "anchor_count": anchor_count,
                "anchor_present": int(anchor_count > 0),
                "anchor_distance": float(anchor_distance[region].min()) if anchor_distance is not None else diagonal + 1.0,
                "anchor_missing_in_image": int(anchor_missing),
                "mean_target": float(scores[cls][region].mean()),
                "mean_rival": float(rival[region].mean()),
                "mean_margin": float(margin[region].mean()),
                "median_margin": float(np.median(margin[region])),
                "p10_margin": float(np.quantile(margin[region], .10)),
                "fraction_margin_lt_005": float(np.mean(margin[region] < .05)),
                "rival_ratio": float(scores[cls][region].mean() / (rival[region].mean() + EPS)),
                "normalized_margin": float((scores[cls][region].mean() - rival[region].mean()) /
                                           (scores[cls][region].mean() + rival[region].mean() + EPS)),
                **_geometry(labels, component, diagonal),
            }
            if aligned_basis is not None:
                support = aligned_basis[:, region].mean(1)
                contribution = np.clip(weights[:, cls] * support, 0, None)
                total = float(contribution.sum()) + EPS
                pi = contribution / total
                top5, top10 = order[:5], order[:10]
                row.update({
                    "weighted_query_support": float(np.sum(weights[:, cls] * support)),
                    "top5_weighted_support": float(np.sum(weights[top5, cls] * support[top5])),
                    "top10_weighted_support": float(np.sum(weights[top10, cls] * support[top10])),
                    "top5_normalized_support": float(np.sum(weights[top5, cls] * support[top5]) /
                                                     (np.sum(weights[top5, cls]) + EPS)),
                    "top10_normalized_support": float(np.sum(weights[top10, cls] * support[top10]) /
                                                      (np.sum(weights[top10, cls]) + EPS)),
                    "query_entropy": float(-np.sum(pi * np.log(np.clip(pi, EPS, None)))),
                    "effective_query_count": float(np.exp(-np.sum(pi * np.log(np.clip(pi, EPS, None))))),
                    "top1_contribution_mass": float(np.sort(pi)[-1:].sum()),
                    "top5_contribution_mass": float(np.sort(pi)[-5:].sum()),
                    "top10_contribution_mass": float(np.sort(pi)[-10:].sum()),
                    "top5_support_mean": float(support[top5].mean()),
                    "top5_support_std": float(support[top5].std()),
                    "top5_support_min": float(support[top5].min()),
                })
            rows.append(row)
    return rows


def label_components_with_gt(features: pd.DataFrame, prediction: np.ndarray, truth: np.ndarray,
                             image_id: str, model: str) -> list[dict]:
    """Post-hoc exact M1 labels; called only after no-GT features are persisted."""
    rows = []
    expected = features[(features.image_id == image_id) & (features.model == model)]
    for cls in range(4):
        labels, count = ndimage.label(prediction == cls, structure=STRUCT8)
        subset = expected[expected.class_id == cls]
        if len(subset) != count:
            raise AssertionError(f"Component identity drift: {image_id} {model} C{cls}")
        for component in range(1, count + 1):
            region = labels == component
            overlap = int(np.sum(region & (truth == cls)))
            rows.append({
                "image_id": image_id, "model": model, "class_id": cls,
                "component_id": component, "component_uid": f"{image_id}_c{cls}_r{component:04d}",
                "gt_same_class_overlap": overlap, "is_false_component": int(overlap == 0),
            })
    return rows


def setup_output(output: Path) -> None:
    for name in (
        "provenance", "components", "features", "metrics", "composites", "subtypes",
        "correlation", "sshr_generic", "visualizations/false_high_score",
        "visualizations/false_missed", "visualizations/valid_dangerous",
        "visualizations/high_burden_images", "visualizations/global_component_maps", "report",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)


def run_extract(args, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing populated output: {output}")
    setup_output(output)
    hpath, spath, morph = Path(args.hqmr_checkpoint), Path(args.sshr_checkpoint), Path(args.morphology_output)
    if sha256(hpath) != HQMR_SHA256 or sha256(spath) != SSHR_SHA256:
        raise AssertionError("Frozen checkpoint mismatch")
    morph_result = morph / "morphology_oracle_audit_result.json"
    if sha256(morph_result) != MORPH_RESULT_SHA256:
        raise AssertionError("Morphology Oracle archive mismatch")
    manifest = json.loads((morph / "cache/prediction_manifest.json").read_text())
    if len(manifest) != 3418:
        raise AssertionError("Morphology prediction cache incomplete")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {**CONFIG, "source_commit": commit, "hqmr_checkpoint": str(hpath.resolve()),
              "sshr_checkpoint": str(spath.resolve()), "morphology_output": str(morph.resolve())}
    write_json(output / "provenance/false_component_audit_config.json", config)
    (output / "provenance/false_component_audit_source_commit.txt").write_text(commit + "\n")
    (output / "provenance/false_component_audit_git_diff.patch").write_text(
        subprocess.check_output(["git", "show", "--format=", "--binary", "HEAD"], cwd=ROOT, text=True))
    write_json(output / "provenance/morphology_oracle_archive.json",
               {"path": str(morph.resolve()), "result_sha256": sha256(morph_result),
                "decision": json.loads(morph_result.read_text())["decision"]})
    loader = DataLoader(Stage1_InferDataset(str(Path(args.val_root) / "img"), img_size=224), batch_size=1,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if len(loader) != 3418:
        raise AssertionError("Expected 3418 validation images")
    hqmr = HQMRNet().cuda(); hqmr.load_state_dict(load_state(hpath), strict=True); hqmr.eval()
    sshr = SSHRCAM(4).cuda(); sshr.load_state_dict(load_state(spath), strict=True); sshr.eval()
    hqmr_rows, sshr_rows, checks = [], [], []
    by_id = {row["image_id"]: row for row in manifest}
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]
        original = Image.open(Path(args.val_root) / "img" / f"{image_id}.png")
        image = image.cuda(non_blocking=True)
        h = infer_hqmr_observables(hqmr, image, (original.height, original.width))
        s = infer_sshr_observables(sshr, image, (original.height, original.width))
        frozen = np.load(morph / "cache" / by_id[image_id]["cache"])
        hmatch = bool(np.array_equal(h["prediction"], frozen["hqmr"]))
        smatch = bool(np.array_equal(s["prediction"], frozen["sshr"]))
        if not hmatch or not smatch:
            raise AssertionError(
                f"Frozen prediction-map mismatch at {image_id}: "
                f"hqmr_match={hmatch} hqmr_pixels={int(np.sum(h['prediction'] != frozen['hqmr']))} "
                f"sshr_match={smatch} sshr_pixels={int(np.sum(s['prediction'] != frozen['sshr']))}"
            )
        hqmr_rows.extend(extract_component_features_no_gt(
            image_id, h["prediction"], h["scores"], h["anchors"], h["basis"], h["weights"], "hqmr"))
        sshr_rows.extend(extract_component_features_no_gt(
            image_id, s["prediction"], s["scores"], s["anchors"], model="sshr"))
        checks.append({"image_id": image_id, "hqmr_prediction_exact_match": hmatch,
                       "sshr_prediction_exact_match": smatch})
        if index % 50 == 0 or index == len(loader):
            print(f"FALSE_COMPONENT_EXTRACT_PROGRESS={index}/{len(loader)} HQMR_COMPONENTS={len(hqmr_rows)}", flush=True)
    hqmr_df, sshr_df = pd.DataFrame(hqmr_rows), pd.DataFrame(sshr_rows)
    hqmr_df.to_parquet(output / "components/hqmr_component_features_no_gt.parquet", index=False)
    sshr_df.to_parquet(output / "sshr_generic/sshr_component_generic_features.parquet", index=False)
    hqmr_df[["image_id", "class_id", "component_id", "component_uid", "component_area"]].to_csv(
        output / "components/hqmr_components_index.csv", index=False)
    sshr_df.to_csv(output / "sshr_generic/sshr_component_generic_features.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "provenance/prediction_map_reproduction.csv", index=False)
    write_json(output / "provenance/no_gt_feature_contract.json", {
        "status": "PASS", "extractor_signature": str(inspect.signature(extract_component_features_no_gt)),
        "gt_parameter_present": "truth" in inspect.signature(extract_component_features_no_gt).parameters,
        "hqmr_components": len(hqmr_df), "sshr_components": len(sshr_df),
        "all_prediction_maps_exact_match": True,
    })
    print(f"NO_GT_FEATURE_EXTRACTION=PASS HQMR_COMPONENTS={len(hqmr_df)} SSHR_COMPONENTS={len(sshr_df)}")


def weighted_metrics(y, score, weight=None) -> dict:
    y, score = np.asarray(y, int), np.asarray(score, float)
    weight = None if weight is None else np.asarray(weight, float)
    if len(np.unique(y)) < 2:
        return {"AUROC": None, "AUPRC": None, "n": int(len(y))}
    return {"AUROC": float(roc_auc_score(y, score, sample_weight=weight)),
            "AUPRC": float(average_precision_score(y, score, sample_weight=weight)), "n": int(len(y))}


def percentile_scores(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for feature in features:
        values = FALSE_DIRECTION[feature] * result[feature].astype(float)
        rank = values.groupby(result.class_id).rank(method="average")
        count = values.groupby(result.class_id).transform("count")
        result[f"false_rank_{feature}"] = np.where(count > 1, (rank - 1) / (count - 1), .5)
    return result


def add_family_scores(frame: pd.DataFrame, families) -> pd.DataFrame:
    features = sorted({feature for family in families for feature in FAMILY_FEATURES[family]})
    result = percentile_scores(frame, features)
    for family in families:
        result[f"score_{family}"] = result[[f"false_rank_{feature}" for feature in FAMILY_FEATURES[family]]].mean(axis=1)
    return result


def operating_point(y, score, area, limit: float, weighted: bool) -> dict:
    y, score, area = np.asarray(y, int), np.asarray(score, float), np.asarray(area, float)
    order = np.argsort(-score, kind="stable")
    y, score, area = y[order], score[order], area[order]
    false_total = np.sum(area[y == 1]) if weighted else np.sum(y == 1)
    valid_total = np.sum(area[y == 0]) if weighted else np.sum(y == 0)
    false_cum = valid_cum = 0.0
    best = {"threshold": None, "false_recall": 0.0, "valid_suppression": 0.0}
    for _, group in itertools.groupby(range(len(score)), key=lambda index: score[index]):
        indices = list(group)
        group_weight = area[indices] if weighted else np.ones(len(indices))
        false_cum += float(np.sum(group_weight[np.asarray(y[indices]) == 1]))
        valid_cum += float(np.sum(group_weight[np.asarray(y[indices]) == 0]))
        valid_rate = valid_cum / max(float(valid_total), EPS)
        false_rate = false_cum / max(float(false_total), EPS)
        if valid_rate <= limit + 1e-12 and false_rate >= best["false_recall"]:
            best = {"threshold": float(score[indices[-1]]), "false_recall": false_rate,
                    "valid_suppression": valid_rate}
    return best


def exact_shapley_3(values: dict[int, float]) -> dict[str, float]:
    result = {}
    for index, name in enumerate(FAMILIES):
        total = 0.0
        for subset in range(8):
            if subset & (1 << index):
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
            total += weight * (values[subset | (1 << index)] - values[subset])
        result[name] = float(total)
    return result


def cliffs_delta(false, valid) -> float:
    false, valid = np.asarray(false), np.asarray(valid)
    if not len(false) or not len(valid):
        return float("nan")
    statistic = stats.mannwhitneyu(false, valid, alternative="two-sided").statistic
    return float(2 * statistic / (len(false) * len(valid)) - 1)


def bootstrap_correlations(frame: pd.DataFrame, features: list[str]) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    point, samples = [], {feature: [] for feature in features}
    target = frame["DeltaIoU"].to_numpy(float)
    for feature in features:
        point.append({"feature": feature, "spearman_rho": float(stats.spearmanr(frame[feature], target).statistic)})
    count = len(frame)
    for _ in range(BOOTSTRAP_RESAMPLES):
        chosen = rng.integers(0, count, count)
        for feature in features:
            samples[feature].append(stats.spearmanr(frame[feature].to_numpy()[chosen], target[chosen]).statistic)
    summary = {feature: {"rho": next(row["spearman_rho"] for row in point if row["feature"] == feature),
                         "ci95": [float(np.nanquantile(samples[feature], .025)),
                                  float(np.nanquantile(samples[feature], .975))]}
               for feature in features}
    return point, summary


def _metrics_rows(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    individual, area, per_class = [], [], []
    y, weights = frame.is_false_component, frame.component_area
    for feature in feature_columns:
        score = FALSE_DIRECTION[feature] * frame[feature]
        individual.append({"feature": feature, **weighted_metrics(y, score)})
        area.append({"feature": feature, **weighted_metrics(y, score, weights)})
        for cls, group in frame.groupby("class_id"):
            current = weighted_metrics(group.is_false_component,
                                       FALSE_DIRECTION[feature] * group[feature])
            weighted = weighted_metrics(group.is_false_component,
                                        FALSE_DIRECTION[feature] * group[feature], group.component_area)
            per_class.append({"feature": feature, "class_id": int(cls), **current,
                              "area_weighted_AUROC": weighted["AUROC"],
                              "false_component_prevalence": float(group.is_false_component.mean()),
                              "false_area_prevalence": float(group.loc[group.is_false_component == 1, "component_area"].sum() /
                                                             max(group.component_area.sum(), 1))})
    return individual, area, per_class


def _family_subset_rows(frame: pd.DataFrame) -> tuple[list[dict], dict, dict]:
    rows, values, area_values = [], {0: .5}, {0: .5}
    for subset in range(1, 8):
        names = [FAMILIES[index] for index in range(3) if subset & (1 << index)]
        score = frame[[f"score_{name}" for name in names]].mean(axis=1)
        metric = weighted_metrics(frame.is_false_component, score)
        area_metric = weighted_metrics(frame.is_false_component, score, frame.component_area)
        values[subset], area_values[subset] = metric["AUROC"], area_metric["AUROC"]
        rows.append({"subset": subset, "families": "+".join(names), **metric,
                     "area_weighted_AUROC": area_metric["AUROC"], "area_weighted_AUPRC": area_metric["AUPRC"]})
    return rows, exact_shapley_3(values), exact_shapley_3(area_values)


def _decision(amq, amq_area, area_recall, class_aurocs, shapley, geometry_auc) -> tuple[str, str, dict]:
    classes80 = sum(value is not None and value >= .80 for value in class_aurocs.values())
    classes75 = sum(value is not None and value >= .75 for value in class_aurocs.values())
    meaningful = sum(value >= .05 for value in shapley.values())
    shortcut = amq < .75 and geometry_auc >= .85
    strong = amq >= .85 and amq_area >= .85 and area_recall >= .60 and classes80 >= 3 and meaningful >= 2 and not shortcut
    if strong:
        decision, confidence = "FALSE_COMPONENT_SEPARABLE", "HIGH"
    elif amq >= .85 and amq_area >= .85 and area_recall >= .60 and classes75 < 2:
        decision, confidence = "CLASS_DEPENDENT_COMPONENT_SEPARABILITY", "MEDIUM"
    elif shortcut:
        decision, confidence = "GEOMETRY_SHORTCUT_ONLY", "MEDIUM"
    elif amq >= .75 and amq_area >= .75 and area_recall >= .35:
        decision, confidence = "PARTIAL_FALSE_COMPONENT_SEPARABILITY", "MEDIUM"
    else:
        decision, confidence = "FALSE_COMPONENT_NOT_SEPARABLE", "HIGH" if amq < .70 or area_recall < .25 else "MEDIUM"
    return decision, confidence, {"AMQ_component_AUROC": amq, "AMQ_area_weighted_AUROC": amq_area,
                                  "false_area_recall_at_5pct": area_recall, "classes_AUROC_ge_080": classes80,
                                  "classes_AUROC_ge_075": classes75, "meaningful_AMQ_families": meaningful,
                                  "geometry_only_AUROC": geometry_auc, "GEOMETRY_SHORTCUT_ONLY": shortcut}


def _dominant(shapley: dict[str, float]) -> str:
    ordered = sorted(shapley, key=shapley.get, reverse=True)
    if shapley[ordered[0]] < .05:
        return "MIXED"
    if shapley[ordered[0]] - shapley[ordered[1]] < .02:
        return "MIXED"
    return {"A": "WEAK_ANCHOR", "M": "SEMANTIC_MARGIN", "Q": "QUERY_SUPPORT"}[ordered[0]]


def report_text(result: dict) -> str:
    d, amq, matrix = result["decision"], result["amq"], result["decision_matrix"]
    table = "\n".join([
        "| Evidence | Result | Gate |", "|---|---:|---|",
        f"| AMQ component AUROC | {amq['component_AUROC']:.4f} | 0.85 strong |",
        f"| AMQ area-weighted AUROC | {amq['area_weighted_AUROC']:.4f} | 0.85 strong |",
        f"| False-area recall @5% valid-area suppression | {amq['false_area_recall_at_5pct']:.4f} | 0.60 strong |",
        f"| Classes with AUROC >=0.80 | {matrix['classes_AUROC_ge_080']} | >=3 |",
        f"| Anchor Shapley | {result['shapley']['component']['A']:+.4f} | meaningful >=0.05 |",
        f"| Margin Shapley | {result['shapley']['component']['M']:+.4f} | meaningful >=0.05 |",
        f"| Query Shapley | {result['shapley']['component']['Q']:+.4f} | meaningful >=0.05 |",
        f"| Geometry-only AUROC | {result['amqg']['geometry_only_AUROC']:.4f} | shortcut check |",
        f"| Accessible M1 headroom proxy | {result['accessible_m1_headroom_proxy_pp']:+.4f} pp | descriptive only |",
    ])
    sections = [
        ("Executive Diagnosis", f"**DECISION = {d['decision']}**；**CONFIDENCE = {d['confidence']}**。"),
        ("Frozen Morphology-Oracle Evidence", f"M1 excess Shapley={M1_EXCESS_SHAPLEY_PP:+.5f} pp；full excess={result['morphology_evidence']['full_excess_pp']:+.4f} pp。"),
        ("Why M1 Is the Exact Target", "M1 accounts for nearly all HQMR-specific morphology headroom; M2 remains secondary and is not combined into a future design here."),
        ("Reproduction Gate", str(result["reproduction_gate"])),
        ("Exact M1 Component Definition", "8-connected predicted component R is FALSE iff |R∩GT_c|=0; no size, IoU, or overlap-ratio threshold."),
        ("No-GT Feature Contract", str(result["no_gt_contract"])),
        ("Component Population", str(result["population"])),
        ("Weak Anchor Features", str(result["anchor_analysis"])),
        ("Semantic Margin Features", str(result["margin_analysis"])),
        ("Query-Support Features", str(result["query_analysis"])),
        ("Geometry Features", "Geometry is diagnostic-only and cannot authorize size-based deletion."),
        ("Individual Feature Separability", "See metrics/individual_feature_auroc.csv and metrics/area_weighted_metrics.csv."),
        ("Per-Class Separability", str(result["per_class_amq"])),
        ("Area-Weighted Separability", f"AMQ area-weighted AUROC={amq['area_weighted_AUROC']:.4f}."),
        ("AMQ Composite", str(amq)),
        ("AMQG Geometry Shortcut Check", str(result["amqg"])),
        ("A/M/Q Subset Analysis", "All seven pre-registered equal-weight subsets are in composites/all_amq_subsets.csv; none is treated as model selection."),
        ("Exact A/M/Q Shapley", str(result["shapley"])),
        ("Operating-Point Safety", str(result["operating_points_component"])),
        ("False-Area Recall at Valid-Area Constraints", str(result["operating_points_area"])),
        ("Accessible M1 Headroom Proxy", f"{result['accessible_m1_headroom_proxy_pp']:+.4f} pp. **THIS IS NOT MODEL PERFORMANCE.**"),
        ("False-Component Subtypes", str(result["subtypes"])),
        ("HQMR vs SSHR Generic-Signal Comparison", str(result["generic_comparison"])),
        ("Image-Level Burden Correlation", str(result["correlations"])),
        ("Representative Cases", "Five frozen, automatic groups are listed in visualizations/selection.json and rendered without manual cherry-picking."),
        ("Failure Cases", "False components mis-ranked valid and valid components dangerously ranked false are explicitly visualized."),
        ("Decision Matrix", table),
        ("Exact Separability Decision", f"`DECISION = {d['decision']}`\n\n`CONFIDENCE = {d['confidence']}`"),
        ("Dominant Observable Signal", result["dominant_signal"]),
        ("What Is Preserved", "CCRA, HQMR-v1, H5→H4 hierarchical reconstruction, region-conditioned query update, and the M1 morphology-oracle conclusion."),
        ("What Is Falsified", d["falsified"]),
        ("Exact Next Architecture Target", d["authorized_target"]),
        ("What Must NOT Be Done", "No size-based deletion, GT morphology, opening/closing, CRF, CH restoration, pixel-level CCBP, or HQMR decoder change."),
        ("Final Decision", f"`DECISION = {d['decision']}`\n\n`CONFIDENCE = {d['confidence']}`\n\n**DO NOT TRAIN A NEW MODEL.**"),
    ]
    sentence = (f"Using only GT-free observable signals from frozen HQMR-v1, M1 false components achieve a component-level AUROC of "
                f"{amq['component_AUROC']:.4f}, an area-weighted AUROC of {amq['area_weighted_AUROC']:.4f}, and a false-area recall of "
                f"{amq['false_area_recall_at_5pct']:.4f} at 5% valid-area suppression. The dominant non-GT signal is "
                f"{result['dominant_signal']}. Therefore region-level validation {'is' if d['decision'] == 'FALSE_COMPONENT_SEPARABLE' else 'is not'} "
                f"sufficiently separable to justify the next model.")
    return "# HQMR-v1 M1 False-Component Separability Audit Report\n\n" + "\n\n".join(
        f"## {index} {title}\n\n{body}" for index, (title, body) in enumerate(sections, 1)) + \
        "\n\n## Required Completion Sentence\n\n> **" + sentence + "**\n"


def run_analyze(args, output: Path) -> None:
    config = json.loads((output / "provenance/false_component_audit_config.json").read_text())
    if config["source_commit"] != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip():
        raise AssertionError("Audit source changed after no-GT feature freeze")
    hqmr_features = pd.read_parquet(output / "components/hqmr_component_features_no_gt.parquet")
    sshr_features = pd.read_parquet(output / "sshr_generic/sshr_component_generic_features.parquet")
    morph = Path(args.morphology_output)
    manifest = json.loads((morph / "cache/prediction_manifest.json").read_text())
    labels, hhist, shist, per_image_metrics = [], np.zeros((4, 4), np.int64), np.zeros((4, 4), np.int64), []
    by_id = {row["image_id"]: row for row in manifest}
    for index, row in enumerate(manifest, 1):
        image_id = row["image_id"]
        truth = np.asarray(Image.open(Path(args.val_root) / "mask" / f"{image_id}.png"))
        data = np.load(morph / "cache" / row["cache"])
        for model, features, prediction in (("hqmr", hqmr_features, data["hqmr"]),
                                             ("sshr", sshr_features, data["sshr"])):
            labels.extend(label_components_with_gt(features, prediction, truth, image_id, model))
        hh, ss = foreground_confusion(truth, data["hqmr"]), foreground_confusion(truth, data["sshr"])
        hhist += hh; shist += ss
        per_image_metrics.append({"image_id": image_id,
                                  "DeltaIoU": scores_from_confusion(hh)["mIoU"] - scores_from_confusion(ss)["mIoU"]})
        if index % 200 == 0 or index == len(manifest):
            print(f"FALSE_COMPONENT_LABEL_PROGRESS={index}/{len(manifest)}", flush=True)
    label_df = pd.DataFrame(labels)
    hqmr_labels = label_df[label_df.model == "hqmr"].drop(columns="model")
    sshr_labels = label_df[label_df.model == "sshr"].drop(columns="model")
    hqmr_labels.to_csv(output / "components/hqmr_component_labels_gt.csv", index=False)
    sshr_labels.to_csv(output / "sshr_generic/sshr_component_labels_gt.csv", index=False)
    hqmr = hqmr_features.merge(hqmr_labels, on=["image_id", "class_id", "component_id", "component_uid"], validate="one_to_one")
    sshr = sshr_features.merge(sshr_labels, on=["image_id", "class_id", "component_id", "component_uid"], validate="one_to_one")
    hqmr = add_family_scores(hqmr, ("A", "M", "Q", "G"))
    sshr = add_family_scores(sshr, GENERIC_FAMILIES)
    hqmr["S_AMQ"] = hqmr[["score_A", "score_M", "score_Q"]].mean(axis=1)
    hqmr["S_AMQG"] = hqmr[["score_A", "score_M", "score_Q", "score_G"]].mean(axis=1)
    sshr["S_AMG"] = sshr[["score_A", "score_M", "score_G"]].mean(axis=1)
    hqmr.to_parquet(output / "components/hqmr_component_joined_for_analysis.parquet", index=False)
    # No-GT family tables intentionally contain no oracle label columns.
    id_columns = ["image_id", "class_id", "component_id", "component_uid"]
    hqmr_features[id_columns + list(FALSE_DIRECTION)].to_csv(output / "features/all_observable_features.csv", index=False)
    for family, name in (("A", "anchor_features"), ("M", "margin_features"),
                         ("Q", "query_features"), ("G", "geometry_features")):
        hqmr_features[id_columns + list(FAMILY_FEATURES[family])].to_csv(output / f"features/{name}.csv", index=False)
    feature_columns = [feature for feature in FALSE_DIRECTION if feature in hqmr]
    individual, area_metrics, per_class_metrics = _metrics_rows(hqmr, feature_columns)
    write_csv(output / "metrics/individual_feature_auroc.csv", individual)
    write_csv(output / "metrics/area_weighted_metrics.csv", area_metrics)
    write_csv(output / "metrics/per_class_feature_metrics.csv", per_class_metrics)
    subset_rows, shapley, area_shapley = _family_subset_rows(hqmr)
    write_csv(output / "composites/all_amq_subsets.csv", subset_rows)
    write_csv(output / "composites/amq_shapley.csv", [{"family": key, "shapley": value} for key, value in shapley.items()])
    write_csv(output / "composites/amq_area_shapley.csv", [{"family": key, "shapley": value} for key, value in area_shapley.items()])
    amq_metric = weighted_metrics(hqmr.is_false_component, hqmr.S_AMQ)
    amq_area = weighted_metrics(hqmr.is_false_component, hqmr.S_AMQ, hqmr.component_area)
    amqg_metric = weighted_metrics(hqmr.is_false_component, hqmr.S_AMQG)
    geometry_metric = weighted_metrics(hqmr.is_false_component, hqmr.score_G)
    operating_rows, component_ops, area_ops = [], {}, {}
    for limit in (.01, .02, .05):
        component_ops[str(limit)] = operating_point(hqmr.is_false_component, hqmr.S_AMQ, hqmr.component_area, limit, False)
        area_ops[str(limit)] = operating_point(hqmr.is_false_component, hqmr.S_AMQ, hqmr.component_area, limit, True)
        operating_rows.append({"population": "component", "valid_suppression_limit": limit, **component_ops[str(limit)]})
        operating_rows.append({"population": "area", "valid_suppression_limit": limit, **area_ops[str(limit)]})
    write_csv(output / "metrics/operating_points.csv", operating_rows)
    per_class_amq, per_class_safety = {}, []
    for cls, group in hqmr.groupby("class_id"):
        metric = weighted_metrics(group.is_false_component, group.S_AMQ)
        safety = operating_point(group.is_false_component, group.S_AMQ, group.component_area, .05, True)
        per_class_amq[str(int(cls))] = metric["AUROC"]
        per_class_safety.append({"class_id": int(cls), "component_AUROC": metric["AUROC"], **safety})
    write_csv(output / "metrics/per_class_operating_points.csv", per_class_safety)
    decision, confidence, matrix = _decision(amq_metric["AUROC"], amq_area["AUROC"],
                                              area_ops["0.05"]["false_recall"], per_class_amq,
                                              shapley, geometry_metric["AUROC"])
    dominant = _dominant(shapley)
    authorization = {
        "QUERY_SUPPORT": "Query-Guided Region Validation (QGRV)",
        "WEAK_ANCHOR": "Anchor-Supported Region Validation (ASRV)",
        "SEMANTIC_MARGIN": "Region-Level Class Competition / Margin Validation",
        "MIXED": "Multi-Signal Region Validation",
    }[dominant]
    authorized_target = authorization if decision == "FALSE_COMPONENT_SEPARABLE" else "No region suppressor is authorized by this audit."
    false = hqmr[hqmr.is_false_component == 1]
    valid = hqmr[hqmr.is_false_component == 0]
    margin_analysis = {feature: {"false_mean": float(false[feature].mean()), "valid_mean": float(valid[feature].mean()),
                                 "AUROC": next(row["AUROC"] for row in individual if row["feature"] == feature),
                                 "cliffs_delta_false_vs_valid": cliffs_delta(false[feature], valid[feature]),
                                 "mann_whitney_p": float(stats.mannwhitneyu(false[feature], valid[feature]).pvalue)}
                       for feature in ("mean_margin", "p10_margin", "rival_ratio")}
    anchor_analysis = {"false_anchor_present_rate": float(false.anchor_present.mean()),
                       "valid_anchor_present_rate": float(valid.anchor_present.mean()),
                       "false_anchor_density_mean": float(false.anchor_density.mean()),
                       "valid_anchor_density_mean": float(valid.anchor_density.mean()),
                       "false_anchor_distance_mean": float(false.anchor_distance.mean()),
                       "valid_anchor_distance_mean": float(valid.anchor_distance.mean())}
    query_analysis = {feature: {"false_mean": float(false[feature].mean()), "valid_mean": float(valid[feature].mean()),
                                "AUROC": next(row["AUROC"] for row in individual if row["feature"] == feature)}
                      for feature in ("weighted_query_support", "top5_support_min", "effective_query_count", "top5_contribution_mass")}
    median_margin = float(hqmr.mean_margin.median())
    false = false.copy()
    false["subtype"] = np.select([
        (false.anchor_present == 0) & (false.mean_margin < median_margin),
        (false.anchor_present == 0) & (false.mean_margin >= median_margin),
        (false.anchor_present == 1) & (false.mean_margin < median_margin),
    ], ["F1_no_anchor_low_margin", "F2_no_anchor_high_margin", "F3_anchored_low_margin"],
       default="F4_anchored_high_margin")
    subtype_rows = []
    for (subtype, cls), group in false.groupby(["subtype", "class_id"]):
        subtype_rows.append({"subtype": subtype, "class_id": int(cls), "count": len(group),
                             "area": int(group.component_area.sum()),
                             "M1_area_contribution": float(group.component_area.sum() / max(false.component_area.sum(), 1))})
    write_csv(output / "subtypes/false_component_subtypes.csv", subtype_rows)
    cutoff = float(hqmr.S_AMQ.quantile(.80))
    high_slice = hqmr[(hqmr.anchor_present == 0) & (hqmr.S_AMQ >= cutoff)]
    high_slice_result = {"cutoff": cutoff, "count": len(high_slice),
                         "false_precision": float(high_slice.is_false_component.mean()) if len(high_slice) else None,
                         "false_area_precision": float(high_slice.loc[high_slice.is_false_component == 1, "component_area"].sum() /
                                                       max(high_slice.component_area.sum(), 1)),
                         "class_distribution": high_slice.class_id.value_counts(normalize=True).sort_index().to_dict()}
    write_json(output / "subtypes/high_confidence_false_slice.csv.json", json_ready(high_slice_result))
    high_slice.to_csv(output / "subtypes/high_confidence_false_slice.csv", index=False)
    generic_rows = []
    for model, frame in (("hqmr", hqmr), ("sshr", sshr)):
        for family in GENERIC_FAMILIES:
            metric = weighted_metrics(frame.is_false_component, frame[f"score_{family}"])
            weighted = weighted_metrics(frame.is_false_component, frame[f"score_{family}"], frame.component_area)
            generic_rows.append({"model": model, "family": family, **metric,
                                 "area_weighted_AUROC": weighted["AUROC"], "area_weighted_AUPRC": weighted["AUPRC"]})
    write_csv(output / "sshr_generic/hqmr_vs_sshr_generic_comparison.csv", generic_rows)
    image_metrics = pd.DataFrame(per_image_metrics)
    burden_rows = []
    for image_id, group in hqmr.groupby("image_id"):
        false_group = group[group.is_false_component == 1]
        first = group.iloc[0]
        image_pixels = float(first.component_area / max(first.component_area_fraction_of_image, EPS))
        burden_rows.append({"image_id": image_id, "false_component_count": len(false_group),
                            "false_area": int(false_group.component_area.sum()),
                            "false_area_over_image": float(false_group.component_area.sum() / max(image_pixels, 1)),
                            "false_area_over_predicted_foreground": float(false_group.component_area.sum() / max(group.component_area.sum(), 1)),
                            "mean_AMQ_score": float(group.S_AMQ.mean()), "max_AMQ_score": float(group.S_AMQ.max()),
                            "no_anchor_false_area": int(false_group.loc[false_group.anchor_present == 0, "component_area"].sum()),
                            "low_query_support_false_area": int(false_group.loc[false_group.score_Q >= .5, "component_area"].sum())})
    burden = pd.DataFrame(burden_rows).merge(image_metrics, on="image_id", validate="one_to_one")
    burden.to_csv(output / "correlation/per_image_false_burden.csv", index=False)
    correlation_features = ["false_area", "mean_AMQ_score", "no_anchor_false_area", "low_query_support_false_area"]
    correlation_rows, bootstrap = bootstrap_correlations(burden, correlation_features)
    write_csv(output / "correlation/false_burden_correlations.csv", correlation_rows)
    write_json(output / "correlation/bootstrap.json", json_ready(bootstrap))
    selections = {
        "high_burden_images": burden.nlargest(5, "false_area").image_id.tolist(),
        "false_high_score": false.nlargest(5, "S_AMQ").component_uid.tolist(),
        "false_missed": false.nsmallest(5, "S_AMQ").component_uid.tolist(),
        "valid_dangerous": valid.nlargest(5, "S_AMQ").component_uid.tolist(),
        "largest_hqmr_losses": burden.nsmallest(5, "DeltaIoU").image_id.tolist(),
    }
    write_json(output / "visualizations/selection.json", selections)
    population = {"hqmr_components": len(hqmr), "false_components": int(hqmr.is_false_component.sum()),
                  "valid_components": int((hqmr.is_false_component == 0).sum()),
                  "false_component_prevalence": float(hqmr.is_false_component.mean()),
                  "false_area_prevalence": float(hqmr.loc[hqmr.is_false_component == 1, "component_area"].sum() /
                                                 max(hqmr.component_area.sum(), 1)),
                  "per_class": {str(int(cls)): {"components": len(group), "false_components": int(group.is_false_component.sum()),
                                                "false_prevalence": float(group.is_false_component.mean())}
                                for cls, group in hqmr.groupby("class_id")}}
    morph_result = json.loads((Path(args.morphology_output) / "morphology_oracle_audit_result.json").read_text())
    reproduction = {"status": "PASS", "paired_images": 3418,
                    "hqmr_mIoU": scores_from_confusion(hhist)["mIoU"],
                    "sshr_mIoU": scores_from_confusion(shist)["mIoU"],
                    "prediction_maps_exact_match": True, "same_split": True, "same_class_mapping": True,
                    "same_tta": True, "same_threshold_protocol": True, "same_ignore_policy": True,
                    "same_interpolation": True}
    if abs(reproduction["hqmr_mIoU"] - EXPECTED_HQMR) > 1e-12 or abs(reproduction["sshr_mIoU"] - EXPECTED_SSHR) > 1e-12:
        raise AssertionError("FALSE_COMPONENT_AUDIT_ENGINEERING_BLOCKED")
    write_json(output / "provenance/reproduction_gate.json", reproduction)
    write_json(output / "composites/amq_metrics.json", {**amq_metric, "area_weighted": amq_area,
                                                          "per_class": per_class_amq, "operating_points": area_ops})
    write_json(output / "composites/amqg_metrics.json", {"AMQG": amqg_metric, "G_only": geometry_metric})
    result = json_ready({
        "decision": {"decision": decision, "confidence": confidence, "authorized_target": authorized_target,
                     "falsified": ("The existing frozen signals do not safely identify M1 false components."
                                    if decision != "FALSE_COMPONENT_SEPARABLE"
                                    else "The hypothesis that M1 islands are observationally indistinguishable is falsified.")},
        "reproduction_gate": reproduction, "no_gt_contract": json.loads((output / "provenance/no_gt_feature_contract.json").read_text()),
        "morphology_evidence": {"M1_excess_shapley_pp": M1_EXCESS_SHAPLEY_PP,
                                "full_excess_pp": morph_result["headroom"]["excess_headroom_pp"]},
        "population": population, "anchor_analysis": anchor_analysis, "margin_analysis": margin_analysis,
        "query_analysis": query_analysis, "amq": {"component_AUROC": amq_metric["AUROC"],
                                                    "component_AUPRC": amq_metric["AUPRC"],
                                                    "area_weighted_AUROC": amq_area["AUROC"],
                                                    "area_weighted_AUPRC": amq_area["AUPRC"],
                                                    "false_area_recall_at_5pct": area_ops["0.05"]["false_recall"],
                                                    "valid_area_suppression": area_ops["0.05"]["valid_suppression"]},
        "amqg": {"AMQG_AUROC": amqg_metric["AUROC"], "geometry_only_AUROC": geometry_metric["AUROC"],
                  "GEOMETRY_SHORTCUT_ONLY": matrix["GEOMETRY_SHORTCUT_ONLY"]},
        "per_class_amq": per_class_amq, "per_class_safety": per_class_safety,
        "shapley": {"component": shapley, "area_weighted": area_shapley},
        "operating_points_component": component_ops, "operating_points_area": area_ops,
        "accessible_m1_headroom_proxy_pp": M1_EXCESS_SHAPLEY_PP * area_ops["0.05"]["false_recall"],
        "subtypes": subtype_rows, "high_confidence_slice": high_slice_result,
        "generic_comparison": generic_rows, "correlations": bootstrap,
        "decision_matrix": matrix, "dominant_signal": dominant,
        "source_commit": config["source_commit"], "training_performed": False,
        "oracle_is_model_performance": False,
    })
    write_json(output / "false_component_separability_audit_result.json", result)
    report = output / "report/HQMR_v1_M1_False_Component_Separability_Audit_Report.md"
    report.write_text(report_text(result), encoding="utf-8")
    print(json.dumps({"decision": decision, "confidence": confidence, "amq": result["amq"],
                      "dominant_signal": dominant, "authorized_target": authorized_target}, indent=2))
    print(f"DECISION = {decision}")
    print(f"CONFIDENCE = {confidence}")


def _parse_component_uid(uid: str) -> tuple[str, int, int]:
    prefix, component = uid.rsplit("_r", 1)
    image_id, cls = prefix.rsplit("_c", 1)
    return image_id, int(cls), int(component)


@torch.no_grad()
def run_visualize(args, output: Path) -> None:
    selections = json.loads((output / "visualizations/selection.json").read_text())
    joined = pd.read_parquet(output / "components/hqmr_component_joined_for_analysis.parquet")
    component_destinations, image_destinations = defaultdict(list), defaultdict(list)
    for group in ("false_high_score", "false_missed", "valid_dangerous"):
        for rank, uid in enumerate(selections[group], 1):
            image_id, _, _ = _parse_component_uid(uid)
            component_destinations[image_id].append((group, rank, uid))
    for group in ("high_burden_images", "largest_hqmr_losses"):
        target = "high_burden_images" if group == "high_burden_images" else "global_component_maps"
        for rank, image_id in enumerate(selections[group], 1):
            image_destinations[image_id].append((target, rank))
    wanted = set(component_destinations) | set(image_destinations)
    hqmr = HQMRNet().cuda(); hqmr.load_state_dict(load_state(Path(args.hqmr_checkpoint)), strict=True); hqmr.eval()
    loader = DataLoader(Stage1_InferDataset(str(Path(args.val_root) / "img"), img_size=224), batch_size=1,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    for names, image in loader:
        image_id = names[0]
        if image_id not in wanted:
            continue
        original = np.asarray(Image.open(Path(args.val_root) / "img" / f"{image_id}.png").convert("RGB"))
        truth = np.asarray(Image.open(Path(args.val_root) / "mask" / f"{image_id}.png"))
        h = infer_hqmr_observables(hqmr, image.cuda(non_blocking=True), truth.shape)
        aligned_basis = F.interpolate(torch.from_numpy(h["basis"])[None].float(), size=truth.shape,
                                      mode="bilinear", align_corners=False)[0].numpy()
        if image_id in component_destinations:
            for group, rank, uid in component_destinations[image_id]:
                _, cls, component = _parse_component_uid(uid)
                labels, _ = ndimage.label(h["prediction"] == cls, structure=STRUCT8)
                region = labels == component
                rows, cols = np.where(region); pad = 12
                r0, r1 = max(0, rows.min()-pad), min(region.shape[0], rows.max()+pad+1)
                c0, c1 = max(0, cols.min()-pad), min(region.shape[1], cols.max()+pad+1)
                sl = np.s_[r0:r1, c0:c1]
                rival = np.max(h["scores"][[index for index in range(4) if index != cls]], axis=0)
                order = np.argsort(-h["weights"][:, cls], kind="stable")[:5]
                record = joined[joined.component_uid == uid].iloc[0]
                panels = [(original[sl], "input crop", None), (truth[sl], "GT annotation", "tab10"),
                          (h["prediction"][sl], "HQMR prediction", "tab10"), (region[sl], "component mask", "Reds"),
                          (h["anchors"][cls][sl], "weak-positive anchor", "Greens"),
                          (h["scores"][cls][sl], f"F_c (C{cls})", "magma"), (rival[sl], "max-rival F", "magma"),
                          ((h["scores"][cls]-rival)[sl], "margin", "coolwarm")]
                panels += [(aligned_basis[qid][sl], f"top query B{qid}", "viridis") for qid in order]
                panels += [(np.full(region[sl].shape, record.S_AMQ), f"AMQ={record.S_AMQ:.3f}", "magma"),
                           (truth[sl] == cls, "FALSE" if record.is_false_component else "VALID", "gray")]
                fig, axes = plt.subplots(3, 5, figsize=(17, 10))
                for axis, (value, title, cmap) in zip(axes.flat, panels):
                    axis.imshow(value, cmap=cmap); axis.set_title(title); axis.axis("off")
                fig.suptitle(uid); fig.tight_layout()
                fig.savefig(output / f"visualizations/{group}/{rank:02d}_{uid}.png", dpi=140)
                plt.close(fig)
        if image_id in image_destinations:
            image_rows = joined[joined.image_id == image_id]
            false_map = np.zeros_like(truth, float); amq_map = np.zeros_like(truth, float)
            for record in image_rows.itertuples():
                labels, _ = ndimage.label(h["prediction"] == record.class_id, structure=STRUCT8)
                region = labels == record.component_id
                if record.is_false_component:
                    false_map[region] = record.class_id + 1
                amq_map[region] = record.S_AMQ
            panels = [(original, "input", None), (truth, "GT", "tab10"), (h["prediction"], "HQMR", "tab10"),
                      (false_map, "M1 false components", "tab10"), (amq_map, "component AMQ", "magma")]
            fig, axes = plt.subplots(1, 5, figsize=(19, 4))
            for axis, (value, title, cmap) in zip(axes, panels):
                axis.imshow(value, cmap=cmap); axis.set_title(title); axis.axis("off")
            fig.suptitle(image_id); fig.tight_layout()
            for folder, rank in image_destinations[image_id]:
                fig.savefig(output / f"visualizations/{folder}/{rank:02d}_{image_id}.png", dpi=140)
            plt.close(fig)
        wanted.remove(image_id)
        print(f"FALSE_COMPONENT_VISUAL_PROGRESS remaining={len(wanted)}", flush=True)
        if not wanted:
            break
    if wanted:
        raise AssertionError(f"Visualization images missing: {sorted(wanted)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("extract", "analyze", "visualize"), required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hqmr-checkpoint", required=True)
    parser.add_argument("--sshr-checkpoint", required=True)
    parser.add_argument("--morphology-output", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args(); output = Path(args.output_dir).resolve()
    if args.mode == "extract":
        run_extract(args, output)
    elif args.mode == "analyze":
        run_analyze(args, output)
    else:
        run_visualize(args, output)


if __name__ == "__main__":
    main()
