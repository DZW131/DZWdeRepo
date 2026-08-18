#!/usr/bin/env python3
"""Frozen BCSS validation-only Phase-0R region representation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.resnet38_cls import Net_CAM
from tool import infer_fun
from tool.GenDataset import Stage1_InferDataset
from tools.region_phase0r import (
    IMAGE_SIZE,
    PRIMARY_MIN_AREA,
    REPRESENTATIONS,
    SEED,
    SENSITIVITY_AREAS,
)
from tools.region_phase0r.extractor import RegionAuditExtractor, predict_and_feature
from tools.region_phase0r.probes import (
    classification_metrics,
    decide_phase0r,
    official_scores,
    representation_cluster_metrics,
    run_oof_probe,
)
from tools.region_phase0r.regions import connected_components, extract_image_regions, relabel_predictions
from tools.region_phase0r.report import write_report


A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
EXPECTED_CHECKPOINT_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
EXPECTED_IMAGES = 3418
EXPECTED_SLIDES = 22


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-commit", default="WORKTREE")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--expected-checkpoint-sha256", default="")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False


def validate_scope(args):
    combined = " ".join((args.val_root, args.checkpoint, args.output_dir)).lower()
    if "test" in combined or "luad" in combined:
        raise ValueError("Phase-0R is restricted to BCSS validation only")
    val_root = Path(args.val_root)
    if val_root.name.lower() != "val":
        raise ValueError("--val-root must point exactly to the BCSS val split")
    if not (val_root / "img").is_dir() or not (val_root / "mask").is_dir():
        raise FileNotFoundError("BCSS val/img and val/mask are required")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("config", "parity", "regions", "tables", "figures", "docs"):
        (output / directory).mkdir()


def load_models(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    released = Net_CAM(n_class=4)
    extractor = RegionAuditExtractor(n_class=4)
    released.load_state_dict(state, strict=True)
    extractor.load_state_dict(state, strict=True)
    for key, value in released.state_dict().items():
        if not torch.equal(value, extractor.state_dict()[key]):
            raise AssertionError(f"Extractor changed checkpoint tensor {key}")
    return released.eval(), extractor.eval()


def flatten_metrics(metrics):
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f"{key}.{subkey}"] = float(subvalue)
        elif np.isscalar(value):
            flat[key] = float(value)
    return flat


def released_inference_capture(model, val_root, runtime_args):
    """Capture masks passed to the released metric without altering inference."""
    captured = {}
    original_scores = infer_fun.iouutils.scores

    def capture(ground_truth, prediction, n_class):
        captured["ground_truth"] = np.stack([item.copy() for item in ground_truth])
        captured["prediction"] = np.stack([item.copy() for item in prediction])
        return original_scores(ground_truth, prediction, n_class)

    infer_fun.iouutils.scores = capture
    try:
        metrics = infer_fun.infer(model, str(val_root), 4, runtime_args)
    finally:
        infer_fun.iouutils.scores = original_scores
    if metrics is None or not captured:
        raise RuntimeError("Released inference did not complete")
    return metrics, captured["ground_truth"], captured["prediction"]


def feature_extraction(model, val_root, runtime_args):
    dataset = Stage1_InferDataset(str(val_root / "img"), img_size=IMAGE_SIZE)
    loader = DataLoader(dataset, shuffle=False, batch_size=1,
                        num_workers=runtime_args.num_workers, pin_memory=True)
    model = model.cuda().eval()
    names, ground_truth, predictions, rows = [], [], [], []
    region_parts, bbox_parts, centroid_parts, geometry_parts = [], [], [], []
    for image_index, (name_tuple, tensor) in enumerate(loader):
        image_id = name_tuple[0]
        image_path = val_root / "img" / f"{image_id}.png"
        mask_path = val_root / "mask" / f"{image_id}.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        gt = np.asarray(Image.open(mask_path), dtype=np.uint8)
        prediction, feature = predict_and_feature(
            model, tensor.cuda(non_blocking=True), image.shape[:2], runtime_args
        )
        image_rows, region, bbox, centroid, geometry = extract_image_regions(
            prediction, gt, feature, image_id, image_index
        )
        rows.extend(image_rows)
        region_parts.append(region); bbox_parts.append(bbox)
        centroid_parts.append(centroid); geometry_parts.append(geometry)
        names.append(image_id); ground_truth.append(gt); predictions.append(prediction)
        if (image_index + 1) % 250 == 0:
            print(f"EXTRACT_PROGRESS {image_index + 1}/{len(dataset)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.insert(0, "token_index", np.arange(len(frame), dtype=np.int64))
    tokens = {
        "region": np.concatenate(region_parts),
        "bbox": np.concatenate(bbox_parts),
        "centroid": np.concatenate(centroid_parts),
        "geometry": np.concatenate(geometry_parts),
    }
    tokens["region_geometry"] = np.concatenate((tokens["region"], tokens["geometry"]), axis=1)
    return names, np.stack(ground_truth), np.stack(predictions), frame, tokens


def parity(released_metrics, released_gt, released_prediction, gt, prediction):
    if released_gt.shape != gt.shape or not np.array_equal(released_gt, gt):
        raise AssertionError("Ground-truth order differs from released inference")
    differing = int(np.count_nonzero(released_prediction != prediction))
    extracted_metrics = official_scores(gt, prediction)
    left, right = flatten_metrics(released_metrics), flatten_metrics(extracted_metrics)
    maximum_difference = max(abs(left[key] - right[key]) for key in left)
    result = {
        "differing_pixels": differing,
        "maximum_metric_difference": maximum_difference,
    }
    if differing != 0 or maximum_difference >= 1e-7:
        result["decision"] = "REGION_PHASE0R_PARITY_NOGO"
        raise AssertionError(json.dumps(result))
    result["decision"] = "PARITY_PASS"
    return result, extracted_metrics


def phase_a_tables(frame, output_dir):
    purity_rows = []
    for area_threshold in (1, PRIMARY_MIN_AREA, 32):
        subset = frame[frame.pixel_area >= area_threshold]
        for predicted_class in ("all", 0, 1, 2, 3):
            part = subset if predicted_class == "all" else subset[subset.predicted_class == predicted_class]
            total_pixels = float(part.pixel_area.sum())
            for stratum, lower, upper in (
                (">=0.95", 0.95, np.inf), ("0.80-0.95", 0.80, 0.95),
                ("0.60-0.80", 0.60, 0.80), ("<0.60", -np.inf, 0.60),
            ):
                in_stratum = (part.purity >= lower) & (part.purity < upper)
                purity_rows.append({
                    "min_area": area_threshold,
                    "predicted_class": predicted_class,
                    "purity_stratum": stratum,
                    "n_regions_total": int(len(part)),
                    "region_weighted_mean_purity": float(part.purity.mean()),
                    "median_purity": float(part.purity.median()),
                    "pixel_weighted_purity": float(
                        (part.pixel_area * part.purity).sum() / max(total_pixels, 1.0)
                    ),
                    "stratum_region_count": int(in_stratum.sum()),
                    "stratum_region_fraction": float(in_stratum.mean()),
                    "stratum_pixel_mass": int(part.loc[in_stratum, "pixel_area"].sum()),
                    "stratum_pixel_fraction": float(
                        part.loc[in_stratum, "pixel_area"].sum() / max(total_pixels, 1.0)
                    ),
                })
    pd.DataFrame(purity_rows).to_csv(output_dir / "tables" / "region_purity_by_class.csv", index=False)
    total_wrong = float(frame.wrong_foreground_pixels.sum())
    taxonomy = (
        frame.groupby("taxonomy", as_index=False)
        .agg(n_regions=("region_id", "count"), wrong_pixels=("wrong_foreground_pixels", "sum"))
    )
    taxonomy["wrong_pixel_fraction"] = taxonomy.wrong_pixels / max(total_wrong, 1.0)
    taxonomy.to_csv(output_dir / "tables" / "taxonomy_error_mass.csv", index=False)
    return taxonomy


def run_probes(frame, tokens, ground_truth, predictions, baseline_metrics, output_dir):
    all_rows, primary_artifacts = [], {}
    thresholds = (1, PRIMARY_MIN_AREA, 32)
    for min_area in thresholds:
        selected = frame.pixel_area.to_numpy() >= min_area
        indices = frame.token_index.to_numpy()[selected]
        labels = frame.majority_gt.to_numpy(dtype=np.int64)[selected]
        groups = frame.slide_id.to_numpy()[selected]
        for representation in REPRESENTATIONS:
            oof, fold_ids, metrics, folds = run_oof_probe(tokens[representation][indices], labels, groups)
            selected_frame = frame.loc[selected].copy().reset_index(drop=True)
            relabeled = relabel_predictions(predictions, selected_frame, oof)
            segmentation = official_scores(ground_truth, relabeled)
            gain_pp = 100.0 * (segmentation["Mean IoU"] - baseline_metrics["Mean IoU"])
            entry = {
                "min_area": min_area,
                "representation": representation,
                **metrics,
                "segmentation_miou": float(segmentation["Mean IoU"]),
                "segmentation_mdice": float(segmentation["Mean Dice"]),
                "segmentation_gain_miou_pp": gain_pp,
            }
            all_rows.append(entry)
            if min_area == PRIMARY_MIN_AREA:
                primary_artifacts[representation] = {
                    **entry,
                    "oof": oof,
                    "fold_ids": fold_ids,
                    "folds": folds,
                    "frame": selected_frame,
                    "prediction": relabeled,
                }
    pd.DataFrame(all_rows).to_csv(output_dir / "tables" / "probe_results.csv", index=False)
    fold_rows = []
    for representation, artifact in primary_artifacts.items():
        for row in artifact["folds"]:
            fold_rows.append({"representation": representation, **row})
    pd.DataFrame(fold_rows).to_json(output_dir / "tables" / "groupkfold_details.json", orient="records", indent=2)
    return primary_artifacts


def fold_segmentation_gains(region_artifact, image_slide_ids, ground_truth, predictions):
    gains = []
    frame = region_artifact["frame"]
    for fold in range(5):
        held_slides = set(frame.loc[region_artifact["fold_ids"] == fold, "slide_id"])
        image_ids = [index for index, slide_id in enumerate(image_slide_ids) if slide_id in held_slides]
        raw = official_scores(ground_truth[image_ids], predictions[image_ids])
        relabeled = official_scores(ground_truth[image_ids], region_artifact["prediction"][image_ids])
        gains.append({
            "fold": fold,
            "held_out_slides": sorted(held_slides),
            "baseline_miou": float(raw["Mean IoU"]),
            "region_miou": float(relabeled["Mean IoU"]),
            "gain_miou_pp": 100.0 * (relabeled["Mean IoU"] - raw["Mean IoU"]),
        })
    return gains


def save_cluster_and_strata(frame, tokens, primary_artifacts, output_dir):
    selected = frame.pixel_area.to_numpy() >= PRIMARY_MIN_AREA
    indices = frame.token_index.to_numpy()[selected]
    labels = frame.majority_gt.to_numpy(dtype=np.int64)[selected]
    cluster_rows = []
    for representation in ("region", "bbox", "centroid"):
        cluster_rows.append({
            "representation": representation,
            **representation_cluster_metrics(tokens[representation][indices], labels),
        })
    pd.DataFrame(cluster_rows).to_csv(output_dir / "tables" / "representation_cluster_metrics.csv", index=False)
    artifact = primary_artifacts["region"]
    strata = artifact["frame"].copy()
    strata["oof_prediction"] = artifact["oof"]
    strata["correct"] = strata.oof_prediction == strata.majority_gt
    strata["purity_stratum"] = pd.cut(
        strata.purity, bins=[-np.inf, 0.6, 0.8, 0.95, np.inf],
        labels=["<0.60", "0.60-0.80", "0.80-0.95", ">=0.95"], right=False,
    )
    summary = []
    for name, part in strata.groupby("purity_stratum", observed=True):
        summary.append({
            "purity_stratum": str(name), "n_regions": int(len(part)),
            "mean_purity": float(part.purity.mean()),
            "wrong_pixels": int(part.wrong_foreground_pixels.sum()),
            **classification_metrics(part.majority_gt, part.oof_prediction),
        })
    pd.DataFrame(summary).to_csv(output_dir / "tables" / "region_probe_purity_strata.csv", index=False)


def qualitative_panels(frame, artifact, val_root, gt, raw_prediction, oracle_prediction, output_dir):
    enriched = artifact["frame"].copy()
    enriched["oof_prediction"] = artifact["oof"]
    categories = {
        "A_correctable_success": (enriched.taxonomy == "B_misclassified_pure") & (enriched.oof_prediction == enriched.majority_gt),
        "B_correctable_failure": (enriched.taxonomy == "B_misclassified_pure") & (enriched.oof_prediction != enriched.majority_gt),
        "C_mixed_boundary": enriched.taxonomy == "D_mixed_boundary",
        "D_background_removal": (enriched.taxonomy == "C_false_positive_pure") & (enriched.oof_prediction == 4),
    }
    selected_rows, used_images = [], set()
    for category, condition in categories.items():
        candidates = enriched[condition].groupby(["image_index", "image_id"], as_index=False).agg(
            wrong_pixels=("wrong_foreground_pixels", "sum"), n_regions=("region_id", "count")
        ).sort_values(["wrong_pixels", "image_id"], ascending=[False, True])
        candidates = candidates[~candidates.image_index.isin(used_images)].head(8)
        if len(candidates) < 8:
            raise RuntimeError(f"Insufficient automatic qualitative cases for {category}: {len(candidates)}")
        for item in candidates.itertuples(index=False):
            used_images.add(int(item.image_index))
            selected_rows.append({"category": category, **item._asdict()})
    selection = pd.DataFrame(selected_rows)
    selection.to_csv(output_dir / "tables" / "qualitative_selection.csv", index=False)
    region_prediction = artifact["prediction"]
    colors = np.array([[220, 20, 60], [34, 139, 34], [30, 144, 255], [255, 165, 0], [245, 245, 245]], dtype=np.uint8)
    for category, group in selection.groupby("category", sort=False):
        figure, axes = plt.subplots(8, 8, figsize=(24, 23))
        for row_index, item in enumerate(group.itertuples(index=False)):
            index = int(item.image_index)
            image = np.asarray(Image.open(val_root / "img" / f"{item.image_id}.png").convert("RGB"))
            components = connected_components(raw_prediction[index])
            component_map = np.full(raw_prediction[index].shape, -1, dtype=np.int32)
            purity_map = np.full(raw_prediction[index].shape, np.nan, dtype=np.float32)
            image_regions = frame[frame.image_index == index]
            for ordinal, region_row in enumerate(image_regions.itertuples(index=False)):
                mask = components[int(region_row.predicted_class)][0] == int(region_row.component_label)
                component_map[mask] = ordinal
                purity_map[mask] = float(region_row.purity)
            component_rgb = np.full((*component_map.shape, 3), 245, dtype=np.uint8)
            foreground = component_map >= 0
            palette = plt.get_cmap("tab20")((component_map[foreground] % 20) / 19.0)[:, :3]
            component_rgb[foreground] = (palette * 255).astype(np.uint8)
            purity_rgb = np.full((*purity_map.shape, 3), 245, dtype=np.uint8)
            purity_rgb[foreground] = (plt.get_cmap("viridis")(np.nan_to_num(purity_map[foreground]))[:, :3] * 255).astype(np.uint8)
            raw_wrong = raw_prediction[index] != gt[index]
            region_wrong = region_prediction[index] != gt[index]
            error_difference = np.zeros((*raw_wrong.shape, 3), dtype=np.uint8)
            error_difference[raw_wrong & ~region_wrong] = (0, 210, 0)
            error_difference[~raw_wrong & region_wrong] = (230, 0, 0)
            error_difference[raw_wrong & region_wrong] = (160, 0, 160)
            maps = [image, colors[gt[index]], colors[raw_prediction[index]], component_rgb,
                    purity_rgb, colors[oracle_prediction[index]], colors[region_prediction[index]],
                    error_difference]
            for col, content in enumerate(maps):
                axes[row_index, col].imshow(content)
                axes[row_index, col].axis("off")
                if row_index == 0:
                    axes[row_index, col].set_title([
                        "Input", "GT", "Official", "Components", "Purity",
                        "GT-majority oracle", "Region OOF", "Error difference",
                    ][col])
            axes[row_index, 0].set_ylabel(item.image_id[:18], fontsize=7)
        figure.suptitle(category)
        figure.tight_layout()
        figure.savefig(output_dir / "figures" / f"{category}.png", dpi=130)
        plt.close(figure)


def metric_record(metrics):
    return {
        "mean_iou": float(metrics["Mean IoU"]),
        "mean_dice": float(metrics["Mean Dice"]),
        "class_iou": {str(k): float(v) for k, v in metrics["Class IoU"].items()},
        "class_dice": {str(k): float(v) for k, v in metrics["Dice Coefficients"].items()},
    }


def main():
    args = parse_args()
    validate_scope(args)
    seed_everything()
    output_dir = Path(args.output_dir)
    val_root = Path(args.val_root)
    checkpoint_sha = sha256_file(args.checkpoint)
    expected = args.expected_checkpoint_sha256 or EXPECTED_CHECKPOINT_SHA256
    if expected and not checkpoint_sha.startswith(expected):
        raise ValueError(f"Checkpoint SHA256 mismatch: {checkpoint_sha} does not match {expected}")
    runtime_args = SimpleNamespace(
        dataset="bcss", img_size=IMAGE_SIZE, num_workers=args.num_workers,
        amp_dtype=args.amp_dtype,
    )
    released, extractor = load_models(args.checkpoint)
    print("PHASE_MINUS_ONE_RELEASED_INFERENCE", flush=True)
    released_metrics, released_gt, released_prediction = released_inference_capture(
        released, val_root, runtime_args
    )
    del released
    torch.cuda.empty_cache()
    print("PHASE_MINUS_ONE_EXTRACTOR", flush=True)
    names, gt, predictions, frame, tokens = feature_extraction(extractor, val_root, runtime_args)
    parity_result, baseline_metrics = parity(
        released_metrics, released_gt, released_prediction, gt, predictions
    )
    del released_gt, released_prediction, extractor
    torch.cuda.empty_cache()
    slides = sorted(frame.slide_id.unique())
    if len(names) != EXPECTED_IMAGES or len(slides) != EXPECTED_SLIDES:
        raise AssertionError(f"Dataset mismatch: {len(names)} images, {len(slides)} slides")
    pd.DataFrame({"image_index": range(len(names)), "image_id": names}).to_csv(
        output_dir / "regions" / "image_manifest.csv", index=False
    )
    gt_columns = ["majority_gt", "purity", "taxonomy", "wrong_foreground_pixels"] + [f"gt_pixels_{i}" for i in range(5)]
    frame.drop(columns=gt_columns).to_csv(
        output_dir / "regions" / "region_metadata_gt_free.csv.gz", index=False, compression="gzip"
    )
    frame[["token_index", "region_id", *gt_columns]].to_csv(
        output_dir / "regions" / "region_gt_diagnostics.csv.gz", index=False, compression="gzip"
    )
    np.savez_compressed(output_dir / "parity" / "frozen_masks.npz", ground_truth=gt, prediction=predictions)
    taxonomy = phase_a_tables(frame, output_dir)

    oracle_prediction = relabel_predictions(predictions, frame, frame.majority_gt.to_numpy())
    oracle_metrics = official_scores(gt, oracle_prediction)
    oracle_gain_pp = 100.0 * (oracle_metrics["Mean IoU"] - baseline_metrics["Mean IoU"])
    raw_errors = int(frame.wrong_foreground_pixels.sum())
    oracle_errors = int(np.count_nonzero((oracle_prediction != gt) & (predictions < 4)))
    print(f"ORACLE_GAIN_MIOU_PP {oracle_gain_pp:+.6f}", flush=True)

    primary_artifacts = run_probes(frame, tokens, gt, predictions, baseline_metrics, output_dir)
    image_slide_ids = [name.split("_xmin", 1)[0] for name in names]
    fold_gains = fold_segmentation_gains(
        primary_artifacts["region"], image_slide_ids, gt, predictions
    )
    pd.DataFrame(fold_gains).to_json(output_dir / "tables" / "region_fold_segmentation_gains.json", orient="records", indent=2)
    save_cluster_and_strata(frame, tokens, primary_artifacts, output_dir)
    qualitative_panels(frame, primary_artifacts["region"], val_root, gt, predictions,
                       oracle_prediction, output_dir)

    region = primary_artifacts["region"]
    bbox = primary_artifacts["bbox"]
    centroid = primary_artifacts["centroid"]
    region_geo = primary_artifacts["region_geometry"]
    region_bbox_gain = region["macro_f1"] - bbox["macro_f1"]
    region_centroid_gain = region["macro_f1"] - centroid["macro_f1"]
    geometry_f1_gain = region_geo["macro_f1"] - region["macro_f1"]
    geometry_seg_increment_pp = region_geo["segmentation_gain_miou_pp"] - region["segmentation_gain_miou_pp"]
    geometry_adds_value = geometry_f1_gain >= 0.03 or geometry_seg_increment_pp >= 0.20
    recovery_fraction = (
        region["segmentation_gain_miou_pp"] / oracle_gain_pp if oracle_gain_pp > 0 else float("nan")
    )
    positive_folds = sum(item["gain_miou_pp"] > 0 for item in fold_gains)
    mixed_rows = taxonomy[taxonomy.taxonomy == "D_mixed_boundary"]
    mixed_error_fraction = float(mixed_rows.wrong_pixel_fraction.iloc[0]) if len(mixed_rows) else 0.0
    decision_label = decide_phase0r(
        oracle_gain_pp, region_bbox_gain, region_centroid_gain,
        region["segmentation_gain_miou_pp"], recovery_fraction, positive_folds,
        mixed_error_fraction, geometry_adds_value,
    )
    interpretations = {
        "REGION_REP_STRONG_GO": "Region identity is a strong, cross-slide-decodable bottleneck; the preregistered next-stage region model is justified.",
        "REGION_REP_GO": "Region identity is a useful, cross-slide-decodable bottleneck; the preregistered next-stage region model is justified.",
        "REGION_GEOMETRY_REVIEW": "Appearance-only region identity is insufficient, while fixed geometry adds meaningful value; review a geometry-aware route before implementation.",
        "REGION_SHAPE_BOTTLENECK": "The fixed-support oracle is limited and mixed-boundary errors dominate; representation relabeling cannot address the main shape bottleneck.",
        "REGION_REP_NOGO": "The preregistered region-centric route is not supported by the frozen validation evidence and is closed.",
        "REGION_REP_REVIEW": "Evidence is intermediate and does not satisfy a preregistered Go or No-Go boundary; no model implementation is authorized.",
    }
    summary = {
        "provenance": {
            "audit_commit": args.audit_commit,
            "a0_commit": A0_COMMIT,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "command": " ".join(sys.argv),
        },
        "dataset": {"split": "BCSS validation", "images": len(names), "slides": len(slides), "slide_ids": slides},
        "parity": parity_result,
        "baseline": metric_record(baseline_metrics),
        "oracle": {
            **metric_record(oracle_metrics),
            "gain_miou_pp": oracle_gain_pp,
            "raw_wrong_foreground_pixels": raw_errors,
            "oracle_wrong_foreground_pixels": oracle_errors,
            "recovery_fraction": recovery_fraction,
        },
        "primary": {
            "min_area": PRIMARY_MIN_AREA,
            "n_regions": int((frame.pixel_area >= PRIMARY_MIN_AREA).sum()),
            "pure_fraction": float((frame.loc[frame.pixel_area >= PRIMARY_MIN_AREA, "purity"] >= 0.8).mean()),
        },
        "probes": {
            key: {field: value for field, value in artifact.items()
                  if field not in {"oof", "fold_ids", "folds", "frame", "prediction"}}
            for key, artifact in primary_artifacts.items()
        },
        "decision": {
            "label": decision_label,
            "interpretation": interpretations[decision_label],
            "region_bbox_gain": float(region_bbox_gain),
            "region_centroid_gain": float(region_centroid_gain),
            "region_seg_gain_pp": float(region["segmentation_gain_miou_pp"]),
            "region_geo_seg_gain_pp": float(region_geo["segmentation_gain_miou_pp"]),
            "geometry_f1_gain": float(geometry_f1_gain),
            "geometry_seg_increment_pp": float(geometry_seg_increment_pp),
            "geometry_adds_value": bool(geometry_adds_value),
            "positive_folds": int(positive_folds),
            "mixed_error_fraction": mixed_error_fraction,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        },
    }
    write_report(output_dir, summary)
    (output_dir / "config" / "run_config.json").write_text(
        json.dumps(summary["provenance"] | summary["dataset"] | summary["runtime"], indent=2), encoding="utf-8"
    )
    (output_dir / "parity" / "parity.json").write_text(
        json.dumps(parity_result, indent=2), encoding="utf-8"
    )
    report_source = output_dir / "SSHR_Phase0R_Region_Centric_Feasibility_Report.md"
    report_source.replace(output_dir / "docs" / "phase0r_region_centric_feasibility_audit.md")
    print(f"PHASE0R_DECISION {decision_label}", flush=True)
    print(f"REPORT {output_dir / 'docs' / 'phase0r_region_centric_feasibility_audit.md'}", flush=True)


if __name__ == "__main__":
    main()
