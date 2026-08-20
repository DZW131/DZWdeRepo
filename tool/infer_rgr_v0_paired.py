"""Paired same-forward BCSS validation for frozen-SSHR RGR-v0."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from tool import infer_utils, iouutils
from tool.GenDataset import Stage1_InferDataset
from tool.infer_fun import _get_class_thresholds, _tta_transforms


EXPECTED_VAL = 3_418
PURITY_THRESHOLD = 0.80
VARIANTS = ("base", "isolated", "graph_only", "full")
NODE_GROUPS = ("N=1", "N=2", "N=3-4", "N>=5")


def _normalize(cam):
    array = cam.detach().float().cpu().numpy()
    minimum = array.min(axis=(1, 2), keepdims=True)
    maximum = array.max(axis=(1, 2), keepdims=True)
    return (array - minimum) / (maximum - minimum + 1e-8)


def _prediction(cam, label, image):
    masked = cam * label.reshape(4, 1, 1)
    cam_dict = infer_utils.cam_npy_to_cam_dict(masked, label)
    cam_score, _ = infer_utils.dict2npy(cam_dict, label, image)
    return infer_utils.cam_npy_to_label_map(cam_score).astype(np.uint8)


def _metric_record(metrics):
    return {
        "mIoU": float(metrics["Mean IoU"]),
        "mDice": float(metrics["Mean Dice"]),
        "class_iou": {str(key): float(value) for key, value in metrics["Class IoU"].items()},
        "class_dice": {
            str(key): float(value)
            for key, value in metrics["Dice Coefficients"].items()
        },
    }


def _official_metrics(ground_truth, predictions):
    return _metric_record(iouutils.scores(
        [item.copy() for item in ground_truth],
        [item.copy() for item in predictions],
        n_class=4,
    ))


def _array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def _aggregate_statistics(rows):
    if not rows:
        return {}
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in sorted(rows[0])
    }


def _taxonomy_diagnostics(ground_truth, predictions):
    base_predictions = predictions["base"]
    categories = {
        "A_correct_pure": {},
        "B_misclassified_pure": {},
        "C_false_positive_pure": {},
        "D_mixed_boundary": {},
    }
    for values in categories.values():
        values.update({
            "regions": 0,
            "pixels": 0,
            "base_wrong_pixels": 0,
            "variant_wrong_pixels": {variant: 0 for variant in predictions},
        })

    for truth, base, image_variants in zip(
        ground_truth,
        base_predictions,
        zip(*(predictions[variant] for variant in predictions)),
    ):
        variant_map = dict(zip(predictions, image_variants))
        for predicted_class in range(4):
            count, components = cv2.connectedComponents(
                np.asarray(base == predicted_class, dtype=np.uint8), connectivity=8
            )
            for component_id in range(1, count):
                mask = components == component_id
                area = int(mask.sum())
                if area == 0:
                    continue
                counts = np.bincount(truth[mask].astype(np.int64), minlength=5)
                majority = int(np.argmax(counts))
                purity = float(counts[majority] / area)
                if purity < PURITY_THRESHOLD:
                    category = "D_mixed_boundary"
                elif majority == predicted_class:
                    category = "A_correct_pure"
                elif majority == 4:
                    category = "C_false_positive_pure"
                else:
                    category = "B_misclassified_pure"
                values = categories[category]
                values["regions"] += 1
                values["pixels"] += area
                base_wrong = int(np.count_nonzero(base[mask] != truth[mask]))
                values["base_wrong_pixels"] += base_wrong
                for variant, prediction in variant_map.items():
                    values["variant_wrong_pixels"][variant] += int(
                        np.count_nonzero(prediction[mask] != truth[mask])
                    )

    for values in categories.values():
        values["recovery_pixels"] = {}
        values["recovery_rate"] = {}
        for variant, wrong in values["variant_wrong_pixels"].items():
            recovery = values["base_wrong_pixels"] - wrong
            values["recovery_pixels"][variant] = recovery
            values["recovery_rate"][variant] = (
                recovery / float(max(values["base_wrong_pixels"], 1))
            )
    return categories


def _node_group(count):
    if count == 1:
        return "N=1"
    if count == 2:
        return "N=2"
    if count <= 4:
        return "N=3-4"
    return "N>=5"


def _node_stratified(ground_truth, predictions, node_counts):
    result = {}
    groups = np.asarray([_node_group(count) for count in node_counts])
    for group in NODE_GROUPS:
        selection = np.flatnonzero(groups == group)
        if selection.size == 0:
            result[group] = {"image_count": 0}
            continue
        metrics = {
            variant: _official_metrics(ground_truth[selection], values[selection])
            for variant, values in predictions.items()
        }
        result[group] = {
            "image_count": int(selection.size),
            "metrics": metrics,
            "full_minus_base_pp": 100.0 * (
                metrics["full"]["mIoU"] - metrics["base"]["mIoU"]
            ),
            "full_minus_isolated_pp": 100.0 * (
                metrics["full"]["mIoU"] - metrics["isolated"]["mIoU"]
            ),
            "graph_only_minus_base_pp": 100.0 * (
                metrics["graph_only"]["mIoU"] - metrics["base"]["mIoU"]
            ),
        }
    selection = np.flatnonzero(np.asarray(node_counts) >= 2)
    if selection.size:
        metrics = {
            variant: _official_metrics(ground_truth[selection], values[selection])
            for variant, values in predictions.items()
        }
        result["N>=2_all"] = {
            "image_count": int(selection.size),
            "metrics": metrics,
            "full_minus_base_pp": 100.0 * (
                metrics["full"]["mIoU"] - metrics["base"]["mIoU"]
            ),
            "full_minus_isolated_pp": 100.0 * (
                metrics["full"]["mIoU"] - metrics["isolated"]["mIoU"]
            ),
            "graph_only_minus_base_pp": 100.0 * (
                metrics["graph_only"]["mIoU"] - metrics["base"]["mIoU"]
            ),
        }
    else:
        result["N>=2_all"] = {"image_count": 0}
    return result


def _validate_scope(dataroot, variants):
    path = Path(dataroot)
    combined = str(path).lower()
    if "test" in combined or "luad" in combined:
        raise ValueError("Paired RGR evaluation is BCSS validation only")
    if path.name.lower() != "val" or not (path / "img").is_dir() or not (path / "mask").is_dir():
        raise ValueError("dataroot must point exactly to BCSS val")
    if not variants or "base" not in variants or "full" not in variants:
        raise ValueError("Paired evaluation always requires base and full")
    if not set(variants).issubset(VARIANTS):
        raise ValueError(f"Unknown variants: {variants}")


def _validation_loader(dataroot, args):
    dataset = Stage1_InferDataset(
        data_path=str(Path(dataroot) / "img"), img_size=args.img_size
    )
    if len(dataset) != EXPECTED_VAL:
        raise RuntimeError(f"BCSS validation count {len(dataset)} != {EXPECTED_VAL}")
    return dataset, DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _official_contract(args):
    if getattr(args, "amp_dtype", None) != "bf16" or getattr(args, "dataset", None) != "bcss":
        raise ValueError("Formal paired evaluation requires BCSS BF16")
    thresholds = _get_class_thresholds(args, None, 4)
    expected = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
    if not np.array_equal(thresholds, expected):
        raise RuntimeError("Official BCSS class thresholds changed")
    if _tta_transforms() != (((), ()), ((3,), (2,)), ((2,), (1,))):
        raise RuntimeError("Official three-way TTA changed")
    return thresholds


def paired_rgr_validation(
    model,
    dataroot,
    args,
    variants=VARIANTS,
    return_arrays=False,
):
    """Evaluate all variants from identical base tensors and TTA views."""

    variants = tuple(variants)
    _validate_scope(dataroot, variants)
    model.eval()
    model.cuda()
    dataset, loader = _validation_loader(dataroot, args)
    thresholds = _official_contract(args)
    prediction_lists = {variant: [] for variant in variants}
    names, ground_truth, node_counts = [], [], []
    statistics = []
    base_forward_seconds = graph_seconds = region_seconds = 0.0
    started = time.perf_counter()

    with torch.no_grad():
        for image_index, (name_tuple, image_tensor) in enumerate(loader):
            image_name = name_tuple[0]
            image_path = Path(dataroot) / "img" / f"{image_name}.png"
            mask_path = Path(dataroot) / "mask" / f"{image_name}.png"
            image = np.asarray(Image.open(image_path).convert("RGB"))
            original_size = image.shape[:2]
            image_tensor = image_tensor.cuda(non_blocking=True)
            bases, probabilities = [], []
            for input_flip_dims, _ in _tta_transforms():
                tta_image = torch.flip(image_tensor, dims=input_flip_dims) if input_flip_dims else image_tensor
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    base = model.forward_cam_base(tta_image)
                torch.cuda.synchronize()
                base_forward_seconds += time.perf_counter() - tick
                bases.append(base)
                probabilities.append(base[4])

            probability = torch.stack(probabilities).mean(0).float().cpu().numpy()[0]
            label = (probability > thresholds).astype(np.float32)
            if label.sum() == 0:
                label[int(np.argmax(probability))] = 1.0
            presence = torch.from_numpy(label)[None].cuda()
            tta_variant_cams = {variant: [] for variant in variants}
            tta_28_2, tta_deep = [], []

            for tta_index, (base, (_, cam_flip_dims)) in enumerate(zip(bases, _tta_transforms())):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = model.refine_from_base(base, presence)
                    variant_28_1 = {
                        "base": F.relu(base[1]),
                        "isolated": F.relu(base[1] + result.delta_iso),
                        "graph_only": F.relu(base[1] + result.delta_graph),
                        "full": F.relu(base[1] + result.delta_iso + result.delta_graph),
                    }
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                graph_seconds += elapsed
                region_seconds += elapsed
                statistics.append(result.statistics)
                if tta_index == 0:
                    node_counts.append(result.per_image_region_counts[0])
                for variant in variants:
                    upsampled = F.interpolate(
                        variant_28_1[variant], original_size,
                        mode="bilinear", align_corners=False,
                    )[0]
                    if cam_flip_dims:
                        upsampled = torch.flip(upsampled, dims=cam_flip_dims)
                    tta_variant_cams[variant].append(upsampled)
                for target, cam in ((tta_28_2, F.relu(base[2])), (tta_deep, F.relu(base[3]))):
                    upsampled = F.interpolate(
                        cam, original_size, mode="bilinear", align_corners=False
                    )[0]
                    if cam_flip_dims:
                        upsampled = torch.flip(upsampled, dims=cam_flip_dims)
                    target.append(upsampled)

            normalized_28_2 = _normalize(torch.stack(tta_28_2).mean(0))
            normalized_deep = _normalize(torch.stack(tta_deep).mean(0))
            for variant in variants:
                normalized_28_1 = _normalize(torch.stack(tta_variant_cams[variant]).mean(0))
                fusion = 0.6 * normalized_28_1 + 0.2 * normalized_28_2 + 0.2 * normalized_deep
                prediction_lists[variant].append(_prediction(fusion, label, image))
            names.append(image_name)
            ground_truth.append(np.asarray(Image.open(mask_path), dtype=np.uint8))
            if (image_index + 1) % 250 == 0:
                print(f"RGR_PAIRED_VAL_PROGRESS {image_index + 1}/{len(dataset)}", flush=True)

    gt_array = np.stack(ground_truth)
    predictions = {variant: np.stack(items) for variant, items in prediction_lists.items()}
    final_metrics = {
        variant: _official_metrics(gt_array, prediction)
        for variant, prediction in predictions.items()
    }
    base = final_metrics["base"]
    full = final_metrics["full"]
    isolated = final_metrics.get("isolated", base)
    graph = final_metrics.get("graph_only", base)
    mechanism = _aggregate_statistics(statistics)
    result = {
        "image_count": len(names),
        "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "variants": list(variants),
        "final_metrics": final_metrics,
        "full_minus_base_pp": 100.0 * (full["mIoU"] - base["mIoU"]),
        "isolated_minus_base_pp": 100.0 * (isolated["mIoU"] - base["mIoU"]),
        "graph_only_minus_base_pp": 100.0 * (graph["mIoU"] - base["mIoU"]),
        "full_minus_isolated_pp": 100.0 * (full["mIoU"] - isolated["mIoU"]),
        "full_minus_base_mdice_pp": 100.0 * (full["mDice"] - base["mDice"]),
        "paired_class_iou_delta_pp": {
            key: 100.0 * (full["class_iou"][key] - base["class_iou"][key])
            for key in base["class_iou"]
        },
        "isolated_class_iou_delta_pp": {
            key: 100.0 * (isolated["class_iou"][key] - base["class_iou"][key])
            for key in base["class_iou"]
        },
        "differing_base_full_pixels": int(np.count_nonzero(predictions["base"] != predictions["full"])),
        "differing_isolated_full_pixels": int(np.count_nonzero(predictions.get("isolated", predictions["base"]) != predictions["full"])),
        "prediction_sha256": {
            variant: _array_hash(prediction) for variant, prediction in predictions.items()
        },
        "taxonomy": _taxonomy_diagnostics(gt_array, predictions),
        "node_count_stratified": (
            _node_stratified(gt_array, predictions, node_counts)
            if set(VARIANTS).issubset(predictions)
            else {}
        ),
        "node_counts": {
            "mean": float(np.mean(node_counts)),
            "min": int(np.min(node_counts)),
            "max": int(np.max(node_counts)),
            "groups": {
                group: int(sum(_node_group(count) == group for count in node_counts))
                for group in NODE_GROUPS
            },
        },
        "mechanism": mechanism,
        "runtime": {
            "validation_seconds": time.perf_counter() - started,
            "base_forward_seconds": base_forward_seconds,
            "region_extraction_and_graph_seconds": region_seconds,
            "message_passing_included_seconds": graph_seconds,
            "mean_rgr_seconds_per_image": graph_seconds / len(dataset),
            "rgr_overhead_vs_base_forward_percent": 100.0 * graph_seconds / max(base_forward_seconds, 1e-12),
        },
        "official_fusion": {"cam56": 0.0, "cam28_1": 0.6, "cam28_2": 0.2, "camdeep": 0.2},
        "official_thresholds": thresholds.tolist(),
        "tta": [[list(left), list(right)] for left, right in _tta_transforms()],
        "optimizer_updates_during_evaluation": 0,
    }
    if return_arrays:
        return result, gt_array, predictions, np.asarray(node_counts)
    return result


def official_a0_validation(model, dataroot, args, return_arrays=False):
    """Independent released A0 validation for corrected parity comparison."""

    _validate_scope(dataroot, ("base", "full"))
    model.eval()
    model.cuda()
    dataset, loader = _validation_loader(dataroot, args)
    thresholds = _official_contract(args)
    predictions, ground_truth, names = [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for image_index, (name_tuple, image_tensor) in enumerate(loader):
            image_name = name_tuple[0]
            image_path = Path(dataroot) / "img" / f"{image_name}.png"
            mask_path = Path(dataroot) / "mask" / f"{image_name}.png"
            image = np.asarray(Image.open(image_path).convert("RGB"))
            original_size = image.shape[:2]
            image_tensor = image_tensor.cuda(non_blocking=True)
            cams_28_1, cams_28_2, cams_deep, probabilities = [], [], [], []
            for input_flip_dims, cam_flip_dims in _tta_transforms():
                tta_image = torch.flip(image_tensor, dims=input_flip_dims) if input_flip_dims else image_tensor
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, cam_28_1, cam_28_2, cam_deep, probability = model.forward_cam(tta_image)
                for target, cam in (
                    (cams_28_1, cam_28_1),
                    (cams_28_2, cam_28_2),
                    (cams_deep, cam_deep),
                ):
                    upsampled = F.interpolate(cam, original_size, mode="bilinear", align_corners=False)[0]
                    if cam_flip_dims:
                        upsampled = torch.flip(upsampled, dims=cam_flip_dims)
                    target.append(upsampled)
                probabilities.append(probability)
            probability = torch.stack(probabilities).mean(0).float().cpu().numpy()[0]
            label = (probability > thresholds).astype(np.float32)
            if label.sum() == 0:
                label[int(np.argmax(probability))] = 1.0
            fusion = (
                0.6 * _normalize(torch.stack(cams_28_1).mean(0))
                + 0.2 * _normalize(torch.stack(cams_28_2).mean(0))
                + 0.2 * _normalize(torch.stack(cams_deep).mean(0))
            )
            predictions.append(_prediction(fusion, label, image))
            ground_truth.append(np.asarray(Image.open(mask_path), dtype=np.uint8))
            names.append(image_name)
            if (image_index + 1) % 250 == 0:
                print(f"RGR_A0_PARITY_PROGRESS {image_index + 1}/{len(dataset)}", flush=True)
    prediction_array = np.stack(predictions)
    gt_array = np.stack(ground_truth)
    result = {
        "image_count": len(names),
        "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "metrics": _official_metrics(gt_array, prediction_array),
        "prediction_sha256": _array_hash(prediction_array),
        "runtime_seconds": time.perf_counter() - started,
        "official_thresholds": thresholds.tolist(),
    }
    if return_arrays:
        return result, gt_array, prediction_array
    return result
