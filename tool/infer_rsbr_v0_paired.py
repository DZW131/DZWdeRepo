"""Paired same-forward BCSS validation for the frozen RSBR-v0 pilot."""

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
VARIANTS = ("base", "core_only", "transition_only", "full")


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
    """Measure variant recovery on regions fixed by the paired Base output."""

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


def _validate_scope(dataroot, variants):
    path = Path(dataroot)
    combined = str(path).lower()
    if "test" in combined or "luad" in combined:
        raise ValueError("Paired RSBR evaluation is BCSS validation only")
    if path.name.lower() != "val" or not (path / "img").is_dir() or not (path / "mask").is_dir():
        raise ValueError("dataroot must point exactly to BCSS val")
    if not variants or "base" not in variants or "full" not in variants:
        raise ValueError("Paired evaluation always requires base and full")
    if not set(variants).issubset(VARIANTS):
        raise ValueError(f"Unknown variants: {variants}")


def paired_rsbr_validation(model, dataroot, args, variants=("base", "full")):
    """Evaluate Base and RSBR variants from identical base tensors and TTA views."""

    variants = tuple(variants)
    _validate_scope(dataroot, variants)
    model.eval()
    model.cuda()
    dataset = Stage1_InferDataset(
        data_path=str(Path(dataroot) / "img"), img_size=args.img_size
    )
    if len(dataset) != EXPECTED_VAL:
        raise RuntimeError(f"BCSS validation count {len(dataset)} != {EXPECTED_VAL}")
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if getattr(args, "amp_dtype", None) != "bf16" or getattr(args, "dataset", None) != "bcss":
        raise ValueError("Formal paired evaluation requires BCSS BF16")
    thresholds = _get_class_thresholds(args, None, 4)
    if not np.array_equal(thresholds, np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)):
        raise RuntimeError("Official BCSS class thresholds changed")
    if _tta_transforms() != (((), ()), ((3,), (2,)), ((2,), (1,))):
        raise RuntimeError("Official three-way TTA changed")

    prediction_lists = {variant: [] for variant in variants}
    standalone_lists = {variant: [] for variant in ("base", "full")}
    names, ground_truth = [], []
    statistics = []
    base_forward_seconds = 0.0
    refinement_seconds = 0.0
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
                tta_image = (
                    torch.flip(image_tensor, dims=input_flip_dims)
                    if input_flip_dims else image_tensor
                )
                torch.cuda.synchronize()
                base_started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    base = model.forward_cam_base(tta_image)
                torch.cuda.synchronize()
                base_forward_seconds += time.perf_counter() - base_started
                bases.append(base)
                probabilities.append(base[4])

            probability = torch.stack(probabilities).mean(0).float().cpu().numpy()[0]
            label = (probability > thresholds).astype(np.float32)
            if label.sum() == 0:
                label[int(np.argmax(probability))] = 1.0
            presence = torch.from_numpy(label)[None].cuda()
            tta_variant_cams = {variant: [] for variant in variants}
            tta_28_2, tta_deep = [], []

            for base, (_, cam_flip_dims) in zip(bases, _tta_transforms()):
                torch.cuda.synchronize()
                refinement_started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = model.refine_from_base(base, presence)
                    variant_28_1 = {
                        "base": F.relu(base[1]),
                        "core_only": F.relu(base[1] + result.delta_core),
                        "transition_only": F.relu(base[1] + result.delta_transition),
                        "full": F.relu(base[1] + result.delta_core + result.delta_transition),
                    }
                torch.cuda.synchronize()
                refinement_seconds += time.perf_counter() - refinement_started
                statistics.append(result.statistics)
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

            averaged_28_2 = torch.stack(tta_28_2).mean(0)
            averaged_deep = torch.stack(tta_deep).mean(0)
            normalized_28_2 = _normalize(averaged_28_2)
            normalized_deep = _normalize(averaged_deep)
            averaged_variants = {
                variant: torch.stack(tta_variant_cams[variant]).mean(0)
                for variant in variants
            }
            for variant in variants:
                normalized_28_1 = _normalize(averaged_variants[variant])
                fusion = (
                    0.6 * normalized_28_1
                    + 0.2 * normalized_28_2
                    + 0.2 * normalized_deep
                )
                prediction_lists[variant].append(_prediction(fusion, label, image))
                if variant in standalone_lists:
                    standalone_lists[variant].append(
                        _prediction(normalized_28_1, label, image)
                    )
            names.append(image_name)
            ground_truth.append(np.asarray(Image.open(mask_path), dtype=np.uint8))
            if (image_index + 1) % 250 == 0:
                print(
                    f"RSBR_PAIRED_VAL_PROGRESS {image_index + 1}/{len(dataset)}",
                    flush=True,
                )

    gt_array = np.stack(ground_truth)
    predictions = {
        variant: np.stack(items) for variant, items in prediction_lists.items()
    }
    standalone = {
        variant: np.stack(items) for variant, items in standalone_lists.items()
    }
    final_metrics = {
        variant: _official_metrics(gt_array, prediction)
        for variant, prediction in predictions.items()
    }
    standalone_metrics = {
        variant: _official_metrics(gt_array, prediction)
        for variant, prediction in standalone.items()
    }
    taxonomy = _taxonomy_diagnostics(gt_array, predictions)
    base_metrics = final_metrics["base"]
    full_metrics = final_metrics["full"]
    class_delta = {
        key: 100.0 * (full_metrics["class_iou"][key] - base_metrics["class_iou"][key])
        for key in base_metrics["class_iou"]
    }
    mechanism = _aggregate_statistics(statistics)
    mechanism["transition_to_core_rms"] = (
        mechanism["rms_delta_transition"]
        / (mechanism["rms_delta_core"] + 1e-12)
    )
    result = {
        "image_count": len(names),
        "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "variants": list(variants),
        "final_metrics": final_metrics,
        "standalone_cam28_1_metrics": standalone_metrics,
        "paired_delta_miou_pp": 100.0 * (full_metrics["mIoU"] - base_metrics["mIoU"]),
        "paired_delta_mdice_pp": 100.0 * (full_metrics["mDice"] - base_metrics["mDice"]),
        "paired_class_iou_delta_pp": class_delta,
        "standalone_cam28_1_delta_miou_pp": 100.0 * (
            standalone_metrics["full"]["mIoU"]
            - standalone_metrics["base"]["mIoU"]
        ),
        "differing_base_full_pixels": int(np.count_nonzero(
            predictions["base"] != predictions["full"]
        )),
        "prediction_sha256": {
            variant: _array_hash(prediction) for variant, prediction in predictions.items()
        },
        "taxonomy": taxonomy,
        "mechanism": mechanism,
        "runtime": {
            "validation_seconds": time.perf_counter() - started,
            "base_forward_seconds": base_forward_seconds,
            "rsbr_refinement_seconds": refinement_seconds,
            "mean_rsbr_refinement_seconds_per_image": refinement_seconds / len(dataset),
            "rsbr_overhead_vs_base_forward_percent": (
                100.0 * refinement_seconds / max(base_forward_seconds, 1e-12)
            ),
        },
        "official_fusion": {"cam56": 0.0, "cam28_1": 0.6, "cam28_2": 0.2, "camdeep": 0.2},
        "official_thresholds": thresholds.tolist(),
        "tta": [[list(left), list(right)] for left, right in _tta_transforms()],
        "optimizer_updates_during_evaluation": 0,
    }
    return result
