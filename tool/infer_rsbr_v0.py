"""Validation-only RSBR-v0 inference under the released SSHR protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from tool import infer_utils, iouutils
from tool.GenDataset import Stage1_InferDataset
from tool.infer_fun import _get_class_thresholds, _tta_transforms


@dataclass
class RSBRInferenceResult:
    metrics: Dict
    standalone_metrics: Dict
    names: List[str]
    ground_truth: np.ndarray
    predictions: np.ndarray
    base_predictions: np.ndarray
    diagnostics: Dict[str, float]
    maximum_cam_differences: Dict[str, float]


def _normalize(cam: torch.Tensor) -> np.ndarray:
    array = cam.detach().float().cpu().numpy()
    minimum = np.min(array, axis=(1, 2), keepdims=True)
    maximum = np.max(array, axis=(1, 2), keepdims=True)
    return (array - minimum) / (maximum - minimum + 1e-8)


def _prediction(cam: np.ndarray, label: np.ndarray, image: np.ndarray) -> np.ndarray:
    cam = cam * label.reshape(4, 1, 1)
    cam_dict = infer_utils.cam_npy_to_cam_dict(cam, label)
    cam_score, _ = infer_utils.dict2npy(cam_dict, label, image)
    return infer_utils.cam_npy_to_label_map(cam_score).astype(np.uint8)


def _average_statistics(rows):
    if not rows:
        return {}
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in sorted(rows[0])
    }


def infer_rsbr_validation(model, dataroot, args) -> RSBRInferenceResult:
    if getattr(args, "dataset", None) != "bcss":
        raise ValueError("RSBR-v0 inference is restricted to BCSS")
    if "test" in str(dataroot).lower():
        raise ValueError("RSBR-v0 pilot must not access test")

    model.eval()
    model = model.cuda()
    dataset = Stage1_InferDataset(
        data_path=str(dataroot) + "/img/", img_size=args.img_size
    )
    if len(dataset) != 3418:
        raise RuntimeError(f"BCSS validation count {len(dataset)} != 3418")
    loader = DataLoader(
        dataset, shuffle=False, batch_size=1,
        num_workers=args.num_workers, pin_memory=True,
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else None
    thresholds = _get_class_thresholds(args, None, 4)
    names, ground_truth = [], []
    predictions, base_predictions, standalone_predictions = [], [], []
    statistic_rows = []
    max_diffs = {"CAM56": 0.0, "CAM28_1": 0.0, "CAM28_2": 0.0, "CAMdeep": 0.0}

    with torch.no_grad():
        for image_index, (name_tuple, image_tensor) in enumerate(loader):
            image_name = name_tuple[0]
            image = np.asarray(
                Image.open(str(dataroot) + "/img/" + image_name + ".png").convert("RGB")
            )
            original_size = image.shape[:2]
            image_tensor = image_tensor.cuda(non_blocking=True)
            bases, probabilities = [], []
            for input_flip_dims, _ in _tta_transforms():
                tta_image = (
                    torch.flip(image_tensor, dims=input_flip_dims)
                    if input_flip_dims else image_tensor
                )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
                ):
                    base = model.forward_cam_base(tta_image)
                bases.append(base)
                probabilities.append(base[4])

            probability = torch.stack(probabilities).mean(0).detach().float().cpu().numpy()[0]
            label = (probability > thresholds).astype(np.float32)
            if label.sum() == 0:
                label[int(np.argmax(probability))] = 1.0
            presence = torch.from_numpy(label).to(image_tensor.device)[None]

            refined_tta = {name: [] for name in ("56", "28_1", "28_2", "deep")}
            base_tta = {name: [] for name in ("56", "28_1", "28_2", "deep")}
            for base, (_, cam_flip_dims) in zip(bases, _tta_transforms()):
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
                ):
                    result = model.refine_from_base(base, presence)
                    base_cams = (F.relu(base[0]), F.relu(base[1]), F.relu(base[2]), F.relu(base[3]))
                    refined_cams = (
                        base_cams[0], F.relu(result.refined_cam), base_cams[2], base_cams[3]
                    )
                statistic_rows.append(result.statistics)
                for key, base_cam, refined_cam in zip(
                    ("56", "28_1", "28_2", "deep"), base_cams, refined_cams
                ):
                    base_up = F.interpolate(
                        base_cam, original_size, mode="bilinear", align_corners=False
                    )[0]
                    refined_up = F.interpolate(
                        refined_cam, original_size, mode="bilinear", align_corners=False
                    )[0]
                    if cam_flip_dims:
                        base_up = torch.flip(base_up, dims=cam_flip_dims)
                        refined_up = torch.flip(refined_up, dims=cam_flip_dims)
                    base_tta[key].append(base_up)
                    refined_tta[key].append(refined_up)
                    diff_name = {
                        "56": "CAM56", "28_1": "CAM28_1",
                        "28_2": "CAM28_2", "deep": "CAMdeep",
                    }[key]
                    max_diffs[diff_name] = max(
                        max_diffs[diff_name],
                        float((refined_up - base_up).detach().float().abs().max().item()),
                    )

            averaged_base = {
                key: torch.stack(value).mean(0) for key, value in base_tta.items()
            }
            averaged_refined = {
                key: torch.stack(value).mean(0) for key, value in refined_tta.items()
            }
            base_fusion = (
                0.6 * _normalize(averaged_base["28_1"])
                + 0.2 * _normalize(averaged_base["28_2"])
                + 0.2 * _normalize(averaged_base["deep"])
            )
            refined_fusion = (
                0.6 * _normalize(averaged_refined["28_1"])
                + 0.2 * _normalize(averaged_refined["28_2"])
                + 0.2 * _normalize(averaged_refined["deep"])
            )
            names.append(image_name)
            base_predictions.append(_prediction(base_fusion, label, image))
            predictions.append(_prediction(refined_fusion, label, image))
            standalone_predictions.append(
                _prediction(_normalize(averaged_refined["28_1"]), label, image)
            )
            ground_truth.append(np.asarray(
                Image.open(str(dataroot) + "/mask/" + image_name + ".png"),
                dtype=np.uint8,
            ))
            if (image_index + 1) % 250 == 0:
                print(f"RSBR_VAL_PROGRESS {image_index + 1}/{len(dataset)}", flush=True)

    gt_array = np.stack(ground_truth)
    prediction_array = np.stack(predictions)
    base_array = np.stack(base_predictions)
    standalone_array = np.stack(standalone_predictions)
    metrics = iouutils.scores(
        [item.copy() for item in gt_array],
        [item.copy() for item in prediction_array], n_class=4,
    )
    standalone_metrics = iouutils.scores(
        [item.copy() for item in gt_array],
        [item.copy() for item in standalone_array], n_class=4,
    )
    return RSBRInferenceResult(
        metrics=metrics,
        standalone_metrics=standalone_metrics,
        names=names,
        ground_truth=gt_array,
        predictions=prediction_array,
        base_predictions=base_array,
        diagnostics=_average_statistics(statistic_rows),
        maximum_cam_differences=max_diffs,
    )
