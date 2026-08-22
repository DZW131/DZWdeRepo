"""Frozen official BCSS inference and validation-only HALR diagnostics."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from network.resnet38_cls import Net as TrainingNet
from tool import iouutils
from tool.GenDataset import Stage1_InferDataset


BCSS_THRESHOLDS = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
TTA_TRANSFORMS = (((), ()), ((3,), (2,)), ((2,), (1,)))


def _amp_dtype(name):
    return torch.bfloat16 if name == "bf16" else torch.float16 if name == "fp16" else None


def _presence(probability):
    label = (np.asarray(probability) > BCSS_THRESHOLDS).astype(np.float32)
    if label.sum() == 0:
        label[int(np.argmax(probability))] = 1.0
    return label


def _normalize(cam):
    minimum = cam.min(axis=(1, 2), keepdims=True)
    maximum = cam.max(axis=(1, 2), keepdims=True)
    return (cam - minimum) / (maximum - minimum + 1.0e-8)


def _resize_unflip(cam, size, output_dims):
    cam = F.interpolate(cam, size, mode="bilinear", align_corners=False)[0]
    return torch.flip(cam, dims=output_dims) if output_dims else cam


def infer_bcss(model, dataroot, amp_dtype="bf16", num_workers=4):
    """Run the released single-network 3-way TTA/fusion/metric protocol."""

    model = model.cuda(); model.eval()
    dataset = Stage1_InferDataset(os.path.join(dataroot, "img"), img_size=224)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    dtype = _amp_dtype(amp_dtype)
    predictions, truths = [], []
    torch.cuda.synchronize(); started = time.time()
    with torch.no_grad():
        for image_index, (name_tuple, image) in enumerate(loader):
            image_id = name_tuple[0]
            original_size = np.asarray(
                Image.open(os.path.join(dataroot, "img", image_id + ".png"))
            ).shape[:2]
            image = image.cuda(non_blocking=True)
            cam_views = {name: [] for name in ("28_1", "28_2", "deep")}
            probabilities = []
            for input_dims, output_dims in TTA_TRANSFORMS:
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    _, cam28, cam28_2, camdeep, probability = model.forward_cam(augmented)
                for name, value in (("28_1", cam28), ("28_2", cam28_2), ("deep", camdeep)):
                    cam_views[name].append(_resize_unflip(value, original_size, output_dims))
                probabilities.append(probability[0])
            label = _presence(
                torch.stack(probabilities).mean(0).detach().float().cpu().numpy()
            )
            cams = {
                name: torch.stack(values).mean(0).detach().float().cpu().numpy()
                for name, values in cam_views.items()
            }
            response = (
                0.6 * _normalize(cams["28_1"])
                + 0.2 * _normalize(cams["28_2"])
                + 0.2 * _normalize(cams["deep"])
            )
            response *= label.reshape(4, 1, 1)
            predictions.append(response.argmax(axis=0).astype(np.uint8))
            truths.append(np.asarray(Image.open(os.path.join(dataroot, "mask", image_id + ".png"))))
            if (image_index + 1) % 200 == 0:
                print(f"EVAL_PROGRESS {image_index + 1}/{len(dataset)}", flush=True)
    torch.cuda.synchronize(); elapsed = time.time() - started
    return iouutils.scores(truths, predictions, n_class=4), {
        "images": len(dataset), "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(dataset), "precision": amp_dtype,
        "tta": TTA_TRANSFORMS, "thresholds": BCSS_THRESHOLDS.tolist(),
        "fusion": [0.0, 0.6, 0.2, 0.2], "extra_inference_parameters": 0,
        "double_view_inference": False,
    }


def _four_neighbor_boundary(truth):
    boundary = np.zeros(truth.shape, dtype=bool)
    different = truth[:-1, :] != truth[1:, :]
    boundary[:-1, :] |= different; boundary[1:, :] |= different
    different = truth[:, :-1] != truth[:, 1:]
    boundary[:, :-1] |= different; boundary[:, 1:] |= different
    return boundary & (truth < 4)


def diagnose_gt_present_hierarchy(model, dataroot, amp_dtype="bf16", num_workers=4):
    """GT-present raw-CAM diagnosis; masks enter only after network forward."""

    model = model.cuda(); model.eval()
    dataset = Stage1_InferDataset(os.path.join(dataroot, "img"), img_size=224)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    dtype = _amp_dtype(amp_dtype)
    regions = {
        name: {"pixels": 0, "deep_correct": 0, "raw28_1_correct": 0}
        for name in ("foreground", "boundary", "interior")
    }
    with torch.no_grad():
        for name_tuple, image in loader:
            image_id = name_tuple[0]
            truth = np.asarray(Image.open(os.path.join(dataroot, "mask", image_id + ".png")))
            image = image.cuda(non_blocking=True)
            deep_views, raw_views = [], []
            for input_dims, output_dims in TTA_TRANSFORMS:
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    outputs = TrainingNet.forward(model, augmented)
                    raw28, rawdeep = outputs[6], outputs[8]
                raw_views.append(_resize_unflip(raw28, truth.shape, output_dims))
                deep_views.append(_resize_unflip(rawdeep, truth.shape, output_dims))
            deep = torch.stack(deep_views).mean(0).detach().float().cpu().numpy()
            raw = torch.stack(raw_views).mean(0).detach().float().cpu().numpy()
            presence = np.asarray([np.any(truth == index) for index in range(4)])
            if not presence.any():
                continue
            deep[~presence] = -1.0e4; raw[~presence] = -1.0e4
            deep_prediction = deep.argmax(axis=0); raw_prediction = raw.argmax(axis=0)
            foreground = truth < 4
            boundary = _four_neighbor_boundary(truth)
            masks = {"foreground": foreground, "boundary": boundary, "interior": foreground & ~boundary}
            for name, mask in masks.items():
                regions[name]["pixels"] += int(mask.sum())
                regions[name]["deep_correct"] += int((mask & (deep_prediction == truth)).sum())
                regions[name]["raw28_1_correct"] += int((mask & (raw_prediction == truth)).sum())
    result = {}
    for name, counts in regions.items():
        pixels = counts["pixels"]
        result[name] = {
            **counts,
            "deep_accuracy": counts["deep_correct"] / pixels if pixels else None,
            "raw28_1_accuracy": counts["raw28_1_correct"] / pixels if pixels else None,
            "deep_advantage_pp": 100 * (
                counts["deep_correct"] - counts["raw28_1_correct"]
            ) / pixels if pixels else None,
        }
    return {
        "regions": result, "presence": "GT-present diagnosis only",
        "boundary_definition": "foreground pixels touching a different 4-neighbor label",
        "tta": 3, "gt_enters_forward": False,
    }
