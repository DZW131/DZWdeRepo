"""Frozen official BCSS inference for SSHR A0 and SSR-v2."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

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
    model = model.cuda(); model.eval()
    dataset = Stage1_InferDataset(os.path.join(dataroot, "img"), img_size=224)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    dtype = _amp_dtype(amp_dtype)
    predictions, truths = [], []
    ssrv2_two_pass = hasattr(model, "forward_presence")
    torch.cuda.synchronize(); started = time.time()
    with torch.no_grad():
        for image_index, (name_tuple, image) in enumerate(loader):
            image_id = name_tuple[0]
            original_size = np.asarray(
                Image.open(os.path.join(dataroot, "img", image_id + ".png"))
            ).shape[:2]
            image = image.cuda(non_blocking=True)
            if ssrv2_two_pass:
                probabilities = []
                for input_dims, _ in TTA_TRANSFORMS:
                    augmented = torch.flip(image, dims=input_dims) if input_dims else image
                    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                        probabilities.append(model.forward_presence(augmented)[0])
                probability_tensor = torch.stack(probabilities).mean(0)
                label = _presence(probability_tensor.detach().float().cpu().numpy())
                internal_presence = probability_tensor.new_tensor(label)[None]
            else:
                label = internal_presence = None

            cam_views = {name: [] for name in ("28_1", "28_2", "deep")}
            probabilities = []
            for input_dims, output_dims in TTA_TRANSFORMS:
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    if ssrv2_two_pass:
                        _, cam_28_1, cam_28_2, cam_deep, _ = model.forward_cam(
                            augmented, present_mask=internal_presence
                        )
                    else:
                        _, cam_28_1, cam_28_2, cam_deep, y_deep = model.forward_cam(augmented)
                        probabilities.append(y_deep[0])
                for name, value in (
                    ("28_1", cam_28_1), ("28_2", cam_28_2), ("deep", cam_deep)
                ):
                    cam_views[name].append(_resize_unflip(value, original_size, output_dims))
            if not ssrv2_two_pass:
                probability = torch.stack(probabilities).mean(0).detach().float().cpu().numpy()
                label = _presence(probability)
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
        "seconds_per_image": elapsed / len(dataset),
        "ssrv2_tta_presence_two_pass": bool(ssrv2_two_pass),
        "precision": amp_dtype, "tta": TTA_TRANSFORMS,
        "thresholds": BCSS_THRESHOLDS.tolist(), "fusion": [0.0, 0.6, 0.2, 0.2],
    }


def diagnose_gt_present_teacher(model, dataroot, amp_dtype="bf16", num_workers=4):
    """Validation-only TTA teacher/student accuracy; GT never enters forward."""

    model = model.cuda(); model.eval()
    dataset = Stage1_InferDataset(os.path.join(dataroot, "img"), img_size=224)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    dtype = _amp_dtype(amp_dtype)
    counts = {"foreground": 0, "deep_correct": 0, "raw28_1_correct": 0}
    with torch.no_grad():
        for name_tuple, image in loader:
            image_id = name_tuple[0]
            truth = np.asarray(Image.open(os.path.join(dataroot, "mask", image_id + ".png")))
            image = image.cuda(non_blocking=True)
            deep_views, raw_views = [], []
            for input_dims, output_dims in TTA_TRANSFORMS:
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    deep, raw = model.forward_teacher_logits(augmented)
                deep_views.append(_resize_unflip(deep, truth.shape, output_dims))
                raw_views.append(_resize_unflip(raw, truth.shape, output_dims))
            deep = torch.stack(deep_views).mean(0).detach().float().cpu().numpy()
            raw = torch.stack(raw_views).mean(0).detach().float().cpu().numpy()
            presence = np.asarray([np.any(truth == index) for index in range(4)])
            if not presence.any():
                continue
            deep[~presence] = -1.0e4; raw[~presence] = -1.0e4
            deep_prediction = deep.argmax(axis=0); raw_prediction = raw.argmax(axis=0)
            foreground = truth < 4
            counts["foreground"] += int(foreground.sum())
            counts["deep_correct"] += int((foreground & (deep_prediction == truth)).sum())
            counts["raw28_1_correct"] += int((foreground & (raw_prediction == truth)).sum())
    return {
        **counts,
        "deep_accuracy": counts["deep_correct"] / counts["foreground"],
        "raw28_1_accuracy": counts["raw28_1_correct"] / counts["foreground"],
        "teacher_advantage_pp": 100 * (
            counts["deep_correct"] - counts["raw28_1_correct"]
        ) / counts["foreground"],
        "presence": "GT-present diagnostic only",
        "tta": 3,
    }
