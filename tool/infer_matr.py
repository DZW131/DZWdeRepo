"""Frozen official BCSS inference with standalone CAM28_1 readout."""

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
    fused_predictions, standalone_predictions, truths = [], [], []
    torch.cuda.synchronize(); started = time.time()
    with torch.no_grad():
        for image_index, (name_tuple, image) in enumerate(loader):
            image_id = name_tuple[0]
            original_size = np.asarray(
                Image.open(os.path.join(dataroot, "img", image_id + ".png"))
            ).shape[:2]
            image = image.cuda(non_blocking=True)
            views = {name: [] for name in ("28_1", "28_2", "deep")}
            probabilities = []
            for input_dims, output_dims in TTA_TRANSFORMS:
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    _, cam28, cam28_2, camdeep, probability = model.forward_cam(augmented)
                for name, value in (("28_1", cam28), ("28_2", cam28_2), ("deep", camdeep)):
                    views[name].append(_resize_unflip(value, original_size, output_dims))
                probabilities.append(probability[0])
            label = _presence(
                torch.stack(probabilities).mean(0).detach().float().cpu().numpy()
            )
            cams = {
                name: _normalize(torch.stack(values).mean(0).detach().float().cpu().numpy())
                for name, values in views.items()
            }
            fused = 0.6 * cams["28_1"] + 0.2 * cams["28_2"] + 0.2 * cams["deep"]
            fused *= label.reshape(4, 1, 1)
            standalone = cams["28_1"] * label.reshape(4, 1, 1)
            fused_predictions.append(fused.argmax(axis=0).astype(np.uint8))
            standalone_predictions.append(standalone.argmax(axis=0).astype(np.uint8))
            truths.append(np.asarray(Image.open(os.path.join(dataroot, "mask", image_id + ".png"))))
            if (image_index + 1) % 200 == 0:
                print(f"EVAL_PROGRESS {image_index + 1}/{len(dataset)}", flush=True)
    torch.cuda.synchronize(); elapsed = time.time() - started
    return (
        iouutils.scores(truths, fused_predictions, n_class=4),
        iouutils.scores(truths, standalone_predictions, n_class=4),
        {
            "images": len(dataset), "elapsed_seconds": elapsed,
            "seconds_per_image": elapsed / len(dataset), "precision": amp_dtype,
            "tta": TTA_TRANSFORMS, "thresholds": BCSS_THRESHOLDS.tolist(),
            "fusion": [0.0, 0.6, 0.2, 0.2],
        },
    )
