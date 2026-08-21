"""Shared official BCSS inference for SSHR A0 and S²HR-v1."""

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
    label = (probability > BCSS_THRESHOLDS).astype(np.float32)
    if label.sum() == 0:
        label[int(np.argmax(probability))] = 1.0
    return label


def _normalize(cam):
    minimum = cam.min(axis=(1, 2), keepdims=True)
    maximum = cam.max(axis=(1, 2), keepdims=True)
    return (cam - minimum) / (maximum - minimum + 1.0e-8)


def _resize_and_unflip(cam, original_size, output_dims):
    cam = F.interpolate(cam, original_size, mode="bilinear", align_corners=False)[0]
    return torch.flip(cam, dims=output_dims) if output_dims else cam


def infer_bcss(model, dataroot, amp_dtype="bf16", num_workers=4):
    """Evaluate either released SSHR or S²HR with one frozen postprocess."""

    model = model.cuda()
    model.eval()
    dataset = Stage1_InferDataset(
        data_path=os.path.join(dataroot, "img"), img_size=224
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    dtype = _amp_dtype(amp_dtype)
    predictions, ground_truths = [], []
    s2hr_two_pass = hasattr(model, "forward_presence")
    torch.cuda.synchronize()
    started = time.time()
    with torch.no_grad():
        for image_index, (name_tuple, image) in enumerate(loader):
            image_id = name_tuple[0]
            image_path = os.path.join(dataroot, "img", image_id + ".png")
            original_size = np.asarray(Image.open(image_path)).shape[:2]
            image = image.cuda(non_blocking=True)

            probability_views = []
            if s2hr_two_pass:
                for input_dims, _ in TTA_TRANSFORMS:
                    augmented = torch.flip(image, dims=input_dims) if input_dims else image
                    with torch.autocast(
                        device_type="cuda", dtype=dtype, enabled=dtype is not None
                    ):
                        probability_views.append(model.forward_presence(augmented)[0])
                probability_tensor = torch.stack(probability_views).mean(0)
                probability = probability_tensor.detach().float().cpu().numpy()
                label = _presence(probability)
                internal_presence = probability_tensor.new_tensor(label)[None]
            else:
                label = internal_presence = None

            cams_28_1, cams_28_2, cams_deep = [], [], []
            probability_views = []
            for input_dims, output_dims in TTA_TRANSFORMS:
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with torch.autocast(
                    device_type="cuda", dtype=dtype, enabled=dtype is not None
                ):
                    if s2hr_two_pass:
                        _, cam_28_1, cam_28_2, cam_deep, _ = model.forward_cam(
                            augmented, present_mask=internal_presence
                        )
                    else:
                        _, cam_28_1, cam_28_2, cam_deep, y_deep = model.forward_cam(augmented)
                        probability_views.append(y_deep[0])
                cams_28_1.append(_resize_and_unflip(cam_28_1, original_size, output_dims))
                cams_28_2.append(_resize_and_unflip(cam_28_2, original_size, output_dims))
                cams_deep.append(_resize_and_unflip(cam_deep, original_size, output_dims))

            if not s2hr_two_pass:
                probability = torch.stack(probability_views).mean(0).detach().float().cpu().numpy()
                label = _presence(probability)
            cam_28_1 = torch.stack(cams_28_1).mean(0).detach().float().cpu().numpy()
            cam_28_2 = torch.stack(cams_28_2).mean(0).detach().float().cpu().numpy()
            cam_deep = torch.stack(cams_deep).mean(0).detach().float().cpu().numpy()
            response = (
                0.6 * _normalize(cam_28_1)
                + 0.2 * _normalize(cam_28_2)
                + 0.2 * _normalize(cam_deep)
            )
            response *= label.reshape(4, 1, 1)
            predictions.append(response.argmax(axis=0).astype(np.uint8))
            ground_truths.append(
                np.asarray(Image.open(os.path.join(dataroot, "mask", image_id + ".png")))
            )
            if (image_index + 1) % 200 == 0:
                print(f"EVAL_PROGRESS {image_index + 1}/{len(dataset)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    scores = iouutils.scores(ground_truths, predictions, n_class=4)
    return scores, {
        "images": len(dataset),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(dataset),
        "s2hr_tta_presence_two_pass": bool(s2hr_two_pass),
        "precision": amp_dtype,
        "tta": TTA_TRANSFORMS,
        "thresholds": BCSS_THRESHOLDS.tolist(),
        "fusion": [0.0, 0.6, 0.2, 0.2],
    }
