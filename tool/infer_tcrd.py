"""Official BCSS inference plus observation-only TCRD mechanism diagnostics."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from network.tcrd_dynamics import NEIGHBOR_OFFSETS
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


def _shift(tensor, dy, dx):
    height, width = tensor.shape[-2:]
    padded = F.pad(tensor, (1, 1, 1, 1), mode="constant", value=0)
    return padded[..., 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]


class DiagnosticAccumulator:
    def __init__(self):
        self.sums = {
            "z0_sq": 0.0, "z0_count": 0,
            "diffusion_sq": 0.0, "diffusion_count": 0,
            "reaction_sq": 0.0, "reaction_count": 0,
            "conductance_same": 0.0, "conductance_same_count": 0,
            "conductance_cross": 0.0, "conductance_cross_count": 0,
            "entropy_z0": 0.0, "entropy_zt": 0.0,
            "margin_z0": 0.0, "margin_zt": 0.0,
            "coactive_z0": 0.0, "coactive_zt": 0.0,
            "reaction_pixels": 0,
        }

    def add(self, diagnostics, gt_mask):
        z0 = diagnostics["z0"].float()
        zt = diagnostics["zt"].float()
        diffusion = diagnostics["diffusion_update"].float()
        reaction = diagnostics["reaction_update"].float()
        self.sums["z0_sq"] += z0.square().sum().item()
        self.sums["z0_count"] += z0.numel()
        self.sums["diffusion_sq"] += diffusion.square().sum().item()
        self.sums["diffusion_count"] += diffusion.numel()
        self.sums["reaction_sq"] += reaction.square().sum().item()
        self.sums["reaction_count"] += reaction.numel()

        active = diagnostics.get("active_classes")
        if active is not None:
            active = active[0].bool()
            if int(active.sum()) >= 2:
                for name, evidence in (("z0", z0[0, active]), ("zt", zt[0, active])):
                    probability = torch.softmax(evidence, dim=0)
                    entropy = -(probability * probability.clamp_min(1.0e-8).log()).sum(0)
                    top2 = probability.topk(2, dim=0).values
                    pixels = probability.shape[-2] * probability.shape[-1]
                    self.sums[f"entropy_{name}"] += entropy.sum().item()
                    self.sums[f"margin_{name}"] += (top2[0] - top2[1]).sum().item()
                    self.sums[f"coactive_{name}"] += (top2[1] >= 0.25).sum().item()
                self.sums["reaction_pixels"] += pixels

        conductance = diagnostics.get("conductance")
        if conductance is None:
            return
        height, width = z0.shape[-2:]
        gt = torch.from_numpy(gt_mask.astype(np.float32)).to(z0.device)
        gt = F.interpolate(gt[None, None], (height, width), mode="nearest")[0, 0].long()
        valid_base = torch.ones((1, 1, height, width), device=gt.device)
        for direction, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
            neighbor_gt = _shift(gt[None, None].float(), dy, dx)[0, 0].long()
            valid = _shift(valid_base, dy, dx)[0, 0] > 0
            tissue_pair = valid & (gt < 4) & (neighbor_gt < 4)
            same = tissue_pair & (gt == neighbor_gt)
            cross = tissue_pair & (gt != neighbor_gt)
            values = conductance[0, direction]
            self.sums["conductance_same"] += values[same].sum().item()
            self.sums["conductance_same_count"] += int(same.sum())
            self.sums["conductance_cross"] += values[cross].sum().item()
            self.sums["conductance_cross_count"] += int(cross.sum())

    def result(self):
        z0_rms = (self.sums["z0_sq"] / max(1, self.sums["z0_count"])) ** 0.5
        diffusion_rms = (
            self.sums["diffusion_sq"] / max(1, self.sums["diffusion_count"])
        ) ** 0.5
        reaction_rms = (
            self.sums["reaction_sq"] / max(1, self.sums["reaction_count"])
        ) ** 0.5
        same = self.sums["conductance_same"] / max(
            1, self.sums["conductance_same_count"]
        )
        cross = self.sums["conductance_cross"] / max(
            1, self.sums["conductance_cross_count"]
        )
        pixels = max(1, self.sums["reaction_pixels"])
        return {
            "z0_rms": z0_rms,
            "diffusion_update_rms": diffusion_rms,
            "diffusion_update_ratio": diffusion_rms / max(z0_rms, 1.0e-12),
            "reaction_update_rms": reaction_rms,
            "reaction_update_ratio": reaction_rms / max(z0_rms, 1.0e-12),
            "conductance_same_mean": same,
            "conductance_cross_mean": cross,
            "conductance_same_cross_ratio": same / max(cross, 1.0e-12),
            "conductance_same_edges": self.sums["conductance_same_count"],
            "conductance_cross_edges": self.sums["conductance_cross_count"],
            "present_entropy_z0": self.sums["entropy_z0"] / pixels,
            "present_entropy_zt": self.sums["entropy_zt"] / pixels,
            "present_top1_top2_margin_z0": self.sums["margin_z0"] / pixels,
            "present_top1_top2_margin_zt": self.sums["margin_zt"] / pixels,
            "coactive_overlap_z0": self.sums["coactive_z0"] / pixels,
            "coactive_overlap_zt": self.sums["coactive_zt"] / pixels,
            "coactive_definition": "fraction of pixels with second active-class softmax probability >= 0.25",
        }


def infer_bcss(
    model,
    dataroot: str,
    amp_dtype: str = "bf16",
    num_workers: int = 4,
    prediction_output: str | None = None,
):
    if "test" in dataroot.lower() or "luad" in dataroot.lower():
        raise AssertionError("TCRD utility gate is BCSS validation only")
    model = model.cuda(); model.eval()
    dataset = Stage1_InferDataset(os.path.join(dataroot, "img"), img_size=224)
    if len(dataset) != 3418:
        raise AssertionError(f"Expected 3418 validation images, found {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    dtype = _amp_dtype(amp_dtype)
    fused_predictions, standalone_predictions, truths, image_ids = [], [], [], []
    diagnostic_accumulator = DiagnosticAccumulator()
    torch.cuda.synchronize(); started = time.time()
    with torch.no_grad():
        for image_index, (name_tuple, image) in enumerate(loader):
            image_id = name_tuple[0]
            image_path = os.path.join(dataroot, "img", image_id + ".png")
            mask_path = os.path.join(dataroot, "mask", image_id + ".png")
            original_size = np.asarray(Image.open(image_path)).shape[:2]
            truth = np.asarray(Image.open(mask_path))
            image = image.cuda(non_blocking=True)
            views = {name: [] for name in ("28_1", "28_2", "deep")}
            probabilities = []
            identity_diagnostics = None
            for view_index, (input_dims, output_dims) in enumerate(TTA_TRANSFORMS):
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    result = model.forward_cam(
                        augmented, return_diagnostics=(view_index == 0)
                    )
                if view_index == 0:
                    _, cam28, cam28_2, camdeep, probability, identity_diagnostics = result
                else:
                    _, cam28, cam28_2, camdeep, probability = result
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
            truths.append(truth.astype(np.uint8))
            image_ids.append(image_id)
            diagnostic_accumulator.add(identity_diagnostics, truth)
            if (image_index + 1) % 200 == 0:
                print(f"EVAL_PROGRESS {image_index + 1}/{len(dataset)}", flush=True)
    torch.cuda.synchronize(); elapsed = time.time() - started
    if prediction_output is not None:
        prediction_output = Path(prediction_output)
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_output,
            image_ids=np.asarray(image_ids),
            predictions=np.stack(fused_predictions),
            standalone_predictions=np.stack(standalone_predictions),
            truths=np.stack(truths),
        )
    return (
        iouutils.scores(truths, fused_predictions, n_class=4),
        iouutils.scores(truths, standalone_predictions, n_class=4),
        {
            "images": len(dataset), "elapsed_seconds": elapsed,
            "seconds_per_image": elapsed / len(dataset), "precision": amp_dtype,
            "tta": TTA_TRANSFORMS, "thresholds": BCSS_THRESHOLDS.tolist(),
            "fusion": [0.0, 0.6, 0.2, 0.2],
        },
        diagnostic_accumulator.result(),
    )
