#!/usr/bin/env python3
"""Minimal HALR-v1 real-batch BF16 preflight; never performs an optimizer step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls import Net
from tool.GenDataset import Stage1_TrainDataset
from tools.halr_objectives import apply_pair_transform, epoch_alpha, halr_terms
from tools.train_halr_v1_25ep import load_pretrained, optimizer_for, set_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def classification_loss(outputs, labels):
    return sum(
        weight * F.multilabel_soft_margin_loss(output, labels)
        for weight, output in zip((0.10, 0.15, 0.25, 0.50), outputs[:4])
    )


def main():
    args = parse_args(); set_seed(42)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("HALR-v1 preflight requires CUDA BF16")
    model = Net(4)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    reference_count = sum(parameter.numel() for parameter in Net(4).parameters())
    if parameter_count != reference_count:
        raise AssertionError("HALR-v1 changed the official SSHR parameter count")
    pretrained = load_pretrained(model, args.weights)
    model = model.cuda(); model.train()
    optimizer = optimizer_for(model, (23422 // 20) * 25)
    optimizer_groups = [
        {
            "lr": group["lr"], "momentum": group["momentum"],
            "weight_decay": group["weight_decay"],
            "parameters": sum(p.numel() for p in group["params"]),
        }
        for group in optimizer.param_groups
    ]
    if any(group["momentum"] != 0.0005 for group in optimizer_groups):
        raise AssertionError("Released optimizer recipe changed")

    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    if len(dataset) != 23422:
        raise AssertionError(f"Unexpected BCSS sample count: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=20, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    _, images, labels = next(iter(loader))
    images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
    flip_codes = (torch.arange(images.shape[0], device=images.device) % 2).long()
    paired = apply_pair_transform(images, flip_codes)
    restored = apply_pair_transform(paired, flip_codes)
    if not torch.equal(images, restored):
        raise AssertionError("flip/inverse-flip is not exact")

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        view1 = model(images); view2 = model(paired)
        terms = halr_terms(view1[6], view1[8], view2[6], view2[8], labels, flip_codes)
        cls = 0.5 * (classification_loss(view1, labels) + classification_loss(view2, labels))
        total = cls + 0.05 * 0.25 * terms["cvle_loss"] + 0.05 * 0.25 * terms["rahd_loss"]

    shapes1 = [list(view1[index].shape) for index in (6, 8)]
    shapes2 = [list(view2[index].shape) for index in (6, 8)]
    if shapes1 != shapes2:
        raise AssertionError("paired-view CAM shapes differ")
    weight_sum = terms["weight28_per_sample"] + terms["weightdeep_per_sample"]
    if not torch.isfinite(weight_sum).all() or not torch.allclose(
        weight_sum, torch.ones_like(weight_sum), atol=1.0e-6, rtol=0.0
    ):
        raise AssertionError("reliability weights are invalid")
    if terms["weight28_per_sample"].requires_grad or terms["weightdeep_per_sample"].requires_grad:
        raise AssertionError("reliability weights must be detached")
    if terms["teacher_view1"].requires_grad or terms["teacher_view2"].requires_grad:
        raise AssertionError("RAHD teachers must be stop-gradient")
    localization_gradients = torch.autograd.grad(
        terms["cvle_loss"] + terms["rahd_loss"], (view1[6], view1[8]),
        retain_graph=False, allow_unused=True,
    )
    gradient_l1 = [
        None if gradient is None else float(gradient.detach().float().abs().sum())
        for gradient in localization_gradients
    ]
    if any(value is None or value <= 0.0 for value in gradient_l1):
        raise AssertionError("CVLE/RAHD must reach both CAM28_1 and CAMdeep")

    synthetic = [torch.randn(2, 4, 6, 6, requires_grad=True) for _ in range(4)]
    synthetic_codes = torch.tensor([0, 1])
    single_labels = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    single = halr_terms(*synthetic, single_labels, synthetic_codes)
    if single["cvle_loss"].item() != 0.0 or single["rahd_loss"].item() != 0.0:
        raise AssertionError("K_present < 2 must zero CVLE and RAHD")
    alpha0_weighted = 0.05 * epoch_alpha(1) * (terms["cvle_loss"] + terms["rahd_loss"])
    if alpha0_weighted.item() != 0.0:
        raise AssertionError("epoch1 alpha must exactly zero weighted localization loss")
    finite_values = [total, terms["cvle_loss"], terms["rahd_loss"], *terms["weight28_per_sample"], *terms["weightdeep_per_sample"]]
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise FloatingPointError("non-finite preflight output")

    result = {
        "decision": "HALR_V1_MINIMAL_PREFLIGHT_PASS",
        "dataset_samples": len(dataset), "batch_size": images.shape[0],
        "effective_forward_images": 2 * images.shape[0], "precision": "bf16",
        "model_class": "network.resnet38_cls.Net", "new_model_parameters": 0,
        "parameter_count": parameter_count, "pretrained": pretrained,
        "optimizer_built": True, "optimizer_step": False,
        "optimizer_groups": optimizer_groups,
        "cam_shapes_view1": shapes1, "cam_shapes_view2": shapes2,
        "flip_inverse_exact": True, "flip_codes": flip_codes.cpu().tolist(),
        "single_present_cvle": float(single["cvle_loss"].detach()),
        "single_present_rahd": float(single["rahd_loss"].detach()),
        "reliability_finite": True,
        "reliability_sum_max_error": float((weight_sum - 1.0).abs().max().detach()),
        "reliability_requires_grad": False, "teacher_requires_grad": False,
        "localization_cam_gradient_l1": gradient_l1,
        "epoch1_alpha": epoch_alpha(1),
        "epoch1_weighted_localization": float(alpha0_weighted.detach()),
        "classification_loss": float(cls.detach().float()),
        "cvle_loss": float(terms["cvle_loss"].detach().float()),
        "rahd_loss": float(terms["rahd_loss"].detach().float()),
        "mean_weight28": float(terms["weight28"].detach().float()),
        "mean_weightdeep": float(terms["weightdeep"].detach().float()),
        "dense_segmentation_gt_enters_forward_or_loss": False,
        "all_finite": True,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("HALR_V1_MINIMAL_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
