#!/usr/bin/env python3
"""Minimal two-batch S²HR-v1 preflight; no optimizer step and no science gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls_s2hr import Net
from network.s2hfrm28_1 import S2HFRM28_1
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset
from tools.train_s2hr_25ep import load_pretrained, set_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    model = Net(4)
    pretrained = load_pretrained(model, args.weights)
    model = model.cuda()
    model.train()
    parameter_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer(
        [
            {"params": parameter_groups[0], "lr": 0.01, "weight_decay": 0.0005},
            {"params": parameter_groups[1], "lr": 0.02, "weight_decay": 0.0},
            {"params": parameter_groups[2], "lr": 0.10, "weight_decay": 0.0005},
            {"params": parameter_groups[3], "lr": 0.20, "weight_decay": 0.0},
        ],
        lr=0.01,
        weight_decay=0.0005,
        max_step=(23422 // 20) * 25,
    )
    optimizer_groups = [
        {
            "lr": group["lr"], "momentum": group["momentum"],
            "weight_decay": group["weight_decay"],
            "parameters": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]
    if any(group["momentum"] != 0.0005 for group in optimizer_groups):
        raise AssertionError("Preflight optimizer no longer matches released recipe")

    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    if len(dataset) != 23422:
        raise AssertionError(f"Unexpected parsed BCSS training count: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=20, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    batch_records = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch_index, (_, images, labels) in enumerate(loader):
            if batch_index == 2:
                break
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images, image_label=labels, mode="train")
                logits = outputs[:4]
                loss_components = [
                    F.multilabel_soft_margin_loss(output, labels) for output in logits
                ]
                loss = sum(
                    weight * value
                    for weight, value in zip((0.10, 0.15, 0.25, 0.50), loss_components)
                )
            expected_shapes = [(20, 4)] * 4
            if [tuple(value.shape) for value in logits] != expected_shapes:
                raise AssertionError([value.shape for value in logits])
            tensors = [*outputs[:9], loss]
            if not all(torch.isfinite(value).all() for value in tensors):
                raise FloatingPointError(f"Non-finite preflight batch {batch_index + 1}")
            resolved = model.last_s2hr_diagnostics["resolved_present_mask"]
            if not torch.equal(resolved, labels):
                raise AssertionError("Training forward did not use GT image-level presence")
            batch_records.append({
                "batch": batch_index + 1,
                "loss": float(loss.float().item()),
                "logit_shapes": [list(value.shape) for value in logits],
                "cam_shapes": [list(value.shape) for value in outputs[5:9]],
                "boundary_fraction": float(
                    model.last_s2hr_diagnostics["boundary_fraction"].float().item()
                ),
            })

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        connectivity_outputs = model(images[:1], image_label=labels[:1], mode="train")
        connectivity_loss = sum(
            weight * F.multilabel_soft_margin_loss(output, labels[:1])
            for weight, output in zip((0.10, 0.15, 0.25, 0.50), connectivity_outputs[:4])
        )
    connectivity_gradients = torch.autograd.grad(
        connectivity_loss,
        (
            model.hfrm_28_1.gamma_veto,
            model.hfrm_28_1.gamma_context,
            model.hfrm_28_1.gamma_spatial,
        ),
        allow_unused=False,
    )
    gradient_record = {
        name: float(gradient.detach().float().abs().item())
        for name, gradient in zip(
            ("gamma_veto", "gamma_context", "gamma_spatial"),
            connectivity_gradients,
        )
    }
    if not all(np.isfinite(value) and value > 0.0 for value in gradient_record.values()):
        raise AssertionError(f"Residual connectivity failed: {gradient_record}")

    module = S2HFRM28_1().eval()
    torch.manual_seed(123)
    feature = torch.randn(1, 512, 8, 8)
    deep = torch.randn(1, 4096, 8, 8)
    deep_cam = torch.randn(1, 4, 8, 8)
    raw_cam = torch.randn(1, 4, 8, 8)
    presence = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    classifier = torch.randn(4, 512, 1, 1)
    identity, _ = module(feature, deep, deep_cam, raw_cam, presence, classifier)
    identity_max_difference = float((identity - feature).abs().max().item())
    if identity_max_difference != 0.0:
        raise AssertionError(f"Zero-init identity failed: {identity_max_difference}")

    single = torch.zeros(1, 8, 8, dtype=torch.long)
    two_region = single.clone(); two_region[:, :, 4:] = 1
    single_band = S2HFRM28_1.semantic_boundary_band(single)
    two_band = S2HFRM28_1.semantic_boundary_band(two_region)
    if single_band.count_nonzero() != 0 or two_band.count_nonzero() == 0:
        raise AssertionError("Boundary-band preflight failed")
    source = (REPO_ROOT / "network" / "resnet38_cls_s2hr.py").read_text(encoding="utf-8")
    forbidden = (
        "segmentation_gt", "boundary_gt", "pseudo_mask", "region_gt",
        "prototype", "transformer", "gat", "osmf", "rsbr", "rgr", "crra",
    )
    found = [token for token in forbidden if token in source.lower()]
    if found:
        raise AssertionError(f"Forbidden dense/legacy input or module: {found}")

    result = {
        "decision": "S2HR_MINIMAL_PREFLIGHT_PASS",
        "pretrained": pretrained,
        "optimizer_built": True,
        "optimizer_step": False,
        "optimizer_groups": optimizer_groups,
        "batches": batch_records,
        "single_sample_residual_gradient_abs": gradient_record,
        "zero_init_identity_max_abs_difference": identity_max_difference,
        "gamma_veto_init": float(module.gamma_veto.item()),
        "gamma_context_init": float(module.gamma_context.item()),
        "gamma_spatial_init": float(module.gamma_spatial.item()),
        "rho_boundary_init": float(torch.sigmoid(module.rho_boundary_raw).item()),
        "single_class_boundary_pixels": int(single_band.count_nonzero().item()),
        "two_region_boundary_pixels": int(two_band.count_nonzero().item()),
        "dense_gt_enters_model": False,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "all_finite": True,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("S2HR_MINIMAL_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
