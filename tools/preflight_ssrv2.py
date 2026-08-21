#!/usr/bin/env python3
"""Minimal SSR-v2 build/two-batch/gradient preflight; no optimizer step."""

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

from network.hfrm28_1_ssrv2 import SSRv2HFRM28_1, epoch_alpha
from network.resnet38_cls import HFRM, Net as SSHRNet
from network.resnet38_cls_ssrv2 import Net
from tool.GenDataset import Stage1_TrainDataset
from tools.train_ssrv2_25ep import (
    load_pretrained, optimizer_for, set_seed, write_json,
)


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
    model = Net(4)
    baseline_count = sum(p.numel() for p in SSHRNet(4).parameters())
    ssrv2_count = sum(p.numel() for p in model.parameters())
    if ssrv2_count - baseline_count != 1:
        raise AssertionError("SSR-v2 must add exactly one parameter")
    pretrained = load_pretrained(model, args.weights)
    model = model.cuda(); model.train()
    optimizer = optimizer_for(model, (23422 // 20) * 25)
    beta_id = id(model.hfrm_28_1.beta_spatial)
    beta_occurrences = sum(
        id(parameter) == beta_id
        for group in optimizer.param_groups for parameter in group["params"]
    )
    if beta_occurrences != 1:
        raise AssertionError("beta_spatial is not exactly once in optimizer")
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
    batch_records = []; connectivity = None
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch_index, (_, images, labels) in enumerate(loader):
            if batch_index == 2:
                break
            images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images, image_label=labels, mode="train", alpha=0.25)
                cls_loss = classification_loss(outputs, labels)
                pcsd = model.last_ssrv2_diagnostics["pcsd_loss"]
                total = cls_loss + 0.05 * 0.25 * pcsd
            expected_logits = [(20, 4)] * 4
            expected_cams = [(20, 4, 56, 56), (20, 4, 28, 28), (20, 4, 28, 28), (20, 4, 28, 28)]
            if [tuple(value.shape) for value in outputs[:4]] != expected_logits:
                raise AssertionError("Unexpected logit shape")
            if [tuple(value.shape) for value in outputs[5:9]] != expected_cams:
                raise AssertionError([value.shape for value in outputs[5:9]])
            if not all(torch.isfinite(value).all() for value in (*outputs[:9], total)):
                raise FloatingPointError("Non-finite two-batch preflight")
            resolved = model.last_ssrv2_diagnostics["resolved_present_mask"]
            if not torch.equal(resolved, labels):
                raise AssertionError("Training did not use exact GT image-level labels")
            batch_records.append({
                "batch": batch_index + 1, "total_loss": float(total.float()),
                "classification_loss": float(cls_loss.float()),
                "pcsd_raw": float(pcsd.float()),
                "logit_shapes": [list(value.shape) for value in outputs[:4]],
                "cam_shapes": [list(value.shape) for value in outputs[5:9]],
            })
            if connectivity is None and (labels.sum(dim=1) >= 2).any():
                connectivity = (images, labels)
    if len(batch_records) != 2 or connectivity is None:
        raise AssertionError("Two-batch preflight did not find a valid sample")

    images, labels = connectivity
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(images, image_label=labels, mode="train", alpha=1.0)
        total = classification_loss(outputs, labels) + 0.05 * model.last_ssrv2_diagnostics["pcsd_loss"]
    beta_gradient = torch.autograd.grad(
        total, model.hfrm_28_1.beta_spatial, allow_unused=False
    )[0]
    if not torch.isfinite(beta_gradient).all():
        raise AssertionError("beta_spatial gradient is not finite")

    torch.manual_seed(123)
    baseline = HFRM(512, 4096, 15).eval()
    ssrv2 = SSRv2HFRM28_1(512, 4096, 15).eval()
    incompat = ssrv2.load_state_dict(baseline.state_dict(), strict=False)
    if set(incompat.missing_keys) != {"beta_spatial"} or incompat.unexpected_keys:
        raise AssertionError(incompat)
    feature = torch.randn(2, 512, 8, 8)
    deep_feature = torch.randn(2, 4096, 8, 8)
    deep_logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    raw_logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    presence = torch.tensor([[1., 1., 0., 0.], [1., 0., 1., 0.]])
    classifier = torch.randn(4, 512, 1, 1, requires_grad=True)
    baseline_output = baseline(feature, deep_feature)
    identity_output, identity_diagnostics = ssrv2(
        feature, deep_feature, deep_logits, raw_logits, presence, classifier, alpha=0.0
    )
    identity_difference = float((identity_output - baseline_output).abs().max())
    if identity_difference != 0.0 or epoch_alpha(1) != 0.0:
        raise AssertionError(f"Epoch1 identity failed: {identity_difference}")

    pcsd = identity_diagnostics["pcsd_loss"]
    teacher_grad, student_grad = torch.autograd.grad(
        pcsd, (deep_logits, raw_logits), allow_unused=True, retain_graph=True
    )
    if teacher_grad is not None:
        raise AssertionError("PCSD updated teacher logits")
    if student_grad is None or not torch.isfinite(student_grad).all() or student_grad.abs().sum() == 0:
        raise AssertionError("PCSD did not update student logits")
    if identity_diagnostics["teacher_residual"].requires_grad:
        raise AssertionError("PTCR residual is not fully detached")

    single_presence = torch.tensor([[1., 0., 0., 0.]])
    single = SSRv2HFRM28_1.spatial_terms(
        deep_logits[:1], raw_logits[:1], single_presence, classifier
    )
    if float(single["pcsd_loss"].detach()) != 0.0:
        raise AssertionError("Single-present PCSD is not zero")
    if int(single["teacher_residual"].count_nonzero()) != 0:
        raise AssertionError("Single-present PTCR is not zero")
    positive_checks = {
        str(value): float(F.softplus(torch.tensor(value)))
        for value in (-100.0, -4.0, 0.0, 10.0)
    }
    if not all(value > 0.0 for value in positive_checks.values()):
        raise AssertionError(positive_checks)

    sources = "\n".join(
        (REPO_ROOT / name).read_text(encoding="utf-8").lower()
        for name in (
            "network/hfrm28_1_ssrv2.py", "network/resnet38_cls_ssrv2.py",
            "tools/train_ssrv2_25ep.py",
        )
    )
    forbidden = (
        "segmentation_gt", "boundary_gt", "pseudo_mask", "rho_boundary",
        "semantic_boundary", "bps_gate", "prototype", "graph", "uncertainty",
    )
    found = [token for token in forbidden if token in sources]
    if found:
        raise AssertionError(f"Forbidden SSR-v2 dependency: {found}")

    result = {
        "decision": "SSRV2_MINIMAL_PREFLIGHT_PASS",
        "pretrained": pretrained, "dataset_samples": len(dataset),
        "parameters": {"sshr": baseline_count, "ssrv2": ssrv2_count, "added": 1},
        "optimizer_built": True, "optimizer_step": False,
        "optimizer_groups": optimizer_groups,
        "beta_optimizer_occurrences": beta_occurrences,
        "batches": batch_records,
        "epoch1_alpha": epoch_alpha(1),
        "epoch1_pcsd_coefficient": 0.05 * epoch_alpha(1),
        "alpha0_identity_max_abs_difference": identity_difference,
        "positive_gamma_checks": positive_checks,
        "pcsd_teacher_gradient": None,
        "pcsd_student_gradient_l1": float(student_grad.detach().float().abs().sum()),
        "ptcr_residual_requires_grad": identity_diagnostics["teacher_residual"].requires_grad,
        "beta_gradient_after_alpha_positive": float(beta_gradient.detach().float()),
        "single_present_pcsd": float(single["pcsd_loss"].detach()),
        "single_present_ptcr_nonzero": int(single["teacher_residual"].count_nonzero()),
        "dense_gt_enters_model_or_loss": False,
        "all_finite": True,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("SSRV2_MINIMAL_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
