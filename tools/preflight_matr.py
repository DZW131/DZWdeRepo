#!/usr/bin/env python3
"""Minimal real-batch BF16 MATR-v1 preflight; no optimizer step."""

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

from network.matr_multiprototype_head import MultiPrototypeCAMHead
from network.resnet38_cls import Net as SSHRNet
from network.resnet38_cls_matr import Net
from tool.GenDataset import Stage1_TrainDataset
from tools.matr_objectives import epoch_alpha, ot_mtr_loss
from tools.train_matr_25ep import load_pretrained, optimizer_for, set_seed, write_json


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
        raise RuntimeError("MATR preflight requires CUDA BF16")
    baseline_count = sum(parameter.numel() for parameter in SSHRNet(4).parameters())
    model = Net(4)
    matr_count = sum(parameter.numel() for parameter in model.parameters())
    overhead = matr_count - baseline_count
    overhead_percent = 100.0 * overhead / baseline_count
    if overhead <= 0 or overhead_percent >= 0.10:
        raise AssertionError("MATR parameter budget failed")
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
    raw_optimizer_occurrences = {}
    for name, parameter in (
        ("d_raw", model.ic1.d_raw),
        ("a_logits", model.hfrm_28_1.sacr.a_logits),
        ("beta_adapt", model.hfrm_28_1.sacr.beta_adapt),
    ):
        raw_optimizer_occurrences[name] = sum(
            id(parameter) == id(item)
            for group in optimizer.param_groups for item in group["params"]
        )
    if any(value != 1 for value in raw_optimizer_occurrences.values()):
        raise AssertionError("MATR raw parameter optimizer coverage failed")

    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    if len(dataset) != 23422:
        raise AssertionError(f"Unexpected BCSS sample count: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=20, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    _, images, labels = next(iter(loader))
    images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(images)
        cls = classification_loss(outputs, labels)
        ot = ot_mtr_loss(
            outputs[11], outputs[6], outputs[10], model.ic1.mode_weights(), labels
        )
        total = cls + 0.05 * 0.25 * ot["loss"]
    expected_shapes = {
        "cam56": [20, 4, 56, 56], "cam28_1": [20, 4, 28, 28],
        "cam28_2": [20, 4, 28, 28], "camdeep": [20, 4, 28, 28],
        "mode_logits": [20, 4, 2, 28, 28],
    }
    observed_shapes = {
        "cam56": list(outputs[5].shape), "cam28_1": list(outputs[6].shape),
        "cam28_2": list(outputs[7].shape), "camdeep": list(outputs[8].shape),
        "mode_logits": list(outputs[10].shape),
    }
    if observed_shapes != expected_shapes:
        raise AssertionError(f"CAM shape mismatch: {observed_shapes}")
    sacr = model.hfrm_28_1.last_sacr_diagnostics
    if sacr["delta_rms"].item() != 0.0:
        raise AssertionError("Initial SACR delta must be exactly zero")
    if ot["max_row_marginal_error"].item() >= 5.0e-3:
        raise AssertionError("Sinkhorn row marginal error too large")
    if ot["max_col_marginal_error"].item() >= 1.0e-5:
        raise AssertionError("Sinkhorn prototype marginal error too large")

    ot_d_raw_gradient = torch.autograd.grad(
        ot["loss"], model.ic1.d_raw, retain_graph=True, allow_unused=True
    )[0]
    if ot_d_raw_gradient is None or ot_d_raw_gradient.abs().sum() == 0:
        raise AssertionError("First OT-active batch must update D_raw")
    optimizer.zero_grad(); total.backward()
    offset_gradient = model.hfrm_28_1.sacr.predictor[-1].weight.grad
    beta_gradient = model.hfrm_28_1.sacr.beta_adapt.grad
    if offset_gradient is None or offset_gradient.abs().sum() == 0:
        raise AssertionError("SACR final offset/modulation predictor gradient is inactive")
    if beta_gradient is None or not torch.isfinite(beta_gradient).all():
        raise AssertionError("beta_adapt gradient must be finite")

    probe = MultiPrototypeCAMHead(8, 4, 2).cuda()
    with torch.no_grad():
        probe.d_raw.zero_()
    probe_features = torch.randn(2, 8, 5, 5, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        probe_aggregated, _ = probe(probe_features)
        probe_base = probe.base(probe_features)
    exact_head_identity = torch.equal(probe_aggregated, probe_base)
    if not exact_head_identity:
        raise AssertionError("D1=D2=0 must exactly recover the base CAM logit")

    finite_values = (
        total, cls, ot["loss"], sacr["gamma_adapt"],
        sacr["mean_abs_offset"], sacr["mean_modulation"], sacr["delta_rms"],
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise FloatingPointError("Non-finite MATR preflight output")
    result = {
        "decision": "MATR_V1_MINIMAL_PREFLIGHT_PASS",
        "dataset_samples": len(dataset), "batch_size": images.shape[0],
        "precision": "bf16", "cam_shapes": observed_shapes,
        "parameters": {
            "sshr": baseline_count, "matr": matr_count,
            "added": overhead, "overhead_percent": overhead_percent,
        },
        "pretrained": pretrained, "optimizer_built": True,
        "optimizer_step": False, "optimizer_groups": optimizer_groups,
        "raw_optimizer_occurrences": raw_optimizer_occurrences,
        "classification_loss": float(cls.detach().float()),
        "ot_loss": float(ot["loss"].detach().float()),
        "ot_valid_pairs": float(ot["valid_pairs"]),
        "ot_mean_seeds": float(ot["mean_seeds"]),
        "sinkhorn_max_row_error": float(ot["max_row_marginal_error"]),
        "sinkhorn_max_col_error": float(ot["max_col_marginal_error"]),
        "mode_zero_exact_base_identity": exact_head_identity,
        "initial_sacr_delta_rms": float(sacr["delta_rms"]),
        "initial_offset_mean_abs": float(sacr["mean_abs_offset"]),
        "initial_modulation_mean": float(sacr["mean_modulation"]),
        "initial_gamma_adapt": float(sacr["gamma_adapt"]),
        "ot_d_raw_gradient_l1": float(ot_d_raw_gradient.detach().float().abs().sum()),
        "offset_predictor_gradient_l1": float(offset_gradient.detach().float().abs().sum()),
        "beta_adapt_gradient": float(beta_gradient.detach().float()),
        "epoch1_alpha_ot": epoch_alpha(1),
        "epoch1_weighted_ot": float((0.05 * epoch_alpha(1) * ot["loss"]).detach()),
        "dense_segmentation_gt_enters_training": False,
        "all_finite": True,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("MATR_V1_MINIMAL_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
