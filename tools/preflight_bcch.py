#!/usr/bin/env python3
"""Real batch20 BF16 no-step preflight for EXP-BCCH-001."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import resnet38_cls
from tools.train_bcch_matched import load_bcch_model
from tools.train_wdch_matched import make_loader
from tools.wdch_common import (
    LOSS_WEIGHTS,
    MatchedAugmentationDataset,
    build_optimizer,
    load_schedule,
    restore_named_optimizer_state,
    restore_rng_state,
    sha256_file,
    write_json,
)


def run(args):
    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    loader = make_loader(dataset, schedule, 20, args.num_workers)
    names, image, target = next(iter(loader))
    if image.shape != (20, 3, 224, 224) or target.shape != (20, 4):
        raise AssertionError((image.shape, target.shape))

    restore_rng_state(common["rng_state"])
    model, load_audit = load_bcch_model(common)
    c0 = resnet38_cls.Net(4)
    if sum(p.numel() for p in model.parameters()) != sum(p.numel() for p in c0.parameters()):
        raise AssertionError("BCCH trainable parameter count differs from C0")
    if not torch.equal(
        model.hfrm_28_1.context_conv.weight.detach().cpu(),
        common["model_state"]["hfrm_28_1.context_conv.weight"],
    ):
        raise AssertionError("Epoch20 CH15 weight was not restored exactly")
    optimizer = build_optimizer(model)
    restore = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    optimizer.global_step = int(common["optimizer_global_step"])
    if restore["skipped"]:
        raise AssertionError(restore)
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(grouped) != len({id(p) for p in grouped}):
        raise AssertionError("Duplicate optimizer parameter")
    if {id(p) for p in grouped} != {id(p) for p in trainable}:
        raise AssertionError("Optimizer coverage differs")

    image = image.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    captured = {}

    def capture_input(_module, inputs):
        captured["feature"] = inputs[0]

    hook = model.hfrm_28_1.register_forward_pre_hook(capture_input)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(image)
        losses = [F.multilabel_soft_margin_loss(output, target) for output in outputs[:4]]
        loss = sum(weight * value for weight, value in zip(LOSS_WEIGHTS, losses))
    hook.remove()
    loss.backward()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        selected, raw, boundary, alpha = model.hfrm_28_1.context_with_maps(
            captured["feature"].detach()
        )
    nonfinite = []
    missing_grad = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_grad.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    unchanged = all(
        torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
    )
    selected_residual = selected.detach().float() - captured["feature"].detach().float()
    raw_residual = raw.detach().float() - captured["feature"].detach().float()
    report = {
        "status": "BCCH_PREFLIGHT_PASS",
        "experiment_id": "EXP-BCCH-001",
        "batch_size": int(image.shape[0]),
        "precision": "bf16",
        "official_loss": float(loss.detach()),
        "outputs_finite": all(bool(torch.isfinite(value).all()) for value in outputs),
        "loss_finite": bool(torch.isfinite(loss)),
        "nonfinite_gradients": nonfinite,
        "missing_gradient_parameter_count": len(missing_grad),
        "context_gradient_norm": float(
            model.hfrm_28_1.context_conv.weight.grad.detach().float().norm()
        ),
        "gamma_veto_gradient": float(model.hfrm_28_1.gamma_veto.grad.detach().float()),
        "gamma_context_gradient": float(model.hfrm_28_1.gamma_context.grad.detach().float()),
        "boundary_map_detached": not boundary.requires_grad,
        "boundary_map": {
            "mean": float(boundary.float().mean()),
            "std": float(boundary.float().std(unbiased=False)),
            "min": float(boundary.float().min()),
            "max": float(boundary.float().max()),
        },
        "alpha": {
            "mean": float(alpha.float().mean()),
            "std": float(alpha.float().std(unbiased=False)),
            "min": float(alpha.float().min()),
            "max": float(alpha.float().max()),
        },
        "raw_ch_residual_rms": float(raw_residual.square().mean().sqrt()),
        "selected_ch_residual_rms": float(selected_residual.square().mean().sqrt()),
        "optimizer_coverage_exactly_once": True,
        "optimizer_state_restored_without_skip": True,
        "optimizer_step_performed": False,
        "parameters_unchanged": unchanged,
        "trainable_parameters_equal_c0": True,
        "context_weight_restored_exactly": True,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "schedule_sha256": sha256_file(args.schedule),
        "model_load_audit": load_audit,
        "batch_names": list(names),
        "test_used": False,
        "luad_used": False,
    }
    if (
        not report["outputs_finite"]
        or not report["loss_finite"]
        or nonfinite
        or report["context_gradient_norm"] <= 0.0
        or not report["boundary_map_detached"]
        or not unchanged
        or report["alpha"]["min"] < 0.0
        or report["alpha"]["max"] > 1.0
    ):
        raise AssertionError(report)
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    print("BCCH_PREFLIGHT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
