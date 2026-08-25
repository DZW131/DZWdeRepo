#!/usr/bin/env python3
"""Real batch20 BF16 no-step preflight for EXP-CBCCH-002."""

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
from research.wdch import contrastive_affinity_loss
from tools.train_cbcch_matched import (
    CONTRASTIVE_WEIGHT,
    HARD_FRACTION,
    LOCAL_KERNEL,
    TEMPERATURE,
    load_cbcch_model,
)
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
from tools.wdch_evaluation import forward_cam_compatible


def gradient_norm(parameter):
    if parameter.grad is None:
        return None
    return float(parameter.grad.detach().float().norm())


def audit_variant(common, image, target, variant):
    restore_rng_state(common["rng_state"])
    model, load_audit = load_cbcch_model(common, variant)
    c0 = resnet38_cls.Net(4)
    if sum(p.numel() for p in model.parameters()) != sum(
        p.numel() for p in c0.parameters()
    ):
        raise AssertionError("CBCCH trainable parameter count differs from C0")
    del c0
    optimizer = build_optimizer(model)
    restore = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    optimizer.global_step = int(common["optimizer_global_step"])
    if restore["skipped"]:
        raise AssertionError(restore)
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(grouped) != len({id(parameter) for parameter in grouped}):
        raise AssertionError("Duplicate optimizer parameter")
    if {id(parameter) for parameter in grouped} != {id(parameter) for parameter in trainable}:
        raise AssertionError("Optimizer coverage differs")

    tracked = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in {
            "conv1a.weight",
            "ic1.weight",
            "ic1.bias",
            "hfrm_28_1.context_conv.weight",
            "hfrm_28_1.gamma_context",
            "hfrm_28_1.gamma_veto",
        }
    }
    captured = {}

    def capture_input(_module, inputs):
        captured["feature"] = inputs[0]

    hook = model.hfrm_28_1.register_forward_pre_hook(capture_input)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(image)
            official_losses = [
                F.multilabel_soft_margin_loss(output, target)
                for output in outputs[:4]
            ]
            official = sum(
                weight * value for weight, value in zip(LOSS_WEIGHTS, official_losses)
            )
            contrastive, contrastive_stats = contrastive_affinity_loss(
                captured["feature"],
                model.hfrm_28_1.last_semantic_logits,
                target,
                model.hfrm_28_1.haar,
                variant=variant,
                kernel_size=LOCAL_KERNEL,
                hard_fraction=HARD_FRACTION,
                temperature=TEMPERATURE,
            )
            total = official + CONTRASTIVE_WEIGHT * contrastive
        total.backward()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, mechanism = model.hfrm_28_1.context_with_diagnostics(
                captured["feature"].detach()
            )
            cams = forward_cam_compatible(model, image[:1])
    finally:
        hook.remove()

    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    unchanged = all(
        torch.equal(tracked[name], dict(model.named_parameters())[name].detach().cpu())
        for name in tracked
    )
    context_grad = gradient_norm(model.hfrm_28_1.context_conv.weight)
    report = {
        "variant": variant,
        "batch_size": int(image.shape[0]),
        "precision": "bf16",
        "official_loss": float(official.detach()),
        "contrastive_loss": float(contrastive.detach()),
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "total_loss": float(total.detach()),
        "outputs_finite": all(bool(torch.isfinite(value).all()) for value in outputs),
        "forward_cam_finite": all(bool(torch.isfinite(value).all()) for value in cams),
        "losses_finite": bool(
            torch.isfinite(official) and torch.isfinite(contrastive) and torch.isfinite(total)
        ),
        "nonfinite_gradients": nonfinite,
        "ic1_weight_gradient_norm": gradient_norm(model.ic1.weight),
        "backbone_gradient_norm": gradient_norm(model.b4.conv_branch2a.weight),
        "gamma_context_gradient": gradient_norm(model.hfrm_28_1.gamma_context),
        "gamma_veto_gradient": gradient_norm(model.hfrm_28_1.gamma_veto),
        "dormant_legacy_context_gradient": context_grad,
        "dormant_legacy_context_expected": True,
        "contrastive_stats": contrastive_stats,
        "mechanism": {
            key: float(value.detach().float()) for key, value in mechanism.items()
        },
        "semantic_probe_shared": model.hfrm_28_1._semantic_probe is model.ic1,
        "semantic_probe_duplicate_state_keys": [
            key for key in model.state_dict() if "semantic_probe" in key
        ],
        "boundary_map_detached": not model.hfrm_28_1.haar.analysis_filters.requires_grad,
        "optimizer_coverage_exactly_once": True,
        "optimizer_state_restored_without_skip": True,
        "optimizer_step_performed": False,
        "tracked_parameters_unchanged": unchanged,
        "trainable_parameters_equal_c0": True,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "model_load_audit": load_audit,
    }
    required_positive = (
        report["ic1_weight_gradient_norm"],
        report["backbone_gradient_norm"],
        report["gamma_context_gradient"],
        report["gamma_veto_gradient"],
    )
    if (
        not report["outputs_finite"]
        or not report["forward_cam_finite"]
        or not report["losses_finite"]
        or nonfinite
        or any(value is None or value <= 0.0 for value in required_positive)
        or context_grad is not None
        or not report["semantic_probe_shared"]
        or report["semantic_probe_duplicate_state_keys"]
        or not unchanged
        or contrastive_stats["valid_anchors"] <= 0
    ):
        raise AssertionError(report)
    return report


def run(args):
    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    loader = make_loader(dataset, schedule, 20, args.num_workers)
    names, image, target = next(iter(loader))
    if image.shape != (20, 3, 224, 224) or target.shape != (20, 4):
        raise AssertionError((image.shape, target.shape))
    image = image.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    variants = {
        variant: audit_variant(common, image, target, variant)
        for variant in ("A2", "A3")
    }
    report = {
        "status": "CBCCH_PREFLIGHT_PASS",
        "experiment_id": "EXP-CBCCH-002",
        "variants": variants,
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "schedule_sha256": sha256_file(args.schedule),
        "batch_names": list(names),
        "test_used": False,
        "luad_used": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    print("CBCCH_PREFLIGHT_PASS", flush=True)


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
