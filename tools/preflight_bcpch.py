#!/usr/bin/env python3
"""Real batch20 BF16 no-step preflight for EXP-BCPCH-003."""

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
from tools.train_bcpch_matched import load_bcpch_model
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


def grad_norm(parameter):
    return None if parameter.grad is None else float(parameter.grad.detach().float().norm())


def run(args):
    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    loader = make_loader(dataset, schedule, 20, args.num_workers)
    names, image, target = next(iter(loader))
    if image.shape != (20, 3, 224, 224) or target.shape != (20, 4):
        raise AssertionError((image.shape, target.shape))

    restore_rng_state(common["rng_state"])
    model, load_audit = load_bcpch_model(common)
    c0 = resnet38_cls.Net(4)
    if sum(parameter.numel() for parameter in model.parameters()) != sum(
        parameter.numel() for parameter in c0.parameters()
    ):
        raise AssertionError("BCP-CH trainable parameter count differs from C0")
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

    image = image.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    tracked = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in {
            "b4.conv_branch2a.weight",
            "ic1.weight",
            "fc8.weight",
            "hfrm_28_1.context_conv.weight",
            "hfrm_28_1.gamma_context",
            "hfrm_28_1.gamma_veto",
        }
    }
    captured = {}

    def capture_input(_module, inputs):
        captured["feature"] = inputs[0]
        captured["deep"] = inputs[1]

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
                variant="A3",
                kernel_size=15,
                hard_fraction=0.20,
                temperature=0.07,
            )
            total = official + 0.10 * contrastive
        total.backward()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, mechanism, auxiliary = model.hfrm_28_1.prototype_with_diagnostics(
                captured["feature"].detach(), captured["deep"].detach()
            )
            cams = forward_cam_compatible(model, image[:1])
    finally:
        hook.remove()

    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    current = dict(model.named_parameters())
    unchanged = all(
        torch.equal(value, current[name].detach().cpu())
        for name, value in tracked.items()
    )
    report = {
        "status": "BCPCH_PREFLIGHT_PASS",
        "experiment_id": "EXP-BCPCH-003",
        "batch_size": int(image.shape[0]),
        "precision": "bf16",
        "official_loss": float(official.detach()),
        "contrastive_loss": float(contrastive.detach()),
        "total_loss": float(total.detach()),
        "outputs_finite": all(bool(torch.isfinite(value).all()) for value in outputs),
        "forward_cam_finite": all(bool(torch.isfinite(value).all()) for value in cams),
        "losses_finite": bool(
            torch.isfinite(official) and torch.isfinite(contrastive) and torch.isfinite(total)
        ),
        "nonfinite_gradients": nonfinite,
        "ic1_weight_gradient_norm": grad_norm(model.ic1.weight),
        "fc8_weight_gradient_norm": grad_norm(model.fc8.weight),
        "backbone_gradient_norm": grad_norm(model.b4.conv_branch2a.weight),
        "gamma_context_gradient": grad_norm(model.hfrm_28_1.gamma_context),
        "gamma_veto_gradient": grad_norm(model.hfrm_28_1.gamma_veto),
        "dormant_legacy_context_gradient": grad_norm(
            model.hfrm_28_1.context_conv.weight
        ),
        "contrastive_stats": contrastive_stats,
        "mechanism": {
            key: float(value.detach().float()) for key, value in mechanism.items()
        },
        "prototype_shape": list(auxiliary["class_similarity"].shape),
        "valid_prototype_count": int(auxiliary["valid_prototypes"].sum()),
        "confidence_mask_detached": not auxiliary["confidence_mask"].requires_grad,
        "boundary_map_detached": not auxiliary["boundary"].requires_grad,
        "semantic_probe_shared": model.hfrm_28_1._semantic_probe is model.ic1,
        "presence_probe_shared": model.hfrm_28_1._presence_probe is model.fc8,
        "duplicate_probe_state_keys": [
            key
            for key in model.state_dict()
            if "semantic_probe" in key or "presence_probe" in key
        ],
        "optimizer_coverage_exactly_once": True,
        "optimizer_state_restored_without_skip": True,
        "optimizer_step_performed": False,
        "tracked_parameters_unchanged": unchanged,
        "trainable_parameters_equal_c0": True,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "schedule_sha256": sha256_file(args.schedule),
        "model_load_audit": load_audit,
        "batch_names": list(names),
        "test_used": False,
        "luad_used": False,
    }
    required_positive = (
        report["ic1_weight_gradient_norm"],
        report["fc8_weight_gradient_norm"],
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
        or report["dormant_legacy_context_gradient"] is not None
        or report["valid_prototype_count"] <= 0
        or not report["confidence_mask_detached"]
        or not report["boundary_map_detached"]
        or not report["semantic_probe_shared"]
        or not report["presence_probe_shared"]
        or report["duplicate_probe_state_keys"]
        or not unchanged
    ):
        raise AssertionError(report)
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    print("BCPCH_PREFLIGHT_PASS", flush=True)


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
