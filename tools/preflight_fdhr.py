#!/usr/bin/env python3
"""Batch20 BF16 no-step preflight for all EXP-FDHR-003 variants."""

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

from tools.train_fdhr_matched import load_variant_model
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


def parameter_snapshot(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def run(args):
    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    phase0 = json.loads(Path(args.phase0_summary).read_text(encoding="utf-8"))
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    loader = make_loader(dataset, schedule, 20, args.num_workers)
    names, image, target = next(iter(loader))
    if image.shape != (20, 3, 224, 224) or target.shape != (20, 4):
        raise AssertionError((image.shape, target.shape))
    image = image.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    rows = []

    for variant in ("A", "B", "C"):
        restore_rng_state(common["rng_state"])
        model, load_audit = load_variant_model(variant, common, int(phase0["selected_kernel"]))
        model.train()
        optimizer = build_optimizer(model)
        restore = restore_named_optimizer_state(
            model, optimizer, common["optimizer_named_state"]
        )
        optimizer.global_step = int(common["optimizer_global_step"])
        grouped = [p for group in optimizer.param_groups for p in group["params"]]
        trainable = [p for p in model.parameters() if p.requires_grad]
        if len(grouped) != len({id(p) for p in grouped}):
            raise AssertionError("Duplicate optimizer parameter")
        if {id(p) for p in grouped} != {id(p) for p in trainable}:
            raise AssertionError("Optimizer coverage differs from trainable parameters")
        before = parameter_snapshot(model)
        captured = {}

        def capture_input(_module, inputs):
            captured["feature"] = inputs[0].detach()

        hook = model.hfrm_28_1.register_forward_pre_hook(capture_input)
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(image)
            losses = [F.multilabel_soft_margin_loss(output, target) for output in outputs[:4]]
            loss = sum(w * value for w, value in zip(LOSS_WEIGHTS, losses))
        hook.remove()
        loss.backward()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, frequency = model.hfrm_28_1.wdch.forward_with_diagnostics(
                captured["feature"]
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
        after = parameter_snapshot(model)
        unchanged = all(torch.equal(before[name], after[name]) for name in before)
        ll_grad = model.hfrm_28_1.wdch.ll_context.weight.grad
        row = {
            "variant": variant,
            "batch_size": int(image.shape[0]),
            "precision": "bf16",
            "official_loss": float(loss.detach()),
            "outputs_finite": all(bool(torch.isfinite(value).all()) for value in outputs),
            "loss_finite": bool(torch.isfinite(loss)),
            "nonfinite_gradients": nonfinite,
            "missing_gradient_parameter_count": len(missing_grad),
            "ll_context_gradient_norm": float(ll_grad.detach().float().norm()),
            "gamma_veto_gradient": float(model.hfrm_28_1.gamma_veto.grad.detach().float()),
            "gamma_context_gradient": float(model.hfrm_28_1.gamma_context.grad.detach().float()),
            "optimizer_coverage_exactly_once": True,
            "optimizer_step_performed": False,
            "parameters_unchanged": unchanged,
            "fixed_strength": float(model.hfrm_28_1.wdch.strength),
            "strength_is_parameter": "strength" in dict(model.hfrm_28_1.wdch.named_parameters()),
            "frequency_statistics": {key: float(value) for key, value in frequency.items()},
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "model_load_audit": load_audit,
            "optimizer_restore_skipped": restore["skipped"],
            "input_names_sha256_guard": sha256_file(args.schedule),
        }
        if not row["outputs_finite"] or not row["loss_finite"] or nonfinite:
            raise AssertionError(row)
        if not unchanged or row["strength_is_parameter"] or row["ll_context_gradient_norm"] <= 0:
            raise AssertionError(row)
        rows.append(row)
        del model, optimizer
        torch.cuda.empty_cache()

    report = {
        "status": "FDHR_PREFLIGHT_PASS",
        "experiment_id": "EXP-FDHR-003",
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "schedule_sha256": sha256_file(args.schedule),
        "batch_names": list(names),
        "variants": rows,
        "test_used": False,
        "luad_used": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    print("FDHR_PREFLIGHT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--phase0-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
