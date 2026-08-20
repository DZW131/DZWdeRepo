#!/usr/bin/env python3
"""Disposable one-batch server preflight for the frozen RSBR Stage-1 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_rsbr_v0_stage1_3ep import (
    A0_CHECKPOINT_SHA256,
    PARITY_REQUIRED,
    READINESS_REQUIRED,
    base_state_hashes,
    build_optimizer,
    frozen_gradients_clean,
    frozen_mode_ok,
    grad_norm,
    load_fresh_model,
    make_loader,
    seed_everything,
    sha256_file,
    training_losses,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parity-summary", required=True)
    parser.add_argument("--readiness-summary", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    combined = " ".join(map(str, vars(args).values())).lower()
    if "test" in combined or "luad" in combined:
        raise ValueError("Preflight is BCSS training-only")
    if sha256_file(args.checkpoint) != A0_CHECKPOINT_SHA256:
        raise RuntimeError("A0 checkpoint SHA mismatch")
    parity = json.loads(Path(args.parity_summary).read_text(encoding="utf-8"))
    readiness = json.loads(Path(args.readiness_summary).read_text(encoding="utf-8"))
    if parity.get("decision") != PARITY_REQUIRED or readiness.get("decision") != READINESS_REQUIRED:
        raise RuntimeError("Required parity/readiness PASS artifacts are absent")
    seed_everything()
    runtime = SimpleNamespace(
        train_root=args.train_root,
        batch_size=20,
        img_size=224,
        num_workers=args.num_workers,
        lr=0.01,
        wt_dec=5e-4,
    )
    dataset, loader = make_loader(runtime)
    model, missing, zero_init = load_fresh_model(args.checkpoint)
    model.cuda()
    optimizer = build_optimizer(model, runtime, len(dataset) // 20)
    before = base_state_hashes(model)
    _, images, labels = next(iter(loader))
    images = images.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        total, _, _, _, _, result = training_losses(model, images, labels)
    identity = {
        "delta_core_exact_zero": torch.count_nonzero(result.delta_core).item() == 0,
        "delta_transition_exact_zero": torch.count_nonzero(result.delta_transition).item() == 0,
        "base_refined_cam_exact": torch.equal(
            result.refined_cam,
            result.refined_cam - result.delta_core - result.delta_transition,
        ),
    }
    total.backward()
    region_grad = grad_norm(model.rsbr.region_semantic_head.parameters())
    transition_grad = grad_norm(model.rsbr.transition_head.parameters())
    frozen_grads = frozen_gradients_clean(model)
    finite = bool(
        torch.isfinite(total).item()
        and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.rsbr.parameters()
        )
    )
    optimizer.step()
    after = base_state_hashes(model)
    checks = {
        "parsed_train_samples": len(dataset),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "parity_decision": parity["decision"],
        "readiness_decision": readiness["decision"],
        "missing_keys_expected_rsbr_only": missing,
        "zero_initialization": zero_init,
        "local_identity": identity,
        "mode_contract": frozen_mode_ok(model),
        "update_groups_rsbr_only": (
            {
                id(parameter)
                for group in optimizer.param_groups for parameter in group["params"]
            }
            == {id(item) for item in model.rsbr.parameters()}
        ),
        "region_grad_norm": region_grad,
        "transition_grad_norm": transition_grad,
        "frozen_gradients_clean": frozen_grads,
        "frozen_parameters_unchanged": (
            before["parameter_sha256"] == after["parameter_sha256"]
        ),
        "frozen_buffers_unchanged": before["buffer_sha256"] == after["buffer_sha256"],
        "finite": finite,
        "batch_size": images.shape[0],
    }
    passed = (
        checks["parsed_train_samples"] == 23_422
        and all(zero_init.values())
        and all(identity.values())
        and all(checks["mode_contract"].values())
        and checks["update_groups_rsbr_only"]
        and region_grad > 0
        and transition_grad > 0
        and frozen_grads
        and checks["frozen_parameters_unchanged"]
        and checks["frozen_buffers_unchanged"]
        and finite
        and images.shape[0] == 20
    )
    print(
        "RSBR_STAGE1_PREFLIGHT_PASS" if passed else "RSBR_STAGE1_PREFLIGHT_NOGO",
        json.dumps(checks, sort_keys=True),
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
