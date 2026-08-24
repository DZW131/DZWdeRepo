#!/usr/bin/env python3
"""No-step real-batch BF16 readiness check for the Phase-0-selected WD-CH."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tool.GenDataset import Stage1_TrainDataset
from tools.wdch_common import (
    A0_COMMIT,
    LOSS_WEIGHTS,
    build_optimizer,
    load_wdch_from_a0,
    optimizer_summary,
    set_seed,
    verify_validation_root,
    write_json,
)


def run(args):
    verify_validation_root(args.val_root)
    phase0 = json.loads(Path(args.phase0_summary).read_text(encoding="utf-8"))
    if phase0["phase0_status"] != "PASS":
        raise AssertionError("Phase 0 not passed")
    kernel = int(phase0["selected_kernel"])
    subprocess.check_call(
        [
            "git", "diff", "--quiet", A0_COMMIT, "--",
            "network/resnet38_cls.py", "train_sshr.py", "tool/GenDataset.py",
            "tool/infer_fun.py", "tool/iouutils.py", "tool/torchutils.py",
        ],
        cwd=REPO_ROOT,
    )
    set_seed(42, deterministic=True)
    dataset = Stage1_TrainDataset(
        args.train_root, dataset="bcss", img_size=224
    )
    if len(dataset) != 23422:
        raise AssertionError(len(dataset))
    loader = DataLoader(
        dataset, batch_size=20, shuffle=False, num_workers=0, drop_last=True
    )
    _, image, label = next(iter(loader))
    image = image.cuda(non_blocking=True)
    label = label.cuda(non_blocking=True)
    model, load_audit = load_wdch_from_a0(
        args.checkpoint, kernel, cam=False, device="cuda"
    )
    model.train()
    optimizer = build_optimizer(model)
    optimizer_groups = optimizer_summary(optimizer)
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(grouped) != len({id(parameter) for parameter in grouped}):
        raise AssertionError("Duplicate optimizer parameter")
    if {id(parameter) for parameter in grouped} != {id(parameter) for parameter in trainable}:
        raise AssertionError("Optimizer coverage differs from trainable parameters")
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(image)
        losses = [
            F.multilabel_soft_margin_loss(output, label) for output in outputs[:4]
        ]
        total_loss = sum(
            weight * value for weight, value in zip(LOSS_WEIGHTS, losses)
        )
    total_loss.backward()
    watched = {
        "ll_context": model.hfrm_28_1.wdch.ll_context.weight,
        "gamma_veto": model.hfrm_28_1.gamma_veto,
        "gamma_context": model.hfrm_28_1.gamma_context,
    }
    gradients = {
        name: None
        if parameter.grad is None
        else float(parameter.grad.detach().float().norm())
        for name, parameter in watched.items()
    }
    all_finite = bool(
        torch.isfinite(total_loss)
        and all(torch.isfinite(value).all() for value in outputs if torch.is_tensor(value))
        and all(value is not None and np.isfinite(value) for value in gradients.values())
    )
    result = {
        "status": "PASS" if all_finite else "FAIL",
        "kernel": kernel,
        "batch": 20,
        "precision": "bf16",
        "loss": float(total_loss.detach()),
        "gradients": gradients,
        "all_finite": all_finite,
        "optimizer_groups": optimizer_groups,
        "optimizer_coverage_exactly_once": True,
        "load_audit": load_audit,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_step_performed": False,
        "test_used": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"WDCH_PREFLIGHT_{result['status']}", flush=True)
    if not all_finite:
        raise RuntimeError("WD-CH BF16 no-step preflight failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase0-summary", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
