#!/usr/bin/env python3
"""One-batch batch20/BF16 resource and connectivity smoke for RSBR-v0."""

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

from tools.run_rsbr_v0 import (
    finite_gradients,
    freeze_a0,
    grad_norm,
    load_rsbr_model,
    make_train_loader,
    rsbr_optimizer,
    seed_everything,
    training_loss,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    runtime = SimpleNamespace(
        train_root=args.train_root, batch_size=20, img_size=224,
        num_workers=args.num_workers, lr=0.01, wt_dec=5e-4,
    )
    seed_everything()
    _, loader = make_train_loader(runtime)
    model, missing = load_rsbr_model(args.checkpoint, inference=False)
    freeze_a0(model)
    model.cuda()
    optimizer = rsbr_optimizer(model, runtime, max_step=32)
    _, images, labels = next(iter(loader))
    images = images.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        total, classification, region, residual, result = training_loss(
            model, images, labels
        )
    total.backward()
    baseline_gradient_clean = all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for name, parameter in model.named_parameters()
        if not name.startswith("rsbr.")
    )
    payload = {
        "finite": bool(torch.isfinite(total).item()) and finite_gradients(model.rsbr),
        "loss_total": float(total.detach().float()),
        "loss_classification": float(classification.detach().float()),
        "loss_region_mil": float(region.detach().float()),
        "loss_residual": float(residual.detach().float()),
        "region_head_grad_norm": grad_norm(model.rsbr.region_semantic_head.parameters()),
        "transition_head_grad_norm": grad_norm(model.rsbr.transition_head.parameters()),
        "transition_fraction": result.statistics["transition_fraction"],
        "residual_ratio": result.statistics["residual_ratio"],
        "sshr_gradients_none_or_zero": baseline_gradient_clean,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "missing_keys_expected_rsbr_only": missing,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if not payload["finite"] or not baseline_gradient_clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

