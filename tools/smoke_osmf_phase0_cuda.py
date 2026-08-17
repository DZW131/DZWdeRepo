"""Disposable batch-20 CUDA smoke for the frozen OSMF Phase-0 audit path.

This deliberately uses synthetic tensors, performs no optimizer update, writes
no checkpoint, and never opens a dataset split.  Its only purpose is to prove
that the exact BF16 forward/gradient machinery fits and remains finite before
the single formal 128-real-batch audit is started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls_osmf import Net
from tools.audit_osmf_phase0_128batch import (
    EXPECTED_MISSING_KEYS,
    _all_finite,
    _load_checkpoint,
    forward_objectives,
)
from tools.osmf_phase0_audit import BATCH_SIZE, IMAGE_SIZE, OBJECTIVE_WEIGHTS
from tools.osmf_phase0_audit.gradients import gradient_decomposition


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(20260817)
    torch.cuda.manual_seed_all(20260817)
    model = Net(n_class=4).cuda()
    missing, unexpected = _load_checkpoint(model, args.checkpoint)
    model.train()

    images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda")
    labels = torch.randint(0, 2, (BATCH_SIZE, 4), device="cuda").float()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bundle = forward_objectives(
            model, images, labels, step=4, force_equivariance=True
        )
    ratios, cosines = gradient_decomposition(
        {
            "base": bundle["base"],
            "sem": bundle["sem"],
            "eq": bundle["eq"],
            "orth": bundle["orth"],
            "rec": bundle["rec"],
        },
        bundle["aux"]["input"],
        tuple(model.osmf_28_1.parameters()),
        OBJECTIVE_WEIGHTS,
    )
    bundle["total"].backward()
    torch.cuda.synchronize()

    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    report = {
        "status": "PASS" if _all_finite(bundle) and gradients_finite else "FAIL",
        "synthetic_only": True,
        "optimizer_step_performed": False,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "precision": "bf16",
        "missing_keys": missing,
        "expected_missing_keys": sorted(EXPECTED_MISSING_KEYS),
        "unexpected_keys": unexpected,
        "outputs_finite": _all_finite(bundle),
        "gradients_finite": gradients_finite,
        "gradient_ratios": {row["objective"]: row["ratio"] for row in ratios},
        "gradient_cosines": {row["objective"]: row["cosine"] for row in cosines},
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "device": torch.cuda.get_device_name(),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("OSMF Phase-0 CUDA smoke failed")


if __name__ == "__main__":
    main()
