#!/usr/bin/env python3
"""No-step BF16/optimizer preflight for EXP-WDCH-002 W2."""

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

from tools.train_scwdch_matched import load_w2
from tools.scwdch_constants import (
    EXPECTED_COMMON_EPOCH20_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    EXPERIMENT_ID,
)
from tools.wdch_common import (
    LOSS_WEIGHTS,
    MatchedAugmentationDataset,
    ScheduleBatchSampler,
    build_optimizer,
    load_schedule,
    optimizer_summary,
    restore_named_optimizer_state,
    set_seed,
    sha256_file,
    write_json,
)


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    if sha256_file(args.common_checkpoint) != EXPECTED_COMMON_EPOCH20_SHA256:
        raise AssertionError("Common epoch20 SHA256 mismatch")
    if sha256_file(args.schedule) != EXPECTED_SCHEDULE_SHA256:
        raise AssertionError("Schedule SHA256 mismatch")
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if calibration["checkpoint_sha256"] != sha256_file(args.common_checkpoint):
        raise AssertionError("Calibration/common mismatch")
    scale = float(calibration["scale"])
    kernel = int(calibration["kernel"])
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    loader = DataLoader(
        dataset,
        batch_sampler=ScheduleBatchSampler(schedule, 20),
        num_workers=0,
        pin_memory=True,
    )
    set_seed(42, deterministic=True)
    model, load_audit = load_w2(common, kernel, scale)
    optimizer = build_optimizer(model)
    restore = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    optimizer.global_step = int(common["optimizer_global_step"])
    if [row["name"] for row in restore["skipped"]] != [
        "hfrm_28_1.context_conv.weight"
    ]:
        raise AssertionError(restore)

    captured = {}

    def capture_input(_module, inputs):
        captured["feature"] = inputs[0]

    hook = model.hfrm_28_1.register_forward_pre_hook(capture_input)
    model.train()
    _, image, target = next(iter(loader))
    image = image.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(image)
            losses = [
                F.multilabel_soft_margin_loss(output, target)
                for output in outputs[:4]
            ]
            loss = sum(weight * value for weight, value in zip(LOSS_WEIGHTS, losses))
            feature = captured["feature"]
            wd_output, _, _ = model.hfrm_28_1.wdch.unscaled_forward_with_bands(feature)
            sc_output = model.hfrm_28_1.wdch(feature)
        loss.backward()
    finally:
        hook.remove()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    finite = bool(
        torch.isfinite(loss)
        and torch.isfinite(sc_output).all()
        and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in trainable
        )
    )
    result = {
        "status": "PASS" if finite else "FAIL",
        "experiment_id": EXPERIMENT_ID,
        "batch": int(image.shape[0]),
        "precision": "bf16",
        "loss": float(loss.detach()),
        "fixed_scale": scale,
        "scale_is_buffer": "hfrm_28_1.wdch.scale" in dict(model.named_buffers()),
        "scale_is_parameter": "hfrm_28_1.wdch.scale" in dict(model.named_parameters()),
        "feature_rms": rms(feature),
        "wd_rectification_rms": rms(wd_output - feature),
        "sc_rectification_rms": rms(sc_output - feature),
        "batch_strength_ratio": rms(sc_output - feature)
        / max(rms(wd_output - feature), 1.0e-12),
        "ll_context_grad_rms": rms(model.hfrm_28_1.wdch.ll_context.weight.grad),
        "gamma_context_grad_rms": rms(model.hfrm_28_1.gamma_context.grad),
        "gamma_veto_grad_rms": rms(model.hfrm_28_1.gamma_veto.grad),
        "all_finite": finite,
        "optimizer_coverage_exactly_once": (
            len(grouped) == len({id(parameter) for parameter in grouped})
            and {id(parameter) for parameter in grouped}
            == {id(parameter) for parameter in trainable}
        ),
        "optimizer": optimizer_summary(optimizer),
        "optimizer_global_step": optimizer.global_step,
        "optimizer_step_performed": False,
        "model_load_audit": load_audit,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "validation_used": False,
        "test_used": False,
    }
    if not result["scale_is_buffer"] or result["scale_is_parameter"]:
        raise AssertionError("Scale must be a fixed persistent buffer")
    if not result["optimizer_coverage_exactly_once"] or not finite:
        raise RuntimeError(result)
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    print("SCWDCH_PREFLIGHT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
