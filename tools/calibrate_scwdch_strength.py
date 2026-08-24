#!/usr/bin/env python3
"""Calibrate SC-WDCH strength from epoch20 and the BCSS training set only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import resnet38_cls
from research.wdch import StrengthCalibratedWaveletContext, WaveletDecoupledContext
from tool.GenDataset import Stage1_InferDataset
from tools.wdch_common import (
    EXPECTED_TRAIN,
    read_state,
    set_seed,
    sha256_file,
    write_json,
)
from tools.wdch_evaluation import forward_cam_compatible
from tools.scwdch_constants import (
    EXPECTED_COMMON_EPOCH20_SHA256,
    EXPERIMENT_ID,
)


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def per_sample_rms(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().square().flatten(1).mean(1).sqrt()


def verify_training_only_root(root: str) -> None:
    path = Path(root).resolve()
    lowered = str(path).lower()
    if any(forbidden in lowered for forbidden in ("/val", "\\val", "test", "luad")):
        raise AssertionError("Strength calibration is BCSS training-only")
    if path.name.lower() != "training":
        raise AssertionError(f"Expected a training split directory, received {path}")
    images = list(path.rglob("*.png")) + list(path.rglob("*.jpg"))
    if len(images) != EXPECTED_TRAIN:
        raise AssertionError(
            f"Expected {EXPECTED_TRAIN} training images, found {len(images)}"
        )


def run_pass(model, operator, loader, mode: str):
    captured = {}

    def capture_input(_module, inputs):
        captured["feature"] = inputs[0].detach()

    hook = model.hfrm_28_1.register_forward_pre_hook(capture_input)
    count = 0
    sums = {
        "input": 0.0,
        "ch_output": 0.0,
        "operator_output": 0.0,
        "ch_delta": 0.0,
        "delta": 0.0,
    }
    started = time.time()
    try:
        with torch.no_grad():
            for index, (_, image) in enumerate(loader, start=1):
                image = image.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    forward_cam_compatible(model, image)
                    feature = captured["feature"]
                    ch_output = model.hfrm_28_1.context_conv(feature)
                    operator_output = operator(feature)
                batch = image.shape[0]
                sums["input"] += float(per_sample_rms(feature).sum())
                sums["ch_output"] += float(per_sample_rms(ch_output).sum())
                sums["operator_output"] += float(per_sample_rms(operator_output).sum())
                sums["ch_delta"] += float(per_sample_rms(ch_output - feature).sum())
                sums["delta"] += float(per_sample_rms(operator_output - feature).sum())
                count += batch
                if index % 100 == 0:
                    print(
                        f"SCWDCH_CALIBRATION mode={mode} batch={index}/{len(loader)} "
                        f"samples={count}/{EXPECTED_TRAIN}",
                        flush=True,
                    )
    finally:
        hook.remove()
    if count != EXPECTED_TRAIN:
        raise AssertionError(count)
    return {
        "samples": count,
        "input_rms": sums["input"] / count,
        "ch_output_rms": sums["ch_output"] / count,
        "operator_output_rms": sums["operator_output"] / count,
        "ch_rectification_rms": sums["ch_delta"] / count,
        "rectification_rms": sums["delta"] / count,
        "runtime_seconds": time.time() - started,
    }


def run(args):
    verify_training_only_root(args.train_root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    phase0 = json.loads(Path(args.phase0_summary).read_text(encoding="utf-8"))
    if phase0.get("phase0_status") != "PASS":
        raise AssertionError("WD-CH Phase 0 has not passed")
    kernel = int(phase0["selected_kernel"])
    if kernel != 7:
        raise AssertionError(f"EXP-WDCH-002 expects frozen k*=7, found {kernel}")
    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    if sha256_file(args.common_checkpoint) != EXPECTED_COMMON_EPOCH20_SHA256:
        raise AssertionError("EXP-WDCH-002 common epoch20 SHA256 mismatch")
    if common.get("format") != "WDCH_MATCHED_COMMON_V1":
        raise AssertionError("Unexpected common checkpoint format")
    if common.get("epoch") != 20 or common.get("optimizer_global_step") != 23420:
        raise AssertionError("Calibration requires seed42 epoch20 common state")

    set_seed(42, deterministic=True)
    model = resnet38_cls.Net(4)
    incompat = model.load_state_dict(read_state(args.common_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    model = model.cuda()
    model.eval()
    dataset = Stage1_InferDataset(args.train_root, img_size=224)
    if len(dataset) != EXPECTED_TRAIN:
        raise AssertionError(len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    wdch = WaveletDecoupledContext(512, kernel).cuda()
    wdch.eval()
    raw = run_pass(model, wdch, loader, "WDCH")
    r_ch = raw["ch_rectification_rms"]
    r_wd = raw["rectification_rms"]
    if not np.isfinite(r_ch) or not np.isfinite(r_wd) or r_wd <= 0:
        raise FloatingPointError({"R_CH": r_ch, "R_WD": r_wd})
    scale = r_ch / r_wd

    scwdch = StrengthCalibratedWaveletContext(512, kernel, scale).cuda()
    scwdch.load_state_dict(
        {**wdch.state_dict(), "scale": scwdch.scale.detach().clone()}, strict=True
    )
    scwdch.eval()
    calibrated = run_pass(model, scwdch, loader, "SC-WDCH")
    ratio = calibrated["rectification_rms"] / r_ch
    result = {
        "experiment_id": EXPERIMENT_ID,
        "R_CH": r_ch,
        "R_WD": r_wd,
        "R_SC_WD": calibrated["rectification_rms"],
        "scale": scale,
        "initial_strength_ratio": ratio,
        "initial_gate_a_range": [0.9, 1.1],
        "initial_gate_a_pass": 0.9 <= ratio <= 1.1,
        "checkpoint": "epoch20",
        "checkpoint_path": str(Path(args.common_checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.common_checkpoint),
        "common_global_step": common["optimizer_global_step"],
        "seed": 42,
        "kernel": kernel,
        "split": "BCSS training only",
        "training_samples": EXPECTED_TRAIN,
        "batch_size": 20,
        "precision": "bf16",
        "augmentation": "none; released resize/normalize only",
        "raw_wdch_pass": raw,
        "calibrated_pass": calibrated,
        "source_commit": source_commit(),
        "validation_used": False,
        "test_used": False,
    }
    if not result["initial_gate_a_pass"]:
        raise RuntimeError(f"Initial strength calibration failed: {ratio}")
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    print("SCWDCH_STRENGTH_CALIBRATION_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--phase0-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
