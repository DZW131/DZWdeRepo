#!/usr/bin/env python3
"""Real batch20 BF16 no-step preflight for TCER-R fresh-25 training."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls import Net as OfficialNet
from network.resnet38_cls_tcrd_gate import Net as TCERNet
from tool.GenDataset import Stage1_TrainDataset
from tools.tcer_r_full25_common import (
    BATCH_SIZE, EXPECTED_PRETRAINED_SHA256, MAX_STEPS,
    build_official_optimizer, official_classification_loss,
    set_official_seed, sha256_file, write_json,
)


def run(args):
    if sha256_file(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise AssertionError("Pretrained SHA mismatch")
    if "test" in args.train_root.lower() or "luad" in args.train_root.lower():
        raise AssertionError("BCSS training path guard failed")

    set_official_seed(42)
    official = OfficialNet(4)
    set_official_seed(42)
    candidate = TCERNet(4, branch="R")
    official_state = official.state_dict()
    candidate_state = candidate.state_dict()
    common_exact = all(
        key in candidate_state and torch.equal(value, candidate_state[key])
        for key, value in official_state.items()
    )
    if not common_exact:
        raise AssertionError("R-only common initialization differs from official A0")
    del official, official_state, candidate_state

    candidate = candidate.cuda()
    optimizer = build_official_optimizer(candidate)
    converted = importlib.import_module("network.resnet38d").convert_mxnet_to_torch(
        args.pretrained
    )
    incompat = candidate.load_state_dict(converted, strict=False)
    if incompat.unexpected_keys:
        raise AssertionError(incompat.unexpected_keys)
    backbone_exact = all(
        torch.equal(candidate.state_dict()[key].cpu(), value)
        for key, value in converted.items()
    )
    if not backbone_exact:
        raise AssertionError("Converted backbone was not fully loaded")

    dataset = Stage1_TrainDataset(args.train_root, dataset="bcss", img_size=224)
    if len(dataset) != 23422:
        raise AssertionError(len(dataset))
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True, generator=generator,
    )
    _, image, label = next(iter(loader))
    candidate.train()
    image = image.cuda(); label = label.cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = candidate(image, active_labels=label)
        loss = official_classification_loss(outputs, label)
    loss.backward()
    finite = bool(
        torch.isfinite(loss)
        and all(torch.isfinite(value).all() for value in outputs[:9])
    )
    gradients = {
        name: None if parameter.grad is None else float(parameter.grad.float().norm())
        for name, parameter in candidate.tcrd.named_parameters()
    }
    new_ids = {id(parameter) for parameter in candidate.tcrd.parameters()}
    scratch_ids = [id(parameter) for parameter in optimizer.param_groups[2]["params"]]
    coverage = {str(value): scratch_ids.count(value) for value in new_ids}
    if any(value != 1 for value in coverage.values()):
        raise AssertionError("TCER parameter optimizer coverage failed")
    report = {
        "decision": "TCER_R25_PREFLIGHT_PASS",
        "optimizer_step": False,
        "common_initialization_exact": common_exact,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "pretrained_backbone_exact": backbone_exact,
        "pretrained_missing_keys": incompat.missing_keys,
        "pretrained_unexpected_keys": incompat.unexpected_keys,
        "train_samples": len(dataset), "batch_size": BATCH_SIZE,
        "image_shape": list(image.shape), "label_shape": list(label.shape),
        "epochs": 25, "max_steps": MAX_STEPS,
        "precision": "bf16", "finite": finite, "loss": float(loss.detach()),
        "eta_r": float(candidate.tcrd.eta_r.detach()),
        "competition_matrix": candidate.tcrd.competition_matrix().detach().cpu().tolist(),
        "new_parameter_gradients": gradients,
        "scratch_group_coverage": coverage,
        "optimizer_momentum": [g["momentum"] for g in optimizer.param_groups],
        "initial_group_lrs": [g["lr"] for g in optimizer.param_groups],
        "group_weight_decay": [g["weight_decay"] for g in optimizer.param_groups],
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "validation_during_training": False,
        "final_checkpoint_only": True,
        "test_used": False, "luad_used": False,
    }
    if (
        not finite
        or any(value is None for value in gradients.values())
        or any(value <= 0 for value in gradients.values())
    ):
        raise AssertionError("TCER R-only BF16 gradient readiness failed")
    write_json(args.output, report)
    print(report, flush=True)
    print("TCER_R25_PREFLIGHT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
