#!/usr/bin/env python3
"""Evaluate the existing official BCSS seed42 Full-25 FINAL checkpoint on val."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import resnet38_cls
from tools.lw_shr_common import A0_COMMIT, set_seed, sha256_file, write_json
from tools.lw_shr_evaluation import evaluate_bcss


EXPECTED_BASELINE_SHA256 = (
    "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
)


def parse_tsv(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split("\t", 1)
        rows.setdefault(key, []).append(value)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--environment-tsv", required=True)
    parser.add_argument("--status-tsv", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint_sha256 = sha256_file(args.checkpoint)
    if checkpoint_sha256 != EXPECTED_BASELINE_SHA256:
        raise AssertionError("Existing BCSS seed42 baseline checkpoint SHA256 mismatch")
    environment = parse_tsv(args.environment_tsv)
    status = parse_tsv(args.status_tsv)
    if environment.get("repo_commit") != [A0_COMMIT]:
        raise AssertionError("Existing baseline is not pure official A0")
    if status.get("dataset") != ["bcss"] or status.get("seed") != ["42"]:
        raise AssertionError("Existing baseline is not BCSS seed42")
    if status.get("status", [])[-1:] != ["complete"]:
        raise AssertionError("Existing baseline is incomplete")
    if status.get("checkpoint_sha256") != [checkpoint_sha256]:
        raise AssertionError("Baseline status/checkpoint hash differs")

    set_seed(42, deterministic=False)
    model = resnet38_cls.Net(4).cuda()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Baseline load failed: {incompatible}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_bcss(
        model,
        args.val_root,
        num_workers=args.num_workers,
        prediction_output=output / "epoch25_validation.npz",
        mechanism_diagnostics=False,
    )
    result = {
        "status": "COMPLETE",
        "reference": "existing six-run official reproduction; no retraining",
        "dataset": "BCSS-WSSS",
        "seed": 42,
        "epoch": 25,
        "checkpoint_rule": "FINAL",
        "a0_commit": A0_COMMIT,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "environment_tsv": str(Path(args.environment_tsv).resolve()),
        "status_tsv": str(Path(args.status_tsv).resolve()),
        "environment": environment,
        "training_status": status,
        "validation": evaluation,
        "predictions": str((output / "epoch25_validation.npz").resolve()),
        "test_used_in_this_evaluation": False,
        "baseline_retrained": False,
    }
    write_json(output / "baseline_reference.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print("LW_SHR_FULL25_BASELINE_REFERENCE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
