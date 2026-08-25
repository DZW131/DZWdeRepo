#!/usr/bin/env python3
"""Run EXP-BCCH-001 from the locked common SSHR Epoch-20 state."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import resnet38_bcch
from tools.train_wdch_matched import make_loader, train_epoch
from tools.wdch_common import (
    EXPECTED_TRAIN,
    LOSS_WEIGHTS,
    MatchedAugmentationDataset,
    build_optimizer,
    load_schedule,
    optimizer_summary,
    restore_named_optimizer_state,
    restore_rng_state,
    set_seed,
    sha256_file,
    write_json,
)
from tools.wdch_evaluation import evaluate_bcss


def source_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_bcch_model(common):
    model = resnet38_bcch.Net(4)
    incompat = model.load_state_dict(common["model_state"], strict=False)
    expected = {
        "hfrm_28_1.haar.analysis_filters",
        "hfrm_28_1.haar.synthesis_filters",
    }
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    audit = {
        "expected_missing_fixed_buffers": sorted(expected),
        "unexpected": incompat.unexpected_keys,
        "context_parameter": "hfrm_28_1.context_conv.weight",
        "context_parameter_restored_exactly": True,
        "context_shape": list(model.hfrm_28_1.context_conv.weight.shape),
        "new_trainable_parameter_count_vs_C0": 0,
        "boundary_map": "detached channel-mean HF energy; per-image spatial min-max; bilinear upsample",
        "alpha": "1-B",
    }
    return model.cuda(), audit


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty BCCH output: {output}")
    for name in ("validation", "checkpoints", "predictions"):
        (output / name).mkdir(parents=True, exist_ok=True)

    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    if common.get("format") != "WDCH_MATCHED_COMMON_V1" or common.get("epoch") != 20:
        raise AssertionError("Invalid common Epoch-20 state")
    if common["data_sampler_state"]["schedule_sha256"] != sha256_file(args.schedule):
        raise AssertionError("Schedule differs from common checkpoint")
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    if len(dataset) != EXPECTED_TRAIN:
        raise AssertionError(f"Expected {EXPECTED_TRAIN}, got {len(dataset)}")

    set_seed(42, deterministic=True)
    model, load_audit = load_bcch_model(common)
    optimizer = build_optimizer(model)
    restore_audit = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    optimizer.global_step = int(common["optimizer_global_step"])
    if restore_audit["skipped"]:
        raise AssertionError(
            f"BCCH must restore every original optimizer state: {restore_audit['skipped']}"
        )
    restore_rng_state(common["rng_state"])

    provenance = {
        "experiment_id": "EXP-BCCH-001",
        "source_commit": source_commit(),
        "common_checkpoint": str(Path(args.common_checkpoint).resolve()),
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "common_epoch": common["epoch"],
        "common_global_step": common["optimizer_global_step"],
        "schedule_sha256": sha256_file(args.schedule),
        "model_load_audit": load_audit,
        "optimizer_restore": restore_audit,
        "optimizer_after_restore": optimizer_summary(optimizer),
        "precision": "bf16",
        "loss_weights": LOSS_WEIGHTS,
        "checkpoint_rule": "Epoch25 FINAL only",
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "provenance.json", provenance)

    history = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(20, 25):
        loader = make_loader(dataset, schedule, epoch, args.num_workers)
        record = train_epoch(
            model, optimizer, loader, schedule["model_seeds"], epoch, "BCCH"
        )
        history.append(record)
        write_json(output / "training_history.json", history)
        print("BCCH_BRANCH_EPOCH " + json.dumps(record, sort_keys=True), flush=True)

    prediction_path = output / "predictions" / "epoch25_validation.npz"
    evaluation = evaluate_bcss(
        model,
        args.val_root,
        num_workers=args.num_workers,
        prediction_output=str(prediction_path),
    )
    evaluation.update(
        {
            "variant": "BC-CH",
            "epoch": 25,
            "checkpoint_selection": "none; Epoch25 FINAL only",
        }
    )
    write_json(output / "validation" / "epoch25.json", evaluation)
    checkpoint = output / "checkpoints" / "epoch25_final.pth"
    torch.save(model.state_dict(), checkpoint)
    completion = {
        "status": "BCCH_MATCHED_COMPLETE",
        "experiment_id": "EXP-BCCH-001",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "epochs": [21, 22, 23, 24, 25],
        "global_step": optimizer.global_step,
        "final_validation": evaluation,
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "complete.json", completion)
    print(
        json.dumps(
            {
                "status": completion["status"],
                "mIoU": 100 * evaluation["scores"]["final"]["mIoU"],
                "mDice": 100 * evaluation["scores"]["final"]["mDice"],
                "checkpoint_sha256": completion["checkpoint_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    print("BCCH_MATCHED_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
