#!/usr/bin/env python3
"""Run one EXP-FDHR-003 variant from the locked common epoch20 state."""

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

from network import resnet38_fdhr
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


def load_variant_model(variant, common, kernel):
    model = resnet38_fdhr.Net(
        4, fdhr_variant=variant, wdch_kernel_size=kernel
    )
    state = dict(common["model_state"])
    old = state.pop("hfrm_28_1.context_conv.weight")
    incompat = model.load_state_dict(state, strict=False)
    expected = {
        "hfrm_28_1.wdch.haar.analysis_filters",
        "hfrm_28_1.wdch.haar.synthesis_filters",
        "hfrm_28_1.wdch.ll_context.weight",
        "hfrm_28_1.wdch.strength",
    }
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    audit = {
        "expected_missing": sorted(expected),
        "unexpected": incompat.unexpected_keys,
        "replaced_parameter": "hfrm_28_1.context_conv.weight",
        "replaced_shape": list(old.shape),
        "new_shape": list(model.hfrm_28_1.wdch.ll_context.weight.shape),
        "new_initialization": f"uniform 1/{kernel**2}",
        "variant": variant,
        "fixed_strength": float(model.hfrm_28_1.wdch.strength),
        "new_trainable_parameter_count_vs_W1": 0,
    }
    return model.cuda(), audit


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty variant output: {output}")
    for name in ("validation", "checkpoints", "predictions"):
        (output / name).mkdir(parents=True, exist_ok=True)

    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    if common.get("format") != "WDCH_MATCHED_COMMON_V1" or common.get("epoch") != 20:
        raise AssertionError("Invalid common epoch20 state")
    if common["data_sampler_state"]["schedule_sha256"] != sha256_file(args.schedule):
        raise AssertionError("Schedule differs from common checkpoint")
    phase0 = json.loads(Path(args.phase0_summary).read_text(encoding="utf-8"))
    if phase0.get("phase0_status") != "PASS":
        raise AssertionError("Locked WD-CH Phase 0 has not passed")
    kernel = int(phase0["selected_kernel"])
    if kernel != 7:
        raise AssertionError(f"EXP-FDHR-003 requires locked k=7, got {kernel}")

    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    if len(dataset) != EXPECTED_TRAIN:
        raise AssertionError(f"Expected {EXPECTED_TRAIN} training samples, got {len(dataset)}")
    set_seed(42, deterministic=True)
    model, load_audit = load_variant_model(args.variant, common, kernel)
    optimizer = build_optimizer(model)
    restore_audit = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    optimizer.global_step = int(common["optimizer_global_step"])
    skipped_names = [row["name"] for row in restore_audit["skipped"]]
    if skipped_names != ["hfrm_28_1.context_conv.weight"]:
        raise AssertionError(f"Optimizer restore mismatch: {restore_audit['skipped']}")
    restore_rng_state(common["rng_state"])

    provenance = {
        "experiment_id": "EXP-FDHR-003",
        "variant": args.variant,
        "source_commit": source_commit(),
        "common_checkpoint": str(Path(args.common_checkpoint).resolve()),
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "common_epoch": common["epoch"],
        "common_global_step": common["optimizer_global_step"],
        "schedule_sha256": sha256_file(args.schedule),
        "selected_kernel": kernel,
        "model_load_audit": load_audit,
        "optimizer_restore_skipped": restore_audit["skipped"],
        "optimizer_after_restore": optimizer_summary(optimizer),
        "precision": "bf16",
        "loss_weights": LOSS_WEIGHTS,
        "checkpoint_rule": "epoch25 FINAL only",
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
            model, optimizer, loader, schedule["model_seeds"], epoch, f"FDHR-{args.variant}"
        )
        history.append(record)
        write_json(output / "training_history.json", history)
        print("FDHR_BRANCH_EPOCH " + json.dumps(record, sort_keys=True), flush=True)

    prediction_path = output / "predictions" / "epoch25_validation.npz"
    evaluation = evaluate_bcss(
        model,
        args.val_root,
        num_workers=args.num_workers,
        prediction_output=str(prediction_path),
    )
    evaluation.update(
        {
            "variant": args.variant,
            "epoch": 25,
            "checkpoint_selection": "none; epoch25 FINAL only",
        }
    )
    write_json(output / "validation" / "epoch25.json", evaluation)
    checkpoint = output / "checkpoints" / "epoch25_final.pth"
    torch.save(model.state_dict(), checkpoint)
    completion = {
        "status": "FDHR_MATCHED_VARIANT_COMPLETE",
        "experiment_id": "EXP-FDHR-003",
        "variant": args.variant,
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
                "variant": args.variant,
                "mIoU": 100 * evaluation["scores"]["final"]["mIoU"],
                "mDice": 100 * evaluation["scores"]["final"]["mDice"],
                "checkpoint_sha256": completion["checkpoint_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"FDHR_MATCHED_VARIANT_COMPLETE variant={args.variant}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--phase0-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
