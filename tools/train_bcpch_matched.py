#!/usr/bin/env python3
"""Run the matched BCP-CH continuation for EXP-BCPCH-003."""

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

from network import resnet38_bcpch
from tools.train_cbcch_matched import train_epoch as train_cbcch_epoch
from tools.train_wdch_matched import make_loader
from tools.wdch_common import (
    EXPECTED_TRAIN,
    LOSS_WEIGHTS,
    MatchedAugmentationDataset,
    build_optimizer,
    capture_rng_state,
    load_schedule,
    named_optimizer_state,
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


def load_bcpch_model(common):
    model = resnet38_bcpch.Net(4)
    incompat = model.load_state_dict(common["model_state"], strict=False)
    expected = {
        "hfrm_28_1.haar.analysis_filters",
        "hfrm_28_1.haar.synthesis_filters",
    }
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    model.hfrm_28_1.set_shared_probes(model.ic1, model.fc8)
    audit = {
        "expected_missing_fixed_buffers": sorted(expected),
        "unexpected": incompat.unexpected_keys,
        "context_parameter": "hfrm_28_1.context_conv.weight",
        "context_parameter_restored_exactly": bool(
            torch.equal(
                model.hfrm_28_1.context_conv.weight.detach().cpu(),
                common["model_state"]["hfrm_28_1.context_conv.weight"],
            )
        ),
        "new_trainable_parameter_count_vs_C0": 0,
        "semantic_probe": "existing ic1; non-registering shared reference",
        "presence_probe": "existing fc8; official BCSS thresholds; detached decision",
        "prototype_embedding": "L2(IDWT(LL,0,0,0))",
        "cam_selection": "per-class spatial minmax ReLU(ic1(F)) > 0.70; detached",
        "fallback": "P_prototype=P_affinity when no valid prototype",
        "propagation": "(1-B)*(0.5*P_affinity+0.5*P_prototype)+B*F",
    }
    if not audit["context_parameter_restored_exactly"]:
        raise AssertionError("Epoch20 CH15 parameter did not restore exactly")
    return model.cuda(), audit


def run(args):
    output = Path(args.output_dir)
    recovery_path = output / "checkpoints" / "recovery_latest.pth"
    if output.exists() and any(output.iterdir()) and not recovery_path.exists():
        unsafe = (output / "training_history.json").exists() or (output / "complete.json").exists()
        if unsafe:
            raise FileExistsError(
                f"Refusing non-empty BCP-CH output without recovery: {output}"
            )
    for name in ("validation", "checkpoints", "predictions"):
        (output / name).mkdir(parents=True, exist_ok=True)

    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    if common.get("format") != "WDCH_MATCHED_COMMON_V1" or common.get("epoch") != 20:
        raise AssertionError("Invalid common Epoch20 state")
    schedule_sha = sha256_file(args.schedule)
    common_sha = sha256_file(args.common_checkpoint)
    if common["data_sampler_state"]["schedule_sha256"] != schedule_sha:
        raise AssertionError("Schedule differs from common checkpoint")
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    if len(dataset) != EXPECTED_TRAIN:
        raise AssertionError(f"Expected {EXPECTED_TRAIN}, got {len(dataset)}")

    set_seed(42, deterministic=True)
    model, load_audit = load_bcpch_model(common)
    optimizer = build_optimizer(model)
    restore_audit = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    optimizer.global_step = int(common["optimizer_global_step"])
    if restore_audit["skipped"]:
        raise AssertionError(
            f"BCP-CH must restore all optimizer state: {restore_audit['skipped']}"
        )
    restore_rng_state(common["rng_state"])

    history = []
    start_epoch = 20
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if (
            recovery.get("format") != "BCPCH_MATCHED_RECOVERY_V1"
            or recovery.get("common_checkpoint_sha256") != common_sha
            or recovery.get("schedule_sha256") != schedule_sha
        ):
            raise AssertionError("BCP-CH recovery provenance differs")
        model.load_state_dict(recovery["model_state"], strict=True)
        model.hfrm_28_1.set_shared_probes(model.ic1, model.fc8)
        optimizer = build_optimizer(model)
        resumed_restore = restore_named_optimizer_state(
            model, optimizer, recovery["optimizer_named_state"]
        )
        if resumed_restore["skipped"]:
            raise AssertionError(resumed_restore)
        optimizer.global_step = int(recovery["optimizer_global_step"])
        restore_rng_state(recovery["rng_state"])
        history = list(recovery["history"])
        start_epoch = int(recovery["epoch"])
        if start_epoch < 20 or start_epoch > 25 or len(history) != start_epoch - 20:
            raise AssertionError("Invalid BCP-CH recovery epoch/history")
        print(f"BCPCH_RESUME completed_epoch={start_epoch}", flush=True)

    provenance = {
        "experiment_id": "EXP-BCPCH-003",
        "source_commit": source_commit(),
        "base_commit": "e2cb2d3f946fc008d3fbe3f83208b851c275bcc9",
        "common_checkpoint": str(Path(args.common_checkpoint).resolve()),
        "common_checkpoint_sha256": common_sha,
        "common_epoch": common["epoch"],
        "common_global_step": common["optimizer_global_step"],
        "schedule_sha256": schedule_sha,
        "model_load_audit": load_audit,
        "optimizer_restore": restore_audit,
        "optimizer_after_restore": optimizer_summary(optimizer),
        "precision": "bf16",
        "loss_weights": LOSS_WEIGHTS,
        "contrastive": {
            "weight": 0.10,
            "temperature": 0.07,
            "hard_fraction": 0.20,
            "local_kernel": 15,
        },
        "prototype": {
            "source": "IDWT(LL,0,0,0)",
            "cam_threshold": 0.70,
            "mix": 0.50,
        },
        "checkpoint_rule": "Epoch25 FINAL only",
        "test_used": False,
        "luad_used": False,
    }
    if not (output / "provenance.json").exists():
        write_json(output / "provenance.json", provenance)

    initial_path = output / "validation" / "epoch20_initial.json"
    if not initial_path.exists():
        initial = evaluate_bcss(
            model,
            args.val_root,
            num_workers=args.num_workers,
            prediction_output=None,
        )
        initial.update(
            {
                "variant": "BCP-CH-initial",
                "epoch": 20,
                "checkpoint_selection": "locked common Epoch20; diagnostic only",
            }
        )
        write_json(initial_path, initial)
        print(
            "BCPCH_INITIAL_DIAGNOSTIC "
            + json.dumps(
                {
                    "mIoU": 100 * initial["scores"]["final"]["mIoU"],
                    "gt_boundary_prototype_similarity": initial["feature_diagnostics"][
                        "gt_boundary_prototype_similarity"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    restore_rng_state(common["rng_state"] if start_epoch == 20 else recovery["rng_state"])

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, 25):
        loader = make_loader(dataset, schedule, epoch, args.num_workers)
        record = train_cbcch_epoch(
            model, optimizer, loader, schedule["model_seeds"], epoch, "A3"
        )
        record["variant"] = "BCP-CH"
        history.append(record)
        write_json(output / "training_history.json", history)
        torch.save(
            {
                "format": "BCPCH_MATCHED_RECOVERY_V1",
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_named_state": named_optimizer_state(model, optimizer),
                "optimizer_global_step": optimizer.global_step,
                "rng_state": capture_rng_state(),
                "history": history,
                "common_checkpoint_sha256": common_sha,
                "schedule_sha256": schedule_sha,
            },
            recovery_path,
        )
        print("BCPCH_BRANCH_EPOCH " + json.dumps(record, sort_keys=True), flush=True)

    prediction_path = output / "predictions" / "epoch25_validation.npz"
    evaluation = evaluate_bcss(
        model,
        args.val_root,
        num_workers=args.num_workers,
        prediction_output=str(prediction_path),
    )
    evaluation.update(
        {
            "variant": "BCP-CH",
            "epoch": 25,
            "checkpoint_selection": "none; Epoch25 FINAL only",
        }
    )
    write_json(output / "validation" / "epoch25.json", evaluation)
    checkpoint_path = output / "checkpoints" / "epoch25_final.pth"
    torch.save(model.state_dict(), checkpoint_path)
    completion = {
        "status": "BCPCH_MATCHED_COMPLETE",
        "experiment_id": "EXP-BCPCH-003",
        "variant": "BCP-CH",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epochs": [21, 22, 23, 24, 25],
        "global_step": optimizer.global_step,
        "initial_validation": str(initial_path.resolve()),
        "final_validation": evaluation,
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "complete.json", completion)
    if recovery_path.exists():
        recovery_path.unlink()
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
    print("BCPCH_MATCHED_COMPLETE", flush=True)


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
