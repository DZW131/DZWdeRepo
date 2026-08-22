#!/usr/bin/env python3
"""Run one frozen five-epoch TCRD utility branch from the common A0 checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.tcrd_dynamics import BRANCHES
from tool.infer_tcrd import infer_bcss
from tools.tcrd_common import (
    BRANCH_DIRS, EXPECTED_A0_SHA256, LOSS_WEIGHTS,
    MatchedAugmentationDataset, ScheduleBatchSampler,
    build_optimizer, compact_scores, dataset_fingerprint, load_branch_model,
    load_schedule, set_deterministic_seed, sha256_file, write_json,
)


def mechanism_parameters(model):
    if model.tcrd is None:
        return {
            "steps": 0, "kappa": None, "eta_d": None,
            "eta_r": None, "competition_matrix": None,
        }
    matrix = model.tcrd.competition_matrix()
    return {
        "steps": model.tcrd.steps,
        "kappa": None if model.tcrd.kappa is None else float(model.tcrd.kappa.detach()),
        "eta_d": None if model.tcrd.eta_d is None else float(model.tcrd.eta_d.detach()),
        "eta_r": None if model.tcrd.eta_r is None else float(model.tcrd.eta_r.detach()),
        "competition_matrix": None if matrix is None else matrix.detach().cpu().tolist(),
    }


def evaluate(model, val_root, branch_dir, point, epoch, num_workers, save_predictions):
    prediction_path = None
    if save_predictions:
        prediction_path = branch_dir / "predictions" / "epoch5_validation.npz"
    fused, standalone, runtime, diagnostics = infer_bcss(
        model, val_root, "bf16", num_workers, prediction_path
    )
    record = {
        "branch": model.branch, "point": point, "epoch": epoch,
        "scores": compact_scores(fused),
        "standalone_cam28_1": compact_scores(standalone),
        "runtime": runtime, "diagnostics": diagnostics,
        "mechanism_parameters": mechanism_parameters(model),
        "checkpoint_selection": "none; fixed evaluation points only",
        "test_used": False, "luad_used": False,
    }
    write_json(branch_dir / "validation" / f"{point}.json", record)
    return record


def run(args):
    if args.branch not in BRANCHES:
        raise ValueError(args.branch)
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("TCRD utility training evaluates BCSS validation only")
    if sha256_file(args.a0_checkpoint) != EXPECTED_A0_SHA256:
        raise AssertionError("A0 checkpoint SHA mismatch")
    experiment = Path(args.experiment_dir)
    branch_dir = experiment / BRANCH_DIRS[args.branch]
    if branch_dir.exists():
        raise FileExistsError(f"Branch output already exists: {branch_dir}")
    for name in ("validation", "train", "checkpoints", "predictions"):
        (branch_dir / name).mkdir(parents=True, exist_ok=True)

    schedule = load_schedule(args.schedule)
    metadata = json.loads(Path(args.schedule).with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["schedule_sha256"] != sha256_file(args.schedule):
        raise AssertionError("Schedule SHA mismatch")
    if schedule["indices"].shape != (5, 1171, 20):
        raise AssertionError(f"Frozen schedule shape mismatch: {schedule['indices'].shape}")

    set_deterministic_seed(42)
    dataset = MatchedAugmentationDataset(args.train_root, image_size=224)
    if len(dataset) != 23422:
        raise AssertionError(f"Expected 23,422 BCSS samples, found {len(dataset)}")
    if dataset_fingerprint(dataset.base, args.train_root) != metadata["dataset_order_sha256"]:
        raise AssertionError("Dataset order does not match frozen schedule")

    model, incompat = load_branch_model(args.branch, args.a0_checkpoint, "cuda")
    history = []
    started = time.time()
    print(f"TCRD_BRANCH_START branch={args.branch}", flush=True)
    history.append(evaluate(
        model, args.val_root, branch_dir, "step0", 0, args.num_workers, False
    ))

    optimizer, tail_base_lr, initial_group_lrs = build_optimizer(
        model, steps_per_epoch=1171, epochs=5
    )
    optimizer_initial = [
        {
            "index": index, "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "momentum": group["momentum"],
            "parameters": sum(parameter.numel() for parameter in group["params"]),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]
    write_json(branch_dir / "provenance.json", {
        "branch": args.branch,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "a0_checkpoint": args.a0_checkpoint,
        "a0_sha256": EXPECTED_A0_SHA256,
        "schedule": args.schedule,
        "schedule_sha256": metadata["schedule_sha256"],
        "dataset_order_sha256": metadata["dataset_order_sha256"],
        "common_checkpoint_missing_keys": incompat.missing_keys,
        "common_checkpoint_unexpected_keys": incompat.unexpected_keys,
        "tail_base_lr_derived_at_epoch20_of_25": tail_base_lr,
        "initial_group_lrs": initial_group_lrs,
        "optimizer_groups": optimizer_initial,
        "poly_power": 0.9, "epochs": 5, "batch_size": 20,
        "precision": "bf16", "seed": 42,
        "deterministic_augmentation": True,
        "common_model_seed_per_step": True,
        "all_original_sshr_parameters_trainable": True,
        "segmentation_gt_used_in_training": False,
        "test_used": False, "luad_used": False,
    })

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    epoch_summaries = []
    for epoch in range(5):
        sampler = ScheduleBatchSampler(
            schedule["indices"], schedule["augmentation_seeds"], epoch
        )
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=False,
        )
        model.train()
        epoch_started = time.time()
        loss_sum = 0.0
        samples = 0
        for step, (_, image, label) in enumerate(loader):
            expected_seed = int(schedule["model_seeds"][epoch, step])
            torch.manual_seed(expected_seed)
            torch.cuda.manual_seed_all(expected_seed)
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(image, active_labels=label)
                losses = [
                    F.multilabel_soft_margin_loss(prediction, label)
                    for prediction in output[:4]
                ]
                total_loss = sum(weight * value for weight, value in zip(LOSS_WEIGHTS, losses))
            total_loss.backward()
            optimizer.step()
            batch_size = image.shape[0]
            loss_sum += float(total_loss.detach()) * batch_size
            samples += batch_size
            if (step + 1) % 100 == 0:
                print(
                    f"TCRD_TRAIN branch={args.branch} epoch={epoch + 1}/5 "
                    f"batch={step + 1}/1171 global_step={optimizer.global_step}/5855 "
                    f"loss={loss_sum / samples:.6f} lr={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )
        epoch_summary = {
            "branch": args.branch, "epoch": epoch + 1,
            "classification_loss": loss_sum / samples,
            "epoch_seconds": time.time() - epoch_started,
            "global_step": optimizer.global_step,
            "lr_end": optimizer.param_groups[0]["lr"],
            "mechanism_parameters": mechanism_parameters(model),
            "all_finite": bool(np.isfinite(loss_sum / samples)),
        }
        epoch_summaries.append(epoch_summary)
        print("TCRD_EPOCH_SUMMARY " + json.dumps(epoch_summary, sort_keys=True), flush=True)
        validation = evaluate(
            model, args.val_root, branch_dir, f"epoch{epoch + 1}", epoch + 1,
            args.num_workers, save_predictions=(epoch == 4),
        )
        history.append(validation)
        write_json(branch_dir / "train" / "epoch_history.json", epoch_summaries)
        write_json(branch_dir / "validation" / "history.json", history)

    checkpoint = branch_dir / "checkpoints" / "epoch5_final.pth"
    torch.save(model.state_dict(), checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    completion = {
        "status": "TCRD_UTILITY_BRANCH_COMPLETE",
        "branch": args.branch, "epochs": 5, "steps": optimizer.global_step,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "total_seconds": time.time() - started,
        "final_validation": history[-1],
        "test_used": False, "luad_used": False,
    }
    write_json(branch_dir / "complete.json", completion)
    print(json.dumps({
        "branch": args.branch,
        "epoch5_mIoU": 100 * history[-1]["scores"]["mIoU"],
        "checkpoint_sha256": checkpoint_sha,
    }, indent=2), flush=True)
    print(f"TCRD_BRANCH_COMPLETE branch={args.branch}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=BRANCHES)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
