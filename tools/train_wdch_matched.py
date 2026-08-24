#!/usr/bin/env python3
"""Create epoch20 common state or run one matched WD-CH continuation branch."""

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

from network import resnet38_cls, resnet38_wdch, resnet38d
from tools.wdch_common import (
    A0_COMMIT,
    EXPECTED_TRAIN,
    LOSS_WEIGHTS,
    MatchedAugmentationDataset,
    ScheduleBatchSampler,
    build_optimizer,
    capture_rng_state,
    dataset_fingerprint,
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


def compute_accuracy(probability, labels, threshold=0.2):
    predicted = probability > threshold
    truth = labels > 0.5
    exact = (predicted == truth).all(dim=1).float().mean()
    intersection = (predicted & truth).sum(dim=1).float()
    union = (predicted | truth).sum(dim=1).float()
    jaccard = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return float(exact), float(jaccard.mean())


def load_pretrained(model, path):
    converted = resnet38d.convert_mxnet_to_torch(path)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in converted.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    forbidden_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(("hfrm_", "ic_56", "ic1", "ic2", "fc8", "bn45", "bn52"))
    ]
    unexpected = sorted(set(converted).difference(model_state))
    shape_mismatch = {
        key: {"pretrained": list(value.shape), "model": list(model_state[key].shape)}
        for key, value in converted.items()
        if key in model_state and model_state[key].shape != value.shape
    }
    if forbidden_missing or unexpected or shape_mismatch:
        raise AssertionError(
            f"Pretrained backbone audit failed: forbidden_missing={forbidden_missing}, "
            f"unexpected={unexpected}, shape_mismatch={shape_mismatch}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "loaded_keys": len(compatible),
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "forbidden_backbone_missing": forbidden_missing,
    }


def make_loader(dataset, schedule, epoch, workers):
    return DataLoader(
        dataset,
        batch_sampler=ScheduleBatchSampler(schedule, epoch),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )


def train_epoch(model, optimizer, loader, model_seeds, epoch, label):
    model.train()
    started = time.time()
    loss_sum = 0.0
    exact_sum = 0.0
    accuracy_sum = 0.0
    samples = 0
    for step, (_, image, target) in enumerate(loader):
        seed = int(model_seeds[epoch, step])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(image)
            losses = [
                F.multilabel_soft_margin_loss(output, target)
                for output in outputs[:4]
            ]
            total_loss = sum(
                weight * value for weight, value in zip(LOSS_WEIGHTS, losses)
            )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1} step {step}")
        total_loss.backward()
        optimizer.step()
        batch = image.shape[0]
        exact, accuracy = compute_accuracy(torch.sigmoid(outputs[3]).detach(), target)
        loss_sum += float(total_loss.detach()) * batch
        exact_sum += exact * batch
        accuracy_sum += accuracy * batch
        samples += batch
        if (step + 1) % 100 == 0:
            print(
                f"WDCH_TRAIN branch={label} epoch={epoch + 1}/25 "
                f"batch={step + 1}/1171 global_step={optimizer.global_step}/29275 "
                f"loss={loss_sum/samples:.6f} exact={exact_sum/samples:.6f} "
                f"accuracy={accuracy_sum/samples:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}",
                flush=True,
            )
    return {
        "branch": label,
        "epoch": epoch + 1,
        "global_step": optimizer.global_step,
        "classification_loss": loss_sum / samples,
        "training_exact_match": exact_sum / samples,
        "training_accuracy": accuracy_sum / samples,
        "lr_end": float(optimizer.param_groups[0]["lr"]),
        "seconds": time.time() - started,
        "all_finite": bool(np.isfinite(loss_sum / samples)),
    }


def snapshot(model, optimizer, epoch, schedule_path, history, extra=None):
    value = {
        "format": "WDCH_MATCHED_COMMON_V1",
        "model_state": model.state_dict(),
        "optimizer_named_state": named_optimizer_state(model, optimizer),
        "optimizer_global_step": optimizer.global_step,
        "scaler_state": torch.amp.GradScaler("cuda", enabled=False).state_dict(),
        "epoch": epoch,
        "rng_state": capture_rng_state(),
        "data_sampler_state": {
            "schedule": str(Path(schedule_path).resolve()),
            "schedule_sha256": sha256_file(schedule_path),
            "next_epoch_index": epoch,
        },
        "history": history,
        "source_commit": source_commit(),
        "a0_commit": A0_COMMIT,
    }
    if extra:
        value.update(extra)
    return value


def run_common(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty common output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(42, deterministic=True)
    schedule = load_schedule(args.schedule)
    if schedule["indices"].shape != (25, 1171, 20):
        raise AssertionError(schedule["indices"].shape)
    dataset = MatchedAugmentationDataset(args.train_root)
    if len(dataset) != EXPECTED_TRAIN:
        raise AssertionError(len(dataset))
    model = resnet38_cls.Net(4).cuda()
    pretrained = load_pretrained(model, args.pretrained)
    optimizer = build_optimizer(model)
    initial_optimizer = optimizer_summary(optimizer)
    if any(abs(group["momentum"] - 0.0005) > 1.0e-12 for group in initial_optimizer):
        raise AssertionError("Released optimizer momentum changed")
    history = []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for epoch in range(20):
        loader = make_loader(dataset, schedule, epoch, args.num_workers)
        record = train_epoch(
            model, optimizer, loader, schedule["model_seeds"], epoch, "COMMON_C0"
        )
        history.append(record)
        write_json(output / "training_history.json", history)
        print("WDCH_COMMON_EPOCH " + json.dumps(record, sort_keys=True), flush=True)
        if (epoch + 1) % 5 == 0 and epoch + 1 < 20:
            recovery = output / "common_recovery_latest.pth"
            torch.save(
                snapshot(
                    model, optimizer, epoch + 1, args.schedule, history,
                    {"pretrained_audit": pretrained},
                ),
                recovery,
            )
    common = output / "common_epoch20.pth"
    torch.save(
        snapshot(
            model, optimizer, 20, args.schedule, history,
            {
                "pretrained_audit": pretrained,
                "optimizer_initial": initial_optimizer,
                "dataset_fingerprint": dataset_fingerprint(dataset.base, args.train_root),
            },
        ),
        common,
    )
    completion = {
        "status": "WDCH_COMMON_EPOCH20_COMPLETE",
        "checkpoint": str(common.resolve()),
        "checkpoint_sha256": sha256_file(common),
        "epoch": 20,
        "global_step": optimizer.global_step,
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "optimizer": optimizer_summary(optimizer),
        "pretrained": pretrained,
        "test_used": False,
    }
    write_json(output / "complete.json", completion)
    print(json.dumps(completion, indent=2), flush=True)
    print("WDCH_COMMON_EPOCH20_COMPLETE", flush=True)


def load_branch_model(branch, common, kernel):
    state = dict(common["model_state"])
    if branch == "C0":
        model = resnet38_cls.Net(4)
        incompat = model.load_state_dict(state, strict=True)
        load_audit = {"missing": incompat.missing_keys, "unexpected": incompat.unexpected_keys}
    else:
        model = resnet38_wdch.Net(4, wdch_kernel_size=kernel)
        old = state.pop("hfrm_28_1.context_conv.weight")
        incompat = model.load_state_dict(state, strict=False)
        expected = {
            "hfrm_28_1.wdch.haar.analysis_filters",
            "hfrm_28_1.wdch.haar.synthesis_filters",
            "hfrm_28_1.wdch.ll_context.weight",
        }
        if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
            raise AssertionError(str(incompat))
        load_audit = {
            "expected_missing": sorted(expected),
            "unexpected": incompat.unexpected_keys,
            "replaced_parameter": "hfrm_28_1.context_conv.weight",
            "replaced_shape": list(old.shape),
            "new_shape": list(model.hfrm_28_1.wdch.ll_context.weight.shape),
            "new_initialization": f"uniform 1/{kernel**2}",
        }
    return model.cuda(), load_audit


def run_branch(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty branch output: {output}")
    for name in ("validation", "checkpoints", "predictions"):
        (output / name).mkdir(parents=True, exist_ok=True)
    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    if common.get("format") != "WDCH_MATCHED_COMMON_V1" or common.get("epoch") != 20:
        raise AssertionError("Invalid common epoch20 state")
    if common["data_sampler_state"]["schedule_sha256"] != sha256_file(args.schedule):
        raise AssertionError("Schedule differs from common checkpoint")
    phase0 = json.loads(Path(args.phase0_summary).read_text(encoding="utf-8"))
    if phase0["phase0_status"] != "PASS":
        raise AssertionError("Phase 0 not passed")
    kernel = int(phase0["selected_kernel"])
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    set_seed(42, deterministic=True)
    model, load_audit = load_branch_model(args.branch, common, kernel)
    optimizer = build_optimizer(model)
    restore_audit = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    optimizer.global_step = int(common["optimizer_global_step"])
    expected_skipped = [] if args.branch == "C0" else ["hfrm_28_1.context_conv.weight"]
    skipped_names = [row["name"] for row in restore_audit["skipped"]]
    if skipped_names != expected_skipped:
        raise AssertionError(f"Optimizer restore mismatch: {restore_audit['skipped']}")
    restore_rng_state(common["rng_state"])
    provenance = {
        "branch": args.branch,
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
        "test_used": False,
    }
    write_json(output / "provenance.json", provenance)
    history = []
    validation_history = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(20, 25):
        loader = make_loader(dataset, schedule, epoch, args.num_workers)
        record = train_epoch(
            model, optimizer, loader, schedule["model_seeds"], epoch, args.branch
        )
        history.append(record)
        print("WDCH_BRANCH_EPOCH " + json.dumps(record, sort_keys=True), flush=True)
        prediction_path = (
            output / "predictions" / "epoch25_validation.npz"
            if epoch == 24
            else None
        )
        evaluation = evaluate_bcss(
            model,
            args.val_root,
            num_workers=args.num_workers,
            prediction_output=str(prediction_path) if prediction_path else None,
        )
        evaluation.update(
            {
                "branch": args.branch,
                "epoch": epoch + 1,
                "checkpoint_selection": "none; epoch25 FINAL is primary",
            }
        )
        validation_history.append(evaluation)
        write_json(output / "training_history.json", history)
        write_json(output / "validation" / "history.json", validation_history)
        write_json(output / "validation" / f"epoch{epoch + 1}.json", evaluation)
        print(
            f"WDCH_VALIDATION branch={args.branch} epoch={epoch + 1} "
            f"mIoU={100*evaluation['scores']['final']['mIoU']:.4f} "
            f"mDice={100*evaluation['scores']['final']['mDice']:.4f}",
            flush=True,
        )
    checkpoint = output / "checkpoints" / "epoch25_final.pth"
    torch.save(model.state_dict(), checkpoint)
    completion = {
        "status": "WDCH_MATCHED_BRANCH_COMPLETE",
        "branch": args.branch,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "epochs": [21, 22, 23, 24, 25],
        "global_step": optimizer.global_step,
        "final_validation": validation_history[-1],
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "test_used": False,
    }
    write_json(output / "complete.json", completion)
    print(json.dumps({
        "status": completion["status"],
        "branch": args.branch,
        "mIoU": 100 * completion["final_validation"]["scores"]["final"]["mIoU"],
        "checkpoint_sha256": completion["checkpoint_sha256"],
    }, indent=2), flush=True)
    print(f"WDCH_MATCHED_BRANCH_COMPLETE branch={args.branch}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("common", "branch"), required=True)
    parser.add_argument("--branch", choices=("C0", "W1"))
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root")
    parser.add_argument("--pretrained")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--common-checkpoint")
    parser.add_argument("--phase0-summary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "common":
        if not args.pretrained:
            parser.error("--pretrained is required for common mode")
        run_common(args)
    else:
        for required in ("branch", "val_root", "common_checkpoint", "phase0_summary"):
            if not getattr(args, required):
                parser.error(f"--{required.replace('_', '-')} is required for branch mode")
        run_branch(args)


if __name__ == "__main__":
    main()
