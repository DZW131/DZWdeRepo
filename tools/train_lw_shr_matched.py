#!/usr/bin/env python3
"""Run one protocol-locked LW-SHR Epoch20 -> Epoch25 continuation."""

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

from tools.lw_shr_common import (
    A0_COMMIT,
    LOSS_WEIGHTS,
    MatchedAugmentationDataset,
    ScheduleBatchSampler,
    VARIANT_TO_MODE,
    build_optimizer,
    load_common_checkpoint,
    load_schedule,
    load_variant_from_common,
    optimizer_summary,
    read_json,
    restore_named_optimizer_state,
    sha256_file,
    write_json,
)
from tools.lw_shr_evaluation import evaluate_bcss


def source_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def official_loss(outputs, target):
    losses = [
        F.multilabel_soft_margin_loss(output, target) for output in outputs[:4]
    ]
    return sum(weight * value for weight, value in zip(LOSS_WEIGHTS, losses))


def classification_counts(probability, target, threshold=0.2):
    predicted = probability > threshold
    actual = target > 0.5
    exact = int((predicted == actual).all(dim=1).sum())
    intersection = (predicted & actual).sum(dim=1).float()
    union = (predicted | actual).sum(dim=1).float()
    accuracy = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return exact, float(accuracy.sum())


def parameter_snapshot(model):
    module = model.hfrm_28_1
    return {
        "gamma_veto": float(module.gamma_veto.detach().float()),
        "gamma_context": float(module.gamma_context.detach().float()),
        "lambda_sf": None
        if module.lambda_sf is None
        else float(module.lambda_sf.detach().float()),
        "filters": model.wavelet_bank.diagnostics(),
    }


def named_optimizer_state(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    state = {}
    for parameter, values in optimizer.state.items():
        name = names.get(id(parameter))
        if name is None:
            continue
        state[name] = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in values.items()
        }
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                key: value
                for key, value in group.items()
                if key != "params"
            }
        )
    return {"state": state, "groups": groups}


def save_recovery(path, model, optimizer, variant, epoch, history):
    payload = {
        "format": "LW_SHR_MATCHED_RECOVERY_V1",
        "variant": variant,
        "mode": VARIANT_TO_MODE[variant],
        "epoch": int(epoch),
        "source_commit": source_commit(),
        "a0_commit": A0_COMMIT,
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_named_state": named_optimizer_state(model, optimizer),
        "optimizer_global_step": int(optimizer.global_step),
        "history": history,
    }
    torch.save(payload, path)


def validate_phase0(path, commit):
    phase0 = read_json(path)
    if phase0.get("phase0_status") != "PASS":
        raise AssertionError("Phase-0 did not pass")
    if phase0.get("source_commit") != commit:
        raise AssertionError("Phase-0 source commit differs from training source")
    return phase0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("A1", "A2", "A3"))
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--phase0-summary", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    commit = source_commit()
    phase0 = validate_phase0(args.phase0_summary, commit)
    common = load_common_checkpoint(args.common_checkpoint)
    schedule = load_schedule(args.schedule)
    output = Path(args.output_dir) / args.variant
    checkpoints = output / "checkpoints"
    predictions = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)

    model, load_audit = load_variant_from_common(args.variant, common)
    optimizer = build_optimizer(model)
    restore = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    if restore["skipped"]:
        raise AssertionError(
            f"Old optimizer state was not restored exactly: {restore['skipped']}"
        )
    optimizer.global_step = int(common["optimizer_global_step"])
    if optimizer.global_step != 20 * 1171:
        raise AssertionError("Frozen continuation must start at global step 23420")

    dataset = MatchedAugmentationDataset(args.train_root)
    history = []
    started = time.time()
    initial_parameters = parameter_snapshot(model)
    print(
        "LW_SHR_MATCHED_START "
        + json.dumps(
            {
                "variant": args.variant,
                "mode": VARIANT_TO_MODE[args.variant],
                "commit": commit,
                "optimizer": optimizer_summary(optimizer),
                "global_step": optimizer.global_step,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for schedule_epoch in range(20, 25):
        epoch = schedule_epoch + 1
        loader = DataLoader(
            dataset,
            batch_sampler=ScheduleBatchSampler(schedule, schedule_epoch),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        model.train()
        loss_sum = 0.0
        examples = exact = 0
        accuracy_sum = 0.0
        epoch_started = time.time()
        for step, (_, image, target) in enumerate(loader):
            model_seed = int(schedule["model_seeds"][schedule_epoch, step])
            torch.manual_seed(model_seed)
            torch.cuda.manual_seed_all(model_seed)
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(image)
                loss = official_loss(outputs, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, step {step + 1}"
                )
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(
                    f"Non-finite gradient at epoch {epoch}, step {step + 1}"
                )
            optimizer.step()
            batch = int(target.shape[0])
            batch_exact, batch_accuracy = classification_counts(outputs[4], target)
            examples += batch
            exact += batch_exact
            accuracy_sum += batch_accuracy
            loss_sum += float(loss.detach()) * batch
            if (step + 1) % 100 == 0:
                print(
                    f"LW_SHR_TRAIN {args.variant} epoch={epoch} "
                    f"step={step + 1}/1171 loss={loss_sum / examples:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )

        if optimizer.global_step != epoch * 1171:
            raise AssertionError(
                f"Unexpected global step {optimizer.global_step} after epoch {epoch}"
            )
        training = {
            "epoch": epoch,
            "loss": loss_sum / examples,
            "exact_match": exact / examples,
            "accuracy": accuracy_sum / examples,
            "examples": examples,
            "lr_after_epoch": [float(group["lr"]) for group in optimizer.param_groups],
            "seconds": time.time() - epoch_started,
            "parameters": parameter_snapshot(model),
        }
        evaluation = evaluate_bcss(
            model,
            args.val_root,
            num_workers=args.num_workers,
            prediction_output=(
                predictions / "epoch25_validation.npz" if epoch == 25 else None
            ),
        )
        row = {"training": training, "validation": evaluation}
        history.append(row)
        write_json(output / "history.json", history)
        save_recovery(
            checkpoints / "recovery_latest.pth",
            model,
            optimizer,
            args.variant,
            epoch,
            history,
        )
        score = evaluation["scores"]["final"]
        print(
            f"LW_SHR_EPOCH_COMPLETE variant={args.variant} epoch={epoch} "
            f"val_mIoU={100.0 * score['mIoU']:.6f} "
            f"val_mDice={100.0 * score['mDice']:.6f}",
            flush=True,
        )

    final_checkpoint = checkpoints / "epoch25_final.pth"
    torch.save(
        {
            "format": "LW_SHR_MATCHED_FINAL_V1",
            "variant": args.variant,
            "mode": VARIANT_TO_MODE[args.variant],
            "epoch": 25,
            "source_commit": commit,
            "a0_commit": A0_COMMIT,
            "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
            "schedule_sha256": sha256_file(args.schedule),
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        final_checkpoint,
    )
    completion = {
        "experiment": "LW-SHR Phase-1 Matched Epoch20-to-25 Continuation",
        "status": "COMPLETE",
        "variant": args.variant,
        "mode": VARIANT_TO_MODE[args.variant],
        "source_commit": commit,
        "a0_commit": A0_COMMIT,
        "phase0_summary": str(Path(args.phase0_summary).resolve()),
        "phase0_status": phase0["phase0_status"],
        "common_checkpoint": str(Path(args.common_checkpoint).resolve()),
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "schedule": str(Path(args.schedule).resolve()),
        "schedule_sha256": sha256_file(args.schedule),
        "checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "predictions": str((predictions / "epoch25_validation.npz").resolve()),
        "optimizer_restore": restore,
        "optimizer_initial": optimizer_summary(optimizer),
        "initial_parameters": initial_parameters,
        "final_parameters": parameter_snapshot(model),
        "history": history,
        "runtime_seconds": time.time() - started,
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "completion.json", completion)
    print(json.dumps(completion, indent=2), flush=True)
    print(f"LW_SHR_{args.variant}_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
