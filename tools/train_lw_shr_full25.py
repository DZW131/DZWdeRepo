#!/usr/bin/env python3
"""Train frozen LW-SHR A2 from the official initialization for 25 epochs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import resnet38_cls, resnet38d
from tool.GenDataset import Stage1_TrainDataset
from tools.lw_shr_common import (
    A0_COMMIT,
    LOSS_WEIGHTS,
    build_optimizer,
    capture_rng_state,
    optimizer_summary,
    restore_rng_state,
    set_seed,
    sha256_file,
    write_json,
)
from tools.lw_shr_evaluation import evaluate_bcss


FROZEN_A2_COMMIT = "a91f45dd0f343c850f179398a02fab3075fccac0"
REQUIRED_EPOCHS = (1, 5, 10, 15, 20, 25)
EXPECTED_PRETRAINED_SHA256 = (
    "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16"
)


def source_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)


def official_loss(outputs, target):
    return sum(
        weight * F.multilabel_soft_margin_loss(output, target)
        for weight, output in zip(LOSS_WEIGHTS, outputs[:4])
    )


def classification_counts(probability, target, threshold=0.2):
    predicted = probability > threshold
    actual = target > 0.5
    exact = int((predicted == actual).all(dim=1).sum())
    intersection = (predicted & actual).sum(dim=1).float()
    union = (predicted | actual).sum(dim=1).float()
    accuracy = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return exact, float(accuracy.sum())


def initialize_a2(pretrained_path):
    """Match every common A0 parameter and the official training RNG state."""

    pretrained_sha256 = sha256_file(pretrained_path)
    if pretrained_sha256 != EXPECTED_PRETRAINED_SHA256:
        raise AssertionError("Released pretrained-weight SHA256 mismatch")

    # Reconstruct the exact seed42 A0 initialization used by the six-run baseline.
    set_seed(42, deterministic=False)
    reference = resnet38_cls.Net(4)
    official_training_rng = capture_rng_state()
    pretrained = resnet38d.convert_mxnet_to_torch(pretrained_path)
    pretrained_load = reference.load_state_dict(pretrained, strict=False)
    reference_state = reference.state_dict()

    # Construct A2 independently, then overwrite every common parameter/buffer
    # with the exact A0 initialization. New wavelet parameters retain the frozen
    # Haar/random initialization from the a91f45d implementation.
    set_seed(42, deterministic=False)
    model = resnet38_cls.Net(
        4, wavelet_hfrm_mode="learnable", wavelet_hfrm_stages="28_1"
    )
    a2_load = model.load_state_dict(reference_state, strict=False)
    allowed_prefixes = ("wavelet_bank.", "hfrm_28_1.wavelet_gate.")
    if a2_load.unexpected_keys or not a2_load.missing_keys:
        raise AssertionError(f"Unexpected A2 initialization load: {a2_load}")
    if any(not key.startswith(allowed_prefixes) for key in a2_load.missing_keys):
        raise AssertionError(f"Non-A2 missing initialization keys: {a2_load.missing_keys}")

    candidate_state = model.state_dict()
    common_keys = sorted(set(reference_state).intersection(candidate_state))
    common_max_abs = 0.0
    common_nonidentical = []
    for name in common_keys:
        left = reference_state[name]
        right = candidate_state[name]
        difference = 0.0 if left.numel() == 0 else float((left - right).abs().max())
        common_max_abs = max(common_max_abs, difference)
        if difference != 0.0:
            common_nonidentical.append(name)
    if common_nonidentical:
        raise AssertionError(f"A0/A2 common initialization differs: {common_nonidentical}")

    initialization = {
        "method": "seed42 A0 reference initialization, then strict common-key transfer",
        "a0_commit": A0_COMMIT,
        "frozen_a2_commit": FROZEN_A2_COMMIT,
        "pretrained_path": str(Path(pretrained_path).resolve()),
        "pretrained_sha256": pretrained_sha256,
        "pretrained_missing_keys": list(pretrained_load.missing_keys),
        "pretrained_unexpected_keys": list(pretrained_load.unexpected_keys),
        "a2_missing_keys": list(a2_load.missing_keys),
        "a2_unexpected_keys": list(a2_load.unexpected_keys),
        "common_key_count": len(common_keys),
        "common_initialization_max_abs_diff": common_max_abs,
        "common_initialization_exact": not common_nonidentical,
        "wavelet_initial": model.wavelet_bank.diagnostics(),
        "gate_output_projection_weight_nonzero": int(
            torch.count_nonzero(
                model.hfrm_28_1.wavelet_gate.output_projection.weight
            )
        ),
        "gate_output_projection_bias_nonzero": int(
            torch.count_nonzero(
                model.hfrm_28_1.wavelet_gate.output_projection.bias
            )
        ),
        "training_rng_restored_to_a0_post_initialization": True,
    }
    del reference, reference_state, pretrained
    restore_rng_state(official_training_rng)
    return model.cuda(), initialization


def optimizer_membership_audit(model, optimizer):
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    counts = {}
    for parameter in grouped:
        counts[id(parameter)] = counts.get(id(parameter), 0) + 1
    names = dict(model.named_parameters())
    required = {
        name: parameter
        for name, parameter in names.items()
        if name.startswith("wavelet_bank.")
        or name.startswith("hfrm_28_1.wavelet_gate.")
    }
    rows = []
    failures = []
    for name, parameter in required.items():
        groups = [
            index
            for index, group in enumerate(optimizer.param_groups)
            if any(candidate is parameter for candidate in group["params"])
        ]
        expected_group = 3 if name.endswith(".bias") else 2
        row = {
            "name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "requires_grad": parameter.requires_grad,
            "occurrences": counts.get(id(parameter), 0),
            "optimizer_groups": groups,
            "expected_scratch_group": expected_group,
            "scratch_group": groups == [expected_group],
        }
        rows.append(row)
        if (
            not parameter.requires_grad
            or row["occurrences"] != 1
            or groups != [expected_group]
        ):
            failures.append(row)
    if failures:
        raise AssertionError(f"A2 optimizer membership failure: {failures}")
    return {
        "status": "PASS",
        "new_parameter_count": len(required),
        "new_parameter_numel": sum(parameter.numel() for parameter in required.values()),
        "all_trainable_exactly_once_in_scratch_group": True,
        "optimizer_groups": optimizer_summary(optimizer),
        "parameters": rows,
    }


def group_grad_norm(module):
    total = 0.0
    present = False
    finite = True
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        present = True
        gradient = parameter.grad.detach().float()
        finite = finite and bool(torch.isfinite(gradient).all())
        total += float(gradient.square().sum())
    return None if not present else float(np.sqrt(total)), finite


def gradient_snapshot(model, epoch, step_in_epoch, global_step):
    gate = model.hfrm_28_1.wavelet_gate
    modules = {
        "gate_output_projection": gate.output_projection,
        "ll_branch": gate.ll_branch,
        "lh_branch": gate.lh_branch,
        "hl_branch": gate.hl_branch,
        "hh_branch": gate.hh_branch,
        "gate_fusion": gate.fusion_depthwise,
    }
    row = {
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
    }
    finite = True
    for name, module in modules.items():
        norm, valid = group_grad_norm(module)
        row[f"{name}_grad_norm"] = norm
        finite = finite and valid
    for name, parameter in (
        ("dec_lo", model.wavelet_bank.dec_lo),
        ("dec_hi", model.wavelet_bank.dec_hi),
    ):
        if parameter.grad is None:
            row[f"{name}_grad_norm"] = None
        else:
            gradient = parameter.grad.detach().float()
            finite = finite and bool(torch.isfinite(gradient).all())
            row[f"{name}_grad_norm"] = float(gradient.norm())
    row["all_finite"] = finite
    return row


def parameter_snapshot(model):
    module = model.hfrm_28_1
    return {
        "gamma_veto": float(module.gamma_veto.detach().float()),
        "gamma_context": float(module.gamma_context.detach().float()),
        "filters": model.wavelet_bank.diagnostics(),
    }


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_gradient_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.seed != 42 or args.epochs != 25:
        raise AssertionError("Frozen Phase-1.5 run is exactly seed42/25 epochs")

    output = Path(args.output_dir)
    for name in ("configs", "checkpoints", "diagnostics", "predictions", "validation"):
        (output / name).mkdir(parents=True, exist_ok=True)

    model, initialization = initialize_a2(args.pretrained)
    dataset = Stage1_TrainDataset(
        data_path=args.train_root,
        transform=transforms.Compose([transforms.ToTensor()]),
        dataset="bcss",
        img_size=224,
    )
    if len(dataset) != 23422:
        raise AssertionError(f"Unexpected BCSS training size: {len(dataset)}")
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=data_generator,
    )
    if len(loader) != 1171:
        raise AssertionError(f"Unexpected steps per epoch: {len(loader)}")
    optimizer = build_optimizer(model, max_step=args.epochs * len(loader))
    optimizer_audit = optimizer_membership_audit(model, optimizer)
    write_json(output / "configs" / "initialization_audit.json", initialization)
    write_json(output / "configs" / "a2_full25_optimizer_audit.json", optimizer_audit)
    configuration = {
        "experiment": "LW-SHR Phase-1.5 A2 Full-25",
        "source_commit": source_commit(),
        "frozen_architecture_commit": FROZEN_A2_COMMIT,
        "a0_commit": A0_COMMIT,
        "command": [sys.executable, *sys.argv],
        "dataset": "BCSS-WSSS",
        "train_root": str(Path(args.train_root).resolve()),
        "val_root": str(Path(args.val_root).resolve()),
        "train_samples": len(dataset),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": 20,
        "steps_per_epoch": len(loader),
        "image_size": 224,
        "precision": "bf16",
        "optimizer": "released PolyOptimizer/SGD",
        "base_lr": 0.01,
        "momentum": 0.0005,
        "weight_decay_groups": [0.0005, 0.0, 0.0005, 0.0],
        "poly_power": 0.9,
        "loss_weights": list(LOSS_WEIGHTS),
        "validation_epochs": list(REQUIRED_EPOCHS),
        "checkpoint_rule": "epoch25 FINAL only; no validation selection",
        "test_used": False,
        "luad_used": False,
        "smoke_steps": args.smoke_steps,
    }
    write_json(output / "configs" / "training_config.json", configuration)

    capture_steps = set(range(1, 11))
    for epoch in (1, 5, 10, 20, 25):
        capture_steps.add(epoch * len(loader))
    history = []
    gradients = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    stop_after = args.smoke_steps if args.smoke_steps > 0 else None

    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        model.train()
        loss_sum = accuracy_sum = 0.0
        examples = exact = 0
        epoch_started = time.time()
        for step_index, (_, image, target) in enumerate(loader):
            step_in_epoch = step_index + 1
            global_step = epoch_index * len(loader) + step_in_epoch
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(image)
                loss = official_loss(outputs, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at global step {global_step}")
            loss.backward()
            if global_step in capture_steps:
                snapshot = gradient_snapshot(
                    model, epoch, step_in_epoch, global_step
                )
                if not snapshot["all_finite"]:
                    raise FloatingPointError(f"Non-finite gradient at {global_step}")
                gradients.append(snapshot)
                write_gradient_csv(
                    output / "diagnostics" / "gradient_diagnostics.csv", gradients
                )
            optimizer.step()

            batch = int(target.shape[0])
            batch_exact, batch_accuracy = classification_counts(outputs[4], target)
            examples += batch
            exact += batch_exact
            accuracy_sum += batch_accuracy
            loss_sum += float(loss.detach()) * batch
            if global_step % 100 == 0:
                print(
                    f"LW_SHR_FULL25_TRAIN epoch={epoch} step={step_in_epoch}/1171 "
                    f"global={global_step}/29275 loss={loss_sum / examples:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )
            if stop_after is not None and global_step >= stop_after:
                break

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
        row = {"training": training, "validation": None}
        if stop_after is None and epoch in REQUIRED_EPOCHS:
            row["validation"] = evaluate_bcss(
                model,
                args.val_root,
                num_workers=args.num_workers,
                prediction_output=(
                    output / "predictions" / "epoch25_validation.npz"
                    if epoch == 25
                    else None
                ),
                mechanism_diagnostics=True,
            )
            write_json(
                output / "validation" / f"epoch{epoch:02d}.json",
                row["validation"],
            )
            score = row["validation"]["scores"]["final"]
            print(
                f"LW_SHR_FULL25_EVAL epoch={epoch} "
                f"val_mIoU={100.0 * score['mIoU']:.6f} "
                f"val_mDice={100.0 * score['mDice']:.6f}",
                flush=True,
            )
        history.append(row)
        write_json(output / "training_history.json", history)

        if stop_after is not None:
            break
        recovery = {
            "format": "LW_SHR_A2_FULL25_RECOVERY_V1",
            "epoch": epoch,
            "source_commit": source_commit(),
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "optimizer_global_step": int(optimizer.global_step),
            "data_generator_state": data_generator.get_state(),
            "rng_state": capture_rng_state(),
            "history": history,
        }
        atomic_torch_save(recovery, output / "checkpoints" / "recovery_latest.pth")

    if stop_after is not None:
        smoke = {
            "status": "PASS",
            "steps": stop_after,
            "source_commit": source_commit(),
            "initialization": initialization,
            "optimizer_audit": optimizer_audit,
            "gradient_diagnostics": gradients,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "all_finite": True,
            "test_used": False,
            "validation_used": False,
        }
        write_json(output / "smoke_summary.json", smoke)
        print("LW_SHR_FULL25_SMOKE_PASS", flush=True)
        return

    final_checkpoint = output / "checkpoints" / "epoch25_final.pth"
    atomic_torch_save(
        {
            "format": "LW_SHR_A2_FULL25_FINAL_V1",
            "epoch": 25,
            "seed": 42,
            "source_commit": source_commit(),
            "frozen_architecture_commit": FROZEN_A2_COMMIT,
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        final_checkpoint,
    )
    completion = {
        "status": "COMPLETE",
        "experiment": "LW-SHR Phase-1.5 A2 Full-25",
        "source_commit": source_commit(),
        "frozen_architecture_commit": FROZEN_A2_COMMIT,
        "command": configuration["command"],
        "training_config": str(
            (output / "configs" / "training_config.json").resolve()
        ),
        "checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "predictions": str(
            (output / "predictions" / "epoch25_validation.npz").resolve()
        ),
        "history": history,
        "initialization": initialization,
        "optimizer_audit": optimizer_audit,
        "gradient_diagnostics": gradients,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime_seconds": time.time() - started,
        "test_used": False,
        "luad_used": False,
        "checkpoint_selection": "FINAL epoch25 only",
    }
    write_json(output / "completion.json", completion)
    print(json.dumps(completion, indent=2), flush=True)
    print("LW_SHR_A2_FULL25_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
