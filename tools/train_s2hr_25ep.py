#!/usr/bin/env python3
"""Fresh BCSS seed42, 25-epoch training for frozen S²HR-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
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

from network.resnet38_cls import Net as SSHRNet
from network.resnet38_cls_s2hr import Net
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset


FROZEN = {
    "dataset": "bcss",
    "seed": 42,
    "epochs": 25,
    "batch_size": 20,
    "image_size": 224,
    "base_lr": 0.01,
    "weight_decay_argument": 0.0005,
    "loss_weights": {"56": 0.10, "28_1": 0.15, "28_2": 0.25, "deep": 0.50},
    "precision": "bf16",
    "checkpoint_epochs": (5, 10, 15, 20, 25),
    "expected_train_samples": 23422,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_acc(predicted, truth):
    overlap = len(set(predicted) & set(truth))
    union = len(truth) + len(predicted) - overlap
    return overlap / union if union else 1.0


def source_manifest():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    source_files = (
        "network/resnet38_cls.py",
        "network/resnet38_cls_s2hr.py",
        "network/s2hfrm28_1.py",
        "tests/test_s2hr_preflight.py",
        "tools/preflight_s2hr.py",
        "tools/train_s2hr_25ep.py",
        "tools/eval_s2hr.py",
        "tool/infer_s2hr.py",
    )
    return commit, {
        name: sha256_file(REPO_ROOT / name) for name in source_files
    }


def load_pretrained(model, path):
    path = str(path)
    if path.endswith(".params"):
        from network.resnet38d import convert_mxnet_to_torch

        state = convert_mxnet_to_torch(path)
    elif path.endswith(".pth"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    else:
        raise ValueError("Pretrained weights must be .params or .pth")
    incompat = model.load_state_dict(state, strict=False)
    missing_backbone = [
        key for key in incompat.missing_keys
        if not key.startswith(("hfrm_", "ic_56.", "ic1.", "ic2.", "fc8."))
    ]
    if missing_backbone or incompat.unexpected_keys:
        raise AssertionError(
            f"Pretrained backbone mismatch: missing={missing_backbone}, "
            f"unexpected={incompat.unexpected_keys}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "size_bytes": Path(path).stat().st_size,
        "missing_keys": list(incompat.missing_keys),
        "unexpected_keys": list(incompat.unexpected_keys),
        "missing_backbone_keys": missing_backbone,
    }


def checkpoint(model, output, epoch, common_meta):
    name = f"epoch{epoch:02d}_final.pth" if epoch == 25 else f"epoch{epoch:02d}.pth"
    path = output / "checkpoints" / name
    torch.save(model.state_dict(), path)
    meta = {
        **common_meta,
        "epoch": epoch,
        "filename": name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "primary_final": epoch == 25,
    }
    write_json(path.with_suffix(".json"), meta)
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S²HR formal training requires CUDA BF16")
    output = Path(args.output_dir)
    if output.exists():
        existing_files = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        }
        allowed_preflight = {"provenance/preflight.json"}
        unexpected_files = existing_files - allowed_preflight
        if unexpected_files:
            raise FileExistsError(
                "Refusing output directory containing files other than the "
                f"approved preflight record: {sorted(unexpected_files)}"
            )
    for directory in ("provenance", "train", "checkpoints", "validation", "figures", "docs"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    set_seed(FROZEN["seed"])
    commit, source_hashes = source_manifest()
    config = {
        **FROZEN,
        "checkpoint_epochs": list(FROZEN["checkpoint_epochs"]),
        "trainroot": str(Path(args.trainroot).resolve()),
        "pretrained": str(Path(args.weights).resolve()),
        "num_workers": args.num_workers,
        "network": "network.resnet38_cls_s2hr",
        "optimizer": "released PolyOptimizer/SGD",
        "poly_power": 0.9,
        "augmentation": "released Stage1_TrainDataset horizontal+vertical flips",
    }
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    write_json(output / "provenance" / "training_config.json", config)

    dataset = Stage1_TrainDataset(
        data_path=args.trainroot, dataset="bcss", img_size=FROZEN["image_size"]
    )
    if len(dataset) != FROZEN["expected_train_samples"]:
        raise AssertionError(f"Expected 23422 parsed BCSS samples, got {len(dataset)}")
    generator = torch.Generator()
    generator.manual_seed(FROZEN["seed"])
    loader = DataLoader(
        dataset,
        batch_size=FROZEN["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = Net(n_class=4)
    baseline_parameters = sum(parameter.numel() for parameter in SSHRNet(4).parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_manifest = {
        "sshr_total": baseline_parameters,
        "s2hr_total": total_parameters,
        "new_parameters": total_parameters - baseline_parameters,
        "overhead_fraction": (total_parameters - baseline_parameters) / baseline_parameters,
    }
    pretrained = load_pretrained(model, args.weights)
    model = model.cuda()
    model.train()
    parameter_manifest["official_trainable_after_freeze"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    groups = model.get_parameter_groups()
    group_two_ids = [id(parameter) for parameter in groups[2]]
    for name, parameter in (
        ("gamma_spatial", model.hfrm_28_1.gamma_spatial),
        ("rho_boundary_raw", model.hfrm_28_1.rho_boundary_raw),
    ):
        if group_two_ids.count(id(parameter)) != 1:
            raise AssertionError(f"{name} must occur exactly once in scratch weight group")
    maximum_step = (len(dataset) // FROZEN["batch_size"]) * FROZEN["epochs"]
    optimizer_groups = [
        {"params": groups[0], "lr": 0.01, "weight_decay": 0.0005},
        {"params": groups[1], "lr": 0.02, "weight_decay": 0.0},
        {"params": groups[2], "lr": 0.10, "weight_decay": 0.0005},
        {"params": groups[3], "lr": 0.20, "weight_decay": 0.0},
    ]
    optimizer = torchutils.PolyOptimizer(
        optimizer_groups, lr=0.01, weight_decay=0.0005, max_step=maximum_step
    )
    optimizer_manifest = [
        {
            "index": index,
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "lr": group["lr"],
            "momentum": group["momentum"],
            "weight_decay": group["weight_decay"],
        }
        for index, group in enumerate(optimizer.param_groups)
    ]
    if any(item["momentum"] != 0.0005 for item in optimizer_manifest):
        raise AssertionError(f"Released optimizer momentum changed: {optimizer_manifest}")

    provenance = {
        "base_a0_commit": "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9",
        "s2hr_source_commit": commit,
        "config_sha256": config_hash,
        "source_sha256": source_hashes,
        "pretrained": pretrained,
        "parameters": parameter_manifest,
        "optimizer_param_groups_initial": optimizer_manifest,
        "dataset_samples": len(dataset),
        "steps_per_epoch": len(dataset) // FROZEN["batch_size"],
        "maximum_steps": maximum_step,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "fresh_training": True,
        "loaded_trained_sshr": False,
        "test_or_luad_used": False,
    }
    write_json(output / "provenance" / "manifest.json", provenance)
    common_checkpoint_meta = {
        "source_commit": commit,
        "config_sha256": config_hash,
        "seed": FROZEN["seed"],
        "dataset": "BCSS",
        "selection": "epoch-number only; epoch25 FINAL is primary",
    }

    history = []
    checkpoint_records = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    amp_dtype = torch.bfloat16
    for epoch in range(1, FROZEN["epochs"] + 1):
        epoch_started = time.time()
        model.train()
        sums = {
            "total_loss": 0.0, "loss_56": 0.0, "loss_28_1": 0.0,
            "loss_28_2": 0.0, "loss_deep": 0.0,
            "semantic_discrepancy_abs_mean": 0.0, "boundary_fraction": 0.0,
            "ch_gate_boundary_mean": 0.0, "ch_gate_interior_mean": 0.0,
        }
        examples = exact = 0
        accuracy_sum = 0.0
        torch.cuda.reset_peak_memory_stats()
        for batch_index, (_, images, labels) in enumerate(loader, start=1):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images, image_label=labels, mode="train")
                out_56, out_28_1, out_28_2, out_deep, y_deep = outputs[:5]
                losses = {
                    "56": F.multilabel_soft_margin_loss(out_56, labels),
                    "28_1": F.multilabel_soft_margin_loss(out_28_1, labels),
                    "28_2": F.multilabel_soft_margin_loss(out_28_2, labels),
                    "deep": F.multilabel_soft_margin_loss(out_deep, labels),
                }
                total_loss = sum(
                    FROZEN["loss_weights"][name] * value for name, value in losses.items()
                )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sums["total_loss"] += float(total_loss.detach().float().item())
            for name, value in losses.items():
                sums[f"loss_{name}"] += float(value.detach().float().item())
            diagnostics = model.last_s2hr_diagnostics
            for name in (
                "semantic_discrepancy_abs_mean", "boundary_fraction",
                "ch_gate_boundary_mean", "ch_gate_interior_mean",
            ):
                sums[name] += float(diagnostics[name].float().item())
            probability = y_deep.detach().float().cpu().numpy()
            truth = labels.detach().float().cpu().numpy()
            for observed, target in zip(probability, truth):
                predicted_classes = np.where(observed > 0.2)[0]
                target_classes = np.where(target == 1)[0]
                examples += 1
                exact += int(np.array_equal(predicted_classes, target_classes))
                accuracy_sum += compute_acc(predicted_classes, target_classes)
            if optimizer.global_step % 100 == 0:
                print(
                    f"TRAIN_PROGRESS epoch={epoch:02d}/25 batch={batch_index:04d}/{len(loader)} "
                    f"step={optimizer.global_step}/{maximum_step} "
                    f"loss={sums['total_loss']/batch_index:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )

        batches = len(loader)
        record = {
            "epoch": epoch,
            **{name: value / batches for name, value in sums.items()},
            "training_exact_match": exact / examples,
            "training_accuracy": accuracy_sum / examples,
            "gamma_global_28_1": float(model.hfrm_28_1.gamma_veto.detach().float().item()),
            "gamma_context_28_1": float(model.hfrm_28_1.gamma_context.detach().float().item()),
            "gamma_spatial": float(model.hfrm_28_1.gamma_spatial.detach().float().item()),
            "rho_boundary": float(torch.sigmoid(model.hfrm_28_1.rho_boundary_raw).detach().float().item()),
            "lr_end": float(optimizer.param_groups[0]["lr"]),
            "optimizer_momentum": float(optimizer.param_groups[0]["momentum"]),
            "epoch_seconds": time.time() - epoch_started,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "global_step": optimizer.global_step,
        }
        history.append(record)
        with open(output / "train" / "epoch_history.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print("EPOCH_SUMMARY " + json.dumps(record, sort_keys=True), flush=True)
        if epoch in FROZEN["checkpoint_epochs"]:
            checkpoint_records.append(
                checkpoint(model, output, epoch, common_checkpoint_meta)
            )
            write_json(output / "checkpoints" / "manifest.json", checkpoint_records)

    completion = {
        "status": "S2HR_TRAINING_COMPLETE",
        "epochs": 25,
        "primary_checkpoint": "checkpoints/epoch25_final.pth",
        "total_seconds": time.time() - started,
        "mean_epoch_seconds": float(np.mean([item["epoch_seconds"] for item in history])),
        "peak_cuda_memory_gib": max(item["peak_cuda_memory_gib"] for item in history),
        "final_mechanism": {
            key: history[-1][key]
            for key in (
                "gamma_global_28_1", "gamma_context_28_1", "gamma_spatial",
                "rho_boundary", "semantic_discrepancy_abs_mean", "boundary_fraction",
                "ch_gate_boundary_mean", "ch_gate_interior_mean",
            )
        },
    }
    write_json(output / "train" / "history.json", history)
    write_json(output / "train" / "training_complete.json", completion)
    print("S2HR_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
