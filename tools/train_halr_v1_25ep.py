#!/usr/bin/env python3
"""Fresh frozen-protocol BCSS seed42 25-epoch HALR-v1 training."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from network.resnet38_cls import Net
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset
from tools.halr_objectives import apply_pair_transform, epoch_alpha, halr_terms


FROZEN = {
    "dataset": "bcss", "seed": 42, "epochs": 25, "batch_size": 20,
    "image_size": 224, "base_lr": 0.01, "weight_decay": 0.0005,
    "loss_weights": {"56": 0.10, "28_1": 0.15, "28_2": 0.25, "deep": 0.50},
    "lambda_cv": 0.05, "lambda_hd": 0.05, "tau": 1.0,
    "precision": "bf16", "checkpoint_epochs": (5, 10, 15, 20, 25),
    "expected_train_samples": 23422,
}

APPROVED_PRETRAINED_MISSING = {
    f"{layer}.{state}"
    for layer in ("bn45", "bn52")
    for state in ("weight", "bias", "running_mean", "running_var")
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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
    np.random.seed(worker_seed); random.seed(worker_seed)


def compute_acc(predicted, truth):
    overlap = len(set(predicted) & set(truth))
    union = len(truth) + len(predicted) - overlap
    return overlap / union if union else 1.0


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
        and key not in APPROVED_PRETRAINED_MISSING
    ]
    if missing_backbone or incompat.unexpected_keys:
        raise AssertionError(
            f"Pretrained mismatch: missing={missing_backbone}, "
            f"unexpected={incompat.unexpected_keys}"
        )
    return {
        "path": str(Path(path).resolve()), "sha256": sha256_file(path),
        "size_bytes": Path(path).stat().st_size,
        "missing_keys": list(incompat.missing_keys),
        "approved_missing_keys": sorted(set(incompat.missing_keys) & APPROVED_PRETRAINED_MISSING),
        "unexpected_keys": list(incompat.unexpected_keys),
        "missing_backbone_keys": missing_backbone,
    }


def source_manifest():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    files = (
        "network/resnet38_cls.py", "tools/halr_objectives.py",
        "tools/train_halr_v1_25ep.py", "tools/preflight_halr_v1.py",
        "tools/eval_halr_v1.py", "tool/infer_halr.py",
        "tests/test_halr_v1.py",
    )
    return commit, {name: sha256_file(REPO_ROOT / name) for name in files}


def optimizer_for(model, maximum_step):
    groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer(
        [
            {"params": groups[0], "lr": 0.01, "weight_decay": 0.0005},
            {"params": groups[1], "lr": 0.02, "weight_decay": 0.0},
            {"params": groups[2], "lr": 0.10, "weight_decay": 0.0005},
            {"params": groups[3], "lr": 0.20, "weight_decay": 0.0},
        ],
        lr=0.01, weight_decay=0.0005, max_step=maximum_step,
    )
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    occurrences = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            occurrences[id(parameter)] = occurrences.get(id(parameter), 0) + 1
    if set(occurrences) != trainable or any(count != 1 for count in occurrences.values()):
        raise AssertionError("Official optimizer coverage is not exactly once per trainable parameter")
    return optimizer


def checkpoint(model, output, epoch, common_meta, epoch_record):
    name = f"epoch{epoch:02d}_final.pth" if epoch == 25 else f"epoch{epoch:02d}.pth"
    path = output / "checkpoints" / name
    torch.save(model.state_dict(), path)
    dynamics = {
        key: epoch_record[key]
        for key in (
            "jsd28", "jsddeep", "weight28", "weightdeep",
            "fraction_weight28_gt", "fraction_weightdeep_gt", "hierarchy_agreement",
        )
    }
    meta = {
        **common_meta, "epoch": epoch, "filename": name,
        "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        "primary_final": epoch == 25, "teacher_dynamics": dynamics,
    }
    write_json(path.with_suffix(".json"), meta)
    write_json(output / "mechanism" / f"epoch{epoch:02d}_teacher_dynamics.json", dynamics)
    return meta


def classification_components(outputs, labels):
    return {
        name: F.multilabel_soft_margin_loss(value, labels)
        for name, value in zip(("56", "28_1", "28_2", "deep"), outputs[:4])
    }


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
        raise RuntimeError("Formal HALR-v1 training requires CUDA BF16")
    output = Path(args.output_dir)
    if output.exists():
        existing = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*") if path.is_file()
        }
        unexpected = existing - {"provenance/preflight.json"}
        if unexpected:
            raise FileExistsError(f"Refusing populated output directory: {sorted(unexpected)}")
    for directory in ("provenance", "train", "checkpoints", "validation", "mechanism", "figures", "docs"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    set_seed(FROZEN["seed"])
    commit, source_hashes = source_manifest()
    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    if len(dataset) != FROZEN["expected_train_samples"]:
        raise AssertionError(f"Expected 23422 samples, got {len(dataset)}")
    data_generator = torch.Generator(); data_generator.manual_seed(42)
    pair_generator = torch.Generator(); pair_generator.manual_seed(42)
    loader = DataLoader(
        dataset, batch_size=20, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
        generator=data_generator,
    )

    model = Net(4)
    baseline_parameters = sum(parameter.numel() for parameter in Net(4).parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if total_parameters != baseline_parameters:
        raise AssertionError("HALR-v1 must add zero model parameters")
    pretrained = load_pretrained(model, args.weights)
    model = model.cuda(); model.train()
    maximum_step = len(loader) * 25
    optimizer = optimizer_for(model, maximum_step)
    optimizer_manifest = [
        {
            "index": index, "parameters": sum(p.numel() for p in group["params"]),
            "lr": group["lr"], "momentum": group["momentum"],
            "weight_decay": group["weight_decay"],
        }
        for index, group in enumerate(optimizer.param_groups)
    ]
    if any(item["momentum"] != 0.0005 for item in optimizer_manifest):
        raise AssertionError("Released optimizer momentum changed")

    config = {
        **FROZEN, "checkpoint_epochs": list(FROZEN["checkpoint_epochs"]),
        "alpha_by_epoch": {str(epoch): epoch_alpha(epoch) for epoch in range(1, 26)},
        "trainroot": str(Path(args.trainroot).resolve()),
        "pretrained": str(Path(args.weights).resolve()),
        "num_workers": args.num_workers,
        "network": "network.resnet38_cls (clean official A0)",
        "paired_view": "per-sample seed42 hflip/vflip after official augmentation; exact inverse",
        "optimizer": "released PolyOptimizer/SGD", "poly_power": 0.9,
        "augmentation": "released horizontal+vertical flips before paired-view generation",
        "checkpoint_selection": "epoch25 FINAL only; no validation selection",
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(output / "provenance" / "training_config.json", config)
    provenance = {
        "base_a0_commit": "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9",
        "halr_source_commit": commit, "config_sha256": config_hash,
        "source_sha256": source_hashes, "pretrained": pretrained,
        "parameters": {
            "sshr_total": baseline_parameters, "halr_total": total_parameters,
            "new_parameters": total_parameters - baseline_parameters,
        },
        "optimizer_param_groups_initial": optimizer_manifest,
        "dataset_samples": len(dataset), "steps_per_epoch": len(loader),
        "maximum_steps": maximum_step,
        "python": sys.version, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(), "command": " ".join(sys.argv),
        "fresh_training": True, "loaded_trained_checkpoint": False,
        "validation_during_training": False, "test_or_luad_used": False,
        "segmentation_gt_used_in_training": False,
    }
    write_json(output / "provenance" / "manifest.json", provenance)
    checkpoint_meta = {
        "source_commit": commit, "config_sha256": config_hash,
        "seed": 42, "dataset": "BCSS",
        "selection": "epoch-number only; epoch25 FINAL is primary",
    }

    history = []; checkpoint_records = []
    started = time.time()
    for epoch in range(1, 26):
        epoch_started = time.time(); model.train(); alpha = epoch_alpha(epoch)
        sums = {
            "total_loss": 0.0, "classification_loss": 0.0,
            "loss_56": 0.0, "loss_28_1": 0.0,
            "loss_28_2": 0.0, "loss_deep": 0.0,
            "cvle_loss": 0.0, "rahd_loss": 0.0,
            "cvle_weighted": 0.0, "rahd_weighted": 0.0,
            "jsd28": 0.0, "jsddeep": 0.0,
            "weight28": 0.0, "weightdeep": 0.0,
            "fraction_weight28_gt": 0.0, "fraction_weightdeep_gt": 0.0,
            "hierarchy_agreement": 0.0, "valid_fraction": 0.0,
        }
        examples = exact = 0; accuracy_sum = 0.0
        torch.cuda.reset_peak_memory_stats()
        for batch_index, (_, images, labels) in enumerate(loader, start=1):
            images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            flip_codes = torch.randint(0, 2, (images.shape[0],), generator=pair_generator).to(images.device)
            paired_images = apply_pair_transform(images, flip_codes)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs_view1 = model(images)
                outputs_view2 = model(paired_images)
                components_view1 = classification_components(outputs_view1, labels)
                components_view2 = classification_components(outputs_view2, labels)
                components = {
                    name: 0.5 * (components_view1[name] + components_view2[name])
                    for name in FROZEN["loss_weights"]
                }
                classification_loss = sum(
                    FROZEN["loss_weights"][name] * value
                    for name, value in components.items()
                )
                terms = halr_terms(
                    outputs_view1[6], outputs_view1[8],
                    outputs_view2[6], outputs_view2[8], labels, flip_codes,
                )
                cvle_weighted = FROZEN["lambda_cv"] * alpha * terms["cvle_loss"]
                rahd_weighted = FROZEN["lambda_hd"] * alpha * terms["rahd_loss"]
                total_loss = classification_loss + cvle_weighted + rahd_weighted
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss epoch={epoch} batch={batch_index}")
            optimizer.zero_grad(); total_loss.backward(); optimizer.step()

            values = {
                "total_loss": total_loss, "classification_loss": classification_loss,
                **{f"loss_{name}": value for name, value in components.items()},
                "cvle_loss": terms["cvle_loss"], "rahd_loss": terms["rahd_loss"],
                "cvle_weighted": cvle_weighted, "rahd_weighted": rahd_weighted,
                **{name: terms[name] for name in (
                    "jsd28", "jsddeep", "weight28", "weightdeep",
                    "fraction_weight28_gt", "fraction_weightdeep_gt",
                    "hierarchy_agreement", "valid_fraction",
                )},
            }
            for name, value in values.items():
                sums[name] += float(value.detach().float().item())

            probability = 0.5 * (outputs_view1[4] + outputs_view2[4])
            probability = probability.detach().float().cpu().numpy()
            truth = labels.detach().float().cpu().numpy()
            for observed, target in zip(probability, truth):
                predicted = np.where(observed > 0.2)[0]
                actual = np.where(target == 1)[0]
                examples += 1; exact += int(np.array_equal(predicted, actual))
                accuracy_sum += compute_acc(predicted, actual)
            if optimizer.global_step % 100 == 0:
                print(
                    f"TRAIN_PROGRESS epoch={epoch:02d}/25 batch={batch_index:04d}/{len(loader)} "
                    f"step={optimizer.global_step}/{maximum_step} "
                    f"loss={sums['total_loss']/batch_index:.6f} "
                    f"cvle={sums['cvle_loss']/batch_index:.6f} "
                    f"rahd={sums['rahd_loss']/batch_index:.6f} "
                    f"w28={sums['weight28']/batch_index:.4f} "
                    f"wdeep={sums['weightdeep']/batch_index:.4f} "
                    f"alpha={alpha:.2f} lr={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )
        batches = len(loader)
        record = {
            "epoch": epoch, "alpha": alpha,
            **{name: value / batches for name, value in sums.items()},
            "training_exact_match": exact / examples,
            "training_accuracy": accuracy_sum / examples,
            "gamma_veto_28_1": float(model.hfrm_28_1.gamma_veto.detach().float().item()),
            "gamma_context_28_1": float(model.hfrm_28_1.gamma_context.detach().float().item()),
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
            checkpoint_records.append(checkpoint(model, output, epoch, checkpoint_meta, record))
            write_json(output / "checkpoints" / "manifest.json", checkpoint_records)

    completion = {
        "status": "HALR_V1_TRAINING_COMPLETE", "epochs": 25,
        "primary_checkpoint": "checkpoints/epoch25_final.pth",
        "total_seconds": time.time() - started,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "peak_cuda_memory_gib": max(row["peak_cuda_memory_gib"] for row in history),
        "final_mechanism": {
            key: history[-1][key]
            for key in (
                "jsd28", "jsddeep", "weight28", "weightdeep",
                "fraction_weight28_gt", "fraction_weightdeep_gt",
                "hierarchy_agreement", "gamma_veto_28_1", "gamma_context_28_1",
            )
        },
    }
    write_json(output / "train" / "history.json", history)
    write_json(output / "train" / "training_complete.json", completion)
    print("HALR_V1_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
