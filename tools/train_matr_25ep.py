#!/usr/bin/env python3
"""Fresh frozen-protocol BCSS seed42 25-epoch MATR-v1 training."""

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

from network.resnet38_cls import Net as SSHRNet
from network.resnet38_cls_matr import Net
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset
from tools.matr_objectives import epoch_alpha, ot_mtr_loss


FROZEN = {
    "dataset": "bcss", "seed": 42, "epochs": 25, "batch_size": 20,
    "image_size": 224, "base_lr": 0.01, "weight_decay": 0.0005,
    "loss_weights": {"56": 0.10, "28_1": 0.15, "28_2": 0.25, "deep": 0.50},
    "modes_per_class": 2, "tau_mode": 1.0,
    "seed_fraction": 0.30, "min_seeds": 4, "max_seeds": 128,
    "epsilon_ot": 0.1, "sinkhorn_iterations": 20, "lambda_ot": 0.05,
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
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
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
        "network/resnet38_cls.py", "network/resnet38_cls_matr.py",
        "network/matr_hfrm28.py", "network/matr_sparse_context.py",
        "network/matr_multiprototype_head.py", "tool/sinkhorn.py",
        "tool/infer_matr.py", "tools/matr_objectives.py",
        "tools/train_matr_25ep.py", "tools/preflight_matr.py",
        "tools/eval_matr.py", "tests/test_matr_head.py",
        "tests/test_matr_ot.py", "tests/test_matr_sparse_context.py",
        "tests/test_matr_protocol.py",
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
    if set(occurrences) != trainable or any(value != 1 for value in occurrences.values()):
        raise AssertionError("Optimizer coverage is not exactly once per trainable parameter")
    return optimizer


def checkpoint(model, output, epoch, common_meta, record):
    name = f"epoch{epoch:02d}_final.pth" if epoch == 25 else f"epoch{epoch:02d}.pth"
    path = output / "checkpoints" / name
    torch.save(model.state_dict(), path)
    mechanism_keys = (
        "mode_cosine", "mode_activation_ratio", "ot_mean_seeds_by_class",
        "ot_transport_mass_by_class", "gamma_adapt", "mean_abs_offset",
        "p95_abs_offset", "mean_modulation", "delta_context15_ratio",
    )
    meta = {
        **common_meta, "epoch": epoch, "filename": name,
        "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        "primary_final": epoch == 25,
        "mechanism": {key: record[key] for key in mechanism_keys},
    }
    write_json(path.with_suffix(".json"), meta)
    write_json(output / "mechanism" / f"epoch{epoch:02d}.json", meta["mechanism"])
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
        raise RuntimeError("Formal MATR-v1 training requires CUDA BF16")
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

    set_seed(42)
    commit, source_hashes = source_manifest()
    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    if len(dataset) != 23422:
        raise AssertionError(f"Expected 23422 samples, got {len(dataset)}")
    generator = torch.Generator(); generator.manual_seed(42)
    loader = DataLoader(
        dataset, batch_size=20, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
        generator=generator,
    )
    model = Net(4)
    baseline_parameters = sum(parameter.numel() for parameter in SSHRNet(4).parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    overhead = total_parameters - baseline_parameters
    overhead_percent = 100.0 * overhead / baseline_parameters
    if overhead <= 0 or overhead_percent >= 0.10:
        raise AssertionError(f"MATR parameter overhead outside budget: {overhead_percent:.6f}%")
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
        "alpha_ot_by_epoch": {str(epoch): epoch_alpha(epoch) for epoch in range(1, 26)},
        "trainroot": str(Path(args.trainroot).resolve()),
        "pretrained": str(Path(args.weights).resolve()), "num_workers": args.num_workers,
        "network": "network.resnet38_cls_matr",
        "optimizer": "released PolyOptimizer/SGD", "poly_power": 0.9,
        "augmentation": "released horizontal+vertical flips",
        "checkpoint_selection": "epoch25 FINAL only; no validation selection",
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(output / "provenance" / "training_config.json", config)
    provenance = {
        "base_a0_commit": "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9",
        "matr_source_commit": commit, "config_sha256": config_hash,
        "source_sha256": source_hashes, "pretrained": pretrained,
        "parameters": {
            "sshr_total": baseline_parameters, "matr_total": total_parameters,
            "new_parameters": overhead, "overhead_percent": overhead_percent,
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

    history = []; checkpoint_records = []; started = time.time()
    for epoch in range(1, 26):
        epoch_started = time.time(); model.train(); alpha = epoch_alpha(epoch)
        scalar_sums = {
            name: 0.0 for name in (
                "total_loss", "classification_loss", "loss_56", "loss_28_1",
                "loss_28_2", "loss_deep", "ot_loss", "ot_weighted",
                "ot_mean_seeds", "mean_abs_offset", "p95_abs_offset",
                "mean_modulation", "delta_context_rms", "context15_rms",
                "delta_context15_ratio", "max_row_marginal_error",
                "max_col_marginal_error",
            )
        }
        valid_by_class = np.zeros(4); seed_sum_by_class = np.zeros(4)
        transport_mass_sum = np.zeros((4, 2)); activation_sum = np.zeros(4)
        valid_pairs_total = 0.0; examples = exact = 0; accuracy_sum = 0.0
        torch.cuda.reset_peak_memory_stats()
        for batch_index, (_, images, labels) in enumerate(loader, start=1):
            images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images)
                components = classification_components(outputs, labels)
                classification_loss = sum(
                    FROZEN["loss_weights"][name] * value
                    for name, value in components.items()
                )
                ot = ot_mtr_loss(
                    outputs[11], outputs[6], outputs[10],
                    model.ic1.mode_weights(), labels,
                )
                ot_weighted = 0.05 * alpha * ot["loss"]
                total_loss = classification_loss + ot_weighted
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss epoch={epoch} batch={batch_index}")
            optimizer.zero_grad(); total_loss.backward(); optimizer.step()

            sacr = model.hfrm_28_1.last_sacr_diagnostics
            values = {
                "total_loss": total_loss, "classification_loss": classification_loss,
                **{f"loss_{name}": value for name, value in components.items()},
                "ot_loss": ot["loss"], "ot_weighted": ot_weighted,
                "ot_mean_seeds": ot["mean_seeds"],
                "mean_abs_offset": sacr["mean_abs_offset"],
                "p95_abs_offset": sacr["p95_abs_offset"],
                "mean_modulation": sacr["mean_modulation"],
                "delta_context_rms": sacr["delta_rms"],
                "context15_rms": sacr["context15_rms"],
                "delta_context15_ratio": sacr["delta_context15_ratio"],
                "max_row_marginal_error": ot["max_row_marginal_error"],
                "max_col_marginal_error": ot["max_col_marginal_error"],
            }
            for name, value in values.items():
                scalar_sums[name] += float(value.detach().float().item())
            valid_pairs_total += float(ot["valid_pairs"].item())
            valid_by_class += ot["valid_pairs_by_class"].cpu().numpy()
            seed_sum_by_class += ot["seed_sum_by_class"].cpu().numpy()
            transport_mass_sum += ot["transport_mass_sum"].cpu().numpy()
            activation_sum += ot["mode_activation_ratio"].cpu().numpy()

            probability = outputs[4].detach().float().cpu().numpy()
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
                    f"loss={scalar_sums['total_loss']/batch_index:.6f} "
                    f"ot={scalar_sums['ot_loss']/batch_index:.6f} "
                    f"gamma_adapt={model.hfrm_28_1.sacr.gamma_adapt.item():.6f} "
                    f"alpha={alpha:.2f} lr={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )
        batches = len(loader)
        seed_means = np.divide(
            seed_sum_by_class, valid_by_class,
            out=np.zeros_like(seed_sum_by_class), where=valid_by_class > 0,
        )
        transport_means = np.divide(
            transport_mass_sum, valid_by_class[:, None],
            out=np.zeros_like(transport_mass_sum), where=valid_by_class[:, None] > 0,
        )
        record = {
            "epoch": epoch, "alpha_ot": alpha,
            **{name: value / batches for name, value in scalar_sums.items()},
            "ot_valid_pairs": valid_pairs_total,
            "ot_valid_pairs_by_class": valid_by_class.tolist(),
            "ot_mean_seeds_by_class": seed_means.tolist(),
            "ot_transport_mass_by_class": transport_means.tolist(),
            "mode_cosine": model.ic1.mode_cosine().detach().float().cpu().tolist(),
            "mode_activation_ratio": (activation_sum / batches).tolist(),
            "training_exact_match": exact / examples,
            "training_accuracy": accuracy_sum / examples,
            "gamma_veto_28_1": float(model.hfrm_28_1.gamma_veto.detach().float()),
            "gamma_context_28_1": float(model.hfrm_28_1.gamma_context.detach().float()),
            "gamma_adapt": float(model.hfrm_28_1.sacr.gamma_adapt.detach().float()),
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
        "status": "MATR_V1_TRAINING_COMPLETE", "epochs": 25,
        "primary_checkpoint": "checkpoints/epoch25_final.pth",
        "total_seconds": time.time() - started,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "peak_cuda_memory_gib": max(row["peak_cuda_memory_gib"] for row in history),
        "final_mechanism": {
            key: history[-1][key]
            for key in (
                "mode_cosine", "mode_activation_ratio", "ot_valid_pairs",
                "ot_mean_seeds_by_class", "ot_transport_mass_by_class",
                "gamma_adapt", "mean_abs_offset", "p95_abs_offset",
                "mean_modulation", "delta_context_rms", "context15_rms",
                "delta_context15_ratio",
            )
        },
    }
    write_json(output / "train" / "history.json", history)
    write_json(output / "train" / "training_complete.json", completion)
    print("MATR_V1_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
