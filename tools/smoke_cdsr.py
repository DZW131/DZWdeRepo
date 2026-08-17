"""Run the frozen 20-step real-BCSS CDSR optimization-readiness audit."""

import argparse
import hashlib
import importlib
import json
import math
import platform
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from train_sshr import collect_cdsr_epoch_record


STAGES = {
    "stage1": "hfrm_56",
    "stage2": "hfrm_28_1",
    "stage3": "hfrm_28_2",
}
LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pretrained_state(path):
    path = str(path)
    if path.endswith(".params"):
        return importlib.import_module(
            "network.resnet38d"
        ).convert_mxnet_to_torch(path)
    if path.endswith(".pth"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        return checkpoint
    raise ValueError(f"Unsupported pretrained weights: {path}")


def alpha_parameters(model):
    parameters = {}
    for stage, attribute in STAGES.items():
        gate = getattr(model, attribute).selective_gate
        parameters[f"{stage}.alpha_sem"] = gate.alpha_sem_logit
        parameters[f"{stage}.alpha_ctx"] = gate.alpha_ctx_logit
    return parameters


def make_optimizer(model, lr, weight_decay, max_step):
    groups = model.get_parameter_groups()
    parameters = [
        {"params": groups[0], "lr": lr, "weight_decay": weight_decay},
        {"params": groups[1], "lr": 2 * lr, "weight_decay": 0.0},
        {"params": groups[2], "lr": 10 * lr, "weight_decay": weight_decay},
        {"params": groups[3], "lr": 20 * lr, "weight_decay": 0.0},
    ]
    optimizer = PolyOptimizer(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        max_step=max_step,
    )
    return optimizer, groups


def make_weight_decay_shadows(alpha_map, lr, weight_decay, max_step, device):
    shadows = {
        name: nn.Parameter(parameter.detach().clone().to(device))
        for name, parameter in alpha_map.items()
    }
    optimizer = PolyOptimizer(
        [
            {
                "params": list(shadows.values()),
                "lr": 10 * lr,
                "weight_decay": weight_decay,
            }
        ],
        lr=lr,
        weight_decay=weight_decay,
        max_step=max_step,
    )
    return shadows, optimizer


def classification_loss(outputs, labels):
    return sum(
        weight * F.multilabel_soft_margin_loss(output, labels)
        for weight, output in zip(LOSS_WEIGHTS, outputs[:4])
    )


def optimizer_manifest(optimizer):
    return [
        {
            "index": index,
            "parameter_tensors": len(group["params"]),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "lr": group["lr"],
            "momentum": group.get("momentum", 0.0),
            "weight_decay": group.get("weight_decay", 0.0),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", default="bcss")
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--formal-epochs", default=25, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 20:
        raise ValueError("CDSR readiness requires at least 20 real training steps")
    if not torch.cuda.is_available():
        raise RuntimeError("CDSR batch20 BF16 readiness requires CUDA")

    set_seed(args.seed)
    pretrained_state = load_pretrained_state(args.weights)
    baseline = Net(n_class=4, rectification_mode="uniform")
    baseline_load = baseline.load_state_dict(pretrained_state, strict=False)
    model = Net(n_class=4, rectification_mode="cdsr")
    cdsr_load = model.load_state_dict(pretrained_state, strict=False)
    expected_alpha_keys = {
        name
        for name, _ in model.named_parameters()
        if ".selective_gate." in name
    }
    baseline_missing = set(baseline_load.missing_keys)
    cdsr_missing = set(cdsr_load.missing_keys)
    pretrained_audit_pass = (
        set(cdsr_load.unexpected_keys) == set(baseline_load.unexpected_keys)
        and cdsr_missing - baseline_missing == expected_alpha_keys
        and baseline_missing == cdsr_missing - expected_alpha_keys
    )
    pretrained_audit = {
        "baseline_missing_keys": sorted(baseline_missing),
        "cdsr_missing_keys": sorted(cdsr_missing),
        "baseline_unexpected_keys": sorted(baseline_load.unexpected_keys),
        "cdsr_unexpected_keys": sorted(cdsr_load.unexpected_keys),
        "cdsr_only_missing_keys": sorted(cdsr_missing - baseline_missing),
        "pass": pretrained_audit_pass,
    }
    del baseline, pretrained_state

    device = torch.device("cuda")
    model.to(device)
    alpha_map = alpha_parameters(model)
    dataset = Stage1_TrainDataset(
        data_path=str(args.train_root),
        transform=transforms.Compose([transforms.ToTensor()]),
        dataset=args.dataset,
        img_size=args.image_size,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    max_step = (len(dataset) // args.batch_size) * args.formal_epochs
    optimizer, parameter_groups = make_optimizer(
        model, args.lr, args.weight_decay, max_step
    )
    grouped_ids = [
        id(parameter) for group in parameter_groups for parameter in group
    ]
    trainable_ids = [
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    alpha_ids = {id(parameter) for parameter in alpha_map.values()}
    optimizer_coverage_pass = (
        len(grouped_ids) == len(set(grouped_ids))
        and set(grouped_ids) == set(trainable_ids)
        and alpha_ids.issubset({id(parameter) for parameter in parameter_groups[2]})
    )
    shadows, shadow_optimizer = make_weight_decay_shadows(
        alpha_map, args.lr, args.weight_decay, max_step, device
    )
    initial_logits = {
        name: parameter.detach().float().item()
        for name, parameter in alpha_map.items()
    }
    initial_alphas = {
        name: torch.sigmoid(parameter.detach().float()).item()
        for name, parameter in alpha_map.items()
    }
    ulp_thresholds = {
        name: abs(
            torch.nextafter(
                torch.tensor(value, dtype=torch.float32),
                torch.tensor(math.inf, dtype=torch.float32),
            ).item()
            - value
        )
        for name, value in initial_logits.items()
    }

    model.train()
    records = []
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        _, images, labels = next(iterator)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        shadow_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs, diagnostics = model.forward_with_diagnostics(images)
            loss = classification_loss(outputs, labels)
        loss.backward()

        before = {
            name: parameter.detach().float().item()
            for name, parameter in alpha_map.items()
        }
        gradients = {
            name: (
                parameter.grad.detach().float().item()
                if parameter.grad is not None
                else None
            )
            for name, parameter in alpha_map.items()
        }
        for shadow in shadows.values():
            shadow.grad = torch.zeros_like(shadow)

        finite = (
            torch.isfinite(loss).item()
            and all(torch.isfinite(value).all().item() for value in outputs)
            and all(
                gradient is not None and math.isfinite(gradient)
                for gradient in gradients.values()
            )
        )
        diagnostics_record = collect_cdsr_epoch_record(
            model, diagnostics, step
        )
        optimizer.step()
        shadow_optimizer.step()
        torch.cuda.synchronize(device)

        after = {
            name: parameter.detach().float().item()
            for name, parameter in alpha_map.items()
        }
        shadow_after = {
            name: parameter.detach().float().item()
            for name, parameter in shadows.items()
        }
        records.append(
            {
                "step": step,
                "loss": loss.detach().float().item(),
                "finite": bool(finite),
                "actual_group2_lr": optimizer.param_groups[2]["lr"],
                "shadow_lr": shadow_optimizer.param_groups[0]["lr"],
                "alphas": {
                    name: {
                        "task_gradient": gradients[name],
                        "logit_before": before[name],
                        "logit_after": after[name],
                        "alpha_after": torch.sigmoid(
                            torch.tensor(after[name])
                        ).item(),
                        "shadow_logit_after": shadow_after[name],
                        "task_excess_logit": after[name] - shadow_after[name],
                    }
                    for name in alpha_map
                },
                "diagnostics": diagnostics_record["stages"],
            }
        )
        print(
            f"step={step:02d}/{args.steps} loss={loss.item():.6f} "
            f"finite={bool(finite)}",
            flush=True,
        )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    final = {}
    for name, parameter in alpha_map.items():
        actual_logit = parameter.detach().float().item()
        shadow_logit = shadows[name].detach().float().item()
        task_excess = actual_logit - shadow_logit
        actual_alpha = torch.sigmoid(torch.tensor(actual_logit)).item()
        shadow_alpha = torch.sigmoid(torch.tensor(shadow_logit)).item()
        final[name] = {
            "initial_logit": initial_logits[name],
            "final_actual_logit": actual_logit,
            "final_shadow_logit": shadow_logit,
            "actual_logit_movement": actual_logit - initial_logits[name],
            "weight_decay_only_logit_movement": (
                shadow_logit - initial_logits[name]
            ),
            "task_excess_logit_movement": task_excess,
            "task_excess_relative_to_initial_logit": (
                abs(task_excess) / max(abs(initial_logits[name]), 1e-30)
            ),
            "initial_alpha": initial_alphas[name],
            "final_actual_alpha": actual_alpha,
            "final_shadow_alpha": shadow_alpha,
            "task_excess_alpha_movement": actual_alpha - shadow_alpha,
            "float32_logit_ulp_threshold": ulp_thresholds[name],
            "measurable_task_excess": abs(task_excess) >= ulp_thresholds[name],
        }

    all_finite = all(record["finite"] for record in records)
    shadow_lr_matched = all(
        record["actual_group2_lr"] == record["shadow_lr"]
        for record in records
    )
    task_excess_pass = all(
        item["measurable_task_excess"] for item in final.values()
    )
    result = {
        "protocol": {
            "dataset": args.dataset,
            "train_root": str(args.train_root.resolve()),
            "parsed_training_samples": len(dataset),
            "batch_size": args.batch_size,
            "steps": args.steps,
            "formal_epochs_for_lr_schedule": args.formal_epochs,
            "max_step": max_step,
            "seed": args.seed,
            "image_size": args.image_size,
            "precision": "bf16",
            "loss_weights": list(LOSS_WEIGHTS),
            "optimizer": "official PolyOptimizer/SGD",
            "base_lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": optimizer.param_groups[0].get("momentum", 0.0),
            "need_formula": "R * (1 - (1-D) * (1-U))",
            "weights": str(args.weights.resolve()),
            "weights_size_bytes": args.weights.stat().st_size,
            "weights_sha256": sha256_file(args.weights),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "optimizer_param_groups": optimizer_manifest(optimizer),
        "optimizer_coverage_pass": optimizer_coverage_pass,
        "pretrained_audit": pretrained_audit,
        "initial_alpha_count": len(alpha_map),
        "initial_alphas": initial_alphas,
        "records": records,
        "final_alpha_movements": final,
        "shadow_lr_matched": shadow_lr_matched,
        "all_finite": all_finite,
        "task_excess_pass": task_excess_pass,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": elapsed,
        "readiness_pass": bool(
            optimizer_coverage_pass
            and pretrained_audit_pass
            and shadow_lr_matched
            and all_finite
            and task_excess_pass
            and len(alpha_map) == 6
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "CDSR_READINESS_PASS" if result["readiness_pass"]
        else "CDSR_READINESS_FAIL",
        flush=True,
    )


if __name__ == "__main__":
    main()
