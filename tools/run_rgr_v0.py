#!/usr/bin/env python3
"""Execute the preregistered RGR-v0 parity, readiness and 3-epoch pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.resnet38_cls import Net_CAM as A0NetCAM
from network.resnet38_cls_rgr import Net as RGRNet
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_rgr_v0_paired import official_a0_validation, paired_rgr_validation


A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
A0_CHECKPOINT_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
RSBR_REQUIRED = "RSBR_V0_PILOT_REVIEW"
PARITY_PASS = "RGR_V0_PARITY_PASS"
READINESS_PASS = "RGR_V0_READINESS_PASS"
READINESS_REVIEW = "RGR_V0_READINESS_REVIEW"
READINESS_NOGO = "RGR_V0_READINESS_NOGO"
PILOT_STRONG_GO = "RGR_V0_PILOT_STRONG_GO"
PILOT_GO = "RGR_V0_PILOT_GO"
PILOT_REVIEW = "RGR_V0_PILOT_REVIEW"
PILOT_NOGO = "RGR_V0_PILOT_NOGO"
SEED = 42
N_CLASS = 4
READINESS_STEPS = 32
PILOT_EPOCHS = 3
EXPECTED_TRAIN = 23_422
EXPECTED_VAL = 3_418
BATCH_SIZE = 20
IMAGE_SIZE = 224
LR = 0.01
WEIGHT_DECAY = 0.0005
LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)
LAMBDA_REGION = 0.05
LAMBDA_RESIDUAL = 0.01
PRODUCTION_MIOU_ENVELOPE_PP = 0.01379944
PRODUCTION_DIFFERING_PIXEL_ENVELOPE = 87_808
KNOWN_REPEAT_ENVELOPE_PP = 0.01329944
GRAPH_MODULES = (
    "node_projection",
    "edge_gate",
    "value_projection",
    "message_projection",
    "isolated_head",
    "graph_head",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rsbr-pilot-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(payload), sort_keys=True) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_named_tensors(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().contiguous().cpu()
        if value.dtype == torch.bfloat16:
            value = value.view(torch.int16)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def base_state_hashes(model):
    parameters = [
        (name, value) for name, value in model.named_parameters()
        if not name.startswith("rgr.")
    ]
    buffers = [
        (name, value) for name, value in model.named_buffers()
        if not name.startswith("rgr.")
    ]
    return {
        "parameter_sha256": hash_named_tensors(parameters),
        "buffer_sha256": hash_named_tensors(buffers),
        "parameter_tensors": len(parameters),
        "buffer_tensors": len(buffers),
    }


def rgr_state_hash(model):
    return hash_named_tensors(model.rgr.state_dict().items())


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(False)
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


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def set_frozen_training_mode(model):
    model.eval()
    model.rgr.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("rgr."))


def frozen_mode_ok(model):
    return {
        "all_original_modules_eval": all(
            not module.training
            for name, module in model.named_modules()
            if name and name != "rgr" and not name.startswith("rgr.")
        ),
        "rgr_module_train": model.rgr.training,
        "all_original_parameters_frozen": all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if not name.startswith("rgr.")
        ),
        "all_rgr_parameters_trainable": all(
            parameter.requires_grad for parameter in model.rgr.parameters()
        ),
    }


def load_fresh_rgr(checkpoint):
    state = load_state(checkpoint)
    if any(key.startswith("rgr.") for key in state):
        raise RuntimeError("A0 checkpoint unexpectedly contains RGR parameters")
    model = RGRNet(n_class=N_CLASS)
    incompat = model.load_state_dict(state, strict=False)
    expected = {key for key in model.state_dict() if key.startswith("rgr.")}
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise RuntimeError({
            "missing": incompat.missing_keys,
            "unexpected": incompat.unexpected_keys,
            "expected": sorted(expected),
        })
    set_frozen_training_mode(model)
    zero = {
        "isolated_weight": torch.count_nonzero(model.rgr.isolated_head.weight).item() == 0,
        "isolated_bias": torch.count_nonzero(model.rgr.isolated_head.bias).item() == 0,
        "graph_weight": torch.count_nonzero(model.rgr.graph_head.weight).item() == 0,
        "graph_bias": torch.count_nonzero(model.rgr.graph_head.bias).item() == 0,
    }
    if not all(zero.values()):
        raise RuntimeError({"RGR output zero initialization failed": zero})
    return model, sorted(expected), zero


def load_a0(checkpoint):
    model = A0NetCAM(n_class=N_CLASS)
    incompat = model.load_state_dict(load_state(checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(incompat)
    return model


def validate_inputs(args):
    combined = " ".join((
        args.train_root,
        args.val_root,
        args.checkpoint,
        args.rsbr_pilot_summary,
        args.output_dir,
    )).lower()
    if "test" in combined or "luad" in combined:
        raise ValueError("RGR-v0 is restricted to BCSS train + validation")
    train_root, val_root = Path(args.train_root), Path(args.val_root)
    if not train_root.is_dir():
        raise FileNotFoundError(args.train_root)
    if val_root.name.lower() != "val" or not (val_root / "img").is_dir() or not (val_root / "mask").is_dir():
        raise ValueError("--val-root must point exactly to BCSS val")
    if sha256_file(args.checkpoint) != A0_CHECKPOINT_SHA256:
        raise RuntimeError("A0 checkpoint SHA256 mismatch")
    prior = json.loads(Path(args.rsbr_pilot_summary).read_text(encoding="utf-8"))
    if prior.get("decision") != RSBR_REQUIRED:
        raise RuntimeError(f"RGR locked by RSBR decision={prior.get('decision')}")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    for name in ("parity", "readiness_32b", "pilot_3ep", "checkpoints", "docs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in (
            ROOT / "network" / "rgr_v0.py",
            ROOT / "network" / "resnet38_cls_rgr.py",
            ROOT / "tool" / "infer_rgr_v0_paired.py",
            ROOT / "tools" / "run_rgr_v0.py",
        )
    }
    return prior, source_hashes


def make_loader(args):
    dataset = Stage1_TrainDataset(
        data_path=args.train_root,
        dataset="bcss",
        img_size=IMAGE_SIZE,
    )
    if len(dataset) != EXPECTED_TRAIN:
        raise RuntimeError(f"BCSS parsed train count {len(dataset)} != {EXPECTED_TRAIN}")
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader


def runtime_args(args):
    return SimpleNamespace(
        dataset="bcss", img_size=IMAGE_SIZE,
        num_workers=args.num_workers, amp_dtype="bf16",
    )


def build_optimizer(model, max_steps):
    weights, biases = [], []
    for module in model.rgr.modules():
        if isinstance(module, torch.nn.Linear):
            weights.append(module.weight)
            if module.bias is not None:
                biases.append(module.bias)
    optimizer = torchutils.PolyOptimizer(
        [
            {"params": weights, "lr": 10 * LR, "weight_decay": WEIGHT_DECAY},
            {"params": biases, "lr": 20 * LR, "weight_decay": 0.0},
        ],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        max_step=max_steps,
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    rgr_ids = {id(parameter) for parameter in model.rgr.parameters()}
    if optimizer_ids != rgr_ids:
        raise RuntimeError("Optimizer must contain exactly all RGR parameters")
    return optimizer


def region_mil_loss(region_logits, labels):
    losses = []
    for batch_index, logits in enumerate(region_logits):
        if logits.shape[0] == 0:
            continue
        score = logits.max(dim=0).values[None]
        losses.append(F.multilabel_soft_margin_loss(
            score, labels[batch_index:batch_index + 1]
        ))
    return torch.stack(losses).mean() if losses else labels.sum() * 0.0


def training_losses(model, images, labels):
    outputs = model(images, presence=labels, return_rgr_aux=True)
    out_56, out_28_1, out_28_2, out_deep = outputs[:4]
    result = outputs[-1]
    slots = (
        F.multilabel_soft_margin_loss(out_56, labels),
        F.multilabel_soft_margin_loss(out_28_1, labels),
        F.multilabel_soft_margin_loss(out_28_2, labels),
        F.multilabel_soft_margin_loss(out_deep, labels),
    )
    classification = sum(weight * loss for weight, loss in zip(LOSS_WEIGHTS, slots))
    region = region_mil_loss(result.region_logits, labels)
    residual = (result.delta_iso + result.delta_graph).abs().mean()
    total = classification + LAMBDA_REGION * region + LAMBDA_RESIDUAL * residual
    return total, classification, slots[1], region, residual, result


def grad_norm(parameters):
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.0


def module_gradients(model):
    return {
        name: grad_norm(getattr(model.rgr, name).parameters())
        for name in GRAPH_MODULES
    }


def frozen_gradients_clean(model):
    return all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for name, parameter in model.named_parameters()
        if not name.startswith("rgr.")
    )


def module_snapshots(model):
    return {
        name: {
            key: value.detach().float().cpu().clone()
            for key, value in getattr(model.rgr, name).state_dict().items()
        }
        for name in GRAPH_MODULES
    }


def module_movements(model, initial):
    result = {}
    for name in GRAPH_MODULES:
        current = getattr(model.rgr, name).state_dict()
        squared_update = squared_initial = 0.0
        for key, initial_value in initial[name].items():
            value = current[key].detach().float().cpu()
            squared_update += float((value - initial_value).square().sum().item())
            squared_initial += float(initial_value.square().sum().item())
        absolute = float(np.sqrt(squared_update))
        result[name] = {
            "absolute_update_norm": absolute,
            "relative_update_norm": absolute / (float(np.sqrt(squared_initial)) + 1e-12),
        }
    return result


def summarize_rows(rows):
    keys = sorted(
        key for key, value in rows[0].items()
        if key not in ("step", "epoch") and isinstance(value, (int, float))
    )
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "median": float(np.median([row[key] for row in rows])),
            "p95": float(np.percentile([row[key] for row in rows], 95)),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in keys
    }


def train_step(model, optimizer, images, labels, step, epoch=0):
    images = images.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        total, classification, refined_slot, region, residual, result = training_losses(
            model, images, labels
        )
    total.backward()
    gradients = module_gradients(model)
    finite = bool(
        torch.isfinite(total).item()
        and torch.isfinite(result.delta_iso).all().item()
        and torch.isfinite(result.delta_graph).all().item()
        and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.rgr.parameters()
        )
    )
    frozen_clean = frozen_gradients_clean(model)
    optimizer.step()
    finite = finite and all(
        torch.isfinite(parameter).all().item() for parameter in model.rgr.parameters()
    )
    return {
        "epoch": epoch,
        "step": step,
        "total_loss": float(total.detach().float().item()),
        "classification_loss": float(classification.detach().float().item()),
        "refined_cam28_classification_loss": float(refined_slot.detach().float().item()),
        "region_mil_loss": float(region.detach().float().item()),
        "residual_loss": float(residual.detach().float().item()),
        "finite": int(finite),
        "frozen_gradients_clean": int(frozen_clean),
        "lr_weight": float(optimizer.param_groups[0]["lr"]),
        "lr_bias": float(optimizer.param_groups[1]["lr"]),
        **{f"grad_{name}": value for name, value in gradients.items()},
        **result.statistics,
    }


def _optimizer_record(optimizer):
    return [
        {
            "lr": float(group["lr"]),
            "momentum": float(group["momentum"]),
            "weight_decay": float(group["weight_decay"]),
            "parameter_tensors": len(group["params"]),
            "parameter_elements": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]


def _checkpoint(model, optimizer, epoch, validation, args, provenance):
    return {
        "rgr_state_dict": {
            key: value.detach().cpu() for key, value in model.rgr.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation": validation,
        "a0_checkpoint": args.checkpoint,
        "a0_checkpoint_sha256": A0_CHECKPOINT_SHA256,
        "experiment_commit": args.experiment_commit,
        "provenance": provenance,
        "config": {
            "seed": SEED,
            "epochs": PILOT_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "precision": "BF16",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "loss_weights": LOSS_WEIGHTS,
            "lambda_region": LAMBDA_REGION,
            "lambda_residual": LAMBDA_RESIDUAL,
        },
    }


def save_checkpoint(model, optimizer, epoch, validation, args, output, provenance):
    path = output / "checkpoints" / f"epoch{epoch}_rgr.pth"
    torch.save(_checkpoint(model, optimizer, epoch, validation, args, provenance), path)
    return path, sha256_file(path)


def run_parity(args, output):
    seed_everything()
    rgr, missing, zero = load_fresh_rgr(args.checkpoint)
    rgr.cuda()
    initial_hash = rgr_state_hash(rgr)
    paired, ground_truth, rgr_predictions, _ = paired_rgr_validation(
        rgr,
        args.val_root,
        runtime_args(args),
        variants=("base", "full"),
        return_arrays=True,
    )
    same_process_pass = (
        paired["differing_base_full_pixels"] == 0
        and paired["full_minus_base_pp"] == 0.0
        and paired["full_minus_base_mdice_pp"] == 0.0
        and paired["prediction_sha256"]["base"] == paired["prediction_sha256"]["full"]
    )
    del rgr
    torch.cuda.empty_cache()

    seed_everything()
    a0 = load_a0(args.checkpoint)
    a0.cuda()
    a0.eval()
    a0_summary, a0_ground_truth, a0_predictions = official_a0_validation(
        a0, args.val_root, runtime_args(args), return_arrays=True
    )
    if not np.array_equal(ground_truth, a0_ground_truth):
        raise RuntimeError("Parity ground-truth ordering mismatch")
    independent_delta_pp = 100.0 * (
        paired["final_metrics"]["full"]["mIoU"] - a0_summary["metrics"]["mIoU"]
    )
    independent_differing_pixels = int(
        np.count_nonzero(rgr_predictions["full"] != a0_predictions)
    )
    independent_pass = (
        abs(independent_delta_pp) <= PRODUCTION_MIOU_ENVELOPE_PP
        and independent_differing_pixels <= PRODUCTION_DIFFERING_PIXEL_ENVELOPE
    )
    decision = PARITY_PASS if same_process_pass and independent_pass else "RGR_V0_PARITY_NOGO"
    result = {
        "decision": decision,
        "same_process_exact": same_process_pass,
        "same_process": paired,
        "independent_a0": a0_summary,
        "independent_delta_miou_pp": independent_delta_pp,
        "independent_differing_pixels": independent_differing_pixels,
        "allowed_miou_envelope_pp": PRODUCTION_MIOU_ENVELOPE_PP,
        "allowed_differing_pixels": PRODUCTION_DIFFERING_PIXEL_ENVELOPE,
        "independent_envelope_pass": independent_pass,
        "rgr_initialization_sha256": initial_hash,
        "zero_initialization": zero,
        "missing_keys_expected_rgr_only": missing,
    }
    write_json(output / "parity" / "summary.json", result)
    del a0
    torch.cuda.empty_cache()
    print(decision, json.dumps({
        "independent_delta_miou_pp": independent_delta_pp,
        "independent_differing_pixels": independent_differing_pixels,
    }), flush=True)
    return result


def _deterministic_structure_check(model, images, labels):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        base = model.forward_cam_base(images)
        first = model.refine_from_base(base, labels, collect_structures=True)
        second = model.refine_from_base(base, labels, collect_structures=True)
    return {
        "structures_equal": first.structures == second.structures,
        "region_counts_equal": first.per_image_region_counts == second.per_image_region_counts,
        "zero_identity": bool(
            torch.equal(first.refined_cam, base[1])
            and torch.count_nonzero(first.delta_iso).item() == 0
            and torch.count_nonzero(first.delta_graph).item() == 0
        ),
    }


def run_readiness(args, output, parity):
    seed_everything()
    dataset, loader = make_loader(args)
    model, missing, zero = load_fresh_rgr(args.checkpoint)
    model.cuda()
    frozen_initial = base_state_hashes(model)
    initial_modules = module_snapshots(model)
    optimizer = build_optimizer(model, READINESS_STEPS)
    iterator = iter(loader)
    first = next(iterator)
    first_images = first[1].cuda(non_blocking=True)
    first_labels = first[2].cuda(non_blocking=True)
    deterministic = _deterministic_structure_check(model, first_images, first_labels)
    set_frozen_training_mode(model)
    rows = []
    started = time.time()
    for step in range(1, READINESS_STEPS + 1):
        _, images, labels = first if step == 1 else next(iterator)
        row = train_step(model, optimizer, images, labels, step)
        rows.append(row)
        append_jsonl(output / "readiness_32b" / "steps.jsonl", row)
        print("RGR_READINESS_STEP", json.dumps(row, sort_keys=True), flush=True)
    movements = module_movements(model, initial_modules)
    frozen_final = base_state_hashes(model)
    summary_stats = summarize_rows(rows)
    step1_heads = rows[0]["grad_isolated_head"] > 0 and rows[0]["grad_graph_head"] > 0
    upstream = ("node_projection", "edge_gate", "value_projection", "message_projection")
    upstream_by8 = {
        name: max(row[f"grad_{name}"] for row in rows[:8]) > 0.0 for name in upstream
    }
    upstream_by32 = {
        name: max(row[f"grad_{name}"] for row in rows) > 0.0 for name in upstream
    }
    measurable_movement = {
        name: movements[name]["absolute_update_norm"] > 1e-12 for name in GRAPH_MODULES
    }
    all_finite = all(row["finite"] == 1 and row["frozen_gradients_clean"] == 1 for row in rows)
    frozen_unchanged = frozen_initial == frozen_final
    message_nonzero = any(
        row["multi_node_fraction"] > 0 and row["message_norm"] > 0 for row in rows
    )
    graph_residual_active = rows[-1]["rms_delta_graph"] > 0
    residual_ratio = rows[-1]["residual_ratio"]
    failures = []
    if parity["decision"] != PARITY_PASS:
        failures.append("PARITY_NOT_PASS")
    if not all_finite:
        failures.append("NONFINITE_OR_FROZEN_GRADIENT_VIOLATION")
    if not step1_heads:
        failures.append("OUTPUT_HEAD_INACTIVE_STEP1")
    if not all(upstream_by32.values()) or not all(
        measurable_movement[name] for name in upstream
    ):
        failures.append("GRAPH_PATH_INACTIVE")
    if not graph_residual_active:
        failures.append("GRAPH_RESIDUAL_INACTIVE")
    if not message_nonzero:
        failures.append("MESSAGE_PASSING_ZERO")
    if not frozen_unchanged:
        failures.append("FROZEN_BASE_STATE_VIOLATION")
    if not all(deterministic.values()):
        failures.append("REGION_EXTRACTION_OR_IDENTITY_FAILURE")
    if residual_ratio > 1.0:
        failures.append("RESIDUAL_EXPLOSION")
    if failures:
        decision = READINESS_NOGO
    elif residual_ratio > 0.5 or not all(upstream_by8.values()):
        decision = READINESS_REVIEW
    else:
        decision = READINESS_PASS
    base_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if not name.startswith("rgr.")
    )
    rgr_parameters = model.rgr.trainable_parameter_count()
    result = {
        "decision": decision,
        "failures": failures,
        "steps": READINESS_STEPS,
        "parsed_train_samples": len(dataset),
        "runtime_seconds": time.time() - started,
        "optimizer_groups": _optimizer_record(optimizer),
        "step1_heads_active": step1_heads,
        "upstream_active_by_step8": upstream_by8,
        "upstream_active_by_step32": upstream_by32,
        "module_movements": movements,
        "measurable_module_movement": measurable_movement,
        "graph_residual_active": graph_residual_active,
        "message_nonzero_on_multinode": message_nonzero,
        "deterministic_checks": deterministic,
        "all_finite": all_finite,
        "frozen_base_unchanged": frozen_unchanged,
        "frozen_initial": frozen_initial,
        "frozen_final": frozen_final,
        "final_residual_ratio": residual_ratio,
        "final_graph_to_isolated_rms": rows[-1]["graph_to_isolated_rms"],
        "summary_statistics": summary_stats,
        "parameter_profile": {
            "base_parameters": base_parameters,
            "rgr_parameters": rgr_parameters,
            "overhead_percent": 100.0 * rgr_parameters / base_parameters,
            "under_one_percent": 100.0 * rgr_parameters / base_parameters < 1.0,
        },
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "missing_keys_expected_rgr_only": missing,
        "zero_initialization": zero,
    }
    write_json(output / "readiness_32b" / "summary.json", result)
    print(decision, json.dumps({
        "failures": failures,
        "residual_ratio": residual_ratio,
        "upstream_by8": upstream_by8,
    }, sort_keys=True), flush=True)
    del model
    torch.cuda.empty_cache()
    return result


def pilot_decision(records, safety):
    best = max(records, key=lambda row: (row["full_minus_base_pp"], -row["epoch"]))
    best_graph_increment = max(row["full_minus_isolated_pp"] for row in records)
    final = records[-1]
    positive_classes = sum(value > 0 for value in best["class_delta_pp"].values())
    nonnegative_classes = sum(value >= 0 for value in best["class_delta_pp"].values())
    all_graph_nonpositive = all(row["full_minus_isolated_pp"] <= 0 for row in records)
    if safety["failures"] or best["full_minus_base_pp"] < 0.05 or all_graph_nonpositive or final["full_minus_base_pp"] < 0:
        decision = PILOT_NOGO
    elif (
        best["full_minus_base_pp"] >= 0.30
        and final["full_minus_base_pp"] >= 0.20
        and best_graph_increment >= 0.10
        and nonnegative_classes >= 3
    ):
        decision = PILOT_STRONG_GO
    elif (
        best["full_minus_base_pp"] >= 0.15
        and final["full_minus_base_pp"] >= 0
        and best_graph_increment >= 0.05
        and positive_classes >= 2
    ):
        decision = PILOT_GO
    else:
        decision = PILOT_REVIEW
    return {
        "decision": decision,
        "best_epoch": best["epoch"],
        "best_full_minus_base_pp": best["full_minus_base_pp"],
        "best_full_minus_isolated_pp": best_graph_increment,
        "epoch3_full_minus_base_pp": final["full_minus_base_pp"],
        "positive_classes_at_best": positive_classes,
        "nonnegative_classes_at_best": nonnegative_classes,
    }


def _validation_record(epoch, validation, train_record):
    return {
        "epoch": epoch,
        "base_metrics": validation["final_metrics"]["base"],
        "isolated_metrics": validation["final_metrics"]["isolated"],
        "graph_only_metrics": validation["final_metrics"]["graph_only"],
        "full_metrics": validation["final_metrics"]["full"],
        "full_minus_base_pp": validation["full_minus_base_pp"],
        "isolated_minus_base_pp": validation["isolated_minus_base_pp"],
        "graph_only_minus_base_pp": validation["graph_only_minus_base_pp"],
        "full_minus_isolated_pp": validation["full_minus_isolated_pp"],
        "delta_mdice_pp": validation["full_minus_base_mdice_pp"],
        "class_delta_pp": validation["paired_class_iou_delta_pp"],
        "isolated_class_delta_pp": validation["isolated_class_iou_delta_pp"],
        "taxonomy": validation["taxonomy"],
        "node_count_stratified": validation["node_count_stratified"],
        "mechanism": validation["mechanism"],
        "runtime": validation["runtime"],
        "training": train_record,
    }


def run_pilot(args, output, parity, readiness, provenance):
    seed_everything()
    dataset, loader = make_loader(args)
    steps_per_epoch = len(loader)
    model, missing, zero = load_fresh_rgr(args.checkpoint)
    model.cuda()
    frozen_initial = base_state_hashes(model)
    optimizer = build_optimizer(model, steps_per_epoch * PILOT_EPOCHS)
    initial_rgr_hash = rgr_state_hash(model)
    epoch0 = paired_rgr_validation(
        model, args.val_root, runtime_args(args), variants=("base", "isolated", "graph_only", "full")
    )
    identity = (
        epoch0["differing_base_full_pixels"] == 0
        and epoch0["differing_isolated_full_pixels"] == 0
        and epoch0["full_minus_base_pp"] == 0.0
        and epoch0["full_minus_isolated_pp"] == 0.0
    )
    write_json(output / "pilot_3ep" / "validation_epoch0.json", epoch0)
    if not identity:
        result = {
            "decision": PILOT_NOGO,
            "failures": ["RGR_V0_PILOT_INIT_IDENTITY_NOGO"],
            "epoch0": epoch0,
        }
        write_json(output / "pilot_3ep" / "summary.json", result)
        return result

    records, validations, training_epochs = [], [], []
    checkpoint_hashes = {}
    global_step = 0
    all_finite = True
    frozen_unchanged = True
    both_paths_active = True
    residual_explosion = False
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for epoch in range(1, PILOT_EPOCHS + 1):
        set_frozen_training_mode(model)
        rows = []
        tick = time.time()
        for _, images, labels in loader:
            global_step += 1
            row = train_step(model, optimizer, images, labels, global_step, epoch=epoch)
            rows.append(row)
            append_jsonl(output / "pilot_3ep" / "train" / f"epoch{epoch}_steps.jsonl", row)
            all_finite = all_finite and row["finite"] == 1 and row["frozen_gradients_clean"] == 1
            if global_step % 100 == 0:
                print("RGR_PILOT_TRAIN", json.dumps(row, sort_keys=True), flush=True)
        train_summary = summarize_rows(rows)
        train_record = {
            "epoch": epoch,
            "steps": len(rows),
            "seconds": time.time() - tick,
            "statistics": train_summary,
        }
        training_epochs.append(train_record)
        write_json(output / "pilot_3ep" / "train" / f"epoch{epoch}_summary.json", train_record)
        both_paths_active = both_paths_active and (
            train_summary["grad_isolated_head"]["max"] > 0
            and train_summary["grad_graph_head"]["max"] > 0
            and train_summary["grad_edge_gate"]["max"] > 0
        )
        residual_explosion = residual_explosion or (
            train_summary["residual_ratio"]["max"] > 1.0
            or train_summary["max_abs_delta_iso"]["max"] > 100.0
            or train_summary["max_abs_delta_graph"]["max"] > 100.0
        )
        frozen_before = base_state_hashes(model)
        validation = paired_rgr_validation(
            model, args.val_root, runtime_args(args), variants=("base", "isolated", "graph_only", "full")
        )
        frozen_after = base_state_hashes(model)
        frozen_unchanged = frozen_unchanged and frozen_before == frozen_initial and frozen_after == frozen_initial
        validations.append(validation)
        write_json(output / "pilot_3ep" / f"validation_epoch{epoch}.json", validation)
        record = _validation_record(epoch, validation, train_record)
        records.append(record)
        checkpoint, digest = save_checkpoint(
            model, optimizer, epoch, validation, args, output, provenance
        )
        checkpoint_hashes[checkpoint.name] = digest
        print("RGR_PILOT_EPOCH", json.dumps({
            "epoch": epoch,
            "full_minus_base_pp": validation["full_minus_base_pp"],
            "full_minus_isolated_pp": validation["full_minus_isolated_pp"],
        }), flush=True)

    validation_finite = all(
        np.isfinite(row["full_minus_base_pp"])
        and np.isfinite(row["full_minus_isolated_pp"])
        and all(np.isfinite(value) for value in row["class_delta_pp"].values())
        for row in records
    )
    failures = []
    if not all_finite:
        failures.append("NONFINITE_OR_FROZEN_GRADIENT_VIOLATION")
    if not frozen_unchanged:
        failures.append("FROZEN_BASE_STATE_VIOLATION")
    if not both_paths_active:
        failures.append("GRAPH_PATH_INACTIVE")
    if residual_explosion:
        failures.append("RESIDUAL_EXPLOSION")
    if not validation_finite:
        failures.append("VALIDATION_NONFINITE")
    safety = {
        "failures": failures,
        "all_finite": all_finite,
        "frozen_base_unchanged": frozen_unchanged,
        "both_paths_active": both_paths_active,
        "residual_explosion": residual_explosion,
        "validation_finite": validation_finite,
        "frozen_initial": frozen_initial,
        "frozen_final": base_state_hashes(model),
    }
    decision = pilot_decision(records, safety)
    flags = []
    best_record = next(row for row in records if row["epoch"] == decision["best_epoch"])
    multi = best_record["node_count_stratified"].get("N>=2_all", {})
    if (
        best_record["full_minus_isolated_pp"] > 0
        and multi.get("full_minus_isolated_pp", 0) > 0
    ):
        flags.append("REGION_RELATIONAL_SIGNAL")
    if max(row["runtime"]["rgr_overhead_vs_base_forward_percent"] for row in records) > 50.0:
        flags.append("RGR_RUNTIME_REVIEW")
    result = {
        **decision,
        "secondary_flags": flags,
        "fresh_restart_from_a0": True,
        "initial_rgr_sha256": initial_rgr_hash,
        "zero_initialization": zero,
        "missing_keys_expected_rgr_only": missing,
        "parsed_train_samples": len(dataset),
        "steps_per_epoch": steps_per_epoch,
        "epochs_trained": PILOT_EPOCHS,
        "total_optimizer_updates": global_step,
        "epoch0_identity": epoch0,
        "epochs": records,
        "safety": safety,
        "checkpoint_sha256s": checkpoint_hashes,
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "test_accessed": False,
        "luad_accessed": False,
        "auto_25epoch": False,
    }
    write_json(output / "pilot_3ep" / "summary.json", result)
    print(result["decision"], json.dumps({
        "best_epoch": result["best_epoch"],
        "best_full_minus_base_pp": result["best_full_minus_base_pp"],
        "best_full_minus_isolated_pp": result["best_full_minus_isolated_pp"],
        "secondary_flags": flags,
        "failures": failures,
    }, sort_keys=True), flush=True)
    del model
    torch.cuda.empty_cache()
    return result


def write_delivery_report(output, contract, parity, readiness, pilot):
    executive = mechanism = classes = graph_rows = node_rows = training_rows = ""
    if pilot and "epochs" in pilot:
        for row in pilot["epochs"]:
            executive += (
                f"| {row['epoch']} | {100 * row['base_metrics']['mIoU']:.4f} | "
                f"{100 * row['isolated_metrics']['mIoU']:.4f} | "
                f"{100 * row['graph_only_metrics']['mIoU']:.4f} | "
                f"{100 * row['full_metrics']['mIoU']:.4f} | "
                f"{row['full_minus_base_pp']:+.4f} | {row['full_minus_isolated_pp']:+.4f} |\n"
            )
            for variant, deltas in (
                ("Isolated", row["isolated_class_delta_pp"]),
                ("Full", row["class_delta_pp"]),
            ):
                classes += (
                    f"| {row['epoch']} | {variant} | {deltas['0']:+.4f} | "
                    f"{deltas['1']:+.4f} | {deltas['2']:+.4f} | {deltas['3']:+.4f} |\n"
                )
            graph = row["mechanism"]
            graph_rows += (
                f"| {row['epoch']} | {graph['touch_gate_mean']:.6f} | "
                f"{graph['nontouch_gate_mean']:.6f} | {graph['same_class_gate_mean']:.6f} | "
                f"{graph['different_class_gate_mean']:.6f} | {graph['rms_delta_graph']:.6f} |\n"
            )
            b = row["taxonomy"]["B_misclassified_pure"]["recovery_pixels"]["full"]
            d = row["taxonomy"]["D_mixed_boundary"]["recovery_pixels"]["full"]
            mechanism += f"| {row['epoch']} | {b:+,} | {d:+,} | {graph['graph_to_isolated_rms']:.6f} |\n"
            stats = row["training"]["statistics"]
            training_rows += (
                f"| {row['epoch']} | {row['training']['seconds']:.2f} | "
                f"{row['runtime']['validation_seconds']:.2f} | {stats['total_loss']['mean']:.6f} | "
                f"{stats['grad_isolated_head']['mean']:.6e} | {stats['grad_graph_head']['mean']:.6e} |\n"
            )
        final_nodes = pilot["epochs"][-1]["node_count_stratified"]
        for group, values in final_nodes.items():
            node_rows += (
                f"| {group} | {values.get('image_count', 0)} | "
                f"{values.get('full_minus_base_pp', float('nan')):+.4f} | "
                f"{values.get('full_minus_isolated_pp', float('nan')):+.4f} |\n"
            )

    final_decision = (
        pilot.get("decision") if pilot
        else readiness.get("decision") if readiness
        else parity.get("decision")
    )
    text = f"""# RGR-v0 Minimal Region Graph Reasoning Delivery

## 1. Executive conclusion

Final decision: `{final_decision}`

The run followed the gated sequence: zero-init parity, fresh 32-batch
readiness, and only after readiness PASS a fresh three-epoch frozen-SSHR pilot.
No RSBR trained weight, dense label, test split, LUAD split, other seed, or
25-epoch continuation was used.

## 2. Frozen implementation

- Base commit: `{A0_COMMIT}`
- Experiment commit: `{contract['experiment_commit']}`
- A0 checkpoint SHA256: `{A0_CHECKPOINT_SHA256}`
- Seed / batch / image / precision: 42 / 20 / 224 / BF16
- Loss coefficients: region=0.05, residual=0.01
- RGR parameters: {readiness.get('parameter_profile', {}).get('rgr_parameters', 'not run')}
- Parameter overhead: {readiness.get('parameter_profile', {}).get('overhead_percent', float('nan')):.6f}%
- Source hashes: `{json.dumps(contract['source_hashes'], sort_keys=True)}`

## 3. Stage -1 parity

- Decision: `{parity['decision']}`
- Same-process exact: {parity['same_process_exact']}
- Independent mIoU delta: {parity['independent_delta_miou_pp']:+.8f} pp
- Independent differing pixels: {parity['independent_differing_pixels']:,}
- Corrected envelope pass: {parity['independent_envelope_pass']}

## 4. Stage 0 readiness

- Decision: `{readiness.get('decision', 'not run')}`
- Failures: {readiness.get('failures', 'not run')}
- Step-1 isolated/graph heads active: {readiness.get('step1_heads_active', 'not run')}
- Upstream active by step 8: {readiness.get('upstream_active_by_step8', 'not run')}
- Upstream active by step 32: {readiness.get('upstream_active_by_step32', 'not run')}
- Final residual ratio: {readiness.get('final_residual_ratio', float('nan')):.8f}
- Graph/isolated residual RMS: {readiness.get('final_graph_to_isolated_rms', float('nan')):.8f}
- Frozen SSHR unchanged: {readiness.get('frozen_base_unchanged', 'not run')}
- All finite: {readiness.get('all_finite', 'not run')}

## 5. Three-epoch paired validation

| Epoch | Base | Isolated | Graph-only | Full | Full-Base | Full-Isolated |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | {100 * pilot.get('epoch0_identity', {}).get('final_metrics', {}).get('base', {}).get('mIoU', 0):.4f} | same | same | same | 0 | 0 |
{executive}

## 6. Per-class changes

| Epoch | Variant | C0 delta | C1 delta | C2 delta | C3 delta |
|---:|---|---:|---:|---:|---:|
{classes}

## 7. Graph diagnostics

| Epoch | Touch gate | Non-touch gate | Same-class gate | Diff-class gate | Graph RMS |
|---:|---:|---:|---:|---:|---:|
{graph_rows}

| Epoch | Type-B recovery | Type-D recovery | Graph/isolated RMS |
|---:|---:|---:|---:|
{mechanism}

## 8. Node-count stratification at epoch 3

| Region count | Images | Full-Base | Full-Isolated |
|---|---:|---:|---:|
{node_rows}

## 9. Training and resources

| Epoch | Train seconds | Validation seconds | Mean loss | Isolated grad | Graph grad |
|---:|---:|---:|---:|---:|---:|
{training_rows}

- Pilot peak CUDA memory: {pilot.get('peak_cuda_memory_bytes', 0) / 2**30:.3f} GiB
- Parameter overhead below 1%: {readiness.get('parameter_profile', {}).get('under_one_percent', 'not run')}
- Runtime review: {'RGR_RUNTIME_REVIEW' in pilot.get('secondary_flags', []) if pilot else 'not run'}

## 10. Required scientific answers

1. Full improves A0: {pilot.get('best_full_minus_base_pp', float('nan')) > 0 if pilot else 'not run'}.
2. Full improves isolated correction: {pilot.get('best_full_minus_isolated_pp', float('nan')) > 0 if pilot else 'not run'}.
3. Graph-only positive at any epoch: {any(row['graph_only_minus_base_pp'] > 0 for row in pilot.get('epochs', [])) if pilot else 'not run'}.
4. Maximum graph increment: {pilot.get('best_full_minus_isolated_pp', float('nan')):+.4f} pp.
5. Multi-node relational signal: {'REGION_RELATIONAL_SIGNAL' in pilot.get('secondary_flags', []) if pilot else 'not run'}.
6. Edge gates are reported descriptively above; no gate statistic was tuned.
7. Per-class beneficiaries are reported above.
8. Type-B recovery at epoch 3: {pilot.get('epochs', [{}])[-1].get('taxonomy', {}).get('B_misclassified_pure', {}).get('recovery_pixels', {}).get('full', 'not run') if pilot else 'not run'} pixels.
9. Graph context reduces isolated errors: {pilot.get('best_full_minus_isolated_pp', float('nan')) > 0 if pilot else 'not run'}.
10. Parameter/runtime overhead are reported above.
11. A 25-epoch experiment is not automatically authorized; scientific review is required.
12. Transition-aware graph edges are not authorized by this experiment.

## 11. Commands and artifacts

```bash
{contract['command']}
```

- Parity: `parity/summary.json`
- Readiness: `readiness_32b/summary.json`
- Pilot: `pilot_3ep/summary.json`
- Checkpoints: `checkpoints/epoch1_rgr.pth` through `epoch3_rgr.pth`

## 12. STOP boundary

Execution stops after this report regardless of decision. No test, LUAD,
additional seed, 25-epoch run, deeper GNN, transition head, GAT, Transformer,
prototype, topology change, edge-feature change, or tuning is performed.
"""
    (output / "docs" / "rgr_v0_delivery.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    prior, source_hashes = validate_inputs(args)
    output = Path(args.output_dir)
    contract = {
        "a0_commit": A0_COMMIT,
        "experiment_commit": args.experiment_commit,
        "a0_checkpoint": args.checkpoint,
        "a0_checkpoint_sha256": A0_CHECKPOINT_SHA256,
        "rsbr_prerequisite": prior["decision"],
        "source_hashes": source_hashes,
        "seed": SEED,
        "readiness_steps": READINESS_STEPS,
        "pilot_epochs": PILOT_EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "precision": "BF16",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss_weights": LOSS_WEIGHTS,
        "lambda_region": LAMBDA_REGION,
        "lambda_residual": LAMBDA_RESIDUAL,
        "test_forbidden": True,
        "luad_forbidden": True,
        "auto_25epoch_forbidden": True,
        "command": " ".join(sys.argv),
    }
    write_json(output / "contract.json", contract)
    parity = run_parity(args, output)
    readiness = {}
    pilot = {}
    if parity["decision"] == PARITY_PASS:
        readiness = run_readiness(args, output, parity)
    if readiness.get("decision") == READINESS_PASS:
        provenance = {
            "a0_commit": A0_COMMIT,
            "experiment_commit": args.experiment_commit,
            "checkpoint_sha256": A0_CHECKPOINT_SHA256,
            "parity_decision": parity["decision"],
            "readiness_decision": readiness["decision"],
            "source_hashes": source_hashes,
        }
        pilot = run_pilot(args, output, parity, readiness, provenance)
    write_delivery_report(output, contract, parity, readiness, pilot)
    summary = {
        "final_decision": pilot.get("decision", readiness.get("decision", parity["decision"])),
        "parity": parity["decision"],
        "readiness": readiness.get("decision", "NOT_RUN"),
        "pilot": pilot.get("decision", "NOT_RUN"),
        "test_accessed": False,
        "luad_accessed": False,
        "auto_25epoch": False,
        "report": "docs/rgr_v0_delivery.md",
    }
    write_json(output / "summary.json", summary)
    print("RGR_V0_FINAL", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
