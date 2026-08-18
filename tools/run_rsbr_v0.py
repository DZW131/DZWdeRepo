#!/usr/bin/env python3
"""Execute the preregistered RSBR-v0 parity, readiness, and pilot gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.resnet38_cls import Net_CAM as A0NetCAM
from network.resnet38_cls_rsbr import Net as RSBRNet
from network.resnet38_cls_rsbr import Net_CAM as RSBRNetCAM
from tool import infer_fun, iouutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_rsbr_v0 import infer_rsbr_validation


A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
SEED = 42
N_CLASS = 4
EXPECTED_TRAIN = 23422
EXPECTED_VAL = 3418
LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)
LAMBDA_REGION = 0.05
LAMBDA_RESIDUAL = 0.01


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("all", "parity", "readiness", "pilot"), default="all")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--amp-dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wt-dec", type=float, default=5e-4)
    parser.add_argument("--audit-commit", default="WORKTREE")
    return parser.parse_args()


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def validate_scope(args):
    if args.batch_size != 20 or args.img_size != 224 or args.amp_dtype != "bf16":
        raise ValueError("RSBR-v0 is frozen to batch20, 224x224, and BF16")
    combined = " ".join((args.train_root, args.val_root, args.output_dir)).lower()
    if "luad" in combined or "test" in combined:
        raise ValueError("RSBR-v0 is BCSS train/validation only; test and LUAD are forbidden")
    train_root, val_root = Path(args.train_root), Path(args.val_root)
    if val_root.name.lower() != "val":
        raise ValueError("--val-root must point exactly to BCSS val")
    if not train_root.is_dir() or not (val_root / "img").is_dir() or not (val_root / "mask").is_dir():
        raise FileNotFoundError("Expected BCSS training and val/{img,mask}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
    for name in ("parity", "readiness_32b", "pilot_3ep", "docs", "config"):
        (output / name).mkdir(parents=True, exist_ok=True)


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_a0_model(checkpoint):
    model = A0NetCAM(n_class=N_CLASS)
    model.load_state_dict(load_state(checkpoint), strict=True)
    model.eval()
    return model


def load_rsbr_model(checkpoint, inference=False):
    model = RSBRNetCAM(n_class=N_CLASS) if inference else RSBRNet(n_class=N_CLASS)
    incompat = model.load_state_dict(load_state(checkpoint), strict=False)
    expected = {name for name in model.state_dict() if name.startswith("rsbr.")}
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise RuntimeError({
            "missing": incompat.missing_keys,
            "unexpected": incompat.unexpected_keys,
            "expected_rsbr": sorted(expected),
        })
    return model, sorted(expected)


def runtime_args(args):
    return SimpleNamespace(
        dataset="bcss", img_size=args.img_size,
        num_workers=args.num_workers, amp_dtype=args.amp_dtype,
    )


def metric_flat(metrics):
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f"{key}.{inner_key}"] = float(inner_value)
        elif np.isscalar(value):
            flat[key] = float(value)
    return flat


def official_inference_capture(model, val_root, args):
    captured = {}
    original = infer_fun.iouutils.scores

    def capture(ground_truth, predictions, n_class):
        captured["ground_truth"] = np.stack([item.copy() for item in ground_truth])
        captured["predictions"] = np.stack([item.copy() for item in predictions])
        return original(ground_truth, predictions, n_class)

    infer_fun.iouutils.scores = capture
    try:
        metrics = infer_fun.infer(
            model, str(val_root), N_CLASS, runtime_args(args),
            thr=None, cam_weights=(0.6, 0.2, 0.2),
        )
    finally:
        infer_fun.iouutils.scores = original
    if metrics is None or not captured:
        raise RuntimeError("Released validation inference failed")
    return metrics, captured


def stage_parity(args, output):
    print("RSBR_STAGE_MINUS1_START", flush=True)
    seed_everything()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    a0 = load_a0_model(args.checkpoint).cuda()
    official_started = time.time()
    released_metrics, captured = official_inference_capture(a0, args.val_root, args)
    official_runtime = time.time() - official_started
    del a0
    torch.cuda.empty_cache()
    rsbr, missing = load_rsbr_model(args.checkpoint, inference=True)
    rsbr.eval(); rsbr.cuda()
    rsbr_started = time.time()
    result = infer_rsbr_validation(rsbr, args.val_root, runtime_args(args))
    rsbr_runtime = time.time() - rsbr_started

    metric_left, metric_right = metric_flat(released_metrics), metric_flat(result.metrics)
    metric_difference = max(abs(metric_left[key] - metric_right[key]) for key in metric_left)
    differing_official = int(np.count_nonzero(captured["predictions"] != result.predictions))
    differing_pre_post = int(np.count_nonzero(result.base_predictions != result.predictions))
    cam_ok = all(value == 0.0 for value in result.maximum_cam_differences.values())
    decision = "RSBR_V0_PARITY_PASS" if (
        cam_ok and differing_official == 0 and differing_pre_post == 0
        and metric_difference < 1e-7
    ) else "RSBR_V0_PARITY_NOGO"
    payload = {
        "decision": decision,
        "a0_commit": A0_COMMIT,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "missing_keys_expected_rsbr_only": missing,
        "released_metrics": released_metrics,
        "rsbr_zero_init_metrics": result.metrics,
        "standalone_cam28_1_metrics": result.standalone_metrics,
        "maximum_cam_differences": result.maximum_cam_differences,
        "differing_prediction_pixels_vs_released": differing_official,
        "differing_prediction_pixels_pre_vs_post": differing_pre_post,
        "maximum_metric_difference": metric_difference,
        "runtime_seconds": time.time() - started,
        "official_inference_runtime_seconds": official_runtime,
        "rsbr_inference_runtime_seconds": rsbr_runtime,
        "rsbr_inference_overhead_percent": 100.0 * (rsbr_runtime / official_runtime - 1.0),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "command": " ".join(sys.argv),
    }
    write_json(output / "parity" / "summary.json", payload)
    print(decision, json.dumps(json_ready(payload)), flush=True)
    if decision != "RSBR_V0_PARITY_PASS":
        raise RuntimeError(decision)
    return payload


def make_train_loader(args):
    dataset = Stage1_TrainDataset(
        data_path=args.train_root,
        transform=transforms.Compose([transforms.ToTensor()]),
        dataset="bcss", img_size=args.img_size,
    )
    if len(dataset) != EXPECTED_TRAIN:
        raise RuntimeError(f"BCSS parsed train count {len(dataset)} != {EXPECTED_TRAIN}")
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader


def freeze_a0(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("rsbr."))
    model.train()
    for name, parameter in model.named_parameters():
        if not name.startswith("rsbr."):
            parameter.requires_grad_(False)


def rsbr_optimizer(model, args, max_step):
    weights, biases = [], []
    for module in model.rsbr.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            weights.append(module.weight)
            if module.bias is not None:
                biases.append(module.bias)
    params = [
        {"params": weights, "lr": 10 * args.lr, "weight_decay": args.wt_dec},
        {"params": biases, "lr": 20 * args.lr, "weight_decay": 0.0},
    ]
    return torchutils.PolyOptimizer(
        params, lr=args.lr, weight_decay=args.wt_dec, max_step=max_step
    )


def region_mil_loss(region_logits, labels):
    losses = []
    for batch_index, logits in enumerate(region_logits):
        if logits.shape[0] == 0:
            continue
        region_score = logits.max(dim=0).values[None]
        losses.append(F.multilabel_soft_margin_loss(region_score, labels[batch_index:batch_index + 1]))
    return torch.stack(losses).mean() if losses else labels.sum() * 0.0


def training_loss(model, images, labels):
    outputs = model(images, presence=labels, return_rsbr_aux=True)
    out_56, out_28_1, out_28_2, out_deep = outputs[:4]
    result = outputs[-1]
    classification = (
        LOSS_WEIGHTS[0] * F.multilabel_soft_margin_loss(out_56, labels)
        + LOSS_WEIGHTS[1] * F.multilabel_soft_margin_loss(out_28_1, labels)
        + LOSS_WEIGHTS[2] * F.multilabel_soft_margin_loss(out_28_2, labels)
        + LOSS_WEIGHTS[3] * F.multilabel_soft_margin_loss(out_deep, labels)
    )
    region = region_mil_loss(result.region_logits, labels)
    residual = result.delta_core.abs().mean() + result.delta_transition.abs().mean()
    total = classification + LAMBDA_REGION * region + LAMBDA_RESIDUAL * residual
    return total, classification, region, residual, result


def grad_norm(parameters):
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().square().sum())
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


def parameter_movement(module, initial):
    numerator, denominator = 0.0, 0.0
    for name, parameter in module.named_parameters():
        numerator += float((parameter.detach().float().cpu() - initial[name]).square().sum())
        denominator += float(initial[name].square().sum())
    absolute = float(np.sqrt(numerator))
    relative = float(absolute / (np.sqrt(denominator) + 1e-12))
    return absolute, relative


def snapshot(module):
    return {name: parameter.detach().float().cpu().clone() for name, parameter in module.named_parameters()}


def finite_gradients(module):
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in module.parameters()
    )


def stage_readiness(args, output, parity_payload):
    if parity_payload["decision"] != "RSBR_V0_PARITY_PASS":
        raise RuntimeError("Pilot gate requires parity PASS")
    print("RSBR_STAGE0_32B_START", flush=True)
    seed_everything()
    _, loader = make_train_loader(args)
    model, missing = load_rsbr_model(args.checkpoint, inference=False)
    freeze_a0(model)
    model.cuda()
    optimizer = rsbr_optimizer(model, args, max_step=32)
    initial_region = snapshot(model.rsbr.region_semantic_head)
    initial_transition = snapshot(model.rsbr.transition_head)
    rows, structure_reference = [], None
    all_component_counts = []
    baseline_gradient_clean = True
    all_finite = True
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    for step, (_, images, labels) in enumerate(loader, start=1):
        if step > 32:
            break
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, classification, region, residual, result = training_loss(model, images, labels)
        total.backward()
        region_gradient = grad_norm(model.rsbr.region_semantic_head.parameters())
        transition_gradient = grad_norm(model.rsbr.transition_head.parameters())
        all_finite = (
            all_finite
            and bool(torch.isfinite(total).item())
            and bool(torch.isfinite(result.delta_core).all().item())
            and bool(torch.isfinite(result.delta_transition).all().item())
            and finite_gradients(model.rsbr)
            and all(np.isfinite(value) for value in result.statistics.values())
        )
        for name, parameter in model.named_parameters():
            if name.startswith("rsbr.") or parameter.grad is None:
                continue
            baseline_gradient_clean = baseline_gradient_clean and bool(
                torch.count_nonzero(parameter.grad).item() == 0
            )
        optimizer.step()
        row = {
            "step": step,
            "loss_total": float(total.detach().float()),
            "loss_classification": float(classification.detach().float()),
            "loss_region_mil": float(region.detach().float()),
            "loss_residual": float(residual.detach().float()),
            "region_head_grad_norm": region_gradient,
            "transition_head_grad_norm": transition_gradient,
            **result.statistics,
        }
        rows.append(row)
        all_component_counts.extend(result.per_image_component_counts or [])
        print("RSBR_READINESS_STEP", json.dumps(row, sort_keys=True), flush=True)

        if step == 1:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base = model._base_features_and_logits(images[:1], apply_deep_dropout=False)
                first = model.refine_from_base(base, labels[:1], collect_structures=True)
                second = model.refine_from_base(base, labels[:1], collect_structures=True)
            structure_reference = {
                "identical": first.structures == second.structures,
                "component_count_identical": sum(map(len, first.structures)) == sum(map(len, second.structures)),
            }

    region_movement_abs, region_movement = parameter_movement(
        model.rsbr.region_semantic_head, initial_region
    )
    transition_movement_abs, transition_movement = parameter_movement(
        model.rsbr.transition_head, initial_transition
    )
    last_eight = rows[-8:]
    transition_mean = float(np.mean([row["transition_fraction"] for row in last_eight]))
    residual_ratio = rows[-1]["residual_ratio"]
    mask_degenerate = transition_mean > 0.80 or transition_mean < 0.05
    residual_explosion = residual_ratio > 1.00 or max(
        rows[-1]["max_abs_delta_core"], rows[-1]["max_abs_delta_transition"]
    ) > 100.0
    residual_review = residual_ratio > 0.50
    region_active = any(row["region_head_grad_norm"] > 0 for row in rows)
    transition_active = any(row["transition_head_grad_norm"] > 0 for row in rows)
    deterministic = bool(structure_reference and all(structure_reference.values()))

    if not all_finite or residual_explosion or not region_active or not transition_active \
            or region_movement_abs <= 0 or transition_movement_abs <= 0 or not baseline_gradient_clean \
            or not deterministic:
        decision = "RSBR_V0_READINESS_NOGO"
    elif mask_degenerate or residual_review:
        decision = "RSBR_V0_READINESS_REVIEW"
    else:
        decision = "RSBR_V0_READINESS_PASS"
    payload = {
        "decision": decision,
        "steps": 32,
        "fresh_restart_from_a0": True,
        "missing_keys_expected_rsbr_only": missing,
        "all_finite": all_finite,
        "region_head_active": region_active,
        "transition_head_active": transition_active,
        "region_head_relative_movement": region_movement,
        "transition_head_relative_movement": transition_movement,
        "region_head_l2_movement": region_movement_abs,
        "transition_head_l2_movement": transition_movement_abs,
        "sshr_gradients_none_or_zero": baseline_gradient_clean,
        "deterministic_extraction": structure_reference,
        "last8_transition_fraction_mean": transition_mean,
        "mean_regions_per_image": float(np.mean(all_component_counts)),
        "median_regions_per_image": float(np.median(all_component_counts)),
        "transition_mask_degenerate": mask_degenerate,
        "final_residual_ratio": residual_ratio,
        "residual_review": residual_review,
        "residual_explosion": residual_explosion,
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_groups": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
        "rows": rows,
    }
    write_json(output / "readiness_32b" / "summary.json", payload)
    write_readiness_report(output, parity_payload, payload)
    print(decision, json.dumps(json_ready(payload)), flush=True)
    return payload


def write_readiness_report(output, parity, readiness):
    text = f"""# RSBR-v0 Readiness Report

**Decision: {readiness['decision']}**

## Stage -1 parity

- Decision: `{parity['decision']}`
- Differing prediction pixels: {parity['differing_prediction_pixels_vs_released']}
- Maximum metric difference: {parity['maximum_metric_difference']:.3e}
- Official validation runtime: {parity['official_inference_runtime_seconds']:.2f} s
- RSBR zero-init validation runtime: {parity['rsbr_inference_runtime_seconds']:.2f} s
- Inference overhead: {parity['rsbr_inference_overhead_percent']:.2f}%

## Stage 0 — 32 real BCSS batches

- Finite: {readiness['all_finite']}
- Region head active / L2 movement: {readiness['region_head_active']} / {readiness['region_head_l2_movement']:.6e}
- Transition head active / L2 movement: {readiness['transition_head_active']} / {readiness['transition_head_l2_movement']:.6e}
- A0 gradients None or zero: {readiness['sshr_gradients_none_or_zero']}
- Deterministic extraction: {readiness['deterministic_extraction']}
- Last-8 mean transition fraction: {readiness['last8_transition_fraction_mean']:.6f}
- Mean / median regions per image: {readiness['mean_regions_per_image']:.3f} / {readiness['median_regions_per_image']:.3f}
- Final residual ratio: {readiness['final_residual_ratio']:.6f}
- Runtime: {readiness['runtime_seconds']:.2f} s
- Peak CUDA memory: {readiness['peak_cuda_memory_bytes'] / 2**30:.3f} GiB

The three-epoch pilot is unlocked only by `RSBR_V0_READINESS_PASS`.
"""
    (output / "docs" / "rsbr_v0_readiness_report.md").write_text(text, encoding="utf-8")


def official_metrics(gt, prediction):
    return iouutils.scores(
        [item.copy() for item in gt], [item.copy() for item in prediction], n_class=4
    )


def region_error_diagnostics(gt, base_prediction, refined_prediction):
    kernel = np.ones((3, 3), np.uint8)
    del kernel  # taxonomy uses components only; transition geometry is reported elsewhere.
    totals = {
        "B_misclassified_pure": {"a0_wrong_pixels": 0, "refined_wrong_pixels": 0},
        "D_mixed_boundary": {"a0_wrong_pixels": 0, "refined_wrong_pixels": 0},
    }
    for truth, base, refined in zip(gt, base_prediction, refined_prediction):
        for predicted_class in range(4):
            count, components = cv2.connectedComponents(
                np.asarray(base == predicted_class, dtype=np.uint8), connectivity=8
            )
            for component_id in range(1, count):
                mask = components == component_id
                area = int(mask.sum())
                counts = np.bincount(truth[mask].astype(np.int64), minlength=5)
                majority = int(np.argmax(counts))
                purity = float(counts[majority] / max(area, 1))
                if purity >= 0.80 and majority not in (predicted_class, 4):
                    taxonomy = "B_misclassified_pure"
                elif purity < 0.80:
                    taxonomy = "D_mixed_boundary"
                else:
                    continue
                totals[taxonomy]["a0_wrong_pixels"] += int(np.count_nonzero(base[mask] != truth[mask]))
                totals[taxonomy]["refined_wrong_pixels"] += int(
                    np.count_nonzero(refined[mask] != truth[mask])
                )
    for values in totals.values():
        values["wrong_pixel_change"] = values["refined_wrong_pixels"] - values["a0_wrong_pixels"]
        values["relative_change"] = values["wrong_pixel_change"] / max(values["a0_wrong_pixels"], 1)
    return totals


def metric_record(metrics):
    return {
        "mIoU": float(metrics["Mean IoU"]),
        "mDice": float(metrics["Mean Dice"]),
        "class_iou": {str(key): float(value) for key, value in metrics["Class IoU"].items()},
        "class_dice": {
            str(key): float(value) for key, value in metrics["Dice Coefficients"].items()
        },
    }


def write_pilot_report(output, args, parity, readiness, payload):
    lines = [
        "# RSBR-v0 Delivery",
        "",
        f"**Decision: {payload['decision']}**",
        "",
        "## Frozen control",
        "",
        f"- A0 source: `{A0_COMMIT}` (`baseline/official-a0`).",
        f"- A0 checkpoint SHA256: `{parity['checkpoint_sha256']}`.",
        "- Dataset/split: BCSS training + validation only; image-level labels only.",
        "- Backbone, HFRM, original heads, augmentation, BF16, loss weights, optimizer recipe, TTA, thresholds, fusion and metric remain released A0.",
        "- Trainable modules: zero-initialized 512->4 region residual head and 1541->128->4 transition residual head.",
        "- Frozen values: minimum area 2, internal deviation top 25%, lambda_region=0.05, lambda_residual=0.01.",
        "",
        "## Validation evidence",
        "",
        f"- Stage -1: `{parity['decision']}`; differing pixels={parity['differing_prediction_pixels_vs_released']}; maximum metric difference={parity['maximum_metric_difference']:.3e}.",
        f"- Stage 0: `{readiness['decision']}`; 32 real batches; region movement={readiness['region_head_relative_movement']:.6e}; transition movement={readiness['transition_head_relative_movement']:.6e}.",
        "",
        "## Three-epoch validation results",
        "",
        "| Epoch | mIoU | Delta mIoU (pp) | mDice | CAM28_1 standalone mIoU | Type-D wrong-pixel change |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["epochs"]:
        lines.append(
            f"| {record['epoch']} | {record['validation']['mIoU'] * 100:.4f} | "
            f"{record['delta_miou_pp']:+.4f} | {record['validation']['mDice'] * 100:.4f} | "
            f"{record['standalone_cam28_1']['mIoU'] * 100:.4f} | "
            f"{record['region_error']['D_mixed_boundary']['wrong_pixel_change']:+d} |"
        )
    lines.extend([
        "",
        "## Mechanism and resources",
        "",
        f"- RSBR parameters: {payload['resource_profile']['rsbr_parameters']:,} "
        f"({payload['resource_profile']['additional_parameter_percent']:.4f}% of A0).",
        f"- Pilot runtime: {payload['runtime_seconds']:.1f} s; peak CUDA allocated memory: {payload['peak_cuda_memory_bytes'] / 2**30:.3f} GiB.",
        "- Detailed region/core/transition/residual/MIL statistics are in `pilot_3ep/summary.json`.",
        "",
        "## Exact command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
        "## Scope stop",
        "",
        "No test evaluation, LUAD run, 25-epoch run, graph, transformer, prototype, pseudo-label or decoder expansion was performed.",
    ])
    (output / "docs" / "rsbr_v0_delivery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_pilot(args, output, parity_payload, readiness_payload):
    if readiness_payload["decision"] != "RSBR_V0_READINESS_PASS":
        raise RuntimeError("Three-epoch pilot is locked unless readiness is PASS")
    print("RSBR_STAGE1_3EP_START", flush=True)
    seed_everything()
    dataset, loader = make_train_loader(args)
    model, missing = load_rsbr_model(args.checkpoint, inference=False)
    freeze_a0(model)
    model.cuda()
    max_step = (len(dataset) // args.batch_size) * 3
    optimizer = rsbr_optimizer(model, args, max_step=max_step)
    a0_metrics = parity_payload["released_metrics"]
    a0_miou = float(a0_metrics["Mean IoU"])
    a0_class_iou = {str(key): float(value) for key, value in a0_metrics["Class IoU"].items()}
    epoch_records = []
    transition_alive = True
    residual_stable = True
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    for epoch in range(1, 4):
        model.train()
        for name, parameter in model.named_parameters():
            if not name.startswith("rsbr."):
                parameter.requires_grad_(False)
        loss_rows, statistic_rows = [], []
        transition_epoch_active = False
        for _, images, labels in loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total, classification, region, residual, result = training_loss(model, images, labels)
            total.backward()
            transition_epoch_active = transition_epoch_active or grad_norm(
                model.rsbr.transition_head.parameters()
            ) > 0
            optimizer.step()
            loss_rows.append((
                float(total.detach().float()), float(classification.detach().float()),
                float(region.detach().float()), float(residual.detach().float()),
            ))
            statistic_rows.append(result.statistics)
        transition_alive = transition_alive and transition_epoch_active
        model.eval()
        evaluation = infer_rsbr_validation(model, args.val_root, runtime_args(args))
        diagnosis = region_error_diagnostics(
            evaluation.ground_truth, evaluation.base_predictions, evaluation.predictions
        )
        validation = metric_record(evaluation.metrics)
        standalone = metric_record(evaluation.standalone_metrics)
        average_stats = {
            key: float(np.mean([row[key] for row in statistic_rows]))
            for key in statistic_rows[0]
        }
        residual_stable = residual_stable and average_stats["residual_ratio"] <= 1.0
        record = {
            "epoch": epoch,
            "validation": validation,
            "standalone_cam28_1": standalone,
            "delta_miou_pp": 100.0 * (validation["mIoU"] - a0_miou),
            "class_iou_delta_pp": {
                key: 100.0 * (validation["class_iou"][key] - a0_class_iou[key])
                for key in a0_class_iou
            },
            "loss_total": float(np.mean([row[0] for row in loss_rows])),
            "loss_refined_classification": float(np.mean([row[1] for row in loss_rows])),
            "loss_region_mil": float(np.mean([row[2] for row in loss_rows])),
            "loss_residual": float(np.mean([row[3] for row in loss_rows])),
            "mechanism": average_stats,
            "validation_mechanism": evaluation.diagnostics,
            "region_error": diagnosis,
        }
        epoch_records.append(record)
        torch.save(
            {key: value.detach().cpu() for key, value in model.rsbr.state_dict().items()},
            output / "pilot_3ep" / f"rsbr_epoch_{epoch:02d}.pth",
        )
        print("RSBR_PILOT_EPOCH", json.dumps(json_ready(record)), flush=True)

    best = max(epoch_records, key=lambda item: item["validation"]["mIoU"])
    final = epoch_records[-1]
    positive_classes = sum(delta > 0 for delta in best["class_iou_delta_pp"].values())
    nonnegative_classes = sum(delta >= 0 for delta in best["class_iou_delta_pp"].values())
    typed_decreased = best["region_error"]["D_mixed_boundary"]["wrong_pixel_change"] < 0
    if (
        best["delta_miou_pp"] >= 0.30 and final["delta_miou_pp"] >= 0.20
        and nonnegative_classes >= 3 and residual_stable and typed_decreased
    ):
        decision = "RSBR_V0_PILOT_STRONG_GO"
    elif (
        best["delta_miou_pp"] >= 0.15 and final["delta_miou_pp"] >= 0
        and positive_classes >= 2 and residual_stable and typed_decreased
    ):
        decision = "RSBR_V0_PILOT_GO"
    elif (
        best["delta_miou_pp"] >= 0.05 and final["delta_miou_pp"] >= 0
        and residual_stable and transition_alive and typed_decreased
    ):
        decision = "RSBR_V0_PILOT_REVIEW"
    else:
        decision = "RSBR_V0_PILOT_NOGO"

    a0_parameter_count = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if not name.startswith("rsbr.")
    )
    rsbr_parameter_count = model.rsbr.trainable_parameter_count()
    payload = {
        "decision": decision,
        "fresh_restart_from_a0": True,
        "epochs_trained": 3,
        "missing_keys_expected_rsbr_only": missing,
        "a0_validation": metric_record(a0_metrics),
        "best_epoch": best["epoch"],
        "best_delta_miou_pp": best["delta_miou_pp"],
        "final_delta_miou_pp": final["delta_miou_pp"],
        "positive_classes_at_best": positive_classes,
        "nonnegative_classes_at_best": nonnegative_classes,
        "type_d_error_decreased_at_best": typed_decreased,
        "transition_head_active_every_epoch": transition_alive,
        "residual_stable": residual_stable,
        "epochs": epoch_records,
        "resource_profile": {
            "a0_parameters": a0_parameter_count,
            "rsbr_parameters": rsbr_parameter_count,
            "additional_parameter_percent": 100.0 * rsbr_parameter_count / a0_parameter_count,
        },
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "command": " ".join(sys.argv),
    }
    write_json(output / "pilot_3ep" / "summary.json", payload)
    write_pilot_report(output, args, parity_payload, readiness_payload, payload)
    print(decision, json.dumps(json_ready(payload)), flush=True)
    return payload


def main():
    args = parse_args()
    validate_scope(args)
    output = Path(args.output_dir)
    config = {
        "a0_commit": A0_COMMIT,
        "audit_commit": args.audit_commit,
        "seed": SEED,
        "dataset": "BCSS",
        "train_root": args.train_root,
        "val_root": args.val_root,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "batch_size": args.batch_size,
        "image_size": args.img_size,
        "precision": args.amp_dtype,
        "readiness_batches": 32,
        "pilot_epochs": 3,
        "loss_weights": LOSS_WEIGHTS,
        "lambda_region": LAMBDA_REGION,
        "lambda_residual": LAMBDA_RESIDUAL,
        "test_forbidden": True,
        "command": " ".join(sys.argv),
    }
    write_json(output / "config" / "frozen_contract.json", config)

    parity_path = output / "parity" / "summary.json"
    readiness_path = output / "readiness_32b" / "summary.json"
    parity = stage_parity(args, output) if args.stage in ("all", "parity") else json.loads(parity_path.read_text())
    if args.stage == "parity":
        return
    readiness = stage_readiness(args, output, parity) if args.stage in ("all", "readiness") else json.loads(readiness_path.read_text())
    if args.stage == "readiness":
        return
    if readiness["decision"] != "RSBR_V0_READINESS_PASS":
        print(f"RSBR_STOP_AFTER_{readiness['decision']}", flush=True)
        return
    stage_pilot(args, output, parity, readiness)


if __name__ == "__main__":
    main()
