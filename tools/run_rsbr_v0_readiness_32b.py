#!/usr/bin/env python3
"""Run the gated, fresh-restart 32-batch RSBR-v0 readiness audit."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.resnet38_cls_rsbr import Net as RSBRNet
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset
from tools.rsbr_parity_r1_contract import MIOU_ALLOWANCE_PP, PARITY_PASS, PIXEL_ALLOWANCE


SEED = 42
EXPECTED_TRAIN = 23_422
N_CLASS = 4
AUDIT_STEPS = (0, 1, 2, 4, 8, 16, 24, 32)
LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)
LAMBDA_REGION = 0.05
LAMBDA_RESIDUAL = 0.01
READINESS_PASS = "RSBR_V0_READINESS_PASS"
READINESS_REVIEW = "RSBR_V0_READINESS_REVIEW"
READINESS_NOGO = "RSBR_V0_READINESS_NOGO"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wt-dec", type=float, default=5e-4)
    return parser.parse_args()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


@contextmanager
def preserve_rng_state():
    """Keep observation-only audit forwards from perturbing the update stream."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_states)


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_fresh_model(checkpoint):
    model = RSBRNet(n_class=N_CLASS)
    incompat = model.load_state_dict(load_state(checkpoint), strict=False)
    expected = {key for key in model.state_dict() if key.startswith("rsbr.")}
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise RuntimeError({
            "missing": incompat.missing_keys,
            "unexpected": incompat.unexpected_keys,
            "expected": sorted(expected),
        })
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("rsbr."))
    return model, sorted(expected)


def make_loader(args):
    dataset = Stage1_TrainDataset(
        data_path=args.train_root,
        transform=transforms.Compose([transforms.ToTensor()]),
        dataset="bcss",
        img_size=args.img_size,
    )
    if len(dataset) != EXPECTED_TRAIN:
        raise RuntimeError(f"BCSS parsed train count {len(dataset)} != {EXPECTED_TRAIN}")
    generator = torch.Generator().manual_seed(SEED)
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
    return dataset, loader


def build_optimizer(model, args):
    weights, biases = [], []
    for module in model.rsbr.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            weights.append(module.weight)
            if module.bias is not None:
                biases.append(module.bias)
    groups = [
        {"params": weights, "lr": 10 * args.lr, "weight_decay": args.wt_dec},
        {"params": biases, "lr": 20 * args.lr, "weight_decay": 0.0},
    ]
    optimizer = torchutils.PolyOptimizer(
        groups, lr=args.lr, weight_decay=args.wt_dec, max_step=32
    )
    rsbr_ids = {id(parameter) for parameter in model.rsbr.parameters()}
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if optimizer_ids != rsbr_ids:
        raise RuntimeError("Optimizer coverage is not exactly the RSBR module")
    return optimizer


def region_mil_loss(region_logits, labels):
    losses = []
    for batch_index, logits in enumerate(region_logits):
        if logits.shape[0] == 0:
            continue
        region_score = logits.max(dim=0).values[None]
        losses.append(F.multilabel_soft_margin_loss(
            region_score, labels[batch_index:batch_index + 1]
        ))
    return torch.stack(losses).mean() if losses else labels.sum() * 0.0


def forward_losses(model, images, labels):
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


def tensors_grad_norm(gradients):
    squares = [item.detach().float().square().sum() for item in gradients if item is not None]
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


def parameters_grad_norm(parameters):
    return tensors_grad_norm([parameter.grad for parameter in parameters])


def module_snapshot(module):
    return {name: value.detach().float().cpu().clone() for name, value in module.named_parameters()}


def movement(module, initial):
    numerator = 0.0
    denominator = 0.0
    for name, parameter in module.named_parameters():
        reference = initial[name]
        difference = parameter.detach().float().cpu() - reference
        numerator += float(difference.square().sum())
        denominator += float(reference.square().sum())
    absolute = float(np.sqrt(numerator))
    return absolute, float(absolute / (np.sqrt(denominator) + 1e-12))


def baseline_parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith("rsbr."):
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def baseline_gradients_clean(model):
    return all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for name, parameter in model.named_parameters()
        if not name.startswith("rsbr.")
    )


def module_gradients_finite(module):
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in module.parameters()
    )


def region_logit_stats(result):
    valid = [item.detach().float() for item in result.region_logits if item.numel()]
    if not valid:
        return {
            "region_valid_images": 0,
            "region_valid_image_fraction": 0.0,
            "region_logits_mean": 0.0,
            "region_logits_std": 0.0,
            "region_logits_max": 0.0,
            "region_logits_min": 0.0,
        }
    values = torch.cat([item.flatten() for item in valid])
    return {
        "region_valid_images": len(valid),
        "region_valid_image_fraction": len(valid) / float(len(result.region_logits)),
        "region_logits_mean": float(values.mean().item()),
        "region_logits_std": float(values.std(unbiased=False).item()),
        "region_logits_max": float(values.max().item()),
        "region_logits_min": float(values.min().item()),
    }


def deterministic_structure_check(model, images, labels):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        base = model._base_features_and_logits(images[:1], apply_deep_dropout=False)
        first = model.refine_from_base(base, labels[:1], collect_structures=True)
        second = model.refine_from_base(base, labels[:1], collect_structures=True)
    return {
        "structures_identical": first.structures == second.structures,
        "component_counts_identical": first.per_image_component_counts == second.per_image_component_counts,
    }


def audit_snapshot(model, images, labels, step, initial_region, initial_transition):
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        total, classification, region, residual, result = forward_losses(model, images, labels)
    region_parameters = list(model.rsbr.region_semantic_head.parameters())
    transition_parameters = list(model.rsbr.transition_head.parameters())
    region_task_gradients = torch.autograd.grad(
        region, region_parameters, retain_graph=True, allow_unused=True
    )
    transition_task_gradients = torch.autograd.grad(
        classification, transition_parameters, retain_graph=True, allow_unused=True
    )
    total.backward()
    region_total_grad = parameters_grad_norm(region_parameters)
    transition_total_grad = parameters_grad_norm(transition_parameters)
    region_absolute, region_relative = movement(
        model.rsbr.region_semantic_head, initial_region
    )
    transition_absolute, transition_relative = movement(
        model.rsbr.transition_head, initial_transition
    )
    row = {
        "step": step,
        "loss_total": float(total.detach().float().item()),
        "loss_classification": float(classification.detach().float().item()),
        "loss_region_mil": float(region.detach().float().item()),
        "loss_residual": float(residual.detach().float().item()),
        "region_grad_from_region_loss": tensors_grad_norm(region_task_gradients),
        "transition_grad_from_refined_classification": tensors_grad_norm(
            transition_task_gradients
        ),
        "region_total_grad": region_total_grad,
        "transition_total_grad": transition_total_grad,
        "transition_to_region_grad_ratio": transition_total_grad / (region_total_grad + 1e-30),
        "region_parameter_movement_absolute": region_absolute,
        "region_parameter_movement_relative": region_relative,
        "transition_parameter_movement_absolute": transition_absolute,
        "transition_parameter_movement_relative": transition_relative,
        "finite": bool(
            torch.isfinite(total).item()
            and torch.isfinite(result.delta_core).all().item()
            and torch.isfinite(result.delta_transition).all().item()
            and module_gradients_finite(model.rsbr)
        ),
        "frozen_sshr_gradients_clean": baseline_gradients_clean(model),
        **result.statistics,
        **region_logit_stats(result),
    }
    model.zero_grad(set_to_none=True)
    return row, list(result.per_image_component_counts or [])


def run_update(model, optimizer, images, labels):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        total, classification, region, residual, result = forward_losses(model, images, labels)
    total.backward()
    row = {
        "loss_total": float(total.detach().float().item()),
        "loss_classification": float(classification.detach().float().item()),
        "loss_region_mil": float(region.detach().float().item()),
        "loss_residual": float(residual.detach().float().item()),
        "region_total_grad": parameters_grad_norm(
            model.rsbr.region_semantic_head.parameters()
        ),
        "transition_total_grad": parameters_grad_norm(
            model.rsbr.transition_head.parameters()
        ),
        "finite": bool(
            torch.isfinite(total).item()
            and torch.isfinite(result.delta_core).all().item()
            and torch.isfinite(result.delta_transition).all().item()
            and module_gradients_finite(model.rsbr)
        ),
        "frozen_sshr_gradients_clean": baseline_gradients_clean(model),
        **result.statistics,
        **region_logit_stats(result),
        "component_counts": list(result.per_image_component_counts or []),
    }
    optimizer.step()
    return row


def decide(rows, all_updates, structure, baseline_unchanged):
    end = rows[-1]
    last_eight = all_updates[-8:]
    transition_mean = float(np.mean([item["transition_fraction"] for item in last_eight]))
    rapid_residual_growth = end["residual_ratio"] > max(
        5.0 * rows[4]["residual_ratio"], 0.50
    )
    hard_failures = []
    reviews = []
    if not all(item["finite"] for item in rows + all_updates):
        hard_failures.append("non-finite loss/output/gradient")
    if not baseline_unchanged or not all(
        item["frozen_sshr_gradients_clean"] for item in rows + all_updates
    ):
        hard_failures.append("frozen SSHR changed or received nonzero gradient")
    if not all(structure.values()):
        hard_failures.append("region extraction was not deterministic")
    if end["region_grad_from_region_loss"] <= 0 or end["region_total_grad"] <= 0:
        hard_failures.append("region head inactive at step 32")
    if end["transition_grad_from_refined_classification"] <= 0 or end["transition_total_grad"] <= 0:
        hard_failures.append("transition head inactive at step 32")
    if end["region_parameter_movement_absolute"] <= 0:
        hard_failures.append("region head did not update")
    if end["transition_parameter_movement_absolute"] <= 0:
        hard_failures.append("transition head did not update")
    if end["residual_ratio"] > 1.0 or max(
        end["max_abs_delta_core"], end["max_abs_delta_transition"]
    ) > 100.0:
        hard_failures.append("residual/logit explosion")
    if 0.50 < end["residual_ratio"] <= 1.0:
        reviews.append("end residual ratio is in the 0.50-1.00 review band")
    if transition_mean > 0.80 or transition_mean < 0.05:
        reviews.append("transition fraction is persistently outside [0.05, 0.80]")
    if end["transition_to_region_grad_ratio"] < 0.01:
        reviews.append("step-32 transition/region gradient ratio is below 0.01")
    if end["transition_to_region_grad_ratio"] > 100 and rapid_residual_growth:
        reviews.append("transition/region gradient ratio exceeds 100 with rapid residual growth")
    if end["region_valid_image_fraction"] < 0.50:
        reviews.append("fewer than half of images produce valid region MIL tokens")
    if end["region_logits_std"] == 0.0:
        reviews.append("region logits are constant at step 32")
    if hard_failures:
        return READINESS_NOGO, hard_failures, reviews, transition_mean, rapid_residual_growth
    if reviews:
        return READINESS_REVIEW, hard_failures, reviews, transition_mean, rapid_residual_growth
    return READINESS_PASS, hard_failures, reviews, transition_mean, rapid_residual_growth


def validate_args(args):
    if args.batch_size != 20 or args.img_size != 224:
        raise ValueError("Frozen readiness protocol requires batch20 and 224x224")
    combined = " ".join((args.train_root, args.checkpoint, args.output_dir)).lower()
    if "test" in combined or "luad" in combined:
        raise ValueError("Readiness is BCSS training split only")
    if not Path(args.train_root).is_dir() or not Path(args.checkpoint).is_file():
        raise FileNotFoundError("Training root or checkpoint is missing")
    output = Path(args.output_dir)
    parity_path = output / "parity_r1" / "summary.json"
    if not parity_path.is_file():
        raise FileNotFoundError("Parity R1 summary is required")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity.get("decision") != PARITY_PASS:
        raise RuntimeError(f"Readiness locked by {parity.get('decision')}")
    readiness = output / "readiness_32b"
    if readiness.exists() and any(readiness.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {readiness}")
    readiness.mkdir(parents=True, exist_ok=True)
    (output / "docs").mkdir(parents=True, exist_ok=True)
    return parity


def write_combined_report(output, parity, summary):
    audit_rows = []
    for row in summary["audit_steps"]:
        audit_rows.append(
            f"| {row['step']} | {row['loss_total']:.6f} | {row['loss_region_mil']:.6f} | "
            f"{row['region_grad_from_region_loss']:.3e} | "
            f"{row['transition_grad_from_refined_classification']:.3e} | "
            f"{row['region_total_grad']:.3e} | {row['transition_total_grad']:.3e} | "
            f"{row['transition_to_region_grad_ratio']:.4f} | {row['residual_ratio']:.6f} |"
        )
    end = summary["audit_steps"][-1]
    production = parity["production_comparison"]
    text = f"""# RSBR-v0 Parity R1 and 32-Batch Readiness Delivery

## 1. Executive conclusion

- Corrected parity: **{parity['decision']}**
- Stage-0 readiness: **{summary['decision']}**
- 3-epoch pilot started: **false**
- Test/LUAD accessed: **false / false**

## 2. Corrected two-layer parity

Layer 1 used the fixed 32-image BCSS validation subset in one RSBR process:

- Maximum CAM difference: {parity['same_process_identity']['maximum_cam_difference']:.3e}
- Delta-core exact zero: {parity['same_process_identity']['delta_core_exact_zero']}
- Delta-transition exact zero: {parity['same_process_identity']['delta_transition_exact_zero']}
- Base/refined differing pixels: {parity['same_process_identity']['differing_prediction_pixels']}

Layer 2 used independent production BF16 A0 and RSBR-zero validation runs:

- mIoU difference: {production['mIoU_difference_pp']:.8f} pp / allowance {MIOU_ALLOWANCE_PP:.8f} pp
- Differing pixels: {production['differing_prediction_pixels']:,} / allowance {PIXEL_ALLOWANCE:,}
- Production flags unchanged: {production['production_flags_unchanged']}

## 3. Frozen Stage-0 control

- Dataset / seed: BCSS train / 42
- Parsed samples: {summary['parsed_train_samples']:,}
- Real batches / batch size / image size: 32 / 20 / 224
- Precision: BF16; production cuDNN benchmark and TF32 enabled
- Fresh A0 checkpoint SHA256: `{summary['checkpoint_sha256']}`
- Loss: `L_SSHR_refined28_1 + 0.05 L_region + 0.01 L_res`
- SSHR parameters frozen and absent from the update groups: {summary['frozen_sshr_parameters_unchanged']}
- RSBR model source hashes unchanged: {parity['model_source_hashes_unchanged']}

## 4. Connectivity and dynamics

| Step | Total loss | Region MIL | Region grad from L_region | Transition grad from refined cls | Region total grad | Transition total grad | T/R grad | Residual ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(audit_rows)}

At step 32:

- Region movement absolute / relative: {end['region_parameter_movement_absolute']:.6e} / {end['region_parameter_movement_relative']:.6e}
- Transition movement absolute / relative: {end['transition_parameter_movement_absolute']:.6e} / {end['transition_parameter_movement_relative']:.6e}
- Region valid images: {end['region_valid_images']}/{summary['batch_size']}
- Region logit mean/std/min/max: {end['region_logits_mean']:.6f} / {end['region_logits_std']:.6f} / {end['region_logits_min']:.6f} / {end['region_logits_max']:.6f}
- Mean region token norm: {end['mean_region_token_norm']:.6f}

## 5. Region and mask statistics

- Mean / median regions per image: {summary['region_statistics']['mean_regions_per_image']:.4f} / {summary['region_statistics']['median_regions_per_image']:.4f}
- Mean valid-token regions per image: {summary['region_statistics']['mean_valid_token_regions_per_image']:.4f}
- Mean core / transition fraction: {summary['region_statistics']['mean_core_fraction']:.6f} / {summary['region_statistics']['mean_transition_fraction']:.6f}
- Last-8 transition fraction: {summary['transition_fraction_last8_mean']:.6f}
- Mean tiny / no-region fraction: {summary['region_statistics']['mean_tiny_region_fraction']:.6f} / {summary['region_statistics']['mean_no_region_fraction']:.6f}
- Deterministic extraction repeat: {summary['deterministic_structure_check']}

## 6. Residual and safety

- End residual ratio: {end['residual_ratio']:.6f}
- End core / transition RMS: {end['rms_delta_core']:.6f} / {end['rms_delta_transition']:.6f}
- End maximum absolute core / transition delta: {end['max_abs_delta_core']:.6f} / {end['max_abs_delta_transition']:.6f}
- All finite: {summary['all_finite']}
- Peak CUDA memory: {summary['peak_cuda_memory_bytes'] / 2**30:.3f} GiB
- Runtime: {summary['runtime_seconds']:.2f} s

## 7. Decision evidence

- Hard failures: {summary['hard_failures'] or 'none'}
- Review triggers: {summary['review_triggers'] or 'none'}
- Rapid residual growth: {summary['rapid_residual_growth']}

Final decision: **{summary['decision']}**.

The protocol requires a stop after this report even when readiness passes.
No 3-epoch pilot or formal experiment was launched.

## 8. Exact commands

Parity:

```bash
{parity['command']}
```

Readiness:

```bash
{summary['command']}
```
"""
    (output / "docs" / "rsbr_v0_parity_r1_and_readiness_delivery.md").write_text(
        text, encoding="utf-8"
    )


def main():
    args = parse_args()
    parity = validate_args(args)
    output = Path(args.output_dir)
    readiness_dir = output / "readiness_32b"
    seed_everything()
    dataset, loader = make_loader(args)
    model, missing = load_fresh_model(args.checkpoint)
    model.cuda()
    optimizer = build_optimizer(model, args)
    optimizer_record = [{
        "lr": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
        "momentum": float(group["momentum"]),
        "parameter_tensors": len(group["params"]),
        "parameter_elements": sum(parameter.numel() for parameter in group["params"]),
    } for group in optimizer.param_groups]
    initial_region = module_snapshot(model.rsbr.region_semantic_head)
    initial_transition = module_snapshot(model.rsbr.transition_head)
    baseline_hash_before = baseline_parameter_hash(model)
    audit_rows = []
    update_rows = []
    all_component_counts = []
    structure = None
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    for batch_index, (_, images, labels) in enumerate(loader, start=1):
        if batch_index > 32:
            break
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        if batch_index == 1:
            structure = deterministic_structure_check(model, images, labels)
            with preserve_rng_state():
                row, counts = audit_snapshot(
                    model, images, labels, 0, initial_region, initial_transition
                )
            audit_rows.append(row)
            print("RSBR_READINESS_AUDIT", json.dumps(row, sort_keys=True), flush=True)
        update = run_update(model, optimizer, images, labels)
        update["step"] = batch_index
        update_rows.append(update)
        all_component_counts.extend(update.pop("component_counts"))
        print("RSBR_READINESS_UPDATE", json.dumps(update, sort_keys=True), flush=True)
        if batch_index in AUDIT_STEPS[1:]:
            with preserve_rng_state():
                row, _ = audit_snapshot(
                    model, images, labels, batch_index, initial_region, initial_transition
                )
            audit_rows.append(row)
            print("RSBR_READINESS_AUDIT", json.dumps(row, sort_keys=True), flush=True)

    if len(update_rows) != 32 or [row["step"] for row in audit_rows] != list(AUDIT_STEPS):
        raise RuntimeError("Did not execute the exact 32-update/audit schedule")
    baseline_hash_after = baseline_parameter_hash(model)
    baseline_unchanged = baseline_hash_before == baseline_hash_after
    decision, hard_failures, reviews, transition_last8, rapid_growth = decide(
        audit_rows, update_rows, structure, baseline_unchanged
    )
    region_statistics = {
        "mean_regions_per_image": float(np.mean(all_component_counts)),
        "median_regions_per_image": float(np.median(all_component_counts)),
        "mean_valid_token_regions_per_image": float(np.mean([
            row["semantic_regions_per_image"] for row in update_rows
        ])),
        "mean_core_fraction": float(np.mean([row["core_fraction"] for row in update_rows])),
        "mean_transition_fraction": float(np.mean([
            row["transition_fraction"] for row in update_rows
        ])),
        "mean_tiny_region_fraction": float(np.mean([
            row["tiny_region_fraction"] for row in update_rows
        ])),
        "mean_no_region_fraction": float(np.mean([
            row["no_region_fraction"] for row in update_rows
        ])),
    }
    summary = {
        "decision": decision,
        "parity_decision": parity["decision"],
        "audit_commit": args.audit_commit,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "parsed_train_samples": len(dataset),
        "seed": SEED,
        "batch_size": args.batch_size,
        "image_size": args.img_size,
        "precision": "BF16",
        "updates": len(update_rows),
        "audit_schedule": list(AUDIT_STEPS),
        "loss_weights": list(LOSS_WEIGHTS),
        "lambda_region": LAMBDA_REGION,
        "lambda_residual": LAMBDA_RESIDUAL,
        "optimizer_param_groups_initial": optimizer_record,
        "missing_keys_expected_rsbr_only": missing,
        "fresh_restart_from_a0": True,
        "frozen_sshr_hash_before": baseline_hash_before,
        "frozen_sshr_hash_after": baseline_hash_after,
        "frozen_sshr_parameters_unchanged": baseline_unchanged,
        "deterministic_structure_check": structure,
        "audit_steps": audit_rows,
        "update_steps": update_rows,
        "region_statistics": region_statistics,
        "transition_fraction_last8_mean": transition_last8,
        "rapid_residual_growth": rapid_growth,
        "all_finite": all(row["finite"] for row in audit_rows + update_rows),
        "hard_failures": hard_failures,
        "review_triggers": reviews,
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "command": " ".join(sys.argv),
        "pilot_started": False,
        "test_accessed": False,
        "luad_accessed": False,
    }
    write_json(readiness_dir / "summary.json", summary)
    write_combined_report(output, parity, summary)
    print(decision, json.dumps({
        "hard_failures": hard_failures,
        "review_triggers": reviews,
        "end": audit_rows[-1],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
