#!/usr/bin/env python3
"""LW-SHR Phase-0 engineering, identity, gradient, and BF16 audit."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import resnet38_cls
from network.wavelet_gate import SharedLearnableWaveletBank, SubbandStructuralGate
from tools.lw_shr_common import (
    A0_COMMIT,
    LOSS_WEIGHTS,
    MatchedAugmentationDataset,
    VARIANT_TO_MODE,
    build_optimizer,
    load_common_checkpoint,
    load_schedule,
    load_variant_from_common,
    optimizer_summary,
    restore_named_optimizer_state,
    set_seed,
    sha256_file,
    write_json,
)


def source_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def official_loss(outputs, target):
    losses = [F.multilabel_soft_margin_loss(output, target) for output in outputs[:4]]
    return sum(weight * value for weight, value in zip(LOSS_WEIGHTS, losses))


def scheduled_batch(dataset, schedule, epoch, step, device="cuda"):
    requests = zip(
        schedule["indices"][epoch, step],
        schedule["augmentation_seeds"][epoch, step],
    )
    rows = [dataset[(int(index), int(seed))] for index, seed in requests]
    images = torch.stack([row[1] for row in rows]).to(device)
    labels = torch.stack([row[2] for row in rows]).to(device)
    return images, labels


def optimizer_coverage(model, optimizer, variant):
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    ids = [id(parameter) for parameter in grouped]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"Duplicate optimizer parameter in {variant}")
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    missing = trainable.difference(ids)
    if missing:
        names = [
            name for name, parameter in model.named_parameters() if id(parameter) in missing
        ]
        raise AssertionError(f"Ungrouped trainable parameters in {variant}: {names}")

    scratch_ids = {id(parameter) for parameter in optimizer.param_groups[2]["params"]}
    direct = {}
    for name, parameter in (
        ("dec_lo", model.wavelet_bank.dec_lo),
        ("dec_hi", model.wavelet_bank.dec_hi),
        ("lambda_sf", model.hfrm_28_1.lambda_sf),
    ):
        if parameter is None or not parameter.requires_grad:
            direct[name] = "not_trainable"
        else:
            if id(parameter) not in scratch_ids:
                raise AssertionError(f"{variant} {name} is not in scratch LR group")
            direct[name] = "scratch_group_exactly_once"
    return {
        "all_trainable_covered": True,
        "no_duplicates": True,
        "direct_parameters": direct,
        "groups": optimizer_summary(optimizer),
    }


def shape_audit():
    rows = []
    for channels in (256, 512, 1024):
        for height, width in ((8, 10), (7, 9)):
            bank = SharedLearnableWaveletBank(trainable=True).cuda()
            gate = SubbandStructuralGate(channels).cuda().to(dtype=torch.bfloat16)
            x = torch.randn(
                1, channels, height, width, device="cuda", dtype=torch.bfloat16
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, details = gate(x, bank, return_details=True)
            expected = (int(np.ceil(height / 2)), int(np.ceil(width / 2)))
            band_shapes = {
                name: list(value.shape)
                for name, value in details["subbands"].items()
            }
            if any(tuple(shape[-2:]) != expected for shape in band_shapes.values()):
                raise AssertionError(band_shapes)
            if logits.shape != x.shape or not torch.isfinite(logits).all():
                raise AssertionError("Gate shape/finiteness failure")
            rows.append(
                {
                    "channels": channels,
                    "input_shape": list(x.shape),
                    "band_shapes": band_shapes,
                    "gate_logit_shape": list(logits.shape),
                    "finite": True,
                }
            )
            del bank, gate, x, logits, details
            torch.cuda.empty_cache()
    return rows


def identity_audit(common, image):
    set_seed(42, deterministic=True)
    baseline = resnet38_cls.Net(4).cuda().eval()
    baseline.load_state_dict(common["model_state"], strict=True)
    with torch.no_grad():
        baseline_fp32 = [value.detach().float().cpu() for value in baseline(image[:1])]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            baseline_bf16 = [value.detach().float().cpu() for value in baseline(image[:1])]
    del baseline
    torch.cuda.empty_cache()

    result = {}
    for variant in VARIANT_TO_MODE:
        model, load_audit = load_variant_from_common(variant, common)
        model.eval()
        with torch.no_grad():
            proposed_fp32 = [value.detach().float().cpu() for value in model(image[:1])]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                proposed_bf16 = [value.detach().float().cpu() for value in model(image[:1])]
        fp32_max = max(
            float((left - right).abs().max())
            for left, right in zip(baseline_fp32, proposed_fp32)
        )
        bf16_max = max(
            float((left - right).abs().max())
            for left, right in zip(baseline_bf16, proposed_bf16)
        )
        if fp32_max >= 1.0e-5:
            raise AssertionError(f"{variant} FP32 identity error {fp32_max}")
        result[variant] = {
            "fp32_max_abs_diff": fp32_max,
            "fp32_pass_lt_1e-5": True,
            "bf16_max_abs_diff": bf16_max,
            "checkpoint_load": load_audit,
        }
        del model, proposed_fp32, proposed_bf16
        torch.cuda.empty_cache()
    return result


def gradient_audit(variant, common, batches, schedule):
    model, load_audit = load_variant_from_common(variant, common)
    optimizer = build_optimizer(model)
    restore = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    if restore["skipped"]:
        raise AssertionError(f"Optimizer restore skipped old parameters: {restore['skipped']}")
    optimizer.global_step = int(common["optimizer_global_step"])
    coverage = optimizer_coverage(model, optimizer, variant)
    model.train()
    tracked = {
        "dec_lo": model.wavelet_bank.dec_lo,
        "dec_hi": model.wavelet_bank.dec_hi,
        "output_projection": model.hfrm_28_1.wavelet_gate.output_projection.weight,
    }
    if model.hfrm_28_1.lambda_sf is not None:
        tracked["lambda_sf"] = model.hfrm_28_1.lambda_sf
    initial = {name: parameter.detach().clone() for name, parameter in tracked.items()}
    records = []
    torch.cuda.reset_peak_memory_stats()
    for step, (image, target) in enumerate(batches, start=1):
        torch.manual_seed(int(schedule["model_seeds"][20, step - 1]))
        torch.cuda.manual_seed_all(int(schedule["model_seeds"][20, step - 1]))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(image)
            loss = official_loss(outputs, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {variant} Phase-0 loss")
        loss.backward()
        gradients = {
            name: None
            if parameter.grad is None
            else float(parameter.grad.detach().float().norm())
            for name, parameter in tracked.items()
        }
        all_gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
        updates = {
            name: float((parameter.detach() - initial[name]).float().norm())
            for name, parameter in tracked.items()
        }
        records.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norms": gradients,
                "update_norms_from_t0": updates,
                "all_gradients_finite": bool(all_gradients_finite),
                "all_outputs_finite": bool(all(torch.isfinite(value).all() for value in outputs)),
            }
        )
    if variant in ("A2", "A3"):
        if records[0]["grad_norms"]["dec_lo"] != 0.0 or records[0]["grad_norms"]["dec_hi"] != 0.0:
            raise AssertionError("Identity output layer should block first-step filter gradients")
        if records[1]["grad_norms"]["dec_lo"] <= 0.0 or records[1]["grad_norms"]["dec_hi"] <= 0.0:
            raise AssertionError("Learnable wavelet filters did not open by step 2")
    if records[0]["grad_norms"]["output_projection"] <= 0.0:
        raise AssertionError("Wavelet output projection did not receive task gradient")
    if variant == "A3" and records[0]["grad_norms"]["lambda_sf"] <= 0.0:
        raise AssertionError("A3 lambda_sf did not receive task gradient")
    if not all(row["all_gradients_finite"] and row["all_outputs_finite"] for row in records):
        raise FloatingPointError("Non-finite Phase-0 result")
    result = {
        "variant": variant,
        "mode": VARIANT_TO_MODE[variant],
        "checkpoint_load": load_audit,
        "optimizer_restore": restore,
        "optimizer_coverage": coverage,
        "steps": records,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "filter_diagnostics_after_step2": model.wavelet_bank.diagnostics(),
        "pass": True,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    common = load_common_checkpoint(args.common_checkpoint)
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    batches = [
        scheduled_batch(dataset, schedule, 20, step)
        for step in range(2)
    ]

    started = time.time()
    shape = shape_audit()
    identity = identity_audit(common, batches[0][0])
    gradients = {
        variant: gradient_audit(variant, common, batches, schedule)
        for variant in ("A1", "A2", "A3")
    }
    summary = {
        "experiment": "LW-SHR Phase-0 Engineering Audit",
        "phase0_status": "PASS",
        "source_commit": source_commit(),
        "a0_commit": A0_COMMIT,
        "common_checkpoint": str(Path(args.common_checkpoint).resolve()),
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "schedule": str(Path(args.schedule).resolve()),
        "schedule_sha256": sha256_file(args.schedule),
        "shape_audit": shape,
        "identity_audit": identity,
        "gradient_bf16_audit": gradients,
        "runtime_seconds": time.time() - started,
        "test_used": False,
        "validation_used": False,
    }
    write_json(output / "lw_shr_phase0_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("LW_SHR_PHASE0_PASS", flush=True)


if __name__ == "__main__":
    main()
