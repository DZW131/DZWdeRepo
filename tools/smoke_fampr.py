"""Dataset-free Full FA-MPR optimization and inference smoke test."""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net_CAM
from tool.torchutils import PolyOptimizer


def gradient_summary(parameter):
    gradient = parameter.grad
    if gradient is None:
        return {"present": False, "finite": None, "nonzero": False, "norm": None}
    gradient = gradient.detach().float()
    return {
        "present": True,
        "finite": bool(torch.isfinite(gradient).all().item()),
        "nonzero": bool(gradient.abs().sum().item() > 0.0),
        "norm": gradient.norm().item(),
    }


def make_optimizer(model, max_step):
    parameter_groups = model.get_parameter_groups()
    learning_rate = 0.01
    weight_decay = 5e-4
    optimizer_groups = [
        {
            "params": parameter_groups[0],
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": parameter_groups[1],
            "lr": 2 * learning_rate,
            "weight_decay": 0.0,
        },
        {
            "params": parameter_groups[2],
            "lr": 10 * learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": parameter_groups[3],
            "lr": 20 * learning_rate,
            "weight_decay": 0.0,
        },
    ]
    return PolyOptimizer(
        optimizer_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
        max_step=max_step,
    )


def classification_loss(outputs, labels):
    weights = (0.10, 0.15, 0.25, 0.50)
    return sum(
        weight * F.multilabel_soft_margin_loss(output, labels)
        for weight, output in zip(weights, outputs[:4])
    )


def stage_record(hfrm, diagnostics):
    fampr = hfrm.fampr_context
    return {
        **diagnostics["summary"],
        "gamma_context": hfrm.gamma_context.detach().float().item(),
        "gamma_context_gradient": gradient_summary(hfrm.gamma_context),
        "anchor_logit_gradient": gradient_summary(fampr.anchor_logit),
        "band_predictor_gradient": gradient_summary(
            fampr.frequency_selector.band_weight_network[-1].weight
        ),
        "base_kernel_gradient": gradient_summary(
            fampr.adaptive_kernel.base_kernel
        ),
        "kernel_gate_gradient": gradient_summary(
            fampr.adaptive_kernel.gate_network[-1].weight
        ),
    }


def stage_path_active(stage):
    return all(
        stage[name]["present"]
        and stage[name]["finite"]
        and stage[name]["nonzero"]
        for name in (
            "anchor_logit_gradient",
            "band_predictor_gradient",
            "base_kernel_gradient",
            "kernel_gate_gradient",
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--steps", default=5, type=int)
    parser.add_argument(
        "--amp-dtype", default="bf16", choices=("none", "bf16")
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    device = torch.device(args.device)
    amp_enabled = args.amp_dtype == "bf16" and device.type == "cuda"
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)

    model = Net_CAM(n_class=4, context_mode="fampr").to(device)
    model.train()
    optimizer = make_optimizer(model, max_step=args.steps)
    images = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )
    labels = torch.randint(0, 2, (args.batch_size, 4), device=device).float()
    stage_modules = {
        "stage1": model.hfrm_56,
        "stage2": model.hfrm_28_1,
        "stage3": model.hfrm_28_2,
    }

    records = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            outputs, diagnostics = model._forward_impl(
                images, return_diagnostics=True
            )
            loss = classification_loss(outputs, labels)
        loss.backward()

        stages = {
            name: stage_record(module, diagnostics["fampr"][name])
            for name, module in stage_modules.items()
        }
        finite = {
            "loss": bool(torch.isfinite(loss).item()),
            "outputs": all(
                torch.isfinite(output).all().item() for output in outputs
            ),
            "diagnostics": all(
                diagnostics["fampr"][name]["all_finite"]
                for name in stage_modules
            ),
            "gradients": all(
                all(
                    stage[key]["finite"] in (None, True)
                    for key in (
                        "gamma_context_gradient",
                        "anchor_logit_gradient",
                        "band_predictor_gradient",
                        "base_kernel_gradient",
                        "kernel_gate_gradient",
                    )
                )
                for stage in stages.values()
            ),
        }
        finite["all"] = all(finite.values())
        record = {
            "step": step,
            "loss": loss.detach().float().item(),
            "lr": [group["lr"] for group in optimizer.param_groups],
            "momentum": [
                group.get("momentum", 0.0) for group in optimizer.param_groups
            ],
            "stages": stages,
            "finite_checks": finite,
        }
        optimizer.step()
        record["gamma_context_after_update"] = {
            name: module.gamma_context.detach().float().item()
            for name, module in stage_modules.items()
        }
        record["anchor_lambda_after_update"] = {
            name: module.fampr_context.anchor_lambda.detach().float().item()
            for name, module in stage_modules.items()
        }
        records.append(record)

    active_step = next(
        (
            record["step"]
            for record in records[1:3]
            if all(stage_path_active(stage) for stage in record["stages"].values())
        ),
        None,
    )

    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        inference_outputs = model.forward_cam(images[:1])
    forward_cam_finite = all(
        torch.isfinite(output).all().item() for output in inference_outputs
    )

    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "steps": records,
        "amp_dtype": "bf16" if amp_enabled else "none",
        "forward_cam_finite": forward_cam_finite,
        "optimization_readiness": {
            "assessed": args.steps >= 3,
            "pass": active_step is not None if args.steps >= 3 else None,
            "path_active_step": active_step,
            "criterion": (
                "all three stages have finite nonzero band-predictor, "
                "adaptive-kernel, kernel-gate, and anchor gradients by step 2 or 3"
            ),
        },
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
    }
    if not all(record["finite_checks"]["all"] for record in records):
        raise RuntimeError("non-finite value or gradient detected")
    if not forward_cam_finite:
        raise RuntimeError("forward_cam produced a non-finite tensor")
    if args.steps >= 3 and active_step is None:
        raise RuntimeError("FA-MPR path did not open by step 2 or 3")

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
