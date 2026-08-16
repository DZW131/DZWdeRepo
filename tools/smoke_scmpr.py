"""CUDA optimization-readiness smoke for frozen SC-MPR v1.0."""

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


def gradient_record(parameter):
    gradient = parameter.grad
    if gradient is None:
        return {"present": False, "finite": None, "norm": None}
    gradient = gradient.detach().float()
    return {
        "present": True,
        "finite": bool(torch.isfinite(gradient).all().item()),
        "norm": gradient.norm().item(),
    }


def make_optimizer(model, max_step):
    groups = model.get_parameter_groups()
    lr = 0.01
    weight_decay = 5e-4
    parameters = [
        {"params": groups[0], "lr": lr, "weight_decay": weight_decay},
        {"params": groups[1], "lr": 2 * lr, "weight_decay": 0.0},
        {"params": groups[2], "lr": 10 * lr, "weight_decay": weight_decay},
        {"params": groups[3], "lr": 20 * lr, "weight_decay": 0.0},
    ]
    return PolyOptimizer(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        max_step=max_step,
    )


def classification_loss(outputs, labels):
    return sum(
        weight * F.multilabel_soft_margin_loss(output, labels)
        for weight, output in zip((0.10, 0.15, 0.25, 0.50), outputs[:4])
    )


def stage_record(hfrm, diagnostics):
    context = hfrm.scmpr_context
    ch_norm = diagnostics["original_ch"].detach().float().norm()
    correction_norm = (
        context.beta.detach().float()
        * diagnostics["delta_zero_mean"].detach().float()
    ).norm()
    return {
        "gamma_sem": hfrm.gamma_veto.detach().float().item(),
        "gamma_sem_gradient": gradient_record(hfrm.gamma_veto),
        "gamma_context": hfrm.gamma_context.detach().float().item(),
        "gamma_context_gradient": gradient_record(hfrm.gamma_context),
        "beta": context.beta.detach().float().item(),
        "beta_logit_gradient": gradient_record(context.beta_logit),
        "target_projector_gradient": gradient_record(
            context.semantic_condition.target_projector.weight
        ),
        "correction_to_ch_norm_ratio": (
            correction_norm / ch_norm.clamp_min(1e-12)
        ).item(),
        "summary": diagnostics["summary"],
        "finite": bool(diagnostics["all_finite"]),
    }


def run_steps(model, optimizer, images, labels, steps, amp_enabled):
    records = []
    modules = {
        "stage1": model.hfrm_56,
        "stage2": model.hfrm_28_1,
        "stage3": model.hfrm_28_2,
    }
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=amp_enabled
        ):
            outputs, diagnostics = model._forward_impl(
                images, return_diagnostics=True
            )
            loss = classification_loss(outputs, labels)
        loss.backward()

        record = {
            "step": step,
            "loss": loss.detach().float().item(),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "stages": {
                name: stage_record(module, diagnostics["scmpr"][name])
                for name, module in modules.items()
            },
            "shared": {
                "deep_projector_gradient": gradient_record(
                    model.scmpr_shared.deep_projector.weight
                ),
                "policy_input_gradient": gradient_record(
                    model.scmpr_shared.gate_policy[0].weight
                ),
                "policy_output_gradient": gradient_record(
                    model.scmpr_shared.gate_policy[-1].weight
                ),
            },
            "optimizer": {
                "lr": [group["lr"] for group in optimizer.param_groups],
                "momentum": [
                    group.get("momentum", 0.0)
                    for group in optimizer.param_groups
                ],
            },
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        optimizer.step()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SC-MPR smoke requires CUDA")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    model = Net_CAM(n_class=4, context_mode="sc-mpr").to(device)
    model.train()
    optimizer = make_optimizer(model, max_step=args.steps)
    images = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )
    labels = torch.randint(
        0, 2, (args.batch_size, 4), device=device
    ).float()
    amp_enabled = args.amp_dtype == "bf16"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records = run_steps(
        model, optimizer, images, labels, args.steps, amp_enabled
    )
    model.eval()
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=amp_enabled
    ):
        cams = model.forward_cam(images[:1])
    forward_cam_finite = all(
        bool(torch.isfinite(tensor).all().item()) for tensor in cams
    )
    active_by_step = None
    for record in records:
        watched = [
            record["shared"]["deep_projector_gradient"],
            record["shared"]["policy_input_gradient"],
            record["shared"]["policy_output_gradient"],
        ] + [
            record["stages"][stage]["target_projector_gradient"]
            for stage in ("stage1", "stage2", "stage3")
        ]
        if all(item["present"] and item["norm"] > 0.0 for item in watched):
            active_by_step = record["step"]
            break
    result = {
        "device": torch.cuda.get_device_name(device),
        "pytorch": torch.__version__,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "amp_dtype": args.amp_dtype,
        "steps": records,
        "transition_path_active_by_step": active_by_step,
        "optimization_readiness_pass": (
            active_by_step is not None
            and active_by_step <= 3
            and forward_cam_finite
        ),
        "forward_cam_finite": forward_cam_finite,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not result["optimization_readiness_pass"]:
        raise RuntimeError("SC-MPR optimization-readiness gate failed")


if __name__ == "__main__":
    main()
