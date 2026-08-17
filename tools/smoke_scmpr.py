"""CUDA optimization-readiness smoke for frozen SC-MPR v1.0."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net, Net_CAM
from tool.torchutils import PolyOptimizer


def gradient_record(parameter):
    gradient = parameter.grad
    if gradient is None:
        return {
            "present": False,
            "finite": None,
            "norm": None,
            "abs_sum": None,
            "max_abs": None,
            "nonzero_count": None,
        }
    gradient = gradient.detach().float()
    absolute = gradient.abs()
    return {
        "present": True,
        "finite": bool(torch.isfinite(gradient).all().item()),
        # Accumulate in float64 so gradients near 1e-25 do not report an
        # artificial zero after float32 square-and-sum underflow.
        "norm": gradient.double().norm().item(),
        "abs_sum": absolute.sum().item(),
        "max_abs": absolute.max().item(),
        "nonzero_count": int(torch.count_nonzero(gradient).item()),
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


def tracked_parameter_groups(model):
    stages = {
        "stage1": model.hfrm_56,
        "stage2": model.hfrm_28_1,
        "stage3": model.hfrm_28_2,
    }
    groups = {
        "policy_input": tuple(model.scmpr_shared.gate_policy[0].parameters()),
        "policy_output": tuple(model.scmpr_shared.gate_policy[-1].parameters()),
        "deep_projector": tuple(model.scmpr_shared.deep_projector.parameters()),
    }
    for name, hfrm in stages.items():
        groups[f"target_projector_{name}"] = tuple(
            hfrm.scmpr_context.semantic_condition.target_projector.parameters()
        )
        groups[f"beta_{name}"] = (hfrm.scmpr_context.beta_logit,)
    return groups


def snapshot_parameter_groups(groups):
    return {
        name: tuple(parameter.detach().clone() for parameter in parameters)
        for name, parameters in groups.items()
    }


def parameter_group_norm(parameters):
    squared = sum(
        parameter.detach().double().square().sum()
        for parameter in parameters
    )
    return squared.sqrt().item()


def parameter_group_delta(parameters, reference):
    squared = sum(
        (parameter.detach().double() - initial.double()).square().sum()
        for parameter, initial in zip(parameters, reference)
    )
    return squared.sqrt().item()


def movement_record(groups, reference, decay_shadow=None):
    records = {}
    for name, parameters in groups.items():
        initial_norm = parameter_group_norm(reference[name])
        delta_norm = parameter_group_delta(parameters, reference[name])
        record = {
            "initial_norm": initial_norm,
            "delta_norm": delta_norm,
            "relative_delta": (
                delta_norm / initial_norm if initial_norm > 0.0 else None
            ),
            "finite": bool(
                all(torch.isfinite(parameter).all() for parameter in parameters)
            ),
        }
        if decay_shadow is not None:
            decay_parameters = tuple(
                item["value"] for item in decay_shadow[name]
            )
            decay_delta = parameter_group_delta(
                decay_parameters, reference[name]
            )
            task_excess = parameter_group_delta(
                parameters, decay_parameters
            )
            record.update(
                {
                    "decay_only_delta_norm": decay_delta,
                    "task_excess_delta_norm": task_excess,
                    "task_excess_relative": (
                        task_excess / initial_norm
                        if initial_norm > 0.0
                        else None
                    ),
                }
            )
        records[name] = record
    return records


def make_decay_only_shadow(groups, optimizer):
    optimizer_groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    return {
        name: tuple(
            {
                "value": parameter.detach().clone(),
                "momentum_buffer": torch.zeros_like(parameter),
                "optimizer_group": optimizer_groups[id(parameter)],
            }
            for parameter in parameters
        )
        for name, parameters in groups.items()
    }


def advance_decay_only_shadow(decay_shadow):
    with torch.no_grad():
        for items in decay_shadow.values():
            for item in items:
                group = item["optimizer_group"]
                value = item["value"]
                gradient = group.get("weight_decay", 0.0) * value
                momentum_buffer = item["momentum_buffer"]
                momentum_buffer.mul_(group.get("momentum", 0.0)).add_(
                    gradient
                )
                value.add_(momentum_buffer, alpha=-group["lr"])


def safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def load_real_images(root, count, image_size, device):
    root = Path(root)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[:count]
    if not paths:
        raise RuntimeError(f"No real images found under {root}")
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.size != (image_size, image_size):
            image = TF.resize(
                image,
                [image_size, image_size],
                interpolation=InterpolationMode.BILINEAR,
            )
        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        images.append(image)
    return torch.stack(images).to(device), [str(path) for path in paths]


def initialization_record(
    model, images, amp_enabled, input_kind, precision, paths=None
):
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=amp_enabled
    ):
        _, diagnostics = model.forward_with_diagnostics(images)
    stages = {}
    for name in ("stage1", "stage2", "stage3"):
        stage = diagnostics["scmpr"][name]
        ch = stage["original_ch"].detach().float()
        shift = stage["scmpr_context"].detach().float() - ch
        stages[name] = {
            "gate_fine": stage["summary"]["gate_fine"],
            "gate_morphology": stage["summary"]["gate_morphology"],
            "context_l2_drift_ratio": (
                shift.double().norm() / ch.double().norm().clamp_min(1e-12)
            ).item(),
            "context_rms_drift_ratio": stage["summary"][
                "context_shift_ratio"
            ],
            "finite": bool(stage["all_finite"]),
        }
    model.train(was_training)
    return {
        "input_kind": input_kind,
        "precision": precision,
        "batch_size": images.shape[0],
        "paths": paths,
        "stages": stages,
    }


def stage_record(hfrm, diagnostics):
    context = hfrm.scmpr_context
    ch_norm = diagnostics["original_ch"].detach().float().norm()
    correction_norm = (
        context.beta.detach().float()
        * diagnostics["delta_sc"].detach().float()
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
    parameter_groups = tracked_parameter_groups(model)
    initial_parameters = snapshot_parameter_groups(parameter_groups)
    previous_parameters = snapshot_parameter_groups(parameter_groups)
    decay_shadow = make_decay_only_shadow(parameter_groups, optimizer)
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
            outputs, diagnostics = model.forward_with_diagnostics(images)
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
        output_norm = record["shared"]["policy_output_gradient"]["norm"]
        record["gradient_ratios"] = {
            "policy_input_to_output": safe_ratio(
                record["shared"]["policy_input_gradient"]["norm"],
                output_norm,
            ),
            "deep_projector_to_output": safe_ratio(
                record["shared"]["deep_projector_gradient"]["norm"],
                output_norm,
            ),
            "target_projector_to_output": {
                name: safe_ratio(
                    record["stages"][name]["target_projector_gradient"][
                        "norm"
                    ],
                    output_norm,
                )
                for name in ("stage1", "stage2", "stage3")
            },
        }
        optimizer.step()
        advance_decay_only_shadow(decay_shadow)
        record["parameter_updates"] = movement_record(
            parameter_groups, previous_parameters
        )
        record["cumulative_movement"] = movement_record(
            parameter_groups, initial_parameters, decay_shadow
        )
        previous_parameters = snapshot_parameter_groups(parameter_groups)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return records, movement_record(
        parameter_groups, initial_parameters, decay_shadow
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--real-input-root", type=Path)
    parser.add_argument("--real-input-count", type=int, default=4)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SC-MPR smoke requires CUDA")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    model = Net(n_class=4, context_mode="sc-mpr").to(device)
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

    initialization = []
    for enabled, precision in ((False, "fp32"), (amp_enabled, args.amp_dtype)):
        initialization.append(
            initialization_record(
                model,
                images,
                enabled,
                input_kind="fixed_random",
                precision=precision,
            )
        )
    if args.real_input_root is not None:
        real_images, real_paths = load_real_images(
            args.real_input_root,
            args.real_input_count,
            args.image_size,
            device,
        )
        for enabled, precision in (
            (False, "fp32"),
            (amp_enabled, args.amp_dtype),
        ):
            initialization.append(
                initialization_record(
                    model,
                    real_images,
                    enabled,
                    input_kind="real",
                    precision=precision,
                    paths=real_paths,
                )
            )
        del real_images

    records, final_movement = run_steps(
        model, optimizer, images, labels, args.steps, amp_enabled
    )
    cam_model = Net_CAM(n_class=4, context_mode="sc-mpr").to(device)
    cam_model.load_state_dict(model.state_dict())
    cam_model.eval()
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=amp_enabled
    ):
        cams = cam_model.forward_cam(images[:1])
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
        ] + [
            record["stages"][stage]["beta_logit_gradient"]
            for stage in ("stage1", "stage2", "stage3")
        ]
        if all(item["present"] and item["abs_sum"] > 0.0 for item in watched):
            active_by_step = record["step"]
            break
    final_ratios = records[-1]["gradient_ratios"]
    practical_ratios = [
        final_ratios["policy_input_to_output"],
        final_ratios["deep_projector_to_output"],
        *final_ratios["target_projector_to_output"].values(),
    ]
    initial_safety = all(
        abs(stage[gate]["mean"] - 0.1) < 0.01
        and stage[gate]["std"] > 0.0
        and stage["context_l2_drift_ratio"] < 0.05
        and stage["finite"]
        for source in initialization
        for stage in source["stages"].values()
        for gate in ("gate_fine", "gate_morphology")
    )
    measurable_movement = all(
        item["finite"] and item["task_excess_delta_norm"] > 0.0
        for item in final_movement.values()
    )
    step_finite = all(
        record["loss_finite"]
        and all(stage["finite"] for stage in record["stages"].values())
        and all(
            gradient["present"] and gradient["finite"]
            for gradient in record["shared"].values()
        )
        and all(
            stage[key]["present"] and stage[key]["finite"]
            for stage in record["stages"].values()
            for key in (
                "gamma_sem_gradient",
                "gamma_context_gradient",
                "beta_logit_gradient",
                "target_projector_gradient",
            )
        )
        for record in records
    )
    result = {
        "device": torch.cuda.get_device_name(device),
        "pytorch": torch.__version__,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "amp_dtype": args.amp_dtype,
        "steps": records,
        "initialization_safety": initialization,
        "final_parameter_movement": final_movement,
        "transition_path_active_by_step": active_by_step,
        "practical_gradient_ratio_floor": 1e-8,
        "initialization_safety_pass": initial_safety,
        "practical_gradient_pass": all(
            ratio is not None and ratio > 1e-8
            for ratio in practical_ratios
        ),
        "parameter_movement_pass": measurable_movement,
        "all_steps_finite": step_finite,
        "optimization_readiness_pass": (
            active_by_step is not None
            and active_by_step <= 2
            and initial_safety
            and all(
                ratio is not None and ratio > 1e-8
                for ratio in practical_ratios
            )
            and measurable_movement
            and step_finite
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
