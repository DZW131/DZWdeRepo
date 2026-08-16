"""Profile the exact A0 and Full FA-MPR models under one environment."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.fampr.fampr_context import FrequencyAdaptiveMorphologyContext
from network.resnet38_cls import Net
from tool.torchutils import PolyOptimizer


class OperationCounter:
    """Count module FLOPs plus documented FA-MPR functional estimates."""

    def __init__(self):
        self.module_flops = 0
        self.fampr_functional_flops = 0
        self.handles = []

    def register(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(
                    module.register_forward_hook(self._conv_hook)
                )
            elif isinstance(module, nn.Linear):
                self.handles.append(
                    module.register_forward_hook(self._linear_hook)
                )
            elif isinstance(module, FrequencyAdaptiveMorphologyContext):
                self.handles.append(
                    module.register_forward_hook(self._fampr_hook)
                )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _conv_hook(self, module, inputs, output):
        kernel_macs = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        self.module_flops += 2 * output.numel() * kernel_macs

    def _linear_hook(self, module, inputs, output):
        self.module_flops += 2 * output.numel() * module.in_features

    def _fampr_hook(self, module, inputs, output):
        feature = inputs[0]
        batch, channels, height, width = feature.shape
        elements = batch * channels * height * width
        spatial = batch * height * width

        # Functional operations not visible to Conv/Linear hooks. This is a
        # transparent engineering estimate: three average pools, telescoping
        # bands/energies/fusion, bilinear 9-point sampling, two 3x3 weighted
        # sums, low/high channel gating, and the CH anchor blend.
        lowpass = elements * (3 * 3 + 7 * 7 + 15 * 15)
        band_and_energy = elements * 16
        morphology_and_dilation = spatial * 24
        bilinear_sampling = elements * 9 * 8
        two_kernel_reductions = elements * 2 * (9 + 8)
        gating_and_anchor = elements * 8
        self.fampr_functional_flops += (
            lowpass
            + band_and_energy
            + morphology_and_dilation
            + bilinear_sampling
            + two_kernel_reductions
            + gating_and_anchor
        )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def classification_loss(outputs, labels):
    weights = (0.10, 0.15, 0.25, 0.50)
    return sum(
        weight * F.multilabel_soft_margin_loss(output, labels)
        for weight, output in zip(weights, outputs[:4])
    )


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


def timed_forward(model, sample, device, amp_enabled, warmup, iterations):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                model(sample)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                model(sample)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
    }


def timed_train_step(
    model, sample, labels, device, amp_enabled, warmup, iterations
):
    model.train()
    optimizer = make_optimizer(model, max_step=warmup + iterations)
    timings = []
    for index in range(warmup + iterations):
        if index == warmup:
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            outputs = model(sample)
            loss = classification_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        synchronize(device)
        if index >= warmup:
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
    }


def profile_mode(
    context_mode,
    device,
    batch_size,
    image_size,
    amp_enabled,
    warmup,
    iterations,
):
    torch.manual_seed(42)
    model = Net(n_class=4, context_mode=context_mode).to(device)
    sample = torch.randn(
        batch_size, 3, image_size, image_size, device=device
    )
    labels = torch.randint(0, 2, (batch_size, 4), device=device).float()

    counter = OperationCounter()
    counter.register(model)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        model(sample)
    counter.close()

    forward = timed_forward(
        model, sample, device, amp_enabled, warmup, iterations
    )
    train_step = timed_train_step(
        model,
        sample,
        labels,
        device,
        amp_enabled,
        warmup,
        iterations,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    fampr_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".fampr_context." in name
    )
    result = {
        "context_mode": context_mode,
        "parameters": total_parameters,
        "fampr_parameters": fampr_parameters,
        "module_flops_per_image": counter.module_flops / batch_size,
        "estimated_fampr_functional_flops_per_image": (
            counter.fampr_functional_flops / batch_size
        ),
        "estimated_total_flops_per_image": (
            counter.module_flops + counter.fampr_functional_flops
        )
        / batch_size,
        "forward": forward,
        "train_step": train_step,
    }
    del model, sample, labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def percent_delta(candidate, baseline):
    return 100.0 * (candidate / baseline - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument(
        "--amp-dtype", default="bf16", choices=("none", "bf16")
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")

    device = torch.device(args.device)
    amp_enabled = args.amp_dtype == "bf16" and device.type == "cuda"
    modes = [
        profile_mode(
            mode,
            device,
            args.batch_size,
            args.image_size,
            amp_enabled,
            args.warmup,
            args.iterations,
        )
        for mode in ("ch", "fampr")
    ]
    baseline, fampr = modes
    comparison = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "amp_dtype": "bf16" if amp_enabled else "none",
        "flop_method": (
            "exact Conv2d/Linear multiply-and-add count plus an explicit "
            "functional-operation estimate for FA-MPR pooling, sampling, "
            "kernel reductions, gates, and anchor; one multiply-add is 2 FLOPs"
        ),
        "results": modes,
        "fampr_vs_a0_percent": {
            "parameters": percent_delta(
                fampr["parameters"], baseline["parameters"]
            ),
            "estimated_total_flops": percent_delta(
                fampr["estimated_total_flops_per_image"],
                baseline["estimated_total_flops_per_image"],
            ),
            "forward_median_latency": percent_delta(
                fampr["forward"]["median_ms"],
                baseline["forward"]["median_ms"],
            ),
            "train_median_latency": percent_delta(
                fampr["train_step"]["median_ms"],
                baseline["train_step"]["median_ms"],
            ),
        },
        "parameter_budget": {
            "limit_percent": 10.0,
            "pass": percent_delta(
                fampr["parameters"], baseline["parameters"]
            )
            < 10.0,
        },
    }
    serialized = json.dumps(comparison, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
