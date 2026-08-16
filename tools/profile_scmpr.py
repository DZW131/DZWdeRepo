"""Paired A0/SC-MPR parameter, FLOP, latency, and memory profile."""

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

from network.resnet38_cls import Net
from network.scmpr.scmpr_context import SCMPRContext
from tool.torchutils import PolyOptimizer


class OperationCounter:
    """Count module FLOPs and document SC-MPR functional estimates."""

    def __init__(self):
        self.module_flops = 0
        self.scmpr_functional_flops = 0
        self.handles = []

    def register(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._conv))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear))
            elif isinstance(module, SCMPRContext):
                self.handles.append(module.register_forward_hook(self._scmpr))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def _conv(self, module, inputs, output):
        kernel_macs = (
            module.kernel_size[0] * module.kernel_size[1]
            * module.in_channels // module.groups
        )
        self.module_flops += 2 * output.numel() * kernel_macs

    def _linear(self, module, inputs, output):
        self.module_flops += 2 * output.numel() * module.in_features

    def _scmpr(self, module, inputs, output):
        feature = inputs[0]
        batch, channels, height, width = feature.shape
        elements = batch * channels * height * width
        spatial = batch * height * width
        classes = inputs[3].shape[1]
        # Explicit estimate for operations outside Conv2d/Linear hooks:
        # LP3/LP15, residual/quality normalization, probability softmax and
        # LP3 variation, interpolation/cosine, gates, demeaning, and addition.
        lowpasses = elements * (3 * 3 + 15 * 15)
        residual_quality = elements * 10
        semantic_maps = spatial * (classes * 16 + 32 * 8)
        rectification = elements * 9
        self.scmpr_functional_flops += (
            lowpasses + residual_quality + semantic_maps + rectification
        )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    return PolyOptimizer(parameters, lr=lr, weight_decay=weight_decay, max_step=max_step)


def classification_loss(outputs, labels):
    return sum(
        weight * F.multilabel_soft_margin_loss(output, labels)
        for weight, output in zip((0.10, 0.15, 0.25, 0.50), outputs[:4])
    )


def timed_forward(model, sample, device, amp_enabled, warmup, iterations):
    model.eval()
    timings = []
    with torch.no_grad():
        for index in range(warmup + iterations):
            if index == warmup:
                synchronize(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                model(sample)
            synchronize(device)
            if index >= warmup:
                timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def timed_train(model, sample, labels, device, amp_enabled, warmup, iterations):
    model.train()
    optimizer = make_optimizer(model, warmup + iterations)
    timings = []
    for index in range(warmup + iterations):
        if index == warmup:
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
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
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def profile_mode(mode, device, batch_size, image_size, amp_enabled, warmup, iterations):
    torch.manual_seed(42)
    model = Net(n_class=4, context_mode=mode).to(device)
    sample = torch.randn(batch_size, 3, image_size, image_size, device=device)
    labels = torch.randint(0, 2, (batch_size, 4), device=device).float()
    counter = OperationCounter()
    counter.register(model)
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        model(sample)
    counter.close()
    result = {
        "context_mode": mode,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "scmpr_parameters": sum(parameter.numel() for name, parameter in model.named_parameters() if "scmpr" in name),
        "module_flops_per_image": counter.module_flops / batch_size,
        "estimated_scmpr_functional_flops_per_image": counter.scmpr_functional_flops / batch_size,
        "estimated_total_flops_per_image": (counter.module_flops + counter.scmpr_functional_flops) / batch_size,
        "forward": timed_forward(model, sample, device, amp_enabled, warmup, iterations),
        "train_step": timed_train(model, sample, labels, device, amp_enabled, warmup, iterations),
    }
    del model, sample, labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def percent_delta(candidate, baseline):
    return 100.0 * (candidate / baseline - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    amp_enabled = args.amp_dtype == "bf16" and device.type == "cuda"
    modes = [
        profile_mode(mode, device, args.batch_size, args.image_size, amp_enabled, args.warmup, args.iterations)
        for mode in ("ch", "sc-mpr")
    ]
    baseline, candidate = modes
    deltas = {
        "parameters": percent_delta(candidate["parameters"], baseline["parameters"]),
        "estimated_total_flops": percent_delta(candidate["estimated_total_flops_per_image"], baseline["estimated_total_flops_per_image"]),
        "forward_median_latency": percent_delta(candidate["forward"]["median_ms"], baseline["forward"]["median_ms"]),
        "train_median_latency": percent_delta(candidate["train_step"]["median_ms"], baseline["train_step"]["median_ms"]),
    }
    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "amp_dtype": "bf16" if amp_enabled else "none",
        "flop_method": "Conv2d/Linear multiply-add count plus explicit fixed-filter and tensor-operation estimates; one multiply-add is 2 FLOPs",
        "results": modes,
        "scmpr_vs_a0_percent": deltas,
        "gates": {
            "parameter_limit_percent": 1.0,
            "parameter_pass": deltas["parameters"] < 1.0,
            "flop_limit_percent": 1.0,
            "flop_pass": deltas["estimated_total_flops"] < 1.0,
            "preferred_forward_latency_limit_percent": 15.0,
            "forward_latency_preference_pass": deltas["forward_median_latency"] < 15.0,
            "hard_forward_latency_explanation_threshold_percent": 20.0,
        },
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
