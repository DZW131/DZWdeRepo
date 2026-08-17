"""Profile Full CDSR against exact uniform A0 at batch20/224/BF16."""

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

from network.cdsr import (
    AnalyticalRectificationNeed,
    SelectiveRectificationGate,
)
from network.resnet38_cls import HFRM, Net
from tool.torchutils import PolyOptimizer


LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)


class OperationCounter:
    """Exact Conv/Linear FLOPs plus an explicit CDSR functional estimate."""

    def __init__(self):
        self.module_flops = 0
        self.cdsr_functional_flops = 0
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
            elif isinstance(module, AnalyticalRectificationNeed):
                self.handles.append(
                    module.register_forward_hook(self._need_hook)
                )
            elif isinstance(module, SelectiveRectificationGate):
                self.handles.append(
                    module.register_forward_hook(self._gate_hook)
                )
            elif isinstance(module, HFRM):
                self.handles.append(
                    module.register_forward_hook(self._hfrm_hook)
                )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _conv_hook(self, module, _inputs, output):
        kernel_macs = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        self.module_flops += 2 * output.numel() * kernel_macs

    def _linear_hook(self, module, _inputs, output):
        self.module_flops += 2 * output.numel() * module.in_features

    def _need_hook(self, _module, inputs, output):
        stage_logits, deep_logits = inputs
        stage_elements = stage_logits.numel()
        deep_elements = deep_logits.numel()
        resized_deep_elements = stage_elements
        pixels = output["need_map"].numel()

        # Primitive-operation estimate. A transcendental (exp/log), compare,
        # add, multiply, or divide counts as one operation. It includes the two
        # softmaxes, bilinear resize, JSD, two entropies, reliability, and the
        # frozen probabilistic-OR Need expression.
        softmax = 4 * (stage_elements + deep_elements)
        bilinear_resize = 7 * resized_deep_elements
        jsd_and_entropies = 18 * stage_elements
        pixelwise_need = 10 * pixels
        self.cdsr_functional_flops += (
            softmax + bilinear_resize + jsd_and_entropies + pixelwise_need
        )

    def _gate_hook(self, _module, inputs, _output):
        need_map = inputs[0]
        # Two instances of 1 - alpha * (1 - N): three operations each.
        self.cdsr_functional_flops += 6 * need_map.numel()

    def _hfrm_hook(self, module, inputs, _output):
        if module.rectification_mode == "cdsr":
            feature = inputs[0]
            # Two additional gate-by-residual products.
            self.cdsr_functional_flops += 2 * feature.numel()


def synchronize(device):
    torch.cuda.synchronize(device)


def classification_loss(outputs, labels):
    return sum(
        weight * F.multilabel_soft_margin_loss(output, labels)
        for weight, output in zip(LOSS_WEIGHTS, outputs[:4])
    )


def make_optimizer(model, max_step):
    lr = 0.01
    weight_decay = 5e-4
    groups = model.get_parameter_groups()
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


def timed_forward(model, sample, device, warmup, iterations):
    model.eval()
    timings = []
    with torch.no_grad():
        for index in range(warmup + iterations):
            if index == warmup:
                synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(sample)
            synchronize(device)
            if index >= warmup:
                timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }


def timed_train(model, sample, labels, device, warmup, iterations):
    optimizer = make_optimizer(model, max_step=warmup + iterations)
    model.train()
    timings = []
    for index in range(warmup + iterations):
        if index == warmup:
            synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }


def profile_mode(mode, args, device):
    torch.manual_seed(args.seed)
    model = Net(n_class=4, rectification_mode=mode).to(device)
    sample = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )
    labels = torch.randint(
        0, 2, (args.batch_size, 4), device=device
    ).float()

    counter = OperationCounter()
    counter.register(model)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(sample)
    counter.close()

    forward = timed_forward(
        model, sample, device, args.warmup, args.iterations
    )
    train = timed_train(
        model, sample, labels, device, args.warmup, args.iterations
    )
    result = {
        "mode": mode,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "learnable_cdsr_scalars": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("cdsr_selective_gate.")
        ),
        "module_flops_per_image": counter.module_flops / args.batch_size,
        "estimated_cdsr_functional_flops_per_image": (
            counter.cdsr_functional_flops / args.batch_size
        ),
        "estimated_total_flops_per_image": (
            counter.module_flops + counter.cdsr_functional_flops
        ) / args.batch_size,
        "forward": forward,
        "train_step": train,
    }
    del model, sample, labels
    torch.cuda.empty_cache()
    return result


def percent_delta(candidate, baseline):
    return 100.0 * (candidate / baseline - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CDSR batch20 BF16 profiling requires CUDA")
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")

    device = torch.device("cuda")
    baseline = profile_mode("uniform", args, device)
    cdsr = profile_mode("cdsr", args, device)
    comparison = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "precision": "bf16",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "flop_method": (
            "exact Conv2d/Linear multiply-and-add count plus an explicit "
            "primitive-operation estimate for CDSR softmax, resize, JSD, "
            "entropy, Need, gates, and residual modulation; one multiply-add "
            "is two FLOPs and one transcendental is one estimated operation"
        ),
        "results": [baseline, cdsr],
        "cdsr_vs_a0": {
            "additional_parameters": cdsr["parameters"] - baseline["parameters"],
            "parameter_percent": percent_delta(
                cdsr["parameters"], baseline["parameters"]
            ),
            "estimated_flops_percent": percent_delta(
                cdsr["estimated_total_flops_per_image"],
                baseline["estimated_total_flops_per_image"],
            ),
            "forward_median_latency_percent": percent_delta(
                cdsr["forward"]["median_ms"],
                baseline["forward"]["median_ms"],
            ),
            "train_median_latency_percent": percent_delta(
                cdsr["train_step"]["median_ms"],
                baseline["train_step"]["median_ms"],
            ),
            "forward_peak_memory_percent": percent_delta(
                cdsr["forward"]["peak_cuda_memory_bytes"],
                baseline["forward"]["peak_cuda_memory_bytes"],
            ),
            "train_peak_memory_percent": percent_delta(
                cdsr["train_step"]["peak_cuda_memory_bytes"],
                baseline["train_step"]["peak_cuda_memory_bytes"],
            ),
        },
    }
    deltas = comparison["cdsr_vs_a0"]
    comparison["budget"] = {
        "additional_parameters_eq_2": deltas["additional_parameters"] == 2,
        "estimated_flops_lt_0_1_percent": (
            deltas["estimated_flops_percent"] < 0.1
        ),
        "forward_latency_lt_5_percent": (
            deltas["forward_median_latency_percent"] < 5.0
        ),
        "train_latency_lt_10_percent": (
            deltas["train_median_latency_percent"] < 10.0
        ),
    }
    comparison["budget"]["pass"] = all(comparison["budget"].values())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
