"""CUDA readiness audit for the frozen RDDR Phase-1 UC/DD implementation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from network.resnet38_cls import Net
from tool import torchutils


LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pretrained(model, path):
    if str(path).endswith(".params"):
        state = importlib.import_module(
            "network.resnet38d"
        ).convert_mxnet_to_torch(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def build_optimizer(model, max_step=25 * 1171):
    groups = model.get_parameter_groups()
    return torchutils.PolyOptimizer(
        [
            {"params": groups[0], "lr": 0.01, "weight_decay": 5.0e-4},
            {"params": groups[1], "lr": 0.02, "weight_decay": 0.0},
            {"params": groups[2], "lr": 0.10, "weight_decay": 5.0e-4},
            {"params": groups[3], "lr": 0.20, "weight_decay": 0.0},
        ],
        lr=0.01,
        weight_decay=5.0e-4,
        max_step=max_step,
    )


def official_loss(outputs, labels):
    return sum(
        weight * F.multilabel_soft_margin_loss(output, labels)
        for weight, output in zip(LOSS_WEIGHTS, outputs[:4])
    )


def parameter_summary(model):
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "dda": int(
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name.startswith("dross_disposal.")
            )
        ),
    }


def dda_macs(height=28, width=28):
    return int(
        height * width * (512 * 128 + 128 * 3 * 3 + 128 * 512)
    )


def identity_audit(pretrained):
    torch.manual_seed(42)
    baseline = Net(4, rddr_phase1_mode="none")
    torch.manual_seed(42)
    dd = Net(4, rddr_phase1_mode="dd")
    baseline_audit = load_pretrained(baseline, pretrained)
    dd_audit = load_pretrained(dd, pretrained)
    baseline.cuda().eval()
    dd.cuda().eval()
    image = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.inference_mode():
        expected = baseline(image)
        actual = dd(image)
    differences = [
        float((left - right).abs().max().item())
        for left, right in zip(expected, actual)
    ]
    del baseline, dd, image, expected, actual
    torch.cuda.empty_cache()
    return {
        "max_abs_diff": max(differences),
        "output_max_abs_diff": differences,
        "pass": max(differences) < 1.0e-5,
        "baseline_pretrained": baseline_audit,
        "dd_pretrained": dd_audit,
    }


def run_variant(mode, pretrained, batch_size, steps):
    torch.manual_seed(42)
    model = Net(4, rddr_phase1_mode=mode).cuda()
    pretrained_audit = load_pretrained(model, pretrained)
    model.train()
    optimizer = build_optimizer(model)
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    membership = {
        name: sum(candidate is parameter for candidate in grouped)
        for name, parameter in model.named_parameters()
        if name.startswith("dross_disposal.")
    }
    if not membership or any(count != 1 for count in membership.values()):
        raise AssertionError(f"Invalid DDA optimizer membership: {membership}")
    image = torch.randn(batch_size, 3, 224, 224, device="cuda")
    label = torch.randint(0, 2, (batch_size, 4), device="cuda").float()
    torch.cuda.reset_peak_memory_stats()
    records = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(image)
            loss = official_loss(outputs, label)
        loss.backward()
        gradients = {
            name: float(parameter.grad.float().norm().item())
            if parameter.grad is not None
            else 0.0
            for name, parameter in model.named_parameters()
            if name.startswith("dross_disposal.")
        }
        optimizer.step()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            diagnostics = model.forward_rddr_diagnostics(image[:1])
        q = diagnostics["q"].float()
        records.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "finite": bool(torch.isfinite(loss).item()),
                "q_requires_grad": bool(diagnostics["q"].requires_grad),
                "q_min": float(q.min().item()),
                "q_max": float(q.max().item()),
                "expand_weight_grad_norm": gradients[
                    "dross_disposal.expand.weight"
                ],
                "reduce_weight_grad_norm": gradients[
                    "dross_disposal.reduce.weight"
                ],
                "depthwise_weight_grad_norm": gradients[
                    "dross_disposal.depthwise.weight"
                ],
            }
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = {
        "mode": mode,
        "batch_size": batch_size,
        "steps": steps,
        "precision": "BF16 autocast",
        "finite": all(record["finite"] for record in records),
        "optimizer_membership": membership,
        "optimizer_groups": [
            {
                "index": index,
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "momentum": float(group["momentum"]),
                "parameters": int(sum(parameter.numel() for parameter in group["params"])),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
        "parameters": parameter_summary(model),
        "dda_macs_28x28": dda_macs(),
        "runtime_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "pretrained": pretrained_audit,
        "records": records,
    }
    del model, optimizer, image, label, diagnostics
    torch.cuda.empty_cache()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RDDR Phase-1 smoke requires CUDA")
    result = {
        "pretrained": args.pretrained,
        "pretrained_sha256": sha256_file(args.pretrained),
        "identity": identity_audit(args.pretrained),
        "variants": {
            mode: run_variant(mode, args.pretrained, args.batch_size, args.steps)
            for mode in ("uc", "dd")
        },
    }
    result["pass"] = bool(
        result["identity"]["pass"]
        and all(row["finite"] for row in result["variants"].values())
        and all(
            not record["q_requires_grad"]
            for row in result["variants"].values()
            for record in row["records"]
        )
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
