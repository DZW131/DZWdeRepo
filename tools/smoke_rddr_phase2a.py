"""CUDA readiness audit for frozen RDDR Phase-2A GS/RCS."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net  # noqa: E402
from tool import torchutils  # noqa: E402


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


def optimizer_signature(optimizer):
    return [
        {
            "index": index,
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "momentum": float(group["momentum"]),
            "parameter_tensors": len(group["params"]),
            "parameters": int(sum(p.numel() for p in group["params"])),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def parameter_summary(model):
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "additional": 0,
    }


def initial_equivalence(pretrained):
    torch.manual_seed(42)
    model = Net(4, rddr_context_mode="none")
    audit = load_pretrained(model, pretrained)
    model.cuda().eval()
    image = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.inference_mode():
        model.rddr_context_mode = "none"
        expected = model(image)
        differences = {}
        for mode in ("global", "receiver"):
            model.rddr_context_mode = mode
            actual = model(image)
            differences[mode] = max(
                float((left - right).abs().max().item())
                for left, right in zip(expected, actual)
            )
    del model, image, expected, actual
    torch.cuda.empty_cache()
    return {
        "pretrained": audit,
        "max_abs_diff_at_zero_gamma": differences,
        "pass": max(differences.values()) < 1.0e-6,
    }


def run_variant(mode, pretrained, batch_size, steps):
    torch.manual_seed(42)
    model = Net(4, rddr_context_mode=mode).cuda()
    pretrained_audit = load_pretrained(model, pretrained)
    model.train()
    optimizer = build_optimizer(model)
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
        gamma_gradient = model.hfrm_28_1.gamma_context.grad
        optimizer.step()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            diagnostics = model.forward_rddr_context_diagnostics(image[:1])
        q = diagnostics["q"].float()
        reliability = diagnostics["reliability"].float()
        context_before = diagnostics["context_before"].float()
        context_after = diagnostics["context_after"].float()
        ratio = context_after.square().mean().sqrt() / context_before.square().mean().sqrt()
        records.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "finite": bool(
                    torch.isfinite(loss).item()
                    and all(torch.isfinite(value).all() for value in outputs)
                ),
                "q_requires_grad": bool(diagnostics["q"].requires_grad),
                "q_min": float(q.min().item()),
                "q_max": float(q.max().item()),
                "q_mean": float(q.mean().item()),
                "reliability_shape": list(reliability.shape),
                "reliability_mean": float(reliability.mean().item()),
                "context_rms_ratio": float(ratio.item()),
                "gamma_context": float(model.hfrm_28_1.gamma_context.item()),
                "gamma_context_grad": float(gamma_gradient.float().item()),
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
        "parameters": parameter_summary(model),
        "optimizer_groups": optimizer_signature(optimizer),
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
        raise RuntimeError("RDDR Phase-2A smoke requires CUDA")
    result = {
        "pretrained": args.pretrained,
        "pretrained_sha256": sha256_file(args.pretrained),
        "initial_equivalence": initial_equivalence(args.pretrained),
        "variants": {
            mode: run_variant(mode, args.pretrained, args.batch_size, args.steps)
            for mode in ("global", "receiver")
        },
    }
    signatures = [
        result["variants"][mode]["optimizer_groups"]
        for mode in ("global", "receiver")
    ]
    parameters = [
        result["variants"][mode]["parameters"]
        for mode in ("global", "receiver")
    ]
    result["pass"] = bool(
        result["initial_equivalence"]["pass"]
        and signatures[0] == signatures[1]
        and parameters[0] == parameters[1]
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
