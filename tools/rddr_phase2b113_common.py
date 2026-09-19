"""Shared, side-effect-free math for the Phase-2B1.13 attribution audit.

The formal runner may read checkpoints, but this module never writes a model,
constructs a training loop, or calls an optimizer on the audited parameters.
All vector statistics are accumulated in FP64 on CPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from tools.rddr_phase2b112_common import EPS, FINAL_LRS, UPSTREAM


PREFIX = "rddr_phase2b113_"
A0 = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
CHECKPOINT_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
LAMBDA_ADT = 0.027074256246554088
TRAIN_BATCHES = 128
DIAGNOSTIC_BATCHES = 32
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
POPULATION_NAMES = ("Deep-Win_0", "Shallow-Win_0", "Both-Wrong_0", "Stable-Correct_0")
HISTORICAL_DECISIONS = {
    "phase2b19": "ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE",
    "phase2b110": "RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED",
    "phase2b111": "THIRD_EVIDENCE_OPERATIONAL_HEADROOM_INSUFFICIENT",
    "phase2b112": "ADT_LOCAL_SIGNAL_NOT_TRANSLATING_TO_OPTIMIZATION",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Iterable[Mapping]) -> None:
    rows = list(rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: clean(value) for key, value in row.items()} for row in rows)


def approved(name: str) -> bool:
    return name.split(".", 1)[0] in UPSTREAM


def block_name(name: str) -> str:
    block = name.split(".", 1)[0]
    require(block in UPSTREAM, f"Unapproved parameter: {name}")
    return block


def parameter_manifest(model, optimizer_groups: Sequence[Mapping]) -> tuple[list[dict], list[str]]:
    parameter_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    group_by_name = {}
    for index, group in enumerate(optimizer_groups):
        for parameter in group["params"]:
            group_by_name[parameter_by_id[id(parameter)]] = {
                "optimizer_group": index,
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "momentum": float(group["momentum"]),
                "dampening": float(group["dampening"]),
                "nesterov": bool(group["nesterov"]),
            }
    selected = [(name, parameter) for name, parameter in model.named_parameters() if approved(name)]
    require(len(selected) == 39, f"Approved parameter count must be 39, got {len(selected)}")
    names = [name for name, _ in selected]
    total = sum(parameter.numel() for _, parameter in selected)
    rows = []
    for index, (name, parameter) in enumerate(selected):
        require(name in group_by_name, f"Approved parameter absent from optimizer: {name}")
        rows.append({
            "manifest_index": index,
            "exact_name": name,
            "block": block_name(name),
            "shape": "x".join(map(str, parameter.shape)),
            "numel": parameter.numel(),
            "total_numel": total,
            **group_by_name[name],
        })
    require(sum(row["numel"] for row in rows) == total == 27_275_776,
            f"Unexpected approved numel: {total}")
    return rows, names


def cpu_grad_map(names: Sequence[str], gradients: Sequence[torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    require(len(names) == len(gradients), "Gradient/name length mismatch")
    result = {}
    for name, gradient in zip(names, gradients):
        if gradient is None:
            result[name] = None
        else:
            require(gradient.dtype == torch.float32, f"Gradient is not FP32: {name}/{gradient.dtype}")
            require(bool(torch.isfinite(gradient).all()), f"Nonfinite gradient: {name}")
            result[name] = gradient.detach().cpu()
    return result


def scale_gradients(gradients: Mapping[str, torch.Tensor | None], scalar: float):
    return {name: None if value is None else value * scalar for name, value in gradients.items()}


def add_gradients(left: Mapping[str, torch.Tensor | None],
                  right: Mapping[str, torch.Tensor | None], right_scale: float = 1.0):
    require(left.keys() == right.keys(), "Gradient maps use different parameter manifests")
    result = {}
    for name in left:
        a, b = left[name], right[name]
        if a is None and b is None:
            result[name] = None
        elif a is None:
            result[name] = b * right_scale
        elif b is None:
            result[name] = a.clone()
        else:
            result[name] = a + b * right_scale
    return result


def subset(gradients: Mapping[str, torch.Tensor | None], block: str):
    return {name: value for name, value in gradients.items() if block_name(name) == block}


def vector_dot(left: Mapping[str, torch.Tensor | None], right: Mapping[str, torch.Tensor | None]) -> float:
    require(left.keys() == right.keys(), "Vector maps use different manifests")
    total = 0.0
    for name in left:
        if left[name] is not None and right[name] is not None:
            total += float((left[name].double() * right[name].double()).sum())
    return total


def vector_norm(vector: Mapping[str, torch.Tensor | None]) -> float:
    return math.sqrt(max(0.0, vector_dot(vector, vector)))


def vector_cosine(left: Mapping[str, torch.Tensor | None], right: Mapping[str, torch.Tensor | None]) -> float:
    denominator = vector_norm(left) * vector_norm(right)
    return vector_dot(left, right) / denominator if denominator > 0 else float("nan")


def direction_difference(left: Mapping[str, torch.Tensor | None],
                         right: Mapping[str, torch.Tensor | None]) -> tuple[float, float]:
    cosine = float(np.clip(vector_cosine(left, right), -1.0, 1.0))
    direct = math.sqrt(max(0.0, 2.0 * (1.0 - cosine)))
    return direct, math.sqrt(max(0.0, 2.0 * (1.0 - cosine)))


def relative_difference(left: Mapping[str, torch.Tensor | None],
                        right: Mapping[str, torch.Tensor | None]) -> float:
    difference = add_gradients(left, right, -1.0)
    return vector_norm(difference) / (0.5 * (vector_norm(left) + vector_norm(right)) + EPS)


def norm_ratio(left: Mapping[str, torch.Tensor | None], right: Mapping[str, torch.Tensor | None]) -> float:
    return vector_norm(left) / (vector_norm(right) + EPS)


def zero_aggregate(template: Mapping[str, torch.Tensor | None]):
    return {name: torch.zeros_like(value, dtype=torch.float64) if value is not None else None
            for name, value in template.items()}


def accumulate(aggregate: dict[str, torch.Tensor | None], gradients: Mapping[str, torch.Tensor | None]) -> None:
    require(aggregate.keys() == gradients.keys(), "Aggregate/gradient manifests differ")
    for name, value in gradients.items():
        if value is None:
            continue
        if aggregate[name] is None:
            aggregate[name] = torch.zeros_like(value, dtype=torch.float64)
        aggregate[name].add_(value.double())


def max_relative_error(actual: Mapping[str, torch.Tensor | None],
                       expected: Mapping[str, torch.Tensor | None]) -> float:
    difference = vector_norm(add_gradients(actual, expected, -1.0))
    return difference / (vector_norm(expected) + EPS)


def optimizer_specs(manifest_rows: Sequence[Mapping]) -> dict[str, dict]:
    return {str(row["exact_name"]): {
        key: row[key] for key in ("optimizer_group", "lr", "weight_decay", "momentum", "dampening", "nesterov")
    } for row in manifest_rows}


def virtual_fresh_update(parameters: Mapping[str, torch.Tensor],
                         gradients: Mapping[str, torch.Tensor | None],
                         specs: Mapping[str, Mapping]) -> dict[str, torch.Tensor | None]:
    """Exact first fresh-state SGD displacement on the approved parameter space.

    PyTorch SGD skips a parameter entirely when ``grad is None``. With an empty
    state and nesterov=False, the first momentum buffer equals the weight-decayed
    gradient.  The returned displacement deliberately includes the FP32 rounding
    of PyTorch's in-place ``p.add_(direction, alpha=-lr)``; this matters at the
    frozen final learning rate near 1e-6.
    """
    require(parameters.keys() == gradients.keys() == specs.keys(), "Virtual optimizer manifests differ")
    result = {}
    for name in parameters:
        gradient = gradients[name]
        if gradient is None:
            result[name] = None
            continue
        spec = specs[name]
        require(not spec["nesterov"], "Frozen optimizer unexpectedly uses nesterov")
        require(float(spec["dampening"]) == 0.0, "Frozen optimizer dampening changed")
        parameter = parameters[name].float()
        direction = gradient.float() + float(spec["weight_decay"]) * parameter
        after = torch.add(parameter, direction, alpha=-float(spec["lr"]))
        result[name] = after - parameter
    return result


def dry_run_clone_error(parameters: Mapping[str, torch.Tensor],
                        gradients: Mapping[str, torch.Tensor | None],
                        specs: Mapping[str, Mapping]) -> float:
    """Validate the analytical transform on isolated clones only."""
    analytical = virtual_fresh_update(parameters, gradients, specs)
    clones = {name: torch.nn.Parameter(value.detach().clone()) for name, value in parameters.items()}
    before = {name: value.detach().clone() for name, value in clones.items()}
    groups = []
    for group_index in sorted({int(spec["optimizer_group"]) for spec in specs.values()}):
        names = [name for name, spec in specs.items() if int(spec["optimizer_group"]) == group_index]
        exemplar = specs[names[0]]
        groups.append({
            "params": [clones[name] for name in names],
            "lr": float(exemplar["lr"]),
            "weight_decay": float(exemplar["weight_decay"]),
            "momentum": float(exemplar["momentum"]),
            "dampening": float(exemplar["dampening"]),
            "nesterov": bool(exemplar["nesterov"]),
        })
    optimizer = torch.optim.SGD(groups, lr=1.0)
    for name, parameter in clones.items():
        parameter.grad = None if gradients[name] is None else gradients[name].detach().clone()
    optimizer.step()  # Isolated clones; never the formal model or formal optimizer.
    actual = {name: clones[name].detach() - before[name] for name in clones}
    error = max_relative_error(actual, analytical)
    del optimizer, clones, before, actual
    return error


def vector_metrics(g_main, g_ctx, g_rnd, parameters, specs, lambda_value=LAMBDA_ADT):
    d_dir, d_formula = direction_difference(g_ctx, g_rnd)
    total_a = add_gradients(g_main, g_ctx, lambda_value)
    total_r = add_gradients(g_main, g_rnd, lambda_value)
    update_a = virtual_fresh_update(parameters, total_a, specs)
    update_r = virtual_fresh_update(parameters, total_r, specs)
    return {
        "C_aux": vector_cosine(g_ctx, g_rnd),
        "D_dir": d_dir,
        "D_dir_formula": d_formula,
        "R_norm": norm_ratio(g_ctx, g_rnd),
        "C_ctx_main": vector_cosine(g_ctx, g_main),
        "C_rnd_main": vector_cosine(g_rnd, g_main),
        "DeltaC_main": vector_cosine(g_ctx, g_main) - vector_cosine(g_rnd, g_main),
        "C_tot": vector_cosine(total_a, total_r),
        "rho_ctx": vector_norm(scale_gradients(add_gradients(g_ctx, g_rnd, -1.0), lambda_value)) /
                   (0.5 * (vector_norm(total_a) + vector_norm(total_r)) + EPS),
        "C_update": vector_cosine(update_a, update_r),
        "rho_update": relative_difference(update_a, update_r),
        "main_norm": vector_norm(g_main),
        "ctx_norm": vector_norm(g_ctx),
        "rnd_norm": vector_norm(g_rnd),
        "update_a_norm": vector_norm(update_a),
        "update_r_norm": vector_norm(update_r),
    }, update_a, update_r


def oracle_metrics(g_ctx, g_rnd, g_oracle, update_a, update_r):
    oracle_sq = vector_dot(g_oracle, g_oracle)
    c_ctx = vector_cosine(g_ctx, g_oracle)
    c_rnd = vector_cosine(g_rnd, g_oracle)
    p_ctx = vector_dot(g_ctx, g_oracle) / (oracle_sq + EPS)
    p_rnd = vector_dot(g_rnd, g_oracle) / (oracle_sq + EPS)
    improvement_a = -vector_dot(g_oracle, update_a)
    improvement_r = -vector_dot(g_oracle, update_r)
    return {
        "C_ctx_oracle": c_ctx,
        "C_rnd_oracle": c_rnd,
        "DeltaC_oracle": c_ctx - c_rnd,
        "P_ctx": p_ctx,
        "P_rnd": p_rnd,
        "DeltaP_oracle": p_ctx - p_rnd,
        "I_A": improvement_a,
        "I_R": improvement_r,
        "Adv_oracle": improvement_a - improvement_r,
    }


def cancellation_index(vectors: Sequence[Mapping[str, torch.Tensor | None]]) -> float:
    require(len(vectors) > 0, "Cancellation index needs vectors")
    total = vectors[0]
    for vector in vectors[1:]:
        total = add_gradients(total, vector)
    return 1.0 - vector_norm(total) / (sum(vector_norm(vector) for vector in vectors) + EPS)


def bootstrap(rows: Sequence[Mapping[str, float]], specifications: Sequence[tuple[str, str]]) -> list[dict]:
    """Paired minibatch percentile bootstrap; one shared seed42 resampling matrix."""
    require(len(rows) > 0, "Bootstrap needs minibatches")
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, len(rows), size=(BOOTSTRAP_REPLICATES, len(rows)))
    output = []
    for endpoint, statistic in specifications:
        values = np.asarray([float(row[endpoint]) for row in rows], dtype=np.float64)
        sampled = values[indices]
        estimates = sampled.mean(axis=1) if statistic == "mean" else np.median(sampled, axis=1)
        point = float(values.mean()) if statistic == "mean" else float(np.median(values))
        low, high = np.quantile(estimates, (0.025, 0.975))
        output.append({
            "endpoint": endpoint,
            "statistic": statistic,
            "point": point,
            "ci_low": float(low),
            "ci_high": float(high),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "statistical_unit": "minibatch",
            "n_minibatches": len(rows),
            "method": "paired_percentile",
        })
    return output
