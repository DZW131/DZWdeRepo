"""Gradient and parameter-health helpers for the frozen Phase-0 audit."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference, memory_format=torch.preserve_format)


def gradient_decomposition(
    losses: Mapping[str, torch.Tensor],
    representation: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    weights: Mapping[str, float],
    objective_names: Sequence[str] = ("sem", "eq", "orth", "rec"),
    eps: float = 1e-12,
) -> Tuple[list[dict], list[dict]]:
    """Measure independent objective gradients without populating ``.grad``."""

    gradients_h: Dict[str, torch.Tensor] = {}
    parameter_norms: Dict[str, float] = {}
    parameter_finite: Dict[str, bool] = {}
    targets = (representation, *parameters)
    for name in ("base", *objective_names):
        gradients = torch.autograd.grad(
            losses[name],
            targets,
            retain_graph=True,
            allow_unused=True,
        )
        grad_h = gradients[0]
        if grad_h is None:
            grad_h = _zero_like(representation)
        gradients_h[name] = grad_h.detach().float()
        squared = torch.zeros((), device=representation.device, dtype=torch.float64)
        all_parameter_gradients_finite = True
        for gradient in gradients[1:]:
            if gradient is not None:
                all_parameter_gradients_finite = (
                    all_parameter_gradients_finite
                    and bool(torch.isfinite(gradient).all())
                )
                squared = squared + gradient.detach().double().square().sum()
        parameter_norms[name] = float(squared.sqrt().cpu())
        parameter_finite[name] = all_parameter_gradients_finite and math.isfinite(
            parameter_norms[name]
        )

    base = gradients_h["base"].reshape(-1)
    base_norm = float(torch.linalg.vector_norm(base).cpu())
    ratio_rows = []
    cosine_rows = []
    for name in objective_names:
        gradient = gradients_h[name].reshape(-1)
        norm = float(torch.linalg.vector_norm(gradient).cpu())
        weight = float(weights[name])
        ratio = weight * norm / (base_norm + eps)
        denominator = torch.linalg.vector_norm(base) * torch.linalg.vector_norm(
            gradient
        )
        if float(denominator.cpu()) <= eps:
            cosine = 0.0
        else:
            cosine = float(torch.dot(base, gradient).div(denominator).cpu())
        ratio_rows.append(
            {
                "objective": name,
                "base_grad_norm_h": base_norm,
                "raw_grad_norm_h": norm,
                "weight": weight,
                "weighted_grad_norm_h": weight * norm,
                "ratio": ratio,
                "base_grad_norm_osmf_parameters": parameter_norms["base"],
                "objective_grad_norm_osmf_parameters": parameter_norms[name],
                "finite": bool(
                    torch.isfinite(gradients_h[name]).all()
                    and torch.isfinite(gradients_h["base"]).all()
                    and parameter_finite[name]
                    and parameter_finite["base"]
                ),
            }
        )
        cosine_rows.append(
            {
                "objective": name,
                "cosine": cosine,
                "finite": bool(torch.isfinite(torch.tensor(cosine))),
            }
        )
    return ratio_rows, cosine_rows


def snapshot_parameters(
    module: torch.nn.Module, names: Iterable[str]
) -> Dict[str, torch.Tensor]:
    selected = dict(module.named_parameters())
    return {name: selected[name].detach().float().cpu().clone() for name in names}


def parameter_gradient_rows(
    module: torch.nn.Module, names: Iterable[str]
) -> list[dict]:
    selected = dict(module.named_parameters())
    rows = []
    for name in names:
        parameter = selected[name]
        gradient = parameter.grad
        if gradient is None:
            norm = 0.0
            nonzero_fraction = 0.0
            finite = True
        else:
            detached = gradient.detach().float()
            norm = float(torch.linalg.vector_norm(detached).cpu())
            nonzero_fraction = float(torch.count_nonzero(detached).cpu()) / detached.numel()
            finite = bool(torch.isfinite(detached).all())
        rows.append(
            {
                "parameter": name,
                "grad_norm": norm,
                "nonzero_grad_fraction": nonzero_fraction,
                "finite": finite,
            }
        )
    return rows


def parameter_update_rows(
    module: torch.nn.Module,
    names: Iterable[str],
    initial: Mapping[str, torch.Tensor],
    before_step: Mapping[str, torch.Tensor],
    eps: float = 1e-12,
) -> list[dict]:
    selected = dict(module.named_parameters())
    rows = []
    for name in names:
        current = selected[name].detach().float().cpu()
        step_update = torch.linalg.vector_norm(current - before_step[name])
        cumulative = torch.linalg.vector_norm(current - initial[name])
        initial_norm = torch.linalg.vector_norm(initial[name])
        rows.append(
            {
                "parameter": name,
                "step_update_norm": float(step_update),
                "cumulative_update_norm": float(cumulative),
                "initial_parameter_norm": float(initial_norm),
                "relative_update_norm": float(cumulative / (initial_norm + eps)),
                "finite": bool(torch.isfinite(current).all()),
            }
        )
    return rows


def max_consecutive(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
