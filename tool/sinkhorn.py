"""Small detached FP32 Sinkhorn solver used by MATR-v1 OT-MTR."""

from __future__ import annotations

import math

import torch


def sinkhorn_plan(
    cost: torch.Tensor, epsilon: float = 0.1, iterations: int = 20
) -> torch.Tensor:
    """Return a no-grad balanced transport plan with uniform marginals."""

    if cost.ndim != 2 or cost.shape[0] < 1 or cost.shape[1] != 2:
        raise ValueError("MATR-v1 Sinkhorn expects an Nx2 cost matrix")
    with torch.no_grad(), torch.autocast(device_type=cost.device.type, enabled=False):
        cost32 = cost.detach().float()
        log_kernel = -cost32 / float(epsilon)
        log_row_target = torch.full(
            (cost32.shape[0],), -math.log(cost32.shape[0]),
            device=cost32.device, dtype=torch.float32,
        )
        log_col_target = torch.full(
            (cost32.shape[1],), -math.log(cost32.shape[1]),
            device=cost32.device, dtype=torch.float32,
        )
        log_v = torch.zeros_like(log_col_target)
        for _ in range(iterations):
            log_u = log_row_target - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
            log_v = log_col_target - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        plan = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
    return plan.detach()


def marginal_errors(plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_target = torch.full_like(plan.sum(dim=1), 1.0 / plan.shape[0])
    col_target = torch.full_like(plan.sum(dim=0), 1.0 / plan.shape[1])
    return (
        (plan.sum(dim=1) - row_target).abs().max(),
        (plan.sum(dim=0) - col_target).abs().max(),
    )
