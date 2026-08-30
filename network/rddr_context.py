"""Frozen RDDR Phase-2A dross score and context-suppression primitives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


JS_EPSILON = 1.0e-8
JS_TEMPERATURE = 1.0
JS_MAXIMUM = math.log(2.0)
CONTEXT_MODES = frozenset({"none", "global", "receiver"})


def compute_rddr_dross_score(
    shallow_logits: torch.Tensor,
    deep_logits: torch.Tensor,
    *,
    epsilon: float = JS_EPSILON,
    temperature: float = JS_TEMPERATURE,
) -> torch.Tensor:
    """Return the detached, normalized Phase-0 Jensen-Shannon score.

    Both inputs must be raw four-class logits with identical BCHW shapes.  The
    arithmetic is deliberately performed in FP32 inside the official BF16
    autocast region and exactly matches the frozen Phase-0 expression.
    """

    if shallow_logits.shape != deep_logits.shape:
        raise ValueError(
            "RDDR shallow/deep logits must have identical shapes, got "
            f"{tuple(shallow_logits.shape)} and {tuple(deep_logits.shape)}"
        )
    if shallow_logits.ndim != 4:
        raise ValueError("RDDR logits must be BCHW tensors")
    if temperature != JS_TEMPERATURE:
        raise ValueError("RDDR Phase-2A freezes temperature at 1.0")

    p_shallow = F.softmax(shallow_logits.float() / temperature, dim=1)
    p_deep = F.softmax(deep_logits.float() / temperature, dim=1)
    midpoint = 0.5 * (p_shallow + p_deep)
    js = 0.5 * (
        p_shallow * ((p_shallow + epsilon).log() - (midpoint + epsilon).log())
    ).sum(dim=1, keepdim=True)
    js = js + 0.5 * (
        p_deep * ((p_deep + epsilon).log() - (midpoint + epsilon).log())
    ).sum(dim=1, keepdim=True)
    return (js / JS_MAXIMUM).clamp_(0.0, 1.0).detach()


def context_reliability(q: torch.Tensor, mode: str) -> torch.Tensor:
    """Return the frozen GS or RCS reliability gate for ``q``."""

    if mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown RDDR Phase-2A context mode: {mode}")
    if q.ndim != 4 or q.shape[1] != 1:
        raise ValueError("RDDR q must have shape [B,1,H,W]")
    if mode == "receiver":
        return 1.0 - q
    if mode == "global":
        return (1.0 - q).mean(dim=(-2, -1), keepdim=True)
    return torch.ones(
        (q.shape[0], 1, 1, 1), device=q.device, dtype=q.dtype
    )
