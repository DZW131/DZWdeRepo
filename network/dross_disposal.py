"""RDDR Phase-1 spatial-semantic dross disposal primitives.

The disagreement score is deliberately analytical and detached.  The adapter
only learns which local feature component to subtract; it cannot manipulate
the shallow/deep probes through the score path.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


JS_EPSILON = 1.0e-8
JS_TEMPERATURE = 1.0
JS_MAXIMUM = math.log(2.0)


def compute_rddr_dross_score(
    shallow_logits: torch.Tensor,
    deep_logits: torch.Tensor,
    *,
    epsilon: float = JS_EPSILON,
    temperature: float = JS_TEMPERATURE,
) -> torch.Tensor:
    """Return detached normalized Jensen-Shannon disagreement ``[B,1,H,W]``."""

    if shallow_logits.shape != deep_logits.shape:
        raise ValueError(
            "RDDR shallow/deep logits must have identical shapes, got "
            f"{tuple(shallow_logits.shape)} and {tuple(deep_logits.shape)}"
        )
    if shallow_logits.ndim != 4:
        raise ValueError("RDDR logits must be BCHW tensors")
    if temperature != JS_TEMPERATURE:
        raise ValueError("RDDR Phase-1 freezes temperature at 1.0")

    # Compute the logarithms in FP32 even inside the official BF16 autocast
    # region, then cast q back to the feature dtype at the disposal site.
    shallow_probability = F.softmax(shallow_logits.float() / temperature, dim=1)
    deep_probability = F.softmax(deep_logits.float() / temperature, dim=1)
    mixture = 0.5 * (shallow_probability + deep_probability)
    js = 0.5 * (
        shallow_probability
        * ((shallow_probability + epsilon).log() - (mixture + epsilon).log())
    ).sum(dim=1, keepdim=True)
    js = js + 0.5 * (
        deep_probability
        * ((deep_probability + epsilon).log() - (mixture + epsilon).log())
    ).sum(dim=1, keepdim=True)
    return (js / JS_MAXIMUM).clamp_(0.0, 1.0).detach()


class DrossDisposalAdapter(nn.Module):
    """Predict a removable local component without changing identity at init."""

    def __init__(self, channels: int = 512, hidden_channels: int = 128):
        super().__init__()
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.reduce_activation = nn.GELU()
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.depthwise_activation = nn.GELU()
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        component = self.reduce_activation(self.reduce(feature))
        component = self.depthwise_activation(self.depthwise(component))
        return self.expand(component)
