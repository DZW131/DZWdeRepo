"""Shared semantic projector and compatibility policy for SC-MPR."""

import math

import torch
import torch.nn as nn


def logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    return math.log(probability / (1.0 - probability))


class SharedSCMPRPolicy(nn.Module):
    """One deep projector and one gate policy shared by all HFRM stages."""

    def __init__(
        self,
        deep_channels: int = 4096,
        projection_dim: int = 32,
        condition_channels: int = 6,
        hidden_channels: int = 16,
        gate_init: float = 0.1,
    ):
        super().__init__()
        self.deep_projector = nn.Conv2d(
            deep_channels, projection_dim, kernel_size=1, bias=True
        )
        self.gate_policy = nn.Sequential(
            nn.Conv2d(
                condition_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=True),
        )
        final_layer = self.gate_policy[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, logit(gate_init))

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        return self.gate_policy(conditions)
