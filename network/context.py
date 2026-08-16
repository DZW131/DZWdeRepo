"""Neutral Contextual Homogenization primitives shared by SSHR rectifiers.

This module mirrors the released SSHR CH definition exactly: a stage-specific
depthwise convolution with an odd kernel, no bias, same-size zero padding, and
uniform averaging-kernel initialization.
"""

import torch
import torch.nn as nn


def build_context_conv(in_channels: int, context_kernel: int = 15) -> nn.Conv2d:
    """Build the original SSHR Contextual Homogenization convolution."""
    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}")
    if context_kernel <= 0 or context_kernel % 2 == 0:
        raise ValueError(
            f"context_kernel must be a positive odd integer, got {context_kernel}"
        )

    context_conv = nn.Conv2d(
        in_channels,
        in_channels,
        kernel_size=context_kernel,
        padding=context_kernel // 2,
        groups=in_channels,
        bias=False,
    )
    nn.init.constant_(context_conv.weight, 1.0 / (context_kernel**2))
    return context_conv


def apply_contextual_homogenization(
    context_conv: nn.Conv2d, feature: torch.Tensor
) -> torch.Tensor:
    """Apply CH through the supplied stage-specific depthwise convolution."""
    return context_conv(feature)
