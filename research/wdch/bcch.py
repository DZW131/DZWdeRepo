"""Boundary-aware contextual homogenization for EXP-BCCH-001."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .haar_wavelet import FixedHaarDWT2D


def _detached_boundary_map(
    haar: FixedHaarDWT2D, x: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        _, lh, hl, hh = haar.dwt(x)
        energy = torch.sqrt(lh.square() + hl.square() + hh.square())
        spatial = energy.mean(dim=1, keepdim=True)
        minimum = spatial.amin(dim=(-2, -1), keepdim=True)
        maximum = spatial.amax(dim=(-2, -1), keepdim=True)
        normalized = (spatial - minimum) / (maximum - minimum + 1.0e-6)
        boundary = F.interpolate(
            normalized,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp_(0.0, 1.0)
    return boundary.detach()


class BoundaryAwareContext(nn.Module):
    """Mix original CH15 with identity using a detached Haar boundary map.

    The frozen Phase-1 contract is:

    * E_HF = sqrt(LH^2 + HL^2 + HH^2)
    * average E_HF over channels to obtain one spatial map per image
    * spatial min-max normalization per image
    * bilinear upsampling with align_corners=False
    * alpha = 1 - B
    * output = alpha * CH(F) + (1-alpha) * F

    The boundary map is analytical and detached.  No new trainable parameter is
    introduced relative to the original HFRM context operator.
    """

    def __init__(self, channels: int, context_kernel: int = 15) -> None:
        super().__init__()
        if context_kernel != 15:
            raise ValueError("EXP-BCCH-001 freezes the original CH15 kernel")
        self.channels = int(channels)
        self.context_kernel = int(context_kernel)
        self.haar = FixedHaarDWT2D()
        self.context_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=context_kernel,
            padding=context_kernel // 2,
            groups=channels,
            bias=False,
        )
        nn.init.constant_(self.context_conv.weight, 1.0 / (context_kernel**2))

    def boundary_map(self, x: torch.Tensor) -> torch.Tensor:
        return _detached_boundary_map(self.haar, x)

    def forward_with_maps(self, x: torch.Tensor):
        context = self.context_conv(x)
        boundary = self.boundary_map(x)
        alpha = 1.0 - boundary
        output = alpha * context + (1.0 - alpha) * x
        return output, context, boundary, alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_maps(x)[0]


class HFRMBCCH(nn.Module):
    """Original HFRM equation with BC-CH replacing CH15 at one stage."""

    def __init__(
        self,
        in_channels: int,
        deep_channels: int = 4096,
        context_kernel: int = 15,
    ) -> None:
        super().__init__()
        self.veto_mlp = nn.Sequential(
            nn.Linear(deep_channels, deep_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(deep_channels // 8, in_channels, bias=False),
            nn.Sigmoid(),
        )
        self.haar = FixedHaarDWT2D()
        self.context_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=context_kernel,
            padding=context_kernel // 2,
            groups=in_channels,
            bias=False,
        )
        nn.init.constant_(self.context_conv.weight, 1.0 / (context_kernel**2))
        self.gamma_veto = nn.Parameter(torch.zeros(1))
        self.gamma_context = nn.Parameter(torch.zeros(1))

    def boundary_map(self, x: torch.Tensor) -> torch.Tensor:
        return _detached_boundary_map(self.haar, x)

    def context_with_maps(self, x: torch.Tensor):
        context = self.context_conv(x)
        boundary = self.boundary_map(x)
        alpha = 1.0 - boundary
        output = alpha * context + (1.0 - alpha) * x
        return output, context, boundary, alpha

    def forward(self, feat_nong: torch.Tensor, feat_deep: torch.Tensor):
        batch, channels = feat_nong.shape[:2]
        global_dna = F.adaptive_avg_pool2d(feat_deep, 1).view(batch, -1)
        veto_weights = self.veto_mlp(global_dna).view(batch, channels, 1, 1)
        feat_vetoed = feat_nong * veto_weights
        feat_smoothed = self.context_with_maps(feat_nong)[0]
        return (
            feat_nong
            + self.gamma_veto * feat_vetoed
            + self.gamma_context * feat_smoothed
        )
