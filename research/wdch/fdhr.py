"""Minimal fixed-strength cross-band interactions for EXP-FDHR-003.

The three operators implement the frozen Phase-3 equations at HFRM28_1.
They intentionally add no trainable parameters beyond the WD-CH LL depthwise
convolution already present in W1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .haar_wavelet import FixedHaarDWT2D


VALID_VARIANTS = ("A", "B", "C")


class CrossBandWaveletContext(nn.Module):
    """Apply one frozen cross-band interaction after a Haar decomposition.

    Variant C's ``Pool(HF)`` is made dimensionally explicit as mean pooling
    over the three high-frequency band axis.  Haar coefficients are already
    at LL spatial resolution, so no spatial resize or second downsampling is
    introduced.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        variant: str = "A",
        strength: float = 0.1,
    ) -> None:
        super().__init__()
        variant = str(variant).upper()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unknown FDHR variant: {variant}")
        if kernel_size not in (5, 7, 9):
            raise ValueError("Phase-0-locked WD-CH kernels are 5, 7, or 9")
        if float(strength) != 0.1:
            raise ValueError("EXP-FDHR-003 freezes interaction strength at 0.1")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.variant = variant
        self.haar = FixedHaarDWT2D()
        self.ll_context = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        nn.init.constant_(self.ll_context.weight, 1.0 / (kernel_size**2))
        self.register_buffer("strength", torch.tensor(float(strength)))

    def _interact(self, ll_rectified, lh, hl, hh):
        strength = self.strength.to(dtype=ll_rectified.dtype)
        if self.variant == "A":
            return ll_rectified, (
                (1.0 + strength) * lh,
                (1.0 + strength) * hl,
                (1.0 + strength) * hh,
            )
        if self.variant == "B":
            hf_magnitude = lh.abs() + hl.abs() + hh.abs()
            delta_ll = strength * hf_magnitude * ll_rectified
        else:
            # ``Pool(HF)`` = arithmetic mean over the three-band axis.  The
            # coefficients already share LL's H/2 x W/2 spatial resolution.
            hf_down = torch.stack((lh, hl, hh), dim=0).mean(dim=0)
            delta_ll = strength * hf_down
        return ll_rectified + delta_ll, (lh, hl, hh)

    def forward_with_diagnostics(self, x: torch.Tensor):
        ll, lh, hl, hh = self.haar.dwt(x)
        ll_rectified = self.ll_context(ll)
        base = self.haar.idwt(ll_rectified, lh, hl, hh)
        ll_prime, high_prime = self._interact(ll_rectified, lh, hl, hh)
        output = self.haar.idwt(ll_prime, *high_prime)
        interaction = output - base
        diagnostics = {
            "E_LL": ll.detach().float().square().mean(),
            "E_HF": (
                lh.detach().float().square()
                + hl.detach().float().square()
                + hh.detach().float().square()
            ).mean(),
            "interaction_rms": interaction.detach().float().square().mean().sqrt(),
            "interaction_input_rms": interaction.detach().float().square().mean().sqrt()
            / x.detach().float().square().mean().sqrt().clamp_min(1.0e-12),
        }
        return output, diagnostics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_diagnostics(x)[0]


class HFRMFDHR(nn.Module):
    """Original HFRM equation with one Phase-3 operator at HFRM28_1."""

    def __init__(
        self,
        in_channels: int,
        deep_channels: int = 4096,
        kernel_size: int = 7,
        variant: str = "A",
    ) -> None:
        super().__init__()
        self.veto_mlp = nn.Sequential(
            nn.Linear(deep_channels, deep_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(deep_channels // 8, in_channels, bias=False),
            nn.Sigmoid(),
        )
        # Retain the ``wdch`` attribute so the parameter names and optimizer
        # grouping remain matched to W1.
        self.wdch = CrossBandWaveletContext(
            in_channels, kernel_size=kernel_size, variant=variant, strength=0.1
        )
        self.gamma_veto = nn.Parameter(torch.zeros(1))
        self.gamma_context = nn.Parameter(torch.zeros(1))

    def forward(self, feat_nong: torch.Tensor, feat_deep: torch.Tensor):
        batch, channels = feat_nong.shape[:2]
        global_dna = F.adaptive_avg_pool2d(feat_deep, 1).view(batch, -1)
        veto_weights = self.veto_mlp(global_dna).view(batch, channels, 1, 1)
        feat_vetoed = feat_nong * veto_weights
        feat_smoothed = self.wdch(feat_nong)
        return (
            feat_nong
            + self.gamma_veto * feat_vetoed
            + self.gamma_context * feat_smoothed
        )
