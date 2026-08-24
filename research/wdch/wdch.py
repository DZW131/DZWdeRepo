"""Minimal WD-CH operator and its HFRM28_1 integration."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .haar_wavelet import FixedHaarDWT2D


VALID_BANDS = ("LH", "HL", "HH")


class WaveletDecoupledContext(nn.Module):
    """Apply contextual homogenization to LL and preserve high-frequency bands."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size not in (5, 7, 9):
            raise ValueError("Phase-0-locked WD-CH kernels are 5, 7, or 9")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
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
        self._ablated_bands: frozenset[str] = frozenset()

    @property
    def ablated_bands(self) -> tuple[str, ...]:
        return tuple(sorted(self._ablated_bands))

    def set_ablation(self, bands: Iterable[str] = ()) -> None:
        selected = frozenset(str(band).upper() for band in bands)
        invalid = selected.difference(VALID_BANDS)
        if invalid:
            raise ValueError(f"Unknown Haar bands: {sorted(invalid)}")
        self._ablated_bands = selected

    def forward_with_bands(self, x: torch.Tensor):
        ll, lh, hl, hh = self.haar.dwt(x)
        ll_rectified = self.ll_context(ll)
        high = {"LH": lh, "HL": hl, "HH": hh}
        for name in self._ablated_bands:
            high[name] = torch.zeros_like(high[name])
        output = self.haar.idwt(
            ll_rectified, high["LH"], high["HL"], high["HH"]
        )
        return output, {"LL": ll, "LH": lh, "HL": hl, "HH": hh}, ll_rectified

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_bands(x)[0]

    def identity(self, x: torch.Tensor) -> torch.Tensor:
        return self.haar.reconstruct(x)


class StrengthCalibratedWaveletContext(WaveletDecoupledContext):
    """Fixed-strength SC-WDCH: ``F + scale * (WDCH(F) - F)``.

    ``scale`` is a persistent, non-learnable buffer obtained once from the
    epoch20 training-set-only calibration audit.  The Haar transform, LL
    contextual operator and all high-frequency identity paths are unchanged.
    """

    def __init__(self, channels: int, kernel_size: int, scale: float) -> None:
        if not torch.isfinite(torch.tensor(float(scale))) or float(scale) <= 0:
            raise ValueError("SC-WDCH scale must be finite and positive")
        super().__init__(channels, kernel_size)
        self.register_buffer(
            "scale", torch.tensor(float(scale), dtype=torch.float32), persistent=True
        )

    @property
    def operator_name(self) -> str:
        return f"SC-WDCH{self.kernel_size}(s={float(self.scale):.8f})"

    def unscaled_forward_with_bands(self, x: torch.Tensor):
        return super().forward_with_bands(x)

    def forward_with_bands(self, x: torch.Tensor):
        wd_output, bands, ll_rectified = self.unscaled_forward_with_bands(x)
        output = x + self.scale.to(dtype=x.dtype) * (wd_output - x)
        return output, bands, ll_rectified


class HFRMWDCH(nn.Module):
    """Original HFRM equation with WD-CH replacing only its CH operator."""

    def __init__(
        self,
        in_channels: int,
        deep_channels: int = 4096,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.veto_mlp = nn.Sequential(
            nn.Linear(deep_channels, deep_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(deep_channels // 8, in_channels, bias=False),
            nn.Sigmoid(),
        )
        self.wdch = WaveletDecoupledContext(in_channels, kernel_size)
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


class HFRMSCWDCH(HFRMWDCH):
    """Original HFRM equation with fixed-strength SC-WDCH at its CH path."""

    def __init__(
        self,
        in_channels: int,
        deep_channels: int = 4096,
        kernel_size: int = 7,
        scale: float = 1.0,
    ) -> None:
        super().__init__(in_channels, deep_channels, kernel_size)
        self.wdch = StrengthCalibratedWaveletContext(
            in_channels, kernel_size, scale
        )
