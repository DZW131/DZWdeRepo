"""Minimal learnable Haar analysis and structural gating for LW-SHR.

The frequency path is deliberately restricted to producing a multiplicative
gate for the existing SSHR contextual residual.  It never reconstructs or
replaces the main feature path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_analysis_filters(device=None, dtype=torch.float32):
    """Return LWTformer-compatible, convolution-oriented Haar filters."""

    scale = 1.0 / math.sqrt(2.0)
    dec_lo = torch.tensor([scale, scale], device=device, dtype=dtype)
    # PyWavelets Haar dec_hi is [-s, s]; LWTformer requests flip=True.
    dec_hi = torch.tensor([scale, -scale], device=device, dtype=dtype)
    return dec_lo, dec_hi


class SharedLearnableWaveletBank(nn.Module):
    """One globally shared pair of Haar-initialized 1-D analysis filters."""

    def __init__(self, trainable: bool):
        super().__init__()
        dec_lo, dec_hi = haar_analysis_filters()
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=trainable)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=trainable)

    @property
    def trainable(self):
        return bool(self.dec_lo.requires_grad and self.dec_hi.requires_grad)

    def diagnostics(self):
        haar_lo, haar_hi = haar_analysis_filters(
            device=self.dec_lo.device, dtype=self.dec_lo.dtype
        )

        def one(current, reference):
            current_float = current.detach().float()
            reference_float = reference.detach().float()
            return {
                "values": current_float.cpu().tolist(),
                "l2_drift": float(torch.linalg.vector_norm(current_float - reference_float)),
                "norm": float(torch.linalg.vector_norm(current_float)),
                "cosine_to_haar": float(
                    F.cosine_similarity(current_float[None], reference_float[None]).item()
                ),
            }

        return {
            "dec_lo": one(self.dec_lo, haar_lo),
            "dec_hi": one(self.dec_hi, haar_hi),
            "trainable": self.trainable,
        }


class GroupedDWT2D(nn.Module):
    """Level-1 channel-wise 2-D DWT with fixed replicate padding semantics."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _kernels(dec_lo: torch.Tensor, dec_hi: torch.Tensor):
        lo = dec_lo.reshape(-1)
        hi = dec_hi.reshape(-1)
        ll = torch.outer(lo, lo)
        # Match LWTformer ordering: LL, LH, HL, HH.
        lh = torch.outer(hi, lo)
        hl = torch.outer(lo, hi)
        hh = torch.outer(hi, hi)
        return torch.stack((ll, lh, hl, hh), dim=0)[:, None]

    def forward(self, x: torch.Tensor, bank: SharedLearnableWaveletBank):
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(x.shape)}")
        if x.shape[-2] < 1 or x.shape[-1] < 1:
            raise ValueError("DWT spatial dimensions must be non-empty")

        original_dtype = x.dtype
        channels = x.shape[1]
        pad_right = int(x.shape[-1] % 2)
        pad_bottom = int(x.shape[-2] % 2)

        # The analysis is explicitly FP32 for BF16 stability.  Casting the
        # result back preserves gradients to the FP32 learnable filters.
        with torch.autocast(device_type=x.device.type, enabled=False):
            analysis_input = x.float()
            if pad_right or pad_bottom:
                analysis_input = F.pad(
                    analysis_input,
                    (0, pad_right, 0, pad_bottom),
                    mode="replicate",
                )
            kernels = self._kernels(bank.dec_lo.float(), bank.dec_hi.float())
            kernels = kernels.repeat(channels, 1, 1, 1)
            bands = F.conv2d(analysis_input, kernels, stride=2, groups=channels)
            bands = bands.reshape(
                x.shape[0], channels, 4, bands.shape[-2], bands.shape[-1]
            )
            ll, lh, hl, hh = bands.unbind(dim=2)

        return tuple(band.to(dtype=original_dtype) for band in (ll, lh, hl, hh))


def _project_then_depthwise(in_channels, reduced_channels, kernel_size, padding, gelu=True):
    layers = [
        nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True),
        nn.Conv2d(
            reduced_channels,
            reduced_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=reduced_channels,
            bias=True,
        ),
    ]
    if gelu:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class SubbandStructuralGate(nn.Module):
    """Subband-specific processor whose only output is a CH gate logit."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 16)
        self.channels = int(channels)
        self.reduced_channels = int(reduced)
        self.dwt = GroupedDWT2D()

        self.ll_branch = _project_then_depthwise(
            channels, reduced, kernel_size=3, padding=1, gelu=True
        )
        self.lh_branch = _project_then_depthwise(
            channels, reduced, kernel_size=(1, 3), padding=(0, 1), gelu=True
        )
        self.hl_branch = _project_then_depthwise(
            channels, reduced, kernel_size=(3, 1), padding=(1, 0), gelu=True
        )
        self.hh_branch = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.Conv2d(
                reduced,
                reduced,
                kernel_size=3,
                padding=1,
                groups=reduced,
                bias=True,
            ),
        )

        fused_channels = 4 * reduced
        self.fusion_depthwise = nn.Conv2d(
            fused_channels,
            fused_channels,
            kernel_size=3,
            padding=1,
            groups=fused_channels,
            bias=True,
        )
        self.output_projection = nn.Conv2d(
            fused_channels, channels, kernel_size=1, bias=True
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x, bank, return_details=False):
        ll, lh, hl, hh = self.dwt(x, bank)
        processed = (
            self.ll_branch(ll),
            self.lh_branch(lh),
            self.hl_branch(hl),
            self.hh_branch(hh),
        )
        fused = self.fusion_depthwise(torch.cat(processed, dim=1))
        logits = self.output_projection(fused)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        if not return_details:
            return logits
        return logits, {
            "subbands": {name: value for name, value in zip(("LL", "LH", "HL", "HH"), (ll, lh, hl, hh))},
            "processed_subbands": {
                name: value
                for name, value in zip(("LL", "LH", "HL", "HH"), processed)
            },
            "fused": fused,
        }
