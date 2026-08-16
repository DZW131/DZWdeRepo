"""Full Frequency-Adaptive Morphology-Preserving context branch."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from network.fampr.adaptive_kernel import AdaptiveKernelSpectrum
from network.fampr.adaptive_sampler import SpatiallyAdaptiveDepthwiseSampler
from network.fampr.frequency_selection import MultiBandFrequencySelector


@dataclass(frozen=True)
class FAMPRConfig:
    """Frozen Full FA-MPR configuration for the first controlled experiment."""

    lowpass_kernels: tuple = (3, 7, 15)
    num_bands: int = 4
    band_hidden: int = 16
    morphology_eps: float = 1e-6
    morphology_smooth_kernel: int = 3
    adaptive_kernel_size: int = 3
    dilation_min: float = 1.0
    dilation_max: float = 7.0
    padding_mode: str = "border"
    align_corners: bool = True
    internal_sampling_fp32: bool = True
    adakern_reduction: int = 16
    anchor_lambda_init: float = 0.25

    def __post_init__(self):
        expected = {
            "lowpass_kernels": (3, 7, 15),
            "num_bands": 4,
            "band_hidden": 16,
            "morphology_eps": 1e-6,
            "morphology_smooth_kernel": 3,
            "adaptive_kernel_size": 3,
            "dilation_min": 1.0,
            "dilation_max": 7.0,
            "padding_mode": "border",
            "align_corners": True,
            "internal_sampling_fp32": True,
            "adakern_reduction": 16,
            "anchor_lambda_init": 0.25,
        }
        if asdict(self) != expected:
            raise ValueError(
                "The first Full FA-MPR experiment uses the frozen v1.0 config; "
                "configuration tuning is disabled"
            )

    @classmethod
    def from_value(
        cls, value: Optional[Union["FAMPRConfig", Mapping[str, Any]]]
    ) -> "FAMPRConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(f"Unsupported FAMPR config: {type(value)!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FrequencyAdaptiveMorphologyContext(nn.Module):
    """Frequency-adaptive context with the original CH15 as a stable anchor."""

    def __init__(
        self,
        channels: int,
        config: Optional[Union[FAMPRConfig, Mapping[str, Any]]] = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.config = FAMPRConfig.from_value(config)
        self.frequency_selector = MultiBandFrequencySelector(
            lowpass_kernels=self.config.lowpass_kernels,
            band_hidden=self.config.band_hidden,
            morphology_eps=self.config.morphology_eps,
            morphology_smooth_kernel=self.config.morphology_smooth_kernel,
        )
        self.adaptive_kernel = AdaptiveKernelSpectrum(
            channels=self.channels,
            reduction=self.config.adakern_reduction,
        )
        self.adaptive_sampler = SpatiallyAdaptiveDepthwiseSampler(
            padding_mode=self.config.padding_mode,
            align_corners=self.config.align_corners,
            internal_sampling_fp32=self.config.internal_sampling_fp32,
        )
        anchor_logit = math.log(
            self.config.anchor_lambda_init
            / (1.0 - self.config.anchor_lambda_init)
        )
        self.anchor_logit = nn.Parameter(torch.tensor([anchor_logit]))

    @property
    def anchor_lambda(self) -> torch.Tensor:
        return torch.sigmoid(self.anchor_logit)

    @staticmethod
    def _relative_norm(delta: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        delta_norm = delta.float().norm()
        anchor_norm = anchor.float().norm().clamp_min(1e-12)
        return delta_norm / anchor_norm

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
        values = tensor.detach().float().reshape(-1)
        return {
            "mean": values.mean().item(),
            "std": values.std(unbiased=False).item(),
            "min": values.min().item(),
            "max": values.max().item(),
            "p10": torch.quantile(values, 0.10).item(),
            "p50": torch.quantile(values, 0.50).item(),
            "p90": torch.quantile(values, 0.90).item(),
        }

    @classmethod
    def summarize_diagnostics(
        cls, diagnostics: Mapping[str, torch.Tensor]
    ) -> Dict[str, Any]:
        return {
            "morphology": cls._tensor_stats(diagnostics["morphology_map"]),
            "dilation": cls._tensor_stats(diagnostics["dilation_map"]),
            "band_weights": cls._tensor_stats(diagnostics["band_weights"]),
            "kernel_gate_low": cls._tensor_stats(
                diagnostics["kernel_gate_low"]
            ),
            "kernel_gate_high": cls._tensor_stats(
                diagnostics["kernel_gate_high"]
            ),
            "anchor_lambda": diagnostics["anchor_lambda"].detach().float().item(),
            "adaptive_vs_ch_ratio": diagnostics[
                "adaptive_vs_ch_ratio"
            ].detach().float().item(),
            "fampr_vs_ch_ratio": diagnostics[
                "fampr_vs_ch_ratio"
            ].detach().float().item(),
            "all_finite": bool(diagnostics["all_finite"]),
        }

    def forward(
        self,
        feature: torch.Tensor,
        original_ch: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        if feature.shape != original_ch.shape:
            raise ValueError(
                "feature and original_ch must have identical shapes, got "
                f"{tuple(feature.shape)} and {tuple(original_ch.shape)}"
            )
        x_fs, morphology, frequency_diagnostics = self.frequency_selector(feature)
        dilation = self.config.dilation_min + (1.0 - morphology) * (
            self.config.dilation_max - self.config.dilation_min
        )
        kernel_low, kernel_high, gate_low, gate_high = self.adaptive_kernel(x_fs)
        y_low, y_high = self.adaptive_sampler(
            x_fs, dilation, kernel_low, kernel_high
        )
        adaptive_context = gate_low * y_low + gate_high * y_high
        anchor_lambda = self.anchor_lambda
        fampr_context = original_ch + anchor_lambda * (
            adaptive_context - original_ch
        )

        if not return_diagnostics:
            return fampr_context

        adaptive_vs_ch_ratio = self._relative_norm(
            adaptive_context - original_ch, original_ch
        )
        fampr_vs_ch_ratio = self._relative_norm(
            fampr_context - original_ch, original_ch
        )
        finite_tensors = (
            x_fs,
            morphology,
            dilation,
            kernel_low,
            kernel_high,
            gate_low,
            gate_high,
            original_ch,
            adaptive_context,
            fampr_context,
        )
        diagnostics = {
            **frequency_diagnostics,
            "morphology_map": morphology,
            "dilation_map": dilation,
            "kernel_low": kernel_low,
            "kernel_high": kernel_high,
            "kernel_gate_low": gate_low,
            "kernel_gate_high": gate_high,
            "anchor_lambda": anchor_lambda,
            "original_ch": original_ch,
            "adaptive_context": adaptive_context,
            "fampr_context": fampr_context,
            "adaptive_vs_ch_ratio": adaptive_vs_ch_ratio,
            "fampr_vs_ch_ratio": fampr_vs_ch_ratio,
            "all_finite": all(
                torch.isfinite(tensor).all().item() for tensor in finite_tensors
            ),
        }
        diagnostics["summary"] = self.summarize_diagnostics(diagnostics)
        return fampr_context, diagnostics
