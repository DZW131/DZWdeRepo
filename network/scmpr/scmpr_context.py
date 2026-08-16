"""Semantic-Conditioned Morphology-Preserving HFRM context."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from network.scmpr.compatibility_policy import SharedSCMPRPolicy, logit
from network.scmpr.frequency_proposal import FixedFrequencyProposal
from network.scmpr.semantic_condition import StageSemanticCondition


@dataclass(frozen=True)
class SCMPRConfig:
    """Frozen v1.0 configuration; validation-driven tuning is disabled."""

    lowpass_fine_kernel: int = 3
    lowpass_morphology_kernel: int = 15
    padding_mode: str = "replicate"
    eps: float = 1e-6
    quality_clamp: float = 5.0
    projection_dim: int = 32
    condition_channels: int = 6
    policy_hidden_channels: int = 16
    gate_init: float = 0.1
    beta_max: float = 0.5
    beta_init: float = 0.1
    semantic_stop_gradient: bool = True
    residual_spatial_demean: bool = True

    def __post_init__(self):
        expected = {
            "lowpass_fine_kernel": 3,
            "lowpass_morphology_kernel": 15,
            "padding_mode": "replicate",
            "eps": 1e-6,
            "quality_clamp": 5.0,
            "projection_dim": 32,
            "condition_channels": 6,
            "policy_hidden_channels": 16,
            "gate_init": 0.1,
            "beta_max": 0.5,
            "beta_init": 0.1,
            "semantic_stop_gradient": True,
            "residual_spatial_demean": True,
        }
        if asdict(self) != expected:
            raise ValueError(
                "SC-MPR v1.0 uses one frozen configuration; tuning is disabled"
            )

    @classmethod
    def from_value(
        cls, value: Optional[Union["SCMPRConfig", Mapping[str, Any]]]
    ) -> "SCMPRConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(f"Unsupported SC-MPR config: {type(value)!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SCMPRContext(nn.Module):
    """Stage-local proposal/condition modules plus bounded residual strength."""

    def __init__(
        self,
        channels: int,
        config: Optional[Union[SCMPRConfig, Mapping[str, Any]]] = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.config = SCMPRConfig.from_value(config)
        self.frequency_proposal = FixedFrequencyProposal(
            eps=self.config.eps,
            quality_clamp=self.config.quality_clamp,
        )
        self.semantic_condition = StageSemanticCondition(
            target_channels=channels,
            projection_dim=self.config.projection_dim,
            eps=self.config.eps,
        )
        beta_fraction = self.config.beta_init / self.config.beta_max
        self.beta_logit = nn.Parameter(
            torch.tensor(logit(beta_fraction), dtype=torch.float32)
        )

    @property
    def beta(self) -> torch.Tensor:
        return self.config.beta_max * torch.sigmoid(self.beta_logit)

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
        values = tensor.detach().float().flatten()
        quantiles = torch.quantile(
            values, torch.tensor([0.05, 0.50, 0.95], device=values.device)
        )
        return {
            "mean": values.mean().item(),
            "std": values.std(unbiased=False).item(),
            "q05": quantiles[0].item(),
            "q50": quantiles[1].item(),
            "q95": quantiles[2].item(),
            "min": values.min().item(),
            "max": values.max().item(),
        }

    @classmethod
    def summarize_diagnostics(
        cls, diagnostics: Mapping[str, torch.Tensor]
    ) -> Dict[str, Any]:
        context_rms = diagnostics["original_ch"].detach().float().square().mean().sqrt()
        correction_rms = (
            diagnostics["scmpr_context"].detach().float()
            - diagnostics["original_ch"].detach().float()
        ).square().mean().sqrt()
        residual_rms = diagnostics["delta_zero_mean"].detach().float().square().mean().sqrt()
        residual_mean = diagnostics["delta_zero_mean"].detach().float().mean()
        return {
            "gate_fine": cls._tensor_stats(diagnostics["gate_fine"]),
            "gate_morphology": cls._tensor_stats(
                diagnostics["gate_morphology"]
            ),
            "quality_fine": cls._tensor_stats(diagnostics["quality_fine"]),
            "quality_morphology": cls._tensor_stats(
                diagnostics["quality_morphology"]
            ),
            "confidence": cls._tensor_stats(diagnostics["confidence"]),
            "uncertainty": cls._tensor_stats(diagnostics["uncertainty"]),
            "variation": cls._tensor_stats(diagnostics["variation"]),
            "compatibility": cls._tensor_stats(diagnostics["compatibility"]),
            "beta": diagnostics["beta"].detach().float().item(),
            "residual_rms": residual_rms.item(),
            "residual_mean": residual_mean.item(),
            "context_drift_rms": correction_rms.item(),
            "context_drift_ratio": (
                correction_rms / context_rms.clamp_min(1e-12)
            ).item(),
            "all_finite": bool(diagnostics["all_finite"]),
        }

    def forward(
        self,
        feature: torch.Tensor,
        original_ch: torch.Tensor,
        deep_feature: torch.Tensor,
        deep_cam_logits: torch.Tensor,
        shared_policy: SharedSCMPRPolicy,
        shared_semantic=None,
        return_diagnostics: bool = False,
    ):
        residuals, qualities = self.frequency_proposal(feature)
        semantic = self.semantic_condition(
            feature,
            deep_feature,
            deep_cam_logits,
            shared_policy.deep_projector,
            shared_semantic=shared_semantic,
        )
        policy_input = torch.cat(
            (
                qualities["fine"],
                qualities["morphology"],
                semantic["confidence"],
                semantic["uncertainty"],
                semantic["variation"],
                semantic["compatibility"],
            ),
            dim=1,
        )
        gate_logits = shared_policy(policy_input)
        gates = torch.sigmoid(gate_logits)
        gate_fine = gates[:, 0:1]
        gate_morphology = gates[:, 1:2]
        delta = (
            gate_fine * residuals["fine"]
            + gate_morphology * residuals["morphology"]
        )
        delta_zero_mean = delta - delta.mean(dim=(-2, -1), keepdim=True)
        scmpr_context = original_ch + self.beta * delta_zero_mean

        if not return_diagnostics:
            return scmpr_context

        finite_tensors = (
            residuals["fine"],
            residuals["morphology"],
            qualities["fine"],
            qualities["morphology"],
            semantic["confidence"],
            semantic["uncertainty"],
            semantic["variation"],
            semantic["compatibility"],
            gate_fine,
            gate_morphology,
            delta,
            delta_zero_mean,
            original_ch,
            scmpr_context,
        )
        diagnostics = {
            "residual_fine": residuals["fine"],
            "residual_morphology": residuals["morphology"],
            "quality_fine": qualities["fine"],
            "quality_morphology": qualities["morphology"],
            "confidence": semantic["confidence"],
            "uncertainty": semantic["uncertainty"],
            "variation": semantic["variation"],
            "compatibility": semantic["compatibility"],
            "gate_fine": gate_fine,
            "gate_morphology": gate_morphology,
            "beta": self.beta,
            "delta": delta,
            "delta_zero_mean": delta_zero_mean,
            "original_ch": original_ch,
            "scmpr_context": scmpr_context,
            "all_finite": all(
                bool(torch.isfinite(tensor).all().item())
                for tensor in finite_tensors
            ),
            "shared_policy_id": id(shared_policy),
        }
        diagnostics["summary"] = self.summarize_diagnostics(diagnostics)
        return scmpr_context, diagnostics
