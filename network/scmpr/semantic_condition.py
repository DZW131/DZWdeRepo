"""Read-only deep-semantic conditions for stage-local SC-MPR policies."""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.scmpr.frequency_proposal import fixed_lowpass


class StageSemanticCondition(nn.Module):
    """Build confidence, uncertainty, variation, and compatibility maps."""

    def __init__(
        self,
        target_channels: int,
        projection_dim: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.target_projector = nn.Conv2d(
            target_channels, projection_dim, kernel_size=1, bias=True
        )
        self.eps = float(eps)

    @staticmethod
    def _resize(tensor: torch.Tensor, spatial_size) -> torch.Tensor:
        if tensor.shape[-2:] == tuple(spatial_size):
            return tensor
        return F.interpolate(
            tensor, size=spatial_size, mode="bilinear", align_corners=False
        )

    def forward(
        self,
        target_feature: torch.Tensor,
        deep_feature: torch.Tensor,
        deep_cam_logits: torch.Tensor,
        deep_projector: nn.Conv2d,
    ) -> Dict[str, torch.Tensor]:
        spatial_size = target_feature.shape[-2:]

        # The semantic anchor and backbone features are read-only. Projector
        # weights remain learnable because detach is applied only to inputs.
        logits = deep_cam_logits.detach()
        probabilities = torch.softmax(logits.float(), dim=1).to(logits.dtype)
        probabilities = self._resize(probabilities, spatial_size)
        confidence = probabilities.max(dim=1, keepdim=True).values

        class_count = probabilities.shape[1]
        entropy = -(
            probabilities.float()
            * torch.log(probabilities.float().clamp_min(self.eps))
        ).sum(dim=1, keepdim=True)
        if class_count > 1:
            entropy = entropy / math.log(class_count)
        uncertainty = entropy.clamp(0.0, 1.0).to(probabilities.dtype)

        probability_lowpass = fixed_lowpass(probabilities, 3)
        variation = (
            probabilities - probability_lowpass
        ).abs().mean(dim=1, keepdim=True)

        target_projection = self.target_projector(target_feature.detach())
        deep_projection = deep_projector(deep_feature.detach())
        deep_projection = self._resize(deep_projection, spatial_size)
        target_projection = F.normalize(
            target_projection, p=2, dim=1, eps=self.eps
        )
        deep_projection = F.normalize(
            deep_projection, p=2, dim=1, eps=self.eps
        )
        compatibility = (
            1.0 + (target_projection * deep_projection).sum(dim=1, keepdim=True)
        ) * 0.5
        compatibility = compatibility.clamp(0.0, 1.0)

        return {
            "probabilities": probabilities,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "variation": variation,
            "compatibility": compatibility,
            "target_projection": target_projection,
            "deep_projection": deep_projection,
        }
