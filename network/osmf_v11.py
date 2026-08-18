"""OSMF-v1.1 semantic-preservation factorization.

The factorization/reconstruction graph is frozen from OSMF-v1.0.  Version 1.1
removes the randomly initialized auxiliary classifier; semantic specialization
is instead supervised by the pretrained SSHR ``ic1`` response geometry.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.osmf import (
    OSMF_EQUIVARIANCE_INTERVAL,
    OSMF_LAMBDA_MORPH,
    OSMF_LAMBDA_ORTH,
    OSMF_LAMBDA_REC,
    OSMF_LAMBDA_SEM,
    _ieee_cuda_conv_precision,
    cross_subspace_covariance,
    inverse_align_morphology,
    orthogonality_loss,
    reconstruction_cosine,
    reconstruction_loss,
    spatial_equivariance_loss,
)


class OSMFV11Factorizer(nn.Module):
    """Split and exactly reconstruct post-HFRM ``H28_1``.

    The complementary 256/256 channel-selection initialization is identical to
    OSMF-v1.0, but this module has exactly four trainable projection tensors and
    no classifier, MLP, prototype, or learned semantic policy.
    """

    def __init__(self, in_channels: int = 512) -> None:
        super().__init__()
        if in_channels < 2:
            raise ValueError("OSMF requires at least two input channels")
        self.in_channels = int(in_channels)
        self.semantic_channels = (self.in_channels + 1) // 2
        self.morphology_channels = self.in_channels // 2
        self.p_sem = nn.Conv2d(
            self.in_channels, self.semantic_channels, kernel_size=1, bias=False
        )
        self.p_morph = nn.Conv2d(
            self.in_channels, self.morphology_channels, kernel_size=1, bias=False
        )
        self.u_sem = nn.Conv2d(
            self.semantic_channels, self.in_channels, kernel_size=1, bias=False
        )
        self.u_morph = nn.Conv2d(
            self.morphology_channels, self.in_channels, kernel_size=1, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for layer in (self.p_sem, self.p_morph, self.u_sem, self.u_morph):
                layer.weight.zero_()
            semantic_index = torch.arange(self.semantic_channels)
            morphology_index = torch.arange(self.morphology_channels)
            self.p_sem.weight[semantic_index, semantic_index, 0, 0] = 1.0
            self.p_morph.weight[
                morphology_index,
                self.semantic_channels + morphology_index,
                0,
                0,
            ] = 1.0
            self.u_sem.weight[semantic_index, semantic_index, 0, 0] = 1.0
            self.u_morph.weight[
                self.semantic_channels + morphology_index,
                morphology_index,
                0,
                0,
            ] = 1.0

    def factorize(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=feature.device.type, enabled=False):
            feature_fp32 = feature.float()
            if feature.is_cuda:
                with _ieee_cuda_conv_precision():
                    return self.p_sem(feature_fp32), self.p_morph(feature_fp32)
            return self.p_sem(feature_fp32), self.p_morph(feature_fp32)

    def reconstruct_branches(
        self, semantic: torch.Tensor, morphology: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=semantic.device.type, enabled=False):
            semantic_fp32 = semantic.float()
            morphology_fp32 = morphology.float()
            if semantic.is_cuda:
                with _ieee_cuda_conv_precision():
                    return self.u_sem(semantic_fp32), self.u_morph(morphology_fp32)
            return self.u_sem(semantic_fp32), self.u_morph(morphology_fp32)

    def forward(
        self, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        semantic, morphology = self.factorize(feature)
        semantic_reconstruction, morphology_reconstruction = (
            self.reconstruct_branches(semantic, morphology)
        )
        reconstruction = semantic_reconstruction + morphology_reconstruction
        return reconstruction, {
            "input": feature,
            "semantic": semantic,
            "morphology": morphology,
            "semantic_reconstruction": semantic_reconstruction,
            "morphology_reconstruction": morphology_reconstruction,
            "reconstruction": reconstruction,
        }

    def forward_inference(self, feature: torch.Tensor) -> torch.Tensor:
        return self(feature)[0]


def semantic_preservation_agreement(
    student_response: torch.Tensor,
    teacher_response: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean spatial cosine agreement across the class-channel dimension."""

    if student_response.shape != teacher_response.shape:
        raise ValueError("Teacher and student semantic responses must match")
    student = F.normalize(student_response.float(), p=2, dim=1, eps=eps)
    teacher = F.normalize(
        teacher_response.detach().float(), p=2, dim=1, eps=eps
    )
    return (student * teacher).sum(dim=1).mean()


def semantic_preservation_loss(
    student_response: torch.Tensor,
    teacher_response: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return 1.0 - semantic_preservation_agreement(
        student_response, teacher_response, eps=eps
    )


__all__ = [
    "OSMFV11Factorizer",
    "OSMF_EQUIVARIANCE_INTERVAL",
    "OSMF_LAMBDA_SEM",
    "OSMF_LAMBDA_MORPH",
    "OSMF_LAMBDA_ORTH",
    "OSMF_LAMBDA_REC",
    "semantic_preservation_agreement",
    "semantic_preservation_loss",
    "spatial_equivariance_loss",
    "cross_subspace_covariance",
    "orthogonality_loss",
    "reconstruction_cosine",
    "reconstruction_loss",
    "inverse_align_morphology",
]

