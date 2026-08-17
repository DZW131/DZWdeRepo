"""Objective-Induced Semantic-Morphology Factorization (OSMF-v1.0).

The factorizer is initialized as an exact channel-partition identity so that
inserting it before the frozen CAM28_1 head does not change the A0 model at
initialization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


OSMF_LAMBDA_SEM = 0.20
OSMF_LAMBDA_MORPH = 0.20
OSMF_LAMBDA_ORTH = 0.05
OSMF_LAMBDA_REC = 0.10
OSMF_EQUIVARIANCE_INTERVAL = 4


class OSMFFactorizer(nn.Module):
    """Split, specialize, and exactly reconstruct a hierarchy feature.

    ``P_sem`` selects the first ``ceil(C/2)`` channels at initialization and
    ``P_morph`` selects the remaining channels. The two reconstruction
    projections place those channels back at their original positions, hence
    ``U_sem(P_sem(H)) + U_morph(P_morph(H)) == H`` at initialization.
    """

    def __init__(self, in_channels: int, n_class: int) -> None:
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
        self.semantic_classifier = nn.Linear(self.semantic_channels, n_class)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Install the frozen complementary-partition initialization."""

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

        nn.init.xavier_uniform_(self.semantic_classifier.weight)
        if self.semantic_classifier.bias is not None:
            nn.init.zeros_(self.semantic_classifier.bias)

    def factorize(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # post-HFRM H28_1 remains FP32 in the released BF16 autocast path.
        # Running the new identity projections under autocast would quantize H
        # before the original ic1 head and violate initialization parity.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            feature_fp32 = feature.float()
            if feature.is_cuda:
                # Released SSHR enables TF32 globally. A TF32 1x1 identity
                # convolution is not bit-exact, so constrain only the four new
                # OSMF projections while leaving the frozen SSHR path untouched.
                with torch.backends.cudnn.flags(allow_tf32=False):
                    return self.p_sem(feature_fp32), self.p_morph(feature_fp32)
            return self.p_sem(feature_fp32), self.p_morph(feature_fp32)

    def reconstruct(
        self, semantic: torch.Tensor, morphology: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type=semantic.device.type, enabled=False):
            semantic_fp32 = semantic.float()
            morphology_fp32 = morphology.float()
            if semantic.is_cuda:
                with torch.backends.cudnn.flags(allow_tf32=False):
                    return self.u_sem(semantic_fp32) + self.u_morph(morphology_fp32)
            return self.u_sem(semantic_fp32) + self.u_morph(morphology_fp32)

    def forward(
        self, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        semantic, morphology = self.factorize(feature)
        reconstruction = self.reconstruct(semantic, morphology)
        semantic_logits = self.semantic_classifier(
            F.adaptive_avg_pool2d(semantic, output_size=1).flatten(1)
        )
        aux = {
            "input": feature,
            "semantic": semantic,
            "morphology": morphology,
            "reconstruction": reconstruction,
            "semantic_logits": semantic_logits,
        }
        return reconstruction, aux

    def forward_inference(self, feature: torch.Tensor) -> torch.Tensor:
        semantic, morphology = self.factorize(feature)
        return self.reconstruct(semantic, morphology)


def semantic_classification_loss(
    semantic_logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Use the same image-level multilabel objective as released SSHR."""

    return F.multilabel_soft_margin_loss(semantic_logits, labels)


def spatial_equivariance_loss(
    morphology_a: torch.Tensor,
    morphology_b_aligned: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Channel-normalized spatial cosine equivariance loss."""

    if morphology_a.shape != morphology_b_aligned.shape:
        raise ValueError("Equivariance inputs must have identical shapes")
    a = F.normalize(morphology_a.float(), p=2, dim=1, eps=eps)
    b = F.normalize(morphology_b_aligned.float(), p=2, dim=1, eps=eps)
    return (1.0 - (a * b).sum(dim=1)).mean()


def cross_subspace_covariance(
    semantic: torch.Tensor, morphology: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Return the standardized semantic-morphology cross-covariance."""

    if semantic.shape[0] != morphology.shape[0] or semantic.shape[2:] != morphology.shape[2:]:
        raise ValueError("Semantic and morphology tensors must share batch/spatial axes")
    s = semantic.float().permute(0, 2, 3, 1).reshape(-1, semantic.shape[1])
    m = morphology.float().permute(0, 2, 3, 1).reshape(-1, morphology.shape[1])
    s = (s - s.mean(dim=0, keepdim=True)) / s.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(eps)
    m = (m - m.mean(dim=0, keepdim=True)) / m.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(eps)
    return s.transpose(0, 1).matmul(m) / float(s.shape[0])


def orthogonality_loss(
    semantic: torch.Tensor, morphology: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    covariance = cross_subspace_covariance(semantic, morphology, eps=eps)
    return covariance.square().sum() / float(
        semantic.shape[1] * morphology.shape[1]
    )


def reconstruction_cosine(
    reconstruction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Mean per-image cosine, with a detached reconstruction target."""

    if reconstruction.shape != target.shape:
        raise ValueError("Reconstruction and target must have identical shapes")
    prediction = reconstruction.float().flatten(1)
    detached_target = target.detach().float().flatten(1)
    return F.cosine_similarity(prediction, detached_target, dim=1, eps=eps).mean()


def reconstruction_loss(
    reconstruction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return 1.0 - reconstruction_cosine(reconstruction, target, eps=eps)


def inverse_align_morphology(
    morphology: torch.Tensor, flip_dimension: int
) -> torch.Tensor:
    """Inverse-align a horizontal (3) or vertical (2) image flip."""

    if flip_dimension not in (2, 3):
        raise ValueError("OSMF-v1.0 only permits vertical or horizontal flips")
    return torch.flip(morphology, dims=(flip_dimension,))
