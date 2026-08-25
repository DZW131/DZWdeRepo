"""Contrastive boundary-aware CH components for EXP-CBCCH-002."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .bcch import _detached_boundary_map
from .haar_wavelet import FixedHaarDWT2D


VALID_VARIANTS = ("A2", "A3")


def _masked_l2(value: torch.Tensor, image_labels: torch.Tensor) -> torch.Tensor:
    mask = image_labels[:, :, None, None].to(dtype=value.dtype)
    return F.normalize(F.relu(value) * mask, dim=1, eps=1.0e-6)


def _structural_orientation_feature(
    haar: FixedHaarDWT2D, feature: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        _, lh, hl, hh = haar.dwt(feature.detach())
        structure = torch.stack(
            (
                lh.abs().mean(dim=1),
                hl.abs().mean(dim=1),
                hh.abs().mean(dim=1),
            ),
            dim=1,
        )
        structure = F.interpolate(
            structure,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return F.normalize(structure, dim=1, eps=1.0e-6).detach()


class LocalSemanticAffinity(nn.Module):
    """Parameter-free local 15x15 semantic affinity propagation."""

    def __init__(self, kernel_size: int = 15, channel_chunk: int = 64) -> None:
        super().__init__()
        if kernel_size != 15:
            raise ValueError("EXP-CBCCH-002 freezes the CH15 neighborhood")
        self.kernel_size = int(kernel_size)
        self.padding = kernel_size // 2
        self.channel_chunk = int(channel_chunk)
        if self.channel_chunk <= 0:
            raise ValueError("channel_chunk must be positive")

    def _affinity(self, semantic: torch.Tensor):
        batch, channels, height, width = semantic.shape
        locations = height * width
        neighbors = self.kernel_size**2
        patches = F.unfold(
            semantic,
            kernel_size=self.kernel_size,
            padding=self.padding,
        ).view(batch, channels, neighbors, locations)
        center = semantic.flatten(2)
        scores = torch.einsum("bcl,bckl->bkl", center, patches)
        validity = F.unfold(
            semantic.new_ones((batch, 1, height, width)),
            kernel_size=self.kernel_size,
            padding=self.padding,
        ).view(batch, neighbors, locations)
        scores = scores.masked_fill(validity == 0, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=1), validity

    def _propagate(self, value: torch.Tensor, semantic: torch.Tensor):
        batch, channels, height, width = value.shape
        locations = height * width
        neighbors = self.kernel_size**2
        affinity, _ = self._affinity(semantic)
        output = []
        for start in range(0, channels, self.channel_chunk):
            stop = min(start + self.channel_chunk, channels)
            patches = F.unfold(
                value[:, start:stop],
                kernel_size=self.kernel_size,
                padding=self.padding,
            ).view(batch, stop - start, neighbors, locations)
            output.append(torch.einsum("bckl,bkl->bcl", patches, affinity))
        return torch.cat(output, dim=1).view(batch, channels, height, width)

    def forward(self, value: torch.Tensor, semantic: torch.Tensor):
        if self.training and (value.requires_grad or semantic.requires_grad):
            return checkpoint(
                self._propagate,
                value,
                semantic,
                use_reentrant=False,
            )
        return self._propagate(value, semantic)

    def forward_with_stats(self, value: torch.Tensor, semantic: torch.Tensor):
        batch, channels, height, width = value.shape
        locations = height * width
        neighbors = self.kernel_size**2
        affinity, validity = self._affinity(semantic)
        propagated_chunks = []
        for start in range(0, channels, self.channel_chunk):
            stop = min(start + self.channel_chunk, channels)
            patches = F.unfold(
                value[:, start:stop],
                kernel_size=self.kernel_size,
                padding=self.padding,
            ).view(batch, stop - start, neighbors, locations)
            propagated_chunks.append(
                torch.einsum("bckl,bkl->bcl", patches, affinity)
            )
        propagated = torch.cat(propagated_chunks, dim=1).view(
            batch, channels, height, width
        )
        log_affinity = affinity.float().clamp_min(1.0e-12).log()
        entropy = -(affinity.float() * log_affinity).sum(dim=1)
        valid_count = validity.sum(dim=1).float().clamp_min(1.0)
        normalized_entropy = entropy / valid_count.log().clamp_min(1.0e-12)
        center_index = neighbors // 2
        stats = {
            "affinity_entropy": normalized_entropy.mean(),
            "affinity_max": affinity.float().amax(dim=1).mean(),
            "affinity_self": affinity[:, center_index].float().mean(),
            "affinity_effective_neighbors": entropy.exp().mean(),
        }
        return propagated, stats


def contrastive_affinity_loss(
    feature: torch.Tensor,
    semantic_logits: torch.Tensor,
    image_labels: torch.Tensor,
    haar: FixedHaarDWT2D,
    variant: str,
    kernel_size: int = 15,
    hard_fraction: float = 0.20,
    temperature: float = 0.07,
):
    """Deterministically select one positive/negative per valid anchor."""

    variant = str(variant).upper()
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown CBCCH variant: {variant}")
    if kernel_size != 15 or hard_fraction != 0.20 or temperature != 0.07:
        raise ValueError("EXP-CBCCH-002 contrastive constants are frozen")

    semantic = _masked_l2(semantic_logits, image_labels)
    structural = _structural_orientation_feature(haar, feature)
    boundary = _detached_boundary_map(haar, feature)
    batch, channels, height, width = semantic.shape
    locations = height * width
    neighbors = kernel_size**2
    padding = kernel_size // 2

    labels = image_labels[:, :, None, None].to(dtype=semantic_logits.dtype)
    masked_response = F.relu(semantic_logits.detach()) * labels
    predicted = masked_response.argmax(dim=1)
    semantic_detached = semantic.detach()

    semantic_patches = F.unfold(
        semantic,
        kernel_size=kernel_size,
        padding=padding,
    ).view(batch, channels, neighbors, locations)
    semantic_detached_patches = semantic_patches.detach()
    structural_patches = F.unfold(
        structural,
        kernel_size=kernel_size,
        padding=padding,
    ).view(batch, 3, neighbors, locations)
    predicted_patches = F.unfold(
        predicted[:, None].float(),
        kernel_size=kernel_size,
        padding=padding,
    ).view(batch, neighbors, locations).long()
    validity = F.unfold(
        semantic.new_ones((batch, 1, height, width)),
        kernel_size=kernel_size,
        padding=padding,
    ).view(batch, neighbors, locations).bool()
    validity[:, neighbors // 2] = False

    center_semantic = semantic_detached.flatten(2)
    center_structural = structural.flatten(2)
    cam_similarity = torch.einsum(
        "bcl,bckl->bkl", center_semantic, semantic_detached_patches
    )
    hf_similarity = torch.einsum(
        "bcl,bckl->bkl", center_structural, structural_patches
    )
    center_class = predicted.flatten(1)
    same_class = predicted_patches == center_class[:, None]
    positive_mask = validity & same_class
    negative_mask = validity & ~same_class

    positive_quality = 0.5 * (cam_similarity + hf_similarity)
    negative_quality = 0.5 * ((1.0 - cam_similarity) + (1.0 - hf_similarity))
    negative_inf = torch.finfo(positive_quality.dtype).min
    positive_index = positive_quality.masked_fill(~positive_mask, negative_inf).argmax(dim=1)
    negative_index = negative_quality.masked_fill(~negative_mask, negative_inf).argmax(dim=1)
    valid_pair = positive_mask.any(dim=1) & negative_mask.any(dim=1)

    if variant == "A3":
        count = max(1, int(math.ceil(hard_fraction * locations)))
        hard_index = boundary.flatten(1).topk(count, dim=1, largest=True, sorted=False).indices
        anchor_mask = torch.zeros(
            (batch, locations), dtype=torch.bool, device=feature.device
        )
        anchor_mask.scatter_(1, hard_index, True)
        valid_pair &= anchor_mask

    gather_index = positive_index[:, None, None].expand(-1, channels, 1, -1)
    positive = semantic_patches.gather(2, gather_index).squeeze(2)
    gather_index = negative_index[:, None, None].expand(-1, channels, 1, -1)
    negative = semantic_patches.gather(2, gather_index).squeeze(2)
    center = semantic.flatten(2)
    positive_similarity = (center * positive).sum(dim=1)
    negative_similarity = (center * negative).sum(dim=1)
    logits = torch.stack((positive_similarity, negative_similarity), dim=-1)
    logits = logits / temperature
    per_anchor = -F.log_softmax(logits.float(), dim=-1)[..., 0]
    if valid_pair.any():
        loss = per_anchor[valid_pair].mean()
        positive_mean = positive_similarity.detach()[valid_pair].float().mean()
        negative_mean = negative_similarity.detach()[valid_pair].float().mean()
    else:
        loss = semantic.sum() * 0.0
        positive_mean = semantic.new_tensor(float("nan"), dtype=torch.float32)
        negative_mean = semantic.new_tensor(float("nan"), dtype=torch.float32)
    stats = {
        "valid_anchors": int(valid_pair.sum()),
        "candidate_anchors": int(batch * locations),
        "valid_anchor_fraction": float(valid_pair.float().mean()),
        "positive_similarity": float(positive_mean),
        "negative_similarity": float(negative_mean),
        "similarity_margin": float(positive_mean - negative_mean),
        "boundary_mean": float(boundary.float().mean()),
    }
    return loss, stats


def _weighted_rms(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    energy = value.float().square().mean(dim=1, keepdim=True)
    return (energy * weight.float()).sum().div(weight.float().sum().clamp_min(1.0e-12)).sqrt()


class HFRMCBCCH(nn.Module):
    """Original HFRM shell with A2 or Full CBCCH contextual propagation."""

    def __init__(
        self,
        in_channels: int,
        deep_channels: int = 4096,
        context_kernel: int = 15,
        variant: str = "A3",
    ) -> None:
        super().__init__()
        variant = str(variant).upper()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unknown CBCCH variant: {variant}")
        if context_kernel != 15:
            raise ValueError("EXP-CBCCH-002 freezes CH15 support")
        self.variant = variant
        self.veto_mlp = nn.Sequential(
            nn.Linear(deep_channels, deep_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(deep_channels // 8, in_channels, bias=False),
            nn.Sigmoid(),
        )
        # Retained under the exact public key for matched checkpoint/optimizer
        # restoration. The frozen Phase-2 equation replaces this aggregation.
        self.context_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=context_kernel,
            padding=context_kernel // 2,
            groups=in_channels,
            bias=False,
        )
        nn.init.constant_(self.context_conv.weight, 1.0 / (context_kernel**2))
        self.haar = FixedHaarDWT2D()
        self.affinity = LocalSemanticAffinity(kernel_size=context_kernel)
        self.gamma_veto = nn.Parameter(torch.zeros(1))
        self.gamma_context = nn.Parameter(torch.zeros(1))
        object.__setattr__(self, "_semantic_probe", None)
        object.__setattr__(self, "_last_semantic_logits", None)

    def set_semantic_probe(self, probe: nn.Module) -> None:
        object.__setattr__(self, "_semantic_probe", probe)

    @property
    def last_semantic_logits(self):
        return object.__getattribute__(self, "_last_semantic_logits")

    def _semantic(self, feature: torch.Tensor) -> torch.Tensor:
        probe = object.__getattribute__(self, "_semantic_probe")
        if probe is None:
            raise RuntimeError("CBCCH semantic probe is not attached")
        logits = probe(feature)
        object.__setattr__(self, "_last_semantic_logits", logits)
        return F.normalize(F.relu(logits), dim=1, eps=1.0e-6)

    def context(self, feature: torch.Tensor):
        semantic = self._semantic(feature)
        propagated = self.affinity(feature, semantic)
        if self.variant == "A2":
            return propagated
        boundary = _detached_boundary_map(self.haar, feature)
        alpha = 1.0 - boundary
        return alpha * propagated + boundary * feature

    def context_with_diagnostics(self, feature: torch.Tensor):
        semantic = self._semantic(feature)
        propagated, affinity = self.affinity.forward_with_stats(feature, semantic)
        boundary = _detached_boundary_map(self.haar, feature)
        alpha = torch.ones_like(boundary) if self.variant == "A2" else 1.0 - boundary
        context = propagated if self.variant == "A2" else alpha * propagated + boundary * feature
        residual = propagated - feature
        input_rms = feature.float().square().mean().sqrt().clamp_min(1.0e-12)
        diagnostics = {
            **affinity,
            "propagation_residual_rms": residual.float().square().mean().sqrt() / input_rms,
            "boundary_propagation_rms": _weighted_rms(residual, boundary) / input_rms,
            "interior_propagation_rms": _weighted_rms(residual, 1.0 - boundary) / input_rms,
            "boundary_map_mean": boundary.float().mean(),
            "boundary_map_std": boundary.float().std(unbiased=False),
            "alpha_mean": alpha.float().mean(),
            "alpha_std": alpha.float().std(unbiased=False),
        }
        return context, diagnostics

    def forward(self, feat_nong: torch.Tensor, feat_deep: torch.Tensor):
        batch, channels = feat_nong.shape[:2]
        global_dna = F.adaptive_avg_pool2d(feat_deep, 1).view(batch, -1)
        veto_weights = self.veto_mlp(global_dna).view(batch, channels, 1, 1)
        feat_vetoed = feat_nong * veto_weights
        feat_context = self.context(feat_nong)
        return (
            feat_nong
            + self.gamma_veto * feat_vetoed
            + self.gamma_context * feat_context
        )
