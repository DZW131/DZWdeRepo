"""Wavelet low-frequency prototype recovery for EXP-BCPCH-003."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bcch import _detached_boundary_map
from .cbcch import LocalSemanticAffinity
from .haar_wavelet import FixedHaarDWT2D


CAM_THRESHOLD = 0.70
PROTOTYPE_MIX = 0.50
BCSS_PRESENCE_THRESHOLDS = (0.8, 0.9, 0.8, 0.6)


def _spatial_minmax(value: torch.Tensor) -> torch.Tensor:
    minimum = value.amin(dim=(-2, -1), keepdim=True)
    maximum = value.amax(dim=(-2, -1), keepdim=True)
    return (value - minimum) / (maximum - minimum + 1.0e-6)


def low_frequency_reconstruction(
    haar: FixedHaarDWT2D, feature: torch.Tensor
) -> torch.Tensor:
    """Return IDWT(LL, 0, 0, 0) at the original spatial resolution."""

    ll, _, _, _ = haar.dwt(feature)
    zero = torch.zeros_like(ll)
    return haar.idwt(ll, zero, zero, zero)


class LowFrequencyPrototypeRecovery(nn.Module):
    """Parameter-free LL prototype construction and per-pixel recovery."""

    def __init__(self, cam_threshold: float = 0.70) -> None:
        super().__init__()
        if cam_threshold != CAM_THRESHOLD:
            raise ValueError("EXP-BCPCH-003 freezes CAM threshold=0.70")
        self.cam_threshold = float(cam_threshold)

    @staticmethod
    def predicted_presence(deep_logits: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(
            F.adaptive_avg_pool2d(deep_logits, 1).flatten(1)
        )
        thresholds = probability.new_tensor(BCSS_PRESENCE_THRESHOLDS)
        presence = probability > thresholds
        empty = ~presence.any(dim=1)
        if empty.any():
            fallback = probability.argmax(dim=1)
            presence = presence.clone()
            presence[empty] = False
            presence[empty, fallback[empty]] = True
        return presence.detach()

    def forward(
        self,
        feature: torch.Tensor,
        cam_logits: torch.Tensor,
        deep_logits: torch.Tensor,
        haar: FixedHaarDWT2D,
        fallback: torch.Tensor,
    ):
        low_frequency = low_frequency_reconstruction(haar, feature)
        embedding = F.normalize(low_frequency, dim=1, eps=1.0e-6)
        cam = _spatial_minmax(F.relu(cam_logits))
        presence = self.predicted_presence(deep_logits)
        confidence = (
            (cam > self.cam_threshold)
            & presence[:, :, None, None]
        ).detach()
        confidence_float = confidence.to(dtype=embedding.dtype)
        count = confidence_float.sum(dim=(-2, -1))
        prototype = torch.einsum(
            "bkhw,bchw->bkc", confidence_float, embedding
        ) / count.clamp_min(1.0).unsqueeze(-1)
        prototype = F.normalize(prototype, dim=-1, eps=1.0e-6)
        valid = (count > 0) & presence
        no_valid = ~valid.any(dim=1)

        similarity = torch.einsum("bchw,bkc->bkhw", embedding, prototype)
        safe_valid = valid.clone()
        if no_valid.any():
            safe_valid[no_valid, 0] = True
        masked_similarity = similarity.masked_fill(
            ~safe_valid[:, :, None, None],
            torch.finfo(similarity.dtype).min,
        )
        weight = torch.softmax(masked_similarity, dim=1)
        prototype_map = torch.einsum("bkhw,bkc->bchw", weight, prototype)
        prototype_map = torch.where(
            no_valid[:, None, None, None], fallback, prototype_map
        )
        state = {
            "low_frequency": low_frequency,
            "embedding": embedding,
            "normalized_cam": cam,
            "confidence_mask": confidence,
            "presence": presence,
            "prototype": prototype,
            "valid_prototypes": valid,
            "class_similarity": similarity,
            "prototype_weight": weight,
            "fallback": no_valid,
        }
        return prototype_map, state


def _weighted_rms(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    energy = value.float().square().mean(dim=1, keepdim=True)
    return (
        (energy * weight.float())
        .sum()
        .div(weight.float().sum().clamp_min(1.0e-12))
        .sqrt()
    )


class HFRMBCPCH(nn.Module):
    """Original HFRM shell with frozen BCP-CH context at HFRM28_1."""

    def __init__(
        self,
        in_channels: int,
        deep_channels: int = 4096,
        context_kernel: int = 15,
    ) -> None:
        super().__init__()
        if context_kernel != 15:
            raise ValueError("EXP-BCPCH-003 freezes local support=15")
        self.veto_mlp = nn.Sequential(
            nn.Linear(deep_channels, deep_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(deep_channels // 8, in_channels, bias=False),
            nn.Sigmoid(),
        )
        # Retained under the public key solely for exact common-state and
        # optimizer restoration. BCP-CH replaces the aggregation equation.
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
        self.prototype_recovery = LowFrequencyPrototypeRecovery()
        self.gamma_veto = nn.Parameter(torch.zeros(1))
        self.gamma_context = nn.Parameter(torch.zeros(1))
        object.__setattr__(self, "_semantic_probe", None)
        object.__setattr__(self, "_presence_probe", None)
        object.__setattr__(self, "_last_semantic_logits", None)

    def set_shared_probes(self, semantic_probe: nn.Module, presence_probe: nn.Module):
        object.__setattr__(self, "_semantic_probe", semantic_probe)
        object.__setattr__(self, "_presence_probe", presence_probe)

    @property
    def last_semantic_logits(self):
        return object.__getattribute__(self, "_last_semantic_logits")

    def _probe(self, feature: torch.Tensor, deep: torch.Tensor):
        semantic_probe = object.__getattribute__(self, "_semantic_probe")
        presence_probe = object.__getattribute__(self, "_presence_probe")
        if semantic_probe is None or presence_probe is None:
            raise RuntimeError("BCP-CH shared ic1/fc8 probes are not attached")
        cam_logits = semantic_probe(feature)
        deep_logits = presence_probe(deep)
        object.__setattr__(self, "_last_semantic_logits", cam_logits)
        semantic = F.normalize(F.relu(cam_logits), dim=1, eps=1.0e-6)
        return semantic, cam_logits, deep_logits

    def _context_components(self, feature: torch.Tensor, deep: torch.Tensor):
        semantic, cam_logits, deep_logits = self._probe(feature, deep)
        affinity = self.affinity(feature, semantic)
        prototype, state = self.prototype_recovery(
            feature, cam_logits, deep_logits, self.haar, affinity
        )
        boundary = _detached_boundary_map(self.haar, feature)
        mixed = PROTOTYPE_MIX * affinity + (1.0 - PROTOTYPE_MIX) * prototype
        context = (1.0 - boundary) * mixed + boundary * feature
        return context, affinity, prototype, boundary, state

    def context(self, feature: torch.Tensor, deep: torch.Tensor):
        return self._context_components(feature, deep)[0]

    def prototype_with_diagnostics(
        self, feature: torch.Tensor, deep: torch.Tensor
    ):
        semantic, cam_logits, deep_logits = self._probe(feature, deep)
        affinity, affinity_stats = self.affinity.forward_with_stats(
            feature, semantic
        )
        prototype, state = self.prototype_recovery(
            feature, cam_logits, deep_logits, self.haar, affinity
        )
        boundary = _detached_boundary_map(self.haar, feature)
        mixed = PROTOTYPE_MIX * affinity + (1.0 - PROTOTYPE_MIX) * prototype
        context = (1.0 - boundary) * mixed + boundary * feature
        input_rms = feature.float().square().mean().sqrt().clamp_min(1.0e-12)
        affinity_residual = affinity - feature
        prototype_residual = prototype - feature
        context_residual = context - feature
        valid = state["valid_prototypes"]
        masked_similarity = state["class_similarity"].masked_fill(
            ~valid[:, :, None, None], -1.0
        )
        maximum_similarity = masked_similarity.amax(dim=1, keepdim=True)
        maximum_similarity = torch.where(
            state["fallback"][:, None, None, None],
            torch.zeros_like(maximum_similarity),
            maximum_similarity,
        )
        diagnostics = {
            **affinity_stats,
            "boundary_map_mean": boundary.float().mean(),
            "boundary_map_std": boundary.float().std(unbiased=False),
            "cam_confidence_fraction": state["confidence_mask"].float().mean(),
            "predicted_presence_per_image": state["presence"].float().sum(1).mean(),
            "valid_prototypes_per_image": valid.float().sum(1).mean(),
            "fallback_fraction": state["fallback"].float().mean(),
            "prototype_similarity_max": maximum_similarity.float().mean(),
            "prototype_similarity_boundary": (
                maximum_similarity.float() * boundary.float()
            ).sum().div(boundary.float().sum().clamp_min(1.0e-12)),
            "affinity_output_rms": affinity.float().square().mean().sqrt() / input_rms,
            "prototype_output_rms": prototype.float().square().mean().sqrt() / input_rms,
            "affinity_residual_rms": affinity_residual.float().square().mean().sqrt() / input_rms,
            "prototype_residual_rms": prototype_residual.float().square().mean().sqrt() / input_rms,
            "context_residual_rms": context_residual.float().square().mean().sqrt() / input_rms,
            "boundary_context_residual_rms": _weighted_rms(context_residual, boundary) / input_rms,
            "interior_context_residual_rms": _weighted_rms(context_residual, 1.0 - boundary) / input_rms,
            "ll_reconstruction_rms": state["low_frequency"].float().square().mean().sqrt() / input_rms,
        }
        auxiliary = {
            "class_similarity": state["class_similarity"],
            "valid_prototypes": valid,
            "boundary": boundary,
            "confidence_mask": state["confidence_mask"],
        }
        return context, diagnostics, auxiliary

    def forward(self, feat_nong: torch.Tensor, feat_deep: torch.Tensor):
        batch, channels = feat_nong.shape[:2]
        global_dna = F.adaptive_avg_pool2d(feat_deep, 1).view(batch, -1)
        veto_weights = self.veto_mlp(global_dna).view(batch, channels, 1, 1)
        feat_vetoed = feat_nong * veto_weights
        feat_context = self.context(feat_nong, feat_deep)
        return (
            feat_nong
            + self.gamma_veto * feat_vetoed
            + self.gamma_context * feat_context
        )
