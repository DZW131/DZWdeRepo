"""S²HFRM28_1: spatial semantics plus boundary-selective SSHR context."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.resnet38_cls import HFRM


BCSS_THRESHOLDS = (0.8, 0.9, 0.8, 0.6)


class S2HFRM28_1(HFRM):
    """Replace only HFRM28_1 while preserving its original GSR and CH15."""

    def __init__(self, in_channels=512, deep_channels=4096, context_kernel=15):
        if in_channels != 512:
            raise ValueError("S²HFRM-v1 is frozen for the 512-channel HFRM28_1 stage")
        super().__init__(
            in_channels=in_channels,
            deep_channels=deep_channels,
            context_kernel=context_kernel,
        )
        self.gamma_spatial = nn.Parameter(torch.zeros(1))
        self.rho_boundary_raw = nn.Parameter(torch.full((1,), -4.0))

    @staticmethod
    def deep_image_probability(deep_cam_logits):
        return torch.sigmoid(F.adaptive_avg_pool2d(deep_cam_logits, 1).flatten(1))

    @staticmethod
    def _fallback_nonempty(mask, probabilities):
        mask = mask.to(dtype=probabilities.dtype)
        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask = mask.clone()
            mask[empty, probabilities[empty].argmax(dim=1)] = 1.0
        return mask

    @classmethod
    def training_presence(cls, image_label, deep_cam_logits):
        if image_label is None:
            raise ValueError("Training S²HR requires image-level classification labels")
        probability = cls.deep_image_probability(deep_cam_logits).detach()
        return cls._fallback_nonempty(image_label > 0.5, probability)

    @classmethod
    def inference_presence(cls, deep_cam_logits):
        probability = cls.deep_image_probability(deep_cam_logits)
        thresholds = probability.new_tensor(BCSS_THRESHOLDS)
        return cls._fallback_nonempty(probability > thresholds, probability)

    @staticmethod
    def semantic_boundary_band(class_map):
        """Detached 8-neighbour transitions expanded by one feature pixel."""

        if class_map.ndim != 3:
            raise ValueError("class_map must have shape [B,H,W]")
        center = class_map
        padded = F.pad(
            class_map[:, None].to(torch.float32),
            (1, 1, 1, 1),
            mode="replicate",
        )[:, 0].to(class_map.dtype)
        height, width = class_map.shape[-2:]
        boundary = torch.zeros_like(class_map, dtype=torch.bool)
        for dy, dx in (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ):
            neighbour = padded[
                :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width
            ]
            boundary |= neighbour != center
        return F.max_pool2d(
            boundary[:, None].to(torch.float32),
            kernel_size=3,
            stride=1,
            padding=1,
        )

    @staticmethod
    def _masked_distribution(logits, present_mask, detach_logits):
        active = present_mask[:, :, None, None] > 0.5
        masked = logits.masked_fill(~active, -1.0e4)
        if detach_logits:
            masked = masked.detach()
        return torch.softmax(masked, dim=1)

    @staticmethod
    def _masked_mean(value, mask, fallback):
        total = mask.sum()
        measured = (value * mask).sum() / total.clamp_min(1.0)
        return torch.where(total > 0, measured, value.new_tensor(fallback))

    def forward(
        self,
        feature,
        deep_feature,
        deep_cam_logits,
        raw_cam28_1_logits,
        present_mask,
        classifier_weight,
    ):
        batch, channels, _, _ = feature.shape
        if channels != 512 or classifier_weight.shape[:2] != (4, 512):
            raise ValueError("Unexpected S²HFRM28_1 feature/classifier shape")

        global_dna = F.adaptive_avg_pool2d(deep_feature, 1).flatten(1)
        global_gate = self.veto_mlp(global_dna).view(batch, channels, 1, 1)
        residual_global = feature * global_gate

        p_deep = self._masked_distribution(
            deep_cam_logits, present_mask, detach_logits=True
        )
        p_shallow = self._masked_distribution(
            raw_cam28_1_logits, present_mask, detach_logits=False
        )
        discrepancy = p_deep - p_shallow
        directions = F.normalize(
            classifier_weight.reshape(4, channels), p=2, dim=1, eps=1.0e-12
        ).detach()
        residual_spatial = torch.einsum("bkhw,kc->bchw", discrepancy, directions)

        class_map = p_deep.argmax(dim=1)
        boundary = self.semantic_boundary_band(class_map).detach()
        rho = torch.sigmoid(self.rho_boundary_raw)
        ch_gate = 1.0 - rho.view(1, 1, 1, 1) * boundary
        context = self.context_conv(feature)

        output = (
            feature
            + self.gamma_veto * residual_global
            + self.gamma_spatial * residual_spatial
            + self.gamma_context * ch_gate * context
        )
        interior = 1.0 - boundary
        diagnostics = {
            "semantic_discrepancy_abs_mean": discrepancy.detach().abs().mean(),
            "boundary_fraction": boundary.mean(),
            "ch_gate_boundary_mean": self._masked_mean(
                ch_gate.detach(), boundary, fallback=1.0
            ),
            "ch_gate_interior_mean": self._masked_mean(
                ch_gate.detach(), interior, fallback=1.0
            ),
            "rho_boundary": rho.detach().reshape(()),
            "present_classes_mean": present_mask.detach().sum(dim=1).mean(),
        }
        return output, diagnostics
