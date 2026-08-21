"""SSR-v2 HFRM28_1 with PCSD and positive teacher-consistent PTCR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.resnet38_cls import HFRM


BCSS_THRESHOLDS = (0.8, 0.9, 0.8, 0.6)


def epoch_alpha(epoch: int) -> float:
    """Frozen epoch ramp shared by PCSD and PTCR."""

    if epoch < 1:
        raise ValueError("Epoch is one-based")
    return min(max((epoch - 1) * 0.25, 0.0), 1.0)


class SSRv2HFRM28_1(HFRM):
    """Original SSHR GSR/CH15 plus PCSD and positive-only PTCR."""

    def __init__(self, in_channels=512, deep_channels=4096, context_kernel=15):
        if in_channels != 512:
            raise ValueError("SSR-v2 is frozen for the 512-channel HFRM28_1 stage")
        super().__init__(
            in_channels=in_channels,
            deep_channels=deep_channels,
            context_kernel=context_kernel,
        )
        self.beta_spatial = nn.Parameter(torch.full((1,), -4.0))

    @property
    def gamma_spatial(self):
        return F.softplus(self.beta_spatial)

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
            raise ValueError("SSR-v2 training requires image-level labels")
        probability = cls.deep_image_probability(deep_cam_logits).detach()
        return cls._fallback_nonempty(image_label > 0.5, probability)

    @classmethod
    def inference_presence(cls, deep_cam_logits):
        probability = cls.deep_image_probability(deep_cam_logits)
        thresholds = probability.new_tensor(BCSS_THRESHOLDS)
        return cls._fallback_nonempty(probability > thresholds, probability)

    @staticmethod
    def _masked_log_distribution(logits, present_mask):
        active = present_mask[:, :, None, None] > 0.5
        return F.log_softmax(logits.masked_fill(~active, -1.0e4), dim=1)

    @classmethod
    def spatial_terms(
        cls,
        deep_cam_logits,
        raw_cam28_1_logits,
        present_mask,
        classifier_weight,
    ):
        """Return detached PTCR residual and differentiable raw PCSD loss."""

        log_teacher = cls._masked_log_distribution(deep_cam_logits, present_mask)
        teacher = log_teacher.softmax(dim=1).detach()
        log_student = cls._masked_log_distribution(raw_cam28_1_logits, present_mask)
        student = log_student.softmax(dim=1)
        valid = present_mask.sum(dim=1) >= 2

        per_pixel_kl = F.kl_div(
            log_student.float(), teacher.float(), reduction="none"
        ).sum(dim=1)
        per_image_kl = per_pixel_kl.mean(dim=(1, 2))
        if valid.any():
            pcsd_loss = per_image_kl[valid].mean()
        else:
            pcsd_loss = raw_cam28_1_logits.sum() * 0.0

        discrepancy = (teacher - student).detach()
        discrepancy = discrepancy * valid[:, None, None, None].to(discrepancy.dtype)
        directions = F.normalize(
            classifier_weight.reshape(4, 512), p=2, dim=1, eps=1.0e-12
        ).detach()
        residual = torch.einsum("bkhw,kc->bchw", discrepancy, directions)

        if valid.any():
            valid_discrepancy = discrepancy[valid]
            mean_abs = valid_discrepancy.abs().float().mean()
            agreement = (
                teacher[valid].argmax(dim=1) == student[valid].argmax(dim=1)
            ).float().mean()
        else:
            mean_abs = discrepancy.float().sum() * 0.0
            agreement = discrepancy.float().sum() * 0.0
        return {
            "pcsd_loss": pcsd_loss,
            "teacher_residual": residual,
            "teacher_distribution": teacher,
            "student_distribution": student,
            "discrepancy_detached": discrepancy,
            "valid_samples": valid,
            "mean_abs_discrepancy": mean_abs,
            "prediction_agreement": agreement,
        }

    def forward(
        self,
        feature,
        deep_feature,
        deep_cam_logits,
        raw_cam28_1_logits,
        present_mask,
        classifier_weight,
        alpha,
    ):
        batch, channels, _, _ = feature.shape
        if channels != 512 or classifier_weight.shape[:2] != (4, 512):
            raise ValueError("Unexpected SSR-v2 feature/classifier shape")
        alpha_tensor = feature.new_tensor(float(alpha))

        global_dna = F.adaptive_avg_pool2d(deep_feature, 1).flatten(1)
        global_gate = self.veto_mlp(global_dna).view(batch, channels, 1, 1)
        residual_global = feature * global_gate
        residual_context = self.context_conv(feature)
        spatial = self.spatial_terms(
            deep_cam_logits,
            raw_cam28_1_logits,
            present_mask,
            classifier_weight,
        )
        effective_gamma = alpha_tensor * self.gamma_spatial
        output = (
            feature
            + self.gamma_veto * residual_global
            + self.gamma_context * residual_context
            + effective_gamma.view(1, 1, 1, 1) * spatial["teacher_residual"]
        )
        diagnostics = {
            **spatial,
            "alpha": alpha_tensor.detach(),
            "beta_spatial": self.beta_spatial.detach().reshape(()),
            "gamma_spatial": self.gamma_spatial.detach().reshape(()),
            "effective_gamma": effective_gamma.detach().reshape(()),
            "mean_present_classes": present_mask.detach().sum(dim=1).float().mean(),
            "valid_fraction": spatial["valid_samples"].detach().float().mean(),
        }
        return output, diagnostics
