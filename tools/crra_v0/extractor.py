"""Frozen A0 H28_1 extraction and exact RGR-v0 coarse proposal."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from network.resnet38_cls import Net_CAM
from tools.crra_v0 import BCSS_THRESHOLDS


class CRRAFeatureExtractor(Net_CAM):
    """Parameter-identical A0 model with an observational H28_1 return."""

    def forward_cam_feature(self, x):
        x = self.conv1a(x)
        x = self.b2(x); x = self.b2_1(x); x = self.b2_2(x)
        x = self.b3(x); x = self.b3_1(x); x = self.b3_2(x)
        feat_56 = x
        x = self.b4(x); x = self.b4_1(x); x = self.b4_2(x)
        x = self.b4_3(x); x = self.b4_4(x); x = self.b4_5(x)
        feat_28_1 = F.relu(self.bn45(x))
        x, _ = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x); x = self.b5_2(x)
        feat_28_2 = F.relu(self.bn52(x))
        x, _ = self.b6(x, get_x_bn_relu=True); x = self.b7(x)
        feat_deep = F.relu(self.bn7(x))

        feat_56 = self.hfrm_56(feat_56, feat_deep)
        feat_28_1 = self.hfrm_28_1(feat_28_1, feat_deep)
        feat_28_2 = self.hfrm_28_2(feat_28_2, feat_deep)
        cam_56 = self.ic_56(feat_56)
        cam_28_1 = self.ic1(feat_28_1)
        cam_28_2 = self.ic2(feat_28_2)
        cam_deep = self.fc8(feat_deep)
        logits = F.adaptive_avg_pool2d(cam_deep, 1).flatten(1)
        return (
            cam_56,
            cam_28_1,
            cam_28_2,
            cam_deep,
            torch.sigmoid(logits),
            feat_28_1,
        )


def _resize_like(cam: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if cam.shape[-2:] == reference.shape[-2:]:
        return cam
    return F.interpolate(cam, reference.shape[-2:], mode="bilinear", align_corners=False)


def spatial_normalize(cam: torch.Tensor) -> torch.Tensor:
    flat = cam.flatten(2)
    minimum = flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
    maximum = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
    return (cam - minimum) / (maximum - minimum + 1e-8)


def presence_from_probability(probability: torch.Tensor) -> torch.Tensor:
    thresholds = probability.new_tensor(BCSS_THRESHOLDS)
    presence = (probability > thresholds).to(probability.dtype)
    empty = presence.sum(dim=1) == 0
    if empty.any():
        presence[empty, probability[empty].argmax(dim=1)] = 1.0
    return presence


def rgr_coarse_proposal(
    cam_56: torch.Tensor,
    cam_28_1: torch.Tensor,
    cam_28_2: torch.Tensor,
    cam_deep: torch.Tensor,
    probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact detached 0/0.6/0.2/0.2 RGR-v0 proposal on H28_1."""

    with torch.no_grad():
        reference = cam_28_1
        normalized = [
            spatial_normalize(F.relu(_resize_like(cam, reference).detach()).float())
            for cam in (cam_56, cam_28_1, cam_28_2, cam_deep)
        ]
        presence = presence_from_probability(probability.detach().float())
        fused = (
            0.0 * normalized[0]
            + 0.6 * normalized[1]
            + 0.2 * normalized[2]
            + 0.2 * normalized[3]
        )
        fused = fused * presence.view(-1, 4, 1, 1)
        return fused.argmax(dim=1), presence


def extract_batch(model, images: torch.Tensor, amp_dtype: str = "bf16"):
    """Extract canonical, unflipped frozen features and proposals."""

    dtype = torch.bfloat16 if amp_dtype == "bf16" else None
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=dtype, enabled=dtype is not None
    ):
        outputs = model.forward_cam_feature(images)
    proposal, presence = rgr_coarse_proposal(*outputs[:5])
    return (
        proposal.detach().cpu().numpy().astype(np.uint8),
        outputs[5].detach().float().cpu(),
        presence.detach().float().cpu().numpy(),
    )
