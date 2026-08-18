"""A0-compatible ResNet38 SSHR network with optional RSBR-v0 refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from network.resnet38_cls import Net as A0Net
from network.rsbr_v0 import RSBRRefinement, RSBRResult


BCSS_THRESHOLDS = (0.8, 0.9, 0.8, 0.6)


class Net(A0Net):
    def __init__(self, n_class):
        super().__init__(n_class)
        self.rsbr = RSBRRefinement(feature_channels=512, n_class=n_class)
        self.from_scratch_layers.append(self.rsbr)

    def _base_features_and_logits(self, x, apply_deep_dropout=True):
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

        feat_56_rectified = self.hfrm_56(feat_56, feat_deep)
        feat_28_1_rectified = self.hfrm_28_1(feat_28_1, feat_deep)
        feat_28_2_rectified = self.hfrm_28_2(feat_28_2, feat_deep)
        cam_56 = self.ic_56(feat_56_rectified)
        cam_28_1 = self.ic1(feat_28_1_rectified)
        cam_28_2 = self.ic2(feat_28_2_rectified)
        deep_input = self.dropout7(feat_deep) if apply_deep_dropout else feat_deep
        cam_deep = self.fc8(deep_input)
        out_deep = F.adaptive_avg_pool2d(self.fc8(feat_deep), 1).flatten(1)
        return (
            cam_56, cam_28_1, cam_28_2, cam_deep,
            torch.sigmoid(out_deep), feat_28_1_rectified,
        )

    def refine_from_base(
        self,
        base,
        presence,
        collect_structures=False,
    ) -> RSBRResult:
        cam_56, cam_28_1, cam_28_2, cam_deep, _, feature = base
        return self.rsbr(
            feature, cam_56, cam_28_1, cam_28_2, cam_deep, presence,
            collect_structures=collect_structures,
        )

    def forward_cam_base(self, x):
        """Dropout-free base path shared by validation and ``Net_CAM``."""
        return self._base_features_and_logits(x, apply_deep_dropout=False)

    def forward(self, x, presence=None, return_rsbr_aux=False):
        base = self._base_features_and_logits(x, apply_deep_dropout=True)
        cam_56, cam_28_1, cam_28_2, cam_deep, y_deep, feature = base
        if presence is None:
            thresholds = y_deep.new_tensor(BCSS_THRESHOLDS)
            presence = (y_deep > thresholds).to(y_deep.dtype)
            empty = presence.sum(dim=1) == 0
            if empty.any():
                presence[empty, y_deep[empty].argmax(dim=1)] = 1.0
        result = self.refine_from_base(base, presence)
        out_56 = F.adaptive_avg_pool2d(cam_56, 1).flatten(1)
        out_28_1 = F.adaptive_avg_pool2d(result.refined_cam, 1).flatten(1)
        out_28_2 = F.adaptive_avg_pool2d(cam_28_2, 1).flatten(1)
        out_deep = F.adaptive_avg_pool2d(cam_deep, 1).flatten(1)
        outputs = (
            out_56, out_28_1, out_28_2, out_deep, y_deep,
            cam_56, result.refined_cam, cam_28_2, cam_deep, feature,
        )
        return (*outputs, result) if return_rsbr_aux else outputs


class Net_CAM(Net):
    def forward(self, x):
        return self._base_features_and_logits(x, apply_deep_dropout=False)[4]

    def forward_cam(self, x, presence=None, return_rsbr_aux=False):
        base = self.forward_cam_base(x)
        cam_56, _, cam_28_2, cam_deep, y_deep, _ = base
        if presence is None:
            thresholds = y_deep.new_tensor(BCSS_THRESHOLDS)
            presence = (y_deep > thresholds).to(y_deep.dtype)
            empty = presence.sum(dim=1) == 0
            if empty.any():
                presence[empty, y_deep[empty].argmax(dim=1)] = 1.0
        result = self.refine_from_base(base, presence)
        outputs = (
            F.relu(cam_56), F.relu(result.refined_cam),
            F.relu(cam_28_2), F.relu(cam_deep), y_deep,
        )
        return (*outputs, result) if return_rsbr_aux else outputs
