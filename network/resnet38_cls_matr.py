"""Clean SSHR with only MATR-v1 OT-MTR and SACR at HFRM28_1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.matr_hfrm28 import MATR_HFRM28_1
from network.matr_multiprototype_head import MultiPrototypeCAMHead
from network.resnet38_cls import Net as SSHRNet


class Net(SSHRNet):
    def __init__(self, n_class):
        if n_class != 4:
            raise ValueError("Frozen MATR-v1 expects four BCSS foreground classes")
        super().__init__(n_class)
        self.hfrm_28_1 = MATR_HFRM28_1()
        self.ic1 = MultiPrototypeCAMHead(512, n_class, modes=2)
        self.from_scratch_layers = [
            self.ic_56, self.ic1, self.ic2, self.fc8,
            self.hfrm_56, self.hfrm_28_1, self.hfrm_28_2,
        ]

    def forward(self, x):
        x = self.conv1a(x)
        x = self.b2(x); x = self.b2_1(x); x = self.b2_2(x)
        x = self.b3(x); x = self.b3_1(x); x = self.b3_2(x)
        feat_56 = x
        x = self.b4(x); x = self.b4_1(x); x = self.b4_2(x)
        x = self.b4_3(x); x = self.b4_4(x); x = self.b4_5(x)
        feat_28_1 = F.relu(self.bn45(x))
        x, _ = self.b5(x, get_x_bn_relu=True); x = self.b5_1(x); x = self.b5_2(x)
        feat_28_2 = F.relu(self.bn52(x))
        x, _ = self.b6(x, get_x_bn_relu=True); x = self.b7(x)
        feat_deep = F.relu(self.bn7(x))

        feat_56_rectified = self.hfrm_56(feat_56, feat_deep)
        feat_28_1_rectified = self.hfrm_28_1(feat_28_1, feat_deep)
        feat_28_2_rectified = self.hfrm_28_2(feat_28_2, feat_deep)
        cam_56 = self.ic_56(feat_56_rectified)
        cam_28_1, mode_logits = self.ic1(feat_28_1_rectified)
        cam_28_2 = self.ic2(feat_28_2_rectified)
        feat_deep_drop = self.dropout7(feat_deep)
        cam_deep = self.fc8(feat_deep_drop)

        out_56 = F.adaptive_avg_pool2d(cam_56, 1).view(x.size(0), -1)
        out_28_1 = F.adaptive_avg_pool2d(cam_28_1, 1).view(x.size(0), -1)
        out_28_2 = F.adaptive_avg_pool2d(cam_28_2, 1).view(x.size(0), -1)
        out_deep = F.adaptive_avg_pool2d(cam_deep, 1).view(x.size(0), -1)
        y_deep = torch.sigmoid(out_deep)
        return (
            out_56, out_28_1, out_28_2, out_deep, y_deep,
            cam_56, cam_28_1, cam_28_2, cam_deep, feat_56_rectified,
            mode_logits, feat_28_1_rectified,
        )

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        extra_scratch = (
            self.ic1.d_raw,
            self.hfrm_28_1.sacr.a_logits,
            self.hfrm_28_1.sacr.beta_adapt,
        )
        for parameter in extra_scratch:
            if any(id(parameter) == id(existing) for group in groups for existing in group):
                raise AssertionError("MATR raw parameter was already assigned")
            groups[2].append(parameter)
        return groups


class Net_CAM(Net):
    def forward(self, x):
        return super().forward(x)[4]

    def forward_cam(self, x):
        x = self.conv1a(x)
        x = self.b2(x); x = self.b2_1(x); x = self.b2_2(x)
        x = self.b3(x); x = self.b3_1(x); x = self.b3_2(x)
        feat_56 = x
        x = self.b4(x); x = self.b4_1(x); x = self.b4_2(x)
        x = self.b4_3(x); x = self.b4_4(x); x = self.b4_5(x)
        feat_28_1 = F.relu(self.bn45(x))
        x, _ = self.b5(x, get_x_bn_relu=True); x = self.b5_1(x); x = self.b5_2(x)
        feat_28_2 = F.relu(self.bn52(x))
        x, _ = self.b6(x, get_x_bn_relu=True); x = self.b7(x)
        feat_deep = F.relu(self.bn7(x))

        feat_56 = self.hfrm_56(feat_56, feat_deep)
        feat_28_1 = self.hfrm_28_1(feat_28_1, feat_deep)
        feat_28_2 = self.hfrm_28_2(feat_28_2, feat_deep)
        cam_56 = F.relu(self.ic_56(feat_56))
        cam_28_1, _ = self.ic1(feat_28_1)
        cam_28_1 = F.relu(cam_28_1)
        cam_28_2 = F.relu(self.ic2(feat_28_2))
        cam_deep = F.relu(self.fc8(feat_deep))
        out_deep = F.adaptive_avg_pool2d(self.fc8(feat_deep), 1).view(x.size(0), -1)
        return cam_56, cam_28_1, cam_28_2, cam_deep, torch.sigmoid(out_deep)
