"""Frozen SSHR A0 with OSMF-v1.1 only at post-HFRM H28_1."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from network.osmf_v11 import OSMFV11Factorizer
from network.resnet38_cls import Net as SSHRNet


class Net(SSHRNet):
    def __init__(self, n_class: int):
        super().__init__(n_class=n_class)
        with torch.random.fork_rng(devices=[]):
            self.osmf_28_1 = OSMFV11Factorizer(in_channels=512)
        self.from_scratch_layers.append(self.osmf_28_1)

    def _forward_hierarchy(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.conv1a(x)
        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)
        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)
        feat_56 = x
        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)
        feat_28_1 = F.relu(self.bn45(x))
        x, _ = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)
        feat_28_2 = F.relu(self.bn52(x))
        x, _ = self.b6(x, get_x_bn_relu=True)
        x = self.b7(x)
        feat_deep = F.relu(self.bn7(x))

        feat_56_rectified = self.hfrm_56(feat_56, feat_deep)
        feat_28_1_rectified = self.hfrm_28_1(feat_28_1, feat_deep)
        feat_28_2_rectified = self.hfrm_28_2(feat_28_2, feat_deep)
        feat_28_1_osmf, osmf_aux = self.osmf_28_1(feat_28_1_rectified)
        return (
            feat_56_rectified,
            feat_28_1_osmf,
            feat_28_2_rectified,
            feat_deep,
            osmf_aux,
        )

    def _frozen_ic1_response(self, feature: torch.Tensor) -> torch.Tensor:
        """Apply pretrained ic1 geometry without an auxiliary gradient to ic1."""

        bias = None if self.ic1.bias is None else self.ic1.bias.detach()
        return F.conv2d(
            feature,
            self.ic1.weight.detach(),
            bias,
            stride=self.ic1.stride,
            padding=self.ic1.padding,
            dilation=self.ic1.dilation,
            groups=self.ic1.groups,
        )

    def _attach_semantic_responses(self, aux: Dict[str, torch.Tensor]) -> None:
        teacher = self._frozen_ic1_response(aux["input"].detach()).detach()
        student = self._frozen_ic1_response(aux["semantic_reconstruction"])
        aux["semantic_teacher_response"] = teacher
        aux["semantic_student_response"] = student

    def forward_with_aux(self, x: torch.Tensor):
        (
            feat_56_rectified,
            feat_28_1_osmf,
            feat_28_2_rectified,
            feat_deep,
            osmf_aux,
        ) = self._forward_hierarchy(x)
        self._attach_semantic_responses(osmf_aux)

        cam_56 = self.ic_56(feat_56_rectified)
        cam_28_1 = self.ic1(feat_28_1_osmf)
        cam_28_2 = self.ic2(feat_28_2_rectified)
        cam_deep = self.fc8(self.dropout7(feat_deep))
        out_56 = F.adaptive_avg_pool2d(cam_56, 1).flatten(1)
        out_28_1 = F.adaptive_avg_pool2d(cam_28_1, 1).flatten(1)
        out_28_2 = F.adaptive_avg_pool2d(cam_28_2, 1).flatten(1)
        out_deep = F.adaptive_avg_pool2d(cam_deep, 1).flatten(1)
        y_deep = torch.sigmoid(out_deep)
        outputs = (
            out_56,
            out_28_1,
            out_28_2,
            out_deep,
            y_deep,
            cam_56,
            cam_28_1,
            cam_28_2,
            cam_deep,
            feat_56_rectified,
        )
        return outputs, osmf_aux

    def forward(self, x: torch.Tensor):
        return self.forward_with_aux(x)[0]

    def forward_morphology(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_hierarchy(x)[-1]["morphology"]

    def forward_osmf_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._forward_hierarchy(x)[-1]


class Net_CAM(Net):
    def forward(self, x: torch.Tensor):
        return super().forward(x)[4]

    def forward_cam(self, x: torch.Tensor):
        (
            feat_56_rectified,
            feat_28_1_osmf,
            feat_28_2_rectified,
            feat_deep,
            _,
        ) = self._forward_hierarchy(x)
        cam_56 = F.relu(self.ic_56(feat_56_rectified))
        cam_28_1 = F.relu(self.ic1(feat_28_1_osmf))
        cam_28_2 = F.relu(self.ic2(feat_28_2_rectified))
        cam_deep = F.relu(self.fc8(feat_deep))
        out_deep = F.adaptive_avg_pool2d(self.fc8(feat_deep), 1).flatten(1)
        return cam_56, cam_28_1, cam_28_2, cam_deep, torch.sigmoid(out_deep)
