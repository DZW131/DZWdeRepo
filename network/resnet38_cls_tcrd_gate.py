"""Official SSHR with the frozen TCRD-v0 CAM28_1 utility-gate dynamics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from network.resnet38_cls import Net as SSHRNet
from network.tcrd_dynamics import BRANCHES, TCRDDynamics


class Net(SSHRNet):
    def __init__(self, n_class: int, branch: str = "C0"):
        if branch not in BRANCHES:
            raise ValueError(f"Unknown branch: {branch}")
        super().__init__(n_class)
        self.branch = branch
        self.tcrd = None if branch == "C0" else TCRDDynamics(branch, n_class=n_class, steps=3)
        self.register_buffer(
            "bcss_thresholds", torch.tensor([0.8, 0.9, 0.8, 0.6]), persistent=False
        )

    def _features(self, x):
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
        return feat_56, feat_28_1, feat_28_2, feat_deep

    def _rectified(self, feat_56, feat_28_1, feat_28_2, feat_deep):
        return (
            self.hfrm_56(feat_56, feat_deep),
            self.hfrm_28_1(feat_28_1, feat_deep),
            self.hfrm_28_2(feat_28_2, feat_deep),
        )

    def _apply_dynamics(self, z0, feature, active_classes, return_diagnostics=False):
        if self.tcrd is None:
            diagnostics = {
                "z0": z0.detach().float(), "zt": z0.detach().float(),
                "conductance": None,
                "diffusion_update": torch.zeros_like(z0).detach().float(),
                "reaction_update": torch.zeros_like(z0).detach().float(),
                "active_classes": None if active_classes is None else active_classes.detach().bool(),
            }
            return (z0, diagnostics) if return_diagnostics else z0
        return self.tcrd(z0, feature, active_classes, return_diagnostics)

    def forward(self, x, active_labels=None):
        if self.branch in ("R", "DR") and active_labels is None:
            raise ValueError("Training reaction branches require image-level GT labels")
        feat_56, feat_28_1, feat_28_2, feat_deep = self._features(x)
        rect_56, rect_28_1, rect_28_2 = self._rectified(
            feat_56, feat_28_1, feat_28_2, feat_deep
        )
        cam_56 = self.ic_56(rect_56)
        z0_28_1 = self.ic1(rect_28_1)
        cam_28_1 = self._apply_dynamics(z0_28_1, rect_28_1, active_labels)
        cam_28_2 = self.ic2(rect_28_2)
        cam_deep = self.fc8(self.dropout7(feat_deep))
        # Keep the released SSHR pooling expression bit-for-bit for C0 parity.
        outputs = [
            F.avg_pool2d(
                cam, kernel_size=(cam.size(2), cam.size(3)), padding=0
            ).view(cam.size(0), -1)
            for cam in (cam_56, cam_28_1, cam_28_2, cam_deep)
        ]
        y_deep = torch.sigmoid(outputs[3])
        return (*outputs, y_deep, cam_56, cam_28_1, cam_28_2, cam_deep, rect_56)

    def _predicted_presence(self, y_deep):
        active = y_deep > self.bcss_thresholds.to(y_deep)
        empty = active.sum(dim=1) == 0
        if empty.any():
            fallback = y_deep.argmax(dim=1)
            active = active.clone()
            rows = torch.nonzero(empty, as_tuple=False).flatten()
            active[rows, fallback[rows]] = True
        return active

    def forward_cam(self, x, return_diagnostics: bool = False):
        feat_56, feat_28_1, feat_28_2, feat_deep = self._features(x)
        rect_56, rect_28_1, rect_28_2 = self._rectified(
            feat_56, feat_28_1, feat_28_2, feat_deep
        )
        raw_deep = self.fc8(feat_deep)
        y_deep = torch.sigmoid(F.adaptive_avg_pool2d(raw_deep, 1).flatten(1))
        active = self._predicted_presence(y_deep)
        z0 = self.ic1(rect_28_1)
        if return_diagnostics:
            zt, diagnostics = self._apply_dynamics(z0, rect_28_1, active, True)
            diagnostics["active_classes"] = active.detach()
        else:
            zt = self._apply_dynamics(z0, rect_28_1, active, False)
            diagnostics = None
        result = (
            F.relu(self.ic_56(rect_56)),
            F.relu(zt),
            F.relu(self.ic2(rect_28_2)),
            F.relu(raw_deep),
            y_deep,
        )
        return (*result, diagnostics) if return_diagnostics else result

    def get_parameter_groups(self):
        groups = tuple(list(group) for group in super().get_parameter_groups())
        if self.tcrd is not None:
            groups[2].extend(parameter for parameter in self.tcrd.parameters() if parameter.requires_grad)
        return groups


class Net_CAM(Net):
    def forward(self, x):
        return self.forward_cam(x)[4]
