"""S²HR-v1 network: official SSHR with only HFRM28_1 reconstructed."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from network.resnet38_cls import Net as SSHRNet
from network.s2hfrm28_1 import S2HFRM28_1


class Net(SSHRNet):
    def __init__(self, n_class):
        if n_class != 4:
            raise ValueError("S²HR-v1 is frozen for four BCSS foreground classes")
        super().__init__(n_class=n_class)
        official_hfrm_28_1 = self.hfrm_28_1
        reconstructed_hfrm_28_1 = S2HFRM28_1(
            in_channels=512, deep_channels=4096, context_kernel=15
        )
        incompat = reconstructed_hfrm_28_1.load_state_dict(
            official_hfrm_28_1.state_dict(), strict=False
        )
        if set(incompat.missing_keys) != {"gamma_spatial", "rho_boundary_raw"}:
            raise AssertionError(
                f"Unexpected HFRM28_1 reconstruction keys: {incompat.missing_keys}"
            )
        if incompat.unexpected_keys:
            raise AssertionError(
                f"Unexpected official HFRM28_1 keys: {incompat.unexpected_keys}"
            )
        self.hfrm_28_1 = reconstructed_hfrm_28_1
        self.from_scratch_layers = [
            self.ic_56,
            self.ic1,
            self.ic2,
            self.fc8,
            self.hfrm_56,
            self.hfrm_28_1,
            self.hfrm_28_2,
        ]
        self.last_s2hr_diagnostics = None

    def _extract_features(self, x):
        x = self.conv1a(x)
        x = self.b2(x); x = self.b2_1(x); x = self.b2_2(x)
        x = self.b3(x); x = self.b3_1(x); x = self.b3_2(x)
        feature_56 = x

        x = self.b4(x); x = self.b4_1(x); x = self.b4_2(x)
        x = self.b4_3(x); x = self.b4_4(x); x = self.b4_5(x)
        feature_28_1 = F.relu(self.bn45(x))

        x, _ = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x); x = self.b5_2(x)
        feature_28_2 = F.relu(self.bn52(x))

        x, _ = self.b6(x, get_x_bn_relu=True)
        x = self.b7(x)
        feature_deep = F.relu(self.bn7(x))
        return feature_56, feature_28_1, feature_28_2, feature_deep

    def _resolve_presence(
        self, deep_cam_logits, image_label=None, mode=None, present_mask=None
    ):
        if present_mask is not None:
            probability = self.hfrm_28_1.deep_image_probability(deep_cam_logits)
            return self.hfrm_28_1._fallback_nonempty(present_mask > 0.5, probability)
        if mode is None:
            mode = "train" if image_label is not None else "inference"
        if mode == "train":
            return self.hfrm_28_1.training_presence(image_label, deep_cam_logits)
        if mode == "inference":
            if image_label is not None:
                raise ValueError("Inference must not receive image_label")
            return self.hfrm_28_1.inference_presence(deep_cam_logits)
        raise ValueError(f"Unknown S²HR forward mode: {mode}")

    def _rectify(
        self,
        feature_56,
        feature_28_1,
        feature_28_2,
        feature_deep,
        image_label=None,
        mode=None,
        present_mask=None,
    ):
        deep_cam_logits = self.fc8(feature_deep)
        raw_cam28_1_logits = self.ic1(feature_28_1)
        resolved_presence = self._resolve_presence(
            deep_cam_logits,
            image_label=image_label,
            mode=mode,
            present_mask=present_mask,
        )
        rectified_56 = self.hfrm_56(feature_56, feature_deep)
        rectified_28_1, diagnostics = self.hfrm_28_1(
            feature_28_1,
            feature_deep,
            deep_cam_logits,
            raw_cam28_1_logits,
            resolved_presence,
            self.ic1.weight,
        )
        rectified_28_2 = self.hfrm_28_2(feature_28_2, feature_deep)
        diagnostics = {
            **diagnostics,
            "resolved_present_mask": resolved_presence.detach(),
        }
        self.last_s2hr_diagnostics = diagnostics
        return rectified_56, rectified_28_1, rectified_28_2, deep_cam_logits

    def forward(self, x, image_label=None, mode=None, present_mask=None):
        feature_56, feature_28_1, feature_28_2, feature_deep = self._extract_features(x)
        rectified_56, rectified_28_1, rectified_28_2, _ = self._rectify(
            feature_56,
            feature_28_1,
            feature_28_2,
            feature_deep,
            image_label=image_label,
            mode=mode,
            present_mask=present_mask,
        )

        cam_56 = self.ic_56(rectified_56)
        cam_28_1 = self.ic1(rectified_28_1)
        cam_28_2 = self.ic2(rectified_28_2)
        cam_deep = self.fc8(self.dropout7(feature_deep))

        out_56 = F.adaptive_avg_pool2d(cam_56, 1).flatten(1)
        out_28_1 = F.adaptive_avg_pool2d(cam_28_1, 1).flatten(1)
        out_28_2 = F.adaptive_avg_pool2d(cam_28_2, 1).flatten(1)
        out_deep = F.adaptive_avg_pool2d(cam_deep, 1).flatten(1)
        y_deep = torch.sigmoid(out_deep)
        return (
            out_56,
            out_28_1,
            out_28_2,
            out_deep,
            y_deep,
            cam_56,
            cam_28_1,
            cam_28_2,
            cam_deep,
            rectified_56,
        )

    def forward_presence(self, x):
        *_, feature_deep = self._extract_features(x)
        deep_cam_logits = self.fc8(feature_deep)
        return self.hfrm_28_1.deep_image_probability(deep_cam_logits)

    def forward_cam(self, x, present_mask=None):
        feature_56, feature_28_1, feature_28_2, feature_deep = self._extract_features(x)
        rectified_56, rectified_28_1, rectified_28_2, deep_cam_logits = self._rectify(
            feature_56,
            feature_28_1,
            feature_28_2,
            feature_deep,
            image_label=None,
            mode="inference",
            present_mask=present_mask,
        )
        cam_56 = F.relu(self.ic_56(rectified_56))
        cam_28_1 = F.relu(self.ic1(rectified_28_1))
        cam_28_2 = F.relu(self.ic2(rectified_28_2))
        cam_deep = F.relu(deep_cam_logits)
        y_deep = self.hfrm_28_1.deep_image_probability(deep_cam_logits)
        return cam_56, cam_28_1, cam_28_2, cam_deep, y_deep

    def get_parameter_groups(self):
        groups = super().get_parameter_groups()
        scratch_ids = {id(parameter) for parameter in groups[2]}
        if id(self.hfrm_28_1.gamma_spatial) not in scratch_ids:
            groups[2].append(self.hfrm_28_1.gamma_spatial)
            scratch_ids.add(id(self.hfrm_28_1.gamma_spatial))
        if id(self.hfrm_28_1.rho_boundary_raw) not in scratch_ids:
            groups[2].append(self.hfrm_28_1.rho_boundary_raw)
        return groups


class Net_CAM(Net):
    def forward(self, x, image_label=None, mode=None, present_mask=None):
        return super().forward(
            x,
            image_label=image_label,
            mode=mode,
            present_mask=present_mask,
        )[4]
