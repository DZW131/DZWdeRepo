"""Exact official prediction plus frozen post-HFRM H28_1 extraction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from network.resnet38_cls import Net_CAM
from tool import infer_utils
from tool.infer_fun import _get_class_thresholds, _tta_transforms


class RegionAuditExtractor(Net_CAM):
    """Parameter-identical A0 extractor with an observational feature return."""

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
        feat_56_rectified = self.hfrm_56(feat_56, feat_deep)
        feat_28_1_rectified = self.hfrm_28_1(feat_28_1, feat_deep)
        feat_28_2_rectified = self.hfrm_28_2(feat_28_2, feat_deep)
        cam_56 = F.relu(self.ic_56(feat_56_rectified))
        cam_28_1 = F.relu(self.ic1(feat_28_1_rectified))
        cam_28_2 = F.relu(self.ic2(feat_28_2_rectified))
        cam_deep = F.relu(self.fc8(feat_deep))
        out_deep = F.avg_pool2d(
            self.fc8(feat_deep),
            kernel_size=(feat_deep.size(2), feat_deep.size(3)),
            padding=0,
        ).flatten(1)
        return (
            cam_56, cam_28_1, cam_28_2, cam_deep,
            torch.sigmoid(out_deep), feat_28_1_rectified,
        )


def predict_and_feature(model, image_tensor, original_size, args):
    """Replicate released inference exactly and return canonical H28_1."""

    cams_28_1, cams_28_2, cams_deep, probabilities = [], [], [], []
    canonical_feature = None
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else None
    with torch.no_grad():
        for index, (input_flip_dims, cam_flip_dims) in enumerate(_tta_transforms()):
            tta_image = (
                torch.flip(image_tensor, dims=input_flip_dims)
                if input_flip_dims else image_tensor
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                _, cam_28_1, cam_28_2, cam_deep, probability, feature = (
                    model.forward_cam_feature(tta_image)
                )
                cam_28_1 = F.interpolate(
                    cam_28_1, original_size, mode="bilinear", align_corners=False
                )[0]
                cam_28_2 = F.interpolate(
                    cam_28_2, original_size, mode="bilinear", align_corners=False
                )[0]
                cam_deep = F.interpolate(
                    cam_deep, original_size, mode="bilinear", align_corners=False
                )[0]
            if index == 0:
                canonical_feature = feature[0].detach().float().cpu()
            if cam_flip_dims:
                cam_28_1 = torch.flip(cam_28_1, dims=cam_flip_dims)
                cam_28_2 = torch.flip(cam_28_2, dims=cam_flip_dims)
                cam_deep = torch.flip(cam_deep, dims=cam_flip_dims)
            cams_28_1.append(cam_28_1)
            cams_28_2.append(cam_28_2)
            cams_deep.append(cam_deep)
            probabilities.append(probability)

    c_28_1 = torch.stack(cams_28_1).mean(0)
    c_28_2 = torch.stack(cams_28_2).mean(0)
    c_deep = torch.stack(cams_deep).mean(0)
    probability = torch.stack(probabilities).mean(0).detach().float().cpu().numpy()[0]
    thresholds = _get_class_thresholds(args, None, 4)
    label = (probability > thresholds).astype(np.float32)
    if label.sum() == 0:
        label[int(np.argmax(probability))] = 1.0

    def normalize(cam):
        array = cam.detach().float().cpu().numpy()
        minimum = np.min(array, axis=(1, 2), keepdims=True)
        maximum = np.max(array, axis=(1, 2), keepdims=True)
        return (array - minimum) / (maximum - minimum + 1e-8)

    cam = 0.6 * normalize(c_28_1) + 0.2 * normalize(c_28_2) + 0.2 * normalize(c_deep)
    cam *= label.reshape(4, 1, 1)
    cam_dict = infer_utils.cam_npy_to_cam_dict(cam, label)
    cam_score, _ = infer_utils.dict2npy(cam_dict, label, np.zeros((*original_size, 3), dtype=np.uint8))
    prediction = infer_utils.cam_npy_to_label_map(cam_score)
    return prediction.astype(np.uint8), canonical_feature
