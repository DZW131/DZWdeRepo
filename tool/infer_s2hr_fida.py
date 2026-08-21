"""Frozen S²HR-v1 counterfactual inference for FIDA-v0.

All variants for one image reuse the same three backbone/TTA feature sets and
the same deployed presence mask.  Checkpoint parameters are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tool.infer_s2hr import (
    TTA_TRANSFORMS,
    _amp_dtype,
    _normalize,
    _presence,
    _resize_and_unflip,
)


PRIMARY_VARIANTS = {
    "V00": {"bps": False, "spatial": "zero"},
    "V10": {"bps": True, "spatial": "zero"},
    "V01": {"bps": False, "spatial": "learned"},
    "V11": {"bps": True, "spatial": "learned"},
}


@dataclass
class ViewBase:
    feature_28_1: torch.Tensor
    residual_global: torch.Tensor
    residual_spatial: torch.Tensor
    context: torch.Tensor
    boundary: torch.Tensor
    cam_56: torch.Tensor
    cam_28_2: torch.Tensor
    cam_deep: torch.Tensor
    deep_logits: torch.Tensor
    raw_28_1_logits: torch.Tensor
    output_dims: tuple[int, ...]


def masked_argmax(logits, presence):
    logits = np.asarray(logits, dtype=np.float32)
    active = np.asarray(presence, dtype=bool)
    if not active.any():
        raise ValueError("Presence mask must contain at least one class")
    masked = logits.copy()
    masked[~active] = -1.0e4
    return masked.argmax(axis=0).astype(np.uint8)


class FIDAInstrumentor:
    """Compute all frozen counterfactuals without rerunning the backbone."""

    def __init__(self, model, amp_dtype="bf16"):
        self.model = model.cuda().eval()
        self.dtype = _amp_dtype(amp_dtype)
        self.amp_name = amp_dtype
        self.backbone_forwards = 0

    def _autocast(self):
        return torch.autocast(
            device_type="cuda", dtype=self.dtype, enabled=self.dtype is not None
        )

    def _extract_features_once(self, image):
        with self._autocast():
            features = self.model._extract_features(image)
        self.backbone_forwards += 1
        return features

    def _rectified_cam_28_1(self, base, bps, spatial):
        module = self.model.hfrm_28_1
        if spatial == "learned":
            gamma_spatial = module.gamma_spatial
        elif spatial == "positive":
            gamma_spatial = module.gamma_spatial.abs()
        elif spatial == "zero":
            gamma_spatial = torch.zeros_like(module.gamma_spatial)
        else:
            raise ValueError(f"Unknown spatial counterfactual: {spatial}")
        if bps:
            rho = torch.sigmoid(module.rho_boundary_raw)
            ch_gate = 1.0 - rho.view(1, 1, 1, 1) * base.boundary
        else:
            ch_gate = torch.ones_like(base.boundary)
        with self._autocast():
            rectified = (
                base.feature_28_1
                + module.gamma_veto * base.residual_global
                + gamma_spatial * base.residual_spatial
                + module.gamma_context * ch_gate * base.context
            )
            return F.relu(self.model.ic1(rectified))

    @staticmethod
    def _postprocess(cams, presence):
        response = (
            0.6 * _normalize(cams["cam_28_1"])
            + 0.2 * _normalize(cams["cam_28_2"])
            + 0.2 * _normalize(cams["cam_deep"])
        )
        response *= np.asarray(presence, dtype=np.float32).reshape(4, 1, 1)
        return response.argmax(axis=0).astype(np.uint8), response

    def audit_image(self, image, original_size):
        """Return all preregistered variants from exactly three base forwards."""

        image = image.cuda(non_blocking=True)
        feature_views = []
        probability_views = []
        # Presence is obtained from the same base features later reused by variants.
        for input_dims, output_dims in TTA_TRANSFORMS:
            augmented = torch.flip(image, dims=input_dims) if input_dims else image
            features = self._extract_features_once(augmented)
            with self._autocast():
                deep_logits = self.model.fc8(features[-1])
                probability_views.append(
                    self.model.hfrm_28_1.deep_image_probability(deep_logits)[0]
                )
            feature_views.append((features, deep_logits, tuple(output_dims)))

        probability_tensor = torch.stack(probability_views).mean(0)
        probability = probability_tensor.detach().float().cpu().numpy()
        deployed_presence = _presence(probability)
        presence_tensor = probability_tensor.new_tensor(deployed_presence)[None]

        bases = []
        for features, deep_logits, output_dims in feature_views:
            feature_56, feature_28_1, feature_28_2, feature_deep = features
            module = self.model.hfrm_28_1
            with self._autocast():
                raw_28_1_logits = self.model.ic1(feature_28_1)
                p_deep = module._masked_distribution(
                    deep_logits, presence_tensor, detach_logits=True
                )
                p_shallow = module._masked_distribution(
                    raw_28_1_logits, presence_tensor, detach_logits=False
                )
                directions = F.normalize(
                    self.model.ic1.weight.reshape(4, 512),
                    p=2,
                    dim=1,
                    eps=1.0e-12,
                ).detach()
                residual_spatial = torch.einsum(
                    "bkhw,kc->bchw", p_deep - p_shallow, directions
                )
                global_dna = F.adaptive_avg_pool2d(feature_deep, 1).flatten(1)
                global_gate = module.veto_mlp(global_dna).view(1, 512, 1, 1)
                residual_global = feature_28_1 * global_gate
                boundary = module.semantic_boundary_band(p_deep.argmax(dim=1)).detach()
                context = module.context_conv(feature_28_1)
                cam_56 = F.relu(
                    self.model.ic_56(self.model.hfrm_56(feature_56, feature_deep))
                )
                cam_28_2 = F.relu(
                    self.model.ic2(self.model.hfrm_28_2(feature_28_2, feature_deep))
                )
                cam_deep = F.relu(deep_logits)
            bases.append(
                ViewBase(
                    feature_28_1,
                    residual_global,
                    residual_spatial,
                    context,
                    boundary,
                    cam_56,
                    cam_28_2,
                    cam_deep,
                    deep_logits,
                    raw_28_1_logits,
                    output_dims,
                )
            )

        definitions = {**PRIMARY_VARIANTS, "Splus": {"bps": False, "spatial": "positive"}}
        collected = {
            name: {key: [] for key in ("cam_56", "cam_28_1", "cam_28_2", "cam_deep")}
            for name in definitions
        }
        deep_logits_views, raw_logits_views, boundary_views = [], [], []
        for base in bases:
            with self._autocast():
                common = {
                    "cam_56": base.cam_56,
                    "cam_28_2": base.cam_28_2,
                    "cam_deep": base.cam_deep,
                }
                for name, definition in definitions.items():
                    variant_cams = {
                        **common,
                        "cam_28_1": self._rectified_cam_28_1(base, **definition),
                    }
                    for key, cam in variant_cams.items():
                        collected[name][key].append(
                            _resize_and_unflip(cam, original_size, base.output_dims)
                        )
                deep_logits_views.append(
                    _resize_and_unflip(base.deep_logits, original_size, base.output_dims)
                )
                raw_logits_views.append(
                    _resize_and_unflip(base.raw_28_1_logits, original_size, base.output_dims)
                )
                boundary = torch.flip(base.boundary[0], dims=base.output_dims) if base.output_dims else base.boundary[0]
                boundary_views.append(boundary[0])

        variants = {}
        for name, cam_lists in collected.items():
            cams = {
                key: torch.stack(values).mean(0).detach().float().cpu().numpy()
                for key, values in cam_lists.items()
            }
            prediction, response = self._postprocess(cams, deployed_presence)
            variants[name] = {
                "cams": cams,
                "prediction": prediction,
                "response": response,
            }
        diagnostics = {
            "probability": probability,
            "deployed_presence": deployed_presence,
            "deep_logits": torch.stack(deep_logits_views).mean(0).detach().float().cpu().numpy(),
            "raw_28_1_logits": torch.stack(raw_logits_views).mean(0).detach().float().cpu().numpy(),
            "identity_view_boundary_28": bases[0].boundary[0, 0].detach().bool().cpu().numpy(),
            "tta_boundaries_28": [
                boundary.detach().bool().cpu().numpy() for boundary in boundary_views
            ],
            "tta_boundary_vote_28": torch.stack(boundary_views).mean(0).detach().float().cpu().numpy(),
            "base_forward_count": len(bases),
        }
        return variants, diagnostics

    def released_image(self, image, original_size):
        """Exact released S²HR two-pass image inference used only for parity."""

        image = image.cuda(non_blocking=True)
        probability_views = []
        with torch.no_grad():
            for input_dims, _ in TTA_TRANSFORMS:
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with self._autocast():
                    probability_views.append(self.model.forward_presence(augmented)[0])
            probability_tensor = torch.stack(probability_views).mean(0)
            presence = _presence(probability_tensor.detach().float().cpu().numpy())
            presence_tensor = probability_tensor.new_tensor(presence)[None]
            cams = {key: [] for key in ("cam_56", "cam_28_1", "cam_28_2", "cam_deep")}
            for input_dims, output_dims in TTA_TRANSFORMS:
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with self._autocast():
                    values = self.model.forward_cam(
                        augmented, present_mask=presence_tensor
                    )[:4]
                    for key, cam in zip(cams, values):
                        cams[key].append(
                            _resize_and_unflip(cam, original_size, output_dims)
                        )
            averaged = {
                key: torch.stack(values).mean(0).detach().float().cpu().numpy()
                for key, values in cams.items()
            }
        prediction, response = self._postprocess(averaged, presence)
        return {
            "cams": averaged,
            "prediction": prediction,
            "response": response,
            "presence": presence,
        }
