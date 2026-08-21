"""One-backbone-forward audit instrumentation for frozen SSHR HFRM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from network.resnet38_cls import Net_CAM
from tools.hma_v0 import BCSS_THRESHOLDS, STAGES, VARIANTS


class HMAAuditNet(Net_CAM):
    """A0-compatible model that exposes counterfactual HFRM variants."""

    @staticmethod
    def _stage_variants(module, feature, deep_feature):
        pooled = F.adaptive_avg_pool2d(deep_feature, 1).flatten(1)
        gate = module.veto_mlp(pooled).view(feature.shape[0], feature.shape[1], 1, 1)
        gsr_residual = feature * gate
        context = module.context_conv(feature)
        variants = {
            "raw": feature,
            "gsr": feature + module.gamma_veto * gsr_residual,
            "ch": feature + module.gamma_context * context,
            "full": (
                feature
                + module.gamma_veto * gsr_residual
                + module.gamma_context * context
            ),
        }
        return gate, gsr_residual, context, variants

    def forward_hfrm_audit(self, x, apply_deep_dropout=False):
        """Run the backbone exactly once and derive all HFRM variants."""

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

        raw_features = {"56": feat_56, "28_1": feat_28_1, "28_2": feat_28_2}
        modules = {
            "56": self.hfrm_56,
            "28_1": self.hfrm_28_1,
            "28_2": self.hfrm_28_2,
        }
        heads = {"56": self.ic_56, "28_1": self.ic1, "28_2": self.ic2}
        gates, gsr_residuals, contexts, feature_variants = {}, {}, {}, {}
        cam_logits = {variant: {} for variant in VARIANTS}
        cam_relu = {variant: {} for variant in VARIANTS}
        pooled_logits = {variant: {} for variant in VARIANTS}
        for stage in STAGES:
            gate, gsr_residual, context, variants = self._stage_variants(
                modules[stage], raw_features[stage], feat_deep
            )
            gates[stage] = gate
            gsr_residuals[stage] = gsr_residual
            contexts[stage] = context
            feature_variants[stage] = variants
            for variant in VARIANTS:
                logits = heads[stage](variants[variant])
                cam_logits[variant][stage] = logits
                cam_relu[variant][stage] = F.relu(logits)
                pooled_logits[variant][stage] = F.adaptive_avg_pool2d(logits, 1).flatten(1)

        deep_input = self.dropout7(feat_deep) if apply_deep_dropout else feat_deep
        deep_logits = self.fc8(deep_input)
        deep_relu = F.relu(deep_logits)
        deep_pooled = F.adaptive_avg_pool2d(deep_logits, 1).flatten(1)
        return {
            "raw_features": raw_features,
            "feat_deep": feat_deep,
            "gates": gates,
            "gsr_residuals": gsr_residuals,
            "contexts": contexts,
            "feature_variants": feature_variants,
            "cam_logits": cam_logits,
            "cam_relu": cam_relu,
            "pooled_logits": pooled_logits,
            "deep_logits": deep_logits,
            "deep_relu": deep_relu,
            "deep_pooled": deep_pooled,
            "y_deep": torch.sigmoid(deep_pooled),
        }


def presence_from_probability(probability: torch.Tensor) -> torch.Tensor:
    thresholds = probability.new_tensor(BCSS_THRESHOLDS)
    presence = (probability > thresholds).to(probability.dtype)
    empty = presence.sum(dim=1) == 0
    if empty.any():
        presence[empty, probability[empty].argmax(dim=1)] = 1.0
    return presence
