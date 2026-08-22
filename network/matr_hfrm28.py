"""Original SSHR HFRM28_1 plus the zero-anchored SACR correction."""

from __future__ import annotations

import torch.nn.functional as F

from network.matr_sparse_context import SparseAdaptiveContext
from network.resnet38_cls import HFRM


class MATR_HFRM28_1(HFRM):
    def __init__(self):
        super().__init__(in_channels=512, deep_channels=4096, context_kernel=15)
        self.sacr = SparseAdaptiveContext(512)
        self.last_sacr_diagnostics = None

    def forward(self, feat_nong, feat_deep):
        batch, channels, _, _ = feat_nong.shape
        global_dna = F.adaptive_avg_pool2d(feat_deep, 1).view(batch, -1)
        veto_weights = self.veto_mlp(global_dna).view(batch, channels, 1, 1)
        feat_vetoed = feat_nong * veto_weights
        feat_smoothed = self.context_conv(feat_nong)
        delta_context, diagnostics = self.sacr(feat_nong)
        context_rms = feat_smoothed.float().square().mean().sqrt()
        diagnostics = {
            **diagnostics,
            "context15_rms": context_rms.detach(),
            "delta_context15_ratio": (
                diagnostics["delta_rms"] / context_rms.detach().clamp_min(1.0e-12)
            ),
        }
        self.last_sacr_diagnostics = diagnostics
        return (
            feat_nong
            + self.gamma_veto * feat_vetoed
            + self.gamma_context * feat_smoothed
            + self.sacr.gamma_adapt.to(feat_nong.dtype) * delta_context
        )
