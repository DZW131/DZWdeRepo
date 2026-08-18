"""Frozen SSHR A0 plus OSMF-v1.3 at post-HFRM H28_1."""

from __future__ import annotations

import torch

from network.osmf_v13 import OSMFV13Factorizer
from network.resnet38_cls_osmf_v12 import Net as V12Net, Net_CAM as V12NetCAM


class Net(V12Net):
    def __init__(self, n_class: int):
        super().__init__(n_class=n_class)
        # Replace only the versioned factorizer object. Both implementations
        # have the identical four projection tensors and exact initialization.
        with torch.random.fork_rng(devices=[]):
            self.osmf_28_1 = OSMFV13Factorizer(in_channels=512)
        self.from_scratch_layers[-1] = self.osmf_28_1


class Net_CAM(Net):
    forward_cam = V12NetCAM.forward_cam

    def forward(self, x):
        return super().forward(x)[4]
