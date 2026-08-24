"""SSHR with fixed-strength SC-WDCH at HFRM28_1 only.

The released ``network.resnet38_cls`` and WD-CH v1 implementation remain
untouched.  The sole model change is CH15 -> SC-WDCH in HFRM28_1.
"""

from __future__ import annotations

from network import resnet38_cls as baseline
from research.wdch import HFRMSCWDCH


def _install_scwdch(model, kernel_size: int, scale: float) -> None:
    model.hfrm_28_1 = HFRMSCWDCH(
        in_channels=512,
        deep_channels=4096,
        kernel_size=kernel_size,
        scale=scale,
    )
    model.from_scratch_layers = [
        model.ic_56,
        model.ic1,
        model.ic2,
        model.fc8,
        model.hfrm_56,
        model.hfrm_28_1,
        model.hfrm_28_2,
    ]
    model.wdch_kernel_size = int(kernel_size)
    model.scwdch_scale = float(scale)


def _parameter_groups(model):
    groups = baseline.Net.get_parameter_groups(model)
    groups[2].append(model.hfrm_28_1.gamma_veto)
    groups[2].append(model.hfrm_28_1.gamma_context)
    return groups


class Net(baseline.Net):
    def __init__(
        self,
        n_class: int,
        wdch_kernel_size: int = 7,
        scwdch_scale: float = 1.0,
    ) -> None:
        super().__init__(n_class)
        _install_scwdch(self, wdch_kernel_size, scwdch_scale)

    def get_parameter_groups(self):
        return _parameter_groups(self)


class Net_CAM(baseline.Net_CAM):
    def __init__(
        self,
        n_class: int,
        wdch_kernel_size: int = 7,
        scwdch_scale: float = 1.0,
    ) -> None:
        super().__init__(n_class)
        _install_scwdch(self, wdch_kernel_size, scwdch_scale)

    def get_parameter_groups(self):
        return _parameter_groups(self)
