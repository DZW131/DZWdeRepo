"""SSHR with WD-CH at HFRM28_1 only.

The released ``network.resnet38_cls`` remains untouched and is the C0 model.
"""

from __future__ import annotations

from network import resnet38_cls as baseline
from research.wdch import HFRMWDCH


def _install_wdch(model, kernel_size: int) -> None:
    model.hfrm_28_1 = HFRMWDCH(
        in_channels=512, deep_channels=4096, kernel_size=kernel_size
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


def _parameter_groups(model):
    groups = baseline.Net.get_parameter_groups(model)
    # The released grouping handles scalar gammas through an explicit HFRM
    # isinstance check.  HFRMWDCH preserves those two scalars but has a distinct
    # class, so add them explicitly to the same scratch-weight group.
    groups[2].append(model.hfrm_28_1.gamma_veto)
    groups[2].append(model.hfrm_28_1.gamma_context)
    return groups


class Net(baseline.Net):
    def __init__(self, n_class: int, wdch_kernel_size: int = 7):
        super().__init__(n_class)
        _install_wdch(self, wdch_kernel_size)

    def get_parameter_groups(self):
        return _parameter_groups(self)


class Net_CAM(baseline.Net_CAM):
    def __init__(self, n_class: int, wdch_kernel_size: int = 7):
        super().__init__(n_class)
        _install_wdch(self, wdch_kernel_size)

    def get_parameter_groups(self):
        return _parameter_groups(self)
