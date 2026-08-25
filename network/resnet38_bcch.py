"""SSHR with frozen boundary-aware CH at HFRM28_1 only."""

from __future__ import annotations

from network import resnet38_cls as baseline
from research.wdch import HFRMBCCH


def _install_bcch(model) -> None:
    model.hfrm_28_1 = HFRMBCCH(
        in_channels=512,
        deep_channels=4096,
        context_kernel=15,
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
    model.bcch_contract = "detached-channel-mean-spatial-minmax-alpha=1-B"


def _parameter_groups(model):
    groups = baseline.Net.get_parameter_groups(model)
    groups[2].append(model.hfrm_28_1.gamma_veto)
    groups[2].append(model.hfrm_28_1.gamma_context)
    return groups


class Net(baseline.Net):
    def __init__(self, n_class: int):
        super().__init__(n_class)
        _install_bcch(self)

    def get_parameter_groups(self):
        return _parameter_groups(self)


class Net_CAM(baseline.Net_CAM):
    def __init__(self, n_class: int):
        super().__init__(n_class)
        _install_bcch(self)

    def get_parameter_groups(self):
        return _parameter_groups(self)
