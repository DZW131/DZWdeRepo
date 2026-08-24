"""SSHR with one frozen EXP-FDHR-003 operator at HFRM28_1."""

from __future__ import annotations

from network import resnet38_cls as baseline
from research.wdch import HFRMFDHR


def _install_fdhr(model, kernel_size: int, variant: str) -> None:
    model.hfrm_28_1 = HFRMFDHR(
        in_channels=512,
        deep_channels=4096,
        kernel_size=kernel_size,
        variant=variant,
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
    model.fdhr_variant = str(variant).upper()
    model.wdch_kernel_size = int(kernel_size)


def _parameter_groups(model):
    groups = baseline.Net.get_parameter_groups(model)
    groups[2].append(model.hfrm_28_1.gamma_veto)
    groups[2].append(model.hfrm_28_1.gamma_context)
    return groups


class Net(baseline.Net):
    def __init__(self, n_class: int, fdhr_variant: str, wdch_kernel_size: int = 7):
        super().__init__(n_class)
        _install_fdhr(self, wdch_kernel_size, fdhr_variant)

    def get_parameter_groups(self):
        return _parameter_groups(self)


class Net_CAM(baseline.Net_CAM):
    def __init__(self, n_class: int, fdhr_variant: str, wdch_kernel_size: int = 7):
        super().__init__(n_class)
        _install_fdhr(self, wdch_kernel_size, fdhr_variant)

    def get_parameter_groups(self):
        return _parameter_groups(self)
