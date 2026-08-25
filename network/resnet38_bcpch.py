"""SSHR with frozen BCP-CH Phase-3 at HFRM28_1 only."""

from __future__ import annotations

from network import resnet38_cls as baseline
from research.wdch import HFRMBCPCH


def _install_bcpch(model) -> None:
    model.hfrm_28_1 = HFRMBCPCH(
        in_channels=512,
        deep_channels=4096,
        context_kernel=15,
    )
    model.hfrm_28_1.set_shared_probes(model.ic1, model.fc8)
    model.from_scratch_layers = [
        model.ic_56,
        model.ic1,
        model.ic2,
        model.fc8,
        model.hfrm_56,
        model.hfrm_28_1,
        model.hfrm_28_2,
    ]
    model.bcpch_contract = (
        "IDWT(LL,0,0,0)-prototype; CAM>0.7; predicted-fc8-presence; "
        "Y=(1-B)*(0.5*P_affinity+0.5*P_prototype)+B*F"
    )


def _parameter_groups(model):
    groups = baseline.Net.get_parameter_groups(model)
    groups[2].append(model.hfrm_28_1.gamma_veto)
    groups[2].append(model.hfrm_28_1.gamma_context)
    return groups


class Net(baseline.Net):
    def __init__(self, n_class: int):
        super().__init__(n_class)
        _install_bcpch(self)

    def get_parameter_groups(self):
        return _parameter_groups(self)


class Net_CAM(baseline.Net_CAM):
    def __init__(self, n_class: int):
        super().__init__(n_class)
        _install_bcpch(self)

    def get_parameter_groups(self):
        return _parameter_groups(self)
