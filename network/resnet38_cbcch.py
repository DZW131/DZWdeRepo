"""SSHR with frozen CBCCH Phase-2 at HFRM28_1 only."""

from __future__ import annotations

from network import resnet38_cls as baseline
from research.wdch import HFRMCBCCH


def _install_cbcch(model, variant: str) -> None:
    model.hfrm_28_1 = HFRMCBCCH(
        in_channels=512,
        deep_channels=4096,
        context_kernel=15,
        variant=variant,
    )
    # Reuse the existing CAM28_1 classifier as the semantic probe. The
    # reference is intentionally non-registering so state_dict keys and the
    # optimizer contain no duplicate parameters.
    model.hfrm_28_1.set_semantic_probe(model.ic1)
    model.from_scratch_layers = [
        model.ic_56,
        model.ic1,
        model.ic2,
        model.fc8,
        model.hfrm_56,
        model.hfrm_28_1,
        model.hfrm_28_2,
    ]
    model.cbcch_contract = (
        "local15-semantic-affinity; A2=P; "
        "A3=(1-B)*P+B*F; contrastive-top20-boundary"
    )


def _parameter_groups(model):
    groups = baseline.Net.get_parameter_groups(model)
    groups[2].append(model.hfrm_28_1.gamma_veto)
    groups[2].append(model.hfrm_28_1.gamma_context)
    return groups


class Net(baseline.Net):
    def __init__(self, n_class: int, variant: str = "A3"):
        super().__init__(n_class)
        _install_cbcch(self, variant)

    def get_parameter_groups(self):
        return _parameter_groups(self)


class Net_CAM(baseline.Net_CAM):
    def __init__(self, n_class: int, variant: str = "A3"):
        super().__init__(n_class)
        _install_cbcch(self, variant)

    def get_parameter_groups(self):
        return _parameter_groups(self)
