from pathlib import Path

import torch

from network.ccac import ccac_complete
from network.ccac_net import CCACNet
from network.gcqm_net import GCQMNet


ROOT = Path(__file__).resolve().parents[1]


def _case(classes=4):
    torch.manual_seed(11)
    mask = torch.rand(2, classes, 7, 6, requires_grad=True)
    feature = torch.randn(2, 12, 7, 6, requires_grad=True)
    return mask, feature, ccac_complete(mask, feature)


def test_output_is_finite_bounded_and_fill_only():
    mask, _, result = _case()
    restored = result["restored"]
    assert torch.isfinite(restored).all()
    assert float(restored.min()) >= 0 and float(restored.max()) <= 1
    assert torch.all(restored >= mask.detach() - 1e-7)
    assert result["diagnostics"]["max_negative_change"] == 0


def test_exactly_two_iterations_and_shape_preserved():
    mask, _, result = _case()
    assert result["restored"].shape == mask.shape
    assert len(result["iteration_deltas"]) == 2


def test_ccac_affinity_nonnegative_row_sum_and_self_loop():
    _, _, result = _case()
    affinity = result["affinity"]
    assert torch.all(affinity >= 0)
    assert torch.allclose(affinity.sum(1), torch.ones_like(affinity[:, 0]), atol=1e-6)
    assert torch.all(affinity[:, 4] > 0)


def test_ccac_local_3x3_only():
    mask = torch.zeros(1, 1, 7, 7); mask[0, 0, 3, 3] = 1
    feature = torch.ones(1, 2, 7, 7)
    one = ccac_complete(mask, feature, iterations=1)["restored"]
    assert float(one[0, 0, 3, 5]) == 0 and float(one[0, 0, 1, 3]) == 0


def test_pixel_feature_and_affinity_are_detached_but_mask_has_gradient():
    mask, feature, result = _case()
    result["restored"].sum().backward()
    assert mask.grad is not None and torch.isfinite(mask.grad).all()
    assert feature.grad is None and not result["affinity"].requires_grad


def test_rival_gate_protects_competing_class():
    mask = torch.zeros(1, 2, 3, 3); mask[:, 1] = 1
    feature = torch.ones(1, 2, 3, 3)
    guarded = ccac_complete(mask, feature, use_rival=True)["restored"]
    unguarded = ccac_complete(mask, feature, use_rival=False)["restored"]
    assert torch.equal(guarded[:, 0], mask[:, 0])
    assert torch.all(unguarded[:, 0] >= guarded[:, 0])


def test_ccac_rival_gate_range_and_detached():
    mask = torch.zeros(1, 2, 3, 3, requires_grad=True)
    with torch.no_grad(): mask[:, 0, :, 0] = 1; mask[:, 1, :, 1] = .8
    out = ccac_complete(mask, torch.ones(1, 2, 3, 3), iterations=1)["restored"][:, 0].sum()
    cross = torch.autograd.grad(out, mask, retain_graph=True)[0][:, 1]
    assert torch.equal(cross, torch.zeros_like(cross))


def test_no_change_when_consensus_not_higher():
    mask = torch.ones(1, 2, 4, 4) * .7
    out = ccac_complete(mask, torch.randn(1, 3, 4, 4))["restored"]
    assert torch.equal(out, mask)


def test_fill_when_neighbors_high():
    mask = torch.zeros(1, 2, 3, 3); mask[:, 0] = 1; mask[:, 0, 1, 1] = 0
    out = ccac_complete(mask, torch.ones(1, 2, 3, 3), iterations=1)["restored"]
    assert float(out[0, 0, 1, 1]) > 0


def test_unrelated_negative_cosine_neighbor_does_not_contribute():
    mask = torch.zeros(1, 1, 3, 3); mask[0, 0, 1, 0] = 1
    feature = torch.zeros(1, 2, 3, 3); feature[:, 0] = 1; feature[0, :, 1, 0] = torch.tensor([-1., 0.])
    out = ccac_complete(mask, feature, iterations=1)["restored"]
    assert float(out[0, 0, 1, 1]) == 0


def test_uniform_and_off_ablation_contracts():
    mask, feature, _ = _case()
    uniform = ccac_complete(mask, feature, affinity_mode="uniform")
    off = ccac_complete(mask, feature, iterations=0)
    assert torch.equal(off["restored"], mask)
    assert uniform["diagnostics"]["affinity_mode"] == "uniform"


def test_zero_parameter_and_state_dict_delta():
    gcqm, ccac = GCQMNet(), CCACNet()
    assert sum(p.numel() for p in ccac.parameters()) == sum(p.numel() for p in gcqm.parameters())
    assert list(ccac.state_dict()) == list(gcqm.state_dict())


def test_stage_and_loss_wiring_is_frozen():
    source = (ROOT / "network/ccac_net.py").read_text(encoding="utf-8")
    assert "stage_index < 2" in source and "iterations=2" in source
    assert 'mixture_class_bce(ccac["restored"]' in source
    assert 'output["base_primary_output"]' in source
    assert 'output["primary_output"] = output["stages"][-1]["ccac"]["restored"]' in source


def test_no_new_auxiliary_loss():
    source = (ROOT / "network/ccac_net.py").read_text(encoding="utf-8")
    assert "loss_ccac" not in source and '"new_auxiliary_loss": 0' in source
