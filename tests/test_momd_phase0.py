import inspect

import torch

from network.cqrf_net import CQRFNet, STAGE_WEIGHTS
from network.hqrf_targets import IGNORE, circular_locality
from network.momd import mixture_class_bce, mixture_decode, resize_responsibility, route_one_class
from network.momd_net import MOMDNet
from tools.momd_diagnostics import _exact_top_fraction_support


def _inputs(batch=2, queries=196, classes=4, memory_hw=(7, 7), output_hw=(56, 56)):
    spatial = memory_hw[0] * memory_hw[1]
    r = torch.softmax(torch.randn(batch, queries, spatial, classes), dim=1).requires_grad_()
    logits = torch.randn(batch, queries, *output_hw, requires_grad=True)
    locality = circular_locality(14, output_hw, 1)
    return logits, r, locality


def test_momd_resize_contract_and_detach():
    _, r, _ = _inputs(batch=1)
    value = resize_responsibility(r, (7, 7), (56, 56), 2)
    assert value.shape == (1, 196, 56, 56)
    assert not value.requires_grad
    assert torch.allclose(value.sum(1), torch.ones_like(value[:, 0]), atol=1e-5)


def test_momd_locality_fallback_and_query_axis_renorm():
    _, r, locality = _inputs(batch=1)
    routed = route_one_class(r, (7, 7), (56, 56), locality, 0)
    assert torch.allclose(routed["routing"].sum(1), torch.ones(1, 56, 56), atol=1e-5)
    zero = torch.zeros_like(locality)
    fallback = route_one_class(r, (7, 7), (56, 56), zero, 0)
    assert fallback["fallback"].all()
    assert torch.allclose(fallback["routing"].sum(1), torch.ones(1, 56, 56), atol=1e-5)


def test_momd_mass_conservation_posterior_and_range():
    logits, r, locality = _inputs(batch=1)
    result = mixture_decode(logits, r, (7, 7), locality, materialize=True)
    assert result["mixture"].shape == (1, 4, 56, 56)
    assert torch.equal(result["primary_output"], result["mixture"])
    assert float(result["mixture"].min().detach()) >= 0 and float(result["mixture"].max().detach()) <= 1
    assert result["routing_sum_error_max"] <= 1e-5
    assert result["contribution_sum_error_max"] <= 1e-6
    assert result["posterior_sum_error_max"] <= 1e-4
    assert torch.allclose(result["contribution"], result["routing"] * result["base_probability"][:, :, None])
    assert not result["routing"].requires_grad and result["base_probability"].requires_grad


def test_momd_identical_experts_preserve_probability():
    _, r, locality = _inputs(batch=1)
    logits = torch.full((1, 196, 56, 56), torch.logit(torch.tensor(.73)))
    final = mixture_decode(logits, r, (7, 7), locality)["mixture"]
    assert torch.allclose(final, torch.full_like(final, .73), atol=1e-5)


def test_momd_stage23_class_level_loss_and_gradients():
    probability = torch.sigmoid(torch.randn(2, 4, 56, 56, requires_grad=True))
    target = torch.full((2, 4, 7, 7), IGNORE, dtype=torch.int8)
    target[:, :, :2, :2] = 1; target[:, :, 2:4, :2] = 0
    present = torch.ones(2, 4, dtype=torch.bool)
    loss, detail = mixture_class_bce(probability, target, present)
    loss.backward()
    assert torch.isfinite(loss) and probability.grad_fn is not None
    assert detail["class_level_mixture_bce"] and not detail["query_assignment"]
    assert not detail["ownership_weighted_bce"] and not detail["second_responsibility_weighting"]


def test_momd_zero_parameters_and_frozen_stage_contract():
    cqrf = CQRFNet(); momd = MOMDNet()
    assert sum(p.numel() for p in cqrf.parameters()) == sum(p.numel() for p in momd.parameters())
    assert dict(cqrf.named_parameters()).keys() == dict(momd.named_parameters()).keys()
    assert STAGE_WEIGHTS == (.20, .30, .50)


def test_momd_stage_presence_primary_and_pmec_diagnostic_source():
    source = inspect.getsource(MOMDNet.forward)
    assert 'if stage_index >= 2' in source
    assert '"primary_output": stages[-1]["momd"]["mixture"]' in source
    assert 'pmec(final["base_mask_logits"].detach()' in source
    assert '"pmec_diagnostic_only"] = True' in source
    for prohibited in ("topk", "load_balance", "entropy_loss", "aux_loss", "ownership_routed_bce", "relative"):
        assert prohibited not in source.lower()


def test_momd_implementation_has_no_rpmc_or_comd_path():
    source = (inspect.getsource(mixture_decode) + inspect.getsource(mixture_class_bce)).lower()
    assert "softmax" not in source
    assert "argmax" not in source
    assert "p_class" not in source
    assert "relative" not in source
    assert "gate" not in source


def test_contribution_support_is_exact_top20_under_zero_ties():
    value=torch.zeros(3,56,56); value[:,0,0]=1
    support=_exact_top_fraction_support(value,.20)
    assert torch.equal(support.flatten(1).sum(1),torch.full((3,),628))
