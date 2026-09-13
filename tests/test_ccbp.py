from pathlib import Path

import pytest
import torch

from network.ccbp import CCBP, balanced_competitive_loss
from network.hqmr import class_mixture
from tools.eval_ccbp_full25_bcss_seed42 import decide


ROOT = Path(__file__).resolve().parents[1]


def inputs(batch=2, queries=5, height=4, width=4):
    torch.manual_seed(42)
    basis = torch.rand(batch, queries, height * 2, width * 2)
    weights = torch.softmax(torch.randn(batch, queries, 4), dim=1)
    query = torch.randn(batch, queries, 256)
    key4 = torch.randn(batch, 256, height, width)
    prior = torch.sigmoid(torch.randn(batch, 4))
    present = torch.ones(batch, 4, dtype=torch.bool)
    return basis, weights, query, key4, prior, present


def test_identity_init_pq():
    module = CCBP(); assert torch.equal(module.query_projection.weight, torch.eye(256))


def test_identity_init_pz():
    module = CCBP(); assert torch.equal(module.semantic_projection.weight[:, :, 0, 0], torch.eye(256))


def test_gamma_init_and_range():
    module = CCBP(); assert float(module.gamma) == pytest.approx(5.0, abs=1e-6)
    for value in (-100.0, 100.0):
        module.theta_gamma.data.fill_(value); assert 1 <= float(module.gamma) <= 20


def test_class_prototype_weighting_and_detach():
    query = torch.randn(1, 3, 256, requires_grad=True); weights = torch.softmax(torch.randn(1, 3, 4), 1).requires_grad_()
    result = CCBP.class_prototype(query, weights)
    expected = torch.einsum("bqc,bqd->bcd", weights.detach(), query.detach())
    assert torch.equal(result, expected) and not result.requires_grad


def test_rival_excludes_self():
    logits = torch.tensor([[[[9.]], [[3.]], [[2.]], [[1.]]]])
    rival = CCBP.rival(logits, torch.ones(1, 4, dtype=torch.bool))
    assert float(rival[0, 0]) == 3 and float(rival[0, 1]) == 9


def test_gate_range():
    result = CCBP()(*inputs()); assert bool((result["gate"] > 0).all() and (result["gate"] <= 1).all())


def test_rival_suppression_and_winner_no_suppression():
    module = CCBP(); basis, weights, query, key4, prior, present = inputs(batch=1)
    result = module(basis, weights, query, key4, prior, present)
    winners = result["logits"] >= result["rival"]
    assert torch.equal(result["gate_h4"][winners], torch.ones_like(result["gate_h4"][winners]))
    losers = ~winners; assert bool((result["gate_h4"][losers] < 1).all())


def test_single_present_class_gate_one():
    values = list(inputs(batch=1)); values[-1] = torch.tensor([[False, False, True, False]])
    assert torch.equal(CCBP()(*values)["gate"], torch.ones(1, 4, 8, 8))


def test_gate_upsample_shape_and_class_conditioned_basis_shape():
    basis, *rest = inputs(); result = CCBP()(basis, *rest)
    assert result["gate"].shape == (2, 4, 8, 8)
    assert result["purified_basis"].shape == (2, 5, 4, 8, 8)


def test_off_is_exact_hqmr_mixture():
    basis, weights, query, key4, prior, present = inputs()
    result = CCBP()(basis, weights, query, key4, prior, present, mode="off")
    assert torch.allclose(result["mixture"], class_mixture(basis, weights), atol=1e-7, rtol=0)
    assert torch.equal(result["purified_basis"], basis[:, :, None].expand(-1, -1, 4, -1, -1))


def test_balanced_ce_is_mean_of_class_means():
    logits = torch.tensor([[[[2., 1.]], [[0., 3.]], [[0., 0.]], [[0., 0.]]]])
    positive = torch.zeros_like(logits, dtype=torch.bool); positive[0, 0, 0, 0] = True; positive[0, 1, 0, 1] = True
    loss, detail = balanced_competitive_loss(logits, positive, torch.ones(1, 4, dtype=torch.bool))
    one = torch.nn.functional.cross_entropy(logits[0, :, 0, 0][None], torch.tensor([0]))
    two = torch.nn.functional.cross_entropy(logits[0, :, 0, 1][None], torch.tensor([1]))
    assert torch.allclose(loss, (one + two) / 2) and detail["classes_used"] == 2


def test_background_and_uncertain_are_ignored():
    logits = torch.randn(1, 4, 2, 2, requires_grad=True); positive = torch.zeros(1, 4, 2, 2, dtype=torch.bool)
    loss, detail = balanced_competitive_loss(logits, positive, torch.ones(1, 4, dtype=torch.bool))
    assert float(loss) == 0 and detail["background_ignored"] and detail["uncertain_ignored"]


def test_q_and_k4_inputs_stopgrad_but_purifier_trains():
    basis, weights, query, key4, prior, present = inputs(batch=1); query.requires_grad_(); key4.requires_grad_(); prior.requires_grad_()
    module = CCBP(); result = module(basis, weights, query, key4, prior, present)
    positive = torch.zeros(1, 4, 4, 4, dtype=torch.bool); positive[:, 0, 0, 0] = True; positive[:, 1, 1, 1] = True
    loss, _ = balanced_competitive_loss(result["logits"], positive, present); loss.backward()
    assert query.grad is None and key4.grad is None and prior.grad is None
    assert module.query_projection.weight.grad is not None and module.semantic_projection.weight.grad is not None
    assert module.theta_gamma.grad is not None


def test_base_loss_has_no_purifier_gradient():
    values = list(inputs(batch=1)); values[0].requires_grad_(); module = CCBP(); result = module(*values); result["mixture"].sum().backward()
    assert all(parameter.grad is None for parameter in module.parameters())


def test_purifier_loss_has_no_base_gradient():
    values = list(inputs(batch=1)); values[0].requires_grad_(); values[1].requires_grad_(); values[2].requires_grad_(); values[3].requires_grad_()
    module = CCBP(); result = module(*values); positive = torch.zeros(1, 4, 4, 4, dtype=torch.bool); positive[:, 0, 0, 0] = True
    loss, _ = balanced_competitive_loss(result["logits"], positive, values[-1]); loss.backward()
    assert all(value.grad is None for value in values[:4])


def test_grad_base_total_equals_base():
    values = list(inputs(batch=1)); values[0].requires_grad_(); module = CCBP(); result = module(*values)
    base = result["mixture"].sum(); positive = torch.zeros(1, 4, 4, 4, dtype=torch.bool); positive[:, 0, 0, 0] = True
    aux, _ = balanced_competitive_loss(result["logits"], positive, values[-1])
    grad_base = torch.autograd.grad(base, values[0], retain_graph=True)[0]
    grad_total = torch.autograd.grad(base + aux, values[0])[0]
    assert torch.equal(grad_base, grad_total)


def test_all_classes_are_symmetric():
    source = (ROOT / "network/ccbp.py").read_text(); assert "class == 2" not in source and "class == 3" not in source


def test_no_threshold_topk_or_morphology():
    source = (ROOT / "network/ccbp.py").read_text().lower()
    assert "topk" not in source and "top_k" not in source and "morpholog" not in source and "threshold" not in source


def test_stage1_stage2_are_not_modified_by_ccbp_net():
    source = (ROOT / "network/ccbp_net.py").read_text()
    assert 'stage3 = output["stages"][2]' in source and 'output["stages"][0]' not in source and 'output["stages"][1]' not in source


def test_hqmr_feature_tap_does_not_change_basis():
    source = (ROOT / "network/hqmr.py").read_text()
    assert '"key4": k4' in source and '"basis": final_logits.sigmoid()' in source


def test_parameter_delta_is_131073():
    module = CCBP(); assert sum(p.numel() for p in module.parameters()) == 2 * 256 * 256 + 1


def test_finite_bf16():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip("CUDA BF16 required")
    module = CCBP().cuda(); values = [value.cuda() for value in inputs(batch=1)]
    with torch.autocast("cuda", dtype=torch.bfloat16): result = module(*values)
    assert all(torch.isfinite(result[key]).all() for key in ("logits", "gate", "purified_basis", "mixture"))


def test_supported_ablation_modes():
    assert CCBP.MODES == {"full", "off", "raw_space", "no_g_prior", "mean_rival", "hard_gate"}


def test_frozen_decision_precedence_and_strong_go():
    passed = {"passed": True, "worsened": False}; stable = {"passed": False, "worsened": False}
    classes = {str(cls): 0.0 for cls in range(4)}
    assert decide(.60, .10, .40, classes, {"2": .1, "3": .1}, passed, passed, False, .25) == "CCBP_FULL25_STRONG_GO"
    assert decide(.60, .10, -.30, classes, {"2": .1, "3": .1}, stable, stable, False, .25) == "CCBP_FULL25_NOGO"
    assert decide(.10, -.10, 0.0, classes, {"2": .1, "3": .1}, stable, stable, False, .10) == "CCBP_FULL25_NEUTRAL"
