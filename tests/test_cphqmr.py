from pathlib import Path

import pytest
import torch

from network.cphqmr import CPHQMR, DetailGuidedSpatialRestoration, coverage_preserving_fusion
from network.cphqmr_net import CPHQMRNet
from network.hqmr_net import HQMRNet
from tools.eval_cphqmr_full25_bcss_seed42 import MODES, decide
from tools.run_cphqmr_full25_bcss_seed42 import health_summary


ROOT = Path(__file__).resolve().parents[1]


def _inputs(queries=5, device="cpu"):
    torch.manual_seed(7)
    return (torch.randn(2, queries, 256, device=device), torch.randn(2, 256, 4, 4, device=device),
            torch.randn(2, 128, 4, 4, device=device), torch.randn(2, 256, 8, 8, device=device))


def test_cov_query_equals_ln_ccra_query():
    module = CPHQMR(); q, h5, h4, _ = _inputs()
    out = module(q, h5, h4); assert torch.equal(out["q_cov"], module.query_norm(q))


def test_cov_query_never_region_updated():
    module = CPHQMR(); q, h5, h4, _ = _inputs(); out = module(q, h5, h4)
    assert torch.equal(out["q_cov"], module.query_norm(q)) and not torch.equal(out["q_cov"], out["q_disc"])


def test_disc_query_region_updated():
    out = CPHQMR()(*_inputs()[:3]); assert float((out["q_disc"] - out["q_cov"]).detach().abs().sum()) > 0


def test_cov_disc_states_separate():
    out = CPHQMR()(*_inputs()[:3]); assert out["q_cov"].data_ptr() != out["q_disc"].data_ptr()


def test_h4_cov_logit_uses_qcov():
    module = CPHQMR(); q, h5, h4, _ = _inputs(); out = module(q, h5, h4)
    k4, _ = module.scale4(h4); expected = out["C5"] + torch.einsum("bqd,bdhw->bqhw", out["q_cov"], k4) / 16
    assert torch.allclose(out["C4"], expected)


def test_h4_disc_logit_uses_qdisc():
    module = CPHQMR(); q, h5, h4, _ = _inputs(); out = module(q, h5, h4)
    k4, _ = module.scale4(h4); expected = out["C5"] + torch.einsum("bqd,bdhw->bqhw", out["q_disc"], k4) / 16
    assert torch.allclose(out["D4"], expected)


def test_cfr_if_c_le_d_equals_d():
    c, d = torch.tensor([-2., 0., 1.]), torch.tensor([-1., 0., 2.])
    assert torch.equal(coverage_preserving_fusion(c, d)["fused"], d)


def test_cfr_if_c_gt_d_between_d_and_c():
    c, d = torch.tensor([2., 0., -1.]), torch.tensor([1., -2., -3.])
    fused = coverage_preserving_fusion(c, d)["fused"]
    assert torch.all(fused > d) and torch.all(fused <= c)


def test_cfr_no_threshold_or_tunable_lambda():
    source = (ROOT / "network/cphqmr.py").read_text()
    assert "threshold" not in source.lower() and "lambda" not in source.lower() and "topk" not in source.lower()


def test_h4_is_semantic_endpoint_and_no_qk_affinity_on_h3_primary():
    source = (ROOT / "network/cphqmr.py").read_text()
    assert '"h4_semantic_endpoint": True' in source
    assert 'mode == "old_f3_semantic"' in source and source.count("direct_affinity(q_disc, h3)") == 1


def test_stage2_stops_at_h4():
    out = CPHQMR()(*_inputs()[:3]); assert out["dgsr"] is None and out["basis"].shape[-2:] == (4, 4)


def test_stage3_semantic_stops_at_h4():
    out = CPHQMR()(*_inputs()); assert out["semantic_logits"].shape[-2:] == (4, 4) and out["basis"].shape[-2:] == (8, 8)


def test_dgsr_query_and_class_agnostic():
    module = DetailGuidedSpatialRestoration(); logits = torch.randn(1, 2, 4, 4); logits[:, 1] = logits[:, 0]
    out = module(logits, torch.randn(1, 256, 8, 8)); assert torch.equal(out["logits"][:, 0], out["logits"][:, 1])
    assert out["gate"].shape[1] == 1 and out["kernel"].shape[1] == 9


def test_dynamic_kernel_shape_sum_one_nonnegative():
    out = DetailGuidedSpatialRestoration()(torch.randn(2, 3, 4, 4), torch.randn(2, 256, 8, 8))
    assert out["kernel"].shape == (2, 9, 8, 8) and torch.all(out["kernel"] >= 0)
    assert torch.allclose(out["kernel"].sum(1), torch.ones(2, 8, 8), atol=1e-6)


def test_restore_gate_range():
    gate = DetailGuidedSpatialRestoration()(torch.randn(1, 2, 4, 4), torch.randn(1, 256, 8, 8))["gate"]
    assert float(gate.detach().min()) >= 0 and float(gate.detach().max()) <= 1


def test_output_is_convex_blend():
    out = DetailGuidedSpatialRestoration()(torch.randn(1, 2, 4, 4), torch.randn(1, 256, 8, 8))
    expected = (1 - out["gate"]) * out["upsampled"] + out["gate"] * out["restored"]
    assert torch.equal(out["logits"], expected)


def test_bilinear_align_corners_false():
    assert 'mode="bilinear", align_corners=False' in (ROOT / "network/cphqmr.py").read_text()


def test_dgsr_gradient_to_h3():
    h3 = torch.randn(1, 256, 8, 8, requires_grad=True)
    DetailGuidedSpatialRestoration()(torch.randn(1, 2, 4, 4), h3)["logits"].sum().backward()
    assert h3.grad is not None and float(h3.grad.abs().sum()) > 0


def test_dgsr_no_query_semantic_projection():
    names = [name for name, _ in DetailGuidedSpatialRestoration().named_parameters()]
    assert not any("query" in name or "key" in name for name in names)


def test_ccra_and_tri_state_unchanged():
    source = (ROOT / "network/cphqmr_net.py").read_text()
    assert "super().forward" in source and "tri_state" not in source and "ccra2" not in source and "ccra3" not in source


def test_w_detached():
    source = (ROOT / "network/cphqmr_net.py").read_text(); assert '["weights"].detach()' in source


def test_stage1_and_loss_weights_unchanged():
    source = (ROOT / "network/cphqmr_net.py").read_text()
    assert 'mask_losses = [output["losses"]["loss_mask_stage1"]]' in source and "STAGE_WEIGHTS" in source
    assert '.50 * output["losses"]["loss_deep"] + .25 * output["losses"]["loss_pca"] + .25 * loss_mask' in source


def test_no_aux_loss_or_propagation():
    source = (ROOT / "network/cphqmr_net.py").read_text()
    assert '"new_auxiliary_loss": False' in source and '"propagation": False' in source
    assert "CCAC" not in source and "DFSC" not in source and "mcc_complete" not in source


def test_synthetic_coverage_rescue():
    out = coverage_preserving_fusion(torch.tensor([3.]), torch.tensor([-2.]))["fused"]
    assert -2 < float(out) <= 3


def test_synthetic_discrimination_preservation():
    d, c = torch.tensor([10.]), torch.tensor([12.]); fused = coverage_preserving_fusion(c, d)["fused"]
    assert float((fused - d).abs()) < 1e-3


def test_synthetic_dgsr_edge_and_shared_restoration():
    logits = torch.tensor([[[[-5., -5.], [5., 5.]], [[-5., -5.], [5., 5.]]]])
    kernel = torch.zeros(1, 9, 2, 2); kernel[:, 4] = 1
    restored = DetailGuidedSpatialRestoration.dynamic_restore(logits, kernel)
    assert torch.equal(restored[:, 0], restored[:, 1]) and torch.equal(restored, logits)


def test_parameter_delta_is_lighter_than_hqmr_v1():
    old, new = sum(p.numel() for p in HQMRNet().parameters()), sum(p.numel() for p in CPHQMRNet().parameters())
    assert old == 112_269_530 and new == 111_956_196 and new - old == -313_334


def test_epoch5_health_is_diagnostic_only():
    rows = [{"near_full_fraction": 1.0, "empty_fraction": 0.0,
             "all_query_masks_identical": False, "finite": True}]
    summary = health_summary(rows)
    assert summary["diagnostic_only"] is True
    assert summary["action"] == "CONTINUE_FULL25_UNCHANGED"


def test_final_ablation_set_is_frozen():
    assert list(MODES.values()) == ["full", "discriminative_only", "coverage_only", "simple_average", "bilinear_only", "old_f3_semantic"]


def test_decision_strong_go_contract():
    decision = decide(.6, .1, 1.2, {"0": 0., "1": -.5, "2": .2, "3": .1},
                      {"CoverageGain": .06, "UncoveredReduction": .01, "PurityDelta": -.01, "RivalDelta": .01, "improved": True},
                      {"dual_state": True})
    assert decision == "CPHQMR_FULL25_STRONG_GO"


def test_all_ablation_modes_finite_and_bounded():
    module = CPHQMR(); q, h5, h4, h3 = _inputs(queries=2)
    for mode in module.MODES:
        out = module(q, h5, h4, h3, mode); assert torch.isfinite(out["basis"]).all()
        assert 0 <= float(out["basis"].detach().min()) <= float(out["basis"].detach().max()) <= 1


def test_cuda_bf16_autocast():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip("CUDA BF16 required")
    module = CPHQMR().cuda(); inputs = _inputs(queries=2, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16): out = module(*inputs)
    assert torch.isfinite(out["basis"]).all()
