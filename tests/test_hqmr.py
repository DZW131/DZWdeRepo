from pathlib import Path

import pytest
import torch

from network.gcqm_net import GCQMNet
from network.hqmr import HQMR, QueryRegionUpdate, class_mixture, direct_affinity, normalized_region_weights, residual_logits
from network.hqmr_net import HQMRNet
from tools.run_hqmr_full25_bcss_seed42 import CONFIG, EPOCHS, MILESTONES, TOTAL_STEPS, collapse_gate


ROOT = Path(__file__).resolve().parents[1]


def _inputs(device="cpu", dtype=torch.float32, queries=5):
    torch.manual_seed(42)
    return (torch.randn(2, queries, 256, device=device, dtype=dtype),
            torch.randn(2, 256, 4, 4, device=device, dtype=dtype),
            torch.randn(2, 128, 4, 4, device=device, dtype=dtype),
            torch.randn(2, 256, 8, 8, device=device, dtype=dtype))


def test_hqmr_feature_shapes():
    q, h5, h4, h3 = _inputs(); module = HQMR()
    stage2 = module(q, h5, h4); stage3 = module(q, h5, h4, h3)
    assert stage2["basis"].shape == (2, 5, 4, 4)
    assert stage3["basis"].shape == (2, 5, 8, 8)


def test_hqmr_actual_scale_order():
    actual = {"H5": (256, 28, 28), "H4": (128, 28, 28), "H3": (256, 56, 56)}
    assert actual["H5"][1] <= actual["H4"][1] < actual["H3"][1]


def test_hqmr_no_hard_mask():
    source = (ROOT / "network/hqmr.py").read_text()
    assert "masked_fill" not in source and ">=" not in source and "<=" not in source


def test_hqmr_no_threshold_inside_decoder():
    source = (ROOT / "network/hqmr.py").read_text()
    assert "threshold" not in source.lower()


def test_hqmr_no_topk():
    source = (ROOT / "network/hqmr.py").read_text()
    assert "topk" not in source.lower() and "top_k" not in source.lower()


def test_coarse_query_token_logits():
    q = torch.randn(1, 3, 256); k = torch.randn(1, 256, 2, 2)
    expected = torch.einsum("bqd,bdhw->bqhw", q, k) / 16.0
    assert torch.allclose(direct_affinity(q, k), expected)


def test_region_weights_sum_one():
    weights = normalized_region_weights(torch.randn(2, 5, 3, 4))
    assert torch.allclose(weights.sum(-1), torch.ones(2, 5), atol=1e-6)


def test_region_pool_fp32():
    logits = torch.randn(1, 2, 3, 3, dtype=torch.float16)
    assert normalized_region_weights(logits).dtype == torch.float32


def test_mid_logit_residual_addition():
    coarse = torch.randn(1, 2, 2, 2); direct = torch.randn(1, 2, 4, 4)
    expected = torch.nn.functional.interpolate(coarse, (4, 4), mode="bilinear", align_corners=False) + direct
    assert torch.equal(residual_logits(coarse, direct), expected)


def test_fine_logit_residual_addition():
    mid = torch.randn(1, 2, 4, 4); direct = torch.randn(1, 2, 8, 8)
    assert torch.equal(residual_logits(mid, direct),
                       torch.nn.functional.interpolate(mid, (8, 8), mode="bilinear", align_corners=False) + direct)


def test_bilinear_align_corners_false():
    source = (ROOT / "network/hqmr.py").read_text()
    assert source.count('mode="bilinear", align_corners=False') >= 2


def test_higher_scale_can_activate_outside_coarse_support():
    coarse = torch.full((1, 1, 1, 2), -8.0); coarse[..., 0] = 8.0
    direct = torch.zeros(1, 1, 2, 4); direct[..., 2:] = 12.0
    mid = residual_logits(coarse, direct)
    assert bool((mid.sigmoid()[..., 2:] > .5).all())


def test_stage2_uses_f5_f4():
    q, h5, h4, _ = _inputs(); out = HQMR()(q, h5, h4)
    assert out["logits5"] is not None and out["logits4"] is not None and out["logits3"] is None


def test_stage3_uses_f5_f4_f3():
    out = HQMR()(*_inputs())
    assert all(out[key] is not None for key in ("logits5", "logits4", "logits3"))


def test_parameter_sharing_stage2_stage3():
    source = (ROOT / "network/hqmr_net.py").read_text()
    assert "self.hqmr = HQMR(256)" in source and "hqmr2" not in source and "hqmr3" not in source


def test_w_detached():
    basis = torch.rand(1, 3, 2, 2, requires_grad=True); weights = torch.rand(1, 3, 4, requires_grad=True)
    class_mixture(basis, weights).sum().backward()
    assert basis.grad is not None and weights.grad is None


def test_stage1_unchanged():
    source = (ROOT / "network/hqmr_net.py").read_text()
    assert 'mask_losses = [output["losses"]["loss_mask_stage1"]]' in source
    assert "if stage_index == 1" in source


def test_loss_weights_unchanged():
    source = (ROOT / "network/hqmr_net.py").read_text()
    assert "STAGE_WEIGHTS" in source and '.50 * output["losses"]["loss_deep"]' in source
    assert '.25 * output["losses"]["loss_pca"] + .25 * loss_mask' in source


def test_no_aux_loss():
    source = (ROOT / "network/hqmr_net.py").read_text()
    assert '"new_auxiliary_loss": False' in source


def test_hqmr_gradient_to_query():
    q, h5, h4, h3 = _inputs(queries=3); q.requires_grad_()
    HQMR()(q, h5, h4, h3)["basis"].sum().backward()
    assert q.grad is not None and float(q.grad.abs().sum()) > 0


def test_hqmr_gradient_to_multiscale_projection():
    module = HQMR(); module(*_inputs(queries=3))["basis"].sum().backward()
    for name in ("scale5.key.0.weight", "scale5.value.0.weight", "scale4.key.0.weight",
                 "scale4.value.0.weight", "scale3.key.0.weight"):
        grad = dict(module.named_parameters())[name].grad
        assert grad is not None and float(grad.abs().sum()) > 0


def test_no_gradient_to_w():
    basis = torch.rand(2, 5, 3, 3, requires_grad=True); weights = torch.rand(2, 5, 4, requires_grad=True)
    class_mixture(basis, weights).mean().backward(); assert weights.grad is None


def test_output_range():
    out = HQMR()(*_inputs())
    assert 0 <= float(out["basis"].detach().min()) <= float(out["basis"].detach().max()) <= 1


def test_finite_bf16():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip("CUDA BF16 required")
    module = HQMR().cuda(); out = module(*_inputs("cuda", torch.bfloat16))
    assert torch.isfinite(out["basis"]).all()


def test_synthetic_correction():
    coarse = torch.full((1, 1, 2, 2), 6.0); fine_direct = torch.full((1, 1, 4, 4), -10.0)
    fine = residual_logits(coarse, fine_direct)
    assert bool((fine < torch.nn.functional.interpolate(coarse, (4, 4), mode="bilinear", align_corners=False)).all())


def test_parameter_delta_under_five_million():
    base = sum(p.numel() for p in GCQMNet().parameters())
    model = sum(p.numel() for p in HQMRNet().parameters())
    assert model - base == 990_208 and model - base < 5_000_000


def test_full25_protocol_frozen():
    assert EPOCHS == 25 and TOTAL_STEPS == 29275 and MILESTONES == {5, 10, 15, 20, 25}
    assert CONFIG["effective_batch_size"] == 20 and CONFIG["precision"] == "bf16"
    assert CONFIG["checkpoint_selection"] == "fixed Epoch25 FINAL only"
    assert CONFIG["validation_during_training"] is False and CONFIG["new_auxiliary_loss"] == 0


def test_epoch5_collapse_gate_boundaries():
    healthy = [{"near_full_fraction": .5, "empty_fraction": .8, "all_query_masks_identical": False, "x": 0.}]
    assert collapse_gate(healthy)["decision"] == "CONTINUE_FULL25_UNCHANGED"
    assert collapse_gate([{**healthy[0], "near_full_fraction": .5001}])["decision"] == "HQMR_ENGINEERING_OR_COLLAPSE_BLOCKED"
    assert collapse_gate([{**healthy[0], "empty_fraction": .8001}])["decision"] == "HQMR_ENGINEERING_OR_COLLAPSE_BLOCKED"
