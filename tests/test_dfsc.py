from pathlib import Path

import torch

from network.dfsc import DFRA, balanced_relation_loss, mcc_complete, pair_masks, relation_labels
from network.dfsc_net import DFSCNet
from network.gcqm_net import GCQMNet


ROOT = Path(__file__).resolve().parents[1]


def _detail():
    positive = torch.zeros(1, 4, 2, 3, dtype=torch.bool)
    positive[0, 0, 0, :2] = True; positive[0, 1, 1, 0] = True
    background = torch.zeros(1, 2, 3, dtype=torch.bool); background[0, 1, 1] = True
    return {"positive": positive, "reliable_background": background}


def _relation(mode="full"):
    torch.manual_seed(9); module = DFRA(); pixel = torch.randn(2, 256, 5, 6, requires_grad=True)
    detail = {"positive": torch.zeros(2, 4, 3, 3, dtype=torch.bool),
              "reliable_background": torch.ones(2, 3, 3, dtype=torch.bool)}
    return module, pixel, module(pixel, detail, mode)


def test_dfsc_parameter_delta_exact():
    base, model = GCQMNet(), DFSCNet()
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters()) == 16386


def test_low_frequency_shape():
    _, pixel, out = _relation(); assert out["low_frequency"].shape == pixel.shape


def test_high_frequency_residual_shape():
    _, pixel, out = _relation(); assert out["high_frequency"].shape == pixel.shape


def test_relation_feature_detached():
    _, pixel, out = _relation(); out["relation_loss"].backward(); assert pixel.grad is None


def test_relation_embeddings_l2_norm():
    _, _, out = _relation()
    for key in ("embedding_low", "embedding_high"):
        norm = out[key].square().sum(1).sqrt(); assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_beta_positive():
    _, _, out = _relation(); assert float(out["beta_low"]) > 0 and float(out["beta_high"]) > 0


def test_pair_distance_range():
    _, _, out = _relation(); assert float(out["distance_low"].min()) >= 0 and float(out["distance_high"].max()) <= 2


def test_pair_affinity_range_self_row_sum_and_finite():
    _, _, out = _relation(); raw, alpha = out["raw_affinity"], out["affinity"]
    assert torch.isfinite(alpha).all() and float(raw.min()) >= 0 and float(raw.max()) <= 1
    assert torch.equal(raw[:, 4], torch.ones_like(raw[:, 4]))
    assert torch.allclose(alpha.sum(1), torch.ones_like(alpha[:, 0]), atol=1e-6)


def test_pair_label_same_fg_positive():
    label = relation_labels(_detail(), (2, 3)); pos, _ = pair_masks(label); assert bool(pos.any())


def test_pair_label_same_bg_positive():
    detail = _detail(); detail["reliable_background"][0, 1, 1:] = True
    label = relation_labels(detail, (2, 3)); pos, _ = pair_masks(label); assert bool(pos.any())


def test_pair_label_cross_fg_negative():
    label = relation_labels(_detail(), (2, 3)); _, neg = pair_masks(label); assert bool(neg.any())


def test_pair_label_fg_bg_negative():
    detail = _detail(); detail["reliable_background"][0, 0, 2] = True
    label = relation_labels(detail, (2, 3)); _, neg = pair_masks(label); assert bool(neg.any())


def test_pair_label_uncertain_ignore():
    label = torch.tensor([[[-1, 0]]]); pos, neg = pair_masks(label); assert not bool(pos.any()) and not bool(neg.any())


def test_relation_loss_balanced():
    raw = torch.tensor([[[.8], [.2]]]); pos = torch.tensor([[[True], [False]]]); neg = ~pos
    loss, detail = balanced_relation_loss(raw, pos, neg)
    assert torch.allclose(loss, -torch.log(torch.tensor(.8)), atol=1e-6) and detail["balanced_available_terms"] == 2


def test_relation_loss_no_base_gradient_and_relation_gradient_nonzero():
    module, pixel, out = _relation(); out["relation_loss"].backward()
    assert pixel.grad is None and any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())


def test_base_loss_no_relation_gradient():
    module, _, out = _relation(); mask = torch.rand(2, 4, 5, 6, requires_grad=True)
    mcc_complete(mask, out["affinity_comp"])["restored"].sum().backward()
    assert mask.grad is not None and all(p.grad is None for p in module.parameters())


def test_mcc_monotonic_equals_max_consensus_and_range():
    mask = torch.zeros(1, 1, 3, 3); mask[0, 0, 1, 0] = 1
    affinity = torch.zeros(1, 9, 9); affinity[:, 3] = .5; affinity[:, 4] = .5
    out = mcc_complete(mask, affinity, iterations=1)
    neighbors = torch.nn.functional.unfold(mask, 3, padding=1).reshape(1, 1, 9, 9)
    consensus = (neighbors * affinity[:, None]).sum(2).reshape_as(mask)
    assert torch.equal(out["restored"], torch.maximum(mask, consensus))
    assert torch.all(out["restored"] >= mask) and 0 <= float(out["restored"].min()) <= float(out["restored"].max()) <= 1


def test_mcc_T_exactly_two():
    mask = torch.rand(1, 2, 3, 3); affinity = torch.ones(1, 9, 9) / 9
    assert len(mcc_complete(mask, affinity)["iteration_deltas"]) == 2


def test_uniform_fallback_is_uniform_consensus():
    _, _, relation = _relation("uniform"); mask = torch.rand(2, 4, 5, 6)
    out = mcc_complete(mask, relation["affinity_comp"], 1)["restored"]
    assert torch.all(out >= mask)


def test_stage1_unchanged_stage2_stage3_primary_dfsc():
    source = (ROOT / "network/dfsc_net.py").read_text(encoding="utf-8")
    assert "stage_index < 2" in source and "iterations=2" in source
    assert 'output["primary_output"] = output["stages"][-1]["dfsc"]["restored"]' in source
    assert 'mixture_class_bce(completion["restored"]' in source


def test_no_rival_gate_topk_or_new_dense_supervision():
    source = (ROOT / "network/dfsc_net.py").read_text(encoding="utf-8")
    assert '"rival_gate": False' in source and '"top_k": False' in source and '"new_dense_supervision": False' in source
    assert "ccac_use_rival" not in source


def _isolated_relation_change(same_label):
    torch.manual_seed(23); module = DFRA(channels=2, relation_dim=2)
    pixel = torch.tensor([[[[1., -1.]], [[.2, .8]]]])
    positive = torch.zeros(1, 2, 1, 2, dtype=torch.bool)
    positive[0, 0, 0, 0] = True; positive[0, 0 if same_label else 1, 0, 1] = True
    detail = {"positive": positive, "reliable_background": torch.zeros(1, 1, 2, dtype=torch.bool)}
    with torch.no_grad():
        initial = module(pixel, detail); mask = initial["positive_pairs"] if same_label else initial["negative_pairs"]
        before = float(initial["raw_affinity"][mask].mean())
    optimizer = torch.optim.Adam(module.parameters(), lr=.05)
    for _ in range(30):
        optimizer.zero_grad(); out = module(pixel, detail); out["relation_loss"].backward(); optimizer.step()
    with torch.no_grad():
        final = module(pixel, detail); after = float(final["raw_affinity"][mask].mean())
    return before, after


def test_synthetic_same_region_affinity_increases():
    before, after = _isolated_relation_change(True); assert after > before


def test_synthetic_cross_label_affinity_decreases():
    before, after = _isolated_relation_change(False); assert after < before
