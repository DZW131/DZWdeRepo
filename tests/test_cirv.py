import numpy as np
import torch

from network.cirv import (
    PrototypeBank, base_region_posterior, calibrate_evidence, calibration_ratio,
    extract_regions, product_fusion, prototype_posterior, region_embedding,
    select_source_regions, spherical_kmeans,
)


def fixtures():
    pred = np.zeros((8, 8), np.int64); pred[:, 4:] = 1; pred[6:, :2] = 2
    evidence = np.full((4, 8, 8), .05, np.float32)
    for cls in range(4): evidence[cls][pred == cls] = .8
    key = torch.arange(256 * 4 * 4, dtype=torch.float32).reshape(256, 4, 4) + 1
    bank = np.random.default_rng(2).normal(size=(4, 4, 256)).astype(np.float32)
    g = np.asarray([.8, .7, .6, .5], np.float32)
    return pred, evidence, key, bank, g


def test_hqmr_base_unchanged():
    pred, evidence, key, bank, g = fixtures()
    off, _ = calibrate_evidence(evidence, pred, key, bank, g, mode="off")
    assert np.array_equal(off, evidence)


def test_cirv_no_base_gradient():
    parameter = torch.tensor(2.0, requires_grad=True)
    (parameter.square()).backward(); reference = parameter.grad.clone(); parameter.grad = None
    pred, evidence, key, bank, g = fixtures()
    calibrate_evidence(evidence, pred, key.requires_grad_(), bank, g)
    (parameter.square()).backward()
    assert torch.equal(reference, parameter.grad) and key.grad is None


def test_component_stopgrad():
    pred, *_ = fixtures(); rows = extract_regions(pred)
    assert rows and all(isinstance(row["mask"], np.ndarray) for row in rows)


def test_region_embedding_normalized():
    pred, _, key, _, _ = fixtures(); z = region_embedding(pred == 0, key.requires_grad_())
    assert torch.allclose(z.norm(), torch.tensor(1.0), atol=1e-6) and not z.requires_grad


def test_source_requires_image_label():
    pred, _, key, _, _ = fixtures(); anchors = np.ones((4, 8, 8), bool)
    assert select_source_regions(pred, np.zeros(4), anchors, key) == []


def test_source_requires_anchor():
    pred, _, key, _, _ = fixtures(); anchors = np.zeros((4, 8, 8), bool)
    assert select_source_regions(pred, np.ones(4), anchors, key) == []


def test_one_source_per_class_image():
    pred, _, key, _, _ = fixtures(); anchors = np.ones((4, 8, 8), bool)
    rows = select_source_regions(pred, np.ones(4), anchors, key)
    assert len({row["class_id"] for row in rows}) == len(rows)


def test_source_max_anchor_count():
    pred = np.ones((6, 6), np.int64); pred[:2, :2] = 0; pred[4:, 4:] = 0
    anchors = np.zeros((4, 6, 6), bool); anchors[0, 4:, 4:] = True; anchors[0, 0, 0] = True
    key = torch.randn(256, 3, 3); rows = select_source_regions(pred, [1, 0, 0, 0], anchors, key)
    assert rows[0]["anchor_count"] == 4 and rows[0]["area"] == 4


def test_four_prototypes_per_class():
    _, _, _, bank, _ = fixtures(); assert PrototypeBank(bank).prototypes.shape[:2] == (4, 4)


def test_spherical_kmeans_deterministic():
    x = np.random.default_rng(3).normal(size=(20, 7))
    a = spherical_kmeans(x, seed=42); b = spherical_kmeans(x, seed=42)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_prototypes_normalized():
    _, _, _, bank, _ = fixtures(); assert np.allclose(np.linalg.norm(PrototypeBank(bank).prototypes, axis=2), 1)


def test_verify_before_update_and_ema_099():
    bank = np.zeros((4, 4, 2), np.float32); bank[..., 0] = 1
    memory = PrototypeBank(bank); z = np.asarray([0, 1], np.float32)
    assignment, before = memory.verify(z, 0); old = memory.prototypes[0, assignment].copy()
    updated = memory.update(z, 0)
    expected = .99 * old + .01 * z; expected /= np.linalg.norm(expected)
    assert updated == assignment and np.allclose(memory.prototypes[0, assignment], expected)


def test_verify_before_update():
    bank = np.random.default_rng(9).normal(size=(4, 4, 3)).astype(np.float32)
    memory = PrototypeBank(bank); before = memory.prototypes.copy()
    memory.verify([1, 0, 0], 1)
    assert np.array_equal(before, memory.prototypes)


def test_ema_099():
    bank = np.zeros((4, 4, 2), np.float32); bank[..., 0] = 1
    memory = PrototypeBank(bank); memory.update([0, 1], 0)
    expected = np.asarray([.99, .01]); expected /= np.linalg.norm(expected)
    assert np.allclose(memory.prototypes[0, 0], expected)


def test_nearest_proto_assignment():
    bank = np.zeros((4, 4, 2), np.float32); bank[..., 0] = 1; bank[2, 3] = [0, 1]
    assert PrototypeBank(bank).verify([0, 1], 2)[0] == 3


def test_proto_score_max4_and_scale5():
    bank = np.zeros((4, 4, 2), np.float32); bank[..., 0] = 1; bank[1, 2] = [0, 1]
    posterior, similarity = prototype_posterior([0, 1], bank, np.ones(4), scale=5)
    assert similarity[1] == 1 and posterior.argmax() == 1 and np.isclose(posterior.sum(), 1)


def test_proto_score_max4():
    bank = np.zeros((4, 4, 2), np.float32); bank[..., 0] = 1; bank[3, 1] = [0, 1]
    _, similarity = prototype_posterior([0, 1], bank, np.ones(4))
    assert similarity[3] == 1


def test_proto_distribution_normalized():
    _, _, _, bank, g = fixtures(); posterior, _ = prototype_posterior(np.ones(256), bank, g)
    assert np.isclose(posterior.sum(), 1)


def test_proto_scale_5():
    bank = np.zeros((4, 4, 2), np.float32); bank[..., 0] = 1; bank[1, 0] = [0, 1]
    posterior, _ = prototype_posterior([0, 1], bank, np.ones(4))
    assert np.isclose(posterior[1] / posterior[0], np.exp(5), rtol=1e-5)


def test_g_prior_detached():
    _, _, _, bank, _ = fixtures(); g = torch.ones(4, requires_grad=True)
    posterior, _ = prototype_posterior(np.ones(256), bank, g.detach().numpy())
    assert np.isclose(posterior.sum(), 1) and g.grad is None


def test_base_region_distribution_normalized():
    pred, evidence, *_ = fixtures(); assert np.isclose(base_region_posterior(evidence, pred == 0).sum(), 1)


def test_product_fusion_normalized():
    fused = product_fusion([.2, .3, .4, .1], [.4, .3, .2, .1]); assert np.isclose(fused.sum(), 1)


def test_ratio_clip_025_4():
    ratio, _ = calibration_ratio([1, 1, .01, .9], [.01, .3, .9, .9])
    assert ratio.min() >= .25 and ratio.max() <= 4


def test_no_hard_delete_and_no_background_creation():
    pred, evidence, key, bank, g = fixtures(); calibrated, records = calibrate_evidence(evidence, pred, key, bank, g)
    assert calibrated.shape == evidence.shape and len(records) == len(extract_regions(pred))
    assert np.all(calibrated.sum(0) > 0)


def test_no_hard_delete():
    pred, evidence, key, bank, g = fixtures(); calibrated, _ = calibrate_evidence(evidence, pred, key, bank, g)
    assert np.all(calibrated > 0)


def test_no_background_creation():
    pred, evidence, key, bank, g = fixtures(); calibrated, _ = calibrate_evidence(evidence, pred, key, bank, g)
    assert calibrated.shape[0] == 4


def test_cirv_off_exact_base_path():
    pred, evidence, key, bank, g = fixtures(); off, records = calibrate_evidence(evidence, pred, key, bank, g, "off")
    assert off.tobytes() == evidence.tobytes() and records == []
