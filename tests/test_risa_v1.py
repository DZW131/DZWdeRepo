import torch

from network.risa_v1 import RISA, risa_losses


def sample(batch=2, queries=7):
    torch.manual_seed(7)
    return (
        torch.randn(batch, 256, 12, 12),
        torch.randn(batch, 512, 6, 6),
        torch.randn(batch, 1024, 6, 6),
        torch.sigmoid(torch.randn(batch, queries, 12, 12)),
    )


def test_risa_shape():
    out = RISA()(*sample(), hard_refinement=True)
    assert out["identity_prob"].shape == (2, 7, 4)
    assert out["presence_prob"].shape == (2, 4)
    assert out["class_map"].shape == (2, 4, 12, 12)


def test_risa_gradient_isolation():
    tensors = tuple(value.requires_grad_() for value in sample())
    module = RISA()
    out = module(*tensors)
    out["class_map"].mean().backward()
    assert all(value.grad is None for value in tensors)
    assert any(parameter.grad is not None for parameter in module.parameters())


def test_risa_no_gt_dependency():
    module = RISA()
    arguments = sample()
    first = module(*arguments)["identity_prob"]
    second = module(*arguments)["identity_prob"]
    torch.testing.assert_close(first, second)


def test_region_pooling():
    module = RISA()
    f3, f4, f5, responsibility = sample(batch=1, queries=1)
    responsibility.fill_(1.)
    out = module(f3, f4, f5, responsibility, hard_refinement=False)
    feature = out["identity_feature"].float().flatten(2).mean(-1)
    expected = module.mean_norm(feature[:, None])
    torch.testing.assert_close(out["region_mean"], expected)


def test_hard_rival_selection():
    out = RISA()(*sample(batch=1, queries=3))
    assert out["selected_index"].shape[-1] == int(12 * 12 * .20)
    assert int(out["selected_index"].min()) >= 0
    assert int(out["selected_index"].max()) < 12 * 12


def test_presence_range():
    presence = RISA()(*sample())["presence_prob"]
    assert bool(((presence >= 0) & (presence <= 1)).all())


def test_class_probability_normalization():
    probability = RISA()(*sample())["identity_prob"]
    torch.testing.assert_close(probability.sum(-1), torch.ones_like(probability[..., 0]))


def test_losses_finite():
    output = RISA()(*sample())
    losses = risa_losses(output, torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]]), True)
    assert all(torch.isfinite(value) for value in losses.values())
