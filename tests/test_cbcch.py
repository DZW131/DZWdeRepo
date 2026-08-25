import pytest
import torch
import torch.nn.functional as F

from network import resnet38_cls
from network.resnet38_cbcch import Net as CBCCHNet
from research.wdch import (
    FixedHaarDWT2D,
    HFRMCBCCH,
    LocalSemanticAffinity,
    contrastive_affinity_loss,
)


def test_local_affinity_is_parameter_free_and_normalized():
    torch.manual_seed(7)
    operator = LocalSemanticAffinity()
    semantic = F.normalize(torch.rand(2, 4, 8, 8), dim=1)
    affinity, validity = operator._affinity(semantic)
    assert list(operator.parameters()) == []
    torch.testing.assert_close(
        affinity.sum(dim=1), torch.ones_like(affinity[:, 0]), rtol=1e-6, atol=1e-6
    )
    assert torch.count_nonzero(affinity[validity == 0]) == 0


def test_channel_chunking_is_exactly_equivalent():
    torch.manual_seed(8)
    value = torch.randn(2, 7, 8, 8)
    semantic = F.normalize(torch.rand(2, 4, 8, 8), dim=1)
    chunked = LocalSemanticAffinity(channel_chunk=2)(value, semantic)
    unchunked = LocalSemanticAffinity(channel_chunk=64)(value, semantic)
    torch.testing.assert_close(chunked, unchunked, rtol=1.0e-6, atol=1.0e-6)


def test_a2_and_a3_follow_the_confirmed_final_equations():
    torch.manual_seed(9)
    probe = torch.nn.Conv2d(3, 4, 1)
    value = torch.randn(1, 3, 8, 8)
    deep = torch.randn(1, 6, 2, 2)
    a2 = HFRMCBCCH(3, deep_channels=6, variant="A2")
    a3 = HFRMCBCCH(3, deep_channels=6, variant="A3")
    a2.set_semantic_probe(probe)
    a3.set_semantic_probe(probe)
    semantic = F.normalize(F.relu(probe(value)), dim=1, eps=1.0e-6)
    propagated = a2.affinity(value, semantic)
    torch.testing.assert_close(a2.context(value), propagated)
    boundary = a3.context_with_diagnostics(value)[1]["boundary_map_mean"]
    selected = a3.context(value)
    from research.wdch.bcch import _detached_boundary_map

    boundary_map = _detached_boundary_map(a3.haar, value)
    torch.testing.assert_close(
        selected, (1.0 - boundary_map) * propagated + boundary_map * value
    )
    assert 0.0 <= float(boundary) <= 1.0


def test_existing_ic1_probe_is_shared_without_duplicate_state_or_parameters():
    model = CBCCHNet(4, variant="A3")
    c0 = resnet38_cls.Net(4)
    assert isinstance(model.hfrm_28_1, HFRMCBCCH)
    assert model.hfrm_28_1._semantic_probe is model.ic1
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in c0.parameters()
    )
    state = model.state_dict()
    assert "hfrm_28_1.context_conv.weight" in state
    assert not any("semantic_probe" in key for key in state)
    assert sum(key == "ic1.weight" for key in state) == 1
    groups = model.get_parameter_groups()
    grouped = [parameter for group in groups for parameter in group]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in trainable
    }


@pytest.mark.parametrize("variant", ["A2", "A3"])
def test_contrastive_loss_is_finite_and_updates_existing_probe(variant):
    torch.manual_seed(10)
    feature = torch.randn(2, 3, 8, 8, requires_grad=True)
    probe = torch.nn.Conv2d(3, 4, 1)
    logits = probe(feature)
    labels = torch.ones(2, 4)
    loss, stats = contrastive_affinity_loss(
        feature, logits, labels, FixedHaarDWT2D(), variant=variant
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["valid_anchors"] > 0
    assert probe.weight.grad is not None
    assert torch.isfinite(probe.weight.grad).all()
    assert float(probe.weight.grad.norm()) > 0.0
    if variant == "A3":
        assert stats["valid_anchors"] <= 2 * int(0.20 * 64 + 0.999999)


def test_frozen_constants_and_variants_are_rejected():
    with pytest.raises(ValueError, match="CH15"):
        LocalSemanticAffinity(kernel_size=7)
    with pytest.raises(ValueError, match="Unknown CBCCH variant"):
        HFRMCBCCH(3, deep_channels=6, variant="A4")
    feature = torch.randn(1, 3, 8, 8)
    logits = torch.randn(1, 4, 8, 8)
    labels = torch.ones(1, 4)
    with pytest.raises(ValueError, match="constants are frozen"):
        contrastive_affinity_loss(
            feature,
            logits,
            labels,
            FixedHaarDWT2D(),
            variant="A3",
            temperature=0.1,
        )
