from __future__ import annotations

import inspect

import pytest
import torch

from network.osmf import (
    OSMF_EQUIVARIANCE_INTERVAL,
    OSMF_LAMBDA_MORPH,
    OSMF_LAMBDA_ORTH,
    OSMF_LAMBDA_REC,
    OSMF_LAMBDA_SEM,
    OSMFFactorizer,
    cross_subspace_covariance,
    inverse_align_morphology,
    orthogonality_loss,
    reconstruction_cosine,
    reconstruction_loss,
    semantic_classification_loss,
    spatial_equivariance_loss,
)


@pytest.mark.parametrize("channels,expected", [(512, (256, 256)), (5, (3, 2))])
def test_channel_partition_is_frozen(channels, expected):
    module = OSMFFactorizer(channels, n_class=4)
    assert (module.semantic_channels, module.morphology_channels) == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_partition_initialization_is_exact_identity(dtype):
    module = OSMFFactorizer(8, n_class=4).to(dtype=dtype)
    feature = torch.randn(2, 8, 7, 7).to(dtype=dtype)
    reconstruction, aux = module(feature)
    assert torch.equal(reconstruction, feature)
    assert torch.equal(aux["input"], feature)


def test_semantic_and_morphology_select_complementary_channels():
    module = OSMFFactorizer(5, n_class=4)
    feature = torch.arange(5.0).reshape(1, 5, 1, 1)
    _, aux = module(feature)
    assert torch.equal(aux["semantic"].flatten(), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(aux["morphology"].flatten(), torch.tensor([3.0, 4.0]))


def test_forward_inference_matches_training_reconstruction():
    module = OSMFFactorizer(8, n_class=4)
    feature = torch.randn(2, 8, 5, 5)
    reconstruction, _ = module(feature)
    assert torch.equal(module.forward_inference(feature), reconstruction)


def test_semantic_head_is_gap_plus_linear_only():
    module = OSMFFactorizer(8, n_class=4)
    children = list(module.semantic_classifier.modules())
    assert len(children) == 1
    assert isinstance(children[0], torch.nn.Linear)


def test_semantic_loss_matches_released_multilabel_loss():
    logits = torch.randn(3, 4)
    labels = torch.randint(0, 2, (3, 4)).float()
    expected = torch.nn.functional.multilabel_soft_margin_loss(logits, labels)
    assert torch.equal(semantic_classification_loss(logits, labels), expected)


def test_reconstruction_target_is_stop_gradient():
    prediction = torch.randn(2, 8, 4, 4, requires_grad=True)
    target = torch.randn(2, 8, 4, 4, requires_grad=True)
    reconstruction_loss(prediction, target).backward()
    assert prediction.grad is not None
    assert target.grad is None


def test_initial_reconstruction_cosine_is_one():
    module = OSMFFactorizer(8, n_class=4)
    feature = torch.randn(2, 8, 4, 4)
    reconstruction = module.forward_inference(feature)
    assert reconstruction_cosine(reconstruction, feature).item() == pytest.approx(
        1.0, abs=1e-7
    )


def test_equivariance_loss_is_zero_after_inverse_flip():
    morphology = torch.randn(2, 8, 7, 7)
    flipped = torch.flip(morphology, dims=(3,))
    aligned = inverse_align_morphology(flipped, flip_dimension=3)
    assert spatial_equivariance_loss(morphology, aligned).item() == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.parametrize("dimension", [0, 1, 4, -1])
def test_only_frozen_geometric_flips_are_allowed(dimension):
    with pytest.raises(ValueError):
        inverse_align_morphology(torch.randn(1, 2, 3, 3), dimension)


def test_cross_covariance_and_orthogonality_are_finite():
    semantic = torch.randn(2, 6, 5, 5)
    morphology = torch.randn(2, 4, 5, 5)
    covariance = cross_subspace_covariance(semantic, morphology)
    loss = orthogonality_loss(semantic, morphology)
    assert covariance.shape == (6, 4)
    assert torch.isfinite(covariance).all()
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_osmf_constants_are_frozen():
    assert OSMF_LAMBDA_SEM == 0.20
    assert OSMF_LAMBDA_MORPH == 0.20
    assert OSMF_LAMBDA_ORTH == 0.05
    assert OSMF_LAMBDA_REC == 0.10
    assert OSMF_EQUIVARIANCE_INTERVAL == 4


def test_factorizer_has_no_forbidden_module_family():
    source = inspect.getsource(OSMFFactorizer).lower()
    for forbidden in (
        "attention",
        "transformer",
        "prototype",
        "fft",
        "frequency",
        "uncertainty",
        "router",
    ):
        assert forbidden not in source


def test_exactly_one_auxiliary_classifier():
    module = OSMFFactorizer(512, n_class=4)
    classifiers = [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]
    assert classifiers == [module.semantic_classifier]


def test_all_factorizer_parameters_receive_gradient():
    module = OSMFFactorizer(8, n_class=4)
    feature = torch.randn(2, 8, 5, 5, requires_grad=True)
    reconstruction, aux = module(feature)
    labels = torch.randint(0, 2, (2, 4)).float()
    loss = (
        reconstruction.square().mean()
        + semantic_classification_loss(aux["semantic_logits"], labels)
        + orthogonality_loss(aux["semantic"], aux["morphology"])
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_osmf_network_checkpoint_delta_is_only_new_osmf_keys():
    from network.resnet38_cls import Net as A0Net
    from network.resnet38_cls_osmf import Net as OSMFNet

    baseline = A0Net(n_class=4)
    osmf = OSMFNet(n_class=4)
    incompatible = osmf.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "osmf_28_1.p_sem.weight",
        "osmf_28_1.p_morph.weight",
        "osmf_28_1.u_sem.weight",
        "osmf_28_1.u_morph.weight",
        "osmf_28_1.semantic_classifier.weight",
        "osmf_28_1.semantic_classifier.bias",
    }


def test_optimizer_groups_cover_every_osmf_parameter_once():
    from network.resnet38_cls_osmf import Net as OSMFNet

    model = OSMFNet(n_class=4)
    groups = model.get_parameter_groups()
    grouped = [parameter for group in groups for parameter in group]
    for parameter in model.osmf_28_1.parameters():
        assert sum(candidate is parameter for candidate in grouped) == 1


def test_full_network_forward_cam_is_exact_at_initialization():
    from network.resnet38_cls import Net_CAM as A0NetCAM
    from network.resnet38_cls_osmf import Net_CAM as OSMFNetCAM

    torch.manual_seed(20260817)
    baseline = A0NetCAM(n_class=4)
    osmf = OSMFNetCAM(n_class=4)
    osmf.load_state_dict(baseline.state_dict(), strict=False)
    baseline.eval()
    osmf.eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        baseline_outputs = baseline.forward_cam(image)
        osmf_outputs = osmf.forward_cam(image)
        aux = osmf.forward_osmf_features(image)
    assert torch.equal(aux["input"], aux["reconstruction"])
    assert all(
        torch.equal(baseline_tensor, osmf_tensor)
        for baseline_tensor, osmf_tensor in zip(baseline_outputs, osmf_outputs)
    )


def test_osmf_module_initialization_does_not_advance_global_rng():
    from network.resnet38_cls import Net as A0Net
    from network.resnet38_cls_osmf import Net as OSMFNet

    torch.manual_seed(123)
    A0Net(n_class=4)
    expected_next = torch.rand(8)
    torch.manual_seed(123)
    OSMFNet(n_class=4)
    actual_next = torch.rand(8)
    assert torch.equal(actual_next, expected_next)
