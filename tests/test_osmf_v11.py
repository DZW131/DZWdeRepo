from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from network.osmf_v11 import (
    OSMF_EQUIVARIANCE_INTERVAL,
    OSMF_LAMBDA_MORPH,
    OSMF_LAMBDA_ORTH,
    OSMF_LAMBDA_REC,
    OSMF_LAMBDA_SEM,
    OSMFV11Factorizer,
    semantic_preservation_agreement,
    semantic_preservation_loss,
)


def test_v11_has_exactly_four_projection_parameters_and_no_classifier():
    module = OSMFV11Factorizer(512)
    assert tuple(dict(module.named_parameters())) == (
        "p_sem.weight",
        "p_morph.weight",
        "u_sem.weight",
        "u_morph.weight",
    )
    assert sum(parameter.numel() for parameter in module.parameters()) == 524288
    assert not any(isinstance(child, torch.nn.Linear) for child in module.modules())
    assert "semantic_classifier" not in inspect.getsource(OSMFV11Factorizer)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v11_partition_initialization_is_exact_identity(dtype):
    module = OSMFV11Factorizer(8)
    feature = torch.randn(2, 8, 7, 7).to(dtype=dtype)
    reconstruction, aux = module(feature)
    assert torch.equal(reconstruction, feature.float())
    assert torch.equal(
        aux["semantic_reconstruction"] + aux["morphology_reconstruction"],
        reconstruction,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v11_cuda_batch20_bf16_exact_identity_under_a0_tf32():
    module = OSMFV11Factorizer(512).cuda()
    feature = torch.randn(20, 512, 28, 28, device="cuda")
    backend = torch.backends.cudnn.conv
    original = backend.fp32_precision
    backend.fp32_precision = "tf32"
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reconstruction = module.forward_inference(feature)
        assert backend.fp32_precision == "tf32"
    finally:
        backend.fp32_precision = original
    assert torch.equal(reconstruction, feature)


def test_semantic_preservation_is_class_channel_cosine():
    teacher = torch.randn(2, 4, 5, 5, requires_grad=True)
    student = torch.randn(2, 4, 5, 5, requires_grad=True)
    expected = (
        F.normalize(student.float(), dim=1)
        * F.normalize(teacher.detach().float(), dim=1)
    ).sum(dim=1).mean()
    agreement = semantic_preservation_agreement(student, teacher)
    loss = semantic_preservation_loss(student, teacher)
    assert torch.allclose(agreement, expected)
    assert torch.allclose(loss, 1.0 - expected)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_identical_semantic_responses_have_zero_loss():
    response = torch.randn(2, 4, 5, 5)
    assert semantic_preservation_loss(response, response).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_auxiliary_semantic_loss_cannot_update_ic1_but_updates_semantic_path():
    from network.resnet38_cls_osmf_v11 import Net

    model = Net(n_class=4)
    model.eval()
    _, aux = model.forward_with_aux(torch.randn(2, 3, 32, 32))
    loss = semantic_preservation_loss(
        aux["semantic_student_response"], aux["semantic_teacher_response"]
    )
    loss.backward()
    assert not aux["semantic_teacher_response"].requires_grad
    assert model.ic1.weight.grad is None
    assert model.ic1.bias.grad is None
    assert model.osmf_28_1.p_sem.weight.grad is not None
    assert model.osmf_28_1.u_sem.weight.grad is not None
    assert model.osmf_28_1.p_morph.weight.grad is None
    assert model.osmf_28_1.u_morph.weight.grad is None


def test_original_sshr_loss_still_updates_ic1():
    from network.resnet38_cls_osmf_v11 import Net

    model = Net(n_class=4)
    model.eval()
    outputs, _ = model.forward_with_aux(torch.randn(2, 3, 32, 32))
    outputs[1].square().mean().backward()
    assert model.ic1.weight.grad is not None
    assert model.ic1.bias.grad is not None


def test_student_response_reuses_detached_ic1_geometry_exactly():
    from network.resnet38_cls_osmf_v11 import Net

    model = Net(n_class=4)
    model.eval()
    _, aux = model.forward_with_aux(torch.randn(1, 3, 32, 32))
    expected_student = F.conv2d(
        aux["semantic_reconstruction"],
        model.ic1.weight.detach(),
        model.ic1.bias.detach(),
    )
    expected_teacher = F.conv2d(
        aux["input"].detach(),
        model.ic1.weight.detach(),
        model.ic1.bias.detach(),
    )
    assert torch.equal(aux["semantic_student_response"], expected_student)
    assert torch.equal(aux["semantic_teacher_response"], expected_teacher)


def test_a0_checkpoint_delta_is_only_four_projection_keys():
    from network.resnet38_cls import Net as A0Net
    from network.resnet38_cls_osmf_v11 import Net as V11Net

    baseline = A0Net(n_class=4)
    v11 = V11Net(n_class=4)
    incompatible = v11.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "osmf_28_1.p_sem.weight",
        "osmf_28_1.p_morph.weight",
        "osmf_28_1.u_sem.weight",
        "osmf_28_1.u_morph.weight",
    }


def test_v11_optimizer_groups_cover_every_new_parameter_once():
    from network.resnet38_cls_osmf_v11 import Net

    model = Net(n_class=4)
    grouped = [parameter for group in model.get_parameter_groups() for parameter in group]
    for parameter in model.osmf_28_1.parameters():
        assert sum(candidate is parameter for candidate in grouped) == 1


def test_v11_full_cam_path_is_exact_at_initialization():
    from network.resnet38_cls import Net_CAM as A0NetCAM
    from network.resnet38_cls_osmf_v11 import Net_CAM as V11NetCAM

    torch.manual_seed(20260817)
    baseline = A0NetCAM(n_class=4)
    v11 = V11NetCAM(n_class=4)
    incompatible = v11.load_state_dict(baseline.state_dict(), strict=False)
    assert len(incompatible.missing_keys) == 4
    baseline.eval()
    v11.eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        baseline_outputs = baseline.forward_cam(image)
        v11_outputs = v11.forward_cam(image)
        aux = v11.forward_osmf_features(image)
    assert torch.equal(aux["input"], aux["reconstruction"])
    assert all(
        torch.equal(left, right)
        for left, right in zip(baseline_outputs, v11_outputs)
    )


def test_v11_constants_remain_frozen():
    assert OSMF_LAMBDA_SEM == 0.20
    assert OSMF_LAMBDA_MORPH == 0.20
    assert OSMF_LAMBDA_ORTH == 0.05
    assert OSMF_LAMBDA_REC == 0.10
    assert OSMF_EQUIVARIANCE_INTERVAL == 4


def test_v11_module_creation_does_not_advance_global_rng():
    from network.resnet38_cls import Net as A0Net
    from network.resnet38_cls_osmf_v11 import Net as V11Net

    torch.manual_seed(123)
    A0Net(n_class=4)
    expected = torch.rand(8)
    torch.manual_seed(123)
    V11Net(n_class=4)
    actual = torch.rand(8)
    assert torch.equal(actual, expected)

