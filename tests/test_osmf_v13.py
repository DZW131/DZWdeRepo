from __future__ import annotations

from pathlib import Path

import pytest
import torch

from network.osmf_v13 import (
    OSMF_LAMBDA_ORTH,
    OSMF_LAMBDA_REC,
    OSMF_LAMBDA_SEM,
    OSMF_LAMBDA_STRUCT,
    OSMF_STRUCTURAL_INTERVAL,
    OSMFV13Factorizer,
    affinity_equivariance_error,
    inverse_align_affinity,
    local_affinity_map,
    structural_affinity_loss,
)


def test_v13_frozen_contract_and_four_projection_tensors():
    module = OSMFV13Factorizer(512)
    assert tuple(dict(module.named_parameters())) == (
        "p_sem.weight", "p_morph.weight", "u_sem.weight", "u_morph.weight"
    )
    assert (OSMF_LAMBDA_SEM, OSMF_LAMBDA_STRUCT, OSMF_LAMBDA_ORTH, OSMF_LAMBDA_REC) == (0.05, 0.05, 0.05, 0.10)
    assert OSMF_STRUCTURAL_INTERVAL == 4


@pytest.mark.parametrize("flip_dimension", (2, 3))
def test_direction_aware_affinity_alignment_is_exact(flip_dimension):
    feature = torch.randn(2, 7, 9, 11)
    flipped = torch.flip(feature, dims=(flip_dimension,))
    assert affinity_equivariance_error(feature, flipped, flip_dimension).item() == pytest.approx(0.0, abs=1e-7)
    assert structural_affinity_loss(feature, flipped, flip_dimension).item() == pytest.approx(0.0, abs=1e-8)


def test_affinity_shape_mask_and_invalid_boundary_exclusion():
    feature = torch.randn(2, 5, 6, 7)
    affinity, mask = local_affinity_map(feature)
    assert affinity.shape == (2, 8, 6, 7)
    assert mask.shape == (1, 8, 6, 7)
    assert mask.dtype == torch.bool
    assert 0 < mask.count_nonzero() < mask.numel()
    aligned, aligned_mask = inverse_align_affinity(affinity, mask, 3)
    assert aligned.shape == affinity.shape
    assert aligned_mask.shape == mask.shape


def test_structural_loss_backpropagates_to_both_inputs():
    first = torch.randn(2, 8, 7, 7, requires_grad=True)
    second = torch.randn(2, 8, 7, 7, requires_grad=True)
    loss = structural_affinity_loss(first, second, 3)
    loss.backward()
    assert first.grad is not None and first.grad.norm() > 0
    assert second.grad is not None and second.grad.norm() > 0


def test_v13_exact_initial_identity_and_checkpoint_delta():
    from network.resnet38_cls import Net as A0Net
    from network.resnet38_cls_osmf_v13 import Net as V13Net

    module = OSMFV13Factorizer(8)
    feature = torch.randn(2, 8, 7, 7)
    reconstruction, _ = module(feature)
    assert torch.equal(reconstruction, feature)
    a0, v13 = A0Net(4), V13Net(4)
    incompatible = v13.load_state_dict(a0.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "osmf_28_1.p_sem.weight", "osmf_28_1.p_morph.weight",
        "osmf_28_1.u_sem.weight", "osmf_28_1.u_morph.weight",
    }


def test_cli_has_no_validation_test_luad_epochs_or_checkpoint_save():
    source = Path("tools/audit_osmf_v13_gradient_gate.py").read_text(encoding="utf-8")
    assert 'add_argument("--val' not in source
    assert 'add_argument("--test' not in source
    assert 'add_argument("--luad' not in source
    assert 'add_argument("--epochs' not in source
    assert "torch.save(" not in source
