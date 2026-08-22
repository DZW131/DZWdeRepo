"""Frozen training-only objectives for HALR-v1.

HALR does not modify the SSHR model.  This module only constructs an exact
paired flip view and computes CVLE/RAHD from CAM28_1 and CAMdeep.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def epoch_alpha(epoch: int) -> float:
    """Return the preregistered epoch 1--5 localization-loss ramp."""

    if epoch < 1:
        raise ValueError("epoch is one-indexed")
    return min(1.0, 0.25 * (epoch - 1))


def apply_pair_transform(tensor: torch.Tensor, flip_codes: torch.Tensor) -> torch.Tensor:
    """Apply per-sample hflip (code 0) or vflip (code 1).

    Both transforms are self-inverse, so this function is also used to align
    view-2 outputs back into view-1 coordinates.  No interpolation is used.
    """

    if tensor.ndim != 4:
        raise ValueError("paired transform expects BCHW input")
    if flip_codes.ndim != 1 or flip_codes.numel() != tensor.shape[0]:
        raise ValueError("one flip code is required per sample")
    if not torch.all((flip_codes == 0) | (flip_codes == 1)):
        raise ValueError("flip codes must be 0 (horizontal) or 1 (vertical)")
    selector = (flip_codes == 0).view(-1, 1, 1, 1)
    return torch.where(selector, torch.flip(tensor, dims=(-1,)), torch.flip(tensor, dims=(-2,)))


def _present_probabilities(cams: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    present = labels > 0.5
    masked = cams.float().masked_fill(~present[:, :, None, None], -1.0e4)
    return F.softmax(masked, dim=1)


def _jsd_per_sample(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(first.dtype).tiny
    midpoint = 0.5 * (first + second)
    first_kl = first * (first.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    second_kl = second * (second.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    return 0.5 * (first_kl.sum(dim=1) + second_kl.sum(dim=1)).mean(dim=(1, 2))


def _kl_per_sample(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(student.dtype).tiny
    values = teacher * (
        teacher.clamp_min(eps).log() - student.clamp_min(eps).log()
    )
    return values.sum(dim=1).mean(dim=(1, 2))


def _valid_mean(values: torch.Tensor, valid: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if bool(valid.any()):
        return values[valid].mean()
    return reference.sum() * 0.0


def halr_terms(
    cam28_view1: torch.Tensor,
    camdeep_view1: torch.Tensor,
    cam28_view2: torch.Tensor,
    camdeep_view2: torch.Tensor,
    labels: torch.Tensor,
    flip_codes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the frozen CVLE and RAHD terms and observation-only diagnostics."""

    expected = cam28_view1.shape
    if any(value.shape != expected for value in (camdeep_view1, cam28_view2, camdeep_view2)):
        raise ValueError("CAM28_1 and CAMdeep must share BxCxHxW shape across views")
    if labels.shape != expected[:2]:
        raise ValueError("labels must be BxC and match the CAM class dimension")

    p28_view1 = _present_probabilities(cam28_view1, labels)
    pdeep_view1 = _present_probabilities(camdeep_view1, labels)
    p28_view2 = apply_pair_transform(_present_probabilities(cam28_view2, labels), flip_codes)
    pdeep_view2 = apply_pair_transform(_present_probabilities(camdeep_view2, labels), flip_codes)

    valid = labels.gt(0.5).sum(dim=1) >= 2
    jsd28 = _jsd_per_sample(p28_view1, p28_view2)
    jsddeep = _jsd_per_sample(pdeep_view1, pdeep_view2)
    cvle = 0.5 * _valid_mean(jsd28, valid, cam28_view1)
    cvle = cvle + 0.5 * _valid_mean(jsddeep, valid, camdeep_view1)

    reliability28 = torch.exp(-jsd28.detach())
    reliabilitydeep = torch.exp(-jsddeep.detach())
    denominator = reliability28 + reliabilitydeep + 1.0e-8
    weight28 = reliability28 / denominator
    weightdeep = reliabilitydeep / denominator
    weight28_map = weight28[:, None, None, None]
    weightdeep_map = weightdeep[:, None, None, None]

    teacher_view1 = (
        weight28_map * p28_view1 + weightdeep_map * pdeep_view1
    ).detach()
    teacher_view2 = (
        weight28_map * p28_view2 + weightdeep_map * pdeep_view2
    ).detach()
    rahd_per_sample = 0.25 * (
        _kl_per_sample(teacher_view1, p28_view1)
        + _kl_per_sample(teacher_view1, pdeep_view1)
        + _kl_per_sample(teacher_view2, p28_view2)
        + _kl_per_sample(teacher_view2, pdeep_view2)
    )
    rahd = _valid_mean(rahd_per_sample, valid, cam28_view1)

    agreement_view1 = (p28_view1.argmax(dim=1) == pdeep_view1.argmax(dim=1)).float()
    agreement_view2 = (p28_view2.argmax(dim=1) == pdeep_view2.argmax(dim=1)).float()
    agreement = 0.5 * (
        agreement_view1.mean(dim=(1, 2)) + agreement_view2.mean(dim=(1, 2))
    )

    return {
        "cvle_loss": cvle,
        "rahd_loss": rahd,
        "valid_samples": valid,
        "valid_fraction": valid.float().mean(),
        "jsd28": _valid_mean(jsd28, valid, cam28_view1),
        "jsddeep": _valid_mean(jsddeep, valid, camdeep_view1),
        "weight28": _valid_mean(weight28, valid, cam28_view1),
        "weightdeep": _valid_mean(weightdeep, valid, camdeep_view1),
        "fraction_weight28_gt": _valid_mean((weight28 > weightdeep).float(), valid, cam28_view1),
        "fraction_weightdeep_gt": _valid_mean((weightdeep > weight28).float(), valid, cam28_view1),
        "hierarchy_agreement": _valid_mean(agreement, valid, cam28_view1),
        "weight28_per_sample": weight28,
        "weightdeep_per_sample": weightdeep,
        "teacher_view1": teacher_view1,
        "teacher_view2": teacher_view2,
        "p28_view1": p28_view1,
        "pdeep_view1": pdeep_view1,
        "p28_view2_aligned": p28_view2,
        "pdeep_view2_aligned": pdeep_view2,
    }
