"""Future OSMF-v1.2 training entry point; not authorized by Phase-0 runs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from network.osmf_v12 import (
    OSMF_EQUIVARIANCE_INTERVAL,
    OSMF_LAMBDA_MORPH,
    OSMF_LAMBDA_ORTH,
    OSMF_LAMBDA_REC,
    OSMF_LAMBDA_SEM,
    cross_subspace_covariance,
    inverse_align_morphology,
    orthogonality_loss,
    reconstruction_cosine,
    reconstruction_loss,
    semantic_preservation_agreement,
    semantic_preservation_loss,
    spatial_equivariance_loss,
)


def sshr_classification_loss(outputs, labels):
    out_56, out_28_1, out_28_2, out_deep = outputs[:4]
    return (
        0.10 * F.multilabel_soft_margin_loss(out_56, labels)
        + 0.15 * F.multilabel_soft_margin_loss(out_28_1, labels)
        + 0.25 * F.multilabel_soft_margin_loss(out_28_2, labels)
        + 0.50 * F.multilabel_soft_margin_loss(out_deep, labels)
    )


def compute_step_losses(model, images, labels, optimizer_step):
    outputs, aux = model.forward_with_aux(images)
    loss_sshr = sshr_classification_loss(outputs, labels)
    loss_sem_pres = semantic_preservation_loss(
        aux["semantic_student_response"], aux["semantic_teacher_response"]
    )
    loss_orth = orthogonality_loss(aux["semantic"], aux["morphology"])
    loss_rec = reconstruction_loss(aux["reconstruction"], aux["input"])
    compute_equivariance = (
        optimizer_step + 1
    ) % OSMF_EQUIVARIANCE_INTERVAL == 0
    if compute_equivariance:
        block = (optimizer_step + 1) // OSMF_EQUIVARIANCE_INTERVAL
        flip_dimension = 3 if block % 2 else 2
        morphology_b = model.forward_morphology(
            torch.flip(images, dims=(flip_dimension,))
        )
        morphology_b = inverse_align_morphology(morphology_b, flip_dimension)
        loss_eq = spatial_equivariance_loss(aux["morphology"], morphology_b)
    else:
        loss_eq = loss_sshr.new_zeros(())
    total = (
        loss_sshr
        + OSMF_LAMBDA_SEM * loss_sem_pres
        + OSMF_LAMBDA_MORPH * loss_eq
        + OSMF_LAMBDA_ORTH * loss_orth
        + OSMF_LAMBDA_REC * loss_rec
    )
    diagnostics = {
        "loss_total": total.detach(),
        "loss_sshr": loss_sshr.detach(),
        "loss_sem_pres": loss_sem_pres.detach(),
        "loss_eq": loss_eq.detach(),
        "loss_orth": loss_orth.detach(),
        "loss_rec": loss_rec.detach(),
        "semantic_agreement": semantic_preservation_agreement(
            aux["semantic_student_response"], aux["semantic_teacher_response"]
        ).detach(),
        "reconstruction_cosine": reconstruction_cosine(
            aux["reconstruction"], aux["input"]
        ).detach(),
        "cross_covariance": cross_subspace_covariance(
            aux["semantic"], aux["morphology"]
        )
        .square()
        .mean()
        .sqrt()
        .detach(),
        "equivariance_computed": compute_equivariance,
    }
    return total, diagnostics


def main():
    raise SystemExit(
        "OSMF-v1.2 Phase 1 is not authorized. Use the parity/readiness audit tools."
    )


if __name__ == "__main__":
    main()
