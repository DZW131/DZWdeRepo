"""Preregistered OSMF-v1.3 loss helper; not a full-training entry point."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from network.osmf_v13 import (
    OSMF_LAMBDA_ORTH,
    OSMF_LAMBDA_REC,
    OSMF_LAMBDA_SEM,
    OSMF_LAMBDA_STRUCT,
    OSMF_STRUCTURAL_INTERVAL,
    cross_subspace_covariance,
    orthogonality_loss,
    reconstruction_cosine,
    reconstruction_loss,
    semantic_preservation_agreement,
    semantic_preservation_loss,
    structural_affinity_loss,
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
    compute_structural = (optimizer_step + 1) % OSMF_STRUCTURAL_INTERVAL == 0
    if compute_structural:
        block = (optimizer_step + 1) // OSMF_STRUCTURAL_INTERVAL
        flip_dimension = 3 if block % 2 else 2
        second = model.forward_morphology(torch.flip(images, dims=(flip_dimension,)))
        loss_struct = structural_affinity_loss(
            aux["morphology"], second, flip_dimension
        )
    else:
        loss_struct = loss_sshr.new_zeros(())
    total = (
        loss_sshr
        + OSMF_LAMBDA_SEM * loss_sem_pres
        + OSMF_LAMBDA_STRUCT * loss_struct
        + OSMF_LAMBDA_ORTH * loss_orth
        + OSMF_LAMBDA_REC * loss_rec
    )
    diagnostics = {
        "loss_total": total.detach(),
        "loss_sshr": loss_sshr.detach(),
        "loss_sem_pres": loss_sem_pres.detach(),
        "loss_struct": loss_struct.detach(),
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
        ).square().mean().sqrt().detach(),
        "structural_computed": compute_structural,
    }
    return total, diagnostics


def main():
    raise SystemExit(
        "OSMF-v1.3 full training is not authorized; use the gated audit tools."
    )


if __name__ == "__main__":
    main()
