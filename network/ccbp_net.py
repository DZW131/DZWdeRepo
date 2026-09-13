"""Frozen CCRA + HQMR-v1 with Stage3-only CCBP."""
from __future__ import annotations

import torch

from network.ccbp import CCBP, balanced_competitive_loss
from network.cqrf_net import STAGE_WEIGHTS
from network.hqmr_net import HQMRNet
from network.momd import mixture_class_bce


class CCBPNet(HQMRNet):
    def __init__(self):
        super().__init__()
        self.ccbp = CCBP(256)

    def forward(self, image, labels, step=0, run_pmec=False, hqmr_mode="full", ccbp_mode="full"):
        output = super().forward(image, labels, step=step, run_pmec=run_pmec, hqmr_mode=hqmr_mode)
        stage3 = output["stages"][2]; decoded = stage3["hqmr"]
        decoded["base_mixture"] = decoded["mixture"]
        purified = self.ccbp(decoded["basis"], decoded["weights"], decoded["query4"],
                             decoded["key4"], output["deep_gate"], labels.bool(), mode=ccbp_mode)
        decoded["ccbp"] = purified; decoded["purified_basis"] = purified["purified_basis"]
        decoded["mixture"] = purified["mixture"]; decoded["primary_output"] = purified["mixture"]
        stage3["primary_output"] = purified["mixture"]
        stage3_loss, detail = mixture_class_bce(purified["mixture"], output["targets"], labels.bool())
        stage3["mask_detail"] = detail
        mask_losses = [output["losses"]["loss_mask_stage1"], output["losses"]["loss_mask_stage2"], stage3_loss.float()]
        with torch.autocast(device_type=image.device.type, enabled=False):
            loss_mask = sum(weight * loss for weight, loss in zip(STAGE_WEIGHTS, mask_losses))
            loss_base = .50 * output["losses"]["loss_deep"] + .25 * output["losses"]["loss_pca"] + .25 * loss_mask
            loss_ccbp, purifier_detail = balanced_competitive_loss(purified["logits"], output["target_detail"]["positive"], labels.bool())
            total = loss_base + loss_ccbp
        output["primary_output"] = purified["mixture"]
        output["losses"].update({"loss": total, "loss_base": loss_base, "loss_ccbp": loss_ccbp,
            "loss_mask": loss_mask, "loss_mask_stage3": stage3_loss.float()})
        output["ccbp_detail"] = purifier_detail
        output["ccbp_configuration"] = {"stage": 3, "dimension": 256, "mode": ccbp_mode,
            "query_tensor": "HQMR.query4", "semantic_tensor": "HQMR.key4",
            "class_weights_detached": True, "query_input_detached": True,
            "semantic_input_detached": True, "class_prior_detached": True,
            "gate_detached_for_base": True, "all_classes_symmetric": True,
            "threshold": False, "top_k": False, "morphology": False}
        return output


__all__ = ["CCBPNet"]
