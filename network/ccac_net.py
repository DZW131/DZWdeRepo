"""GCQM with zero-parameter class-conditioned affinity completion at Stage2/3."""
from __future__ import annotations

import torch

from network.ccac import ccac_complete
from network.cqrf_net import STAGE_WEIGHTS
from network.gcqm_net import GCQMNet
from network.momd import mixture_class_bce


class CCACNet(GCQMNet):
    def forward(self, image, labels, step=0, run_pmec=False,
                ccac_mode: str = "feature", ccac_use_rival: bool = True):
        output = super().forward(image, labels, step=step, run_pmec=run_pmec)
        mask_losses = [output["losses"]["loss_mask_stage1"]]
        for stage_index, stage in enumerate(output["stages"], 1):
            if stage_index < 2:
                stage["ccac"] = None
                continue
            ccac = ccac_complete(stage["gcqm"]["mixture"], output["pixel_feature"], iterations=2,
                                 affinity_mode=ccac_mode, use_rival=ccac_use_rival)
            stage["ccac"] = ccac
            stage["primary_output"] = ccac["restored"]
            loss, detail = mixture_class_bce(ccac["restored"], output["targets"], labels.bool())
            stage["mask_detail"] = detail
            mask_losses.append(loss.float())
        with torch.autocast(device_type=image.device.type, enabled=False):
            loss_mask = sum(weight * loss for weight, loss in zip(STAGE_WEIGHTS, mask_losses))
            total = .50 * output["losses"]["loss_deep"] + .25 * output["losses"]["loss_pca"] + .25 * loss_mask
        output["base_primary_output"] = output["primary_output"]
        output["primary_output"] = output["stages"][-1]["ccac"]["restored"]
        output["losses"].update({
            "loss": total, "loss_mask": loss_mask,
            "loss_mask_stage2": mask_losses[1], "loss_mask_stage3": mask_losses[2],
        })
        output["ccac_configuration"] = {
            "iterations": 2, "affinity_mode": ccac_mode, "use_rival": bool(ccac_use_rival),
            "stages": [2, 3], "new_parameters": 0, "new_auxiliary_loss": 0,
        }
        return output


__all__ = ["CCACNet"]
