"""Frozen CCRA/GCQM backbone with learned DFRA and monotone consensus completion."""
from __future__ import annotations

import torch

from network.cqrf_net import STAGE_WEIGHTS
from network.dfsc import DFRA, mcc_complete
from network.gcqm_net import GCQMNet
from network.momd import mixture_class_bce


class DFSCNet(GCQMNet):
    def __init__(self):
        super().__init__()
        self.dfra = DFRA(256, 32)

    def forward(self, image, labels, step=0, run_pmec=False, relation_mode="full", completion_update="mcc"):
        output = super().forward(image, labels, step=step, run_pmec=run_pmec)
        relation = self.dfra(output["pixel_feature"], output["target_detail"], mode=relation_mode)
        mask_losses = [output["losses"]["loss_mask_stage1"]]
        for stage_index, stage in enumerate(output["stages"], 1):
            if stage_index < 2:
                stage["dfsc"] = None
                continue
            completion = mcc_complete(stage["gcqm"]["mixture"], relation["affinity_comp"],
                                      iterations=2, update=completion_update)
            stage["dfsc"] = completion; stage["primary_output"] = completion["restored"]
            loss, detail = mixture_class_bce(completion["restored"], output["targets"], labels.bool())
            stage["mask_detail"] = detail; mask_losses.append(loss.float())
        with torch.autocast(device_type=image.device.type, enabled=False):
            loss_mask = sum(weight * loss for weight, loss in zip(STAGE_WEIGHTS, mask_losses))
            loss_base = .50 * output["losses"]["loss_deep"] + .25 * output["losses"]["loss_pca"] + .25 * loss_mask
            total = loss_base + relation["relation_loss"]
        output["base_primary_output"] = output["primary_output"]
        output["primary_output"] = output["stages"][-1]["dfsc"]["restored"]
        output["relation"] = relation
        output["losses"].update({"loss": total, "loss_base": loss_base, "loss_relation": relation["relation_loss"],
            "loss_mask": loss_mask, "loss_mask_stage2": mask_losses[1], "loss_mask_stage3": mask_losses[2]})
        output["dfsc_configuration"] = {"relation_mode": relation_mode, "completion_update": completion_update,
            "iterations": 2, "stages": [2, 3], "rival_gate": False, "top_k": False,
            "new_dense_supervision": False}
        return output


__all__ = ["DFSCNet"]
