"""Frozen CCRA allocation with HQMR Stage2/3 primary masks."""
from __future__ import annotations

import torch

from network.cqrf_net import STAGE_WEIGHTS
from network.gcqm_net import GCQMNet
from network.hqmr import HQMR, class_mixture
from network.momd import mixture_class_bce


class HQMRNet(GCQMNet):
    def __init__(self):
        super().__init__()
        self.hqmr = HQMR(256)

    def forward(self, image, labels, step=0, run_pmec=False, hqmr_mode="full"):
        output = super().forward(image, labels, step=step, run_pmec=run_pmec, gcqm_weights_only=True)
        h5 = output["query_detail"]["context_feature"]
        h4 = output["pixel_detail"]["F4_context"]
        h3 = output["features"]["F3"]
        mask_losses = [output["losses"]["loss_mask_stage1"]]
        for stage_index, stage in enumerate(output["stages"], 1):
            if stage_index == 1:
                stage["hqmr"] = None
                continue
            decoded = self.hqmr(stage["query"], h5, h4, h3 if stage_index == 3 else None, mode=hqmr_mode)
            decoded["weights"] = stage["gcqm"]["weights"].detach()
            decoded["mixture"] = class_mixture(decoded["basis"], decoded["weights"])
            decoded["primary_output"] = decoded["mixture"]
            loss, detail = mixture_class_bce(decoded["mixture"], output["targets"], labels.bool())
            stage["hqmr"] = decoded
            stage["primary_output"] = decoded["mixture"]
            stage["mask_detail"] = detail
            mask_losses.append(loss.float())
        with torch.autocast(device_type=image.device.type, enabled=False):
            loss_mask = sum(weight * loss for weight, loss in zip(STAGE_WEIGHTS, mask_losses))
            total = .50 * output["losses"]["loss_deep"] + .25 * output["losses"]["loss_pca"] + .25 * loss_mask
        output["base_primary_output"] = output["primary_output"]
        output["primary_output"] = output["stages"][-1]["hqmr"]["mixture"]
        output["losses"].update({"loss": total, "loss_mask": loss_mask,
            "loss_mask_stage2": mask_losses[1], "loss_mask_stage3": mask_losses[2]})
        output["hqmr_configuration"] = {
            "mode": hqmr_mode, "dimension": 256, "stages": [2, 3],
            "features": {"H5": "CHPF(F5)", "H4": "CHPF(F4)", "H3": "raw F3"},
            "shared_stage2_stage3": True, "class_weights_detached": True,
            "propagation": False, "hard_threshold": False, "top_k": False,
            "new_auxiliary_loss": False,
        }
        return output


__all__ = ["HQMRNet"]
