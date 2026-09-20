"""Frozen HQMR-v1 with optional RACC-A and RACC-G Phase-0 heads."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from network.hqmr import class_mixture
from network.hqmr_net import HQMRNet
from network.racc import RACCController


class RACCNet(HQMRNet):
    def __init__(self):
        super().__init__()
        self.racc = RACCController(hidden_dim=8, alpha_max=4.0, topk_ratio=0.05)

    def freeze_hqmr(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("racc."))

    def forward(self, image, labels, step=0, run_pmec=False, hqmr_mode="full",
                enable_arbitration=True, enable_presence=True):
        arbitrator = self.racc.arbitration if enable_arbitration else None
        output = super().forward(image, labels, step=step, run_pmec=run_pmec,
                                 hqmr_mode=hqmr_mode, hqmr_arbitrator=arbitrator)
        stage3 = output["stages"][2]["hqmr"]
        c4 = class_mixture(stage3["logits4"].sigmoid(), stage3["weights"])
        local = self.racc.presence(c4.detach(), stage3["mixture"].detach())
        stage3["class_mixture4"] = c4
        output["racc"] = {"enable_arbitration": bool(enable_arbitration),
                          "enable_presence": bool(enable_presence),
                          "alpha_stage2": output["stages"][1]["hqmr"].get("alpha4"),
                          "alpha_stage3": stage3.get("alpha4"),
                          "local_presence": local}
        with torch.autocast(device_type=image.device.type, enabled=False):
            presence_loss = F.binary_cross_entropy_with_logits(local["logits"].float(), labels.float())
            total = output["losses"]["loss"] + (presence_loss if enable_presence else 0.0)
        output["losses"].update({"loss_hqmr": output["losses"]["loss"],
                                 "loss_presence": presence_loss, "loss": total})
        output["racc_configuration"] = {"arbitration": bool(enable_arbitration),
            "presence": bool(enable_presence), "alpha_max": 4.0, "topk_ratio": .05,
            "presence_threshold": .5, "presence_loss_weight": 1.0, "old_parameters_frozen": True}
        return output


__all__ = ["RACCNet"]
