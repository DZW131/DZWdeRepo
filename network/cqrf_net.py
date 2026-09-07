"""CQRF-Net Phase-0 class-conditioned responsibility model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from network.cqrf_query import (
    CCRALayer, CHPF, CQRFPixelDecoder, CrossFirstFocusDecoder,
    DualConfidenceAllocator, FocusPatchQueries, MaskEmbedding, Projection,
    sine_position_2d,
)
from network.hqrf_backbone import HQRFBackbone
from network.hqrf_pmec import pmec
from network.hqrf_targets import circular_locality, masked_query_bce, normalize_cam, radius_at_step, tri_state_targets


STAGE_WEIGHTS = (.20, .30, .50)


class CQRFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HQRFBackbone()
        self.deep_head = nn.Conv2d(4096, 4, 1, bias=False)
        nn.init.xavier_uniform_(self.deep_head.weight)
        self.patch_queries = FocusPatchQueries()
        self.semantic_projection = Projection(4096, 256)
        self.decoder1 = CrossFirstFocusDecoder()
        self.context_projection = Projection(1024, 256)
        self.f5_chpf = CHPF(256)
        self.ccra2 = CCRALayer()
        self.pixel_decoder = CQRFPixelDecoder()
        self.f4_memory_projection = Projection(128, 256)
        self.ccra3 = CCRALayer()
        self.mask_embeddings = nn.ModuleList([MaskEmbedding() for _ in range(3)])
        self.pca_heads = nn.ModuleList([DualConfidenceAllocator() for _ in range(3)])

    @staticmethod
    def _memory(feature):
        return feature.flatten(2).transpose(1, 2)

    def forward(self, image, labels, step=0, run_pmec=False):
        if labels is None:
            raise ValueError("CQRF Phase-0 requires train-split image labels")
        features = self.backbone(image)
        deep_cam_logits = self.deep_head(features["FD"])
        deep_logits = F.adaptive_avg_pool2d(deep_cam_logits, 1).flatten(1)
        deep_gate = deep_logits.float().sigmoid()
        normalized_cam = normalize_cam(deep_cam_logits)
        targets, target_detail = tri_state_targets(normalized_cam.detach(), labels.bool())

        content0, base = self.patch_queries(image)
        semantic_feature = self.semantic_projection(features["FD"].detach())
        kp1 = sine_position_2d(image.shape[0], *semantic_feature.shape[-2:], 256, image.device, semantic_feature.dtype)
        query1, attention1 = self.decoder1(content0 + base, self._memory(semantic_feature) + kp1)
        confidence1 = self.pca_heads[0](query1)

        context_raw = self.context_projection(features["F5"].detach())
        context_feature = self.f5_chpf(context_raw)
        kp2 = sine_position_2d(image.shape[0], *context_feature.shape[-2:], 256, image.device, context_feature.dtype)
        query2, ccra2 = self.ccra2(query1, base, self._memory(context_feature), kp2, confidence1["p_class"], deep_gate)
        confidence2 = self.pca_heads[1](query2)

        pixel_feature, pixel_detail = self.pixel_decoder(features["F4"], features["F3"])
        f4_memory = self.f4_memory_projection(pixel_detail["F4_context"].detach())
        kp3 = sine_position_2d(image.shape[0], *f4_memory.shape[-2:], 256, image.device, f4_memory.dtype)
        query3, ccra3 = self.ccra3(query2, base, self._memory(f4_memory), kp3, confidence2["p_class"], deep_gate)
        confidence3 = self.pca_heads[2](query3)

        stages = []
        for index, (query, confidence, detail, memory_hw) in enumerate((
            (query1, confidence1, {"cross_attention": attention1["cross_attention"]}, semantic_feature.shape[-2:]),
            (query2, confidence2, ccra2, context_feature.shape[-2:]),
            (query3, confidence3, ccra3, f4_memory.shape[-2:]),
        )):
            embedding = self.mask_embeddings[index](query)
            logits = torch.einsum("bqd,bdhw->bqhw", embedding, pixel_feature)
            stages.append({"query": query, "mask_embedding": embedding, "mask_logits": logits,
                           "confidence": confidence, "detail": detail, "memory_hw": memory_hw})

        radius = radius_at_step(step)
        locality = circular_locality(14, pixel_feature.shape[-2:], radius, pixel_feature.device)
        mask_losses, pca_losses = [], []
        for stage in stages:
            mask_loss, mask_detail = masked_query_bce(
                stage["mask_logits"], targets, labels.bool(), stage["confidence"]["p_class"],
                stage["confidence"]["joint"], locality,
            )
            stage["mask_detail"] = mask_detail
            mask_losses.append(mask_loss.float())
            with torch.autocast(device_type=image.device.type, enabled=False):
                pca_losses.append(F.binary_cross_entropy(stage["confidence"]["image_probability"], labels.float()))
        with torch.autocast(device_type=image.device.type, enabled=False):
            loss_deep = F.multilabel_soft_margin_loss(deep_logits.float(), labels.float())
            loss_pca = sum(w * v for w, v in zip(STAGE_WEIGHTS, pca_losses))
            loss_mask = sum(w * v for w, v in zip(STAGE_WEIGHTS, mask_losses))
            total = .50 * loss_deep + .25 * loss_pca + .25 * loss_mask

        output = {
            "features": features, "deep_cam_logits": deep_cam_logits, "deep_logits": deep_logits,
            "deep_gate": deep_gate, "normalized_cam": normalized_cam, "targets": targets,
            "target_detail": target_detail, "stages": stages, "pixel_feature": pixel_feature,
            "pixel_detail": pixel_detail, "query_detail": {
                "B": base, "semantic_feature": semantic_feature, "context_raw": context_raw,
                "context_feature": context_feature, "f4_memory": f4_memory,
            }, "radius": radius, "locality": locality,
            "losses": {"loss": total, "loss_deep": loss_deep, "loss_pca": loss_pca, "loss_mask": loss_mask,
                       **{f"loss_pca_stage{i+1}": v for i, v in enumerate(pca_losses)},
                       **{f"loss_mask_stage{i+1}": v for i, v in enumerate(mask_losses)}},
        }
        if run_pmec:
            final = stages[-1]
            region, rows = pmec(final["mask_logits"].detach(), final["confidence"]["joint"].detach(),
                                final["confidence"]["p_class"].detach(), labels.bool(), locality)
            output["pmec_region"], output["pmec_rows"] = region, rows
        return output

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            scratch = not name.startswith("backbone.")
            groups[2 * int(scratch) + int(name.endswith("bias"))].append(parameter)
        values = [id(v) for group in groups for v in group]
        if len(values) != len(set(values)):
            raise AssertionError("Duplicate optimizer parameter")
        return groups
