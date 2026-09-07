"""FQRF-Net Phase-0 focus-preservation model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from network.hqrf_backbone import HQRFBackbone
from network.hqrf_pmec import pmec
from network.hqrf_targets import circular_locality, masked_query_bce, normalize_cam, radius_at_step, tri_state_targets
from network.fqrf_query import (
    CHPF, CrossFirstFocusDecoder, DualConfidenceAllocator, DynamicFocusPosition,
    FocusPatchQueries, MaskEmbedding, PixelDecoder, Projection,
    previous_mask_visibility, sine_position_2d,
)


STAGE_WEIGHTS = (0.20, 0.30, 0.50)


class FQRFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HQRFBackbone()
        self.deep_head = nn.Conv2d(4096, 4, 1, bias=False)
        nn.init.xavier_uniform_(self.deep_head.weight)
        self.patch_queries = FocusPatchQueries()

        self.semantic_projection = Projection(4096, 256)
        self.decoder1 = CrossFirstFocusDecoder()
        self.focus_update1 = DynamicFocusPosition()

        self.context_projection = Projection(1024, 256)
        self.f5_chpf = CHPF(256)
        self.decoder2 = CrossFirstFocusDecoder()
        self.focus_update2 = DynamicFocusPosition()

        self.pixel_decoder = PixelDecoder()
        self.f4_memory_projection = Projection(128, 256)
        self.decoder3 = CrossFirstFocusDecoder()

        self.mask_embeddings = nn.ModuleList([MaskEmbedding() for _ in range(3)])
        self.pca_heads = nn.ModuleList([DualConfidenceAllocator() for _ in range(3)])

    @staticmethod
    def _memory(feature):
        return feature.flatten(2).transpose(1, 2)

    def query_decode(self, image, fd, f5, f4_context):
        """Propagate (Qc,Qp,M,A); detach only CNN-side inputs to trainable projections."""
        content0, base = self.patch_queries(image)

        semantic_feature = self.semantic_projection(fd.detach())
        position_d = sine_position_2d(image.shape[0], *semantic_feature.shape[-2:], 256, image.device, semantic_feature.dtype)
        query1, attention1 = self.decoder1(content0 + base, self._memory(semantic_feature) + position_d)

        context_raw = self.context_projection(f5.detach())
        context_feature = self.f5_chpf(context_raw)
        position5 = sine_position_2d(image.shape[0], *context_feature.shape[-2:], 256, image.device, context_feature.dtype)
        position2 = self.focus_update1(attention1["cross_attention"], position_d, base)

        stage1_embedding = self.mask_embeddings[0](query1)
        # Pixel masks are supplied after the shared pixel decoder in forward.
        detail = {
            "Q0": content0, "B": base, "Q1": query1, "Qp1": base, "Qp2": position2,
            "semantic_feature": semantic_feature, "context_raw": context_raw,
            "context_feature": context_feature, "position_d": position_d, "position5": position5,
            "decoder1_attention": attention1, "stage1_embedding": stage1_embedding,
        }
        return detail

    def _finish_query_decode(self, detail, pixel_feature, f4_context):
        query1, base = detail["Q1"], detail["B"]
        mask1 = torch.einsum("bqd,bdhw->bqhw", detail["stage1_embedding"], pixel_feature)
        visible2, health2 = previous_mask_visibility(mask1, detail["context_feature"].shape[-2:])
        query2, attention2 = self.decoder2(
            query1 + detail["Qp2"], self._memory(detail["context_feature"]) + detail["position5"], visible2
        )
        position3 = self.focus_update2(attention2["cross_attention"], detail["position5"], base)
        embedding2 = self.mask_embeddings[1](query2)
        mask2 = torch.einsum("bqd,bdhw->bqhw", embedding2, pixel_feature)

        f4_memory = self.f4_memory_projection(f4_context.detach())
        position4 = sine_position_2d(query2.shape[0], *f4_memory.shape[-2:], 256, query2.device, f4_memory.dtype)
        visible3, health3 = previous_mask_visibility(mask2, f4_memory.shape[-2:])
        query3, attention3 = self.decoder3(query2 + position3, self._memory(f4_memory) + position4, visible3)
        embedding3 = self.mask_embeddings[2](query3)
        mask3 = torch.einsum("bqd,bdhw->bqhw", embedding3, pixel_feature)

        confidence1 = self.pca_heads[0](query1)
        confidence2 = self.pca_heads[1](query2)
        confidence3 = self.pca_heads[2](query3)
        return [
            {"query": query1, "position": detail["Qp1"], "mask_embedding": detail["stage1_embedding"],
             "mask_logits": mask1, "confidence": confidence1, "attention": detail["decoder1_attention"],
             "memory_hw": detail["semantic_feature"].shape[-2:], "masked_attention": None},
            {"query": query2, "position": detail["Qp2"], "mask_embedding": embedding2,
             "mask_logits": mask2, "confidence": confidence2, "attention": attention2,
             "memory_hw": detail["context_feature"].shape[-2:], "masked_attention": health2},
            {"query": query3, "position": position3, "mask_embedding": embedding3,
             "mask_logits": mask3, "confidence": confidence3, "attention": attention3,
             "memory_hw": f4_memory.shape[-2:], "masked_attention": health3},
        ], {**detail, "Q2": query2, "Q3": query3, "Qp3": position3, "f4_memory": f4_memory, "position4": position4}

    def forward(self, image, labels, step=0, run_pmec=False):
        if labels is None:
            raise ValueError("FQRF Phase-0 requires train-split image labels")
        features = self.backbone(image)
        deep_cam_logits = self.deep_head(features["FD"])
        deep_logits = F.adaptive_avg_pool2d(deep_cam_logits, 1).flatten(1)
        normalized_cam = normalize_cam(deep_cam_logits)
        targets, target_detail = tri_state_targets(normalized_cam.detach(), labels.bool())

        pixel_feature, pixel_detail = self.pixel_decoder(features["F4"], features["F3"])
        initial = self.query_decode(image, features["FD"], features["F5"], pixel_detail["F4_context"])
        stages, query_detail = self._finish_query_decode(initial, pixel_feature, pixel_detail["F4_context"])

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
            loss_pca = sum(weight * loss for weight, loss in zip(STAGE_WEIGHTS, pca_losses))
            loss_mask = sum(weight * loss for weight, loss in zip(STAGE_WEIGHTS, mask_losses))
            total = 0.50 * loss_deep + 0.25 * loss_pca + 0.25 * loss_mask

        final = stages[-1]
        output = {
            "features": features, "deep_cam_logits": deep_cam_logits, "deep_logits": deep_logits,
            "normalized_cam": normalized_cam, "targets": targets, "target_detail": target_detail,
            "stages": stages, "query_detail": query_detail, "pixel_feature": pixel_feature,
            "pixel_detail": pixel_detail, "query": final["query"], "mask_embedding": final["mask_embedding"],
            "mask_logits": final["mask_logits"], "confidence": final["confidence"],
            "radius": radius, "locality": locality, "mask_detail": final["mask_detail"],
            "losses": {
                "loss": total, "loss_deep": loss_deep, "loss_pca": loss_pca, "loss_mask": loss_mask,
                **{f"loss_pca_stage{index+1}": value for index, value in enumerate(pca_losses)},
                **{f"loss_mask_stage{index+1}": value for index, value in enumerate(mask_losses)},
            },
        }
        if run_pmec:
            region, rows = pmec(
                final["mask_logits"].detach(), final["confidence"]["joint"].detach(),
                final["confidence"]["p_class"].detach(), labels.bool(), locality,
            )
            output["pmec_region"] = region
            output["pmec_rows"] = rows
        return output

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            scratch = not name.startswith("backbone.")
            bias = name.endswith("bias")
            groups[2 * int(scratch) + int(bias)].append(parameter)
        if len({id(value) for group in groups for value in group}) != sum(len(group) for group in groups):
            raise AssertionError("Duplicate optimizer parameter")
        return groups

