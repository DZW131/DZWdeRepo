"""Train-cohort diagnostics and preregistered CQRF Phase-0 gates."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
import torch
from torch.nn import functional as F


def _mean(values): return float(np.mean(values)) if values else 0.0
def _median(values): return float(np.median(values)) if values else 0.0
def _p90(values): return float(np.quantile(values, .9)) if values else 0.0


def _upper(n, device):
    return torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)


def _pair_iou(binary):
    inter = (binary[:, None] & binary[None]).flatten(2).sum(-1).float()
    union = (binary[:, None] | binary[None]).flatten(2).sum(-1).float().clamp_min(1)
    return (inter / union)[_upper(binary.shape[0], binary.device)]


def _pair_cosine(value):
    value = F.normalize(value.float(), dim=-1)
    return (value @ value.T)[_upper(value.shape[0], value.device)]


def _components(binary):
    value = binary.astype(np.uint8)
    seen = np.zeros_like(value, dtype=bool)
    count = 0
    for y, x in zip(*np.where(value)):
        if seen[y, x]: continue
        count += 1; stack = [(int(y), int(x))]; seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0 <= ny < value.shape[0] and 0 <= nx < value.shape[1] and value[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx] = True; stack.append((ny,nx))
    return count


@torch.no_grad()
def batch_health(output, labels):
    values = defaultdict(list); locality = output["locality"]; all_finite = True
    for stage_index, stage in enumerate(output["stages"], 1):
        prefix = f"s{stage_index}"; logits = stage["mask_logits"].float(); probability = logits.sigmoid()
        embedding = stage["mask_embedding"].float(); score = stage["confidence"]["joint"].float().max(-1).values
        top20 = torch.argsort(score, dim=1, descending=True, stable=True)[:, :20]
        all_finite &= bool(torch.isfinite(logits).all() and torch.isfinite(embedding).all())
        for image, indices in enumerate(top20):
            binary = probability[image, indices] > .5
            local = locality[indices]
            local_area = (binary & local).flatten(1).sum(1).float() / local.flatten(1).sum(1).clamp_min(1)
            values[f"{prefix}_global_area"].extend(binary.flatten(1).float().mean(1).cpu().tolist())
            values[f"{prefix}_local_area"].extend(local_area.cpu().tolist())
            values[f"{prefix}_pair_iou"].extend(_pair_iou(binary).cpu().tolist())
            values[f"{prefix}_embedding_cosine"].extend(_pair_cosine(embedding[image, indices]).cpu().tolist())
            values[f"{prefix}_components"].extend([_components(v.cpu().numpy()) for v in binary])
        confidence = stage["confidence"]; p_class = confidence["p_class"].float(); assigned = p_class.argmax(-1)
        present_pairs = int(labels.sum()); covered = 0; absent_dominance = []
        for image in range(labels.shape[0]):
            covered += sum(bool((assigned[image] == cls).any()) for cls in torch.where(labels[image].bool())[0].tolist())
            indices = top20[image]; mass = score[image, indices]; absent = ~labels[image, assigned[image, indices]].bool()
            absent_dominance.append(float(mass[absent].sum() > .5 * mass.sum()))
        values[f"{prefix}_present_pairs"].append(present_pairs)
        values[f"{prefix}_covered"].append(covered)
        values[f"{prefix}_absent_dominance"].extend(absent_dominance)
        values[f"{prefix}_top1_confidence"].extend(score.max(1).values.cpu().tolist())

        if stage_index >= 2:
            detail = stage["detail"]; rc = detail["responsibility_class"].float()
            all_finite &= bool(torch.isfinite(rc).all())
            values[f"{prefix}_integrity_max"].append(detail["integrity_max"])
            values[f"{prefix}_integrity_mean"].append(detail["integrity_mean"])
            update = detail["projected_update"].float(); delta = detail["query_delta"].float()
            old = output["stages"][stage_index-2]["query"].float(); new = stage["query"].float()
            values[f"{prefix}_update_norm"].extend(update.norm(dim=-1).flatten().cpu().tolist())
            values[f"{prefix}_delta_norm"].extend(delta.norm(dim=-1).flatten().cpu().tolist())
            values[f"{prefix}_old_new_cosine"].extend(F.cosine_similarity(old, new, dim=-1).flatten().cpu().tolist())
            height, width = stage["memory_hw"]
            yy, xx = torch.meshgrid(torch.linspace(0,1,height,device=rc.device), torch.linspace(0,1,width,device=rc.device), indexing="ij")
            coords = torch.stack((yy.flatten(), xx.flatten()), -1)
            for image in range(labels.shape[0]):
                for cls in torch.where(labels[image].bool())[0].tolist():
                    maps = rc[image, :, :, cls]; mass = maps.sum(-1); distribution = mass / mass.sum().clamp_min(1e-8)
                    entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum()
                    values[f"{prefix}_dominant_share"].append(float(distribution.max()))
                    values[f"{prefix}_effective_queries"].append(float(entropy.exp()))
                    values[f"{prefix}_normalized_entropy"].append(float(entropy / math.log(maps.shape[0])))
                    top = torch.argsort(mass, descending=True, stable=True)[:5]; selected = maps[top]
                    cutoff = torch.quantile(selected, .8, dim=1, keepdim=True)
                    support = (selected >= cutoff).reshape(5, height, width)
                    values[f"{prefix}_resp_pair_iou"].extend(_pair_iou(support).cpu().tolist())
                    normalized = selected / selected.sum(-1, keepdim=True).clamp_min(1e-8)
                    centroids = normalized @ coords
                    values[f"{prefix}_resp_centroid_distance"].extend(torch.cdist(centroids, centroids)[_upper(5, rc.device)].cpu().tolist())
                    values[f"{prefix}_distinct_peaks"].append(float(torch.unique(selected.argmax(-1)).numel() / 5))

    final = output["stages"][-1]; logits = final["mask_logits"].float(); probability = logits.sigmoid()
    joint = final["confidence"]["joint"].float(); target_hw = logits.shape[-2:]
    positive = F.interpolate(output["target_detail"]["positive"].float(), target_hw, mode="nearest").bool()
    background = F.interpolate(output["target_detail"]["reliable_background"][:,None].float(), target_hw, mode="nearest")[:,0].bool()
    for image in range(logits.shape[0]):
        for cls in torch.where(labels[image].bool())[0].tolist():
            ranked = torch.argsort(joint[image,:,cls], descending=True, stable=True)[:8]
            rival = positive[image,[c for c in range(4) if c != cls]].any(0); pos = positive[image,cls]; bg = background[image]
            for query in ranked.tolist():
                support = locality[query]; selected = probability[image,query] > .5
                if bool((pos & support).any()):
                    values["positive_recall"].append(float((selected & pos & support).sum() / (pos & support).sum()))
                    values["positive_logit"].extend(logits[image,query][pos & support].cpu().tolist())
                if bool((rival & support).any()):
                    values["rival_leakage"].append(float((selected & rival & support).sum() / (rival & support).sum()))
                    values["negative_logit"].extend(logits[image,query][rival & support].cpu().tolist())
                if bool((bg & support).any()):
                    values["background_leakage"].append(float((selected & bg & support).sum() / (bg & support).sum()))
                    values["negative_logit"].extend(logits[image,query][bg & support].cpu().tolist())
    gate = output["deep_gate"].float()
    values["deep_present"].extend(gate[labels.bool()].cpu().tolist())
    values["deep_absent"].extend(gate[~labels.bool()].cpu().tolist())
    for prefix, raw, context in (("F5", output["query_detail"]["context_raw"], output["query_detail"]["context_feature"]),
                                 ("F4", output["pixel_detail"]["F4_raw"], output["pixel_detail"]["F4_context"])):
        values[f"{prefix}_residual"].append(float((context.float()-raw.float()).square().mean().sqrt()))
        values[f"{prefix}_cosine"].append(float(F.cosine_similarity(raw.float(),context.float(),dim=1).mean()))
    values["all_finite"].append(float(all_finite))
    return values


def summarize(snapshot, batches, pmec_rows, model):
    merged = defaultdict(list)
    for batch in batches:
        for key, value in batch.items(): merged[key].extend(value)
    area=[]; red=[]; emb=[]; pca=[]; integrity=[]; utilization=[]; complement=[]; updates=[]; fragments=[]
    for stage in (1,2,3):
        p=f"s{stage}"; local=merged[f"{p}_local_area"]; pairs=merged[f"{p}_pair_iou"]; cosine=merged[f"{p}_embedding_cosine"]
        area.append({"snapshot":snapshot,"stage":stage,"global_mean":_mean(merged[f"{p}_global_area"]),"global_median":_median(merged[f"{p}_global_area"]),"local_mean":_mean(local),"local_median":_median(local),"local_fraction_gt_090":_mean([v>.90 for v in local]),"local_fraction_lt_005":_mean([v<.05 for v in local])})
        red.append({"snapshot":snapshot,"stage":stage,"pair_iou_mean":_mean(pairs),"pair_iou_median":_median(pairs),"pair_iou_fraction_gt_090":_mean([v>.90 for v in pairs]),"pair_iou_fraction_gt_075":_mean([v>.75 for v in pairs])})
        emb.append({"snapshot":snapshot,"stage":stage,"cosine_mean":_mean(cosine),"cosine_median":_median(cosine),"cosine_p90":_p90(cosine),"cosine_fraction_gt_095":_mean([v>.95 for v in cosine])})
        pca.append({"snapshot":snapshot,"stage":stage,"present_class_query_coverage":sum(merged[f"{p}_covered"])/max(sum(merged[f"{p}_present_pairs"]),1),"absent_class_dominance":_mean(merged[f"{p}_absent_dominance"]),"top1_confidence":_mean(merged[f"{p}_top1_confidence"]),"all_finite":bool(min(merged["all_finite"] or [0]))})
        fragments.append({"snapshot":snapshot,"stage":stage,"component_mean":_mean(merged[f"{p}_components"]),"component_p90":_p90(merged[f"{p}_components"]),"fraction_gt_5":_mean([v>5 for v in merged[f"{p}_components"]])})
        if stage >= 2:
            integrity.append({"snapshot":snapshot,"stage":stage,"max_abs_sum_error":max(merged[f"{p}_integrity_max"] or [0]),"mean_abs_sum_error":_mean(merged[f"{p}_integrity_mean"])})
            utilization.append({"snapshot":snapshot,"stage":stage,"dominant_share_median":_median(merged[f"{p}_dominant_share"]),"dominant_share_mean":_mean(merged[f"{p}_dominant_share"]),"effective_queries_median":_median(merged[f"{p}_effective_queries"]),"effective_queries_mean":_mean(merged[f"{p}_effective_queries"]),"normalized_entropy_median":_median(merged[f"{p}_normalized_entropy"])})
            complement.append({"snapshot":snapshot,"stage":stage,"responsibility_iou_mean":_mean(merged[f"{p}_resp_pair_iou"]),"responsibility_iou_median":_median(merged[f"{p}_resp_pair_iou"]),"centroid_distance_mean":_mean(merged[f"{p}_resp_centroid_distance"]),"distinct_peak_fraction":_mean(merged[f"{p}_distinct_peaks"])})
            updates.append({"snapshot":snapshot,"stage":stage,"projected_update_norm_mean":_mean(merged[f"{p}_update_norm"]),"projected_update_norm_p90":_p90(merged[f"{p}_update_norm"]),"query_delta_norm_mean":_mean(merged[f"{p}_delta_norm"]),"query_delta_norm_p90":_p90(merged[f"{p}_delta_norm"]),"old_new_cosine_mean":_mean(merged[f"{p}_old_new_cosine"])})
    pos=_mean(merged["positive_logit"]); neg=_mean(merged["negative_logit"])
    semantic={"snapshot":snapshot,"positive_seed_recall":_mean(merged["positive_recall"]),"rival_leakage":_mean(merged["rival_leakage"]),"background_leakage":_mean(merged["background_leakage"]),"positive_mean_logit":pos,"negative_mean_logit":neg,"positive_negative_gap":pos-neg}
    pmec_health={"snapshot":snapshot,"present_class_pairs":len(pmec_rows),"candidate_coverage":_mean([r["candidate_masks"]>=1 for r in pmec_rows]),"differs_from_top1":_mean([r["differs_from_top1"] for r in pmec_rows]),"candidate_masks_per_pair":_mean([r["candidate_masks"] for r in pmec_rows]),"groups_per_pair":_mean([r["region_groups"] for r in pmec_rows])}
    present=_mean(merged["deep_present"]); absent=_mean(merged["deep_absent"])
    return {"stagewise_mask_area":area,"stagewise_query_redundancy":red,"stagewise_embedding_diversity":emb,"responsibility_integrity":integrity,"responsibility_utilization":utilization,"responsibility_complementarity":complement,"query_update_health":updates,"over_fragmentation":fragments,"semantic_selectivity":semantic,"pca_health":pca,"pmec_health":pmec_health,"deep_gate_health":{"snapshot":snapshot,"present_mean":present,"absent_mean":absent,"present_absent_gap":present-absent},"chpf_health":{"snapshot":snapshot,"gamma5":float(model.f5_chpf.gamma),"gamma4":float(model.pixel_decoder.f4_chpf.gamma),"F5_context_residual_norm":_mean(merged["F5_residual"]),"F4_context_residual_norm":_mean(merged["F4_residual"]),"F5_raw_context_cosine":_mean(merged["F5_cosine"]),"F4_raw_context_cosine":_mean(merged["F4_cosine"])},"all_finite":bool(min(merged["all_finite"] or [0]))}


def _row(summary,key,stage): return next(r for r in summary[key] if r["stage"]==stage)


def apply_epoch2_screen(summary):
    red=_row(summary,"stagewise_query_redundancy",3); area=_row(summary,"stagewise_mask_area",3); sem=summary["semantic_selectivity"]; util=_row(summary,"responsibility_utilization",3); integ=_row(summary,"responsibility_integrity",3)
    checks={"no_competitive_effect":red["pair_iou_median"]>=.93 and red["pair_iou_fraction_gt_090"]>=.55,"responsibility_monopoly":util["dominant_share_median"]>.90 and util["effective_queries_median"]<1.5,"rival_leakage":sem["rival_leakage"]>.70,"background_leakage":sem["background_leakage"]>.50,"nonpositive_gap":sem["positive_negative_gap"]<=0,"high_area":area["local_median"]>.90 or area["local_fraction_gt_090"]>.80,"tiny_area":area["local_median"]<.05,"responsibility_normalization":integ["max_abs_sum_error"]>1e-5,"nonfinite":not summary["all_finite"]}
    failed=[k for k,v in checks.items() if v]
    return {"decision":"CQRF_CCRA_NOGO" if failed else "CONTINUE_TO_EPOCH5","checks":checks,"failed_criteria":failed}


def apply_final_gate(history):
    summary=history[-1]; red=_row(summary,"stagewise_query_redundancy",3); area=_row(summary,"stagewise_mask_area",3); sem=summary["semantic_selectivity"]; util=_row(summary,"responsibility_utilization",3); pca=_row(summary,"pca_health",3); pmec=summary["pmec_health"]
    def snapshot_name(value):
        if "snapshot" in value:
            return value["snapshot"]
        for key in ("semantic_selectivity", "pmec_health", "deep_gate_health"):
            if isinstance(value.get(key), dict) and "snapshot" in value[key]:
                return value[key]["snapshot"]
        for key in ("stagewise_query_redundancy", "stagewise_mask_area"):
            if value.get(key):
                return value[key][0].get("snapshot")
        return None
    by={snapshot_name(s):s for s in history}; epochs=[by.get(f"epoch{i}") for i in (3,4,5)]
    stable=len(history)>=3 and all(epochs) and _row(epochs[1],"stagewise_query_redundancy",3)["pair_iou_median"]<=_row(epochs[0],"stagewise_query_redundancy",3)["pair_iou_median"]+.05 and red["pair_iou_median"]<=_row(epochs[1],"stagewise_query_redundancy",3)["pair_iou_median"]+.05 and red["pair_iou_fraction_gt_090"]<=_row(epochs[0],"stagewise_query_redundancy",3)["pair_iou_fraction_gt_090"]+.05
    checks={"A_median_iou":red["pair_iou_median"]<.75,"A_high_iou":red["pair_iou_fraction_gt_090"]<.30,"B_no_rebound":stable,"C_rival":sem["rival_leakage"]<.30,"C_background":sem["background_leakage"]<.10,"C_gap":sem["positive_negative_gap"]>5,"D_median_area":.15<=area["local_median"]<=.75,"D_high_area":area["local_fraction_gt_090"]<.40,"D_tiny_area":area["local_fraction_lt_005"]<.40,"E_dominant_share":util["dominant_share_median"]<.75,"E_effective_queries":util["effective_queries_median"]>1.5,"F_pca_coverage":pca["present_class_query_coverage"]>=.93,"F_absent_dominance":pca["absent_class_dominance"]<=.20,"F_finite":bool(pca["all_finite"] and summary["all_finite"]),"G_pmec_candidates":pmec["candidate_coverage"]>=.85,"G_pmec_difference":pmec["differs_from_top1"]>=.70}
    go=all(checks.values())
    strong=go and red["pair_iou_median"]<.60 and red["pair_iou_fraction_gt_090"]<.20 and sem["rival_leakage"]<.20 and sem["background_leakage"]<.05 and sem["positive_negative_gap"]>10 and pca["present_class_query_coverage"]>=.95 and pca["absent_class_dominance"]<=.10 and pmec["candidate_coverage"]>=.90
    return {"decision":"CQRF_PHASE0_STRONG_GO" if strong else "CQRF_PHASE0_GO" if go else "CQRF_CCRA_NOGO","checks":checks,"strong_go":strong}
