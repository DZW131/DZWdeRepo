"""Render the preregistered MOMD Phase-0 Markdown report."""
from pathlib import Path


def _row(summary,key,stage=3): return next((x for x in summary.get(key,[]) if x.get("stage")==stage),{})
def _f(row,key):
    value=row.get(key); return "n/a" if value is None else f"{float(value):.4f}"


def render_report(output,result):
    output=Path(output); s=result.get("final_summary",{}); gate=result.get("gate",{}); decision=result["decision"]
    r=_row(s,"routing_integrity"); cap=_row(s,"capacity_preservation"); neg=_row(s,"negative_capacity")
    cc=_row(s,"contribution_complementarity"); tr=_row(s,"r_to_q_transfer"); con=_row(s,"expert_owner_contrast")
    area=_row(s,"primary_mask_health"); frag=_row(s,"fragmentation"); pca=_row(s,"pca_health"); deep=s.get("deep_gate_health",{})
    sections=[
      ("Executive Decision",f"Preregistered train-only result: `{decision}`. No validation/test segmentation labels or metrics were used."),
      ("CQRF Frozen Ownership Evidence","The audited CQRF allocator was preserved byte-for-byte at the architecture boundary."),
      ("RPMC-v1 Bypass Failure","Historical context: additive ownership priors were bypassable."),
      ("COMD-v1 Over-Suppression Failure","Historical context: multiplicative ownership ceilings suppressed semantic capacity."),
      ("New Allocation-vs-Attenuation Diagnosis","MOMD treats ownership as mixture allocation, not output attenuation."),
      ("Literature-Migration Principle","A normalized mixture preserves common evidence amplitude while permitting expert specialization."),
      ("MOMD Hypothesis","CCRA ownership can route expert evidence without destroying class-mask capacity."),
      ("Architectural Delta","Only parameter-free responsibility resizing, locality routing, and convex mixture decoding were added."),
      ("CCRA Frozen Contract","Stage2/3 CCRA, priors, detach points, FFNs, memories, and temperature remained frozen."),
      ("MOMD Detach Boundary","R is detached; gradients reach semantic expert logits through F, but not CCRA through the routing side path."),
      ("Responsibility Resize","FP32 bilinear resize to 56x56 with align_corners=False."),
      ("Locality Reuse","The exact Full25 circular locality schedule was reused."),
      ("Query-Axis Mixture Routing","A is normalized over 196 queries for every pixel and class."),
      ("Routing Sum Integrity",f"max |sum A-1|={_f(r,'max_A_sum_error')}."),
      ("Query Semantic Experts","B is sigmoid of each unchanged base mask logit."),
      ("Class-Conditioned Mixture F",f"Endpoint CPR={_f(cap,'CPR')}; median positive F={_f(cap,'positive_F_median')}."),
      ("Contribution Maps C",f"max |sum C-F|={_f(r,'max_C_sum_error')}."),
      ("Posterior Contribution Shares Q",f"max posterior sum error={_f(r,'max_Q_sum_error')}."),
      ("No Double Ownership Weighting","Stage2/3 loss supervises F directly; no per-query R weighting is present."),
      ("Gradient-Routing Logic","Detached A scales expert-logit gradients through the convex mixture only."),
      ("Tri-State Contract","Frozen CAM positives, rival/background negatives, ignore pixels, and >=4 valid-pixel rule."),
      ("Stage1 Loss","Original CQRF query-level loss is unchanged."),
      ("Stage2/3 Mixture Loss","Class-level BCEWithLogits after clamped probability-to-logit conversion."),
      ("Total Loss Contract","Stage .20/.30/.50; macro .50 deep + .25 PCA + .25 mask."),
      ("Primary Stage3 Output","Stage3 F_c is the sole primary foreground output."),
      ("PMEC Diagnostic-Only Status","Historical PMEC uses ungated base masks and is excluded from loss/decision output."),
      ("Parameter Count",f"MOMD parameter delta={result.get('momd_delta_parameters','n/a')}."),
      ("Historical CQRF Compatibility Audit",str(result.get("cqrf_compatibility",{}).get("nontriviality",{}))),
      ("E1-E5 Routing Health","See `momd_routing_integrity.csv` and `momd_locality_fallback.csv`."),
      ("E1-E5 Capacity Preservation",f"CPR={_f(cap,'CPR')}; empty fraction={_f(cap,'present_class_empty_fraction')}."),
      ("Positive-Seed Recall",f"Median recall={_f(cap,'positive_seed_recall_median')}."),
      ("Semantic Selectivity",f"Rival={_f(neg,'rival_leakage')}; background={_f(neg,'background_leakage')}; probability gap={_f(neg,'probability_gap')}."),
      ("Class-Mask Area",f"Median={_f(area,'median_area')}; <1%={_f(area,'fraction_area_lt_001')}; >90%={_f(area,'fraction_area_gt_090')}."),
      ("Fragmentation",f"Component p90={_f(frag,'component_p90')}."),
      ("Contribution Complementarity",f"Posterior support IoU={_f(cc,'posterior_support_iou_median')}; distinct peaks={_f(cc,'distinct_contribution_peaks')}."),
      ("R→Q Transfer",f"T_R→Q={_f(tr,'T_R_to_Q')}."),
      ("Expert Owner Contrast",f"Median={_f(con,'median')}."),
      ("Expert Ownership Correlation","See `momd_expert_ownership_correlation.csv`."),
      ("Base Expert Redundancy","See `momd_base_expert_redundancy.csv`; overlap alone is not a rejection condition."),
      ("Routed Expert Utilization","See `momd_routed_expert_utilization.csv`."),
      ("PCA Health",f"Coverage={_f(pca,'present_class_query_coverage')}; absent dominance={_f(pca,'absent_class_dominance')}."),
      ("Deep Gate Health",f"Present-absent gap={_f(deep,'present_absent_gap')}."),
      ("Query Update Health","Frozen CCRA query-update diagnostics are recorded by stage and snapshot."),
      ("Gradient Health","Per-module gradient RMS/zero/nonfinite statistics are recorded every 100 steps."),
      ("Gradient Routing Audit","A-to-|dL/dZ| correlations are reported at registered snapshots."),
      ("Visual Evidence","Eight fixed train images are rendered at registered visual snapshots."),
      ("Epoch2 Catastrophic Screen",str(result.get("epoch2_screen"))),
      ("E5 GO Gate","\n".join(f"- {k}: {v}" for k,v in gate.get("gate_groups",{}).items()) or "Not reached."),
      ("STRONG GO Gate",f"Strong GO={gate.get('strong_go',False)}."),
      ("Scientific Interpretation","MOMD is accepted only if it conserves evidence, stays selective, and transfers ownership into complementary semantic contributions."),
      ("Exact Decision",decision),
      ("Full25 Recommendation","Freeze CCRA+MOMD and start a fresh official-init Full25 only for GO/STRONG_GO; otherwise archive MOMD-v1 without rescue tuning."),
    ]
    assert len(sections)==52
    lines=["# MOMD Phase-0 — Mass-Preserving Ownership Mixture Decoding Report",""]
    for i,(title,body) in enumerate(sections,1): lines += [f"## {i}. {title}","",body,""]
    lines.append(f"DECISION = {decision}")
    path=output/"MOMD_Phase0_Mass_Preserving_Ownership_Mixture_Decoding_Report.md"
    path.write_text("\n".join(lines)+"\n",encoding="utf-8"); return path
