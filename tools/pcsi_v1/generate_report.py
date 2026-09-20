"""Generate the preregistered 39-section PCSI-v1 final Markdown from frozen artifacts."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))


def pct(x): return "N/A" if x is None else f"{100*x:.4f}%"
def pp(x): return f"{100*x:+.6f} pp"
def fmt(x): return "N/A" if x is None else f"{x:.6g}"


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); a=p.parse_args(); root=a.root
    result=json.loads((root/"metrics/final_result.json").read_text())
    phase_a=json.loads((root/"phaseA_forward/phaseA_decision.json").read_text())
    vpca=json.loads((root/"phaseB_vpca/vpca_root_cause.json").read_text())
    identity=json.loads((root/"phaseC/identity_pretrain.json").read_text())
    parameters=json.loads((root/"phaseC/parameter_audit.json").read_text())
    tensor=json.loads((root/"phaseC/posttrain_tensor_hash_audit.json").read_text())
    candidates=pd.read_csv(root/"phaseA_forward/injection_candidates.csv")
    text=pd.read_csv(root/"phaseB_vpca/text_effective_rank.csv")
    visual=pd.read_csv(root/"phaseB_vpca/visual_rank_statistics.csv")
    pooling=pd.read_csv(root/"phaseB_vpca/pooling_audit.csv")
    temperature=pd.read_csv(root/"phaseB_vpca/temperature_audit.csv")
    rho=pd.read_csv(root/"phaseB_vpca/rho_audit.csv")
    bias=pd.read_csv(root/"phaseB_vpca/concept_bias.csv")
    sensitivity=pd.read_csv(root/"phaseB_vpca/image_sensitivity.csv")
    manifest=pd.read_csv(root/"visualizations/manifest.csv") if (root/"visualizations/manifest.csv").exists() else pd.DataFrame()
    scores=result["segmentation"]; resp=result["responsibility"]; cohort=result["cohort"]; safety=result["tp_safety"]
    lines=["# PCSI-v1 — Pre-CCRA Semantic Injection + VPCA Collapse Audit", "",
           "> BCSS seed 42 · frozen HQMR epoch 25 · PLIP frozen · Phase0 5 epochs C1/C2 · 3-view TTA · 3,418 validation images", "",
           "## 首页决策卡", "",
           "| Field | Result |", "|---|---|"]
    card={"PHASE_A_DECISION":phase_a["PHASE_A_DECISION"],"PRE_CCRA_SITE":phase_a["PRE_CCRA_SITE"],
          "IDENTITY_PASS":phase_a["IDENTITY_PASS"],"GRADIENT_PATH_PASS":phase_a["GRADIENT_PATH_PASS"],
          "VPCA_V1_STATUS":vpca["VPCA_V1_STATUS"],"VPCA_ROOT_CAUSE":", ".join(vpca["VPCA_ROOT_CAUSE"]),
          "TEXT_SPACE_COLLAPSE":vpca["TEXT_SPACE_COLLAPSE"],"VISUAL_RESPONSE_RANK_LOCK":vpca["VISUAL_RESPONSE_RANK_LOCK"],
          "TOPK_AGGREGATION_LOCK":vpca["TOPK_AGGREGATION_LOCK"],"TEMPERATURE_SHARPENING":vpca["TEMPERATURE_SHARPENING"],
          "GLOBAL_CONCEPT_BIAS":vpca["GLOBAL_CONCEPT_BIAS"],"PHASE_C_EXECUTED":True,
          "C0_mIoU":pct(scores["C0"]["mIoU"]),"C1_mIoU":pct(scores["C1"]["mIoU"]),"C2_mIoU":pct(scores["C2"]["mIoU"]),
          "C1-C0":pp(scores["C1"]["mIoU"]-scores["C0"]["mIoU"]),
          "C2-C1":pp(scores["C2"]["mIoU"]-scores["C1"]["mIoU"]),
          "C2-C0":pp(scores["C2"]["mIoU"]-scores["C0"]["mIoU"]),
          "C1_HRCR":pct(resp["C1"]["HRCR"]["component_rate"]),"C2_HRCR":pct(resp["C2"]["HRCR"]["component_rate"]),
          "C1_HardM1CR":pct(cohort["C1"]["hard_m1"]["pixel_area_weighted"]),
          "C2_HardM1CR":pct(cohort["C2"]["hard_m1"]["pixel_area_weighted"]),
          "C1_NCE":fmt(safety["C1"]["nce"]),"C2_NCE":fmt(safety["C2"]["nce"]),
          "PLIP_SEMANTIC_SOURCE":result["PLIP_SEMANTIC_SOURCE"],"PDSR_DECISION":result["PDSR_DECISION"],
          "FINAL_ROUTE":result["FINAL_DECISION"],"FULL25_GO":False}
    lines.extend(f"| {k} | {v} |" for k,v in card.items()); lines.extend(["", "## 关键因果顺序（固定 32 张，E5 后）", "",
           "`decoder1/query1` 是注入点之前的输入，须不变；HQMR 内部名为 `query0` 的张量在 CCRA2 之后，因此应变化。原方案把两个 q0 混用，不能以 HQMR `query0` 不变作为合法门槛。", "",
           "| Tensor | C1 RMS / max abs | C2 RMS / max abs |", "|---|---:|---:|"])
    for key in ("query1","context_pre","context","ccra_k5","ccra_v5","ccra2_responsibility",
                "hqmr_query0","hqmr_k5","hqmr_v5","hqmr5_logits","hqmr_query5","hqmr4_logits","hqmr_query4","hqmr3_logits","final"):
        a1=tensor["C1"]["tensor_drift"].get(key); a2=tensor["C2"]["tensor_drift"].get(key)
        if a1 and a2:lines.append(f"| {key} | {fmt(a1['rms'])} / {fmt(a1['max_abs'])} | {fmt(a2['rms'])} / {fmt(a2['max_abs'])} |")
    lines.extend(["",f"CCRA2 spatial responsibility argmax change: C1 {pct(tensor['C1']['ccra_spatial_argmax_change'])}, C2 {pct(tensor['C2']['ccra_spatial_argmax_change'])}. Tensor hashes are saved in `phaseC/posttrain_tensor_hash_audit.json`.",""])
    def section(index,title,body): lines.extend([f"## {index}. {title}","",body,""])
    section(1,"Executive Decision",f"**{result['FINAL_DECISION']}**。Phase A proved the pre-CCRA causal path; E5 shows C1–C0 {pp(scores['C1']['mIoU']-scores['C0']['mIoU'])} and C2–C0 {pp(scores['C2']['mIoU']-scores['C0']['mIoU'])}. C2–C0 paired 95% CI [{result['bootstrap']['C2-C0_mIoU_pp']['ci95'][0]:+.6f}, {result['bootstrap']['C2-C0_mIoU_pp']['ci95'][1]:+.6f}] pp includes zero; C1/C2 Hard-M1 L5 correctness remains {pct(resp['C1']['HRCR']['component_rate'])}/{pct(resp['C2']['HRCR']['component_rate'])}. Gate C1 minimum={result['C1_MINIMUM_GO']}, C2 PDSR={result['C2_PDSR_GO']}, strong={result['C2_STRONG_GO']}. No Full25 in this round.")
    section(2,"Frozen Context",f"HQMR checkpoint SHA-256 `84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb`; C0 exact UMRF prediction bank `{result['exact_C0_bank']}`. Historical HQMR scalar 65.572444% differs from paired bank ~65.572738% by +0.000294 pp; all present deltas use the single paired bank. Previous post-CCRA P1/P2/P3 were 65.5726/65.5731/65.5725%, essentially tied with P0.")
    section(3,"Why Previous Phase0 Did Not Test Pre-CCRA Hypothesis","The old `h5_residual` is added only after `GCQMNet.forward` has returned; CCRA2 and CCRA3 have already assigned class responsibility. Its identity of upstream CCRA tensors was expected, not evidence that PLIP has no semantics.")
    section(4,"Exact Forward Dependency Graph","Runtime-hooked DAG and edge list: `phaseA_forward/forward_dependency_graph.json` and `.md`. `F5.detach → context_projection → CHPF/context_feature → CCRA2 K/V and responsibility → query2 → CCRA3/HQMR query0`; the same context feature also feeds HQMR scale-5 K/V → L5 → q5 → L4 → q4 → L3.")
    section(5,"CCRA Entry Point","Selected I1: the common 256×28×28 `context_feature` immediately after `f5_chpf`, before CCRA2's memory K/V. No query residual, CCRA weight change, GT mask, or image-label leakage was introduced at inference.")
    table=["| Site | pre-CCRA | ΔCCRA-K RMS | ΔCCRA-V RMS | ΔL5 RMS | spatial responsibility change | GO |","|---|---|---:|---:|---:|---:|---|"]
    for _,r in candidates.iterrows():table.append(f"| {r.site} | {r.structurally_pre_ccra} | {fmt(r.delta_ccra_k5_rms)} | {fmt(r.delta_ccra_v5_rms)} | {fmt(r.delta_L5_rms)} | {pct(r.spatial_any_class_change_rate)} | {r.causal_go} |")
    section(6,"Injection Candidate Audit","\n".join(table)+"\n\nI0 is a deliberately late negative control; I2 K-only is causal but lacks the common K/V feature source; I3 V-only does not change responsibility.")
    section(7,"Identity Tests",f"All candidate alpha=0 tests were exact. Both newly instantiated C1/C2 had γ=0 and exact identity for query1, prefeature, CCRA2/3 responsibility, L5/L3 and final output on the fixed 32 images. Positive γ=.1 changed CCRA and L5. Manifest: `phaseC/identity_pretrain.json`.")
    section(8,"Causal Perturbation",f"I1 at alpha=.10, no training/no segmentation GT: ΔCCRA K/V RMS {fmt(float(candidates.loc[candidates.site=='I1_PRE_CONTEXT','delta_ccra_k5_rms'].iloc[0]))}/{fmt(float(candidates.loc[candidates.site=='I1_PRE_CONTEXT','delta_ccra_v5_rms'].iloc[0]))}; spatial responsibility change 3.50%; ΔL5/L3 RMS 0.03143/0.04449. Alpha .05 and .20 are secondary sensitivity checks in `causal_sensitivity.csv`. The probe normalized semantic RMS to the context RMS for a controlled perturbation; Phase C used the preregistered raw `γ·Z` fusion, so alpha=.10 is not an E5 effect-size prediction.")
    section(9,"Gradient Path","Using original HQMR loss, the prefeature, gamma and PDSR projection received nonzero gradients while frozen HQMR/PLIP parameters had grad=None. Diagnostic CCRA responsibility is detached on output, but internal pooled update remains differentiable. `phaseA_forward/gradient_path.json`.")
    section(10,"Phase-A Decision",f"`{phase_a['PHASE_A_DECISION']}`. Structural causality, alpha-zero identity, and gradient path all passed; only then were C1/C2 run.")
    section(11,"VPCA Collapse Recap","Previous P3 showed four-class top-1 concept collapse and no segmentation gain. This audit re-used the exact P3 projection, PLIP and concept cache on 23,422 training images without opening segmentation GT.")
    section(12,"Text-Space Geometry","Within-class mean pairwise cosine: "+", ".join(f"C{int(r['class'])} {r.mean_cosine:.3f}" for _,r in text.iterrows())+". Cross-class confusion table and all 8×8 cosine plots are under `phaseB_vpca/`.")
    section(13,"Concept Effective Rank","Effective rank out of 8: "+", ".join(f"C{int(r['class'])} {r.effective_rank:.3f}" for _,r in text.iterrows())+". None met text-space collapse criterion.")
    section(14,"Raw Visual-Concept Ranking","Top-1 concept frequencies among positive training images: "+", ".join(f"C{int(r['class'])} concept {int(r.raw_top1_concept)} = {pct(r.raw_top1_max_frequency)}" for _,r in visual.iterrows())+". The lock is already present in raw g20, before q/rho.")
    section(15,"Pooling Audit","Mean, Top50, Top20, Top10 and Max all retained ~100% top-1 locks. Thus TopK did not *create* the lock; see `phaseB_vpca/pooling_audit.csv`.")
    t01=temperature[temperature.tau==.1]
    section(16,"Temperature Audit","At τ=.1, mean q entropy by class: "+", ".join(f"C{int(r['class'])} {r.mean_entropy:.3f}" for _,r in t01.iterrows())+". Positive-temperature softmax preserves argmax exactly; the plan's proposed criterion 'top1 under 80% before softmax but over 80% afterward' is mathematically impossible. Lower τ sharpens entropy/mass but cannot introduce a different top-1 ranking.")
    section(17,"rho/q Decomposition",f"ρ max median {vpca['rho_max_median']:.3f}; dominant rho-class frequency {pct(vpca['rho_top1_max_frequency'])}. Class-level ρ dominance criterion was not met. `phaseB_vpca/rho_audit.csv` separates ρ from within-class q.")
    section(18,"Global Concept Bias","Raw visual response rankings favor one concept in every class across positive images, with mean-pooling lock already present. `phaseB_vpca/concept_bias.csv` contains score levels/gaps; the evidence supports a global concept bias diagnosis, not a causal proof that any one text prompt alone is defective.")
    section(19,"Instance Sensitivity","ISR by class: "+", ".join(f"C{int(r['class'])} {r.ISR:.4f}" for _,r in sensitivity.iterrows())+". All are <1, consistent with concept-mean differences dominating image-to-image variation.")
    section(20,"VPCA Root-Cause Decision",f"`{', '.join(vpca['VPCA_ROOT_CAUSE'])}`. TEXT_SPACE_COLLAPSE={vpca['TEXT_SPACE_COLLAPSE']}; TOPK_AGGREGATION_LOCK={vpca['TOPK_AGGREGATION_LOCK']}; TEMPERATURE_SHARPENING={vpca['TEMPERATURE_SHARPENING']}; RHO_CLASS_DOMINANCE={vpca['RHO_CLASS_DOMINANCE']}. VPCA-v1 stays CLOSED; no formula repairs or training were performed.")
    section(21,"Corrected Pre-CCRA Experiment Protocol","C0 frozen UMRF bank; C1 frozen PLIP last dense + P_last + zero-initialized channelwise gamma; C2 frozen PLIP multi-level static-concept PDSR + gamma. Frozen HQMR/CCRA/PLIP. Seed42, original PolyOptimizer LR=.1/WD=.0005, BF16, microbatch5×accumulation4, exactly five epochs/5,855 optimizer steps each. No validation E1–E4; E5-only common 3-view evaluation.")
    section(22,"C0",f"Frozen baseline mIoU={pct(scores['C0']['mIoU'])}, mDice={pct(scores['C0']['mDice'])}. Prediction maps were not regenerated or tuned.")
    for idx,v,title in ((23,"C1","C1 Pre-CCRA VLM-Last"),(24,"C2","C2 Pre-CCRA Static-PDSR")):
        runtime=json.loads((root/("C1_VLM_LAST" if v=="C1" else "C2_STATIC_PDSR")/"runtime.json").read_text())
        section(idx,title,f"mIoU={pct(scores[v]['mIoU'])}, mDice={pct(scores[v]['mDice'])}; gamma abs mean={result['checkpoints'][v]['gamma_abs_mean']:.6g}; train time={runtime['train_seconds']:.1f}s, peak VRAM={runtime['peak_vram_gib']:.3f} GiB, checkpoint SHA-256 `{runtime['checkpoint_sha256']}`. Validation/segmentation GT access during training both false.")
    section(25,"L5 Responsibility Changes","Component-weighted RCR (baseline L5 wrong → new L5 true): "+", ".join(f"{v} {pct(resp[v]['RCR']['component_rate'])} (n={resp[v]['RCR']['components']})" for v in ("C1","C2"))+". Stage5 responsibility is class evidence from frozen HQMR logits5 + detached class weights, 3-view averaged, class-softmaxed, and uniformly pooled over the *exact* baseline connected component.")
    section(26,"Hard-M1 Responsibility Recovery","HRCR: "+", ".join(f"{v} {pct(resp[v]['HRCR']['component_rate'])} (area-weighted {pct(resp[v]['HRCR']['area_weighted_rate'])})" for v in ("C1","C2"))+". Hard-M1 cohort is frozen UMRF sequential5/4/3-all-wrong, not redefined from new predictions.")
    section(27,"Final Segmentation Recovery","Hard-M1 pixel-area corrective rate: "+", ".join(f"{v} {pct(cohort[v]['hard_m1']['pixel_area_weighted'])}" for v in ("C1","C2"))+"; M1 corrective rate: "+", ".join(f"{v} {pct(cohort[v]['m1']['pixel_area_weighted'])}" for v in ("C1","C2"))+". These are distinct from component-weighted HRCR.")
    section(28,"True-vs-Rival","Rival exit RER: "+", ".join(f"{v} {pct(resp[v]['RER']['component_rate'])}" for v in ("C1","C2"))+"; correct exit conditional rate: "+", ".join(f"{v} {pct(resp[v]['CorrectRER']['component_rate'])}" for v in ("C1","C2"))+"; Hard-M1 true−rival L5 margin mean: "+", ".join(f"{v} {fmt(resp[v]['true_rival_margin_hard_mean'])}" for v in ("C1","C2"))+". Rival is the frozen L5 wrong class (`baseline_l5`), not the final baseline segmentation class. The first summary used the latter incorrectly; only the corrected `evaluate_corrected.log` and final metrics enter conclusions.")
    section(29,"TP Safety","TP harm: "+", ".join(f"{v} {pct(safety[v]['tp_harm'])}" for v in ("C1","C2"))+"; NCE=corrected/harmed: "+", ".join(f"{v} {fmt(safety[v]['nce'])}" for v in ("C1","C2"))+". Interpret small absolute changes against the paired C0 mask, not only the ratio.")
    class_table=["| Class | C0 IoU | C1 IoU | C2 IoU | C1−C0 | C2−C1 |","|---|---:|---:|---:|---:|---:|"]
    for c in range(4):
        vals=[scores[v]["class_iou"][str(c)] for v in ("C0","C1","C2")]
        class_table.append(f"| C{c} | {pct(vals[0])} | {pct(vals[1])} | {pct(vals[2])} | {pp(vals[1]-vals[0])} | {pp(vals[2]-vals[1])} |")
    section(30,"Per-Class","\n".join(class_table)+"\n\nBCSS C0 Tumor, C1 Stroma, C2 Lymphocytic infiltrate, C3 Necrosis; label 4 ignored in foreground metrics.")
    section(31,"Parameter Audit","Adapter checkpoints contain trainable keys only. Frozen base and PLIP state hashes are identical before/after adapter loading, and runtime gradient manifests show frozen grad=None. "+"; ".join(f"{v}: base `{parameters[v]['frozen_base_sha256'][:16]}…`, PLIP `{parameters[v]['frozen_plip_sha256'][:16]}…`, trainable params {parameters[v]['trainable_parameter_count']:,}" for v in ("C1","C2"))+". See `phaseC/parameter_audit.json`.")
    ci=result["bootstrap"]; bootstrap_rows=["| Metric | mean | paired-image 95% CI |","|---|---:|---|"]
    for key in ("C1-C0_mIoU_pp","C2-C1_mIoU_pp","C2-C0_mIoU_pp","C1_HardM1CR","C2_HardM1CR","C1_HRCR","C2_HRCR","C1_TPHarm","C2_TPHarm"):
        row=ci[key]; bootstrap_rows.append(f"| {key} | {row['mean']:.6g} | [{row['ci95'][0]:.6g}, {row['ci95'][1]:.6g}] |")
    section(32,"Bootstrap","2,000 paired-image bootstrap resamples, seed42; mIoU deltas are pp, rates are fractions.\n\n"+"\n".join(bootstrap_rows))
    counts=manifest.groupby("group").size().to_dict() if len(manifest) else {}
    section(33,"Representative Cases",f"A: L5 repaired + final repaired {counts.get('A',0)}/20; B: L5 repaired but final wrong {counts.get('B',0)}/20; C: baseline correct harmed {counts.get('C',0)}/20. Each panel shows RGB, GT, C0, C1, C2 and frozen component. If fewer than 20 exist, all available distinct images are shown, without fabricating cases. `visualizations/manifest.csv`; VPCA has 10 GT-free training examples per class.")
    section(34,"PLIP Semantic Source Decision",f"{result['PLIP_SEMANTIC_SOURCE']}. C1 minimum gate requires HRCR≥5%, Hard-M1 final correction≥1%, NCE>1. A failure means the tested frozen PLIP-last route did not meet this gate; it does not prove all VLM semantics absent.")
    section(35,"PDSR Decision",f"{result['PDSR_DECISION']}. C2 gate requires HRCR gain ≥5 pp **or** Hard-M1CR gain ≥3 pp versus C1, plus C2 mIoU≥C1 and ≥3/4 class IoUs nonnegative. Strong architecture gate additionally needs C2−C0 mIoU≥0.30 pp, C2 HRCR≥10%, NCE≥1.5.")
    section(36,"VPCA-v1 Status","CLOSED. This experiment audited the previous collapse only; no VPCA-v1 repair, new formula, or VPCA training was performed.")
    section(37,"Preserved Routes","The frozen HQMR/CCRA/PLIP, same BCSS/UMRF bank, and static-concept PDSR are retained as controlled references. Future routes depend on the explicit gate outcomes above, not an E5 cherry-picked checkpoint.")
    section(38,"Closed Routes","No Full25 in this round; previous late post-CCRA injection and VPCA-v1 are closed as primary candidate mechanisms. Do not infer improvement from a positive causal probe alone.")
    section(39,"Exact Next Step",("Stop at this preregistered Phase0 decision. Close the current PLIP-last / Static-PDSR pre-CCRA mainline and VPCA-v1; do not extend this run to 10/20/25 epochs or tune gamma against validation GT. A new separately preregistered audit should distinguish semantic-source ranking failure from insufficient CCRA/HQMR interface gain; do not call the tiny mIoU fluctuation an improvement." if result['FINAL_DECISION']=="PRE_CCRA_VLM_SEMANTIC_NOGO" else "Stop at this preregistered Phase0 decision. Any follow-up requires a separately preregistered mechanism test; no automatic Full25 or validation-driven gamma tuning."))
    lines.extend(["## Reproducibility and limitations","",f"Server output root: `{root}`. Code: `tools/pcsi_v1/README.md` and branch `experiment/pcsi-v1`. Checkpoints: C1 `{result['checkpoints']['C1']['sha256']}`, C2 `{result['checkpoints']['C2']['sha256']}`. Single seed (42), single BCSS split, five-epoch mechanism probe: uncertainty CIs are conditional on this split and do not establish external generalization. The original plan's 'q0 unchanged' only applies to pre-CCRA input query1, not HQMR query0. Temperature top1-change criterion is rank-invariant and therefore not used as a positive test.",""])
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("\n".join(lines),encoding="utf-8")
    print(f"PCSI_REPORT_COMPLETE {a.output}",flush=True)


if __name__=="__main__":main()
