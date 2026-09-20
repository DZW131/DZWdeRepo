"""Render the sealed 25-section Oracle action separability report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{100*x:.2f}%"


def number(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.4f}"


def table(rows, columns):
    frame=pd.DataFrame(rows)[columns]
    def cell(value):
        if pd.isna(value): return "—"
        if isinstance(value,(float,np.floating)): return f"{value:.4f}"
        return str(value).replace("|","\\|").replace("\n"," ")
    lines=["| "+" | ".join(columns)+" |",
           "|"+"|".join("---" for _ in columns)+"|"]
    for row in frame.itertuples(index=False,name=None):
        lines.append("| "+" | ".join(cell(v) for v in row)+" |")
    return "\n".join(lines)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    out=args.output
    anchor=json.loads((out/"00_reproduction_gate.json").read_text())
    manifest=json.loads((out/"feature_manifest.json").read_text())
    leakage=json.loads((out/"leakage_audit.json").read_text())
    ad=json.loads((out/"arbitration/decision.json").read_text())
    gd=json.loads((out/"gate/decision.json").read_text())
    a=pd.read_parquet(out/"arbitration/observable_features.parquet")
    al=pd.read_parquet(out/"arbitration/oracle_action_labels.parquet")
    g=pd.read_parquet(out/"gate/gate_off_pairs.parquet")
    gl=pd.read_parquet(out/"gate/rescue_oracle_labels.parquet")
    au=pd.read_csv(out/"arbitration/univariate_metrics.csv")
    au2=pd.read_csv(out/"arbitration/univariate_metrics_ap2.csv")
    gu=pd.read_csv(out/"gate/univariate_metrics.csv")
    ap1=pd.read_csv(out/"arbitration/linear_probe_cv.csv")
    ap2=pd.read_csv(out/"arbitration/linear_probe_cv_ap2.csv")
    apl_tree=pd.read_csv(out/"arbitration/shallow_probe_cv.csv")
    ap2_tree=pd.read_csv(out/"arbitration/shallow_probe_cv_ap2.csv")
    gp=pd.read_csv(out/"gate/linear_probe_cv.csv")
    gp_tree=pd.read_csv(out/"gate/shallow_probe_cv.csv")
    aw=pd.read_csv(out/"arbitration/weighted_metrics.csv")
    gw=pd.read_csv(out/"gate/weighted_metrics.csv")
    asub=pd.read_csv(out/"arbitration/subgroup_metrics.csv")
    gsub=pd.read_csv(out/"gate/subgroup_metrics.csv")
    ar=json.loads((out/"arbitration/gain_ranking.json").read_text())
    gr=json.loads((out/"gate/gain_ranking.json").read_text())
    a_dec=ad["decision"];g_dec=gd["decision"]
    a_go=a_dec in ("ARBITRATION_SEPARABLE","STRONG_ARBITRATION_GO")
    g_go=g_dec in ("GATE_RESCUE_SEPARABLE","STRONG_GATE_GO")
    if a_go and g_go: route="RACC-v2 unified design is eligible for separate pre-registration"
    elif a_go: route="Arbitration only; close Gate rescue"
    elif g_go: route="Gate false-exclusion risk only; close dynamic alpha"
    elif "WEAK" in (a_dec+g_dec): route="No new model training; weak/inconclusive action information"
    else: route="CLOSE Oracle-to-RACC translation route"
    a_prev=float(al.intervene.mean());up=float((al.action=="UP").mean());down=float((al.action=="DOWN").mean())
    g_prev=float(gl.beneficial_rescue.mean())
    ap1_final=ap1.iloc[-1];ap2_final=ap2.iloc[-1];gp_final=gp.iloc[-1]
    image_groups=a.image_id.map(lambda s:"-".join(str(s).split("_")[0].split("-")[:3])).nunique()
    g_groups=g.image_id.map(lambda s:"-".join(str(s).split("_")[0].split("-")[:3])).nunique()
    section=[]
    def add(title,body): section.append((title,body))
    add("Executive Decision",f"**ARBITRATION_DECISION = {a_dec}**  \n"
        f"**ARBITRATION_CONFIDENCE = {ad['confidence']}**  \n"
        f"**GATE_DECISION = {g_dec}**  \n"
        f"**GATE_CONFIDENCE = {gd['confidence']}**  \n"
        f"**FINAL_ROUTE = {route}**\n\n"
        f"A intervene prevalence = {pct(a_prev)}; UP = {pct(up)}; DOWN = {pct(down)}. "
        f"A best single-feature AUROC = {number(au.auroc.max())}; AP1/AP2 linear AUROC = {number(ap1_final.auroc)}/{number(ap2_final.auroc)}; "
        f"AP1/AP2 Recall@P80 = {pct(ap1_final.recall_p80)}/{pct(ap2_final.recall_p80)}; "
        f"AP1/AP2 AUPRC enrichment = {number(ap1_final.enrichment)}/{number(ap2_final.enrichment)}.\n\n"
        f"G beneficial-rescue prevalence = {pct(g_prev)}; G best single-feature AUROC = {number(gu.auroc.max())}; "
        f"G linear AUROC = {number(gp_final.auroc)}; Recall@P80/P90 = {pct(gp_final.recall_p80)}/{pct(gp_final.recall_p90)}; "
        f"AUPRC enrichment = {number(gp_final.enrichment)}.")
    add("Reproduction Gate",f"PASS={anchor['pass']}; six exact frozen anchors (mIoU): "
        f"HQMR {pct(anchor['observed']['hqmr'])}, DLAG A {pct(anchor['observed']['oracle_A1'])}, "
        f"B {pct(anchor['observed']['oracle_B1'])}, AB {pct(anchor['observed']['oracle_AB'])}, "
        f"RACC-A {pct(anchor['observed']['racc_A'])}, RACC-G {pct(anchor['observed']['racc_G'])}. "
        f"HQMR checkpoint SHA256 `{manifest['checkpoint_sha256']}`. Historical DLAG bank's alpha=1 baseline differs slightly from the independently archived HQMR anchor due to its sealed bank evaluation; the exact DLAG prediction bank is used for action labels, not silently conflated with the HQMR anchor. All source hashes in `00_reproduction_gate.json`.")
    add("Anti-Leakage Audit",f"Feature pass completed before GT/Oracle label reads; feature parquet hashes and baseline prediction hash sealed and rechecked. "
        f"Baseline prediction equals DLAG alpha=1 bank; M1 identity {int(al.m1_subset.sum())}. "
        f"IDs/patient grouping, GT, purity, Oracle gain and labels excluded from predictors. "
        f"Ten within-patient label permutations per AP1/AP2/G; any AUROC>0.55: {leakage['permutation_any_above_055']}. "
        f"Fold imputation/scaling fit only on train. Probe uses Oracle labels and is **information-content only, not a deployable weak-supervision model**.")
    add("Frozen Observable Manifest",f"{len(a)} baseline predicted components, {len(g)} deep-gate-OFF image-class pairs, 3418 BCSS validation images, "
        f"{image_groups}/{g_groups} patient groups (A/G). A features={len(manifest['features_A'])}; G features={len(manifest['features_G'])}; "
        f"fixed spatial support threshold={manifest['support_threshold']}. "
        f"Feature table SHA256: `{manifest['sha256']}`. Full names in `feature_manifest.json`.")
    add("Arbitration Oracle Action Definition","For every baseline predicted 8-connected component, compare correct pixels within its valid GT pixels across frozen alpha bank {0,.25,.5,1,1.5,2,3,4}; choose maximal gain using DLAG tie-order. Positive gain and alpha<1 = DOWN, alpha>1 = UP, otherwise KEEP. M1 is secondary only.")
    add("Arbitration Action Prevalence",f"KEEP {(al.action=='KEEP').sum()}, DOWN {(al.action=='DOWN').sum()}, UP {(al.action=='UP').sum()} among {len(al)}. "
        f"Component-weighted intervene {pct(a_prev)}, area-weighted {pct(aw.loc[aw.weighting=='area','prevalence'].iloc[0])}. "
        f"M1 count {int(al.m1_subset.sum())}; non-M1 {len(al)-int(al.m1_subset.sum())}.")
    add("Arbitration Univariate Separability",table(au.head(12),["feature","orientation","auroc","auprc","enrichment","recall_p80","recall_p90","precision_top1"]))
    add("Arbitration Probe Results",f"AP1 grouped 5-fold OOF, L2 logistic C=1 max_iter=2000 and depth-3 tree.\n\n"
        +table(pd.concat([ap1.tail(1).assign(probe="linear"),apl_tree.tail(1).assign(probe="tree")]),
               ["probe","ablation","n_features","auroc","auprc","enrichment","recall_p80","recall_p90","recall_p95"])
        +"\n\n"+table(aw,["weighting","prevalence","auroc","auprc","recall_p80"]))
    add("UP-vs-DOWN Separability",f"AP2 only among {int(al.intervene.sum())} actionable components; UP prevalence within actionable = {pct((al.loc[al.intervene==1,'action']=='UP').mean())}.\n\n"
        +table(pd.concat([ap2.tail(1).assign(probe="linear"),ap2_tree.tail(1).assign(probe="tree")]),
               ["probe","ablation","n_features","auroc","auprc","enrichment","recall_p80","recall_p90"])
        +"\n\nBest single features:\n\n"+table(au2.head(8),["feature","orientation","auroc","auprc","enrichment","recall_p80"]))
    add("Arbitration Gain Ranking",f"AP1 OOF Spearman(score, maximal correct-pixel gain)={number(ar['spearman'])}; top10% mean gain={number(ar['top10_mean_gain'])}, overall={number(ar['overall_mean_gain'])}, enrichment={number(ar['top10_gain_enrichment'])}. No action is inferred from GT at test time; gain is evaluation-only.")
    add("Arbitration Feature Ablation",f"Fixed cumulative A0 disagreement → A1 strength → A2 query concentration → A3 cross-stage class → A4 TTA → A5 geometry.\n\n"
        +table(ap1,["ablation","n_features","auroc","auprc","enrichment","recall_p80"])
        +"\n\nAP2:\n\n"+table(ap2,["ablation","n_features","auroc","auprc","enrichment","recall_p80"]))
    add("Arbitration Subgroups",table(asub,["subgroup","value","n","prevalence","auroc","auprc","recall_p80"]))
    add("Gate Rescue Oracle Definition","Universe is every gate-OFF (image,class) pair. Frozen HQMR re-inference forces on **only** that class, leaving other gates and CAM untouched. Primary label is positive iff whole-image valid-pixel correct count strictly increases. Class presence is secondary explanatory label only.")
    add("Gate Rescue Prevalence",f"Beneficial {int(gl.beneficial_rescue.sum())}/{len(gl)} = {pct(g_prev)}; class actually present {pct(gl.class_present.mean())}. "
        f"Among present classes, action beneficial {pct(gl.loc[gl.class_present==1,'beneficial_rescue'].mean())}; "
        f"pair vs benefit/harm-pixel weighted prevalence {pct(gw.iloc[0].prevalence)} / {pct(gw.iloc[1].prevalence)}.")
    add("Gate Univariate Separability",table(gu.head(12),["feature","orientation","auroc","auprc","enrichment","recall_p80","recall_p90","precision_top1"]))
    add("Gate Probe Results",table(pd.concat([gp.tail(1).assign(probe="linear"),gp_tree.tail(1).assign(probe="tree")]),
        ["probe","ablation","n_features","auroc","auprc","enrichment","recall_p80","recall_p90","recall_p95"])
        +"\n\n"+table(gw,["weighting","prevalence","auroc","auprc","recall_p80"]))
    add("Gate High-Precision Operating Points",table(gp.tail(1),["auroc","auprc","enrichment","recall_p80","recall_p90","recall_p95","precision_top1","precision_top5","precision_top10","precision_top20"]))
    add("Gate Gain Ranking",f"OOF Spearman(score, signed force-on net correct-pixel gain)={number(gr['spearman'])}; top10% mean gain={number(gr['top10_mean_gain'])}, overall={number(gr['overall_mean_gain'])}; signed overall<=0 makes ratio inapplicable ({number(gr['top10_gain_enrichment'])}).")
    add("Gate Feature Ablation","Fixed cumulative G0 deep gate → G1 TTA → G2 C3/C4 local support → G3 cross-stage contradiction → G4 query support.\n\n"
        +table(gp,["ablation","n_features","auroc","auprc","enrichment","recall_p80","recall_p90"]))
    add("Gate Subgroups",table(gsub,["subgroup","value","n","prevalence","auroc","auprc","recall_p80"]))
    add("RACC-v1 Failure Reinterpretation",f"Frozen RACC-A +0.0563 pp with nearly global alpha (mean≈1.258, SD≈0.051), RACC-G −4.6597 pp and rescue precision 6.37%. "
        f"This audit distinguishes weak original formulation from absent separable information. AP1/AP2/G OOF AUROCs are {number(ap1_final.auroc)}/{number(ap2_final.auroc)}/{number(gp_final.auroc)}; compare fixed ablation trajectories, not a post-hoc feature search.")
    add("2×2 Decision Matrix",f"A={a_dec}; G={g_dec}; route={route}. Thresholds applied to **linear OOF**: A both AUROC≥.80, Recall@P80≥20%, AP enrichment≥2; G AUROC≥.88, enrichment≥3, Recall@P80≥20%, @P90≥10%. NOGO if A either AUROC<.70 or @P80<10%; G AUROC<.80 or @P80<10%. Weak is not training authorization.")
    add("What Is Preserved","CCRA, HQMR E25, BCSS Seed42, UCRF mechanism, DLAG counterfactual/oracle results, all sealed checksums and GT-free tables remain preserved. No segmentation parameter updates and no new Full25/checkpoint.")
    add("What Is Closed",("Gate rescue" if not g_go else "No Gate route closed")+"; "+("dynamic-alpha learned arbitration" if not a_go else "No Arbitration route closed")+" under the predeclared operating requirements. A high Oracle ceiling alone is not evidence of GT-free action identifiability.")
    add("Exact Next Architecture Step",("Design the smallest action reliability target from the first ablation family that materially raises OOF precision; pre-register a separate weak-supervision mechanism before training." if a_go or g_go else "Do not train RACC-v2 or tune alpha/gate thresholds. Archive Oracle-to-RACC translation and investigate a different implementable mechanism.")
        +" The present Oracle-supervised probes are diagnostic only, not deployable models. Independent seeds would be required for any future generalization claim.")
    assert len(section)==25
    title="# Oracle Action GT-Free Separability Audit — BCSS Seed42\n\n"
    report=title+"\n\n".join(f"## {i}. {name}\n\n{body}" for i,(name,body) in enumerate(section,1))
    report+="\n\n## Reproduction and artifact paths\n\n"
    report+="- Re-run: `python audits/oracle_action_separability_v1/reproduction_gate.py ...`, then `freeze_observables.py`, `build_labels.py`, `analyze.py`, `generate_report.py` in that order.\n"
    report+="- Server output: `/home/duyanhong/experiments/Oracle_Action_Separability_v1_BCSS_Seed42`.\n"
    report+="- Source: `audits/oracle_action_separability_v1/` on branch `audit/oracle-action-separability-v1`.\n"
    path=out/"Oracle_Action_GTFree_Separability_Audit_Report.md"
    path.write_text(report,encoding="utf-8")
    print(json.dumps({"report":str(path),"A":a_dec,"G":g_dec,"route":route,"sections":len(section)}),flush=True)


if __name__=="__main__":main()
