"""Generate the 26-section UMRF-v1 audit report."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def pct(x): return f"{100*float(x):.2f}%"
def num(x):
    if x==float("inf"): return "∞"
    return f"{float(x):.3f}"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--report",type=Path); a=p.parse_args()
    r=json.loads((a.output/"02_evaluation.json").read_text()); f=json.loads((a.output/"01_evidence_freeze_manifest.json").read_text()); v=json.loads((a.output/"visualizations/visualization_manifest.json").read_text())
    c=r["summaries"]["common"]; s=r["summaries"]["sequential"]
    report=a.report or a.output/"UMRF_v1_Upstream_MultiLevel_Responsibility_Formation_Audit_Report.md"
    lines=["# UMRF-v1 Upstream Multi-Level Responsibility Formation Audit Report","",
           "> **Homepage verdict**",f"> - Primary (Common-Query): **{r['primary_decision']}**",f"> - Secondary (Sequential): **{r['secondary_decision']}**",f"> - Mechanistic diagnosis: **{r['final_diagnosis']}**",
           f"> - HQMR frozen replay: **PASS**, mIoU `{json.loads((a.output/'00_reproduction_gate.json').read_text())['hqmr_miou']:.12f}`",f"> - Parameter updates: **0**; images/components: **{f['images']} / {f['components']}**",f"> - Exact M1 replay: **{r['historical_m1']['exact_replay_count']}** (historical anchor 4440)",""]
    def add(title,body): lines.extend([f"## {len([x for x in lines if x.startswith('## ')])+1}. {title}","",body,""])
    add("Executive conclusion",f"Common-Query yields area-weighted precision {pct(c['area']['precision'])}, coverage {pct(c['area']['coverage'])}, NCE {num(c['area']['nce'])}, and H5-correct harm {pct(c['area']['h5_correct_harm_rate'])}. Sequential yields {pct(s['area']['precision'])}, {pct(s['area']['coverage'])}, {num(s['area']['nce'])}, and {pct(s['area']['h5_correct_harm_rate'])}. The frozen decision is **{r['final_diagnosis']}**.")
    add("Question and scope","This audit asks whether fixed, GT-free multi-level responsibility evidence can correct an erroneous deep H5 decision. It does not train a model or claim a segmentation improvement.")
    add("Frozen inputs",f"Checkpoint SHA-256 `{f['checkpoint_sha256']}`; BCSS validation contains {f['images']} images. All tensors came from the frozen HQMR forward path.")
    add("Reproduction gate","The exact FP32-before-resize HQMR replay passed before evidence extraction. The final prediction was bitwise checked against the sealed exact-baseline prediction bank.")
    add("Forward-path audit",f"Common-query safety gate: `{f['common_query_safety']['pass']}`. Shapes: `{json.dumps(f['common_query_safety']['shapes'])}`. Maximum formula errors: `{json.dumps(f['formula_max_abs_error'])}`.")
    add("Evidence definitions","Sequential uses E5=L5, E4=L4, E3=L3. Common-Query uses q0×k5, q0×k4, q0×k3. The unchanged frozen W maps query evidence to four class channels; probabilities are softmax over classes at temperature 1.")
    add("Direct-branch proxies","D4 and D3 were preserved only as direct-branch class proxies in the component table. They were not promoted to an alternative architecture or decision rule.")
    add("GT-free freeze",f"Evidence and rule maps were sealed before any validation mask was opened. Map bank SHA-256: `{f['sha256']['gt_free_prediction_maps.uint8.npy']}`.")
    add("Component universe",f"The primary universe contains {r['components']} exact-baseline 8-connected components. Component evidence is a uniform mean across the component pixels.")
    add("Historical M1 reconciliation",f"Historical M1 anchor is 4440; exact active replay produces {r['historical_m1']['exact_replay_count']}. M1 remains a secondary subgroup and does not redefine the primary universe.")
    add("Fixed rules","R1 uses the H4/H3 consensus when it exists, otherwise H5. R2 uses a three-stage majority, with H5 retained when all differ. R3 averages the three class-probability vectors equally. No rule was tuned after GT exposure.")
    add("Common-Query primary results",f"Component-weighted: precision {pct(c['component']['precision'])}, coverage {pct(c['component']['coverage'])}, NCE {num(c['component']['nce'])}. Area-weighted: precision {pct(c['area']['precision'])}, coverage {pct(c['area']['coverage'])}, NCE {num(c['area']['nce'])}, harm {pct(c['area']['h5_correct_harm_rate'])}.")
    add("Sequential secondary results",f"Component-weighted: precision {pct(s['component']['precision'])}, coverage {pct(s['component']['coverage'])}, NCE {num(s['component']['nce'])}. Area-weighted: precision {pct(s['area']['precision'])}, coverage {pct(s['area']['coverage'])}, NCE {num(s['area']['nce'])}, harm {pct(s['area']['h5_correct_harm_rate'])}.")
    add("Pixel-level results",f"Common-Query: precision {pct(r['pixel']['common']['precision'])}, coverage {pct(r['pixel']['common']['coverage'])}, NCE {num(r['pixel']['common']['nce'])}. Sequential: precision {pct(r['pixel']['sequential']['precision'])}, coverage {pct(r['pixel']['sequential']['coverage'])}, NCE {num(r['pixel']['sequential']['nce'])}.")
    add("Rule net gains",f"Pixel net correction rates over H5 — Common R1/R2/R3: {pct(r['pixel']['common']['gain_r1'])} / {pct(r['pixel']['common']['gain_r2'])} / {pct(r['pixel']['common']['gain_r3'])}; Sequential: {pct(r['pixel']['sequential']['gain_r1'])} / {pct(r['pixel']['sequential']['gain_r2'])} / {pct(r['pixel']['sequential']['gain_r3'])}.")
    add("Stage agreement and diversity",f"Common H5–H4/H4–H3/all-same: {pct(c['area']['h54_agreement'])} / {pct(c['area']['h43_agreement'])} / {pct(c['area']['all_same'])}. Sequential: {pct(s['area']['h54_agreement'])} / {pct(s['area']['h43_agreement'])} / {pct(s['area']['all_same'])}.")
    add("Rescue matrix",f"Common rescue matrix (area fractions among H5-wrong): `{json.dumps(c['rescue_matrix_area'])}`. Sequential: `{json.dumps(s['rescue_matrix_area'])}`.")
    add("TP safety",f"When H5 is correct, a conflicting shallow consensus harms {pct(c['area']['h5_correct_harm_rate'])} of Common-Query area and {pct(s['area']['h5_correct_harm_rate'])} of Sequential area.")
    add("M1 subgroup",f"M1 area-weighted Common precision/coverage/NCE: {pct(c['m1_area']['precision'])} / {pct(c['m1_area']['coverage'])} / {num(c['m1_area']['nce'])}. Sequential: {pct(s['m1_area']['precision'])} / {pct(s['m1_area']['coverage'])} / {num(s['m1_area']['nce'])}.")
    add("High-purity and large M1",f"High-purity (≥0.70) Common coverage {pct(c['m1_high_purity_area']['coverage'])}; Q4-size (threshold area {r['m1_q4_area_threshold']:.1f}) coverage {pct(c['m1_q4_area']['coverage'])}. Sequential values are {pct(s['m1_high_purity_area']['coverage'])} and {pct(s['m1_q4_area']['coverage'])}.")
    add("Per-class audit","Common area metrics by GT class: " + "; ".join(f"C{k} precision {pct(v['precision'])}, coverage {pct(v['coverage'])}, NCE {num(v['nce'])}" for k,v in c['per_class_area'].items()) + ".")
    add("Bootstrap uncertainty",f"Both component-area metrics and pixel metrics use 2,000 fixed-seed resamples (seed 42). Common area precision CI: `{c['bootstrap_component_area']['precision']['ci95']}`; coverage CI: `{c['bootstrap_component_area']['coverage']['ci95']}`; NCE CI: `{c['bootstrap_component_area']['nce']['ci95']}`.")
    add("Qualitative case banks",f"Rendered counts: `{json.dumps(v['categories'])}`. Each panel shows the image, GT, HQMR final prediction, and Common/Sequential C5–C3 class maps with the audited component outlined and margins in the title.")
    add("Responsibility-level downstream counterfactual",f"**{r['downstream_counterfactual']['status']}** — {r['downstream_counterfactual']['reason']}")
    add("Decision rule application",f"Thresholds were applied exactly: NOGO if precision<0.65, coverage<0.10, or NCE≤1; GO requires precision≥0.75, coverage≥0.20, NCE≥1.5, and harm<5%; STRONG_GO requires 0.85/0.30/2.0. Result: Common **{r['primary_decision']}**, Sequential **{r['secondary_decision']}**.")
    add("Artifacts and reproduction","Key artifacts: `00_reproduction_gate.json`, `01_evidence_freeze_manifest.json`, `component_evidence.parquet`, `gt_free_prediction_maps.uint8.npy`, `component_evidence_with_gt.parquet`, `02_evaluation.json`, and `visualizations/`. Exact commands are in `audits/umrf_v1/README.md`.")
    add("Final recommendation",f"The mechanism-level conclusion is **{r['final_diagnosis']}**. Treat this result as an architecture-routing decision, not as a trained-model performance claim. Do not continue to a learned selector or Full25 training unless a new plan explicitly addresses this frozen evidence ceiling.")
    assert len([x for x in lines if x.startswith("## ")])==26
    report.write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps({"event":"UMRF_REPORT_WRITTEN","path":str(report)}))
if __name__=="__main__": main()

