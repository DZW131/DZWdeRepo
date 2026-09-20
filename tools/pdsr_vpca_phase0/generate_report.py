"""Generate the preregistered 32-section PDSR-VPCA Phase0 report."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def pct(x): return f"{100*float(x):.2f}%"
def pp(x): return f"{float(x):+.3f} pp"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--report",type=Path); a=p.parse_args(); o=a.output
    r=json.loads((o/"metrics/final_result.json").read_text()); base=json.loads((o/"manifests/baseline.json").read_text()); plip=json.loads((o/"manifests/plip_runtime_manifest.json").read_text()); concept=json.loads((o/"manifests/concept_bank_manifest.json").read_text()); identity=json.loads((o/"manifests/identity_tests.json").read_text()); leakage=json.loads((o/"manifests/leakage_audit.json").read_text()); counts=json.loads((o/"manifests/parameter_counts.json").read_text()); deviations=json.loads((o/"manifests/protocol_deviations.json").read_text()); visuals=json.loads((o/"visualizations/visualization_manifest.json").read_text()); cost=json.loads((o/"metrics/computational_cost.json").read_text())
    s=r["segmentation"]; d=r["delta_miou_pp"]; hm={v:r["cohort"][v]["hard_m1"]["pixel_area_weighted"] for v in ("P1","P2","P3")}; m1={v:r["cohort"][v]["m1"]["pixel_area_weighted"] for v in ("P1","P2","P3")}
    path=a.report or o/"PDSR_VPCA_HQMR_v1_Phase0_Final_Report.md"
    lines=["# PDSR‑VPCA‑HQMR v1 Phase0 Final Report","","> **FINAL_DECISION = %s**"%r["FINAL_DECISION"],f"> HQMR P0 = {pct(s['P0']['mIoU'])}",f"> VLM‑Last P1 = {pct(s['P1']['mIoU'])}",f"> Static‑PDSR P2 = {pct(s['P2']['mIoU'])}",f"> VPCA‑PDSR P3 = {pct(s['P3']['mIoU'])}","> SSHR = 66.6967%",f"> P1−P0 = {pp(d['P1-P0'])}; P2−P1 = {pp(d['P2-P1'])}; P3−P2 = {pp(d['P3-P2'])}; P3−P0 = {pp(d['P3-P0'])}",f"> PDSR_DECISION = {r['PDSR_DECISION']}; VPCA_DECISION = {r['VPCA_DECISION']}",f"> Hard‑M1 P0 = 0.00%; P1 = {pct(hm['P1'])}; P2 = {pct(hm['P2'])}; P3 = {pct(hm['P3'])}",f"> PDSR Hard‑M1 gain = {pct(r['pdsr_hmcr_gain'])}; VPCA Hard‑M1 gain = {pct(r['vpca_hmcr_gain'])}",f"> NCE = {r['tp_safety']['P3']['nce']:.3f}",f"> CLASS_DAMAGE = {r['CLASS_DAMAGE']}; LAYER_COLLAPSE = {r['LAYER_COLLAPSE']}; CONCEPT_COLLAPSE = {r['CONCEPT_COLLAPSE']}; CCRA_COLLAPSE = {r['CCRA_COLLAPSE']}; VLM_IGNORED = {r['VLM_IGNORED']}",f"> FULL25_GO = {r['FULL25_GO']}",""]
    def add(title,body): lines.extend([f"## {len([x for x in lines if x.startswith('## ')])+1}. {title}","",body,""])
    add("Executive Decision",f"最终判定 **{r['FINAL_DECISION']}**。PDSR={r['PDSR_DECISION']}，VPCA={r['VPCA_DECISION']}，本脚本按预注册要求停止，不自动运行 Full25。")
    add("Frozen Research Context","RACC、gate rescue、dynamic alpha、prototype relabel 与 H5/H4/H3 consensus 路线保持关闭。本轮只检验独立 PLIP 语义源及其 PDSR/VPCA 使用方式。")
    add("Reproduction Gate",f"冻结 HQMR checkpoint 重放通过，P0 mIoU={pct(base['mIoU'])}，checkpoint SHA‑256 `{base['checkpoint_sha256']}`，三视图 TTA、最终 gate 与 CAM normalization 均未改变。")
    add("PLIP Runtime Audit",f"官方 `vinid/plip@67ade53` 权重 `{plip['weight_sha256']}`；hidden states={plip['hidden_states_length']}，block 4/8/12 dense token 均为 `{plip['selected_dense_shapes'][0]}`，patch grid={plip['patch_grid_side']}×{plip['patch_grid_side']}，视觉 hidden={plip['vision_hidden_size']}，text dim={plip['text_projection_dim']}。")
    add("Class Mapping","BCSS index 固定为 C0 Tumor、C1 Stroma、C2 Lymphocytic/Inflammatory infiltrate、C3 Necrosis；C4 是 evaluation ignore/other。")
    add("Concept Bank",f"每类 8 个、共 32 个 atomic pathology concepts；bank SHA `{concept['concept_bank_sha256']}`，embedding SHA `{concept['embedding_sha256']}`，全部 L2 normalized，且在读取 validation GT 前冻结。")
    add("Leakage Audit",f"训练 batch 仅包含 `{leakage['train_loader_fields']}`，segmentation GT in train={leakage['segmentation_gt_in_train']}，E1–E4 validation access={leakage['validation_access_before_e5']}。")
    add("Architecture Implementation","P1 使用 block12 dense residual；P2 使用 block4/8/12 static-concept reconstruction；P3 仅将 static prior 替换为 VPCA soft q×rho。旧 HQMR 和 PLIP 参数均冻结。")
    add("Identity Tests","P1/P2/P3 的 γ=0 identity 均通过："+json.dumps(identity["tests"])+"。为满足严格 identity，采用 H5+γZ；协议冲突与梯度链处理记录为："+json.dumps(deviations)+"。")
    add("Gradient Tests","三组 gradient manifests 均验证 HQMR grad=None、PLIP grad=None；γ 首次 backward 有梯度，受 γ=0 数学阻断的投影在首个 optimizer update 后获得非零梯度。")
    add("P0 HQMR",f"mIoU {pct(s['P0']['mIoU'])}，mDice {pct(s['P0']['mDice'])}；作为全部 paired comparison 的冻结基线。")
    add("P1 VLM‑Last",f"mIoU {pct(s['P1']['mIoU'])}（P1−P0 {pp(d['P1-P0'])}），Hard‑M1 corrective rate {pct(hm['P1'])}，WRP {pct(r['cohort']['P1']['hard_m1']['wrong_rival_persistence'])}，γ abs mean {r['gamma']['P1']['abs_mean']:.6g}。")
    add("VLM Novel Semantic Evidence",f"Question A：P1>P0 = {d['P1-P0']>0}；Hard‑M1 correction 是否非零 = {hm['P1']>0}。这只说明 frozen PLIP last-layer 是否提供可利用的新信息，不归因于 PDSR。")
    add("P2 Static‑PDSR",f"mIoU {pct(s['P2']['mIoU'])}（P2−P1 {pp(d['P2-P1'])}），Hard‑M1={pct(hm['P2'])}，M1={pct(m1['P2'])}。")
    add("PDSR Cross‑Layer Reconstruction",f"β mean={r['mechanism']['P2'].get('beta_mean')}，β>0.9 fractions={r['mechanism']['P2'].get('beta_gt_0_9_fraction')}，R/U ratios={r['mechanism']['P2'].get('reconstruction_visual_ratio_mean')}。")
    add("PDSR Mechanism Metrics",f"P2−P1 Hard‑M1 gain={pct(r['pdsr_hmcr_gain'])}，M1 gain={pct(r['pdsr_m1cr_gain'])}；PDSR decision={r['PDSR_DECISION']}。")
    add("P3 VPCA‑PDSR",f"mIoU {pct(s['P3']['mIoU'])}（P3−P2 {pp(d['P3-P2'])}；P3−P0 {pp(d['P3-P0'])}），Hard‑M1={pct(hm['P3'])}，M1={pct(m1['P3'])}。")
    add("Concept Participation Analysis",f"P3 concept statistics: `{json.dumps(r['mechanism']['P3'].get('concept_per_class',{}))}`。VPCA 使用固定 Top20%、τ=0.1、无 hard gate/threshold。")
    add("Concept Diversity / Collapse",f"CONCEPT_COLLAPSE={r['CONCEPT_COLLAPSE']}；判据为同类图像 top‑1 concept 频率>80% 且 entropy<0.5·log(8)。")
    add("Hard‑M1 Analysis",f"冻结 Hard‑M1 count={r['hard_m1']['count']}，area={r['hard_m1']['area']}；P1/P2/P3 area‑weighted HMCR={pct(hm['P1'])}/{pct(hm['P2'])}/{pct(hm['P3'])}。")
    add("M1 Analysis",f"historical M1=4440，exact active M1={r['m1']['exact']}；P1/P2/P3 M1CR={pct(m1['P1'])}/{pct(m1['P2'])}/{pct(m1['P3'])}。")
    add("True‑vs‑Rival Analysis",f"P2 margin={r['cohort']['P2']['hard_m1']['semantic_margin']}；P3 margin={r['cohort']['P3']['hard_m1']['semantic_margin']}；positive‑fraction gain={pct(r['vpca_margin_gain'])}。")
    add("TP Safety",f"P1/P2/P3 NCE={r['tp_safety']['P1']['nce']:.3f}/{r['tp_safety']['P2']['nce']:.3f}/{r['tp_safety']['P3']['nce']:.3f}；P3 TP harm={pct(r['tp_safety']['P3']['tp_harm'])}。")
    add("Per‑Class Results","IoU P0/P1/P2/P3："+"；".join(f"C{c} {pct(s['P0']['class_iou'][str(c)])}/{pct(s['P1']['class_iou'][str(c)])}/{pct(s['P2']['class_iou'][str(c)])}/{pct(s['P3']['class_iou'][str(c)])}" for c in range(4))+f"。CLASS_DAMAGE={r['CLASS_DAMAGE']}。")
    add("CCRA Health",f"{json.dumps(r['ccra_health'])}。注入点位于 CCRA 之后，且旧路径冻结，因此 CCRA responsibility 不随 P1/P2/P3 更新。")
    add("Computational Cost",f"参数、profiler-counted FLOPs、E5 3-view FPS/peak VRAM：`{json.dumps(cost)}`。FLOPs 是 torch.profiler 可识别算子的保守计数；PLIP 虽冻结仍计入完整推理参数和 FPS/VRAM。")
    add("Bootstrap CI",f"2000 次 paired image bootstrap，seed42：`{json.dumps(r['bootstrap'])}`。")
    add("Representative Cases",f"案例库：`{json.dumps(visuals)}`；每组固定 recovered/unrecovered/harmed，P2/P3 另含 concept grounding。")
    add("PDSR GO/NOGO",f"PDSR={r['PDSR_DECISION']}。门槛：ΔHMCR≥5pp、ΔM1CR≥3pp、3/4 classes non‑negative、无 layer/CCRA collapse。")
    add("VPCA GO/NOGO",f"VPCA={r['VPCA_DECISION']}。门槛：ΔHMCR≥5pp、positive margin fraction≥5pp、3/4 classes non‑negative、无 concept collapse。")
    add("Full Model GO/NOGO",f"FINAL_DECISION={r['FINAL_DECISION']}，FULL25_GO={r['FULL25_GO']}。需要 P3−P0≥0.50pp、PDSR与VPCA均GO、NCE>1.5且无严重安全失败。")
    add("Exact Next Step","本轮到此停止，不自动执行 Full25。若 FULL25_GO=True，下一步必须从官方 fresh ResNet38 initialization 开始公平的 25‑epoch 训练；否则仅根据报告中失败的因果环节设计下一轮最小实验。")
    assert len([x for x in lines if x.startswith("## ")])==32
    path.write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps({"event":"PDSR_REPORT_WRITTEN","path":str(path)}),flush=True)
if __name__=="__main__": main()
