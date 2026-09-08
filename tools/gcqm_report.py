"""Render the preregistered GCQM Phase-0 Markdown report."""
from pathlib import Path
import json


def _r(s,k,stage=3): return next((x for x in s.get(k,[]) if x.get("stage")==stage),{})
def _f(x): return f"{x:.6f}" if isinstance(x,(int,float)) else str(x)


def render_report(output,payload):
    output=Path(output); s=payload.get("final_summary",{}); gate=payload.get("gate",{}); decision=payload.get("decision","GCQM_ENGINEERING_BLOCKED")
    sem=_r(s,"primary_semantic_health"); sens=_r(s,"query_identity_sensitivity"); gain=_r(s,"vs_pca_gain"); ni=_r(s,"vs_pixel_noninferiority"); wc=_r(s,"weight_class_conditioning"); wu=_r(s,"weight_utilization"); b=_r(s,"B_basis_health"); con=_r(s,"weight_conservation"); ccra=_r(s,"responsibility_complementarity")
    gates="\n".join(f"| {k} | {'PASS' if v else 'FAIL'} |" for k,v in gate.get("gate_groups",{}).items()) or "| unavailable | FAIL |"
    trajectory=[]
    for item in payload.get("summary_history",[]):
        a=_r(item,"primary_semantic_health"); d=_r(item,"query_identity_sensitivity")
        if a: trajectory.append(f"| {a['snapshot']} | {_f(a.get('CPR'))} | {_f(a.get('positive_recall'))} | {_f(a.get('probability_gap'))} | {_f(d.get('D_perm_mean'))} |")
    text=f"""# GCQM Phase-0 Global Class-Conditioned Query Mixture 实验报告

## 1 Executive Decision

最终结论：**{decision}**。

## 2 FOMD evidence that motivates simplification

FOMD 的 pixel-wise ownership 无增益且 B locality 失败，但 query identity、CCRA-vs-PCA 与最终 mask 健康，因此检验更小的全局类条件 query mixture。

## 3 Removed claims

不再声称 pixel-wise ownership 必要，也不再声称 B 是 localized 或 class-specific expert。

## 4 GCQM hypothesis

`w(i,c)=normalize_q(mean_x A(i,x,c))`，`F(c,x)=Σ_i w(i,c)B(i,x)`。

## 5 Frozen architecture

沿用 FOMD lineage `{payload.get('source_commit','unknown')}`；ResNet38、196 queries、CCRA、CHPF、PCA/deep、tri-state 和优化协议保持冻结。

## 6 CCRA spatial evidence

E5 responsibility IoU `{_f(ccra.get('responsibility_iou_median'))}`，distinct peaks `{_f(ccra.get('distinct_peak_fraction'))}`。

## 7 Global query pooling

A 仅作为权重估计中间量，空间平均后沿 query 轴重新归一。

## 8 w class-conditioned allocation

JS median `{_f(wc.get('JS_median'))}`；top1 class difference `{_f(wc.get('top1_difference'))}`。

## 9 Query mask bases B

B 仍为原始 query mask logit 的 sigmoid，只要求非退化。

## 10 GCQM mixture F

Stage2/3 的唯一 primary mask 与 mask loss 输入为 `ΣwB`。

## 11 Loss contract

总损失保持 deep/PCA/mask = 0.50/0.25/0.25，新增 auxiliary loss 为 0。

## 12 Zero parameter delta

GCQM 参数增量 `{payload.get('gcqm_delta_parameters','unknown')}`。

## 13 Historical compatibility audit

正式训练前的 archived FOMD endpoint 审计结果为 `{payload.get('fomd_compatibility',{}).get('decision','unknown')}`，且没有修改预注册门槛。

## 14 Weight conservation

max weight-sum error `{_f(con.get('max_weight_sum_error'))}`；max C/F error `{_f(con.get('max_contribution_sum_error'))}`。

## 15 Weight class-conditioning

详见 `gcqm_weight_class_conditioning.csv`。

## 16 Weight utilization

dominant share `{_f(wu.get('dominant_share'))}`；effective queries `{_f(wu.get('effective_queries'))}`；top5 mass `{_f(wu.get('top5_mass'))}`。

## 17 Primary class-mask health

| Metric | E5 |
| --- | ---: |
| CPR | {_f(sem.get('CPR'))} |
| Positive recall | {_f(sem.get('positive_recall'))} |
| Positive F median | {_f(sem.get('positive_F_median'))} |
| Rival leakage | {_f(sem.get('rival_leakage'))} |
| Background leakage | {_f(sem.get('background_leakage'))} |
| Probability gap | {_f(sem.get('probability_gap'))} |

## 18 Query-identity sensitivity

D_perm mean/std/min/max = `{_f(sens.get('D_perm_mean'))}` / `{_f(sens.get('D_perm_std'))}` / `{_f(sens.get('D_perm_min'))}` / `{_f(sens.get('D_perm_max'))}`。

## 19 Full-vs-permutation

8 个固定 derangements 的逐项结果保存在 `gcqm_perm_semantic_health.csv`。

## 20 GCQM-vs-PCA

gap gain `{_f(gain.get('gap_gain'))}`；rival reduction `{_f(gain.get('rival_reduction'))}`；confidence gain `{_f(gain.get('confidence_gain'))}`。

## 21 GCQM-vs-pixelwise noninferiority

gap/recall/confidence delta = `{_f(ni.get('gap_delta'))}` / `{_f(ni.get('recall_delta'))}` / `{_f(ni.get('confidence_delta'))}`。

## 22 B basis health

empty fraction `{_f(b.get('empty_fraction'))}`；area>90% fraction `{_f(b.get('fraction_area_gt_090'))}`；component p90 `{_f(b.get('component_p90'))}`。

## 23 Contribution utilization

权重×basis 空间质量的贡献分布保存在 `gcqm_contribution_utilization.csv`。

## 24 CCRA health

CCRA normalization、complementarity 与 utilization 均保留在 summary/gate artifacts。

## 25 PCA health

PCA present coverage 与 absent dominance 见 `gcqm_pca_health.csv`。

## 26 Deep gate health

Deep present/absent 及 gap 见 `gcqm_deep_gate_health.csv`。

## 27 Query update health

Stage2/3 update norm、delta 与 cosine 见 `gcqm_query_update_health.csv`。

## 28 Gradient/detach audit

w 与 pixel/PCA/perm/uniform references 均 detached；只有 GCQM primary 通过 B/mask pathway 反向传播。

## 29 Visual evidence

固定 train cohort 可视化保存在 `visualizations/`。

## 30 E2 catastrophic screen

`{json.dumps(payload.get('epoch2_screen',{}),ensure_ascii=False)}`

## 31 E5 Gate A-J

| Gate | Result |
| --- | --- |
{gates}

## 32 STRONG GO

Strong GO：`{gate.get('strong_go',False)}`。

## 33 Scientific Interpretation

GCQM 只在最终掩码健康、query identity 有因果贡献、明显优于 PCA、且不劣于 pixel-wise FOMD 时成立；失败 gate 精确限定该最小模型的不足。

## 34 Exact Decision

`{decision}`。

## 35 Full25 Recommendation

只有 GO/STRONG_GO 才冻结模型并进入 fresh BCSS Seed42 Full25；本任务在 Phase-0 后停止。

### Snapshot trajectory

| Snapshot | CPR | Recall | Gap | D_perm |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(trajectory)}

DECISION = {decision}
"""
    path=output/"GCQM_Phase0_Global_Class_Conditioned_Query_Mixture_Report.md"; path.write_text(text,encoding="utf-8"); return path


__all__=["render_report"]
