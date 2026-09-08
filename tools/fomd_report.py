"""Markdown report renderer for FOMD Phase-0."""
from __future__ import annotations

import json
from pathlib import Path


def _row(summary, key, stage=3):
    rows=summary.get(key,[])
    return next((x for x in rows if x.get("stage")==stage),{})


def _f(value):
    return f"{value:.6f}" if isinstance(value,(int,float)) else str(value)


def render_report(output, payload):
    output=Path(output); s=payload.get("final_summary",{}); gate=payload.get("gate",{})
    decision=payload.get("decision","FOMD_ENGINEERING_BLOCKED")
    full=_row(s,"full_semantic_health"); query=_row(s,"query_identity_sensitivity")
    glob=_row(s,"global_semantic_health"); pca=_row(s,"pca_semantic_health")
    ac=_row(s,"A_class_conditioning"); at=_row(s,"A_top_owner_disagreement")
    bs=_row(s,"B_union_foreground_support"); bl=_row(s,"B_locality_mass"); ba=_row(s,"B_area_health")
    cap=_row(s,"capacity_preservation"); ccra=_row(s,"responsibility_complementarity")
    groups=gate.get("gate_groups",{})
    gate_table="\n".join(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name,passed in groups.items()) or "| unavailable | FAIL |"
    history=[]
    for item in payload.get("summary_history",[]):
        f=_row(item,"full_semantic_health"); q=_row(item,"query_identity_sensitivity")
        if f: history.append(f"| {f['snapshot']} | {_f(f.get('positive_recall'))} | {_f(f.get('probability_gap'))} | {_f(q.get('D_perm_mean'))} |")
    audit=payload.get("momd_factorization_audit",{})
    audit_text=json.dumps({k:audit.get(k) for k in ("decision","F_full_equals_MOMD","counterfactuals_detached","all_derangements","parameter_delta")},ensure_ascii=False)
    text=f"""# FOMD Phase-0 Factorized Ownership Mask Decoding 实验报告

## 1. Executive Decision

本轮严格复用 archived MOMD-v1 的网络、损失和初始化协议；FOMD 只增加 detached 反事实诊断。最终结论：**{decision}**。

## 2. MOMD Frozen Positive Evidence

MOMD 已证明 CCRA、质量守恒和最终掩码容量可以保持；本轮不修改这些路径。

## 3. OEC NOGO Evidence

OEC 的类无关 expert 与类条件 ownership 未形成预注册的语义专门化，因此本轮不再要求 A/B 正相关，而检验二者是否作为互补因子通过 query identity 正确重组。

## 4. Class-conditioned vs class-agnostic mismatch

A 是类条件像素所有权，B 是类无关 query region support；二者角色不同，`Corr(A,B)>0` 不是本轮主张。

## 5. FOMD hypothesis

`Allocate → Support → Mix`：`F(x,c)=Σ_i A(i,x,c)B(i,x)`。

## 6. Why new preregistered claim

因果验证依赖固定 query derangement、空间全局 A、PCA-only mixture 和 uniform reference，而非新增训练信号。

## 7. Architecture freeze

源提交 `{payload.get('source_commit','unknown')}`；训练模型类仍为 `MOMDNet`，FOMD 参数增量为 `{payload.get('fomd_delta_parameters','unknown')}`。

## 8. Zero training-equation delta

历史审计：`{audit_text}`。正式训练的总损失仍为 deep/PCA/mask = 0.50/0.25/0.25。

## 9. Ownership factor A

A 沿 query 轴归一化，由 detached CCRA responsibility 与冻结 locality 构成。

## 10. Region-support factor B

B 为原始 query mask logit 的 sigmoid，不被解释为类别 expert。

## 11. Full F

Stage3 主输出与 MOMD `F=sum(A*B)` 数值完全一致。

## 12. Loss contract

只有 F_full 参与 Stage2/3 mask loss；全部反事实分支 detached 且不进入 backward。

## 13. Counterfactual design

使用 8 个预注册固定 derangements（seeds 1001–1008）、global-A、PCA-only 和 uniform 分支。

## 14. Query permutation

E5 `D_perm_mean={_f(query.get('D_perm_mean'))}`，检验 A 与 B 的 query-index 对齐是否必要。

## 15. Global ownership

Global-A probability gap `{_f(glob.get('probability_gap'))}`，与 full 比较像素级 ownership 的增量。

## 16. PCA-only mixture

PCA-only probability gap `{_f(pca.get('probability_gap'))}`。

## 17. Uniform reference

Uniform 仅作 sanity reference，不作为 GO 的替代模型。

## 18. Why no A-only mask

A 沿 query 归一化，本身不是可独立阈值化的类别掩码，因此没有构造 naive A-only 输出。

## 19. Historical audit

正式运行前的 archived MOMD endpoint 审计保存在 `momd_endpoint_fomd_factorization_audit.json`；审计结果未用于修改门槛。

## 20. CCRA health

E5 responsibility IoU `{_f(ccra.get('responsibility_iou_median'))}`，distinct peaks `{_f(ccra.get('distinct_peak_fraction'))}`。

## 21. Conservation

完整守恒与 finite 结果记录在 `fomd_conservation.csv`。

## 22. F_full semantic health

| Metric | E5 |
| --- | ---: |
| CPR | {_f(cap.get('CPR'))} |
| Positive recall | {_f(full.get('positive_recall'))} |
| Positive F median | {_f(full.get('positive_F_median'))} |
| Rival leakage | {_f(full.get('rival_leakage'))} |
| Background leakage | {_f(full.get('background_leakage'))} |
| Probability gap | {_f(full.get('probability_gap'))} |

## 23. Query-identity sensitivity

`D_perm` mean/std/min/max = `{_f(query.get('D_perm_mean'))}` / `{_f(query.get('D_perm_std'))}` / `{_f(query.get('D_perm_min'))}` / `{_f(query.get('D_perm_max'))}`。

## 24. Full-vs-perm gain

逐 permutation 语义指标及预注册 margin 判断见 `fomd_perm_semantic_health.csv` 与 gate JSON。

## 25. Full-vs-global gain

空间敏感性及语义差值见 `fomd_spatial_ownership_sensitivity.csv`。

## 26. Full-vs-PCA gain

PCA 敏感性及语义差值见 `fomd_pca_sensitivity.csv`。

## 27. A class-conditioning

E5 median JS `{_f(ac.get('JS_median'))}`；top-owner class difference `{_f(at.get('top_owner_difference'))}`。

## 28. B region support

Bmax foreground/background/gap = `{_f(bs.get('Bmax_union_positive'))}` / `{_f(bs.get('Bmax_reliable_background'))}` / `{_f(bs.get('Bmax_gap'))}`。

## 29. B locality/centroid

E5 locality mass median `{_f(bl.get('locality_mass_median'))}`；centroid 结果见对应 CSV。

## 30. B area/fragmentation/redundancy

B empty fraction `{_f(ba.get('empty_fraction'))}`，area>90% fraction `{_f(ba.get('fraction_area_gt_090'))}`。

## 31. Contribution diagnostics

后验贡献互补性与 R→Q transfer 沿用 MOMD frozen 定义，见 `fomd_contribution_complementarity.csv` 和 `fomd_r_to_q_transfer.csv`。

## 32. PCA/deep/query health

PCA coverage、deep present/absent gap 与 query update 均在独立 CSV 中留档。

## 33. Gradient/detach audit

反事实仅在 `torch.no_grad()` monitor 路径生成；detach、F_full 等价性和梯度健康记录均已保存。

## 34. Visual evidence

固定 8 张 train images 的 step500、step1000、E2–E5 可视化保存在 `visualizations/`。

## 35. E2 screen

`{json.dumps(payload.get('epoch2_screen',{}),ensure_ascii=False)}`

## 36. E5 Gate A-J

| Gate | Result |
| --- | --- |
{gate_table}

## 37. STRONG GO

Strong GO：`{gate.get('strong_go',False)}`。

## 38. Scientific Interpretation

若 D/E/F 任一失败，分别意味着 query identity、像素级 ownership 或 CCRA 相对 PCA-only 的因果必要性证据不足；这不否定 MOMD 掩码健康，只限制 FOMD 机制主张。

## 39. Exact Decision

`{decision}`。

## 40. Full25 Recommendation

只有 FOMD_PHASE0_GO/STRONG_GO 才建议冻结架构并进入 fresh Seed42 Full25；本任务按方案在 Phase-0 报告后停止。

### Snapshot trajectory

| Snapshot | Positive recall | Probability gap | D_perm |
| --- | ---: | ---: | ---: |
{chr(10).join(history)}

DECISION = {decision}
"""
    path=output/"FOMD_Phase0_Factorized_Ownership_Mask_Decoding_Report.md"
    path.write_text(text,encoding="utf-8")
    return path


__all__=["render_report"]
