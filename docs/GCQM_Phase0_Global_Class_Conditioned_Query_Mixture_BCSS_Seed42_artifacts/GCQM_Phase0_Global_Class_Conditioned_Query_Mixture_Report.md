# GCQM Phase-0 Global Class-Conditioned Query Mixture 实验报告

## 1 Executive Decision

最终结论：**GCQM_PHASE0_STRONG_GO**。

## 2 FOMD evidence that motivates simplification

FOMD 的 pixel-wise ownership 无增益且 B locality 失败，但 query identity、CCRA-vs-PCA 与最终 mask 健康，因此检验更小的全局类条件 query mixture。

## 3 Removed claims

不再声称 pixel-wise ownership 必要，也不再声称 B 是 localized 或 class-specific expert。

## 4 GCQM hypothesis

`w(i,c)=normalize_q(mean_x A(i,x,c))`，`F(c,x)=Σ_i w(i,c)B(i,x)`。

## 5 Frozen architecture

沿用 FOMD lineage `f327b91bba88bdf45391d991a13b214e9ab101ff`；ResNet38、196 queries、CCRA、CHPF、PCA/deep、tri-state 和优化协议保持冻结。

## 6 CCRA spatial evidence

E5 responsibility IoU `0.193916`，distinct peaks `0.942857`。

## 7 Global query pooling

A 仅作为权重估计中间量，空间平均后沿 query 轴重新归一。

## 8 w class-conditioned allocation

JS median `0.290240`；top1 class difference `1.000000`。

## 9 Query mask bases B

B 仍为原始 query mask logit 的 sigmoid，只要求非退化。

## 10 GCQM mixture F

Stage2/3 的唯一 primary mask 与 mask loss 输入为 `ΣwB`。

## 11 Loss contract

总损失保持 deep/PCA/mask = 0.50/0.25/0.25，新增 auxiliary loss 为 0。

## 12 Zero parameter delta

GCQM 参数增量 `0`。

## 13 Historical compatibility audit

正式训练前的 archived FOMD endpoint 审计结果为 `PROCEED`，且没有修改预注册门槛。

## 14 Weight conservation

max weight-sum error `0.000000`；max C/F error `0.000000`。

## 15 Weight class-conditioning

详见 `gcqm_weight_class_conditioning.csv`。

## 16 Weight utilization

dominant share `0.017337`；effective queries `139.103188`；top5 mass `0.075435`。

## 17 Primary class-mask health

| Metric | E5 |
| --- | ---: |
| CPR | 1.468692 |
| Positive recall | 1.000000 |
| Positive F median | 0.827721 |
| Rival leakage | 0.039219 |
| Background leakage | 0.003409 |
| Probability gap | 0.695568 |

## 18 Query-identity sensitivity

D_perm mean/std/min/max = `0.122051` / `0.088568` / `0.000139` / `0.318784`。

## 19 Full-vs-permutation

8 个固定 derangements 的逐项结果保存在 `gcqm_perm_semantic_health.csv`。

## 20 GCQM-vs-PCA

gap gain `0.406845`；rival reduction `0.528324`；confidence gain `0.271720`。

## 21 GCQM-vs-pixelwise noninferiority

gap/recall/confidence delta = `0.027053` / `0.024907` / `0.037319`。

## 22 B basis health

empty fraction `0.006364`；area>90% fraction `0.000000`；component p90 `7.000000`。

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

`{"decision": "CONTINUE_TO_E5_UNCHANGED", "checks": {"engineering": false, "semantic": false, "specificity": false, "ccra": false, "monopoly": false}, "failed_criteria": []}`

## 31 E5 Gate A-J

| Gate | Result |
| --- | --- |
| A_CCRA_preserved | PASS |
| B_GCQM_conservation | PASS |
| C_primary_mask_healthy | PASS |
| D_query_identity | PASS |
| E_beats_PCA | PASS |
| F_pixel_unnecessary | PASS |
| G_class_conditioned_allocation | PASS |
| H_B_nondegenerate | PASS |
| I_late_stability | PASS |
| J_PCA_deep | PASS |

## 32 STRONG GO

Strong GO：`True`。

## 33 Scientific Interpretation

GCQM 只在最终掩码健康、query identity 有因果贡献、明显优于 PCA、且不劣于 pixel-wise FOMD 时成立；失败 gate 精确限定该最小模型的不足。

## 34 Exact Decision

`GCQM_PHASE0_STRONG_GO`。

## 35 Full25 Recommendation

只有 GO/STRONG_GO 才冻结模型并进入 fresh BCSS Seed42 Full25；本任务在 Phase-0 后停止。

### Snapshot trajectory

| Snapshot | CPR | Recall | Gap | D_perm |
| --- | ---: | ---: | ---: | ---: |
| step0250 | 1.089396 | 0.812179 | 0.102284 | 0.036280 |
| step0500 | 2.134026 | 0.888856 | 0.273565 | 0.105750 |
| step1000 | 1.607896 | 0.998941 | 0.364595 | 0.116158 |
| epoch1 | 1.682120 | 0.975711 | 0.408370 | 0.099657 |
| epoch2 | 1.492513 | 1.000000 | 0.507708 | 0.092127 |
| epoch3 | 1.160639 | 1.000000 | 0.544423 | 0.091997 |
| epoch4 | 1.161632 | 1.000000 | 0.571026 | 0.079381 |
| epoch5 | 1.468692 | 1.000000 | 0.695568 | 0.122051 |

DECISION = GCQM_PHASE0_STRONG_GO
