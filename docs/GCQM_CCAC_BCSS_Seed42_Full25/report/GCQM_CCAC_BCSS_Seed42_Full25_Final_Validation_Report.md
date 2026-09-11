# GCQM+CCAC BCSS Seed42 Full25 Final Validation Report

## 1 Executive Decision

**DECISION = CCAC_FULL25_NOGO**

## 2 Failure-Anatomy Motivation

冻结前序结论为 `MISSING_SPATIAL_COHERENCE (HIGH)`；本轮只修复内部 FN、孔洞与碎片化。

## 3 Frozen CCRA/GCQM Backbone

CCRA、global w、query allocation、Stage1 与 pixel decoder 主体全部冻结。

## 4 CCAC Design

3×3 detached cosine affinity（self=1、row-normalized），rival stop-gradient gate，fill-only 更新，Stage2/3 固定 T=2。

## 5 Literature Migration Provenance

仅迁移 structure-aware affinity 与 weak-response completion 原则；未引入新监督、superpixel、prototype 或外部模型。

## 6 Zero-Parameter / Gradient Contract

参数增量=0；新增辅助损失=0；affinity、rival 与 w side path 均 detached。

## 7 Engineering Validation

单元/回归与 2-step smoke 均通过；训练 finite=True。

## 8 Fresh Full25 Protocol

BCSS train、Seed42、fresh official MXNet init、BF16、batch20、25 epochs/29275 steps；训练期无 validation。

## 9 Training Completion

完成 25 epochs/29275 steps，用时 86.57 分钟。

## 10 E25 Seal

E25 FINAL SHA256 `848631927607bc2832c07cbbafdf0ba71ed3482ff1cec8a056abb2bbca7c80a6`，先封存后评价。

## 11 Main mIoU/mDice

SSHR=66.6967/79.7977；old GCQM=64.3543/78.0756；GCQM+CCAC=64.6169/78.2703。

## 12 Comparison vs SSHR

ΔmIoU=-2.0798 pp，paired 95% CI [-2.7477, -1.4243]。

## 13 Comparison vs old GCQM

ΔmIoU=+0.2626 pp，paired 95% CI [+0.2107, +0.3146]。

## 14 Per-Class Metrics

| Class | SSHR IoU | old GCQM IoU | CCAC IoU | Δ vs SSHR pp |
|---:|---:|---:|---:|---:|
| 0 | 76.3582 | 74.6965 | 74.8947 | -1.4635 |
| 1 | 70.1575 | 67.1114 | 67.4925 | -2.6650 |
| 2 | 57.3211 | 55.4846 | 55.6975 | -1.6236 |
| 3 | 62.9500 | 60.1247 | 60.3829 | -2.5672 |

## 15 Paired Bootstrap

BCSS validation 3418 张，10,000 次 paired bootstrap，seed=20260911。

## 16 H1 FN Recovery

normalized ΔFN=+0.021050，FN_RECOVERY=False；normalized ΔFP=-2.913048。

## 17 H2 Interior Recovery

interior loss=+0.022303，INTERIOR_RECOVERY=False。

## 18 Boundary Safety

boundary loss=+0.011235；BOUNDARY_OVERSMOOTHING_RISK=False。

## 19 Contact Safety

contact excess loss=-0.010055；NEW_CONTACT_LEAKAGE=False。

## 20 Morphology Recovery

gaps={'components': 0.24959029826286483, 'hole_count': 0.21730580137659786, 'compactness': 0.4034550591974779, 'fragmentation_index': 42.19250476891058}；MORPHOLOGY_RECOVERY=False。

## 21 FP Overcompletion Check

OVERCOMPLETION=False，FP CI=[-7.884612586458075, 0.2475804282217594]。

## 22 CCRA Survival

E25 train-only CCRA monitor retained 10 rows; exact table is preserved in mechanism/ccra_health.csv.

## 23 CCAC Completion Statistics

E25 Stage2/3 completion rows: [{'snapshot': 'epoch25', 'stage': 2, 'iterations': 2, 'completion_mass_mean': 0.0067371247569099, 'changed_fraction': 0.6617556214332581, 'rival_protected_mass_mean': 0.0007367146317847, 'max_negative_change': 0.0, 'affinity_entropy_mean': 2.1680914759635925, 'positive_neighbor_affinity_fraction': 1.0, 'use_rival': 1.0, 'base_mean': 0.1686650663614273, 'restored_mean': 0.17540218308568, 'base_positive_fraction': 0.1386818382889032, 'restored_positive_fraction': 0.1391576621681451}, {'snapshot': 'epoch25', 'stage': 3, 'iterations': 2, 'completion_mass_mean': 0.0066367975668981, 'changed_fraction': 0.6690399497747421, 'rival_protected_mass_mean': 0.0007356957212323, 'max_negative_change': 0.0, 'affinity_entropy_mean': 2.1680914759635925, 'positive_neighbor_affinity_fraction': 1.0, 'use_rival': 1.0, 'base_mean': 0.1670910269021988, 'restored_mean': 0.1737278252840042, 'base_positive_fraction': 0.1367062907665968, 'restored_positive_fraction': 0.1371895894408226}]

## 24 E25 Causal Ablations

| Variant | mIoU | mDice |
|---|---:|---:|
| full_ccac | 64.6169 | 78.2703 |
| ccac_off | 64.1820 | 77.9499 |
| uniform_local | 64.6215 | 78.2737 |
| no_rival | 64.6227 | 78.2746 |

## 25 Complexity

trainable params=111,279,322（delta 0）；train peak=6.252 GiB；CCAC inference=0.1113 s/image。

## 26 Qualitative Recovery Cases

样例按与 old GCQM 的逐图 ΔIoU 自动排序，索引见 `visualizations/selection.json`。

## 27 Failure Cases

最大退化样例同样自动保留，未人工筛选。

## 28 Scientific Interpretation

本轮得到的是一个可解释的 **NOGO**，而不是完全无效：

- 在同一 CCAC-trained checkpoint 上，开启 completion 后由 64.1820 提升至 64.6169 mIoU，即算子贡献约 **+0.4349 pp**；但该 checkpoint 的 CCAC-OFF base 比旧 GCQM 低约 **-0.1723 pp**，因此相对旧 GCQM 的最终净增益只剩 **+0.2626 pp**。
- normalized FN gap 从旧值 +0.0227 降至 +0.02105，只恢复约 **7.3%**；interior loss 从 +0.0241 降至 +0.02230，只恢复约 **7.5%**，均远低于预注册的 50% recovery 门槛。
- components、hole count、compactness gap 分别仅缩小约 **14.3% / 11.2% / 15.7%**，没有一项达到 50%，故 `MORPHOLOGY_RECOVERY=False`。
- 没有引入新的主要副作用：boundary oversmoothing、contact leakage 与显著 FP overcompletion 均为 false。这说明方向具有安全性，但当前作用强度不足。
- E25 Stage3 虽有约 66.9% 像素发生非零数值变化，但平均 completion mass 只有 0.00664，>0.5 的像素比例仅从 13.6706% 增至 13.7190%。即“许多像素被轻微抬升”，真正跨过决策边界的像素很少。
- affinity entropy 约 2.168，接近 9 邻域均匀分布的理论上限 ln(9)=2.197，且 positive-neighbor affinity fraction=1.0；对应地 uniform-local 64.6215 略高于 full CCAC 64.6169。这意味着当前 detached pixel feature 的局部 cosine affinity 缺少足够的结构选择性。
- no-rival 为 64.6227，也略高于 full CCAC；rival-protected mass 仅约 0.000736。因此本实验没有支持“rival gate 提供独立性能价值”的论文主张。

综合而言：局部 fill-only completion 能产生小而统计显著的增益，但当前 affinity 几乎退化为均匀局部传播，且两跳补全不足以修复已定位的内部连通性问题。不能把这次结果解释为 CCAC 已验证成功。

## 29 Exact Decision

`DECISION = CCAC_FULL25_NOGO`

## 30 Next Step

不建议直接做 multi-seed，也不建议在本轮结果上调 T、邻域或阈值。下一轮设计应首先回答一个更具体的问题：为什么 Pixel Decoder feature 的 3×3 cosine affinity 接近均匀分布，以及如何获得具有组织边界选择性的 affinity。新的方案应继续保留本轮已证明安全的 fill-only 思路，同时把“feature affinity 是否真的优于 uniform propagation”设为必须通过的独立因果门槛。

DECISION = CCAC_FULL25_NOGO
