# GCQM BCSS Seed42 Full25 Final Validation Report

## 1 Executive Decision

最终结论：**GCQM_FULL25_NOGO**。GCQM−SSHR foreground mIoU 为 -2.3424 pp，paired 95% CI [-3.0121, -1.6730]。

## 2 Frozen GCQM Model

模型冻结为 CCRA → spatial mean(A) → detached global class-conditioned w → query mask bases B → F=ΣwB。

## 3 Phase-0 STRONG GO Provenance

冻结架构提交 `f327b91bba88bdf45391d991a13b214e9ab101ff`，Phase-0 A–J 全部通过。

## 4 Full25 Protocol

BCSS Seed42、官方 MXNet 初始化、batch20、BF16、25 epochs、29275 steps、E25 FINAL only。

## 5 Source / Config / Init Freeze

训练源码 `adf285dbd540906456f712a502512e0fd3952e75`；E25 SHA256 `6e1b909fc86a870e652213831521e8ff552371a083f85faad7dac3a21d969d0f`。

## 6 Dataset / Evaluator Provenance

训练仅访问 BCSS training；封存 E25 后才运行同一 BCSS validation evaluator。

## 7 Training Completion

完成 25 epochs / 29275 steps，用时 82.65 分钟。

## 8 Engineering Health

训练全程 finite；峰值显存 5.785 GiB。

## 9 E25 Checkpoint Seal

E25 checkpoint 在任何 segmentation metric 产生前完成哈希和封存，未做 best checkpoint 选择。

## 10 SSHR Comparability Audit

协议审计：**COMPARABLE**。复算 SSHR 与已封存结果逐项一致。

## 11 Main Segmentation Metrics

| Model | Seed | Epoch | fg mIoU | mDice | Params | Inference |
|---|---:|---:|---:|---:|---:|---:|
| SSHR B0 | 42 | 25 | 66.6967 | 79.7977 | 112,709,714 | 0.0150 s/img |
| GCQM | 42 | 25 | 64.3543 | 78.0756 | 111,279,322 | 0.0328 s/img |
| Delta | - | - | -2.3424 pp | -1.7221 pp | - | - |

## 12 mIoU / mDice Delta

ΔmIoU=-2.3424 pp；ΔmDice=-1.7221 pp。

## 13 Per-Class IoU

| Class | SSHR | GCQM | Δ pp |
|---:|---:|---:|---:|
| 0 | 76.3582 | 74.6965 | -1.6617 |
| 1 | 70.1575 | 67.1114 | -3.0460 |
| 2 | 57.3211 | 55.4846 | -1.8365 |
| 3 | 62.9500 | 60.1247 | -2.8254 |

## 14 Per-Class Dice

| Class | SSHR | GCQM | Δ pp |
|---:|---:|---:|---:|
| 0 | 86.5944 | 85.5157 | -1.0787 |
| 1 | 82.4618 | 80.3194 | -2.1424 |
| 2 | 72.8715 | 71.3699 | -1.5016 |
| 3 | 77.2630 | 75.0973 | -2.1657 |

## 15 Per-Image Paired Delta

共 3418 个严格配对 validation 图像，逐图数据见 CSV。

## 16 Bootstrap Confidence Interval

10,000 次 paired bootstrap，seed=20260910；mIoU 95% CI [-3.0121, -1.6730] pp。

## 17 Improved / Worsened Fractions

改善 16.91%；下降 52.19%；持平 30.90%。

## 18 Full25 Mechanism Survival

E5/E10/E15/E20/E25 的 train-only GCQM、query permutation、PCA 对照和 B 健康记录均已保存。

## 19 Query Allocation Health

E25 Stage3 JS=0.3637，top1 difference=1.0000，D_perm=0.1026。

## 20 GCQM-vs-PCA Diagnostic

仅作为 train-only 机制生存证据，不用于 checkpoint 选择。

## 21 Complexity / VRAM / Runtime

GCQM 参数 111,279,322；SSHR 参数 112,709,714；GCQM inference 0.0328 s/image。

## 22 Qualitative Wins

按逐图 ΔIoU 自动选取最大的 5 例，未人工挑图。

## 23 Similar Cases

按 |逐图 ΔIoU| 自动选取最接近 0 的 5 例。

## 24 Failure Cases

按逐图 ΔIoU 自动选取最小的 5 例，不隐藏失败案例。

## 25 Scientific Interpretation

GCQM 是否把 Phase-0 机制健康转化为真实分割增益，以预注册 E25 paired comparison 为唯一性能结论。

## 26 Exact Performance Decision

`GCQM_FULL25_NOGO`

## 27 Multi-Seed Recommendation

GO/STRONG_GO：冻结并进入同协议 multi-seed（11/17/42）；其余结论按预注册规则执行，不恢复旧机制。

DECISION = GCQM_FULL25_NOGO
