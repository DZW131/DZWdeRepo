# RDDR Phase-2B1.5 Adjudication Bias Decomposition & Third-Evidence Audit

**最终判定：`SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED`。**

同分支来源偏差得到支持；对称化明显改善裁决，第三证据探针也满足预注册门槛。但class2仅勉强达到宽松门槛、class3证据不足，不能宣称所有类别反转已解决。这是零训练机制审计，不是新模型的官方mIoU提升。

除注明百分比/pp外，表中均使用0–1比例；pp为百分点。所有语义指标为原生28-grid，不是官方224/TTA/final-CAM指标。Phase-2B1 NOGO保持不变。

## 1. Provenance / immutable inputs

- Pure A0：`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Probe commit：`ec7abb6a2d889b6dad7f20b4539806395e93d37b`
- Analysis commit：`a09b51de5eef82973f908dd68fd0ef84cb933b6a`
- Checkpoint：`/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- Checkpoint SHA256：`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Native cache：`/home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz`
- Cache SHA256：`767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a`
- Derived observations SHA256：`237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514`
- Statistics SHA256：`acf2f821b7d161a3166d4bd9885b617d4dfc3d316cbdf95723e5680d673ffdd9`

全部3418 validation图像、C0 Full25 seed42。仅复用缓存中的ps/pd及冻结GT/分组；未加载模型、未前向、未创建optimizer、未反传、未写checkpoint，未访问test/LUAD/其他seed。旧报告、缓存、baseline均未删除或覆盖。

## 2. Exact commands / environment / resources

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b15
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b15_bias_decomposition_audit.py --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz --previous-report /home/duyanhong/experiments/RDDR_PHASE2B1/report_r1 --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --output /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/analyze_rddr_phase2b15.py --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1 --output /home/duyanhong/experiments/RDDR_PHASE2B15/report_r1
```

NVIDIA GeForce RTX 5090 D v2；Python 3.10.20 / PyTorch 2.11.0+cu128 / NumPy 1.23.5。缓存来自冻结BF16前向，本轮以batch1/FP32计算概率邻域，不重新softmax或调整temperature。

旧分数校验阶段 3.07s；新探针 6.94s；含校验/压缩总计 18.34s；全量统计和10,000 bootstrap 34.12s。CUDA allocated峰值 15.57MiB，reserved 26.00MiB。衍生观测 287.62MiB，不保存全数据集pair张量。

## 3. Phase-2B1 exact reproduction / populations

| Field | Maximum absolute difference |
|---|---|
| T_SS | 0 |
| T_DS | 0 |
| Delta_old | 0 |
| ctx_S | 0 |

先完成全量旧分数校验，再计算新探针；严格容忍上限1e-7未放宽。原生q、Top20、boundary和Q1–Q5直接复用，不重新挑选。

| Population | Targets |
|---|---|
| all | 2479143 |
| hard_disagreement | 587701 |
| adjudication | 497629 |
| Deep_Win | 314730 |
| Shallow_Win | 182899 |
| Both_Wrong | 393677 |
| Top20 | 485451 |
| Bottom80 | 1993692 |

GT类别0–3为metric targets；背景4/ignore255不计分，但仍可作为无监督support的邻居位置。旧by_CH标签表示raw→完整HFRM而非CH-only因果干预；历史缓存逐图人数已核验，但不声称拥有不存在的原始历史像素哈希。

## 4. Four-way support definitions

```text
T_ab(i)=mean_j clip(1-JS(p_a(i),p_b(j))/ln2,0,1)
T_SS=S<-S; T_SD=S<-D; T_DS=D<-S; T_DD=D<-D
```

natural log、epsilon1e-8、temperature1；15×15/r7、排除self、仅图内邻居。完整四类概率，不one-hot，不加入source reliability/q weighting/边界项/距离权重。计算函数只接受ps/pd，GT诊断与它隔离。

## 5. T_SS / T_SD / T_DS / T_DD statistics

| Support | Mean | Std | p05 | p25 | Median/p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|
| T_SS | 0.861603 | 0.099898 | 0.658249 | 0.823360 | 0.887859 | 0.931204 | 0.966861 |
| T_SD | 0.741500 | 0.184582 | 0.397907 | 0.628756 | 0.756744 | 0.894810 | 0.988408 |
| T_DS | 0.737317 | 0.174717 | 0.374911 | 0.643575 | 0.781571 | 0.871664 | 0.941502 |
| T_DD | 0.802548 | 0.197381 | 0.401278 | 0.686554 | 0.858636 | 0.974863 | 0.999639 |

全部45个固定分组的同类统计保存在support_matrix.csv。

## 6. Same-family bias: B_S / B_D / B_family

```text
B_S=T_SS-T_SD; B_D=T_DD-T_DS
B_family=.5*(B_S+B_D)
```

| Bias | Mean | Std | p05 | p25 | Median/p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|
| B_S | 0.120103 | 0.150471 | -0.100031 | 0.010339 | 0.114093 | 0.212755 | 0.376540 |
| B_D | 0.065231 | 0.153726 | -0.177667 | -0.021475 | 0.064724 | 0.139485 | 0.321785 |
| B_family | 0.092667 | 0.105493 | 0.001543 | 0.022598 | 0.060890 | 0.124825 | 0.300401 |

均值及10,000 image-bootstrap 95%CI：

| Group | B_S [CI] | B_D [CI] | B_family [CI] |
|---|---|---|---|
| all | 0.120103 [0.117159, 0.123044] | 0.065231 [0.062490, 0.068062] | 0.092667 [0.090356, 0.095053] |
| hard_disagreement | 0.278789 [0.274000, 0.283659] | 0.105913 [0.098876, 0.113178] | 0.192351 [0.186543, 0.198340] |
| Deep_Win | 0.304527 [0.297513, 0.311465] | 0.167242 [0.156191, 0.178251] | 0.235885 [0.226907, 0.244801] |
| Shallow_Win | 0.224576 [0.221409, 0.227711] | 0.034655 [0.030258, 0.039037] | 0.129616 [0.126167, 0.133077] |
| Both_Wrong | 0.189575 [0.186046, 0.193183] | -0.008146 [-0.012374, -0.003834] | 0.090714 [0.088293, 0.093349] |
| Top20 | 0.287980 [0.282907, 0.293053] | 0.166219 [0.157926, 0.174701] | 0.227099 [0.220537, 0.233837] |
| Bottom80 | 0.079226 [0.077390, 0.081021] | 0.040641 [0.038676, 0.042587] | 0.059933 [0.059127, 0.060723] |
| class0 | 0.094079 [0.091104, 0.097041] | 0.024305 [0.022137, 0.026456] | 0.059192 [0.057964, 0.060455] |
| class1 | 0.109583 [0.106359, 0.112816] | 0.064271 [0.061706, 0.066857] | 0.086927 [0.085144, 0.088762] |
| class2 | 0.128467 [0.123450, 0.133683] | 0.103258 [0.097471, 0.108909] | 0.115863 [0.113150, 0.118565] |
| class3 | 0.349177 [0.336028, 0.361649] | 0.258637 [0.237365, 0.279662] | 0.303907 [0.287433, 0.320053] |
| boundary | 0.158819 [0.155688, 0.162102] | 0.033166 [0.029271, 0.037079] | 0.095993 [0.093559, 0.098527] |
| interior | 0.116684 [0.113639, 0.119730] | 0.068062 [0.065228, 0.070989] | 0.092373 [0.089940, 0.094885] |

全体两个分支的同源偏好均为正、CI下界均大于0，Gate A通过。这个结论是聚合意义的，不能说每个像素都同向；B_S/B_D的p05仍可为负。

## 7. Source-branch reversal

| Score | Mean score | Pooled AUC | Image AUC | BA | Deep-Win recall | Shallow-Win recall |
|---|---|---|---|---|---|---|
| old | -0.124286 | 0.638180 | 0.734850 | 0.593973 | 0.261653 | 0.926293 |
| dsrc | 0.061048 | 0.824801 | 0.792569 | 0.724887 | 0.880050 | 0.569724 |

仅更换邻居来源，mean Delta从负转正，Deep-Win recall由26.17%变为88.00%。这支持来源分支影响绝对支持尺度。deep-source探针即使部分指标更高也不被选作新primary，未搜索阈值。

## 8. Delta_old vs preregistered Delta_sym

```text
S_S_sym=.5*(T_SS+T_SD); S_D_sym=.5*(T_DS+T_DD)
Delta_sym=S_D_sym-S_S_sym
```

代数上Delta_sym=Delta_old+B_family；B_family是逐位置的双源观测量，不是调出来的常数偏移。因此不能把分数均值变化与B_family均值当成两份独立证据。

| Score | Mean score | Pooled AUC | Image AUC | BA | Deep-Win recall | Shallow-Win recall |
|---|---|---|---|---|---|---|
| old | -0.124286 | 0.638180 | 0.734850 | 0.593973 | 0.261653 | 0.926293 |
| sym | -0.031619 | 0.788282 | 0.784842 | 0.715627 | 0.640314 | 0.790939 |


## 9. Zero-point bias shrinkage

| Group | Old mean | Old median | Sym mean | Sym median | BiasShrink |
|---|---|---|---|---|---|
| all | -0.124286 | -0.080376 | -0.031619 | -0.016857 | 0.745595 |
| Deep_Win | -0.170007 | -0.142318 | 0.065878 | 0.061754 | 0.612499 |
| Shallow_Win | -0.288588 | -0.268283 | -0.158973 | -0.146401 | 0.449137 |

全体BiasShrink=74.56%，95%CI [73.54%, 75.60%]。这是平均零点偏移缩小的描述，不是‘解释了多少训练性能差距’的因果比例。

## 10. Adjudication AUROC / AP / BA / recalls

| Score | Pooled AUROC | Image AUROC | Pooled AP | Image AP | Accuracy | BA | Macro F1 | Deep recall | Shallow recall | AUC images |
|---|---|---|---|---|---|---|---|---|---|---|
| old | 0.638180 | 0.734850 | 0.765099 | 0.822545 | 0.505935 | 0.593973 | 0.490333 | 0.261653 | 0.926293 | 3180 |
| sym | 0.788282 | 0.784842 | 0.847240 | 0.836048 | 0.695675 | 0.715627 | 0.691649 | 0.640314 | 0.790939 | 3180 |

Primary只在exactly-one-correct hard conflict上计算。图内单一标签AUROC=NA，不填0.5；无正例AP=NA，全正例AP=1。所有zero sign固定score>0选deep，其余选shallow。Sym imageAUROC的95%CI为[0.777130, 0.792815]；与old的配对差=0.049991，CI [0.045288, 0.054654]。

## 11. Symmetric anchor utility

```text
wD_sym=S_D_sym/(S_S_sym+S_D_sym+eps)
p_anchor_sym=(1-wD_sym)*ps+wD_sym*pd
```

| Estimator | Accuracy | mIoU | Dice | NLL | Brier |
|---|---|---|---|---|---|
| fixed_average | 0.774087 | 0.573520 | 0.723531 | 0.681292 | 0.341498 |
| anchor_old | 0.779292 | 0.569087 | 0.718405 | 0.669878 | 0.334832 |
| anchor_sym | 0.785383 | 0.593171 | 0.740518 | 0.659663 | 0.326990 |

相对FixedAvg：accuracy +1.1297pp，mIoU +1.9651pp，mIoU差值95%CI [1.8105, 2.1300]pp。相对old anchor：mIoU +2.4084pp。

指标从总体4×4 confusion计算，zero-union类别排除为NA，不做GT背景覆盖。NLL使用-log(pGT+eps)，Brier为四类平方误差之和。各分组mIoU不能按像素数平均还原总体。

## 12. Deep/shallow safety

| Group | Targets | FixedAvg acc | Old acc | Sym acc | Sym-Fixed |
|---|---|---|---|---|---|
| Deep_Correct | 1902567 | 0.986263 | 0.970332 | 0.986796 | 0.000533 |
| Top20_Deep_Correct | 288998 | 0.969387 | 0.869587 | 0.968716 | -0.000671 |
| Deep_Wrong | 576576 | 0.073956 | 0.148903 | 0.120770 | 0.046814 |
| Top20_Deep_Wrong | 196453 | 0.056110 | 0.235405 | 0.163708 | 0.107598 |
| Shallow_Correct | 1770736 | 0.920049 | 0.944324 | 0.935223 | 0.015174 |
| Top20_Shallow_Correct | 171778 | 0.460018 | 0.664392 | 0.582694 | 0.122676 |
| Shallow_Wrong | 708407 | 0.409239 | 0.366776 | 0.410844 | 0.001605 |
| Top20_Shallow_Wrong | 313673 | 0.676351 | 0.584771 | 0.675940 | -0.000411 |

Deep-Wrong总体与Top20仍分别高于FixedAvg约4.6814/10.7598pp，但救援幅度小于old anchor；与此同时Deep-Correct及Shallow-Wrong损失大幅减少。不能只报救援或只报整体提升。本阶段D门槛针对第三证据的rescue/harm，不偷换为上一轮Gate D。

## 13. Per-class support matrix and semantic IoU

| Class | T_SS | T_SD | T_DS | T_DD | B_S | B_D | B_family | Old mean | Sym mean | Deep-Win | Shallow-Win | Both-Wrong |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| class0 | 0.885070 | 0.790990 | 0.792803 | 0.817108 | 0.094079 | 0.024305 | 0.059192 | -0.092267 | -0.033075 | 43802 | 88423 | 111834 |
| class1 | 0.861512 | 0.751929 | 0.743960 | 0.808231 | 0.109583 | 0.064271 | 0.086927 | -0.117553 | -0.030625 | 121742 | 82309 | 133582 |
| class2 | 0.790036 | 0.661569 | 0.653672 | 0.756930 | 0.128467 | 0.103258 | 0.115863 | -0.136364 | -0.020501 | 74850 | 11749 | 101570 |
| class3 | 0.865708 | 0.516532 | 0.508033 | 0.766670 | 0.349177 | 0.258637 | 0.303907 | -0.357675 | -0.053768 | 74336 | 418 | 46691 |

各类裁决能力：

| Class | Old pooled AUC | Sym pooled AUC | Old image AUC | Sym image AUC | Old BA | Sym BA |
|---|---|---|---|---|---|---|
| class0 | 0.866710 | 0.838710 | 0.825913 | 0.790294 | 0.736861 | 0.768956 |
| class1 | 0.827822 | 0.863576 | 0.781674 | 0.796925 | 0.652146 | 0.781379 |
| class2 | 0.319921 | 0.537167 | 0.275458 | 0.451437 | 0.412252 | 0.527845 |
| class3 | 0.051692 | 0.584706 | 0.071672 | 0.340851 | 0.408576 | 0.567818 |

下面是**全体混淆矩阵**的各类IoU，不是仅筛选该GT类后得到的macro mIoU：

| Class | FixedAvg IoU | Old IoU | Sym IoU |
|---|---|---|---|
| 0 | 0.699399 | 0.701335 | 0.708893 |
| 1 | 0.640964 | 0.651215 | 0.654297 |
| 2 | 0.514291 | 0.513676 | 0.522020 |
| 3 | 0.439427 | 0.410121 | 0.487472 |


## 14. All 12 ordered prediction pairs

| S→D pair | Targets | Deep-Win | Shallow-Win | Winner prevalence | Old image AUC | Sym image AUC | AUC images | Old BA | Sym BA | Support |
|---|---|---|---|---|---|---|---|---|---|---|
| pair0_1 | 201473 | 101044 | 71666 | 0.585050 | 0.552529 | 0.596759 | 1073 | 0.621501 | 0.728721 | DEFINED |
| pair0_2 | 44977 | 30821 | 3438 | 0.899647 | 0.553231 | 0.636400 | 103 | 0.615035 | 0.781590 | DEFINED |
| pair0_3 | 63094 | 32874 | 13319 | 0.711666 | 0.445592 | 0.595887 | 134 | 0.501879 | 0.694004 | DEFINED |
| pair1_0 | 66979 | 39164 | 22806 | 0.631983 | 0.577207 | 0.618997 | 920 | 0.672743 | 0.717454 | DEFINED |
| pair1_2 | 76360 | 43655 | 27055 | 0.617381 | 0.514648 | 0.560793 | 637 | 0.563816 | 0.682136 | DEFINED |
| pair1_3 | 86266 | 39959 | 32448 | 0.551867 | 0.473687 | 0.562838 | 164 | 0.501579 | 0.714694 | DEFINED |
| pair2_0 | 5604 | 3972 | 644 | 0.860485 | 0.495859 | 0.592029 | 23 | 0.723764 | 0.745557 | DEFINED |
| pair2_1 | 35079 | 19452 | 10522 | 0.648962 | 0.521091 | 0.559132 | 552 | 0.600820 | 0.623933 | DEFINED |
| pair2_3 | 3995 | 1503 | 583 | 0.720518 | 0 | 0 | 1 | 0.521260 | 0.727452 | DEFINED |
| pair3_0 | 1273 | 666 | 194 | 0.774419 | 0.517468 | 0.636859 | 8 | 0.745132 | 0.768885 | DEFINED |
| pair3_1 | 1888 | 1246 | 151 | 0.891911 | 0.504167 | 0.566667 | 8 | 0.711360 | 0.791308 | DEFINED |
| pair3_2 | 713 | 374 | 73 | 0.836689 | 0 | 0 | 1 | 0.640228 | 0.643964 | LOW_SUPPORT |

3→2的Shallow-Win仅73，标记LOW_SUPPORT，不据此下结论。2→3虽通过像素人数门槛，其image AUROC也仅由1张同时含正负例的图决定；0值不能当成稳健反向信号。3→0/3→1仅8张可计算AUROC图，保留这一限制，不新增门槛、不选择最佳pair。全部四项支持度及pooled AUROC也随ordered_pairs.csv保存。

## 15. Class priors / confidence (no calibration)

| GT group | Head | Pred0 | Pred1 | Pred2 | Pred3 | Mean p0 | Mean p1 | Mean p2 | Mean p3 | Max confidence | Entropy nats |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | shallow | 0.458676 | 0.439594 | 0.086816 | 0.014914 | 0.426801 | 0.405377 | 0.119481 | 0.048341 | 0.740731 | 0.677926 |
| all | deep | 0.363608 | 0.443158 | 0.118025 | 0.075209 | 0.363864 | 0.436968 | 0.122249 | 0.076919 | 0.943320 | 0.146711 |
| class0 | shallow | 0.838741 | 0.141588 | 0.016101 | 0.003569 | 0.721703 | 0.190846 | 0.051699 | 0.035752 | 0.784872 | 0.589988 |
| class0 | deep | 0.792508 | 0.164828 | 0.014040 | 0.028624 | 0.783373 | 0.167350 | 0.017792 | 0.031486 | 0.952057 | 0.124471 |
| class1 | shallow | 0.193531 | 0.755599 | 0.043638 | 0.007233 | 0.227514 | 0.639009 | 0.090602 | 0.042875 | 0.741528 | 0.681443 |
| class1 | deep | 0.097179 | 0.793345 | 0.058202 | 0.051274 | 0.103512 | 0.775560 | 0.067813 | 0.053116 | 0.942388 | 0.150695 |
| class2 | shallow | 0.218055 | 0.323568 | 0.454678 | 0.003700 | 0.229663 | 0.310902 | 0.422721 | 0.036715 | 0.683646 | 0.780964 |
| class2 | deep | 0.056909 | 0.273060 | 0.649726 | 0.020305 | 0.063535 | 0.276121 | 0.638416 | 0.021928 | 0.921681 | 0.196117 |
| class3 | shallow | 0.376556 | 0.405465 | 0.048051 | 0.169928 | 0.340048 | 0.361085 | 0.102238 | 0.196629 | 0.569497 | 1.006210 |
| class3 | deep | 0.114037 | 0.153838 | 0.055225 | 0.676900 | 0.118797 | 0.152534 | 0.058438 | 0.670230 | 0.940182 | 0.155760 |

在GT class3，shallow输出class0/1合计约78.20%，class3仅16.99%；deep输出class3为67.69%。deep平均max confidence=0.9402，shallow=0.5695，概率几何明显不同。这些是条件预测分布/置信度差异，不是训练集class prior的因果估计。未进行校准。

## 16. Candidate-class soft mass

| Group (hard only) | Targets | M_s^S | M_d^S | d-s S margin | M_s^D | M_d^D | d-s D margin |
|---|---|---|---|---|---|---|---|
| Deep_Win | 314730 | 0.314007 | 0.439343 | 0.125336 | 0.142155 | 0.754068 | 0.611913 |
| Shallow_Win | 182899 | 0.507100 | 0.274605 | -0.232495 | 0.440168 | 0.439792 | -0.000376 |
| class0 | 152759 | 0.421363 | 0.371909 | -0.049453 | 0.329540 | 0.527799 | 0.198259 |
| class1 | 229638 | 0.382458 | 0.381999 | -0.000459 | 0.262362 | 0.588800 | 0.326439 |
| class2 | 116575 | 0.308708 | 0.396261 | 0.087553 | 0.180558 | 0.633279 | 0.452721 |
| class3 | 88729 | 0.375991 | 0.234949 | -0.141042 | 0.109484 | 0.752279 | 0.642795 |

所有12个ordered pair的mass和class2/3按winner状态分解保存在candidate_mass.csv。candidate由ps/pd argmax确定，mass是source概率邻域均值，GT不进入其计算。

## 17. GT contextual availability (audit only)

| Group (hard only) | Targets | Same GT | S candidate | D candidate | Other FG | Background | Ignore |
|---|---|---|---|---|---|---|---|
| Deep_Win | 314730 | 0.867565 | 0.065504 | 0.867565 | 0.039465 | 0.027466 | 0 |
| Shallow_Win | 182899 | 0.795934 | 0.795934 | 0.125204 | 0.044132 | 0.034731 | 0 |
| class0 | 152759 | 0.850335 | 0.513367 | 0.331867 | 0.136192 | 0.018573 | 0 |
| class1 | 229638 | 0.805284 | 0.326351 | 0.497127 | 0.132401 | 0.044121 | 0 |
| class2 | 116575 | 0.815573 | 0.127729 | 0.605680 | 0.237805 | 0.028787 | 0 |
| class3 | 88729 | 0.900846 | 0.039343 | 0.781900 | 0.160601 | 0.018155 | 0 |

所有比例以图内非self合法邻居为分母。S候选+D候选+otherFG+background+ignore=1；sameGT与这些项重叠，不再相加。mask nearest到28-grid后计算GT邻域，分数仍完全GT-blind。GT里有正确类别不等于网络已编码正确语义，只能帮助区分可用性不足与表征/score偏差。

## 18. Dedicated class2/class3 root-cause decomposition

| Class | Deep-Win | Shallow-Win | B_S | B_D | B_family | Old Delta | Sym Delta | Old image AUC | Sym image AUC | Mass/GT targets | S mass margin | D mass margin | GT same | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| class2 | 74850 | 11749 | 0.128467 | 0.103258 | 0.115863 | -0.136364 | -0.020501 | 0.275458 | 0.451437 | 116575 | 0.087553 | 0.452721 | 0.815573 | PASS |
| class3 | 74336 | 418 | 0.349177 | 0.258637 | 0.303907 | -0.357675 | -0.053768 | 0.071672 | 0.340851 | 88729 | -0.141042 | 0.642795 | 0.900846 | UNDERPOWERED |

支持度均值覆盖该GT类全体；mass/GT均值只覆盖其中hard disagreement，单独列出人数。

| Winner state | Targets | S d-s margin | D d-s margin | GT same |
|---|---|---|---|---|
| class2_Deep_Win | 74850 | 0.106268 | 0.533450 | 0.848651 |
| class2_Shallow_Win | 11749 | 0.191388 | 0.282777 | 0.683726 |
| class3_Deep_Win | 74336 | -0.168173 | 0.725774 | 0.918953 |
| class3_Shallow_Win | 418 | 0.226226 | 0.111764 | 0.802672 |

**class2：** image AUROC从0.2755升至0.4514，95%CI [0.4256,0.4771]。仅因预注册门槛是0.45而PASS，CI仍全部低于0.5，不能写成已恢复正确排序。Deep-Win的GT同类邻居比例84.87%；Shallow-Win仍有68.37%，但两种source的候选mass都平均偏向错误deep候选，提示方向错误并非仅由GT邻域缺失解释。

**class3：** B_family=0.3039明显较大，mean Delta从-0.3577移到-0.05377；image AUROC从0.0717升至0.3409，CI [0.2786,0.4058]。418个Shallow-Win使其按合同UNDERPOWERED，不是PASS，也不按该门槛计FAIL；低于0.5的观察仍如实保留。Deep-Win处GT同类邻居91.90%，shallow-source候选margin却为-0.1682，而deep-source为+0.7258，支持语义表征方向差异而非纯邻域缺少正确类别。Shallow-Win中GT同类比例80.27%，但source mass又可偏错误deep；对称化不是万能校正。

Class-specific hard-conflict pair组成如下（全部12类均保留）。固定GT类和prediction pair后，winner标签通常天然单一，其AUROC=NA而非0.5；这张表只解释组成，不从条件pair推导阈值或规则。

| Class/pair | Targets | Fraction of class hard | Deep-Win | Shallow-Win | S margin | D margin | GT same | Support |
|---|---|---|---|---|---|---|---|---|
| class2_pair0_1 | 21986 | 0.188600 | 0 | 0 | 0.063687 | 0.419279 | 0.772143 | LOW_SUPPORT |
| class2_pair0_2 | 30821 | 0.264388 | 30821 | 0 | 0.168783 | 0.680030 | 0.861565 | LOW_SUPPORT |
| class2_pair0_3 | 2368 | 0.020313 | 0 | 0 | -0.293440 | 0.088740 | 0.821482 | LOW_SUPPORT |
| class2_pair1_0 | 2367 | 0.020305 | 0 | 0 | 0.063677 | 0.060353 | 0.818353 | LOW_SUPPORT |
| class2_pair1_2 | 43655 | 0.374480 | 43655 | 0 | 0.060299 | 0.428974 | 0.839578 | LOW_SUPPORT |
| class2_pair1_3 | 3025 | 0.025949 | 0 | 0 | -0.307590 | -0.051540 | 0.818987 | LOW_SUPPORT |
| class2_pair2_0 | 644 | 0.005524 | 0 | 644 | 0.155508 | 0.064697 | 0.867989 | LOW_SUPPORT |
| class2_pair2_1 | 10522 | 0.090259 | 0 | 10522 | 0.211129 | 0.310013 | 0.666130 | LOW_SUPPORT |
| class2_pair2_3 | 583 | 0.005001 | 0 | 583 | -0.125256 | 0.032125 | 0.797744 | LOW_SUPPORT |
| class2_pair3_0 | 31 | 0.000266 | 0 | 0 | 0.257712 | 0.252327 | 0.971864 | LOW_SUPPORT |
| class2_pair3_1 | 199 | 0.001707 | 0 | 0 | 0.352112 | 0.510917 | 0.776639 | LOW_SUPPORT |
| class2_pair3_2 | 374 | 0.003208 | 374 | 0 | 0.320134 | 0.648917 | 0.843448 | LOW_SUPPORT |
| class3_pair0_1 | 6777 | 0.076379 | 0 | 0 | -0.079891 | 0.154092 | 0.801886 | LOW_SUPPORT |
| class3_pair0_2 | 1817 | 0.020478 | 0 | 0 | -0.049372 | 0.338577 | 0.910068 | LOW_SUPPORT |
| class3_pair0_3 | 32874 | 0.370499 | 32874 | 0 | -0.160892 | 0.723636 | 0.927346 | LOW_SUPPORT |
| class3_pair1_0 | 2642 | 0.029776 | 0 | 0 | 0.150659 | 0.303103 | 0.733424 | LOW_SUPPORT |
| class3_pair1_2 | 1699 | 0.019148 | 0 | 0 | -0.050556 | 0.186289 | 0.849791 | LOW_SUPPORT |
| class3_pair1_3 | 39959 | 0.450349 | 39959 | 0 | -0.181558 | 0.733403 | 0.913653 | LOW_SUPPORT |
| class3_pair2_0 | 356 | 0.004012 | 0 | 0 | 0.181564 | 0.187886 | 0.834735 | LOW_SUPPORT |
| class3_pair2_1 | 684 | 0.007709 | 0 | 0 | 0.214268 | 0.282246 | 0.756871 | LOW_SUPPORT |
| class3_pair2_3 | 1503 | 0.016939 | 1503 | 0 | 0.028445 | 0.569699 | 0.876277 | LOW_SUPPORT |
| class3_pair3_0 | 194 | 0.002186 | 0 | 194 | 0.259194 | 0.058029 | 0.817754 | LOW_SUPPORT |
| class3_pair3_1 | 151 | 0.001702 | 0 | 151 | 0.186662 | 0.085037 | 0.727942 | LOW_SUPPORT |
| class3_pair3_2 | 73 | 0.000823 | 0 | 73 | 0.220451 | 0.309850 | 0.917166 | LOW_SUPPORT |

以上分解区分了family bias、条件预测/置信度偏移、pair组成、context方向错误和统计能力限制；并未识别各机制的独立因果贡献比例。

## 19. Three context sources and independence

```text
ctx_S=mean_j ps(j); ctx_D=mean_j pd(j)
ctx_sym=.5*(ctx_S+ctx_D)
```

| Group | Context | =shallow | =deep | Different n | Different rate |
|---|---|---|---|---|---|
| all | ctx_S | 0.775825 | 0.770787 | 309629 | 0.124894 |
| all | ctx_D | 0.734513 | 0.849525 | 249593 | 0.100677 |
| all | ctx_sym | 0.754160 | 0.837817 | 241241 | 0.097308 |
| hard_disagreement | ctx_S | 0.440061 | 0.418807 | 82944 | 0.141133 |
| hard_disagreement | ctx_D | 0.210068 | 0.695228 | 55658 | 0.094705 |
| hard_disagreement | ctx_sym | 0.273665 | 0.626562 | 58637 | 0.099774 |
| Both_Wrong | ctx_S | 0.530765 | 0.507568 | 161462 | 0.410138 |
| Both_Wrong | ctx_D | 0.496018 | 0.574209 | 155388 | 0.394709 |
| Both_Wrong | ctx_sym | 0.517902 | 0.561547 | 154221 | 0.391745 |
| Top20_Both_Wrong | ctx_S | 0.377202 | 0.310658 | 43557 | 0.467791 |
| Top20_Both_Wrong | ctx_D | 0.288276 | 0.555847 | 34954 | 0.375397 |
| Top20_Both_Wrong | ctx_sym | 0.317091 | 0.466836 | 38849 | 0.417229 |

shallow和deep预测相同时，两列可以重叠。‘第三证据’是提供不同预测/救援的操作性定义，不是统计独立性证明；ctx本身仍来自同一模型的两组概率。

## 20. Both-Wrong third-class rescue

| Group | Context | Targets | Accuracy | mIoU | Different n | Correct third n | Rescue rate | Rescue precision |
|---|---|---|---|---|---|---|---|---|
| Both_Wrong | ctx_S | 393677 | 0.337558 | 0.150815 | 161462 | 132889 | 0.337558 | 0.823036 |
| Both_Wrong | ctx_D | 393677 | 0.335542 | 0.202953 | 155388 | 132095 | 0.335542 | 0.850098 |
| Both_Wrong | ctx_sym | 393677 | 0.336057 | 0.186479 | 154221 | 132298 | 0.336057 | 0.857847 |
| Top20_Both_Wrong | ctx_S | 93112 | 0.386202 | 0.177209 | 43557 | 35960 | 0.386202 | 0.825585 |
| Top20_Both_Wrong | ctx_D | 93112 | 0.326736 | 0.188383 | 34954 | 30423 | 0.326736 | 0.870372 |
| Top20_Both_Wrong | ctx_sym | 93112 | 0.361758 | 0.197772 | 38849 | 33684 | 0.361758 | 0.867049 |

ctx_sym Both-Wrong accuracy=33.6057%，CI [32.3216%, 34.8766%]。由于两个原假设都错，context正确就必然不同于两者，因此accuracy与ThirdClassRescueRate严格恒等，不能算两份独立证据。

## 21. One-correct intrusion / harm

| Context | Targets | Different n | Third wrong n | Intrusion rate | Harm rate |
|---|---|---|---|---|---|
| ctx_S | 497629 | 48534 | 48534 | 0.097530 | 0.097530 |
| ctx_D | 497629 | 20840 | 20840 | 0.041879 | 0.041879 |
| ctx_sym | 497629 | 22523 | 22523 | 0.045261 | 0.045261 |

ctx_sym harm=4.5261%，CI [4.2627%, 4.7964%]。Exactly-one-correct时第三类必错，intrusion与harm严格相等。它不衡量context改选另一个错误原候选造成的全部损失。

## 22. Four semantic states / evidence roles

| State | Targets | shallow | deep | fixed_average | ctx_S | ctx_D | ctx_sym | anchor_old | anchor_sym |
|---|---|---|---|---|---|---|---|---|---|
| Deep_Win | 314730 | 0 | 1 | 0.916957 | 0.584117 | 0.897385 | 0.857062 | 0.820656 | 0.920179 |
| Shallow_Win | 182899 | 1 | 0 | 0.225955 | 0.723115 | 0.492791 | 0.604170 | 0.460976 | 0.372862 |
| Both_Wrong | 393677 | 0 | 0 | 0.003338 | 0.337558 | 0.335542 | 0.336057 | 0.003917 | 0.003650 |
| Both_Correct | 1587837 | 1 | 1 | 1 | 0.937252 | 0.953796 | 0.959381 | 1 | 1 |

文件名沿用规格three_state_roles，但实际完整列出四种状态：Both-Correct、Deep-Only、Shallow-Only、Neither-Correct。Context有救援功能，也可能损害本来正确的状态，不能仅用Neither-Correct证据替代整体评估。

## 23. Context-winner diagnostic

Delta_ctx=JS(ctx,ps)-JS(ctx,pd)，正值更接近deep，零值归shallow。

| Score | Mean score | Pooled AUC | Image AUC | BA | Deep-Win recall | Shallow-Win recall |
|---|---|---|---|---|---|---|
| ctx_S | -0.103073 | 0.626265 | 0.723261 | 0.575218 | 0.214968 | 0.935467 |
| ctx_D | 0.014302 | 0.808293 | 0.778172 | 0.725021 | 0.815763 | 0.634279 |
| ctx_sym | -0.055341 | 0.748658 | 0.764111 | 0.665480 | 0.489223 | 0.841738 |

JS(mean context,hypothesis)不等于mean JS(neighbor,hypothesis)；这些是辅助诊断，不替换Delta_sym primary。

## 24. Boundary / interior

| Group | B_S | B_D | B_family | Old image AUC | Sym image AUC | Old BA | Sym BA | ctx_sym BW rescue |
|---|---|---|---|---|---|---|---|---|
| boundary | 0.158819 | 0.033166 | 0.095993 | 0.530532 | 0.552859 | 0.504611 | 0.535295 | 0.191541 |
| interior | 0.116684 | 0.068062 | 0.092373 | 0.746441 | 0.801592 | 0.604087 | 0.736479 | 0.370000 |

直接复用Phase-2B1 frozen boundary：224-grid FG-FG transition的欧氏距离<=7px，再nearest投影到28；不重新定义边界或用于权重。

## 25. Frozen conflict Q1–Q5

| Group | B_S | B_D | B_family | Old mean | Sym mean | Old image AUC | Sym image AUC | Old BA | Sym BA | Neither n | Targets | ctx_sym BW rescue |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | -0.025194 | 0.046669 | 0.010738 | -0.022376 | -0.011639 | 0.786508 | 0.827315 | 0.682585 | 0.740034 | 39647 | 495830 | 0.328020 |
| Q2 | 0.047433 | 0.028758 | 0.038096 | -0.069077 | -0.030981 | 0.758352 | 0.771621 | 0.681300 | 0.744028 | 70207 | 495828 | 0.353982 |
| Q3 | 0.110190 | 0.031727 | 0.070958 | -0.110589 | -0.039630 | 0.735576 | 0.759354 | 0.653167 | 0.725816 | 91056 | 495829 | 0.340505 |
| Q4 | 0.176230 | 0.054925 | 0.115578 | -0.157019 | -0.041441 | 0.733233 | 0.768535 | 0.615809 | 0.719075 | 112621 | 495828 | 0.321148 |
| Q5 | 0.291854 | 0.164076 | 0.227965 | -0.262367 | -0.034403 | 0.760018 | 0.790739 | 0.588089 | 0.716254 | 80146 | 495828 | 0.340229 |

Neither-Correct prevalence为表中Neither n/Targets。所有分位边界沿用Phase-2B1，ties归较低组；不按本轮结果重新分箱，Q5好不能单独推进。

## 26. Paired image bootstrap / independent verification

| Metric | Observed | 95% low | 95% high |
|---|---|---|---|
| old_image_auroc | 0.734850 | 0.726086 | 0.743701 |
| old_balanced_accuracy | 0.593973 | 0.587891 | 0.600202 |
| old_deep_win_recall | 0.261653 | 0.250529 | 0.273078 |
| old_shallow_win_recall | 0.926293 | 0.922011 | 0.930392 |
| sym_image_auroc | 0.784842 | 0.777130 | 0.792815 |
| sym_balanced_accuracy | 0.715627 | 0.707953 | 0.723314 |
| sym_deep_win_recall | 0.640314 | 0.628538 | 0.652104 |
| sym_shallow_win_recall | 0.790939 | 0.781909 | 0.799878 |
| sym_minus_old_image_auroc | 0.049991 | 0.045288 | 0.054654 |
| sym_minus_old_balanced_accuracy | 0.121654 | 0.115811 | 0.127393 |
| sym_minus_old_deep_win_recall | 0.378661 | 0.368509 | 0.388615 |
| sym_minus_old_shallow_win_recall | -0.135353 | -0.142160 | -0.128752 |
| class2_sym_image_auroc | 0.451437 | 0.425590 | 0.477130 |
| class3_sym_image_auroc | 0.340851 | 0.278647 | 0.405788 |
| anchor_sym_minus_fixed_average_accuracy | 0.011297 | 0.010716 | 0.011902 |
| anchor_sym_minus_fixed_average_miou | 0.019651 | 0.018105 | 0.021300 |
| anchor_sym_minus_anchor_old_accuracy | 0.006092 | 0.004723 | 0.007518 |
| anchor_sym_minus_anchor_old_miou | 0.024084 | 0.020581 | 0.027441 |
| ctx_sym_Both_Wrong_accuracy | 0.336057 | 0.323216 | 0.348766 |
| ctx_sym_ThirdClassRescueRate | 0.336057 | 0.323216 | 0.348766 |
| ctx_sym_ThirdClassHarmRate | 0.045261 | 0.042627 | 0.047964 |
| mean_Delta_old | -0.124286 | -0.127176 | -0.121445 |
| mean_Delta_sym | -0.031619 | -0.033141 | -0.030096 |
| BiasShrink | 0.745595 | 0.735393 | 0.756009 |

10,000次、seed42、全部3418图像作为cluster成组重采样；AUROC按image mean，BA/语义指标按confusion重加，bias按sum/count。45个分组的B_S/B_D/B_family CI另见第6节及CSV；全部159列bootstrap replicate随交付保存。

独立验证PASS：不导入审计helper的NumPy/SciPy重算116212个图像-AUROC组合；全部45组原生confusion和支持度分布重算；9个固定真实位置显式枚举邻居。支持度最大误差9.01e-08、context误差6.23e-08、GT composition误差2.91e-08。另用与分析器不同的索引求和方式复算32个bootstrap/159列，最大差1.33e-15；全部10,000次CI分位点一致。23项单元测试及真实图像smoke通过。

## 27. Frozen Gate A/B/C/D

| Gate | Result |
|---|---|
| A | PASS |
| B | PASS |
| C | UNDERPOWERED |
| D | PASS |

A：两种family bias mean和CI下界都>0。B：imageAUC>=.70、BA>=.62、双方recall>=.55、全体mean Delta绝对值缩小超过一半，全部满足。

C：class2点估计0.451437>=.45，按原门槛PASS；class3只有418个Shallow-Win，固定为UNDERPOWERED。按用户确认的汇总规则：无powered FAIL且存在underpowered => C UNDERPOWERED。没有以置信区间或另一个AUROC定义事后替换原门槛。D：ctx_sym满足accuracy/CI/rescue/harm条件。

## 28. Strong-signal flags

STRONG_SYMMETRY_SIGNAL = TRUE

STRONG_THIRD_EVIDENCE_SIGNAL = TRUE

THIRD_EVIDENCE_SUPPORTED = TRUE

强信号是预注册辅助标志，不覆盖class3欠充分证据、不解锁训练。

## 29. Scientific interpretation and limits

1. **同源偏差得到支持。** shallow-source和deep-source的绝对支持方向相反，两个family bias均显著为正；对称化将全局零点偏移缩小约74.56%，并改善固定零阈值的双向召回。不能把这一比例解读为训练性能差距的解释率。
2. **对称化有机制价值，但类别反转尚未完全解决。** 总体image AUROC、BA和anchor诊断均改善；class2的CI仍低于0.5，class3的image AUROC仍偏低且Shallow-Win不足。pooled class2/3 AUROC已超过0.5，但不能据此替换冻结的image-balanced primary；跨图混合和图内排序是不同问题。
3. **context可提供候选之外的语义救援。** ctx_sym在Both-Wrong纠正33.61%，one-correct第三类侵入4.53%，满足本阶段第三证据门槛。这不等于统计独立、更不等于context-only就是最终第三分支。
4. **本轮不支持直接进入模型训练。** 所有提升来自固定概率缓存上的机制探针；未证明官方final-CAM或训练后mIoU会提高。未挑选更好的context source、prediction pair、阈值或类别规则。原Phase-2B1 NOGO没有改写。

按实施交付流程保留冻结合同、独立分支、可复算证据、完整报告和运行说明；PR仅供审核，停止于当前审计。

## 30. Exact decision / STOP

A/B通过，C UNDERPOWERED，D通过。按结果前确认的决策优先级，应使用SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED，而不是BIAS_RESOLVED。不启动Phase-2B2、不新增模型、不训练、不访问test。所有数值证据和限制已经交付，等待用户另行决定。

DECISION = SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED
