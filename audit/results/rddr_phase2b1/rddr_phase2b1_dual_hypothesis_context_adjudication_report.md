# RDDR Phase-2B1 Dual-Hypothesis Context Adjudication Audit

**最终判定：`RDDR_PHASE2B1_NOGO`。** Gate A/B/C/D = PASS / FAIL / FAIL / PASS。

Delta 有排序信号：image AUROC=0.734850。但固定 sign 的 Deep-Win recall=26.1653%，BA=59.3973%；anchor 相对 FixedAvg 的 accuracy=+0.5205pp，mIoU=-0.4433pp。不能将排序能力、accuracy提升或安全改善替代方向召回率及mIoU门槛。

所有表中比例默认0–1，pp=百分点；NLL/Brier不是百分比。统计基于原生28-grid，**不是官方224/TTA final-CAM指标**。

## 1. Provenance / frozen evidence

- Pure A0：`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Extraction commit：`82e10afe85af1bda69a1f0e0f8de003110178d08`
- Analysis commit：`abc8ff28471aae94fda3a422932b7032f3b4ef9d`
- Checkpoint：`/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- Checkpoint SHA256：`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Native observation SHA256：`767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a`
- Sufficient statistics SHA256：`56eed74d93172456a1149a06691ce7eb37e2aec1cc2acf1366e0cc236fd1671e`

Phase0已证明冲突信号存在；Phase1 feature disposal、Phase2A receiver suppression和Phase2B0固定deep-anchor路线均未通过各自冻结门槛。本轮从A0独立开始，不继承旧模型改造；不删除旧实验，不改变此前NOGO。

## 2. Exact commands / environment / resources

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b1_dual_hypothesis_audit.py --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --population-cache /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/diagnostics/frozen_phase0_populations --phase0-results /home/duyanhong/experiments/RDDR_PHASE0_586f402/formal --output /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/analyze_rddr_phase2b1.py --input /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1 --output /home/duyanhong/experiments/RDDR_PHASE2B1/report_r1
```

NVIDIA GeForce RTX 5090 D v2；Python 3.10.20 / PyTorch 2.11.0+cu128 / NumPy 1.23.5。batch1，BF16前向、FP32 softmax/support。benchmark=False，matmul=none，conv=tf32，与冻结Phase0 backend一致。

提取总耗时32.29s（含观测压缩落盘），forward/support=13.92s；离线统计和10000 bootstrap=16.88s。CUDA峰值allocated=0.552GiB，reserved=0.629GiB；原生概率观测缓存=229.69MiB。缓存不是模型checkpoint，不保存全数据集pair张量。

## 3. Tensor / preprocessing / GT contract

未修改的A0 Net.forward加只读hook。F28_raw=[1,512,28,28]、Ddeep=[1,4096,28,28]，Ls=ic1(F28_raw)、Ld=fc8(Ddeep)，ps/pd=softmax(logits.float())。不对logits施加ReLU、CAM归一化、presence或TTA；原网络内部激活不改。图像224/bilinear/ImageNet normalization，eval、requires_grad=False、no_grad。

GT及历史mask从224 nearest到28；只在metric target中保留0–3，4/255不计入metric。GT不进入support、anchor或context；包括GT背景位置在内的所有合法邻居都参与无监督support。没有background预测自动修正。

## 4. Conflict / frozen groups

q=clip(JS(ps,pd)/ln2,0,1)。Hard disagreement=argmax(ps)!=argmax(pd)。Top20严格复用历史mask，不重新选择。

| Group | 224 count | 28 count |
|---|---|---|
| all | 158639345 | 2479143 |
| Corrected_by_CH | 19934592 | 305318 |
| Still_Wrong | 24262754 | 378130 |
| Harmed_by_CH | 4443224 | 69617 |
| Stable_Correct | 109998775 | 1726078 |
| Top20 | 31727873 | 485451 |
| Bottom80 | 126911472 | 1993692 |

全部3418缓存SHA及逐图历史人数通过；重新提取native q最大误差=0.0。原始历史逐像素文件没有保留，本轮复用Phase2A按原代码/backend重建、逐图人数exact的immutable cache；不声称与不存在的原始像素哈希比较。

## 5. Neighborhood

固定15×15/radius7，仅图内邻居，排除self。角落63、中部224个source。无距离权重、Top-k或邻居softmax。支持度用合法邻居数除法，不把padding计入分母。

## 6. Two full-distribution hypotheses

hS=ps(i)，hD=pd(i)，e_j=ps(j)。四类完整概率分布；不one-hot，不预设deep正确，不乘source/target reliability。

## 7. Support equations and measurements

```text
cS_ij=clip(1-JS(ps_i,ps_j)/ln2,0,1)
cD_ij=clip(1-JS(pd_i,ps_j)/ln2,0,1)
SS_i=mean_j(cS_ij); SD_i=mean_j(cD_ij)
```

自然对数、epsilon=1e-8，epsilon位置与Phase0逐项一致；temperature=1。

| Group | Mean SS | Mean SD | Mean Delta | Mean wD |
|---|---|---|---|---|
| all | 0.861603 | 0.737317 | -0.124286 | 0.455310 |
| Deep_Win | 0.814649 | 0.644642 | -0.170007 | 0.435536 |
| Shallow_Win | 0.862945 | 0.574356 | -0.288588 | 0.388061 |
| Both_Wrong | 0.805757 | 0.619016 | -0.186741 | 0.427597 |

## 8. Primary Delta and sign

Delta=SD-SS。**Delta>0选deep；否则选shallow，包括Delta=0。** 不调阈值、不改变符号。Eligible winner population中零值tie=0。离线重算wd/anchor/FixedAvg相对CUDA原生观测误差：{'wd_cpu_cuda_max_difference': 0.0, 'anchor_cpu_cuda_max_difference': 0.0, 'fixed_average_max_difference': 0.0}。主指标使用保存的CUDA预测概率。

## 9. Adjudication population

Foreground=2,479,143；hard disagreement=587,701；exactly-one-correct conflict=497,629。只有最后一组形成二分类裁决目标。Agreement、both-wrong不会被悄悄作为Shallow-Win负例。

## 10. Deep-Win / Shallow-Win prevalence

Deep-Win=314,730，Shallow-Win=182,899；Deep-Win prevalence=0.632459。Y=1表示deep正确/shallow错误，Y=0相反。GT仅构建审计标签，不提供给support。

## 11. Exact pooled / image-balanced AUROC and AP

| Group | Pooled AUROC | Image AUROC | Pooled AP | Image AP | Prevalence | AUC images |
|---|---|---|---|---|---|---|
| all | 0.638180 | 0.734850 | 0.765099 | 0.822545 | 0.632459 | 3180 |
| Top20 | 0.658978 | 0.752728 | 0.806289 | 0.852752 | 0.680950 | 2915 |
| hard_disagreement | 0.638180 | 0.734850 | 0.765099 | 0.822545 | 0.632459 | 3180 |

使用精确排序和tie处理，不使用4096bin近似。AUROC缺少任一标签则NA；AP无正例为NA，全正例为1。Primary是image-balanced AUROC，不能结果出来后换成pooled。

## 12. Primary AUROC confidence interval

| AUROC | 95% low | 95% high | Eligible images |
|---|---|---|---|
| 0.734850 | 0.726086 | 0.743701 | 3180 |

在全部3418图上重采样；每次忽略该指标未定义的图像。无正负两类图像没有人为赋AUROC=0.5。

## 13. Fixed sign decision accuracy / BA / F1

| Group | Winner targets | Sign accuracy | Pooled BA | Macro F1 | Deep-Win recall | Shallow-Win recall |
|---|---|---|---|---|---|---|
| all | 497629 | 0.505935 | 0.593973 | 0.490333 | 0.261653 | 0.926293 |
| Top20 | 323902 | 0.451251 | 0.580706 | 0.439037 | 0.222995 | 0.938417 |
| Bottom80 | 173727 | 0.607891 | 0.631370 | 0.586771 | 0.352197 | 0.910543 |

主BA=0.593973，95%CI=[0.587891,0.600202]。辅助image-balanced BA=0.663067；按确认合同，不能替代主pooled BA。

## 14. Winner recalls / strength diagnostic

Deep-Win recall=26.1653%（门槛55%）；Shallow-Win recall=92.6293%（门槛55%）。Gate B要求BA和双方recall同时满足，不以总体accuracy或AUC替代。

Strength=abs(Delta)，在全部FG一次冻结higher-quantile边界：[0.027495861053466797, 0.06410473585128784, 0.1186361312866211, 0.22104030847549438]。

| Group | Winner targets | Sign accuracy | Pooled BA | Macro F1 | Deep-Win recall | Shallow-Win recall |
|---|---|---|---|---|---|---|
| Strength1 | 36852 | 0.505861 | 0.537075 | 0.476511 | 0.480022 | 0.594128 |
| Strength2 | 47962 | 0.519724 | 0.597701 | 0.511180 | 0.438025 | 0.757376 |
| Strength3 | 65927 | 0.519453 | 0.615621 | 0.519141 | 0.384713 | 0.846530 |
| Strength4 | 101249 | 0.521901 | 0.615509 | 0.514298 | 0.304898 | 0.926120 |
| Strength5 | 245639 | 0.493045 | 0.547093 | 0.412099 | 0.108598 | 0.985587 |

Strength只观察margin与正确性的关系，不选择运行子集或修改权重。

## 15. Contextual anchor

```text
wD=SD/(SS+SD+eps); wS=1-wD
p_anchor=wS*ps+wD*pd
p_fixed=.5*ps+.5*pd
```

无q-dependent beta、learned gate、temperature或搜索。Context-only为mean_j(ps_j)，不参与anchor公式。

## 16. Foreground semantic utility

| Estimator | Accuracy | mIoU | Dice | NLL | Brier |
|---|---|---|---|---|---|
| shallow | 0.714253 | 0.436349 | 0.581730 | 0.784837 | 0.406249 |
| deep | 0.767429 | 0.566286 | 0.717333 | 1.784933 | 0.409590 |
| fixed_average | 0.774087 | 0.573520 | 0.723531 | 0.681292 | 0.341498 |
| anchor | 0.779292 | 0.569087 | 0.718405 | 0.669878 | 0.334832 |
| context_only | 0.781395 | 0.489692 | 0.614260 | 0.643973 | 0.336364 |

Anchor-FixedAvg accuracy=+0.5205pp，mIoU=-0.4433pp。四类mIoU/macroDice从总confusion计算，zero-union/denominator类别为NA并排除，不设为完美。NLL=-log(pGT+eps)，Brier为四类直接平方误差之和再平均。

| Class | Fixed IoU | Anchor IoU | Delta pp |
|---|---|---|---|
| 0 | 0.699399 | 0.701335 | 0.193645 |
| 1 | 0.640964 | 0.651215 | 1.025145 |
| 2 | 0.514291 | 0.513676 | -0.061509 |
| 3 | 0.439427 | 0.410121 | -2.930513 |

原生28-grid weak-logit诊断不能直接解释为官方final-CAM复现增益。

## 17. Conflict-only utility

| Group | FG targets | Image AUROC | Sign BA | Fixed acc | Anchor acc | Acc delta | mIoU delta |
|---|---|---|---|---|---|---|---|
| hard_disagreement | 587701 | 0.734850 | 0.593973 | 0.563611 | 0.585568 | 0.021957 | 0.014339 |
| adjudication | 497629 | 0.734850 | 0.593973 | 0.662986 | 0.688459 | 0.025473 | 0.013751 |

## 18. Frozen Top20 / Bottom80

| Group | FG targets | Image AUROC | Sign BA | Fixed acc | Anchor acc | Acc delta | mIoU delta |
|---|---|---|---|---|---|---|---|
| Top20 | 485451 | 0.752728 | 0.580706 | 0.599801 | 0.612945 | 0.013144 | 0.014558 |
| Bottom80 | 1993692 | 0.729992 | 0.631370 | 0.816524 | 0.819796 | 0.003272 | 0.003659 |

Top20 anchor-fixed accuracy delta=+1.3144pp。Top20 nearest投影后不强制20%，没有按本轮Delta重新取Top20。

## 19. Deep-Correct / Deep-Wrong and hard safety

| Group | Targets | Shallow | Deep | FixedAvg | Anchor | Anchor-Fixed | Anchor-Shallow |
|---|---|---|---|---|---|---|---|
| Deep_Correct | 1902567 | 0.834576 | 1 | 0.986263 | 0.970332 | -0.015931 | 0.135756 |
| Deep_Wrong | 576576 | 0.317216 | 0 | 0.073956 | 0.148903 | 0.074948 | -0.168313 |
| Top20_Deep_Correct | 288998 | 0.236808 | 1 | 0.969387 | 0.869587 | -0.099800 | 0.632779 |
| Top20_Deep_Wrong | 196453 | 0.526034 | 0 | 0.056110 | 0.235405 | 0.179295 | -0.290629 |

Global Deep-Wrong delta=+7.4948pp；Top20 Deep-Wrong=+17.9295pp。

| Deep-Wrong stratum | Targets | Fixed | Anchor | Delta | <=-10pp |
|---|---|---|---|---|---|
| all | 576576 | 0.073956 | 0.148903 | 0.074948 | False |
| Top20 | 196453 | 0.056110 | 0.235405 | 0.179295 | False |
| Bottom80 | 380123 | 0.083178 | 0.104198 | 0.021020 | False |
| hard_disagreement | 272971 | 0.156211 | 0.314517 | 0.158306 | False |
| Q1 | 40025 | 0.005022 | 0.005022 | 0 | False |
| Q2 | 74629 | 0.031744 | 0.032682 | 0.000938 | False |
| Q3 | 107952 | 0.084130 | 0.088651 | 0.004521 | False |
| Q4 | 153281 | 0.115970 | 0.133011 | 0.017041 | False |
| Q5 | 200689 | 0.065838 | 0.265366 | 0.199528 | False |
| boundary | 101269 | 0.041572 | 0.084369 | 0.042797 | False |
| interior | 475307 | 0.080855 | 0.162653 | 0.081798 | False |
| class0 | 200257 | 0.097400 | 0.183874 | 0.086474 | False |
| class1 | 215891 | 0.093909 | 0.213237 | 0.119329 | False |
| class2 | 113319 | 0.024780 | 0.025900 | 0.001121 | False |
| class3 | 47109 | 0.001146 | 0.001295 | 0.000149 | False |

所有15个分层提前固定，下降参照均为FixedAvg。检查汇总分层，不按单图/像素或事后子集触发。D还要求全体>=-2pp、Top20>=-3pp。空分层为NA并报告覆盖率，不伪造通过值。

## 20. Symmetric shallow-correct / shallow-wrong audit

| Group | Targets | Shallow | Deep | FixedAvg | Anchor | Anchor-Fixed | Anchor-Shallow |
|---|---|---|---|---|---|---|---|
| Shallow_Correct | 1770736 | 1 | 0.896710 | 0.920049 | 0.944324 | 0.024275 | -0.055676 |
| Shallow_Wrong | 708407 | 0 | 0.444279 | 0.409239 | 0.366776 | -0.042463 | 0.366776 |
| Top20_Shallow_Correct | 171778 | 1 | 0.398404 | 0.460018 | 0.664392 | 0.204374 | -0.335608 |
| Top20_Shallow_Wrong | 313673 | 0 | 0.703156 | 0.676351 | 0.584771 | -0.091579 | 0.584771 |

安全改善不等于没有shallow bias；同时观察浅层正确与浅层错误的代价。

## 21. Both-wrong / third-class recovery

| Estimator | Accuracy | mIoU | Dice | NLL | Brier |
|---|---|---|---|---|---|
| shallow | 0 | 0 | 0 | 2.384482 | 1.239530 |
| deep | 0 | 0 | 0 | 8.614239 | 1.757884 |
| fixed_average | 0.003338 | 0.001161 | 0.002317 | 2.834425 | 1.426194 |
| anchor | 0.003917 | 0.001337 | 0.002667 | 2.720414 | 1.373477 |
| context_only | 0.337558 | 0.150815 | 0.247417 | 1.287542 | 0.737902 |

| Group | Targets | Different from both | Correct third class |
|---|---|---|---|
| Both_Wrong | 393677 | 1829 | 1542 |

该纠正仅secondary signal，不改变任何门槛。

## 22. Context consensus diagnostic

| Group | JS(ctx,ps) | JS(ctx,pd) | Fraction closer to deep |
|---|---|---|---|
| all | 0.052440 | 0.155513 | 0.065613 |
| Top20 | 0.062740 | 0.272500 | 0.112106 |
| Deep_Win | 0.072877 | 0.211991 | 0.214968 |
| Shallow_Win | 0.040954 | 0.256842 | 0.064533 |
| Both_Wrong | 0.078320 | 0.226049 | 0.065353 |

JS(mean_j ps_j,hypothesis)与mean_j JS(ps_j,hypothesis)并不相等；共识只是解释性参照，未替换primary mean support。

## 23. Frozen HFRM transition groups

| Group | Mean Delta | Mean SS | Mean SD | Fixed acc | Anchor acc | Delta acc |
|---|---|---|---|---|---|---|
| Corrected_by_CH | -0.181179 | 0.808977 | 0.627798 | 0.690267 | 0.653090 | -0.037178 |
| Still_Wrong | -0.189682 | 0.815008 | 0.625326 | 0.171647 | 0.126755 | -0.044892 |
| Harmed_by_CH | -0.204260 | 0.818661 | 0.614401 | 0.415818 | 0.458983 | 0.043165 |
| Stable_Correct | -0.096670 | 0.882851 | 0.786181 | 0.935339 | 0.957485 | 0.022146 |

历史by-CH命名实际是raw→full HFRM transition，包括semantic veto+context，**不是CH-only因果干预**。

## 24. Frozen q quintiles

沿用并校验Phase2B0 native q边界：[0.020935675129294395, 0.072734534740448, 0.163648784160614, 0.3369627296924591]。tie归较低分组。

| Bin | Hard prevalence | Deep-Win n | Shallow-Win n | Image AUC | Sign BA | Accuracy delta | mIoU delta |
|---|---|---|---|---|---|---|---|
| Q1 | 0.001722 | 348 | 378 | 0.786508 | 0.682585 | 0.000002 | 0.000002 |
| Q2 | 0.020372 | 4105 | 4422 | 0.758352 | 0.681300 | 0.000147 | 0.000115 |
| Q3 | 0.075937 | 14695 | 16896 | 0.735576 | 0.653167 | 0.000873 | 0.000749 |
| Q4 | 0.205458 | 43330 | 40660 | 0.733233 | 0.615809 | 0.004167 | 0.004660 |
| Q5 | 0.881802 | 252252 | 120543 | 0.760018 | 0.588089 | 0.020836 | 0.021113 |

## 25. Boundary / interior

| Group | FG targets | Image AUROC | Sign BA | Fixed acc | Anchor acc | Acc delta | mIoU delta |
|---|---|---|---|---|---|---|---|
| boundary | 201144 | 0.530532 | 0.504611 | 0.502302 | 0.510062 | 0.007761 | -0.003383 |
| interior | 2277999 | 0.746441 | 0.604087 | 0.798085 | 0.803064 | 0.004979 | -0.004543 |

| Group | Deep-Wrong n | Deep-Wrong delta |
|---|---|---|
| boundary | 101269 | 0.042797 |
| interior | 475307 | 0.081798 |

沿用fullres FG-FG 8-neighbor transition欧氏距离<=7px的边界，先224构造再nearest投影；未进入support计算。

## 26. Per-class adjudication

| Class | Deep-Win n | Shallow-Win n | Pooled AUC | Image AUC | Sign BA | Fixed acc | Anchor acc | Delta |
|---|---|---|---|---|---|---|---|---|
| class0 | 43802 | 88423 | 0.866710 | 0.825913 | 0.736861 | 0.809624 | 0.827660 | 0.018036 |
| class1 | 121742 | 82309 | 0.827822 | 0.781674 | 0.652146 | 0.801496 | 0.824435 | 0.022940 |
| class2 | 74850 | 11749 | 0.319921 | 0.275458 | 0.412252 | 0.633501 | 0.617489 | -0.016012 |
| class3 | 74336 | 418 | 0.051692 | 0.071672 | 0.408576 | 0.654404 | 0.494681 | -0.159722 |

## 27. Support calibration

| Lower inclusive | Upper | Upper inclusive | Winner n | Mean wD | P(Deep-Win) |
|---|---|---|---|---|---|
| 0 | 0.200000 | False | 13588 | 0.161526 | 0.090668 |
| 0.200000 | 0.400000 | False | 171148 | 0.322266 | 0.580942 |
| 0.400000 | 0.600000 | False | 302689 | 0.476113 | 0.676916 |
| 0.600000 | 0.800000 | False | 10148 | 0.644653 | 0.898995 |
| 0.800000 | 1 | True | 56 | 0.824709 | 0.946429 |

固定五桶，空桶NA。wD是support ratio，不自动等价于校准后的P(deep正确)。无温度、偏移或分桶后重校准。

## 28. Echo diagnostics

| Group | Anchor=deep | Anchor=shallow | Neither n |
|---|---|---|---|
| all | 0.928773 | 0.832539 | 4042 |
| Top20 | 0.780050 | 0.428612 | 1550 |
| hard_disagreement | 0.699536 | 0.293586 | 4042 |
| Deep_Correct | 0.970332 | 0.863563 | 1295 |
| Deep_Wrong | 0.791635 | 0.730166 | 2747 |

shallow/deep相同的像素同时计入两列，故全体两列不应强制相加为1；hard disagreement最能区分echo倾向。

## 29. Winner oracle reference

Exactly-one-correct conflict上的oracle winner accuracy=1.000000；固定sign实际accuracy=0.505935，距oracle=49.4065pp。Oracle只用于诊断，不调公式、不向anchor传递GT。

## 30. Paired bootstrap / independent verification

| Metric | Observed | 95% low | 95% high | Eligible images |
|---|---|---|---|---|
| adjudication_image_auroc | 0.734850 | 0.726086 | 0.743701 | 3180 |
| sign_balanced_accuracy | 0.593973 | 0.587891 | 0.600202 | 3410 |
| anchor_fixed_accuracy_delta | 0.005205 | 0.003580 | 0.006855 | 3416 |
| anchor_fixed_miou_delta | -0.004433 | -0.008295 | -0.000372 | 3416 |
| Top20_anchor_fixed_accuracy_delta | 0.013144 | 0.005171 | 0.021391 | 3405 |
| Deep_Wrong_anchor_fixed_accuracy_delta | 0.074948 | 0.071495 | 0.078578 | 3318 |
| Top20_Deep_Wrong_anchor_fixed_accuracy_delta | 0.179295 | 0.171577 | 0.187051 | 3164 |

10000次、seed42、相同图像索引配对；AUROC按image mean，BA重加2×2confusion，accuracy/mIoU重加4×4confusion，没有target/pair-level naive bootstrap。所有replicate均保留。

独立NumPy/SciPy验证：PASS；3418图AUROC用rankdata单独重算，9个固定真实位置用显式邻居重算support（max error=1.1920929e-07）；另复算32个完整bootstrap replicate、全部10000次CI分位点和15个安全分层。21项单元测试日志及独立verification.json随交付保存。

## 31. Frozen Gate A/B/C/D

| Gate | Result |
|---|---|
| A | PASS |
| B | FAIL |
| C | FAIL |
| D | PASS |

A: imageAUC>=.65且CI下界>.5。B: pooledBA>=.60且双方recall>=.55。C: accuracy、mIoU都提升，且至少一项差值CI下界>0。D:全体/Top20 Deep-Wrong满足-2/-3pp容忍，所有预注册安全分层不触及-10pp。BA的CI跨0.60不会替代其point门槛，更不能替代Deep-Win recall门槛。

## 32. STRONG_SIGNAL

STRONG_SIGNAL = FALSE。需要image AUROC>=.70、anchor-fixed mIoU>=+1pp、全体Deep-Wrong delta>=0同时满足。它是辅助标志，不能覆盖硬门槛。

## 33. Scientific interpretation / boundaries

本轮应分清三个问题：Delta能否排序winner、固定零阈值能否给出可靠双向判断、支持比值融合能否提升语义指标。本次AUC=0.7349，支持局部上下文包含winner排序信息；但Deep-Win recall=26.17%，固定sign明显偏向shallow。AUROC对单调平移不敏感，因此高AUC本身不保证零阈值决策有效；这里仅解释差异，不执行任何平移/调参。

Anchor准确率提高0.5205pp而macro mIoU变化-0.4433pp，不能宣称整体分割质量提升。全局class3 IoU差为-2.9305pp，类别分布与macro/pixel加权的不同解释了两种指标不能互相替代；不据此新增class规则。Top20/Bottom80和q分位组分别重算的mIoU均可能提升，而合并后mIoU下降：各组mIoU不能按样本量平均还原整体mIoU，每类union及其权重也随预测改变。完整confusion可逐项相加，分层/总体数字已经独立复核，不用最佳分组替代全体结果。

排序信号并非跨类普适：class2/class3的image AUROC分别为0.2755/0.0717，均低于0.5；class3的Shallow-Win仅418个，需保留不平衡背景，不能只引用整体AUC。Strength最大一组也未呈现更高sign正确率，abs(Delta)不等于已校准置信度。

还需区别hard sign和soft anchor：前者偏向shallow，后者仍可能更多echo deep，因为anchor融合完整概率分布而非按sign硬切换，两者预测不必一致。这里观察两者局限，不进行阈值平移或重新校准。

Deep-Wrong安全明显改善（全体+7.4948pp，Top20 +17.9295pp），但这只说明相对FixedAvg没有重现上轮的deep-following灾难，不能抵消B/C失败。所有class/Top20/Strength/consensus/oracle结果仅解释，不替换primary，不搜索阈值/温度/窗口/权重，不追加训练。

本轮按研究项目实施交付流程保存冻结合同、独立A0分支、可运行命令、测试、CSV/JSON及完整报告；旧实验及checkpoint不删除不覆盖。

## 34. Exact decision / STOP

按用户在看结果前批准的优先级：D失败优先UNSAFE；否则A或B失败为NOGO；否则C失败为FUSION_UTILITY_FAIL；全过才GO。本次A通过、B失败、C失败、D通过，因此使用预先补齐的B失败规则，而非事后选择标签。**不启动Phase2B2训练，不访问test/LUAD或其他seed。报告与PR交付后停止。**

DECISION = RDDR_PHASE2B1_NOGO
