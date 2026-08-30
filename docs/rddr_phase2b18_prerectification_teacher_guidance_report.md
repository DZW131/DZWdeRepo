# RDDR Phase-2B1.8 Pre-Rectification Teacher Guidance & Hierarchy-Safety Audit

完整实验报告｜BCSS validation 3418张｜C0 Full25 seed42｜零更新、零搜索

**结论：A/B/C/D/E = PASS / PASS / FAIL / FAIL / PASS。整体语义收益成立，但错误像素纠正能力与层级安全未达标，不进入 Full25。**

本轮的进展是：teacher→raw 的 all/Top20 梯度收益均为正，四类Mean_dM也全部为正。主要风险是 Shallow-Win（浅层正确、深层错误）HHCR=96.2258%，teacher在该组硬标签正确率仅37.2862%。Raw-Wrong BenefitRate=60.6942%，低于70%门限。

这不是一次训练结果：optimizer构建/step均为0，checkpoint、BN、原推理不变；未访问test/LUAD。native28 teacher mIoU提升不能当作最终融合分割mIoU的训练提升。所有rate/AUC默认0–1，pp为百分点；dM/dQ为单位步长负梯度的局部导数，不是预测翻转率或真实长期collapse发生率。

## 1. Provenance / SHA / commands

纯A0：`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`；分支`feature/rddr-phase2b18-prerect-guidance`，PR目标`baseline/official-a0`。实际GPU执行commit：`56269a0bc292d8bc1c28e382c0980fcd8072ad4d`；独立复核commit：`df1fbb9306403ab311e67e2881d3c3c423bd76ea`。原网络、训练、推理、metric源码未改，仅新增tools/tests/docs/results。

用户批准的[执行合同](rddr_phase2b18_contract.md)在新结果产生前提交。输入规格SHA256：`cc4f8588fdf04962ed447511144f6d03c2ffb7a6c7185b78c4ff5ac6b11553ca`。

| asset | path | SHA256 |
| --- | --- | --- |
| native | /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz | 767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a |
| derived | /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz | 237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514 |
| previous | /home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1/rddr_phase2b16_gradient_observations.npz | 5ab5a048e932f27726cea7507685354f984432fb66b542b87b805cea24a72bd5 |
| checkpoint | /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth | 509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579 |

新观察NPZ SHA256：`740b2e80c9182e701509f5ed7a6fab4c8f1b6ddcd585df418ccefa9f288d8e52`，保留在`/home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1/rddr_phase2b18_observations.npz`。不上传权重或大缓存到Git。正式统计与验证目录为`/home/duyanhong/experiments/RDDR_PHASE2B18/report_r1`。

环境：NVIDIA GeForce RTX 5090 D v2，PyTorch `2.11.0+cu128`，NumPy `1.23.5`；BF16 forward/backward; FP32 loss/logit/q gradients; FP64 statistics。原环境未升级。工作目录`/home/duyanhong/DZWdeRepo-rddr-phase2b18`。

已执行的实际命令：

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b18_audit.py \
  --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz \
  --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz \
  --previous /home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1/rddr_phase2b16_gradient_observations.npz \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --val-images /home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img \
  --output /home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1
```

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/analyze_rddr_phase2b18.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1 \
  --output /home/duyanhong/experiments/RDDR_PHASE2B18/report_r1
```

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/verify_rddr_phase2b18.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1 \
  --report /home/duyanhong/experiments/RDDR_PHASE2B18/report_r1
```

## 2. Frozen historical evidence

Phase2B1.5 symmetric contextual ranking相对旧实现：image-AUROC 0.734850→0.784842，BA 0.593973→0.715627。Sym teacher相对FixedAvg native28 mIoU +1.9651pp，95% CI[+1.8105,+2.1300]pp。

Phase2B1.6 teacher→rect无条件KL为`TEACHER_SIGNAL_PRESENT_GRADIENT_UNSAFE`；Phase2B1.7 acceptance为`CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED`。这两条路线保持No-Go，本轮未恢复post-HFRM KL、Δacceptance、feature disposal或receiver suppression。

## 3. Raw / teacher / rect 的相对位置

| prediction | accuracy | native28_mIoU |
| --- | --- | --- |
| raw | 0.714253 | 0.436349 |
| symmetric teacher | 0.785383 | 0.593171 |
| post-HFRM rect (frozen Phase16) | 0.817788 | 0.637895 |

raw < teacher < rect；因此移动监督位置有明确动机，但“teacher比raw强”不自动证明每种纠正都安全。rect行是既有诊断参考，本轮没有额外rect-guidance probe，也没有重新跑官方全量最终融合分割指标。

## 4. Tensor contract 与冻结重放

```text
F28_raw: [B,512,28,28]
L_s_raw, p_s: [B,4,28,28]
Ddeep: [B,4096,28,28]
L_d, p_d, p_teacher: [B,4,28,28]
q: [B,28,28]
```

| quantity | max_abs |
| --- | --- |
| ps | 0 |
| pd | 0 |
| teacher | 0 |
| q | 5.96046448e-08 |
| raw_frozen_head_logits | 0 |
| rect_logits | 0 |

旧native缓存没有raw logits。本轮从真实网络新提取raw logits，验证原ic1与frozen-head两条路径完全一致；softmax得到的ps/pd与旧缓存完全一致，teacher完全一致。q重算最大差5.96046e-8，在预注册1e-7容差内；loss使用冻结缓存q。未使用log(ps)反造logits。

## 5. Frozen-head student formulation

```text
F28_raw = ReLU(bn45(b4_5(...b4(feat56))))
L_s_student = conv2d(F28_raw, ic1.weight.detach(), ic1.bias.detach())
```

读取原HFRM28_1输入，不使用rectified输出作为student。39个批准的上游参数张量临时允许求导：b4/b4_1…b4_5以及bn45（含其BN affine）。所有BN仍eval、buffer不变。该临时梯度审计不改变原训练的BN冻结政策；b3及更早、b5及更深、全部HFRM和primary ic1均不接收梯度。

## 6. PRG loss

```text
L_PRG = sum_i q_i KL(p_teacher_i || softmax(L_s_student)_i) / (sum_i q_i + 1e-8)
```

teacher、q、deep source全部detach；q仅用于强调supervision，不作为接受器或正确性选择器。KL逐类采用 t*(log(t+1e-8)−log(p+1e-8))。系数1；无lambda、threshold、temperature或q指数搜索。主审计batch1、每图全部784位置作为loss分母，包括背景/ignore；GT只进入事后诊断。

## 7. Uraw / FAraw controls

```text
Uraw = mean_i KL(sym_teacher || p_s_student)
FixedAvg = stopgrad(0.5*p_s + 0.5*p_d)
FAraw = sum_i q_i KL(FixedAvg || p_s_student)/(sum_i q_i+eps)
```

| stratum | loss | targets | benefit_rate | harm_rate | zero_rate | mean_dm | median_dm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | Uraw | 2479143 | 0.789280 | 0.210718 | 2.82355637e-06 | 1.59301205e-04 | 1.00336652e-04 |
| all | FAraw | 2479143 | 0.789280 | 0.210718 | 2.01682598e-06 | 2.43160873e-04 | 3.63666368e-05 |
| all | PRG | 2479143 | 0.789280 | 0.210718 | 2.82355637e-06 | 2.83648257e-04 | 3.53718024e-05 |
| Top20 | Uraw | 485451 | 0.719677 | 0.280313 | 1.02997007e-05 | 3.64528230e-04 | 5.39633256e-04 |
| Top20 | FAraw | 485451 | 0.719681 | 0.280313 | 6.17982041e-06 | 7.18043486e-04 | 8.36937746e-04 |
| Top20 | PRG | 485451 | 0.719677 | 0.280313 | 1.02997007e-05 | 8.76922180e-04 | 7.88453559e-04 |
| Raw_Correct | Uraw | 1770736 | 0.862226 | 0.137772 | 1.69421077e-06 | 9.80658876e-05 | 8.85261779e-05 |
| Raw_Correct | FAraw | 1770736 | 0.862226 | 0.137772 | 1.69421077e-06 | 2.68504320e-05 | 2.67161613e-05 |
| Raw_Correct | PRG | 1770736 | 0.862226 | 0.137772 | 1.69421077e-06 | 4.86979179e-05 | 2.60616694e-05 |
| Raw_Wrong | Uraw | 708407 | 0.606942 | 0.393052 | 5.64647159e-06 | 3.12365164e-04 | 2.40949805e-04 |
| Raw_Wrong | FAraw | 708407 | 0.606945 | 0.393052 | 2.82323580e-06 | 7.83851019e-04 | 2.84735579e-04 |
| Raw_Wrong | PRG | 708407 | 0.606942 | 0.393052 | 5.64647159e-06 | 8.70930742e-04 | 2.59484812e-04 |

数据：[rddr_phase2b18_raw_gradient.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_raw_gradient.csv)。

仅上述三种训练时probe；shared-head是同一PRG的梯度路径诊断，不是第四个teacher或另一个损失方案。

## 8. Raw vs teacher semantic metrics

| model | accuracy | miou | dice | nll | brier | iou_class0 | iou_class1 | iou_class2 | iou_class3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.714253 | 0.436349 | 0.581730 | 0.784837 | 0.406249 | 0.626177 | 0.586830 | 0.375579 | 0.156809 |
| FixedAvg | 0.774087 | 0.573520 | 0.723531 | 0.681292 | 0.341498 | 0.699399 | 0.640964 | 0.514291 | 0.439427 |
| teacher | 0.785383 | 0.593171 | 0.740518 | 0.659663 | 0.326990 | 0.708893 | 0.654297 | 0.522020 | 0.487472 |

数据：[rddr_phase2b18_teacher_raw_metrics.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_teacher_raw_metrics.csv)。

GT0–3前景的native28 pooled4×4混淆矩阵；bg4/ignore255排除，不用官方background overwrite；union为0的类别标NA并从宏平均排除。NLL=-log(p_GT+eps)，Brier为四类平方误差之和，不除4。实际有效前景2,479,143个，分布于3,416张图；其余2张仍参与无GT损失与身份检查。

Teacher−raw accuracy=+7.1130pp，95% CI[+6.6530,+7.5727]pp；mIoU=+15.6822pp，95% CI[+14.8152,+16.5384]pp。历史raw/teacher完整精度指标复算一致。

## 9. Repair / Harm / NetRepair

| stratum | targets | repair | harm | net_repair | net_repair_rate |
| --- | --- | --- | --- | --- | --- |
| all | 2479143 | 291045 | 114703 | 176342 | 0.071130 |
| Top20 | 485451 | 212024 | 71684 | 140340 | 0.289092 |
| Bottom80 | 1993692 | 79021 | 43019 | 36002 | 0.018058 |
| Q1 | 495830 | 182 | 177 | 5 | 1.00841014e-05 |
| Q2 | 495828 | 2305 | 2024 | 281 | 5.66728785e-04 |
| Q3 | 495829 | 9491 | 7655 | 1836 | 0.003703 |
| Q4 | 495828 | 35260 | 21904 | 13356 | 0.026937 |
| Q5 | 495828 | 243807 | 82943 | 160864 | 0.324435 |
| Raw_Correct | 1770736 | 0 | 114703 | -114703 | -0.064777 |
| Raw_Wrong | 708407 | 291045 | 0 | 291045 | 0.410844 |
| Deep-Win | 314730 | 289608 | 0 | 289608 | 0.920179 |
| Shallow-Win | 182899 | 0 | 114703 | -114703 | -0.627138 |
| Both-Wrong | 393677 | 1437 | 0 | 1437 | 0.003650 |
| Stable-Correct | 1587837 | 0 | 0 | 0 | 0 |
| class0 | 965133 | 41476 | 58919 | -17443 | -0.018073 |
| class1 | 1044692 | 112837 | 46313 | 66524 | 0.063678 |
| class2 | 323515 | 67147 | 9100 | 58047 | 0.179426 |
| class3 | 145803 | 69585 | 371 | 69214 | 0.474709 |
| boundary | 201144 | 21881 | 20418 | 1463 | 0.007273 |
| interior | 2277999 | 269164 | 94285 | 174879 | 0.076769 |

数据：[rddr_phase2b18_teacher_raw_transition.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_teacher_raw_transition.csv)。

总体Repair291,045，Harm114,703，NetRepair+176,342（95% CI[164,899.8,187,869.025]）。NetRepair_rate=teacher−raw accuracy，是同一个估计量，不是第二份独立证据。

## 10. TeacherAdvRaw

```text
TeacherAdvRaw_i = p_teacher(GT)-p_raw(GT)
```

| stratum | targets | mean | median | positive_fraction | negative_fraction | zero_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2479143 | 0.074164 | 0.041327 | 0.761540 | 0.238447 | 1.25043210e-05 |
| Top20 | 485451 | 0.149893 | 0.248341 | 0.612466 | 0.387534 | 0 |
| Bottom80 | 1993692 | 0.055725 | 0.035450 | 0.797839 | 0.202146 | 1.55490417e-05 |
| Q1 | 495830 | 0.006526 | 0.005211 | 0.894762 | 0.105175 | 6.25214287e-05 |
| Q2 | 495828 | 0.031748 | 0.037640 | 0.824891 | 0.175109 | 0 |
| Q3 | 495829 | 0.068166 | 0.094008 | 0.772801 | 0.227199 | 0 |
| Q4 | 495828 | 0.107959 | 0.174006 | 0.704287 | 0.295713 | 0 |
| Q5 | 495828 | 0.156422 | 0.288496 | 0.610960 | 0.389040 | 0 |
| Raw_Correct | 1770736 | 0.048877 | 0.042583 | 0.858930 | 0.141052 | 1.75068446e-05 |
| Raw_Wrong | 708407 | 0.137372 | 0.019705 | 0.518103 | 0.481897 | 0 |
| Deep-Win | 314730 | 0.362168 | 0.355021 | 0.999679 | 3.20909986e-04 | 0 |
| Shallow-Win | 182899 | -0.216452 | -0.202756 | 0.007857 | 0.992143 | 0 |
| Both-Wrong | 393677 | -0.042344 | -0.037744 | 0.133102 | 0.866898 | 0 |
| Stable-Correct | 1587837 | 0.079440 | 0.055147 | 0.956963 | 0.043017 | 1.95234146e-05 |
| class0 | 965133 | 0.036124 | 0.025497 | 0.766767 | 0.233233 | 0 |
| class1 | 1044692 | 0.076315 | 0.052657 | 0.780726 | 0.219245 | 2.96738177e-05 |
| class2 | 323515 | 0.110279 | 0.049704 | 0.697458 | 0.302542 | 0 |
| class3 | 145803 | 0.230426 | 0.273130 | 0.731665 | 0.268335 | 0 |
| boundary | 201144 | 0.023408 | 0.001655 | 0.514189 | 0.485801 | 9.94312532e-06 |
| interior | 2277999 | 0.078646 | 0.044154 | 0.783381 | 0.216606 | 1.27304709e-05 |

数据：[rddr_phase2b18_teacher_advantage.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_teacher_advantage.csv)。

GT只定义观察量，绝不用于损失、teacher或训练样本过滤。

## 11. Raw GT-margin gradient

```text
g_s=dL/dL_s_student
v=-g_s  # unit step, NOT unit-normalized vector
dM=v_GT - max(v_k for k tied at CURRENT maximal non-GT logit)
```

冻结非GT最大logit并列位置2111个；PRG dM=0有7个。使用精确tied-max方向导数，不随意取某个并列竞争类。严格>0/<0符号，不增设近零阈值。logit/q梯度FP32，dM、dQ、norm和统计FP64累加。

## 12. Primary Benefit / Harm / Mean_dM

| stratum | loss | targets | benefit_rate | harm_rate | zero_rate | mean_dm | median_dm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | PRG | 2479143 | 0.789280 | 0.210718 | 2.82355637e-06 | 2.83648257e-04 | 3.53718024e-05 |
| Top20 | PRG | 485451 | 0.719677 | 0.280313 | 1.02997007e-05 | 8.76922180e-04 | 7.88453559e-04 |
| Bottom80 | PRG | 1993692 | 0.806227 | 0.193772 | 1.00316398e-06 | 1.39189926e-04 | 2.01013838e-05 |
| Q1 | PRG | 495830 | 0.891880 | 0.108120 | 0 | 1.30389037e-06 | 3.94443958e-07 |
| Q2 | PRG | 495828 | 0.820024 | 0.179974 | 2.01682842e-06 | 2.23766379e-05 | 1.93447656e-05 |
| Q3 | PRG | 495829 | 0.777575 | 0.222425 | 0 | 1.08930251e-04 | 1.18460721e-04 |
| Q4 | PRG | 495828 | 0.729215 | 0.270785 | 0 | 3.24946181e-04 | 3.94882452e-04 |
| Q5 | PRG | 495828 | 0.727704 | 0.272284 | 1.21009705e-05 | 9.60685815e-04 | 9.93195128e-04 |
| Raw_Correct | PRG | 1770736 | 0.862226 | 0.137772 | 1.69421077e-06 | 4.86979179e-05 | 2.60616694e-05 |
| Raw_Wrong | PRG | 708407 | 0.606942 | 0.393052 | 5.64647159e-06 | 8.70930742e-04 | 2.59484812e-04 |
| Deep-Win | PRG | 314730 | 1 | 0 | 0 | 0.002004 | 0.001496 |
| Shallow-Win | PRG | 182899 | 0.037720 | 0.962269 | 1.09349969e-05 | -0.001211 | -8.68234230e-04 |
| Both-Wrong | PRG | 393677 | 0.292707 | 0.707283 | 1.01606139e-05 | -3.51806384e-05 | -2.99312137e-05 |
| Stable-Correct | PRG | 1587837 | 0.957199 | 0.042800 | 6.29787566e-07 | 1.93766078e-04 | 4.19780390e-05 |
| class0 | PRG | 965133 | 0.782449 | 0.217549 | 2.07225325e-06 | 1.05588971e-04 | 1.31245774e-05 |
| class1 | PRG | 1044692 | 0.806956 | 0.193041 | 3.82887971e-06 | 3.17372034e-04 | 4.84156981e-05 |
| class2 | PRG | 323515 | 0.748958 | 0.251039 | 3.09104678e-06 | 5.02674123e-04 | 7.14764155e-05 |
| class3 | PRG | 145803 | 0.797316 | 0.202684 | 0 | 7.34679991e-04 | 5.95461315e-04 |
| boundary | PRG | 201144 | 0.566505 | 0.433490 | 4.97156266e-06 | 7.06899036e-05 | 3.85221267e-06 |
| interior | PRG | 2277999 | 0.808950 | 0.191047 | 2.63389053e-06 | 3.02452170e-04 | 3.85199710e-05 |

数据：[rddr_phase2b18_raw_gradient.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_raw_gradient.csv)。

all Benefit78.9280% > Harm21.0718%；Top20 Benefit71.9677% > Harm28.0313%。all、Top20与四类共6/6组Mean_dM>0，超过所需5/6，Gate B通过。分母始终为该stratum全部有效前景，零dM不剔除。

## 13. Raw-Correct protection

| stratum | teacher_accuracy | repair | harm | net_repair | benefit_rate | harm_rate | mean_dm | mean_dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw_Correct | 0.935223 | 0 | 114703 | -114703 | 0.862226 | 0.137772 | 4.86979179e-05 | -5.02883834e-05 |

数据：[rddr_phase2b18_correct_wrong_safety.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_correct_wrong_safety.csv)。

Raw-Correct PRG HarmRate=13.7772%，95% CI[13.3821,14.1790]%，满足≤50%。该总体保护指标不错，但会掩盖Shallow-Win子群的严重风险，不能替代层级安全门限。

## 14. Raw-Wrong correction

| stratum | teacher_accuracy | repair | harm | net_repair | benefit_rate | harm_rate | mean_dm | mean_dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw_Wrong | 0.410844 | 291045 | 0 | 291045 | 0.606942 | 0.393052 | 8.70930742e-04 | -1.85316930e-04 |

数据：[rddr_phase2b18_correct_wrong_safety.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_correct_wrong_safety.csv)。

Raw-Wrong BenefitRate=60.6942%，95% CI[59.7997,61.5708]%，不足70%。因此Gate C失败来自错误像素的纠正能力不足，而非Raw-Correct总体伤害率超标。

## 15. Deep-Win

| stratum | teacher_accuracy | repair | harm | net_repair | benefit_rate | harm_rate | mean_dm | mean_dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deep-Win | 0.920179 | 289608 | 0 | 289608 | 1 | 0 | 0.002004 | -3.21980956e-04 |

数据：[rddr_phase2b18_deep_shallow_win.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_deep_shallow_win.csv)。

314,730个位置；teacher正确率92.0179%，PRG BenefitRate100%，Mean_dM=+0.00200433。当deep正确、raw错误时，这种指导几乎总能提高GT margin。

## 16. Shallow-Win

| stratum | teacher_accuracy | repair | harm | net_repair | benefit_rate | harm_rate | mean_dm | mean_dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shallow-Win | 0.372862 | 0 | 114703 | -114703 | 0.037720 | 0.962269 | -0.001211 | -2.26890530e-04 |

数据：[rddr_phase2b18_deep_shallow_win.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_deep_shallow_win.csv)。

182,899个位置；teacher正确率37.2862%，hard-label harm62.7138%；PRG HarmRate96.2269%，Mean_dM=-0.00121071。浅层独有的正确信息没有得到保护。

此处teacher硬标签正确率不等同Phase2B1.5 support-sign的Shallow-Win recall=79.0939%。前者评价混合概率的argmax，后者评价support分数选择方向；不能因之前ranking较好就假定混合teacher会保留浅层标签。

## 17. dQ directional derivative

```text
g_q = d[JS(softmax(L_s), stopgrad(p_d))/ln2]/dL_s
dQ = sum_k g_q[k]*(-g_s[k])
```

q导数在独立图计算，不回流到PRG的q权重。逐像素q先求和再对logits求导，得到block-diagonal逐像素导数，不是对图像均值求导造成的额外1/784缩放。

| stratum | dQ_mean | dQ_median | dQ_negative_fraction | dQ_positive_fraction |
| --- | --- | --- | --- | --- |
| all | -8.88723487e-05 | -1.38969105e-05 | 0.999987 | 1.33110514e-05 |
| Top20 | -2.90459301e-04 | -2.30018855e-04 | 0.999975 | 2.47192817e-05 |
| Bottom80 | -3.97872406e-05 | -4.74614099e-06 | 0.999989 | 1.05332218e-05 |
| Q1 | -3.14029622e-08 | -4.39400711e-09 | 0.999996 | 4.03364056e-06 |
| Q2 | -1.77777434e-06 | -1.08470468e-06 | 0.999982 | 1.81514557e-05 |
| Q3 | -1.81026139e-05 | -1.40022918e-05 | 0.999988 | 1.21009461e-05 |
| Q4 | -9.81468660e-05 | -8.01235249e-05 | 0.999982 | 1.81514557e-05 |
| Q5 | -3.26303588e-04 | -2.61436686e-04 | 0.999986 | 1.41177989e-05 |
| Raw_Correct | -5.02883834e-05 | -4.93565169e-06 | 0.999993 | 6.77684308e-06 |
| Raw_Wrong | -1.85316930e-04 | -1.15583015e-04 | 0.999970 | 2.96439759e-05 |
| Deep-Win | -3.21980956e-04 | -2.51112839e-04 | 0.999994 | 6.35465319e-06 |
| Shallow-Win | -2.26890530e-04 | -1.94710905e-04 | 0.999989 | 1.09349969e-05 |
| Both-Wrong | -7.60591654e-05 | -2.58949422e-05 | 0.999952 | 4.82629160e-05 |
| Stable-Correct | -2.99460208e-05 | -2.77506755e-06 | 0.999994 | 6.29787566e-06 |
| class0 | -7.27250349e-05 | -5.34302921e-06 | 0.999997 | 3.10837988e-06 |
| class1 | -8.98667843e-05 | -1.50624748e-05 | 0.999986 | 1.43582989e-05 |
| class2 | -1.16431145e-04 | -3.48353973e-05 | 0.999991 | 9.27314035e-06 |
| class3 | -1.27484329e-04 | -1.23517029e-04 | 0.999918 | 8.23028333e-05 |
| boundary | -1.08486858e-04 | -3.48780137e-05 | 0.999985 | 1.49146880e-05 |
| interior | -8.71404161e-05 | -1.25952594e-05 | 0.999987 | 1.31694527e-05 |

数据：[rddr_phase2b18_hierarchy_direction.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_hierarchy_direction.csv)。

总体99.9987%位置dQ<0：PRG几乎普遍削弱浅深冲突。冲突降低本身既不等于语义改善，也不等于有害collapse。

## 18. CosCollapse

```text
CosCollapse = dQ / (norm(g_q)*norm(v)+1e-8)
```

| stratum | CosCollapse_mean | CosCollapse_median | CosCollapse_negative_fraction | CosCollapse_positive_fraction |
| --- | --- | --- | --- | --- |
| all | -0.870454 | -0.997935 | 0.999987 | 1.33110514e-05 |
| Top20 | -0.994792 | -0.999938 | 0.999975 | 2.47192817e-05 |
| Bottom80 | -0.840179 | -0.995452 | 0.999989 | 1.05332218e-05 |
| Q1 | -0.388963 | -0.304932 | 0.999996 | 4.03364056e-06 |
| Q2 | -0.977971 | -0.989022 | 0.999982 | 1.81514557e-05 |
| Q3 | -0.994497 | -0.998966 | 0.999988 | 1.21009461e-05 |
| Q4 | -0.995069 | -0.999807 | 0.999982 | 1.81514557e-05 |
| Q5 | -0.995773 | -0.999948 | 0.999986 | 1.41177989e-05 |
| Raw_Correct | -0.833783 | -0.996137 | 0.999993 | 6.77684308e-06 |
| Raw_Wrong | -0.962119 | -0.999588 | 0.999970 | 2.96439759e-05 |
| Deep-Win | -0.995741 | -0.999943 | 0.999994 | 6.35465319e-06 |
| Shallow-Win | -0.991933 | -0.999680 | 0.999989 | 1.09349969e-05 |
| Both-Wrong | -0.935240 | -0.998428 | 0.999952 | 4.82629160e-05 |
| Stable-Correct | -0.815566 | -0.994842 | 0.999994 | 6.29787566e-06 |
| class0 | -0.840372 | -0.996287 | 0.999997 | 3.10837988e-06 |
| class1 | -0.871904 | -0.998154 | 0.999986 | 1.43582989e-05 |
| class2 | -0.909461 | -0.998311 | 0.999991 | 9.27314035e-06 |
| class3 | -0.972646 | -0.999875 | 0.999918 | 8.23028333e-05 |
| boundary | -0.924917 | -0.998916 | 0.999985 | 1.49146880e-05 |
| interior | -0.865645 | -0.997802 | 0.999987 | 1.31694527e-05 |

数据：[rddr_phase2b18_hierarchy_direction.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_hierarchy_direction.csv)。

all均值-0.870454，Top20均值-0.994792。保留原eps，未为小梯度调整分母；特别是Q1 norm乘积很小时，eps会减小绝对值，因此不能把该量一概视为无稳定项的标准余弦。

## 19. Beneficial Reconciliation Rate（BRR）

BRR = Deep-Win内部 fraction(dM>0 AND dQ<0)，分母包含该组全部位置。

| stratum | Deep_Win_targets | BRR |
| --- | --- | --- |
| all | 314730 | 0.999994 |
| Top20 | 220561 | 0.999991 |
| Bottom80 | 94169 | 1 |
| boundary | 25404 | 0.999961 |
| interior | 289326 | 0.999997 |

数据：[rddr_phase2b18_reconciliation_collapse.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_reconciliation_collapse.csv)。

总体BRR=99.9994%，95% CI[99.9984,100]%，通过≥60%要求。

## 20. Harmful Hierarchy Collapse Rate（HHCR）

HHCR = Shallow-Win内部 fraction(dM<0 AND dQ<0)，不是总体像素中的比例。

| stratum | Shallow_Win_targets | HHCR |
| --- | --- | --- |
| all | 182899 | 0.962258 |
| Top20 | 103341 | 0.978305 |
| Bottom80 | 79558 | 0.941414 |
| boundary | 26392 | 0.964421 |
| interior | 156507 | 0.961893 |

数据：[rddr_phase2b18_reconciliation_collapse.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_reconciliation_collapse.csv)。

总体HHCR=96.2258%，95% CI[96.0071,96.4334]%，远高于≤30%要求。这是已预注册的局部方向风险事件，不是实际运行Full25后测得96%的信息消失或标签翻转。

## 21. Stable-Correct diversity

| stratum | teacher_accuracy | repair | harm | net_repair | benefit_rate | harm_rate | mean_dm | mean_dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stable-Correct | 1 | 0 | 0 | 0 | 0.957199 | 0.042800 | 1.93766078e-04 | -2.99460208e-05 |

数据：[rddr_phase2b18_deep_shallow_win.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_deep_shallow_win.csv)。

| stratum | dQ_mean | CosCollapse_mean | dQ_negative_fraction |
| --- | --- | --- | --- |
| Stable-Correct | -2.99460208e-05 | -0.815566 | 0.999994 |

数据：[rddr_phase2b18_hierarchy_direction.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_hierarchy_direction.csv)。

两路都正确的1,587,837个位置，teacher hard-label准确率仍100%；PRG Benefit95.7199%，Mean_dM正、dQ负。这类协调与Shallow-Win的有害冲突压缩必须分开解释。

## 22. Both-Wrong diagnostic

| stratum | teacher_accuracy | repair | harm | net_repair | benefit_rate | harm_rate | mean_dm | mean_dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Both-Wrong | 0.003650 | 1437 | 0 | 1437 | 0.292707 | 0.707283 | -3.51806384e-05 | -7.60591654e-05 |

数据：[rddr_phase2b18_deep_shallow_win.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_deep_shallow_win.csv)。

Both-Wrong共393,677；teacher仅纠正1,437个，正确率0.3650%。PRG Benefit29.2707%、Harm70.7283%、Mean_dM负；它拉低了Raw-Wrong总体纠正率。这是secondary观察，不引入third-evidence branch或依据GT挑选训练组。

## 23. Conflict localization

| loss | Top20_mean_G | Bottom80_mean_G | ratio |
| --- | --- | --- | --- |
| Uraw | 5.46256421e-04 | 1.61255234e-04 | 3.387527 |
| FAraw | 0.001414 | 2.05308762e-04 | 6.885680 |
| PRG | 0.001367 | 2.00047395e-04 | 6.835068 |

数据：[rddr_phase2b18_gradient_localization.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_gradient_localization.csv)。

PRG Top20/Bottom80梯度范数比6.83507，高于Uraw3.38753，定位标志为TRUE。这只证明q更强调高冲突位置，不证明这些位置的teacher一定正确。

## 24. q quintiles

| stratum | targets | net_repair | benefit_rate | harm_rate | mean_dm | mean_gradient_norm | mean_dQ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | 495830 | 5 | 0.891880 | 0.108120 | 1.30389037e-06 | 1.39841027e-06 | -3.14029622e-08 |
| Q2 | 495828 | 281 | 0.820024 | 0.179974 | 2.23766379e-05 | 2.64066372e-05 | -1.77777434e-06 |
| Q3 | 495829 | 1836 | 0.777575 | 0.222425 | 1.08930251e-04 | 1.37087981e-04 | -1.81026139e-05 |
| Q4 | 495828 | 13356 | 0.729215 | 0.270785 | 3.24946181e-04 | 4.66864076e-04 | -9.81468660e-05 |
| Q5 | 495828 | 160864 | 0.727704 | 0.272284 | 9.60685815e-04 | 0.001511 | -3.26303588e-04 |

数据：[rddr_phase2b18_q_quintiles.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_q_quintiles.csv)。

Q5 mean G=0.00151134 > Q1=1.39841e-6。Top20/Q1–Q5沿用旧缓存和冻结边界，Top20与Q5不是重新按同一规则选取的等同集合。本轮不只用Q5/Top20训练。

## 25. Per-class semantic safety

| stratum | teacher_accuracy | repair | harm | net_repair | benefit_rate | harm_rate | mean_dm | mean_dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class0 | 0.820668 | 41476 | 58919 | -17443 | 0.782449 | 0.217549 | 1.05588971e-04 | -7.27250349e-05 |
| class1 | 0.819277 | 112837 | 46313 | 66524 | 0.806956 | 0.193041 | 3.17372034e-04 | -8.98667843e-05 |
| class2 | 0.634104 | 67147 | 9100 | 58047 | 0.748958 | 0.251039 | 5.02674123e-04 | -1.16431145e-04 |
| class3 | 0.644637 | 69585 | 371 | 69214 | 0.797316 | 0.202684 | 7.34679991e-04 | -1.27484329e-04 |

数据：[rddr_phase2b18_per_class.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_per_class.csv)。

四类Mean_dM均正，且bootstrap下界均正；但类内总体改善不能抵消跨层winner子群的风险。未创建class-specific阈值或掩码。

## 26. Boundary / interior

复用冻结boundary≤7px、interior>7px标签，未重算尺度或改边界宽度。

| stratum | targets | net_repair | benefit_rate | harm_rate | mean_dm | BRR | HHCR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boundary | 201144 | 1463 | 0.566505 | 0.433490 | 7.06899036e-05 | 0.999961 | 0.964421 |
| interior | 2277999 | 174879 | 0.808950 | 0.191047 | 3.02452170e-04 | 0.999997 | 0.961893 |

数据：[rddr_phase2b18_boundary_interior.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_boundary_interior.csv)。

BRR/HHCR以对应boundary/interior内的Deep-Win/Shallow-Win作分母。无相应人群则NA，不写成0。

## 27. Feature gradient

| stratum | targets | channels | rms | mean_pixel_l2 | max_abs | finite |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2479143 | 512 | 5.04964994e-05 | 5.51926924e-04 | 0.010803 | True |
| Top20 | 485451 | 512 | 1.02798735e-04 | 0.001764 | 0.010803 | True |
| Bottom80 | 1993692 | 512 | 2.44467344e-05 | 2.56845889e-04 | 0.009583 | True |
| Q1 | 495830 | 512 | 1.45747967e-07 | 1.80973357e-06 | 1.52587891e-05 | True |
| Q2 | 495828 | 512 | 1.97405412e-06 | 3.40977332e-05 | 1.77383423e-04 | True |
| Q3 | 495829 | 512 | 9.33646624e-06 | 1.76656564e-04 | 5.76019287e-04 | True |
| Q4 | 495828 | 512 | 3.10616098e-05 | 6.00380969e-04 | 0.001831 | True |
| Q5 | 495828 | 512 | 1.08136886e-04 | 0.001947 | 0.010803 | True |
| class0 | 965133 | 512 | 4.63664460e-05 | 4.53148799e-04 | 0.010132 | True |
| class1 | 1044692 | 512 | 5.15399029e-05 | 5.47783808e-04 | 0.010803 | True |
| class2 | 323515 | 512 | 5.74537797e-05 | 7.19882401e-04 | 0.006775 | True |
| class3 | 145803 | 512 | 5.26185882e-05 | 8.62799756e-04 | 0.005768 | True |

数据：[rddr_phase2b18_feature_gradient.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_feature_gradient.csv)。

来自真实frozen-head PRG对F28_raw的反传。RMS跨该组所有像素×512通道，mean pixelL2先跨通道再平均。完整特征梯度未落盘；逐像素平方和和最大绝对值以小缓存保存，避免多GB冗余。

## 28. Upstream parameter gradient

| parameter | numel | rms | max_abs | nonzero_images | zero_images |
| --- | --- | --- | --- | --- | --- |
| b4.bn_branch2a.weight | 256 | 0.071120 | 1.807927 | 3418 | 0 |
| b4.bn_branch2a.bias | 256 | 0.042449 | 0.791982 | 3418 | 0 |
| b4.conv_branch2a.weight | 1179648 | 0.003330 | 0.388672 | 3418 | 0 |
| b4.bn_branch2b1.weight | 512 | 0.043221 | 1.497679 | 3418 | 0 |
| b4.bn_branch2b1.bias | 512 | 0.020958 | 0.467607 | 3418 | 0 |
| b4.conv_branch2b1.weight | 2359296 | 0.002441 | 0.687500 | 3418 | 0 |
| b4.conv_branch1.weight | 131072 | 0.005491 | 0.921875 | 3418 | 0 |
| b4_1.bn_branch2a.weight | 512 | 0.037664 | 2.548345 | 3418 | 0 |
| b4_1.bn_branch2a.bias | 512 | 0.016440 | 0.506077 | 3418 | 0 |
| b4_1.conv_branch2a.weight | 2359296 | 0.002261 | 0.839844 | 3418 | 0 |
| b4_1.bn_branch2b1.weight | 512 | 0.036322 | 1.535912 | 3418 | 0 |
| b4_1.bn_branch2b1.bias | 512 | 0.013061 | 0.359850 | 3418 | 0 |
| b4_1.conv_branch2b1.weight | 2359296 | 0.001980 | 0.902344 | 3418 | 0 |
| b4_2.bn_branch2a.weight | 512 | 0.022960 | 1.776539 | 3418 | 0 |
| b4_2.bn_branch2a.bias | 512 | 0.012237 | 0.395896 | 3418 | 0 |
| b4_2.conv_branch2a.weight | 2359296 | 0.001209 | 0.287109 | 3418 | 0 |
| b4_2.bn_branch2b1.weight | 512 | 0.020108 | 0.676744 | 3418 | 0 |
| b4_2.bn_branch2b1.bias | 512 | 0.010498 | 0.293974 | 3418 | 0 |
| b4_2.conv_branch2b1.weight | 2359296 | 0.001003 | 0.589844 | 3418 | 0 |
| b4_3.bn_branch2a.weight | 512 | 0.032945 | 1.288128 | 3418 | 0 |
| b4_3.bn_branch2a.bias | 512 | 0.014806 | 0.445743 | 3418 | 0 |
| b4_3.conv_branch2a.weight | 2359296 | 0.001862 | 0.667969 | 3418 | 0 |
| b4_3.bn_branch2b1.weight | 512 | 0.027905 | 1.353177 | 3418 | 0 |
| b4_3.bn_branch2b1.bias | 512 | 0.009339 | 0.318549 | 3418 | 0 |
| b4_3.conv_branch2b1.weight | 2359296 | 0.001335 | 0.523438 | 3418 | 0 |
| b4_4.bn_branch2a.weight | 512 | 0.025472 | 1.000391 | 3418 | 0 |
| b4_4.bn_branch2a.bias | 512 | 0.012515 | 0.266032 | 3418 | 0 |
| b4_4.conv_branch2a.weight | 2359296 | 0.001604 | 0.648438 | 3418 | 0 |
| b4_4.bn_branch2b1.weight | 512 | 0.022136 | 0.858861 | 3418 | 0 |
| b4_4.bn_branch2b1.bias | 512 | 0.008698 | 0.336515 | 3418 | 0 |
| b4_4.conv_branch2b1.weight | 2359296 | 0.001121 | 0.449219 | 3418 | 0 |
| b4_5.bn_branch2a.weight | 512 | 0.043502 | 1.587905 | 3418 | 0 |
| b4_5.bn_branch2a.bias | 512 | 0.015183 | 0.377986 | 3418 | 0 |
| b4_5.conv_branch2a.weight | 2359296 | 0.001754 | 0.996094 | 3418 | 0 |
| b4_5.bn_branch2b1.weight | 512 | 0.025262 | 1.220195 | 3418 | 0 |
| b4_5.bn_branch2b1.bias | 512 | 0.006662 | 0.164423 | 3418 | 0 |
| b4_5.conv_branch2b1.weight | 2359296 | 0.001193 | 0.390625 | 3418 | 0 |
| bn45.weight | 512 | 0.006152 | 0.396980 | 3418 | 0 |
| bn45.bias | 512 | 0.008682 | 0.322609 | 3418 | 0 |
| ic1.weight | 2048 | 0 | 0 | 0 | 3418 |
| ic1.bias | 4 | 0 | 0 | 0 | 3418 |

数据：[rddr_phase2b18_parameter_gradient.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_parameter_gradient.csv)。

所有39个批准上游参数均逐图检查finite；每个b4 block的卷积组汇总能量非零。primary ic1权重/偏置均无梯度；HFRM、deep-only路径及其余未批准参数均无梯度。BN affine仅临时用于诊断，running statistics从未更新。

## 29. Shared-head absorption diagnostic

| mode | feature_l2 | ic1_l2 | upstream_l2 | head_parameter_energy_fraction | head_fraction_per_image_mean | head_fraction_per_image_median |
| --- | --- | --- | --- | --- | --- | --- |
| PRG_frozen | 1.861372 | 0 | 580.970618 | 0 | 0 | 0 |
| PRG_sharedhead | 1.861372 | 58.156914 | 580.970618 | 0.009921 | 0.010273 | 0.009654 |

数据：[rddr_phase2b18_shared_head_diagnostic.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_shared_head_diagnostic.csv)。

各能量先逐图求梯度平方和，再跨图相加；分母仅ic1+批准上游参数，不混入feature-gradient能量。共享头占比0.99212%，远低于50%，吸收风险标志FALSE。frozen/shared特征梯度逐元素max差0，上游梯度总能量相同。本轮没有证据支持“监督主要被分类头吸收”。

该比例依赖参数化与所开放的上游集合，不是未来optimizer更新分配的因果测量，不能单凭它主张解冻head。

## 30. Detach tests 与独立数值验证

**37项unit/integration测试全部PASS，0skip；29项独立检查全部PASS。**

| check | passed |
| --- | --- |
| immutable_sources | 1 |
| observation_hash | 1 |
| full3418_order | 1 |
| frozen_replay | 1 |
| independent_exact_FP32_losses_gradients | 1 |
| independent_exact_FP32_q_derivative | 1 |
| FP64_analytic_KL_and_JS | 1 |
| raw_probability_exact | 1 |
| all_observation_gradients_finite | 1 |
| all_strata_margin_hierarchy | 1 |
| teacher_repair_harm_all_strata | 1 |
| semantic_confusion_nll_brier | 1 |
| feature_gradient_aggregation | 1 |
| frozen_shared_feature_same | 1 |
| 39_upstream_plus_2_head | 1 |
| frozen_head_zero | 1 |
| each_b4_conv_group_active | 1 |
| parameter_energy_bound | 1 |
| shared_energy_denominator | 1 |
| 10000_paired_image_bootstrap | 1 |
| BRR_HHCR_denominators | 1 |
| state_bn_checkpoint_identity | 1 |
| official_prediction_identity | 1 |
| detach_and_no_forbidden_gradients | 1 |
| BF16_batch20 | 1 |
| no_optimizer_test_luad | 1 |
| original_sources_unchanged | 1 |
| independent_all_gates_decision | 1 |
| secondary_and_strong_flags | 1 |

| diagnostic | max_error |
| --- | --- |
| FP32_loss | 0 |
| FP32_loss_gradient | 0 |
| FP32_q_gradient | 0 |
| FP64_KL_formula | 6.93889390e-18 |
| FP64_q_formula | 1.94289029e-16 |
| raw_probability | 0 |
| independent_margin_hierarchy_statistics | 0 |
| semantic_metrics | 0 |
| bootstrap_replicates | 2.27682456e-18 |
| bootstrap_intervals | 7.58941521e-19 |

独立验证器不导入主实现/分析器：重放三种FP32损失与logit梯度、独立q导数，用FP64解析epsilon-KL/JS导数对autograd验证，再独立重算margin/层级/混淆矩阵和bootstrap。FP32 loss、g_s、g_q误差均0；FP64 KL/JS解析差分别约6.94e-18/1.94e-16。本轮没有因为验证失败而调整阈值、容差或实验公式。

[verification.json](../audit/results/rddr_phase2b18/rddr_phase2b18_verification.json)、[测试原始记录](../audit/results/rddr_phase2b18/unit_integration_tests.txt)、[detach记录](../audit/results/rddr_phase2b18/rddr_phase2b18_detach_audit.json)。

## 31. Batch20 BF16 backward smoke 与资源

| batch | PRG_loss | seconds | allocated_GiB | reserved_GiB | conv_energy | head_energy | passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.271977 | 0.085119 | 2.436834 | 2.544922 | 10.565431 | 0 | 1 |

使用固定32张的前20张，真实BF16网络/反传、FP32 loss、整个batch分母。teacher/q/deep来源梯度断开。预算22GiB；这是b4段临时反传显存，不证明Full25全网络解冻显存。

GPU主审计总计82.764s，冻结重放阶段24.359s，真实上游反传45.895s；统计分析7.629s。以上为本次记录，不是重复benchmark或训练耗时预测。

## 32. Zero-update / inference identity

state_dict和BN buffer前后SHA一致：`c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5`。checkpoint SHA前后相同，严格加载missing_keys=[]，unexpected_keys=[]。

固定32个等距索引+seed42无放回抽取其余128图；未看结果挑图。原官方infer路径前后160图、8,028,160个预测像素SHA：`23e333ad8e5168c464cda0cfdaae1bed085bc4d304172166e4ca54b95fca8b93`，在background overwrite之前计算。固定160图raw logits前后完全相同，全3418图raw梯度路径logits与无梯度路径完全相同。

未创建optimizer、未step、未写checkpoint；没有test/LUAD/train split访问、没有新种子、没有完整训练。原推理仍仅运行A0，无teacher/support/q-loss额外计算；本轮没有新增inference FLOPs。

## 33. 10,000次配对图像bootstrap

seed42，从全部3418图有放回抽样。每次重汇总混淆矩阵和分子/分母，不是平均image mIoU，也不是pixel bootstrap。全部19个指标均10000个有效重复；报告percentile95% CI，未做多重检验校正，不据CI挑选子组或更改gate。

| metric | estimate | ci_low | ci_high | valid_resamples |
| --- | --- | --- | --- | --- |
| teacher-raw_accuracy_delta | 0.071130 | 0.066530 | 0.075727 | 10000 |
| teacher_NetRepair_count | 176342 | 164899.800000 | 187869.025000 | 10000 |
| teacher_NetRepair_rate | 0.071130 | 0.066530 | 0.075727 | 10000 |
| PRG:all:mean_dm | 2.83648257e-04 | 2.74570550e-04 | 2.92371670e-04 | 10000 |
| PRG:Top20:mean_dm | 8.76922180e-04 | 8.43489661e-04 | 9.09937779e-04 | 10000 |
| PRG:class0:mean_dm | 1.05588971e-04 | 9.23017469e-05 | 1.18726985e-04 | 10000 |
| PRG:class1:mean_dm | 3.17372034e-04 | 3.06373018e-04 | 3.28201973e-04 | 10000 |
| PRG:class2:mean_dm | 5.02674123e-04 | 4.79605510e-04 | 5.24370684e-04 | 10000 |
| PRG:class3:mean_dm | 7.34679991e-04 | 7.03698299e-04 | 7.66263213e-04 | 10000 |
| PRG:Raw_Correct:harm_rate | 0.137772 | 0.133821 | 0.141790 | 10000 |
| PRG:Raw_Wrong:benefit_rate | 0.606942 | 0.597997 | 0.615708 | 10000 |
| Deep-Win:BRR | 0.999994 | 0.999984 | 1 | 10000 |
| Shallow-Win:HHCR | 0.962258 | 0.960071 | 0.964334 | 10000 |
| Shallow-Win:teacher_accuracy | 0.372862 | 0.363543 | 0.382250 | 10000 |
| PRG:all:benefit_rate | 0.789280 | 0.784245 | 0.794346 | 10000 |
| PRG:all:harm_rate | 0.210718 | 0.205652 | 0.215752 | 10000 |
| PRG:Top20:benefit_rate | 0.719677 | 0.709495 | 0.729402 | 10000 |
| PRG:Top20:harm_rate | 0.280313 | 0.270585 | 0.290501 | 10000 |
| teacher-raw_mIoU_delta | 0.156822 | 0.148152 | 0.165384 | 10000 |

数据：[rddr_phase2b18_bootstrap.csv](../audit/results/rddr_phase2b18/rddr_phase2b18_bootstrap.csv)。

抽样索引流SHA256：`98e6164a3524dde42fc993cac0b5665076f7ebac7f6a73b7420d20c81022d00b`。完整重复值已归档；独立gather复算误差约2.28e-18。

## 34. Gate A / B / C / D / E

| gate | requirement | observed | result |
| --- | --- | --- | --- |
| A | teacher accuracy/mIoU > raw; NetRepair>0; CI lower>0 | +7.1130pp / +15.6822pp; +176342; both CI positive | PASS |
| B | all and Top20 Benefit>Harm | 78.9280>21.0718%; 71.9677>28.0313% | PASS |
| B | Mean_dM>0 in at least5/6 groups | 6/6 | PASS |
| C | Raw_Wrong Benefit≥70% | 60.6942% | FAIL |
| C | Raw_Correct Harm≤50%; NetRepair>0 | 13.7772%; +176342 | PASS |
| D | Deep-Win BRR≥60% | 99.9994% | PASS |
| D | Shallow-Win HHCR≤30% | 96.2258% | FAIL |
| D | Shallow-Win teacher accuracy≥60% | 37.2862% | FAIL |
| D | Q5 mean G>Q1 | .00151134 > .00000139841 | PASS |
| E | finite/detach/upstream/batch20/identity/no-step | all verified | PASS |

汇总A/B/C/D/E=PASS/PASS/FAIL/FAIL/PASS。按批准优先级，E通过、A通过、C失败，最终标签为TEACHER_BETTER_THAN_RAW_BUT_GRADIENT_UNSAFE；D的层级风险也完整报告，不因决策优先级隐藏。

## 35. Secondary flags

CONFLICT_LOCALIZATION_CONFIRMED=TRUE（PRG比Uraw更集中高冲突区域）。

SHARED_HEAD_ABSORPTION_RISK=FALSE（head平方能量占比0.99212%≤50%）。

两项都不覆盖或放宽C/D主门限，不能单独解锁训练。

## 36. STRONG_PRERECT_GUIDANCE_SIGNAL

FALSE。teacher−raw mIoU≥10pp、all/Top20 Mean_dM正、Raw-Correct Harm≤35%、BRR≥70%均满足；但Raw-Wrong Benefit60.6942%未达到80%，HHCR96.2258%远高于20%。不得只列通过部分作为Strong Go。

## 37. Scientific interpretation 与局限

**整体有效，但并非层级安全。** 与teacher→rect不同，teacher→raw的全局梯度收益确实成立。失败不是工程断梯度或分类头吸收，而是两类结构性问题：Both-Wrong纠正有限、Shallow-Win有效分歧被压低。

**为何contextual teacher仍可能变成向deep靠拢？** 在当前冻结点，teacher=(1−wD)·p_s+wD·p_d。忽略log内极小eps项时，softmax-KL对raw logits的导数可直接化简为：

```text
g_PRG ≈ [q/(sum(q)+eps)] * wD * (p_s-p_d)
v_PRG ≈ positive_scalar * (p_d-p_s)
```

这是公式推导，不是新加一个deep-teacher实验。它表明在该冻结点，context的wD主要调节“向deep靠拢”的幅度，并不因Shallow-Win而自动反转方向。实际eps-KL梯度已精确计算和复核；观测上几乎所有位置dQ<0，因此有益reconciliation与有害dissent抹除同时出现。没有将近似等式伪称为含eps实现逐bit恒等式。

Shallow-Win占所有Raw-Correct的一部分，所以Raw-Correct总体Harm仅13.78%并不能排除其子群96.23%的风险。另外，support-sign偏好浅层也不保证混合分布argmax仍选浅层类别。

本轮仅固定checkpoint、单位局部下降方向，没有参数更新、没有联合SSHR分类损失、没有lambda和长期trajectory。因此不能宣称Full25已经塌缩，也不能把native28 teacher的+15.68pp当作最终模型mIoU提升。39个临时上游参数包含BN affine，属于审计白名单，不授权改变未来正式训练的BN冻结。

交付包括完整CSV/JSON、测试、SHA清单、[复现README](README_rddr_phase2b18.md)和[交付摘要](rddr_phase2b18_delivery_summary.md)。独立PR不自动merge；旧实验/缓存保持不动。不追加阈值、lambda、GT保护掩码、third-evidence或新架构补丁。

## 38. Exact final decision

预注册C/D门限未通过。完成本轮报告后停止，不进入Phase-2B2、Full25或test。

DECISION = TEACHER_BETTER_THAN_RAW_BUT_GRADIENT_UNSAFE
