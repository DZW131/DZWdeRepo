# RDDR Phase-2B1.7 Contextual Correction Acceptance Audit

完整实验报告｜BCSS validation-only｜C0 Full25 seed42｜zero-update

**结论：冻结的 contextual acceptance 方案未通过；A/B/C/D 全部 FAIL，工程验证 PASS。** 接纳分数具有一定 winner 排序信号，但不足以安全选择教师纠正。本轮没有训练、没有 optimizer、没有 test/LUAD、没有阈值搜索，也没有改动或删除既有权重。

所有 rate/accuracy/AUC 默认以 0–1 表示，pp 明确表示百分点；dM 是单位负梯度方向下的局部 logit margin 变化，不是 mIoU，不代表真实训练后的预测翻转率。表内显示值经过格式化，CSV/JSON 保留原始精度。

## 1. Provenance、SHA 与实际命令

纯 A0：`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`。分支：`feature/rddr-phase2b17-acceptance`；PR 基线：`baseline/official-a0`。实际 GPU 执行 commit：`c4946123bae64816b5772d9806101aaa916ec38d`；最终独立复核 commit：`131dd55b81c153acc83de4fcb3911718dca40dd0`。仅新增独立审计工具、测试、文档与结果；原网络、训练、推理、指标文件保持不变。

合同：[rddr_phase2b17_contract.md](rddr_phase2b17_contract.md)。用户批准的规格为 `RDDR_Phase2B1_7_Contextual_Correction_Acceptance_Audit_v1.0.md`。

| asset | path | SHA256 |
| --- | --- | --- |
| native | /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz | 767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a |
| derived | /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz | 237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514 |
| previous | /home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1/rddr_phase2b16_gradient_observations.npz | 5ab5a048e932f27726cea7507685354f984432fb66b542b87b805cea24a72bd5 |
| checkpoint | /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth | 509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579 |

checkpoint 为 451,130,207 bytes；native/derived/previous 缓存分别为 240,845,567 / 301,590,961 / 155,185,711 bytes。新观察缓存位于 `/home/duyanhong/experiments/RDDR_PHASE2B17/formal_r1/rddr_phase2b17_observations.npz`，SHA256：`08f4b4480667b8a85689ba069525b7e8f16e309b79d052edc73afa8d1c46aeac`。大缓存不提交 Git。

环境：NVIDIA GeForce RTX 5090 D v2；PyTorch `2.11.0+cu128`；NumPy `1.23.5`；BF16 network / FP32 softmax, support and loss；主审计 batch=1，补充 batch20。全部 3,418 张 validation，28×28 原生审计网格；2,479,143 有效前景位置分布在 3,416 张图。损失仍覆盖全部 2,679,712 原生位置，GT 不控制损失采样。

实际 GPU 命令（已运行，不应覆盖原目录重跑）：

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b17_acceptance_audit.py \
  --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz \
  --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz \
  --previous /home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1/rddr_phase2b16_gradient_observations.npz \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --val-images /home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img \
  --output /home/duyanhong/experiments/RDDR_PHASE2B17/formal_r1
```

实际统计与独立验证命令：

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/analyze_rddr_phase2b17.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B17/formal_r1 \
  --output /home/duyanhong/experiments/RDDR_PHASE2B17/report_r3
```

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/verify_rddr_phase2b17.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B17/formal_r1 \
  --report /home/duyanhong/experiments/RDDR_PHASE2B17/report_r3
```

服务器工作目录：`/home/duyanhong/DZWdeRepo-rddr-phase2b17`。最终小型结果来自 `formal_r1` 与 `report_r3`，本地归档在 `audit/results/rddr_phase2b17/`。早期 `report_r1/r2` 均保留，数值验证修订过程见第25节。

## 2. 冻结的 Phase-2B1.6 证据

上一轮为 `TEACHER_SIGNAL_PRESENT_GRADIENT_UNSAFE`，A/B/C/D = PASS/PASS/FAIL/PASS。以下是原生 CAM28_1 的冻结诊断，并非官方最终融合 segmentation 指标：

| model | accuracy | mIoU |
| --- | --- | --- |
| rect | 81.7788% | 63.7895% |
| symmetric teacher | 78.5383% | 59.3171% |

Repair=88,290；Harm=168,626；NetRepair=-3.2405 pp。CCA all Benefit/Harm=17.9611%/82.0389%，Top20=31.2450%/68.7550%；Rect_Wrong Benefit=86.8583%，Rect_Correct Harm=97.3899%。q 的正权重只缩放逐像素梯度，不能反转方向。本轮检验的不是再次证明 teacher 有信号，而是能否接受有益纠正、拒绝有害纠正。

## 3. p_rect / p_teacher 张量与冻结重放

`p_s/p_d/p_rect/p_teacher: [B,4,28,28]`，FP32；`q: [B,28,28]`。p_rect 从冻结 BF16 forward 产生的 FP32 logits 用 CUDA FP32 softmax 重建，不把上一轮统计时临时 FP64 softmax 当成冻结概率。teacher 仍是原 symmetric adjudication 的浅/深混合。

```text
S_S_sym = 0.5*(T_SS+T_SD)
S_D_sym = 0.5*(T_DS+T_DD)
wD = S_D_sym/(S_S_sym+S_D_sym+1e-8)
p_teacher = (1-wD)*p_s+wD*p_d
q = JS(p_s,p_d)/ln(2)
```

| quantity | replay_max_abs |
| --- | --- |
| T_SS | 0 |
| T_SD | 0 |
| T_DS | 0 |
| T_DD | 0 |
| sym | 0 |
| wD_sym | 0 |
| anchor_sym | 0 |
| q | 5.96046448e-08 |
| U_gradient | 0 |
| CCA_gradient | 0 |
| U_loss | 0 |
| CCA_loss | 0 |

teacher 与旧四项 support 完全一致；q 最大差 5.96046e-8，在预先批准的 1e-7 内，损失使用原缓存 q。U/CCA 损失与梯度完全一致；3,418 张真实网络 logits 重放 max_abs=0。因此“exact replay”不能误写成所有浮点中间量逐 bit 相等。

## 4. S_R / S_T：对称邻域支持

```text
R_S(i)=mean_j[1-JS(p_rect(i),p_s(j))/ln2]
R_D(i)=mean_j[1-JS(p_rect(i),p_d(j))/ln2]
T_S(i)=mean_j[1-JS(p_teacher(i),p_s(j))/ln2]
T_D(i)=mean_j[1-JS(p_teacher(i),p_d(j))/ln2]
S_R=0.5*(R_S+R_D); S_T=0.5*(T_S+T_D)
```

15×15、radius=7、去掉中心自环、仅图内邻居。JS 保留旧实现 log 内 eps=1e-8。新 support 没有额外 clamp、归一化或 offset；GT 不进入构造。下表仅统计有效前景，计算本身覆盖全部位置。

| quantity | count | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| R_S | 2479143 | 0.778082 | 0.122014 | 0.798315 | 0.165347 | 0.997823 |
| R_D | 2479143 | 0.816938 | 0.173606 | 0.858669 | 0.002776 | 1 |
| T_S | 2479143 | 0.855311 | 0.096852 | 0.876396 | 0.136813 | 0.998817 |
| T_D | 2479143 | 0.802827 | 0.149910 | 0.823001 | 0.025404 | 0.999998 |
| S_R | 2479143 | 0.797510 | 0.126901 | 0.811173 | 0.112236 | 0.998910 |
| S_T | 2479143 | 0.829069 | 0.112441 | 0.845476 | 0.099404 | 0.998770 |

数据：[rddr_phase2b17_support_rect_teacher.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_support_rect_teacher.csv)。

## 5. Delta_accept：固定原始分数

```text
Delta_accept = S_T-S_R
m = (Delta_accept > 0)
a = relu(Delta_accept)
```

| quantity | count | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| delta | 2479143 | 0.031559 | 0.070896 | 0.022328 | -0.813164 | 0.780472 |

数据：[rddr_phase2b17_support_rect_teacher.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_support_rect_teacher.csv)。

零阈值固定。有效前景中 Δ=0 有4个，全部拒绝。没有 offset、temperature、分数翻转、class rule 或阈值扫描。平均 S_T>S_R 不是 teacher 比 rect 更正确的证明；必须结合 winner 标签检验。

## 6. Teacher-Win / Rect-Win 诊断人群

仅在 teacher 与 rect 预测不同且恰好一个正确的位置定义 winner 标签。Teacher-Win=88,290，Rect-Win=168,626，合计256,916，正类占比0.343653。其余有效前景并不进入 winner AUROC，但仍进入梯度/覆盖率统计。

3,250张图存在 winner 样本；其中2,547张同时含两类，可计算 image AUROC。排除871张：168张无 winner 样本、703张仅单类。不能把排除图作为 AUC=0.5 填充。

## 7. Winner acceptance AUROC / AP / BA / recalls

| stratum | positive | negative | auroc | image_auroc | auprc | balanced_accuracy | macro_f1 | teacher_win_recall | rect_win_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 88290 | 168626 | 0.621813 | 0.619465 | 0.452854 | 0.587359 | 0.550081 | 0.699139 | 0.475579 |
| Top20 | 48666 | 72297 | 0.634222 | 0.626433 | 0.523360 | 0.592511 | 0.546800 | 0.804853 | 0.380168 |
| Bottom80 | 39624 | 96329 | 0.583977 | 0.583793 | 0.351262 | 0.558244 | 0.530541 | 0.569301 | 0.547187 |

数据：[rddr_phase2b17_acceptance_winner.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_acceptance_winner.csv)。

主 image-AUROC=0.619465，95% CI [0.608924,0.630020]：高于随机，但未达到0.65。固定零阈值 BA=0.587359，TW recall=0.699139，RW recall=0.475579；BA和RW recall均未达门限。TP/FN/FP/TN=61,727/26,563/88,431/80,195。

## 8. Gradient-benefit AUROC

标签复用冻结 CCA 的 exact first-order dM：正445,281，负2,033,862，dM=0为0。原始非GT最大 logit 存在2,698个并列位置，使用上一轮冻结的 tied-max directional derivative，未随意选单个竞争类。

| stratum | positive | negative | auroc | image_auroc | auprc | eligible_images |
| --- | --- | --- | --- | --- | --- | --- |
| all | 445281 | 2033862 | 0.585218 | 0.523684 | 0.277797 | 2889 |
| Top20 | 151679 | 333772 | 0.555888 | 0.574120 | 0.393544 | 2794 |
| Rect_Correct | 52917 | 1974497 | 0.232373 | 0.238441 | 0.015361 | 2651 |
| Rect_Wrong | 392364 | 59365 | 0.860926 | 0.869708 | 0.976387 | 2594 |
| class0 | 113720 | 851413 | 0.585211 | 0.501800 | 0.182031 | 1483 |
| class1 | 186315 | 858377 | 0.647692 | 0.569376 | 0.423873 | 2080 |
| class2 | 90347 | 233168 | 0.405890 | 0.391516 | 0.231633 | 829 |
| class3 | 54899 | 90904 | 0.353547 | 0.477921 | 0.340682 | 333 |
| boundary | 98149 | 102995 | 0.492199 | 0.484881 | 0.493664 | 2102 |
| interior | 347132 | 1930867 | 0.594252 | 0.518767 | 0.257016 | 2860 |

数据：[rddr_phase2b17_gradient_discrimination.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_gradient_discrimination.csv)。

主 image-AUROC=0.523684，95% CI [0.516922,0.530530]，远未达到0.65。合格图2,889张；529张无双标签，其中2张无有效前景、527张仅一种梯度符号。

## 9. q 与 acceptance 是否提供不同证据

| score | auroc | image_auroc | auprc | eligible_images |
| --- | --- | --- | --- | --- |
| q | 0.609447 | 0.616488 | 0.472654 | 2547 |
| delta_accept | 0.621813 | 0.619465 | 0.452854 | 2547 |

数据：[rddr_phase2b17_q_vs_acceptance.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_q_vs_acceptance.csv)。

配对 image-AUROC 差 Δ−q=+0.002978，95% CI [-0.006820,+0.012930]，跨0。q（Need）与Δ（Trust）在定义上不同，但本轮不能声称Δ在 winner 判别上显著优于q。规格中的“证明是两个不同变量”不能取代实测证据；这里如实报告未建立更强判别能力。

## 10. Confidence controls（仅诊断）

| score | auroc | image_auroc | auprc |
| --- | --- | --- | --- |
| teacher_maxconf_minus_rect_maxconf | 0.621863 | 0.634530 | 0.420361 |
| teacher_entropy_minus_rect_entropy | 0.365407 | 0.351809 | 0.263639 |
| JS_teacher_rect | 0.357266 | 0.343663 | 0.267632 |

数据：[rddr_phase2b17_confidence_controls.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_confidence_controls.csv)。

熵采用自然对数，JS控制保持未除ln2的原定义。所有方向预先固定，不对低于0.5的分数取负，不采用看起来更好的 maxconf 分数替代 primary Δ，不做任何融合。

## 11. Hard acceptance：primary consumption probe

```text
L_HA = sum_i q_i*m_i*KL(p_teacher||p_rect) / (sum_i q_i*m_i+1e-8)
```

所有 teacher/q/Δ/m 均 detach。主实验每图 batch1，分母含全部784位置（包括背景与ignore），GT不用于选择损失位置。全拒绝图通过eps安全得到0损失和0梯度；本次真实图全拒绝数为0，人工全拒绝单测通过。未对完整 SSHR 分类目标做优化更新。

## 12. Soft acceptance：唯一 secondary probe

```text
L_SA = sum_i q_i*relu(Delta_i)*KL(p_teacher||p_rect) / (sum_i q_i*relu(Delta_i)+1e-8)
```

| stratum | benefit_rate | harm_rate | zero_rate | mean_dm | median_dm |
| --- | --- | --- | --- | --- | --- |
| all | 0.142161 | 0.696084 | 0.161755 | -1.79455851e-04 | -4.79107166e-06 |
| Top20 | 0.245982 | 0.495902 | 0.258117 | -2.49718995e-04 | 0 |
| Rect_Correct | 0.010351 | 0.843256 | 0.146393 | -5.14157521e-04 | -2.81922985e-05 |
| Rect_Wrong | 0.733741 | 0.035557 | 0.230702 | 0.001323 | 7.62096097e-05 |

数据：[rddr_phase2b17_sa_gradient.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_sa_gradient.csv)。

SA与HA有相同接受位置与逐像素方向，只重分配幅度；未引入sigmoid、power或温度。SA all Mean_dM=-1.7945585e-4，较HA更接近零，但仍为负。

## 13. Acceptance rate 与分母

| stratum | targets | accepted | rejected | acceptance_rate | zero_delta |
| --- | --- | --- | --- | --- | --- |
| all | 2479143 | 2078128 | 401015 | 0.838245 | 4 |
| Top20 | 485451 | 360148 | 125303 | 0.741883 | 0 |
| Bottom80 | 1993692 | 1717980 | 275712 | 0.861708 | 4 |
| Q1 | 495830 | 481157 | 14673 | 0.970407 | 4 |
| Q2 | 495828 | 451060 | 44768 | 0.909711 | 0 |
| Q3 | 495829 | 412841 | 82988 | 0.832628 | 0 |
| Q4 | 495828 | 371135 | 124693 | 0.748516 | 0 |
| Q5 | 495828 | 361935 | 133893 | 0.729961 | 0 |
| Rect_Correct | 2027414 | 1730614 | 296800 | 0.853607 | 4 |
| Rect_Wrong | 451729 | 347514 | 104215 | 0.769298 | 0 |
| Teacher-Win | 88290 | 61727 | 26563 | 0.699139 | 0 |
| Rect-Win | 168626 | 88431 | 80195 | 0.524421 | 0 |
| class0 | 965133 | 804977 | 160156 | 0.834058 | 0 |
| class1 | 1044692 | 873889 | 170803 | 0.836504 | 4 |
| class2 | 323515 | 269513 | 54002 | 0.833077 | 0 |
| class3 | 145803 | 129749 | 16054 | 0.889893 | 0 |
| boundary | 201144 | 162471 | 38673 | 0.807735 | 1 |
| interior | 2277999 | 1915657 | 362342 | 0.840938 | 3 |
| Corrected_by_CH | 305318 | 192815 | 112503 | 0.631522 | 0 |
| Still_Wrong | 378130 | 301990 | 76140 | 0.798641 | 0 |
| Harmed_by_CH | 69617 | 43676 | 25941 | 0.627375 | 0 |
| Stable_Correct | 1726078 | 1539647 | 186431 | 0.891992 | 4 |

数据：[rddr_phase2b17_acceptance_population.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_acceptance_population.csv)。

总体接受83.8245%，不存在“几乎全拒绝”的假安全。Top20与Q5分别沿用原冻结定义，本轮不重新排序或重建边界，因此两组大小不必相等。所有统计率默认整个有效前景stratum作分母。

## 14. Accepted / Rejected teacher quality

| stratum | region | targets | teacher_accuracy | rect_accuracy | accuracy_delta | repair | harm | net_repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | Accepted | 2078128 | 0.819925 | 0.832775 | -0.012850 | 61727 | 88431 | -26704 |
| all | Rejected | 401015 | 0.606381 | 0.740122 | -0.133741 | 26563 | 80195 | -53632 |
| Top20 | Accepted | 360148 | 0.663966 | 0.679634 | -0.015669 | 39169 | 44812 | -5643 |
| Top20 | Rejected | 125303 | 0.582524 | 0.726080 | -0.143556 | 9497 | 27485 | -17988 |

数据：[rddr_phase2b17_accepted_teacher_quality.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_accepted_teacher_quality.csv)。

Accepted: teacher−rect=-1.2850 pp，95% CI [-1.5129,-1.0563] pp；净修复数=-26,704，95%图像bootstrap区间[-31,438.175,-21,953.875]。Rejected中teacher也较差（-13.3741 pp）。虽然拒绝区明显更差，接受区仍不满足teacher>rect。NetRepair_rate与accuracy_delta数学相等，不能视为两份独立成功证据。

## 15. Correction precision / recall / protection

| stratum | Teacher_Win | Rect_Win | accepted_Teacher_Win | accepted_Rect_Win | correction_precision | correction_recall | rect_protection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 88290 | 168626 | 61727 | 88431 | 0.411080 | 0.699139 | 0.475579 |
| Top20 | 48666 | 72297 | 39169 | 44812 | 0.466403 | 0.804853 | 0.380168 |
| Bottom80 | 39624 | 96329 | 22558 | 43619 | 0.340874 | 0.569301 | 0.547187 |

数据：[rddr_phase2b17_selective_correction.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_selective_correction.csv)。

CorrectionPrecision只以接受的winner-conflict为分母，不等同teacher整体accuracy。总体接受的有用纠正61,727少于有害纠正88,431；对Rect-Win的保护率仅47.5579%。

## 16. HA 梯度审计

g=dL/dlogits，下降方向v=-g。GT margin为 z_GT−max(z_nonGT)，dM沿v取精确一阶方向导数。Benefit/Harm/Zero互斥且以整个stratum计数，拒绝位置的0梯度必须保留。

| stratum | loss | benefit_rate | harm_rate | zero_rate | mean_dm | median_dm | active_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | HA | 0.142161 | 0.696084 | 0.161755 | -2.34511453e-04 | -1.24570051e-05 | 0.838245 |
| Top20 | HA | 0.245982 | 0.495902 | 0.258117 | -4.77768753e-04 | 0 | 0.741883 |
| Rect_Correct | HA | 0.010351 | 0.843256 | 0.146393 | -4.57830349e-04 | -5.35692780e-05 | 0.853607 |
| Rect_Wrong | HA | 0.733741 | 0.035557 | 0.230702 | 7.67770569e-04 | 1.46379447e-04 | 0.769298 |
| class0 | HA | 0.089971 | 0.744087 | 0.165942 | -2.84427446e-04 | -1.13880160e-05 | 0.834058 |
| class1 | HA | 0.143910 | 0.692594 | 0.163496 | -1.78223933e-04 | -1.44571477e-05 | 0.836504 |
| class2 | HA | 0.214432 | 0.618645 | 0.166923 | -3.03439727e-04 | -4.45389014e-06 | 0.833077 |
| class3 | HA | 0.314733 | 0.575160 | 0.110107 | -1.54459732e-04 | -1.77228882e-04 | 0.889893 |

数据：[rddr_phase2b17_ha_gradient.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_ha_gradient.csv)。

HA all Benefit14.2161% < Harm69.6084%；Top20 Benefit24.5982% < Harm49.5902%。接受条件下all Harm=83.0407%，仅作辅助，不替换主门限分母。

## 17. U / CCA / HA / SA 对照与 CCA→HA 主比较

| stratum | loss | benefit_rate | harm_rate | zero_rate | mean_dm | median_dm | active_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | U | 0.179611 | 0.820389 | 0 | -1.94769301e-04 | -1.26539846e-04 | 1 |
| all | CCA | 0.179611 | 0.820389 | 0 | -2.89491245e-04 | -5.16416148e-05 | 1 |
| all | HA | 0.142161 | 0.696084 | 0.161755 | -2.34511453e-04 | -1.24570051e-05 | 0.838245 |
| all | SA | 0.142161 | 0.696084 | 0.161755 | -1.79455851e-04 | -4.79107166e-06 | 0.838245 |
| Top20 | U | 0.312450 | 0.687550 | 0 | -2.77904322e-04 | -4.99908230e-04 | 1 |
| Top20 | CCA | 0.312450 | 0.687550 | 0 | -6.80943484e-04 | -7.76608591e-04 | 1 |
| Top20 | HA | 0.245982 | 0.495902 | 0.258117 | -4.77768753e-04 | 0 | 0.741883 |
| Top20 | SA | 0.245982 | 0.495902 | 0.258117 | -2.49718995e-04 | 0 | 0.741883 |
| Rect_Correct | U | 0.026101 | 0.973899 | 0 | -3.35996740e-04 | -2.02362717e-04 | 1 |
| Rect_Correct | CCA | 0.026101 | 0.973899 | 0 | -5.19188394e-04 | -1.19684111e-04 | 1 |
| Rect_Correct | HA | 0.010351 | 0.843256 | 0.146393 | -4.57830349e-04 | -5.35692780e-05 | 0.853607 |
| Rect_Correct | SA | 0.010351 | 0.843256 | 0.146393 | -5.14157521e-04 | -2.81922985e-05 | 0.853607 |
| Rect_Wrong | U | 0.868583 | 0.131417 | 0 | 4.39076402e-04 | 3.47437162e-04 | 1 |
| Rect_Wrong | CCA | 0.868583 | 0.131417 | 0 | 7.41417145e-04 | 2.96493236e-04 | 1 |
| Rect_Wrong | HA | 0.733741 | 0.035557 | 0.230702 | 7.67770569e-04 | 1.46379447e-04 | 0.769298 |
| Rect_Wrong | SA | 0.733741 | 0.035557 | 0.230702 | 0.001323 | 7.62096097e-05 | 0.769298 |
| class0 | CCA | 0.117828 | 0.882172 | 0 | -3.60274951e-04 | -4.13073940e-05 | 1 |
| class0 | HA | 0.089971 | 0.744087 | 0.165942 | -2.84427446e-04 | -1.13880160e-05 | 0.834058 |
| class1 | CCA | 0.178344 | 0.821656 | 0 | -2.55724942e-04 | -6.07920338e-05 | 1 |
| class1 | HA | 0.143910 | 0.692594 | 0.163496 | -1.78223933e-04 | -1.44571477e-05 | 0.836504 |
| class2 | CCA | 0.279267 | 0.720733 | 0 | -2.66860248e-04 | -3.79978701e-05 | 1 |
| class2 | HA | 0.214432 | 0.618645 | 0.166923 | -3.03439727e-04 | -4.45389014e-06 | 0.833077 |
| class3 | CCA | 0.376529 | 0.623471 | 0 | -1.13096810e-04 | -2.30196107e-04 | 1 |
| class3 | HA | 0.314733 | 0.575160 | 0.110107 | -1.54459732e-04 | -1.77228882e-04 | 0.889893 |

数据：[rddr_phase2b17_all_gradient_controls.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_all_gradient_controls.csv)。

HA−CCA all Mean_dM=+5.4979792e-5，CI[+5.1718891e-5,+5.8292067e-5]；all Harm下降12.4305 pp。说明拒绝机制确实减轻一部分伤害，但“负值变得不那么负”不等于安全。接受位置HA/SA没有反转CCA方向，拒绝位置梯度严格为0；相关恒等式及数值误差保存在 gradient_identities.json。

## 18. Rect-Correct protection

| loss | targets | harm_rate | mean_dm | active_fraction |
| --- | --- | --- | --- | --- |
| U | 2027414 | 0.973899 | -3.35996740e-04 | 1 |
| CCA | 2027414 | 0.973899 | -5.19188394e-04 | 1 |
| HA | 2027414 | 0.843256 | -4.57830349e-04 | 0.853607 |
| SA | 2027414 | 0.843256 | -5.14157521e-04 | 0.853607 |

数据：[rddr_phase2b17_correct_wrong_safety.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_correct_wrong_safety.csv)。

CCA Harm=0.973899；HA/SA Harm=0.843256。固定要求HA≤0.486950（CCA的一半），未达到。绝对下降13.0643 pp，95% CI约[12.8471,13.2799] pp；不能把这一下降误读成减半。在已接受的Rect_Correct内，HA伤害率约98.7874%，表明本轮仍接收许多降低正确类别margin的教师方向。

## 19. Rect-Wrong correction

| loss | targets | benefit_rate | harm_rate | mean_dm | active_fraction |
| --- | --- | --- | --- | --- | --- |
| U | 451729 | 0.868583 | 0.131417 | 4.39076402e-04 | 1 |
| CCA | 451729 | 0.868583 | 0.131417 | 7.41417145e-04 | 1 |
| HA | 451729 | 0.733741 | 0.035557 | 7.67770569e-04 | 0.769298 |
| SA | 451729 | 0.733741 | 0.035557 | 0.001323 | 0.769298 |

数据：[rddr_phase2b17_correct_wrong_safety.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_correct_wrong_safety.csv)。

HA Benefit=0.733741，95% CI[0.728212,0.739200]，达到0.60。由CCA的0.868583下降后仍保留纠错能力，故D失败不是“完全不纠错”，主要在正确student保护不足。

## 20. Gradient coverage

| stratum | loss | targets | active_fraction | zero_rate | mean_gradient_norm |
| --- | --- | --- | --- | --- | --- |
| all | HA | 2479143 | 0.838245 | 0.161755 | 4.17084688e-04 |
| all | SA | 2479143 | 0.838245 | 0.161755 | 5.18772919e-04 |
| Top20 | HA | 485451 | 0.741883 | 0.258117 | 0.001192 |
| Top20 | SA | 485451 | 0.741883 | 0.258117 | 0.001644 |
| Bottom80 | HA | 1993692 | 0.861708 | 0.138292 | 2.28455348e-04 |
| Bottom80 | SA | 1993692 | 0.861708 | 0.138292 | 2.44717503e-04 |
| Rect_Correct | HA | 2027414 | 0.853607 | 0.146393 | 3.64496867e-04 |
| Rect_Correct | SA | 2027414 | 0.853607 | 0.146393 | 3.99785300e-04 |
| Rect_Wrong | HA | 451729 | 0.769298 | 0.230702 | 6.53105145e-04 |
| Rect_Wrong | SA | 451729 | 0.769298 | 0.230702 | 0.001053 |

数据：[rddr_phase2b17_gradient_coverage.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_gradient_coverage.csv)。

all ActiveGradientFraction=0.838245，95% CI[0.836472,0.840069]，明显大于0.10。真实参数路径也进行了反传；未创建optimizer，不以是否nonzero取代科学门限。

## 21. q × acceptance 二维审计

| quintile | acceptance | targets | accuracy_delta | net_repair | CCA_mean_dm | HA_mean_dm |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | Accept | 481157 | -0.001143 | -550 | -2.19340654e-06 | -3.63724499e-06 |
| Q1 | Reject | 14673 | -0.110339 | -1619 | -2.25422889e-05 | 0 |
| Q2 | Accept | 451060 | -0.006662 | -3005 | -3.72048555e-05 | -6.02938624e-05 |
| Q2 | Reject | 44768 | -0.158461 | -7094 | -1.18921068e-04 | 0 |
| Q3 | Accept | 412841 | -0.015524 | -6409 | -1.52199705e-04 | -2.33499085e-04 |
| Q3 | Reject | 82988 | -0.140358 | -11648 | -3.06536776e-04 | 0 |
| Q4 | Accept | 371135 | -0.029272 | -10864 | -3.63446787e-04 | -5.17348015e-04 |
| Q4 | Reject | 124693 | -0.134699 | -16796 | -7.08699952e-04 | 0 |
| Q5 | Accept | 361935 | -0.016235 | -5876 | -5.63606268e-04 | -7.29515632e-04 |
| Q5 | Reject | 133893 | -0.123046 | -16475 | -0.001334 | 0 |

数据：[rddr_phase2b17_q_acceptance_grid.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_q_acceptance_grid.csv)。

Q1–Q5沿用冻结q分位边界。Reject格的HA dM为0，这是拒绝操作本身，不是有益方向。二维结果用于描述Need与Trust，不据此选择某个分位区间训练。

## 22. Per-class acceptance 与统计功效

| stratum | Teacher_Win | Rect_Win | eligible_images | image_auroc | balanced_accuracy | power |
| --- | --- | --- | --- | --- | --- | --- |
| class0 | 15861 | 72985 | 1011 | 0.551667 | 0.530857 | SUFFICIENT |
| class1 | 47261 | 58098 | 1589 | 0.708264 | 0.658333 | SUFFICIENT |
| class2 | 12741 | 30816 | 645 | 0.403953 | 0.433642 | SUFFICIENT |
| class3 | 12427 | 6727 | 215 | 0.665475 | 0.586695 | SUFFICIENT |

数据：[rddr_phase2b17_per_class.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_per_class.csv)。

| stratum | acceptance_rate | accepted_accuracy_delta | HA_benefit_rate | HA_harm_rate | HA_mean_dm |
| --- | --- | --- | --- | --- | --- |
| class0 | 0.834058 | -0.036340 | 0.089971 | 0.744087 | -2.84427446e-04 |
| class1 | 0.836504 | 0.011842 | 0.143910 | 0.692594 | -1.78223933e-04 |
| class2 | 0.833077 | -0.051036 | 0.214432 | 0.618645 | -3.03439727e-04 |
| class3 | 0.889893 | 0.045896 | 0.314733 | 0.575160 | -1.54459732e-04 |

数据：[rddr_phase2b17_per_class.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_per_class.csv)。

四类均满足≥500正类、≥500负类、≥30双标签图，不能用underpowered解释本次失败。class2 image-AUROC=0.403953，CI[0.378316,0.429693]，与class1的0.708264差异明显。所有四类HA Mean_dM仍为负。本轮不翻转class2、不排除类别、不创建class-specific规则。

## 23. Boundary / interior

沿用冻结boundary≤7px、interior>7px标记；不改变计算尺度或重新定义边界。

| stratum | winner_image_auroc | winner_pooled_auroc | HA_benefit_rate | HA_harm_rate | HA_mean_dm | accepted_accuracy_delta |
| --- | --- | --- | --- | --- | --- | --- |
| boundary | 0.478573 | 0.494844 | 0.380732 | 0.427003 | 1.84266668e-06 | -0.005724 |
| interior | 0.637011 | 0.644190 | 0.121095 | 0.719843 | -2.55381178e-04 | -0.013454 |

数据：[rddr_phase2b17_boundary_interior.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_boundary_interior.csv)。

边界image-AUROC=0.478573，未表现出可靠winner判别。边界HA Mean_dM虽略正，也不能取代主all/Top20/class门限，更不能只报告该子组作为成功。

## 24. 冻结 HFRM transition groups

**历史名称by_CH实际为raw→完整HFRM transition，不是isolated CH因果分组。**

| stratum | targets | acceptance_rate | mean_delta | teacher_rect_accuracy_delta | HA_mean_dm |
| --- | --- | --- | --- | --- | --- |
| Corrected_by_CH | 305318 | 0.631522 | 0.015023 | -0.237277 | -7.65108904e-04 |
| Still_Wrong | 378130 | 0.798641 | 0.050526 | 0.120300 | 7.15978869e-04 |
| Harmed_by_CH | 69617 | 0.627375 | 0.014734 | 0.287286 | 6.81689455e-04 |
| Stable_Correct | 1726078 | 0.891992 | 0.031008 | -0.042513 | -3.85832026e-04 |

数据：[rddr_phase2b17_hfrm_groups.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_hfrm_groups.csv)。

Still_Wrong/Harmed_by_CH中的正向局部信号与Corrected_by_CH/Stable_Correct中的负向信号并存。这只能解释整体矛盾，不能据GT组别设计训练门控。

## 25. Detach / identity / BF16 / 独立验证

仅允许以下7个既有student参数张量求导；其余参数梯度始终None。teacher/q/Δ/m/a全部detach，语义来源分支与acceptance对rect的分支无梯度。共享ic1具有合法student梯度，不等于teacher分支漏梯度。

| loss | parameter | images | nonzero_images | RMS_min | RMS_max |
| --- | --- | --- | --- | --- | --- |
| HA | hfrm_28_1.context_conv.weight | 3418 | 3418 | 5.03241131e-04 | 0.025319 |
| HA | hfrm_28_1.veto_mlp.0.weight | 3418 | 3418 | 3.56120559e-06 | 1.97344088e-04 |
| HA | hfrm_28_1.veto_mlp.2.weight | 3418 | 3418 | 5.37060204e-06 | 2.37530448e-04 |
| HA | hfrm_28_1.gamma_context | 3418 | 3418 | 0.036444 | 2.716920 |
| HA | hfrm_28_1.gamma_veto | 3418 | 3418 | 5.80357388e-04 | 0.412815 |
| HA | ic1.weight | 3418 | 3418 | 0.005095 | 0.253971 |
| HA | ic1.bias | 3418 | 3418 | 0.005039 | 0.322885 |
| SA | hfrm_28_1.context_conv.weight | 3418 | 3418 | 4.04067730e-04 | 0.046795 |
| SA | hfrm_28_1.veto_mlp.0.weight | 3418 | 3418 | 2.81176732e-06 | 0.001200 |
| SA | hfrm_28_1.veto_mlp.2.weight | 3418 | 3418 | 4.23228074e-06 | 0.001806 |
| SA | hfrm_28_1.gamma_context | 3418 | 3418 | 3.00526619e-04 | 2.874253 |
| SA | hfrm_28_1.gamma_veto | 3418 | 3418 | 0.005122 | 1.934834 |
| SA | ic1.weight | 3418 | 3418 | 0.004038 | 0.485603 |
| SA | ic1.bias | 3418 | 3418 | 0.004517 | 0.453007 |

全部参数和BN buffer的state SHA前后相同：`c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5`。checkpoint SHA前后相同。严格加载missing_keys=[]，unexpected_keys=[]。

固定160图按32个等距索引+seed42抽取其余128图预先选定，未看结果挑图。原官方推理、background overwrite前的8,028,160个预测像素SHA前后相同：`23e333ad8e5168c464cda0cfdaae1bed085bc4d304172166e4ca54b95fca8b93`。raw forward固定160也完全一致；全部3,418张logits与冻结Phase16重放完全一致。不得将160图prediction SHA测试宣称为3418图完整官方分割评估。

| batch | loss_HA | loss_SA | seconds | allocated_GiB | reserved_GiB | budget_GiB | finite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 1.201588 | 1.449261 | 0.110906 | 1.155802 | 1.298828 | 22 | True |

上述显存仅限HFRM28_1/ic1选定参数反传，**不是全网络解冻Full25训练显存证明**。主GPU审计总计50.669s（含缓存加载/前向/验证与产物）；其中parity 6.540s、support 1.997s、实际逐图反传29.749s；统计分析11.965s。这些是本次一次运行耗时，不作重复benchmark或端到端训练速度预测。

**验证结果：29项unit/integration测试PASS、0skip；28项独立复核PASS。** 独立验证不导入主实现/分析器，使用显式邻居gather、SciPy tie-rank AUROC/AP、FP64 epsilon-KL解析导数以及NumPy gather-sum图像bootstrap。

| check | passed |
| --- | --- |
| immutable_source_hashes | True |
| observation_hash | True |
| all3418_order | True |
| frozen_replay | True |
| full3418_independent_support | True |
| rect_fp32_reconstruction | True |
| frozen_winner_counts | True |
| independent_ha_sa_analytic_gradient | True |
| independent_loss_denominator | True |
| all_gradients_finite | True |
| rejected_zero_gradient | True |
| accepted_direction_preserved | True |
| winner_and_gradient_rank_metrics | True |
| acceptance_coverage | True |
| accepted_rejected_quality | True |
| all_denominator_gradient_utility | True |
| fixed_direction_confidence_controls | True |
| paired_bootstrap_all_replicates | True |
| bootstrap_intervals | True |
| state_bn_checkpoint_identity | True |
| prediction_identity | True |
| teacher_acceptance_q_detached | True |
| no_optimizer_test_luad | True |
| bf16_batch20 | True |
| class_power | True |
| independent_gate_decision | True |
| secondary_flags | True |
| original_sources_unchanged | True |

| diagnostic | max_error |
| --- | --- |
| independent_support_max_abs | 3.57627869e-07 |
| independent_support_sign_mismatches | 3 |
| max_original_abs_delta_at_sign_mismatch | 1.19209290e-07 |
| rect_softmax_max_abs | 0 |
| analytic_gradient_max_abs | 9.99456778e-08 |
| loss_max_abs | 7.79369804e-07 |
| independent_fp32_gradient_replay_max_abs | 0 |
| independent_fp64_autograd_vs_formula_max_abs | 1.11022302e-16 |
| bootstrap_replicate_max_abs | 1.80443749e-09 |
| bootstrap_ci_max_abs | 1.59707270e-09 |

**数值事件完整披露。** report_r1已写出统计后，console JSON打印遇到0d NumPy对象序列化错误；仅修复console序列化。report_r2的初版验证使用绝对误差≤2e-8对比FP64解析导数和FP32实测梯度，在SA大小0.583535的梯度上出现9.99457e-8差异（相对1.71e-7），因此该验证当时FAIL。随后增加全3418图独立同精度FP32 autograd精确重放（max_abs=0），以及FP64 autograd对解析式验证（max_abs=1.11022e-16），区分舍入与公式错误。未改变任何主观察数组、损失、接受分数、科学门限或结果。原失败记录保留在服务器report_r2，report_r3为最终通过的复核。

独立support因求和顺序不同max_abs=3.57628e-7；共3个位置的Δ符号在FP32零附近不同，对应原|Δ|≤1.19209e-7。它满足预先实现的独立support 1e-6数值容差，但不是bitwise一致。主结果仍使用原固定Δ，不用独立重算值替换，也不增设“近零带”。bootstrap所有重复估计max差1.80444e-9，CI max差1.59707e-9。

原始证据：[rddr_phase2b17_verification.json](../audit/results/rddr_phase2b17/rddr_phase2b17_verification.json)、[unit_integration_tests.txt](../audit/results/rddr_phase2b17/unit_integration_tests.txt)、[rddr_phase2b17_parameter_gradients.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_parameter_gradients.csv)。

## 26. 10,000次配对图像bootstrap

seed42；从全部3,418图像索引有放回抽样，10,000次，每次重新计算原估计量。image AUROC在该重复内的双标签图上等权；pooled rates用重新汇总的像素计数。HA−CCA与Δ−q配对，不做像素bootstrap。全部报告指标均10,000个有效重复。95%区间为percentile描述区间，未作多重检验校正；不据此搜索成功子组。

| metric | estimate | ci_low | ci_high | valid_resamples |
| --- | --- | --- | --- | --- |
| winner:all:image_auroc | 0.619465 | 0.608924 | 0.630020 | 10000 |
| gradient:all:image_auroc | 0.523684 | 0.516922 | 0.530530 | 10000 |
| winner:class0:image_auroc | 0.551667 | 0.529860 | 0.573273 | 10000 |
| winner:class1:image_auroc | 0.708264 | 0.692895 | 0.723470 | 10000 |
| winner:class2:image_auroc | 0.403953 | 0.378316 | 0.429693 | 10000 |
| winner:class3:image_auroc | 0.665475 | 0.623295 | 0.707222 | 10000 |
| delta-minus-q:image_auroc | 0.002978 | -0.006820 | 0.012930 | 10000 |
| HA:all:mean_dm | -2.34511453e-04 | -2.43820521e-04 | -2.25118177e-04 | 10000 |
| HA:Top20:mean_dm | -4.77768753e-04 | -5.12369023e-04 | -4.43725373e-04 | 10000 |
| HA:class0:mean_dm | -2.84427446e-04 | -2.98611089e-04 | -2.70523410e-04 | 10000 |
| HA:class1:mean_dm | -1.78223933e-04 | -1.93992731e-04 | -1.62270505e-04 | 10000 |
| HA:class2:mean_dm | -3.03439727e-04 | -3.30462527e-04 | -2.75829042e-04 | 10000 |
| HA:class3:mean_dm | -1.54459732e-04 | -2.05790355e-04 | -9.94020392e-05 | 10000 |
| HA:Rect_Correct:harm_rate | 0.843256 | 0.841266 | 0.845268 | 10000 |
| HA:Rect_Wrong:benefit_rate | 0.733741 | 0.728212 | 0.739200 | 10000 |
| HA:all:active_fraction | 0.838245 | 0.836472 | 0.840069 | 10000 |
| HA-CCA:all:mean_dm | 5.49797917e-05 | 5.17188911e-05 | 5.82920665e-05 | 10000 |
| HA-CCA:all:harm_rate | -0.124305 | -0.126341 | -0.122261 | 10000 |
| HA-CCA:Top20:mean_dm | 2.03174731e-04 | 1.88095586e-04 | 2.19192262e-04 | 10000 |
| HA-CCA:Top20:harm_rate | -0.191649 | -0.198631 | -0.184869 | 10000 |
| HA-CCA:Rect_Correct:mean_dm | 6.13580453e-05 | 5.80618008e-05 | 6.46906095e-05 | 10000 |
| HA-CCA:Rect_Correct:harm_rate | -0.130643 | -0.132799 | -0.128471 | 10000 |
| HA-CCA:Rect_Wrong:mean_dm | 2.63534236e-05 | 1.94110417e-05 | 3.34900006e-05 | 10000 |
| HA-CCA:Rect_Wrong:harm_rate | -0.095861 | -0.099410 | -0.092456 | 10000 |
| Accepted:teacher-rect_accuracy_delta | -0.012850 | -0.015129 | -0.010563 | 10000 |
| Accepted:NetRepair_rate | -0.012850 | -0.015129 | -0.010563 | 10000 |
| Accepted:NetRepair_count | -26704 | -31438.175000 | -21953.875000 | 10000 |
| winner:Teacher-Win_recall | 0.699139 | 0.686769 | 0.711003 | 10000 |
| winner:Rect-Win_recall | 0.475579 | 0.464224 | 0.487030 | 10000 |
| winner:zero-sign_BA | 0.587359 | 0.579102 | 0.595715 | 10000 |

数据：[rddr_phase2b17_bootstrap.csv](../audit/results/rddr_phase2b17/rddr_phase2b17_bootstrap.csv)。

随机抽样索引流SHA256：`98e6164a3524dde42fc993cac0b5665076f7ebac7f6a73b7420d20c81022d00b`。完整重复值保存在bootstrap_replicates.csv。

## 27. Gate A / B / C / D：逐条判决

| gate | condition | observed | result |
| --- | --- | --- | --- |
| A | winner image-AUC ≥0.65 | 0.619465 | FAIL |
| A | 95% CI lower >0.50 | 0.608924 | PASS |
| A | zero-sign BA ≥0.60 | 0.587359 | FAIL |
| A | Teacher-Win recall ≥0.55 | 0.699139 | PASS |
| A | Rect-Win recall ≥0.55 | 0.475579 | FAIL |
| B | gradient image-AUC ≥0.65 | 0.523684 | FAIL |
| B | 95% CI lower >0.50 | 0.516922 | PASS |
| C | HA all Benefit > Harm | .142161 < .696084 | FAIL |
| C | HA Top20 Benefit > Harm | .245982 < .495902 | FAIL |
| C | positive Mean_dM in ≥5/6 strata | 0/6; all classes sufficiently powered | FAIL |
| D | HA Rect_Correct Harm ≤0.5×CCA | .843256 > .486950 | FAIL |
| D | HA Rect_Wrong Benefit ≥0.60 | 0.733741 | PASS |
| D | HA all ActiveGradientFraction ≥0.10 | 0.838245 | PASS |

A/B/C/D = **FAIL / FAIL / FAIL / FAIL**。Engineering=PASS。按批准的优先级，工程通过后A失败即给出CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED；其余失败仍全部披露。

## 28. SOFT_ACCEPTANCE_PROMISING

`TRUE`。SA all Mean_dM=-0.000179455851 > HA=-0.000234511453；SA Rect_Correct Harm=HA=0.843255990，满足不超过HA+5pp。这是相对secondary标志，不是主Go；SA的Mean仍负，不能据此绕过A/B/C/D开启训练。

## 29. STRONG_ACCEPTANCE_SIGNAL

`FALSE`。winner/gradient image-AUC均低于0.75；HA all和Top20 Mean_dM均非正；Rect_Correct Harm远高于0.25。虽然Rect_Wrong Benefit≥0.70，不能抵消其余必要条件失败。

## 30. 科学解释、局限与交付边界

1. 冻结的context支持差具有一定winner排序信息，却不足以决定什么时候teacher优于当前rect。接受区teacher accuracy仍显著低于rect，说明“邻域更一致”不等于“更正确”。
2. 局部梯度损害明显多于获益。HA通过拒绝减少伤害，但正确student保护不充分；它不能改变接受位置的教师梯度方向。
3. 失败不是没有工程连通、NaN、空loss或几乎全拒绝：83.82%前景梯度活跃，真实BF16反传和所有复核通过。
4. class2、边界与其他子群表现差异不能授权规则搜索；本轮不翻转类别或替换primary score。
5. 结论针对当前冻结S_T−S_R、零阈值和HA消费规则，不证明所有contextual acceptance思路都不可能。
6. dM仅是固定checkpoint的局部logit导数审计；没有参数步进、没有联合分类loss训练，因此不能把HarmRate解释为预测翻转率或断言Full25必然损失相同幅度的mIoU。

本轮代码和报告按独立分支交付，PR不自动merge。复现命令与目录索引见 [README_rddr_phase2b17.md](README_rddr_phase2b17.md)，交付清单见 [rddr_phase2b17_delivery_summary.md](rddr_phase2b17_delivery_summary.md)。归档只增加新目录，不删除旧实验、不覆盖C0和前三轮缓存。审计到此停止，不进入Phase-2B2/Full25。

## 31. Exact decision

主科学门限失败。不得据soft flag或局部子组擅自推进训练，也不提供本轮协议以外的优化补丁。

DECISION = CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED
