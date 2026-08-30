# RDDR Phase-2B1.6 Trainability & Integration Audit

Symmetric Adjudication · BCSS validation · zero-update · C0 Full25 seed42

结论：**teacher 信号保留，但本轮预注册的 conflict-weighted KL 消费方式未通过梯度安全门。**

Gate A/B/C/D = **PASS / PASS / FAIL / PASS**。没有启动 Full25，没有选择 lambda，没有更新或保存任何模型权重，没有访问 test/LUAD。

## 1. Provenance、冻结资产与样本范围

- Pure A0：`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`。分支：`feature/rddr-phase2b16-trainability`。
- 实际 GPU 审计代码：`417443a9f721b18c647472a1262ee0863b3331f7`；独立复算代码：`1f59baf115320a794e1d46ff84a625867ed51462`。后续提交仅补充报告/交付。
- checkpoint：`/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- checkpoint SHA256：`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Phase2B1 cache：`/home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz`
- SHA256：`767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a`
- Phase2B1.5 cache：`/home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz`
- SHA256：`237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514`
- 本轮梯度观测 SHA256：`5ab5a048e932f27726cea7507685354f984432fb66b542b87b805cea24a72bd5`。服务器 `/home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1` 保留完整 NPZ 和逐图梯度统计。

全部 **3418** validation 图像均执行 forward/backward。native28 共 2,679,712 个像素全部进入 loss 分母；2,479,143 个有效前景像素进入 GT 诊断，分布于 3416 张图像。另 2 张图像并未被丢弃，只是不含 native28 前景 GT。

GT 0–3；background/ignore 只排除于 GT 指标，不排除于 loss。本文所有语义指标均为 native28 四前景类 pooled confusion matrix，absent-union class=NA、不进入 macro mean。**不能与正式224分辨率、TTA/fusion后的 A0 mIoU 直接相减。**

原始 network、train_sshr.py、tool/infer_fun.py、iouutils 和 optimizer 文件零修改。合同见 [冻结合同](rddr_phase2b16_contract.md)。

## 2. Exact commands 与环境

运行目录：`/home/duyanhong/DZWdeRepo-rddr-phase2b16`。GPU `NVIDIA GeForce RTX 5090 D v2`；PyTorch `2.11.0+cu128`；NumPy `1.23.5`。

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b16_trainability_audit.py --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --val-images /home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img --output /home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/analyze_rddr_phase2b16.py --run /home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1 --output /home/duyanhong/experiments/RDDR_PHASE2B16/report_r1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/verify_rddr_phase2b16.py --run /home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1 --report /home/duyanhong/experiments/RDDR_PHASE2B16/report_r1
```

BF16 forward、FP32 softmax/KL；batch1；cudnn.benchmark=False，matmul precision=none，conv precision=tf32，与冻结 cache backend 一致。每个 loss 的系数均为1。本轮没有 optimizer，也没有正式训练命令。

## 3. Teacher/q 重放一致性

| 量 | 最大绝对误差 |
| --- | --- |
| T_SS | 0.000000 |
| T_SD | 0.000000 |
| T_DS | 0.000000 |
| T_DD | 0.000000 |
| sym | 0.000000 |
| wD_sym | 0.000000 |
| anchor_sym | 0.000000 |
| q | 5.960464e-08 |

T_SS/T_SD/T_DS/T_DD、Delta_sym、wD、teacher 均逐位一致；q 的 5.96e-8 差别来自冻结 NumPy division 与 Torch division 的 FP32 舍入，小于合同1e-7。主审计直接使用原始缓存 q，没有重定义。重新前向 p_s、p_d 对缓存最大误差均为0。

独立 FP64 公式复算 q 与冻结 FP32 的差为 2.523090e-07；这是额外跨精度检查，不用于替代或放宽前面的原实现1e-7门限。

## 4. 三个冻结 loss

```text
p_rect = softmax(L_rect.float())
KL_i = sum_k t_ik * (log(t_ik+1e-8) - log(p_rect_ik+1e-8))
U   = mean_i KL(sym_teacher || rect)
FA  = sum_i q_i*KL(fixedavg_teacher || rect)/(sum_i q_i+1e-8)
CCA = sum_i q_i*KL(sym_teacher || rect)/(sum_i q_i+1e-8)
```

| Loss | 逐图均值 | 最小 | 最大 |
| --- | --- | --- | --- |
| U | 0.868045 | 0.261509 | 2.883681 |
| FA | 1.435805 | 0.309569 | 7.404502 |
| CCA | 1.282222 | 0.301828 | 6.034714 |

所有 native28 位置进入 loss；GT 不进入公式。主结果按单图分母归一化，batch20 smoke 按整个 batch 分母归一化，二者原始梯度幅度不能不加说明地互比。

## 5. q 的角色与数学限制

`q = JS(p_s,p_d)/ln2` 保持原公式、原缓存、原 quintile 和 Top20 定义。它只分配 loss 权重，不改变 feature/context 幅度。

对同一 teacher、q>0 时：`g_CCA(i) = [784*q_i/(sum_j q_j+eps)] * g_U(i)`。这是正标量缩放，**可以重新分配梯度大小，但不能修正该像素的方向符号**。

有效前景 q>0 数量=2,479,143，q=0数量=0。正比例梯度最大 FP32 误差=3.725290e-09；dM 符号不一致像素=0。U/CCA BenefitRate 相同不是两个独立发现，也不是实现错误。

## 6. Detach 和梯度范围

teacher/q 全部 detached；真实 batch20 从本批 p_s/p_d 重建 teacher 后，teacher-source 的 p_s.grad、p_d.grad 均为 None。raw CAM 复用 ic1 时，ic1 的 student 梯度合法存在，不能误当作 teacher 泄漏。

只允许 HFRM28_1 的 context_conv、两层 veto_mlp、gamma_context/gamma_veto，以及 ic1.weight/bias 七个 tensor 求导。其余参数梯度全部 None，全部模块 eval，BN buffers 不变。程序同时阻止 optimizer 构造和 checkpoint 写出。

[rddr_phase2b16_detach_audit.json](../audit/results/rddr_phase2b16/rddr_phase2b16_detach_audit.json)

## 7. Teacher / FixedAvg / Rect 的语义指标

| 分布 | Accuracy % | mIoU % | Dice % | NLL | Brier |
| --- | --- | --- | --- | --- | --- |
| rect | 81.7788 | 63.7895 | 77.5312 | 0.840694 | 0.303716 |
| fixed | 77.4087 | 57.3520 | 72.3531 | 0.681292 | 0.341498 |
| teacher | 78.5383 | 59.3171 | 74.0518 | 0.659663 | 0.326990 |

| 分布 | Class0 IoU % | Class1 IoU % | Class2 IoU % | Class3 IoU % |
| --- | --- | --- | --- | --- |
| rect | 74.9891 | 69.7635 | 57.2253 | 53.1800 |
| fixed | 69.9399 | 64.0964 | 51.4291 | 43.9427 |
| teacher | 70.8893 | 65.4297 | 52.2020 | 48.7472 |

Sym teacher 比 FixedAvg **+1.9651 pp mIoU**，但比当前 rectified distribution **-4.4724 pp mIoU**；accuracy 比 rect 低3.2405 pp。Gate A 检验的是“优于 FixedAvg”，不是“优于当前 student”。teacher NLL 较低而 Brier 较高，说明分类正确性与概率校准/置信度不能混为一个指标。

[rddr_phase2b16_teacher_metrics.csv](../audit/results/rddr_phase2b16/rddr_phase2b16_teacher_metrics.csv)

## 8. Repair / Harm / NetRepair

| 分组 | Teacher | Repair | Harm | Net count | Net rate pp |
| --- | --- | --- | --- | --- | --- |
| all | fixed | 90259 | 198601 | -108342 | -4.3701 |
| all | teacher | 88290 | 168626 | -80336 | -3.2405 |
| Top20 | fixed | 50280 | 94855 | -44575 | -9.1822 |
| Top20 | teacher | 48666 | 72297 | -23631 | -4.8678 |
| Bottom80 | fixed | 39979 | 103746 | -63767 | -3.1984 |
| Bottom80 | teacher | 39624 | 96329 | -56705 | -2.8442 |
| Q1 | fixed | 679 | 2849 | -2170 | -0.4377 |
| Q1 | teacher | 678 | 2847 | -2169 | -0.4374 |
| Q2 | fixed | 4425 | 14623 | -10198 | -2.0568 |
| Q2 | teacher | 4419 | 14518 | -10099 | -2.0368 |
| Q3 | fixed | 10680 | 29310 | -18630 | -3.7573 |
| Q3 | teacher | 10626 | 28683 | -18057 | -3.6418 |
| Q4 | fixed | 21286 | 51466 | -30180 | -6.0868 |
| Q4 | teacher | 21094 | 48754 | -27660 | -5.5785 |
| Q5 | fixed | 53189 | 100353 | -47164 | -9.5122 |
| Q5 | teacher | 51473 | 73824 | -22351 | -4.5078 |
| boundary | fixed | 17799 | 20338 | -2539 | -1.2623 |
| boundary | teacher | 17055 | 18303 | -1248 | -0.6205 |
| interior | fixed | 72460 | 178263 | -105803 | -4.6446 |
| interior | teacher | 71235 | 150323 | -79088 | -3.4718 |
| class0 | fixed | 15844 | 83627 | -67783 | -7.0232 |
| class0 | teacher | 15861 | 72985 | -57124 | -5.9188 |
| class1 | fixed | 46618 | 76031 | -29413 | -2.8155 |
| class1 | teacher | 47261 | 58098 | -10837 | -1.0373 |
| class2 | fixed | 13806 | 32076 | -18270 | -5.6473 |
| class2 | teacher | 12741 | 30816 | -18075 | -5.5871 |
| class3 | fixed | 13991 | 6867 | 7124 | 4.8860 |
| class3 | teacher | 12427 | 6727 | 5700 | 3.9094 |
| Deep-Wrong | fixed | 3484 | 186962 | -183478 | -31.8220 |
| Deep-Wrong | teacher | 3913 | 160399 | -156486 | -27.1406 |
| Shallow-Wrong | fixed | 62241 | 97962 | -35721 | -5.0424 |
| Shallow-Wrong | teacher | 59873 | 94457 | -34584 | -4.8819 |
| Both-Wrong | fixed | 399 | 86323 | -85924 | -21.8260 |
| Both-Wrong | teacher | 429 | 86230 | -85801 | -21.7948 |
| Rect_Correct | fixed | 0 | 198601 | -198601 | -9.7958 |
| Rect_Correct | teacher | 0 | 168626 | -168626 | -8.3173 |
| Rect_Wrong | fixed | 90259 | 0 | 90259 | 19.9808 |
| Rect_Wrong | teacher | 88290 | 0 | 88290 | 19.5449 |

整体 teacher 修复88,290个、损伤168,626个，净 -80,336（-3.2405 pp）；比 FixedAvg 的 -108,342 好，但仍是净负向。

`NetRepair rate = teacher accuracy - rect accuracy` 恒等式最大误差=1.075529e-16。相对共同 rect 参照，teacher-vs-fixed 的 NetRepair 差就是 accuracy 差，不能将二者重复计算成独立证据。

## 9. GT probability advantage 与 q 分层 teacher utility

| 分组 | 比较 | Mean ΔP_GT | Median | 正向 % | 负向 % |
| --- | --- | --- | --- | --- | --- |
| all | fixed | -0.123134 | -0.072826 | 16.3973 | 83.6026 |
| all | teacher | -0.117745 | -0.074009 | 16.5167 | 83.4832 |
| all | teacher-minus-fixed | 0.005389 | -1.180172e-05 | 43.5648 | 55.8956 |
| Top20 | fixed | -0.205996 | -0.276477 | 27.6154 | 72.3846 |
| Top20 | teacher | -0.184313 | -0.260047 | 28.1415 | 71.8585 |
| Top20 | teacher-minus-fixed | 0.021683 | 0.012172 | 66.0730 | 33.9262 |
| Bottom80 | fixed | -0.102958 | -0.057860 | 13.6658 | 86.3341 |
| Bottom80 | teacher | -0.101536 | -0.059209 | 13.6861 | 86.3137 |
| Bottom80 | teacher-minus-fixed | 0.001422 | -3.236532e-05 | 38.0842 | 61.2450 |
| Rect_Correct | fixed | -0.175991 | -0.109202 | 2.0577 | 97.9421 |
| Rect_Correct | teacher | -0.169585 | -0.111465 | 1.9375 | 98.0624 |
| Rect_Correct | teacher-minus-fixed | 0.006406 | -3.683567e-05 | 37.8033 | 61.5370 |
| Rect_Wrong | fixed | 0.114094 | 0.056372 | 80.7548 | 19.2452 |
| Rect_Wrong | teacher | 0.114920 | 0.062438 | 81.9498 | 18.0502 |
| Rect_Wrong | teacher-minus-fixed | 8.255936e-04 | 0.001126 | 69.4230 | 30.5763 |

| 分位 | Teacher | NetRepair pp | ΔP_GT | Mean KL |
| --- | --- | --- | --- | --- |
| Q1 | fixed | -0.4377 | -0.018946 | 0.151822 |
| Q1 | teacher | -0.4374 | -0.019006 | 0.152410 |
| Q2 | fixed | -2.0568 | -0.074878 | 0.572546 |
| Q2 | teacher | -2.0368 | -0.075298 | 0.576214 |
| Q3 | fixed | -3.7573 | -0.128977 | 0.957178 |
| Q3 | teacher | -3.6418 | -0.129327 | 0.958263 |
| Q4 | fixed | -6.0868 | -0.177942 | 1.273783 |
| Q4 | teacher | -5.5785 | -0.174829 | 1.242856 |
| Q5 | fixed | -9.5122 | -0.214928 | 1.561783 |
| Q5 | teacher | -4.5078 | -0.190266 | 1.375841 |

高冲突只表示分歧/监督需求，不自动保证 teacher 在该位置可靠。本轮不根据这些统计修改 q。完整 advantage 分层见 [rddr_phase2b16_teacher_advantage.csv](../audit/results/rddr_phase2b16/rddr_phase2b16_teacher_advantage.csv)。

## 10. Logit gradient

| Loss | 分组 | Mean pixel L2 | RMS | Max abs | Finite |
| --- | --- | --- | --- | --- | --- |
| U | all | 3.014689e-04 | 2.162903e-04 | 0.001268 | True |
| U | Top20 | 5.667474e-04 | 3.146206e-04 | 0.001268 | True |
| U | Bottom80 | 2.368753e-04 | 1.845806e-04 | 0.001268 | True |
| FA | all | 5.056201e-04 | 5.058471e-04 | 0.014579 | True |
| FA | Top20 | 0.001508 | 9.896950e-04 | 0.014579 | True |
| FA | Bottom80 | 2.614457e-04 | 2.822866e-04 | 0.010394 | True |
| CCA | all | 4.651258e-04 | 4.447411e-04 | 0.014534 | True |
| CCA | Top20 | 0.001346 | 8.581006e-04 | 0.014534 | True |
| CCA | Bottom80 | 2.506125e-04 | 2.581922e-04 | 0.010275 | True |

直接对 FP32 `L_rect` 求导；返回形状为1×4×28×28。独立解析验证使用带 epsilon 的准确导数，而不是将其近似为 p-t。`a=t*p/(p+eps); g=w*(p*sum(a)-a)`。

FP64 解析梯度与实测 FP32 autograd 最大误差=4.368167e-09。

## 11. 一阶 GT-margin 与 tie 处理

`M=L_GT-max_nonGT L`；`DeltaL=-g`；`dM=DeltaL_GT-max(DeltaL_k | k在当前max非GT并列集合)`。这里取的是方向导数，不随意选择一个 tie index。

前景共 2,698 个像素出现并列最大非GT logit（Top20=298）。使用严格 >0/<0，不调整数值阈值。

**dM<0 表示无穷小辅助梯度降低 GT margin，不等于该像素已经预测错误，更不等于一次真实训练后必然改变类别。**本轮禁止任何真实参数更新，诊断属于已预注册的局部安全代理指标。

## 12. BenefitRate / HarmRate 与 Correct-Wrong 安全性

| Loss | 分组 | Benefit % | Harm % | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- |
| U | all | 17.9611 | 82.0389 | -1.947693e-04 | -1.265398e-04 |
| U | Top20 | 31.2450 | 68.7550 | -2.779043e-04 | -4.999082e-04 |
| U | Bottom80 | 14.7265 | 85.2735 | -1.745265e-04 | -1.009946e-04 |
| U | Rect_Correct | 2.6101 | 97.3899 | -3.359967e-04 | -2.023627e-04 |
| U | Rect_Wrong | 86.8583 | 13.1417 | 4.390764e-04 | 3.474372e-04 |
| FA | all | 17.8477 | 82.1523 | -3.325998e-04 | -5.113664e-05 |
| FA | Top20 | 30.8109 | 69.1891 | -8.541570e-04 | -7.941824e-04 |
| FA | Bottom80 | 14.6912 | 85.3088 | -2.056041e-04 | -3.215987e-05 |
| FA | Rect_Correct | 2.7154 | 97.2846 | -5.731127e-04 | -1.165886e-04 |
| FA | Rect_Wrong | 85.7634 | 14.2366 | 7.468508e-04 | 2.767860e-04 |
| CCA | all | 17.9611 | 82.0389 | -2.894912e-04 | -5.164161e-05 |
| CCA | Top20 | 31.2450 | 68.7550 | -6.809435e-04 | -7.766086e-04 |
| CCA | Bottom80 | 14.7265 | 85.2735 | -1.941752e-04 | -3.277585e-05 |
| CCA | Rect_Correct | 2.6101 | 97.3899 | -5.191884e-04 | -1.196841e-04 |
| CCA | Rect_Wrong | 86.8583 | 13.1417 | 7.414171e-04 | 2.964932e-04 |

CCA 整体 Benefit **17.9611%**、Harm **82.0389%**；Top20 Benefit **31.2450%**、Harm **68.7550%**。Rect_Wrong 有86.8583%得到正向 margin 推力，但 Rect_Correct 有97.3899%的 margin 下降，后者占总体多数。这可能包含向软 teacher 置信度收缩的效应；不能直接等价为97.39%的正确像素被分错。

[rddr_phase2b16_gradient_semantic_utility.csv](../audit/results/rddr_phase2b16/rddr_phase2b16_gradient_semantic_utility.csv)

## 13. Conflict localization

| Loss | Top20 mean G | Bottom80 mean G | Top/Bottom | Q1 | Q5 |
| --- | --- | --- | --- | --- | --- |
| U | 5.667474e-04 | 2.368753e-04 | 2.392598 | 3.537033e-05 | 5.797648e-04 |
| FA | 0.001508 | 2.614457e-04 | 5.769523 | 2.786961e-06 | 0.001655 |
| CCA | 0.001346 | 2.506125e-04 | 5.371269 | 2.794787e-06 | 0.001470 |

CCA Top/Bottom=5.3713，高于 U 的2.3926；Q5>Q1，所以 Gate B PASS。但是权重更集中并未改变逐像素梯度方向；本次加权后整体和Top20的平均 dM 比 U 更负。

## 14. 四类梯度安全与样本充分性

| 类别 | 像素 | 图像 | Benefit % | Harm % | Mean dM | Mean G | Power |
| --- | --- | --- | --- | --- | --- | --- | --- |
| class0 | 965133 | 2018 | 11.7828 | 88.2172 | -3.602750e-04 | 4.121041e-04 | SUFFICIENT |
| class1 | 1044692 | 2676 | 17.8344 | 82.1656 | -2.557249e-04 | 4.594387e-04 | SUFFICIENT |
| class2 | 323515 | 935 | 27.9267 | 72.0733 | -2.668602e-04 | 5.570011e-04 | SUFFICIENT |
| class3 | 145803 | 384 | 37.6529 | 62.3471 | -1.130968e-04 | 6.529907e-04 | SUFFICIENT |

充分性阈值为≥500前景像素且≥30图像。class3 为145,803像素/384图像，**不再沿用上一轮 Shallow-Win 子集的 underpowered 标记**。四类均充分、四类 mean dM 均负；加上 all/Top20，正向为 **0/6**，低于要求的≥5/6。

## 15. F28_rect feature gradient

| 分组 | RMS | Mean pixel L2 | Max abs | Finite |
| --- | --- | --- | --- | --- |
| all | 5.060991e-05 | 5.970106e-04 | 0.008301 | True |
| Top20 | 9.768156e-05 | 0.001731 | 0.008301 | True |
| Bottom80 | 2.935472e-05 | 3.209099e-04 | 0.006012 | True |
| Q1 | 6.378885e-07 | 3.595560e-06 | 1.754761e-04 | True |
| Q2 | 5.194334e-06 | 6.152172e-05 | 7.972717e-04 | True |
| Q3 | 1.688958e-05 | 2.651958e-04 | 0.001503 | True |
| Q4 | 4.353514e-05 | 7.659917e-04 | 0.003113 | True |
| Q5 | 1.029508e-04 | 0.001889 | 0.008301 | True |
| boundary | 5.922881e-05 | 7.818572e-04 | 0.008301 | True |
| interior | 4.977722e-05 | 5.806889e-04 | 0.007874 | True |
| class0 | 5.197881e-05 | 5.429062e-04 | 0.008301 | True |
| class1 | 4.922116e-05 | 5.815168e-04 | 0.008240 | True |
| class2 | 4.940761e-05 | 6.884783e-04 | 0.004700 | True |
| class3 | 5.377696e-05 | 8.632125e-04 | 0.005524 | True |
| Deep-Wrong | 8.078201e-05 | 0.001180 | 0.008301 | True |
| Shallow-Wrong | 7.078701e-05 | 0.001092 | 0.008301 | True |
| Both-Wrong | 6.096526e-05 | 7.729276e-04 | 0.008301 | True |
| Rect_Correct | 4.724520e-05 | 5.354886e-04 | 0.007050 | True |
| Rect_Wrong | 6.355386e-05 | 8.731289e-04 | 0.008301 | True |

所有3418张图像 feature gradient 有限且非零；RMS按512通道统计，pixel L2对512通道求范数。GT分层只用于统计，不参与反向图。

## 16. Parameter gradient

| 参数 | Pooled RMS | Max abs | Mean nonzero fraction | 全零梯度图像数 | Finite |
| --- | --- | --- | --- | --- | --- |
| hfrm_28_1.context_conv.weight | 0.010039 | 0.660156 | 0.992844 | 0 | True |
| hfrm_28_1.veto_mlp.0.weight | 5.785001e-05 | 0.009399 | 0.553485 | 0 | True |
| hfrm_28_1.veto_mlp.2.weight | 7.799452e-05 | 0.024292 | 0.553608 | 0 | True |
| hfrm_28_1.gamma_context | 1.037286 | 2.852733 | 1.000000 | 0 | True |
| hfrm_28_1.gamma_veto | 0.105379 | 0.353083 | 1.000000 | 0 | True |
| ic1.weight | 0.099760 | 5.343750 | 0.997225 | 0 | True |
| ic1.bias | 0.139470 | 0.527344 | 1.000000 | 0 | True |

Pooled RMS = sqrt(sum_image sum_param grad² / (images×parameter_count))。只记录，不进行 optimizer step。逐图、逐参数完整记录见 [rddr_phase2b16_parameter_per_image.csv](../audit/results/rddr_phase2b16/rddr_phase2b16_parameter_per_image.csv)。

## 17. Gradient path attribution

| 分支 | Pooled L2 | L2/total | 平方能量占比 % | 全零图像数 |
| --- | --- | --- | --- | --- |
| context | 208.236967 | 0.618480 | 38.2517 | 0 |
| semantic | 8.209462 | 0.024383 | 0.0595 | 0 |
| head | 264.445321 | 0.785423 | 61.6889 | 0 |

context/semantic/head 三路均连通，并非只有 head 收到梯度。注意 semantic 平方能量占比仅约 **0.05945%**，明显小于 context 38.2517% 和 head 61.6889%；这不触发本轮仅要求非零的 Gate D，但也不能宣称各路得到了同等强度学习。不同参数量/参数化影响此比例，它不是因果贡献率。L2比例本身不要求和为1；平方能量比例和为1。

## 18. 数值稳定性与固定样本

预先冻结32个等距图像索引；seed42从剩余索引抽128个、不放回。160张均包含于全量3418反向审计，并额外重放全部原始 forward 返回tensor，逐位一致。全部loss/logits/logit gradients/feature gradients/parameter gradients有限，未发生NaN/Inf。

[rddr_phase2b16_selection.json](../audit/results/rddr_phase2b16/rddr_phase2b16_selection.json)；[rddr_phase2b16_runtime.json](../audit/results/rddr_phase2b16/rddr_phase2b16_runtime.json)。

## 19. Batch20 BF16 smoke 与资源

- 固定 deterministic32 的前20张，224×224，BF16 forward / FP32 loss，真实 teacher construction + backward。
- Loss=1.267646；耗时=0.100000 s。
- Peak allocated=1.1595 GiB；reserved=1.2871 GiB；预算22 GiB，通过。
- 全量 teacher重放=3.888s；全量3418梯度审计=24.973s；本轮runner=42.433s。

资源值为服务器程序计时器与 CUDA memory API 的实测，不是25epoch训练时间预测。**本轮只开放HFRM28_1/ic1反向，backbone没有梯度图；不能拿1.29GiB推断完全解冻训练的显存。**

[rddr_phase2b16_bf16_smoke.json](../audit/results/rddr_phase2b16/rddr_phase2b16_bf16_smoke.json)

## 20. Inference identity / zero-update identity

- 原始 state_dict 所有parameter+buffer前后哈希：`c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5`（相同）。
- checkpoint文件前后SHA：`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`（相同）。
- 固定160张完整官方推理、共8,028,160像素，原始预测前后SHA：`23e333ad8e5168c464cda0cfdaae1bed085bc4d304172166e4ca54b95fca8b93`（相同）。
- 原始三路TTA、BCSS class thresholds=[0.8,0.9,0.8,0.6]、CAM融合=[0.6,0.2,0.2]、插值、normalization、presence与argmax流程未变。
- 只在audit中限制Dataset到固定160，并拦截最终scores入口计算原始prediction哈希，避免background overwrite影响identity证据。
- 原始checkpoint strict load：missing_keys=[]，unexpected_keys=[]。

CCA helper 不接入model.forward或infer，off不构造teacher/support；on只增加审计loss和梯度，零step后输出不变。没有为未来训练新增CLI或更改训练入口。本轮证实当前零更新路径的identity，不是尚未实现的未来训练runner测试。

[rddr_phase2b16_identity_audit.json](../audit/results/rddr_phase2b16/rddr_phase2b16_identity_audit.json)

## 21. Third-evidence holdout 与为什么不用 forward 注入

teacher严格只有 `wS*p_s+wD*p_d`；没有ctx_sym、第三类恢复、Both-Wrong detector、额外loss。Both-Wrong只作为GT事后分析分组，未被训练模块读取。

历史 Phase1 direct feature manipulation、Phase2A context amplitude manipulation 和 Phase2B1 naive anchor fusion 的风险背景下，本轮只检查teacher-only supervision；不做feature replacement/context residual replacement/direct anchor overwrite/fixed inference fusion。本轮失败也不构成回退到上述路径的授权。

## 22. 10,000次配对 image bootstrap 与独立验证

| 估计量 | Point | 95% CI lower | 95% CI upper |
| --- | --- | --- | --- |
| teacher-fixed_miou | 0.019651 | 0.018105 | 0.021300 |
| teacher-fixed_accuracy | 0.011297 | 0.010716 | 0.011902 |
| teacher-vs-rect_NetRepair | -0.032405 | -0.034847 | -0.029959 |
| fixed-vs-rect_NetRepair | -0.043701 | -0.046381 | -0.040993 |
| CCA:all:mean_dm | -2.894912e-04 | -2.991819e-04 | -2.798003e-04 |
| CCA:Top20:mean_dm | -6.809435e-04 | -7.176667e-04 | -6.445952e-04 |
| CCA:class0:mean_dm | -3.602750e-04 | -3.741680e-04 | -3.463570e-04 |
| CCA:class1:mean_dm | -2.557249e-04 | -2.721233e-04 | -2.393987e-04 |
| CCA:class2:mean_dm | -2.668602e-04 | -2.961828e-04 | -2.368800e-04 |
| CCA:class3:mean_dm | -1.130968e-04 | -1.669129e-04 | -5.476673e-05 |
| CCA:all:benefit_rate | 0.179611 | 0.173122 | 0.186137 |
| CCA-U:all:benefit_rate | 0.000000 | 0.000000 | 0.000000 |
| CCA-U:all:mean_dm | -9.472194e-05 | -9.942879e-05 | -9.006505e-05 |
| CCA-U:Top20:mean_dm | -4.030392e-04 | -4.278967e-04 | -3.789360e-04 |
| CCA-FA:all:benefit_rate | 0.001134 | 9.817808e-04 | 0.001290 |
| CCA-FA:all:mean_dm | 4.310858e-05 | 4.146808e-05 | 4.476581e-05 |
| CCA-FA:Top20:mean_dm | 1.732135e-04 | 1.642573e-04 | 1.824766e-04 |

概率差/accuracy/mIoU表内单位是0–1；换算pp需×100；dM维持原始logit导数单位。image-level、paired、seed42，每次重采样重新pool confusion/count/sum，不是对像素独立bootstrap，也不是平均逐图mIoU。

独立NumPy复算 `PASS`：28项通过。完整复算不导入原loss/analyzer，使用FP64解析梯度、显式真值/预测mask构建confusion、非GT索引gather处理ties，并以gather-sum重做全部bootstrap。

| 独立校验 | 最大绝对差 |
| --- | --- |
| teacher_formula_max_abs | 0.000000 |
| wd_max_abs | 0.000000 |
| q_float64_vs_frozen_max_abs | 2.523090e-07 |
| teacher_metric_max_abs | 0.000000 |
| teacher_advantage_max_abs | 0.000000 |
| loss_float64_vs_autograd_max_abs | 8.511775e-07 |
| analytic_gradient_max_abs | 4.368167e-09 |
| semantic_utility_max_abs | 0.000000 |
| bootstrap_replicate_max_abs | 2.222614e-18 |
| bootstrap_ci_max_abs | 1.084202e-18 |

关键六组CCA mean_dM的95% CI均完全小于0；不是class3样本不足或bootstrap跨零。完整证据见 [rddr_phase2b16_verification.json](../audit/results/rddr_phase2b16/rddr_phase2b16_verification.json) 与 [rddr_phase2b16_bootstrap_replicates.csv](../audit/results/rddr_phase2b16/rddr_phase2b16_bootstrap_replicates.csv)。

## 23. Gate A/B/C/D 判定

| Gate | 结果 | 依据 |
| --- | --- | --- |
| A Teacher superiority | PASS | 优于FixedAvg；ΔmIoU +1.9651pp，95%CI [+1.8105,+2.1300]pp；accuracy/相对NetRepair也更好。后两者不是独立证据。 |
| B Conflict localization | PASS | Top20>Bottom80，5.3713>2.3926，Q5>Q1。 |
| C Gradient semantic utility | FAIL | all/Top20 Benefit<Harm，关键mean_dM正向0/6；四类统计充分。 |
| D Engineering | PASS | 三路非零且有限，detach正确，batch20通过、显存符合预算，零step、权重/推理不变。 |

使用已批准的优先级D>A>B>C。这里不是工程失败，也不是teacher相对FixedAvg失去优势；是局部KL梯度安全性失败。

## 24. Secondary preference flag

**ADJUDICATION_TEACHER_PREFERRED = TRUE**。CCA相对FA的整体BenefitRate +0.1134pp，Top20 mean_dM +1.7321e-4。这是“比FA更好”，不是“绝对安全”：两者整体/Top20 mean_dM都为负。不能用secondary flag覆盖Gate C。

## 25. Scientific interpretation、限制和交付

本轮支持 **Outcome B**：symmetric adjudication的相对语义信号成立，但当前KL-style消费方式没有通过已冻结的GT-margin安全标准。

1. current rect在native28分类上比teacher强；向teacher贴近会同时纠正一些错误、压低大量正确像素的GT margin。
2. q强调冲突幅度，不是teacher正确性；它不能改变U/CCA的像素梯度符号。本次局部加权更集中，但总体方向仍不安全。
3. symmetric teacher较FA好，不能自动推导其优于student，更不能推导Full25增益。
4. “margin下降”可能包含正常软目标置信度收缩。此次结论是预注册代理门失败，**不是已经测得Full25 mIoU下降**，也不证明所有lambda或所有KL训练都失败。
5. 当前只测单位系数辅助loss的局部梯度，没有测它与SSHR分类loss的合成方向、真实更新或长期泛化。不能绕过本合同去做lambda搜索。

本轮完成：独立A0分支、GT-blind helper、单元/真实GPU安全测试、全量3418诊断、10k bootstrap、独立复算、CSV/JSON/Markdown。模型和训练/推理/metric源码未变；没有数据删除或覆盖。仅新增约152MiB服务器梯度观测与文本统计。

如继续研究，需用户另行定义teacher consumption假设与预注册协议；本报告不自行设计、测试或推荐某个未验证补丁。不选择lambda，不启动Full25，不测试其他seed/LUAD/test。命令/目录见 [执行README](README_rddr_phase2b16.md)。

## 26. Exact decision 与停止条件

已完成审计并停止。保留symmetric adjudication信号证据；**本轮conflict-weighted KL路线不进入Full25**。下一阶段必须重新明确并审核执行方案。

[rddr_phase2b16_summary.json](../audit/results/rddr_phase2b16/rddr_phase2b16_summary.json)

DECISION = TEACHER_SIGNAL_PRESENT_GRADIENT_UNSAFE
