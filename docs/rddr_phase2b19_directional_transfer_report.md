# RDDR-Net Phase-2B1.9 Directional Transfer Audit

日期：2026-08-30。Validation-only / zero-training / zero-optimizer-step。

## 1. Provenance / SHA256 / commands

本报告对应已确认合同的完整执行，不是训练结果。唯一 checkpoint 为 C0 Full25 BCSS seed42，全部3418张 validation；无 test、LUAD、train split、optimizer 或参数更新。

纯 A0：`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`。GPU执行 commit：`5f74edae64d36d2eeb4cbb716b37479f67b9ed9d`；独立复核 commit：`6243212531f7e4ba0e76e07af9ca346d99fbe693`。
独立分支：`feature/rddr-phase2b19-directional-transfer`；PR目标：`baseline/official-a0`，不自动合并。
合同文件 SHA256：`97fcd0c381681d7dcce22a324fee44dda0158b44cc97cfca27158fddaf2a9ebe`。

| 资产 | 服务器路径 | SHA256 |
| --- | --- | --- |
| native | /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz | 767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a |
| derived | /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz | 237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514 |
| previous | /home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1/rddr_phase2b18_observations.npz | 740b2e80c9182e701509f5ed7a6fab4c8f1b6ddcd585df418ccefa9f288d8e52 |
| checkpoint | /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth | 509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579 |


新 observation SHA256：`d4f65c519920c010e307ba8f32fb8e110387e0e14db73baa7c43163072ad0f1a`。
随机门 SHA256：`6f5e9e2a667fc96f6246fbd5990249b9f0cbffba4fffa626075c976171876d2a`。

实际命令（已执行，重放时必须使用新输出目录）：

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b19_audit.py --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz --previous /home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1/rddr_phase2b18_observations.npz --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --val-images /home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img --output /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/analyze_rddr_phase2b19.py --run /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1 --output /home/duyanhong/experiments/RDDR_PHASE2B19/report_r1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/verify_rddr_phase2b19.py --run /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1 --report /home/duyanhong/experiments/RDDR_PHASE2B19/report_r1
```

只在独立 tools/tests/docs/results 中新增审计；原 network/tool/train_sshr.py 与 A0 无差异。旧实验全部保留。

## 2. Frozen evidence

复用 Phase2B1 native28、Phase2B1.5 对称裁决、Phase2B1.8 raw logits / q导数 / PRG梯度。
冻结裁决 image-balanced AUROC=0.7848415501，DeepCapture=64.0314%，ShallowProtection=79.0939%。
上一轮 PRG 全部 Deep-Win 几乎均有益，但 Shallow-Win HHCR=96.2258%。该结果不是本轮可调条件。
原 Phase2B1.8 raw native28 mIoU=43.6349%、teacher=59.3171% 仅是历史语义诊断，不能与原官方最终融合 mIoU 混为一谈。

## 3. Convex teacher failure mechanism

忽略极小的 epsilon 项，convex teacher 对 raw 的 KL 梯度方向近似为 `wD*(ps-pd)`：context 权重主要缩放 deep-directed 更新，并没有为 shallow 独有正确信息提供显式拒绝通道。本轮不再构造该 teacher，也不增加共享 head 对照。历史 PRG 只读取已冻结梯度用于比较。

## 4. Directional transfer hypothesis

`q` 只表示 Need，`Delta_sym` 判断哪一层得到 context 支持，`mD` 决定是否允许 deep→shallow。
只检验冻结点的局部机制：选择性减少有害迁移，同时覆盖足够多的 raw 错误；不推断25轮训练最终 mIoU。

## 5. Tensor / precision / denominator contract

| Tensor | Shape | 作用 |
| --- | --- | --- |
| F28_raw | [B,512,28,28] | HFRM28_1之前 |
| L_s / p_s / L_d / p_d | [B,4,28,28] | 复用ic1/deep head |
| q / Delta / mD | [B,28,28] | 全部detach |


真实网络 BF16，概率/loss/logit导数 FP32，统计与内积 FP64。主审计 batch1；loss 分母包括全部784个位置，不借GT排除背景。batch20 smoke 使用整批分母。
诊断分母只含 GT0–3（2,479,143位置），background4/ignore255 排除；未使用官方 background overwrite 修正诊断。拒绝位置计零，不从分母消失。

## 6. Delta / raw-logit replay

| 项目 | 最大绝对差 |
| --- | --- |
| ps | 0 |
| pd | 0 |
| q | 5.960464478e-08 |
| raw_frozen_head_logits | 0 |
| raw_previous_logits | 0 |
| delta_stored | 0 |
| delta_recomputed | 0 |
| supports | 0 |


`S_S=.5*(T_SS+T_SD)`；`S_D=.5*(T_DS+T_DD)`；`Delta=S_D-S_S`。
存储Delta、完整四项support、现场重算Delta完全一致，gate mismatch=0。旧 raw logits与当前head/frozen-head完全一致。
q重算的最大差为5.96046448e-8，在预先批准1e-7内；这是继承浮点算序差异，主loss始终使用冻结缓存q，不替换。独立q导数精确相同。

## 7. DeepCapture / ShallowProtection

| 组 | Deep-Win | Shallow-Win | 双标签图像 | Image AUC | DeepCapture | ShallowProtection | BA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 314730 | 182899 | 3180 | 0.7848415501 | 64.0314% | 79.0939% | 0.7156265907 |


Image AUROC 95%CI：0.7848415501 [0.7771296175, 0.7928154228]；DeepCapture：64.0314% [62.8538%, 65.2104%]；ShallowProtection：79.0939% [78.1909%, 79.9878%]。
Image AUROC按同时存在两类冲突的3180张图像等权平均；不是 pooled AUROC（0.7882824528）。捕获/保护率按像素池化，bootstrap仍以图像为重采样单位。

## 8. Hard direction gate mD

`mD = 1[Delta_sym > 0]`，精确零点，tie拒绝。无偏置、温度、置信度筛选、类别规则或q阈值。
“preserve shallow”在这里指该位置没有直接蒸馏梯度；若未来真的更新共享网络参数，并不保证这些位置的预测绝对不变。本轮无更新，状态与预测恒等另行实测。

## 9. Primary ADT

`L_ADT=sum(q*mD*KL(pd||softmax(L_student)))/(sum(q*mD)+1e-8)`。
`L_student=conv2d(F28_raw,ic1.weight.detach(),ic1.bias.detach())`。
pd/q/Delta/mD全部detach；只有批准的浅层学生路径接收梯度。KL沿用epsilon放在log内部的旧实现，独立FP64解析式验证通过。

## 10. UDT control

`L_UDT=sum(q*KL(pd||ps))/(sum(q)+eps)`。无方向拒绝；仍然有冻结q加权，因此 UDT不是无q加权的普通KL。系数固定1。

## 11. Rate-matched RG control

固定NumPy default_rng42、冻结图像顺序，一次随机实现。每张图在全部784位置中无放回选取与ADT相同数量的位置；不读取GT、q或类别，不反复抽签。
`L_RG=sum(q*m_rand*KL(pd||ps))/(sum(q*m_rand)+eps)`。严格匹配的是全部位置的逐图数量，前景/冲突子组的比例允许自然不同，必须原样报告。| 范围 | 位置数 | ADT rate | RG rate | 差值 | 逐图完全匹配 |
| --- | --- | --- | --- | --- | --- |
| ALL784_GT_BLIND | 2679712 | 28.1285% | 28.1285% | 0.0000% | True |

## 12. Secondary SDT

`aD=relu(Delta_sym)`；`L_SDT=sum(q*aD*KL(pd||ps))/(sum(q*aD)+eps)`。
不增加temperature/power/额外normalization。它仅是附件预注册的次要探针，不能替代ADT主判定；不会因为次要结果有利而改变本轮合同。| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | SDT | 2479143 | 24.5172% | 3.5988% | 71.8840% | 0.001142706647 | 0 |
| Top20 | SDT | 485451 | 37.8425% | 5.2219% | 56.9355% | 0.004297196277 | 0 |
| Raw_Wrong | SDT | 708407 | 35.5865% | 2.9820% | 61.4315% | 0.004048187593 | 0 |
| Deep-Win | SDT | 314730 | 64.0314% | 0.0000% | 35.9686% | 0.008613092985 | 0.001848562795 |
| Shallow-Win | SDT | 182899 | 0.7447% | 20.1609% | 79.0945% | -0.002509320191 | 0 |

## 13. Transfer coverage

| 组 | 全部分母 | Selected | Rejected | 激活率 |
| --- | --- | --- | --- | --- |
| all | 2479143 | 697038 | 1782105 | 28.1161% |
| Top20 | 485451 | 209058 | 276393 | 43.0647% |
| Bottom80 | 1993692 | 487980 | 1505712 | 24.4762% |
| Q1 | 495830 | 24016 | 471814 | 4.8436% |
| Q2 | 495828 | 84586 | 411242 | 17.0595% |
| Q3 | 495829 | 156448 | 339381 | 31.5528% |
| Q4 | 495828 | 205034 | 290794 | 41.3518% |
| Q5 | 495828 | 226954 | 268874 | 45.7727% |
| Raw_Correct | 1770736 | 423816 | 1346920 | 23.9345% |
| Raw_Wrong | 708407 | 273222 | 435185 | 38.5685% |
| Deep-Win | 314730 | 201526 | 113204 | 64.0314% |
| Shallow-Win | 182899 | 38237 | 144662 | 20.9061% |
| Both-Wrong | 393677 | 71696 | 321981 | 18.2119% |
| Stable-Correct | 1587837 | 385579 | 1202258 | 24.2833% |
| class0 | 965133 | 250889 | 714244 | 25.9953% |
| class1 | 1044692 | 296117 | 748575 | 28.3449% |
| class2 | 323515 | 104829 | 218686 | 32.4031% |
| class3 | 145803 | 45203 | 100600 | 31.0028% |
| boundary | 201144 | 42081 | 159063 | 20.9208% |
| interior | 2277999 | 654957 | 1623042 | 28.7514% |


总体前景激活28.1161%，Raw-Wrong激活38.5685%。后者低于40%，因此在当前冻结门下，即使每个被选中的raw错误都获得正dM，也达不到Gate E要求的40%全分母BenefitRate。此为事后解释，不用于修改门控。

## 14. Selected / rejected semantic quality

| 组 | 选择 | N | Raw acc | Deep acc | 差值 | Repair | Harm | NetRepair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | selected | 697038 | 60.8024% | 84.2285% | 23.4261% | 201526 | 38237 | 163289 |
| all | rejected | 1782105 | 75.5803% | 73.8151% | -1.7652% | 113204 | 144662 | -31458 |
| Top20 | selected | 209058 | 22.9223% | 78.7576% | 55.8352% | 138006 | 21278 | 116728 |
| Top20 | rejected | 276393 | 44.8119% | 44.9899% | 0.1780% | 82555 | 82063 | 492 |
| Deep-Win | selected | 201526 | 0.0000% | 100.0000% | 100.0000% | 201526 | 0 | 201526 |
| Deep-Win | rejected | 113204 | 0.0000% | 100.0000% | 100.0000% | 113204 | 0 | 113204 |
| Shallow-Win | selected | 38237 | 100.0000% | 0.0000% | -100.0000% | 0 | 38237 | -38237 |
| Shallow-Win | rejected | 144662 | 100.0000% | 0.0000% | -100.0000% | 0 | 144662 | -144662 |
| Both-Wrong | selected | 71696 | 0.0000% | 0.0000% | 0.0000% | 0 | 0 | 0 |
| Both-Wrong | rejected | 321981 | 0.0000% | 0.0000% | 0.0000% | 0 | 0 | 0 |
| Stable-Correct | selected | 385579 | 100.0000% | 100.0000% | 0.0000% | 0 | 0 | 0 |
| Stable-Correct | rejected | 1202258 | 100.0000% | 100.0000% | 0.0000% | 0 | 0 | 0 |


Repair/Harm是同一冻结点 raw与deep的hard预测交换计数，不是实际参数更新后的纠错数。完整所有组见selected_region_quality.csv。

## 15. DeepSelectionPrecision

| 组 | Precision | Recall | Shallow保护 |
| --- | --- | --- | --- |
| all | 84.0522% | 64.0314% | 79.0939% |
| Top20 | 86.6415% | 62.5704% | 79.4099% |
| Bottom80 | 78.9274% | 67.4532% | 78.6835% |
| Q1 | 73.6686% | 71.5517% | 76.4550% |
| Q2 | 74.6052% | 71.3520% | 77.4536% |
| Q3 | 71.2435% | 69.5951% | 75.5682% |
| Q4 | 76.2766% | 65.5366% | 78.2784% |
| Q5 | 86.8466% | 63.3192% | 79.9316% |
| Raw_Correct | 0.0000% | NA | 79.0939% |
| Raw_Wrong | 100.0000% | 64.0314% | NA |
| Deep-Win | 100.0000% | 64.0314% | NA |
| Shallow-Win | 0.0000% | NA | 79.0939% |
| Both-Wrong | NA | NA | NA |
| Stable-Correct | NA | NA | NA |
| class0 | 64.1911% | 74.3322% | 79.4590% |
| class1 | 86.9724% | 72.2922% | 83.9835% |
| class2 | 87.4805% | 63.0888% | 42.4802% |
| class3 | 99.6073% | 45.3818% | 68.1818% |
| boundary | 53.8417% | 40.3834% | 66.6755% |
| interior | 86.6603% | 66.1078% | 81.1881% |


在exactly-one-correct集合：Precision=84.0522% [83.1464%, 84.9111%]。分母仅selected Deep-Win+selected Shallow-Win，不能解释为全部选中像素的准确率。

## 16. GT-margin exact directional derivative

`v=-dL/dLs`，不做单位范数归一化；`M=Ls[GT]-max_nonGT(Ls)`。
若最大竞争类别打平，dM使用这些当前并列类别中最大的v作为max方向导数，不随意挑一个argmax。共有2111个前景位置存在竞争logit并列。
Benefit/Harm/Zero分别严格使用dM>0/<0/==0。dM是独立logit坐标的局部方向量，不等同于共享参数实际更新后的mIoU或最终训练收益。

## 17. All / Top20 directional utility

| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | UDT | 2479143 | 78.9280% | 21.0718% | 0.0003% | 0.0004863217408 | 7.273322626e-05 |
| all | RG | 2479143 | 23.3702% | 4.8744% | 71.7554% | 0.0004870411298 | 0 |
| all | ADT | 2479143 | 24.5172% | 3.5988% | 71.8840% | 0.0008764349282 | 0 |
| all | SDT | 2479143 | 24.5172% | 3.5988% | 71.8840% | 0.001142706647 | 0 |
| Top20 | UDT | 485451 | 71.9677% | 28.0313% | 0.0010% | 0.001436086992 | 0.001673875435 |
| Top20 | RG | 485451 | 21.3045% | 6.3059% | 72.3896% | 0.00143194475 | 0 |
| Top20 | ADT | 485451 | 37.8425% | 5.2219% | 56.9355% | 0.002753287339 | 0 |
| Top20 | SDT | 485451 | 37.8425% | 5.2219% | 56.9355% | 0.004297196277 | 0 |


ADT all Mean dM CI：0.0008764349282 [0.0008564997475, 0.0008960555787]；Top20：0.002753287339 [0.002671722472, 0.002837791051]。
all、Top20及四类Mean dM均为正（6/6），两组Benefit均高于Harm，Gate B通过。大量Zero是门控设计所致，不得去掉后重新宣称主收益覆盖率更高。

## 18. Deep-Win transfer

| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Deep-Win | UDT | 314730 | 100.0000% | 0.0000% | 0.0000% | 0.003662878906 | 0.003005817416 |
| Deep-Win | RG | 314730 | 30.0480% | 0.0000% | 69.9520% | 0.003645587941 | 0 |
| Deep-Win | ADT | 314730 | 64.0314% | 0.0000% | 35.9686% | 0.005706993123 | 0.003619880474 |
| Deep-Win | SDT | 314730 | 64.0314% | 0.0000% | 35.9686% | 0.008613092985 | 0.001848562795 |
| Deep-Win | PRG_previous | 314730 | 100.0000% | 0.0000% | 0.0000% | 0.002004331467 | 0.001495761826 |

| 组 | Loss | BRR | HHCR | DBR | DCR |
| --- | --- | --- | --- | --- | --- |
| all | UDT | 99.9994% | 96.2258% | 100.0000% | 96.2269% |
| all | RG | 30.0480% | 21.6885% | 30.0480% | 21.6885% |
| all | ADT | 64.0311% | 20.1598% | 64.0314% | 20.1609% |
| all | SDT | 64.0311% | 20.1598% | 64.0314% | 20.1609% |
| all | PRG_previous | 99.9994% | 96.2258% | 100.0000% | 96.2269% |


Deep-Win总数314730；ADT Benefit=64.0314%，Harm=0；被拒绝的35.9686%计Zero。DBR分母仍是全部Deep-Win。

## 19. Shallow-Win protection

| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shallow-Win | UDT | 182899 | 3.7720% | 96.2269% | 0.0011% | -0.002847534292 | -0.002137277101 |
| Shallow-Win | RG | 182899 | 0.9858% | 21.6885% | 77.3257% | -0.002820947934 | 0 |
| Shallow-Win | ADT | 182899 | 0.7447% | 20.1609% | 79.0945% | -0.002134815991 | 0 |
| Shallow-Win | SDT | 182899 | 0.7447% | 20.1609% | 79.0945% | -0.002509320191 | 0 |
| Shallow-Win | PRG_previous | 182899 | 3.7720% | 96.2269% | 0.0011% | -0.001210710787 | -0.0008682342304 |


Shallow-Win总数182899；拒绝率79.0939%，Harm=20.1609%。
Rate下降不代表每次错误迁移的强度下降：ADT Mean dM=-0.002134815991，仍为负。

## 20. BRR_ADT

`BRR=P(mD=1 AND dM>0 AND dQ<0 | Deep-Win)`，全部Deep-Win为分母。
实测64.0311% [62.8528%, 65.2100%]，超过0.60，Gate C通过。它低于历史PRG近100%，即本轮明确牺牲一部分纠错覆盖以保护shallow，而非无代价保留全部Deep-Win收益。

## 21. HHCR_ADT

`HHCR=P(mD=1 AND dM<0 AND dQ<0 | Shallow-Win)`，全部Shallow-Win为分母。
实测20.1598% [19.2819%, 21.0569%]，低于0.30，Gate D通过。
`gq=dq/dLs`由独立图求导；主loss的q已detach。所有被拒绝位置的dQ严格为0。

## 22. PRG versus ADT historical comparison

| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM | BRR | HHCR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | PRG_previous | 2479143 | 78.9280% | 21.0718% | 0.0003% | 0.000283648257 | 3.537180237e-05 | 99.9994% | 96.2258% |
| all | ADT | 2479143 | 24.5172% | 3.5988% | 71.8840% | 0.0008764349282 | 0 | 64.0311% | 20.1598% |
| Top20 | PRG_previous | 485451 | 71.9677% | 28.0313% | 0.0010% | 0.0008769221801 | 0.0007884535589 | 99.9991% | 97.8305% |
| Top20 | ADT | 485451 | 37.8425% | 5.2219% | 56.9355% | 0.002753287339 | 0 | 62.5700% | 20.2079% |
| Deep-Win | PRG_previous | 314730 | 100.0000% | 0.0000% | 0.0000% | 0.002004331467 | 0.001495761826 | 99.9994% | NA |
| Deep-Win | ADT | 314730 | 64.0314% | 0.0000% | 35.9686% | 0.005706993123 | 0.003619880474 | 64.0311% | NA |
| Shallow-Win | PRG_previous | 182899 | 3.7720% | 96.2269% | 0.0011% | -0.001210710787 | -0.0008682342304 | NA | 96.2258% |
| Shallow-Win | ADT | 182899 | 0.7447% | 20.1609% | 79.0945% | -0.002134815991 | 0 | NA | 20.1598% |


HHCR从96.2258%降至20.1598%，减少约76.07个百分点；Deep-Win BRR从99.9994%降至64.0311%。
必须同时报告：Shallow-Win Mean dM从-0.0012107108变为-0.0021348160，更负。各loss都有自身权重和分母，ADT选区归一化会改变强度，所以不能仅按rate下降推断所有风险均改善，也不能把Mean dM更大直接换算成mIoU提升。

## 23. Raw-Correct safety

| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw_Correct | UDT | 1770736 | 86.2226% | 13.7772% | 0.0002% | 5.370084235e-05 | 5.343228531e-05 |
| Raw_Correct | RG | 1770736 | 25.8372% | 3.2348% | 70.9280% | 5.795915871e-05 | 0 |
| Raw_Correct | ADT | 1770736 | 20.0888% | 3.8456% | 76.0656% | 0.0001592784017 | 0 |
| Raw_Correct | SDT | 1770736 | 20.0888% | 3.8456% | 76.0656% | -1.967048953e-05 | 0 |
| Raw_Correct | PRG_previous | 1770736 | 86.2226% | 13.7772% | 0.0002% | 4.869791793e-05 | 2.606166936e-05 |


ADT Harm=3.8456% [3.6861%, 4.0079%]，满足≤30%。本轮No-Go不是因为raw正确位置的总体受害比例超标。

## 24. Raw-Wrong benefit coverage

| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw_Wrong | UDT | 708407 | 60.6942% | 39.3052% | 0.0006% | 0.001567702076 | 0.0005694712563 |
| Raw_Wrong | RG | 708407 | 17.2035% | 8.9728% | 73.8237% | 0.001559575553 | 0 |
| Raw_Wrong | ADT | 708407 | 35.5865% | 2.9820% | 61.4315% | 0.002669041267 | 0 |
| Raw_Wrong | SDT | 708407 | 35.5865% | 2.9820% | 61.4315% | 0.004048187593 | 0 |
| Raw_Wrong | PRG_previous | 708407 | 60.6942% | 39.3052% | 0.0006% | 0.0008709307424 | 0.000259484812 |


ADT Benefit=35.5865% [34.6906%, 36.4944%]，预注册要求≥40%，少4.4135个百分点，CI上界仍低于40%。
这是Gate E唯一失败子项。Raw-Wrong拒绝/零梯度比例高，主全分母覆盖不足；不得用active-only高Benefit替代。

## 25. Both-Wrong

| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM | Mean dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Both-Wrong | UDT | 393677 | 29.2707% | 70.7283% | 0.0010% | -0.0001073132378 | -6.504919656e-05 | -0.0001763668348 |
| Both-Wrong | RG | 393677 | 6.9349% | 16.1462% | 76.9189% | -0.0001081131331 | 0 | -0.0001763742306 |
| Both-Wrong | ADT | 393677 | 12.8458% | 5.3661% | 81.7881% | 0.0002403126718 | 0 | -0.0001277746938 |
| Both-Wrong | SDT | 393677 | 12.8458% | 5.3661% | 81.7881% | 0.0003987168989 | 0 | -0.0001262665398 |
| Both-Wrong | PRG_previous | 393677 | 29.2707% | 70.7283% | 0.0010% | -3.518063837e-05 | -2.99312137e-05 | -7.605916536e-05 |


此组deep与raw的argmax均错误，deep准确率按定义为0。正dM仅表示GT局部margin可能改善，不是deep已经正确。不引入第三证据或特殊修复分支。

## 26. Stable-Correct

| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM | Mean dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stable-Correct | UDT | 1587837 | 95.7199% | 4.2800% | 0.0001% | 0.0003878869111 | 8.624641487e-05 | -5.983008122e-05 |
| Stable-Correct | RG | 1587837 | 28.6998% | 1.1091% | 70.1911% | 0.0003895733158 | 0 | -6.010210868e-05 |
| Stable-Correct | ADT | 1587837 | 22.3170% | 1.9663% | 75.7167% | 0.0004235294365 | 0 | -6.88671851e-05 |
| Stable-Correct | SDT | 1587837 | 22.3170% | 1.9663% | 75.7167% | 0.0002671060755 | 0 | -4.689509312e-05 |
| Stable-Correct | PRG_previous | 1587837 | 95.7199% | 4.2800% | 0.0001% | 0.0001937660784 | 4.197803901e-05 | -2.994602082e-05 |


二者argmax相同且都正确并不保证蒸馏方向对GT margin有利；完整展示harm/zero，不能仅展示selected的优势。

## 27. q × direction grid

| Q | mD | N | Raw acc | Deep acc | 差值 | Benefit | Harm | Mean dM | Mean dQ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 471814 | 92.4383% | 92.3981% | -0.0403% | 0.0000% | 0.0000% | 0 | 0 |
| 1 | 1 | 24016 | 82.0203% | 82.6865% | 0.6662% | 53.5393% | 46.4607% | 5.210347018e-06 | -6.853059946e-07 |
| 2 | 0 | 411242 | 85.4207% | 84.8739% | -0.5469% | 0.0000% | 0.0000% | 0 | 0 |
| 2 | 1 | 84586 | 83.0279% | 85.3120% | 2.2841% | 81.7913% | 18.2087% | 0.0001452535899 | -1.475189814e-05 |
| 3 | 0 | 339381 | 76.2311% | 73.7855% | -2.4456% | 0.0000% | 0.0000% | 0 | 0 |
| 3 | 1 | 156448 | 83.9666% | 87.8650% | 3.8984% | 89.5058% | 10.4942% | 0.0007586652643 | -0.0001066505487 |
| 4 | 0 | 290794 | 62.8380% | 57.0280% | -5.8100% | 0.0000% | 0.0000% | 0 | 0 |
| 4 | 1 | 205034 | 76.6448% | 86.1872% | 9.5423% | 90.2070% | 9.7930% | 0.002458599459 | -0.0005188442965 |
| 5 | 0 | 268874 | 43.9068% | 42.4846% | -1.4222% | 0.0000% | 0.0000% | 0 | 0 |
| 5 | 1 | 226954 | 19.9935% | 79.7117% | 59.7183% | 88.4717% | 11.5279% | 0.006774975636 | -0.001680977388 |


分位边界完全复用冻结资产，不选择有利cell训练。mD=0各格的直接dM/dQ严格为零。

## 28. Per-class / power

| 组 | Deep-Win | Shallow-Win | 双标签图像 | Image AUC | DeepCapture | ShallowProtection | BA | Power |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class0 | 43802 | 88423 | 1654 | 0.7902943397 | 74.3322% | 79.4590% | 0.7689559328 | POWERED |
| class1 | 121742 | 82309 | 2204 | 0.7969253776 | 72.2922% | 83.9835% | 0.7813787502 | POWERED |
| class2 | 74850 | 11749 | 723 | 0.4514365897 | 63.0888% | 42.4802% | 0.5278452772 | POWERED |
| class3 | 74336 | 418 | 107 | 0.3408509823 | 45.3818% | 68.1818% | 0.567817991 | UNDERPOWERED |

| Class | 激活 | Benefit | Harm | Mean dM | BRR | HHCR |
| --- | --- | --- | --- | --- | --- | --- |
| class0 | 25.9953% | 21.7761% | 4.2191% | 0.0005488596202 | 74.3322% | 20.4268% |
| class1 | 28.3449% | 25.8291% | 2.5158% | 0.0009956362164 | 72.2922% | 14.6898% |
| class2 | 32.4031% | 26.2260% | 6.1771% | 0.001119128369 | 63.0888% | 56.2261% |
| class3 | 31.0028% | 29.4706% | 1.5322% | 0.00165220867 | 45.3804% | 27.0335% |


Power同时要求≥500 Deep-Win、≥500 Shallow-Win、≥30双标签图像。class3只有418个Shallow-Win，明确UNDERPOWERED；class3总体Mean dM为正也不能弥补该层级安全性证据不足。
class2/3裁决弱于class0/1，表中如实保留，不创建类别规则。依合同该标记不擅改A–G门槛。

## 29. Boundary / interior

| 组 | Deep-Win | Shallow-Win | 双标签图像 | Image AUC | DeepCapture | ShallowProtection | BA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boundary | 25404 | 26392 | 1838 | 0.5528591669 | 40.3834% | 66.6755% | 0.5352945596 |
| interior | 289326 | 156507 | 3155 | 0.8015919659 | 66.1078% | 81.1881% | 0.736479217 |

| 组 | Benefit | Harm | BRR | HHCR | Mean dM |
| --- | --- | --- | --- | --- | --- |
| boundary | 12.3285% | 8.5924% | 40.3834% | 32.1878% | 0.0001950322217 |
| interior | 25.5935% | 3.1579% | 66.1074% | 18.1315% | 0.0009366017965 |


沿用冻结的boundary≤7px/interior>7px映射，不重估边界宽度或阈值。

## 30. Gradient localization

| Loss | Top20 meanG | Bottom80 meanG | Ratio | Q1 | Q2 | Q3 | Q4 | Q5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UDT | 0.002827381089 | 0.0004106175137 | 6.8856807 | 2.824292245e-06 | 5.375862778e-05 | 0.0002804765554 | 0.000959847494 | 0.003122366535 |
| RG | 0.002810655068 | 0.0004112720684 | 6.834052891 | 2.863504346e-06 | 5.450741951e-05 | 0.0002818721409 | 0.0009610934424 | 0.003105192948 |
| ADT | 0.003161839665 | 0.0004407054031 | 7.174497165 | 1.000440169e-06 | 3.400557766e-05 | 0.000251464639 | 0.001020376701 | 0.003560866467 |
| SDT | 0.004466944415 | 0.0003815715315 | 11.70670254 | 1.802233058e-07 | 9.581966294e-06 | 0.0001096783352 | 0.0007492311408 | 0.005039059711 |


G为四类logit梯度L2，Top20与全局Q5不是同一选择方式；两套冻结分组分别报告，不互相替换。

## 31. Active-only diagnostic

| 组 | Loss | Active N | Benefit | Harm | Mean dM |
| --- | --- | --- | --- | --- | --- |
| all | UDT | 2479143 | 78.9280% | 21.0718% | 0.0004863217408 |
| all | RG | 700224 | 82.7421% | 17.2578% | 0.00172436907 |
| all | ADT | 697038 | 87.2000% | 12.7999% | 0.003117200952 |
| all | SDT | 697038 | 87.2000% | 12.7999% | 0.004064244968 |
| Top20 | UDT | 485451 | 71.9677% | 28.0313% | 0.001436086992 |
| Top20 | RG | 134036 | 77.1606% | 22.8386% | 0.005186211249 |
| Top20 | ADT | 209058 | 87.8737% | 12.1258% | 0.006393374529 |
| Top20 | SDT | 209058 | 87.8737% | 12.1258% | 0.009978466406 |
| Raw_Correct | UDT | 1770736 | 86.2226% | 13.7772% | 5.370084235e-05 |
| Raw_Correct | RG | 514789 | 88.8731% | 11.1267% | 0.0001993639508 |
| Raw_Correct | ADT | 423816 | 83.9327% | 16.0671% | 0.0006654774712 |
| Raw_Correct | SDT | 423816 | 83.9327% | 16.0671% | -8.218482536e-05 |
| Raw_Wrong | UDT | 708407 | 60.6942% | 39.3052% | 0.001567702076 |
| Raw_Wrong | RG | 185435 | 65.7217% | 34.2783% | 0.005957959602 |
| Raw_Wrong | ADT | 273222 | 92.2682% | 7.7318% | 0.006920260876 |
| Raw_Wrong | SDT | 273222 | 92.2682% | 7.7318% | 0.01049609632 |
| Deep-Win | UDT | 314730 | 100.0000% | 0.0000% | 0.003662878906 |
| Deep-Win | RG | 94570 | 100.0000% | 0.0000% | 0.01213255676 |
| Deep-Win | ADT | 201526 | 100.0000% | 0.0000% | 0.008912805025 |
| Deep-Win | SDT | 201526 | 100.0000% | 0.0000% | 0.0134513599 |
| Shallow-Win | UDT | 182899 | 3.7720% | 96.2269% | -0.002847534292 |
| Shallow-Win | RG | 41472 | 4.3475% | 95.6501% | -0.01244088918 |
| Shallow-Win | ADT | 38237 | 3.5620% | 96.4354% | -0.01021146298 |
| Shallow-Win | SDT | 38237 | 3.5620% | 96.4354% | -0.01200282851 |


本表仅说明被选中迁移的质量，不能覆盖主Gate使用的all-denominator。SDT与ADT有相同正权重支撑集，主要改变幅度，不能靠此表解除Raw-Wrong覆盖不足。

## 32. ADT versus random gating

| 组 | ADT rate | RG rate | RG Mean dM | RG Benefit | RG Harm |
| --- | --- | --- | --- | --- | --- |
| all | 28.1161% | 28.2446% | 0.0004870411298 | 23.3702% | 4.8744% |
| Top20 | 43.0647% | 27.6106% | 0.00143194475 | 21.3045% | 6.3059% |
| Raw_Wrong | 38.5685% | 26.1763% | 0.001559575553 | 17.2035% | 8.9728% |
| Deep-Win | 64.0314% | 30.0480% | 0.003645587941 | 30.0480% | 0.0000% |
| Shallow-Win | 20.9061% | 22.6748% | -0.002820947934 | 0.9858% | 21.6885% |


预注册三项配对差（ADT−RG）：

- all Mean dM：0.0003893937984 [0.0003754449203, 0.0004030372835]。
- Shallow-Win Harm：-1.5276% [-2.3469%, -0.6830%]（负值有利）。
- Deep-Win Benefit：33.9834% [32.9802%, 35.0023%]。

三项95%CI均支持ADT，且满足点估计条件，Gate F通过。采用事先固定的OR规则，不声称做过多重比较家族错误率校正；随机对照只有一个seed42 realization，不等同跨随机门seed稳定性。

## 33. Feature / upstream parameter gradients

| 组 | N | RMS | Mean pixel L2 | Maxabs | Finite |
| --- | --- | --- | --- | --- | --- |
| all | 2479143 | 0.0001798648144 | 0.001255065853 | 0.05908203125 | True |
| Top20 | 485451 | 0.0003634284469 | 0.004076096218 | 0.05908203125 | True |
| Bottom80 | 1993692 | 8.982214899e-05 | 0.0005681633568 | 0.05859375 | True |
| Q1 | 495830 | 4.186000365e-07 | 1.297872463e-06 | 0.0001592636108 | True |
| Q2 | 495828 | 6.264634087e-06 | 4.421510174e-05 | 0.001770019531 | True |
| Q3 | 495829 | 3.20712297e-05 | 0.0003268118522 | 0.006286621094 | True |
| Q4 | 495828 | 0.0001104815005 | 0.001319856929 | 0.01483154297 | True |
| Q5 | 495828 | 0.0003853346654 | 0.00458315444 | 0.05908203125 | True |
| Raw_Correct | 1770736 | 0.0001118007482 | 0.0006182698353 | 0.05908203125 | True |
| Raw_Wrong | 708407 | 0.0002863098379 | 0.002846802854 | 0.05859375 | True |
| Deep-Win | 314730 | 0.0003910900013 | 0.005343514729 | 0.05541992188 | True |
| Shallow-Win | 182899 | 0.0003017221225 | 0.002201132311 | 0.05908203125 | True |
| Both-Wrong | 393677 | 0.0001588364601 | 0.0008507753277 | 0.05859375 | True |
| Stable-Correct | 1587837 | 5.876176235e-05 | 0.0004359438384 | 0.01489257812 | True |
| class0 | 965133 | 0.0001751928819 | 0.001065711354 | 0.05883789062 | True |
| class1 | 1044692 | 0.0001743673497 | 0.001221151116 | 0.05908203125 | True |
| class2 | 323515 | 0.0001917249536 | 0.001646905156 | 0.05029296875 | True |
| class3 | 145803 | 0.0002179590395 | 0.001882054585 | 0.04028320312 | True |
| boundary | 201144 | 0.0002089617097 | 0.001345660448 | 0.05908203125 | True |
| interior | 2277999 | 0.0001770660134 | 0.001247066482 | 0.05883789062 | True |
| active | 697038 | 0.0003392101005 | 0.004463871015 | 0.05908203125 | True |
| rejected | 1782105 | 0 | 0 | 0 | True |
| all784 | 2679712 | 0.0001802781028 | 0.001254853957 | 0.05908203125 | True |

| 参数 | Numel | Energy | RMS | Maxabs | 非零图像 |
| --- | --- | --- | --- | --- | --- |
| b4.bn_branch2a.weight | 256 | 34869.39979 | 0.1996255952 | 6.377186775 | 3418 |
| b4.bn_branch2a.bias | 256 | 12462.10886 | 0.1193410236 | 2.531692982 | 3418 |
| b4.conv_branch2a.weight | 1179648 | 349471.0388 | 0.009309864249 | 0.8125 | 3418 |
| b4.bn_branch2b1.weight | 512 | 25355.23349 | 0.1203684858 | 3.947827816 | 3418 |
| b4.bn_branch2b1.bias | 512 | 6030.464645 | 0.05870220005 | 1.484778047 | 3418 |
| b4.conv_branch2b1.weight | 2359296 | 372499.1317 | 0.006796501036 | 1.921875 | 3418 |
| b4.conv_branch1.weight | 131072 | 105917.6437 | 0.01537599266 | 2.21875 | 3418 |
| b4_1.bn_branch2a.weight | 512 | 17433.43633 | 0.09980918063 | 4.938493252 | 3418 |
| b4_1.bn_branch2a.bias | 512 | 3479.496451 | 0.04458997093 | 1.142905235 | 3418 |
| b4_1.conv_branch2a.weight | 2359296 | 308711.1712 | 0.006187268001 | 2.328125 | 3418 |
| b4_1.bn_branch2b1.weight | 512 | 16878.83439 | 0.09820875744 | 3.303385973 | 3418 |
| b4_1.bn_branch2b1.bias | 512 | 2197.34424 | 0.03543464125 | 0.9861736298 | 3418 |
| b4_1.conv_branch2b1.weight | 2359296 | 242124.2116 | 0.005479511779 | 3.0625 | 3418 |
| b4_2.bn_branch2a.weight | 512 | 6870.972874 | 0.06265967058 | 3.628885746 | 3418 |
| b4_2.bn_branch2a.bias | 512 | 2024.544352 | 0.03401281978 | 1.276649714 | 3418 |
| b4_2.conv_branch2a.weight | 2359296 | 94610.74565 | 0.003425255622 | 0.98046875 | 3418 |
| b4_2.bn_branch2b1.weight | 512 | 5605.24127 | 0.05659475 | 1.991761327 | 3418 |
| b4_2.bn_branch2b1.bias | 512 | 1515.730586 | 0.02942998207 | 0.7451755404 | 3418 |
| b4_2.conv_branch2b1.weight | 2359296 | 64290.1593 | 0.002823546901 | 1.4921875 | 3418 |
| b4_3.bn_branch2a.weight | 512 | 14054.88598 | 0.08961746475 | 5.033437729 | 3418 |
| b4_3.bn_branch2a.bias | 512 | 2851.676133 | 0.04036724316 | 1.299284935 | 3418 |
| b4_3.conv_branch2a.weight | 2359296 | 215408.3058 | 0.00516837453 | 1.3984375 | 3418 |
| b4_3.bn_branch2b1.weight | 512 | 10175.83692 | 0.07625425097 | 3.650503635 | 3418 |
| b4_3.bn_branch2b1.bias | 512 | 1126.702555 | 0.02537369002 | 0.7145462036 | 3418 |
| b4_3.conv_branch2b1.weight | 2359296 | 114094.5832 | 0.003761449784 | 1.703125 | 3418 |
| b4_4.bn_branch2a.weight | 512 | 9095.844779 | 0.07209423047 | 3.129076958 | 3418 |
| b4_4.bn_branch2a.bias | 512 | 2266.677046 | 0.03598933431 | 0.7216005325 | 3418 |
| b4_4.conv_branch2a.weight | 2359296 | 171306.6688 | 0.004609033769 | 2.140625 | 3418 |
| b4_4.bn_branch2b1.weight | 512 | 7125.885322 | 0.06381141936 | 3.174556971 | 3418 |
| b4_4.bn_branch2b1.bias | 512 | 1070.787552 | 0.02473606693 | 1.087833643 | 3418 |
| b4_4.conv_branch2b1.weight | 2359296 | 79408.55028 | 0.003138024925 | 1.1875 | 3418 |
| b4_5.bn_branch2a.weight | 512 | 25407.60405 | 0.1204927307 | 7.675104141 | 3418 |
| b4_5.bn_branch2a.bias | 512 | 3082.475823 | 0.04196902015 | 1.214855671 | 3418 |
| b4_5.conv_branch2a.weight | 2359296 | 197296.7254 | 0.004946325515 | 2.15625 | 3418 |
| b4_5.bn_branch2b1.weight | 512 | 8468.450999 | 0.06956342871 | 3.00815177 | 3418 |
| b4_5.bn_branch2b1.bias | 512 | 574.042733 | 0.01811136162 | 0.4001550674 | 3418 |
| b4_5.conv_branch2b1.weight | 2359296 | 91827.4944 | 0.00337449759 | 1.1484375 | 3418 |
| bn45.weight | 512 | 517.29746 | 0.01719290103 | 0.7836874723 | 3418 |
| bn45.bias | 512 | 958.1221262 | 0.02339857586 | 0.6385192871 | 3418 |
| ic1.weight | 2048 | 0 | 0 | 0 | 0 |
| ic1.bias | 4 | 0 | 0 | 0 | 0 |


39个批准参数包含b4..b4_5及bn45；各卷积子组均有非零梯度，ic1两项始终零。BN始终eval，仅诊断时允许其affine求导；这不是修改未来原始训练freeze规则。Feature能量与parameter能量不混作同一分母。

## 34. Detach / no-GT / no-step audit

| 检查 | 值 |
| --- | --- |
| q_detached | True |
| delta_detached | True |
| gate_detached | True |
| deep_source_detached | True |
| primary_ic1_none | True |
| hfrm_none | True |
| upstream_conv_nonzero | True |
| all_other_primary_gradients_none | True |
| rejected_feature_zero | True |
| rejected_logit_zero | True |
| optimizer_created | False |
| optimizer_steps | 0 |
| checkpoint_written | False |


主loss无GT输入，q导数只在独立诊断图计算。深层共享祖先收到合法浅层梯度不被误判为deep-target漏梯度；deep输出、HFRM和head均不接收本轮loss梯度。

## 35. Batch20 BF16 / runtime / memory

| 项目 | 值 |
| --- | --- |
| batch | 20 |
| loss | 1.169971585 |
| active_transfer_fraction | 0.2790816128 |
| seconds | 0.081614329 |
| allocated_bytes | 2616493568 |
| reserved_bytes | 2730491904 |
| upstream_conv_energy | 56.6357701 |
| head_energy | 0 |
| all_finite | True |
| pass | True |


Batch20 peak allocated=2.4368GiB，reserved=2.5430GiB，低于22GiB预设预算。
GPU全量流程68.146s，其中冻结重放23.925s、3418张真实ADT回传32.976s；统计/bootstrap 9.754s。
环境：NVIDIA GeForce RTX 5090 D v2，PyTorch2.11.0+cu128，NumPy1.23.5。这些是审计耗时，不是25轮训练速度预测。Batch20 smoke是选定b4-path的内存，不代表未来全网络训练显存。

## 36. Zero-update identity

| 项目 | 值 |
| --- | --- |
| state_before | c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5 |
| state_after | c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5 |
| bn_before | 948d1a55cad3cbd0aee55b8aaf66a430ad0d788ae34a8693390b814d73eca293 |
| bn_after | 948d1a55cad3cbd0aee55b8aaf66a430ad0d788ae34a8693390b814d73eca293 |
| checkpoint_sha_before | 509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579 |
| checkpoint_sha_after | 509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579 |
| state_unchanged | True |
| bn_unchanged | True |
| prediction_unchanged | True |
| raw_fixed160_exact | True |


固定32+seed42随机128，共160张重放；官方推理前后hash相同：`23e333ad8e5168c464cda0cfdaae1bed085bc4d304172166e4ca54b95fca8b93`，8028160像素。hash在官方background overwrite之前取得。
Checkpoint strict load missing/unexpected=[]/[]。全部3418 gradient finite；权重/BN未更新。

## 37. Bootstrap / independent verification / tests

| 指标 | Estimate | CI low | CI high | 有效次数 |
| --- | --- | --- | --- | --- |
| Delta_image_AUROC | 0.7848415501 | 0.7771296175 | 0.7928154228 | 10000 |
| DeepCapture | 0.6403139199 | 0.6285375386 | 0.6521043883 | 10000 |
| ShallowProtection | 0.7909392616 | 0.7819087873 | 0.7998784687 | 10000 |
| DeepSelectionPrecision | 0.8405216818 | 0.8314637916 | 0.8491111529 | 10000 |
| ADT:all:mean_dm | 0.0008764349282 | 0.0008564997475 | 0.0008960555787 | 10000 |
| ADT:Top20:mean_dm | 0.002753287339 | 0.002671722472 | 0.002837791051 | 10000 |
| ADT:class0:mean_dm | 0.0005488596202 | 0.0005145769824 | 0.0005829798238 | 10000 |
| ADT:class1:mean_dm | 0.0009956362164 | 0.0009684262711 | 0.001021409305 | 10000 |
| ADT:class2:mean_dm | 0.001119128369 | 0.001058905682 | 0.001177013994 | 10000 |
| ADT:class3:mean_dm | 0.00165220867 | 0.001569671584 | 0.001735681811 | 10000 |
| ADT:Raw_Correct:harm_rate | 0.03845576077 | 0.03686118345 | 0.04007940533 | 10000 |
| ADT:Raw_Wrong:benefit_rate | 0.3558646371 | 0.3469058135 | 0.3649438532 | 10000 |
| BRR_ADT | 0.6403107425 | 0.6285278584 | 0.6521000922 | 10000 |
| HHCR_ADT | 0.201597603 | 0.1928186789 | 0.2105691733 | 10000 |
| ADT-RG:all:mean_dm | 0.0003893937984 | 0.0003754449203 | 0.0004030372835 | 10000 |
| ADT-RG:Shallow-Win:harm_rate | -0.01527619068 | -0.02346889931 | -0.00682965717 | 10000 |
| ADT-RG:Deep-Win:benefit_rate | 0.3398341436 | 0.3298024895 | 0.3500227448 | 10000 |
| ADT:all:benefit_rate | 0.2451722228 | 0.2409299839 | 0.2493726634 | 10000 |
| ADT:all:harm_rate | 0.03598824271 | 0.03461180889 | 0.03739629163 | 10000 |
| ADT:Top20:benefit_rate | 0.378425423 | 0.3688255639 | 0.3880024399 | 10000 |
| ADT:Top20:harm_rate | 0.0522194825 | 0.0491504101 | 0.05546085016 | 10000 |
| ADT:Deep-Win:benefit_rate | 0.6403139199 | 0.6285375386 | 0.6521043883 | 10000 |
| ADT:Deep-Win:harm_rate | 0 | 0 | 0 | 10000 |
| ADT:Shallow-Win:benefit_rate | 0.007446732896 | 0.006636794996 | 0.008319991939 | 10000 |
| ADT:Shallow-Win:harm_rate | 0.201608538 | 0.1928206525 | 0.2105801348 | 10000 |


固定10000 image-level paired resamples，seed42。指标按各自估计量重算：AUC对双标签图像均值，其余池化分母；不做pixel bootstrap。RNG序列SHA=`98e6164a3524dde42fc993cac0b5665076f7ebac7f6a73b7420d20c81022d00b`。
独立验证不导入主审计/分析模块：另写FP32 loss/梯度重放、FP64 epsilon-KL/JS解析式、梯形AUROC和直接gather bootstrap。

| 独立检查 | PASS |
| --- | --- |
| immutable_sources | True |
| observation_hash | True |
| full3418_order | True |
| phase2b15_delta_exact | True |
| phase2b18_raw_exact | True |
| q_frozen_replay | True |
| random_seed42_exact | True |
| random_per_image_rate_match | True |
| four_losses_exact_FP32 | True |
| q_gradient_exact_FP32 | True |
| active_direction_analytic_FP64 | True |
| JS_analytic_FP64 | True |
| raw_probability_exact | True |
| all_finite | True |
| rejected_logit_feature_dQ_zero | True |
| all_strata_margin_hierarchy | True |
| all_adjudication_and_power | True |
| selected_rejected_quality | True |
| BRR_HHCR_all_denominators | True |
| feature_statistics | True |
| 39_upstream_plus2_head | True |
| frozen_head_zero | True |
| each_b4_conv_group_active | True |
| parameter_energy_bound | True |
| 10000_paired_image_bootstrap | True |
| state_bn_checkpoint_identity | True |
| official_prediction_identity | True |
| detach_no_forbidden_gradients | True |
| BF16_batch20 | True |
| no_optimizer_test_luad | True |
| original_sources_unchanged | True |
| independent_gates_decision | True |
| secondary_strong_flags | True |

| 误差 | 最大绝对差 |
| --- | --- |
| FP32_loss | 0 |
| FP32_gradient | 0 |
| FP32_q_gradient | 0 |
| FP64_KL_formula | 1.110223025e-16 |
| FP64_JS_formula | 1.942890293e-16 |
| raw_probability | 0 |
| strata_statistics | 0 |
| adjudication | 1.110223025e-16 |
| bootstrap_replicates | 4.773959006e-15 |
| bootstrap_intervals | 1.110223025e-16 |


52项测试全部PASS、0 skips，覆盖附件规定33项测试，并增加公式、全拒绝和判定优先级检查。完整日志见rddr_phase2b19_tests.txt。

## 38. Gate A–G

| Gate | 冻结要求 | 观测 | 结果 |
| --- | --- | --- | --- |
| A | AUC≥.75 / capture≥.60 / protection≥.75 / BA≥.70 | 0.784842 / 0.640314 / 0.790939 / 0.715627 | PASS |
| B | all与Top20 Benefit>Harm；≥5/6 Mean>0 | 两组通过；6/6 | PASS |
| C | BRR≥.60 / DW Benefit≥.60 / Mean>0 | 0.640311 / 0.640314 / 0.0057069931 | PASS |
| D | HHCR≤.30 / SW Harm≤.30 / protection≥.70 | 0.201598 / 0.201609 / 0.790939 | PASS |
| E | RawCorrect Harm≤.30 / RawWrong Benefit≥.40 / active≥.10 | 0.038456 / 0.355865 / 0.281161 | FAIL |
| F | Mean优于RG + 至少一冲突rate改善 + 至少一配对CI有利 | 三个CI均有利；点条件均满足 | PASS |
| G | finite/detach/批准梯度/BF16/identity/no optimizer | 全部通过 | PASS |


Gate E仅Raw-Wrong BenefitRate失败，其余六个Gate均通过。按已确认优先级，G→A→任一B/C/D/E→F→GO；不得因为其他Gate多数通过而改变判定。

## 39. Secondary / strong flags

SOFT_DIRECTIONAL_TRANSFER_PROMISING=True。
STRONG_DIRECTIONAL_TRANSFER_SIGNAL=False。

| 组 | Loss | N | Benefit | Harm | Zero | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | ADT | 2479143 | 24.5172% | 3.5988% | 71.8840% | 0.0008764349282 | 0 |
| all | SDT | 2479143 | 24.5172% | 3.5988% | 71.8840% | 0.001142706647 | 0 |
| Raw_Wrong | ADT | 708407 | 35.5865% | 2.9820% | 61.4315% | 0.002669041267 | 0 |
| Raw_Wrong | SDT | 708407 | 35.5865% | 2.9820% | 61.4315% | 0.004048187593 | 0 |
| Shallow-Win | ADT | 182899 | 0.7447% | 20.1609% | 79.0945% | -0.002134815991 | 0 |
| Shallow-Win | SDT | 182899 | 0.7447% | 20.1609% | 79.0945% | -0.002509320191 | 0 |


SDT满足预先给定的三项次要标准，但本轮仍以ADT A–G判定；Raw-Wrong主覆盖不足不能由soft flag推翻。Strong条件要求Raw-Wrong Benefit≥50%，本轮不满足。

## 40. Scientific interpretation / limitations

本轮证明了什么：对称裁决可以作为有信息的方向开关；相较随机同数量选择，ADT在预注册三项差值上均获得有利CI；大量Shallow-Win位置被拒绝，局部有害迁移发生率显著减少。

本轮没有证明什么：并未形成通过全部readiness门槛的机制。Raw-Wrong纠错覆盖只有35.5865%，即便被选中部分质量较好，仍未达到40%；Shallow-Win剩余更新的平均负向margin幅度更大；class3层级安全性UNDERPOWERED。

因此“裁决有用”不等于“现在可以训练”。方向选择和覆盖之间的权衡仍存在；本轮不会调整阈值、放开某类别、增加第三证据、挑Top20训练、改变loss尺度或改随机seed。局部logit梯度不替代实际优化过程及最终分割评价。所有结论限于C0 seed42固定checkpoint和这套validation诊断。

## 41. Exact decision / stop

A/B/C/D/E/F/G = PASS / PASS / PASS / PASS / FAIL / PASS / PASS。

判定：`ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE`。

停止在完整审计报告，未训练、未选择lambda、未运行test/LUAD、未创建新checkpoint。后续若改变合同，需要独立提出并审核；本轮不自动补救失败项。

机器可读输出与CSV位于`audit/results/rddr_phase2b19/`；服务器大缓存位于`/home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1`，无多GB特征张量落盘。

DECISION = ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE
