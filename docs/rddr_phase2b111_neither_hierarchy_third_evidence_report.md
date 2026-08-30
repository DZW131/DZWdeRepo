# RDDR Phase-2B1.11 — Neither-Hierarchy / Third-Evidence Feasibility Audit

完整实验报告｜BCSS validation-only｜C0 seed42 final Epoch25｜zero training

**结论：第三证据能富集 Both-Wrong，但本轮候选不够可靠，未通过可用性审计。** Gate A/B/C/D/E/F = FAIL / FAIL / PASS / PASS / PASS / FAIL。Gate A 失败来自 CandidatePrecision，而非救回数量不足；这是固定决策标签的含义边界，不能误读为“没有第三证据信号”。

## 1. Provenance / SHA / commands

纯 A0：`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`。从该提交新建 `feature/rddr-phase2b111-third-evidence`，PR base=`baseline/official-a0`。原 `network/`、`tool/`、`train_sshr.py` 不变；只新增审计代码，不引入创新网络。

主运行与独立核验 commit：`57c1c2da9541abd9f70105ebe4902ddf6d7643a3` / `57c1c2da9541abd9f70105ebe4902ddf6d7643a3`。冻结合同 SHA256：`04950d44dabf41fe3b4ec306e88b56fe2628f870cef971011321715eacd274ba`。

全部3,418张既有 validation 缓存，native28位置；以下不是重新测得的224×224官方分割指标。

- `native`：`/home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz`
  SHA256：`767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a`
- `derived`：`/home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz`
  SHA256：`237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514`
- `observations`：`/home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_observations.npz`
  SHA256：`d4f65c519920c010e307ba8f32fb8e110387e0e14db73baa7c43163072ad0f1a`
- `checkpoint`：`/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
  SHA256：`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- `previous_summary`：`/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_summary.json`
  SHA256：`a2137dc73dabf57dc8b1e0457138108127be8bbbe167d04bc8dc213753307850`
- `previous_runtime`：`/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_runtime.json`
  SHA256：`fbd9ccd0741e0e25dff010af2bd7f2a95636cbea002395322c6d86e05f840bdf`
- `previous_identity`：`/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_identity_audit.json`
  SHA256：`9a5e40333ed7431da9838b2b903bb2e69affb8c16b98dbd5b579504fd4c2f071`
- `previous_verification`：`/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_verification.json`
  SHA256：`a8d908bc450940ba295a3800be59065619910a43b369c2499a76bbee7d0dddf9`

实际执行命令（历史记录；复跑必须更换 output，不覆盖 formal_r1）：

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b111
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b111_audit.py --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz --observations /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_observations.npz --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --previous-summary /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_summary.json --previous-runtime /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_runtime.json --previous-identity /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_identity_audit.json --previous-verification /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_verification.json --output /home/duyanhong/experiments/RDDR_PHASE2B111/formal_r1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/verify_rddr_phase2b111.py --run /home/duyanhong/experiments/RDDR_PHASE2B111/formal_r1
RDDR_PHASE2B111_RUN=/home/duyanhong/experiments/RDDR_PHASE2B111/formal_r1 /home/duyanhong/miniconda3/envs/sshr5090/bin/python -m unittest discover -s tests -p 'test_rddr_phase2b111*.py' -v
```

| 工程项目 | 实测 |
| --- | --- |
| 环境 | NVIDIA GeForce RTX 5090 D v2；PyTorch 2.11.0+cu128；NumPy 1.23.5 |
| 主审计耗时 | 15.9748 s |
| 概率-only GPU重放 | 2.8704 s（不是网络forward） |
| GPU peak allocated / reserved | 15.0903 / 24.0000 MiB |
| 主流程进程 peak RSS | 2.4279 GiB |
| 测试 | 54/54，零skip |
| 独立检查 | 29/29 PASS |
| model forward / backward / autograd / optimizer step / checkpoint write | 0 / 0 / 0 / 0 / 0 |

本轮新测文件SHA与原始源码一致性；旧模型state/BN/fixed160预测identity原样继承，**不是本轮重测**。所有张量有限；空人口或单标签统计记NA/null，不能用0冒充。

## 2. Frozen Phase2B1.10 evidence

| 冻结证据 | 数值 |
| --- | --- |
| Raw-Wrong / rejected Raw-Wrong | 708,407 / 435,185 |
| Rejected DW / BW / SW | 113,204 / 321,981 / 144,662 |
| 原 Gate-E 缺口 / residual beneficial | 31,266 / 177,865 |
| S_D utility / winner / interior image AUC | 0.5002 / 0.6083 / 0.5285 |
| BW context救回 / 错误第三类 | 108,541 / 19,165 |
| BW内原候选precision | 84.9929% |

上轮决定 `RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED` 保留；S_D recovery 门槛失败不能靠本轮重新阈值化挽回。上轮84.99%来自 GT-defined Both-Wrong 子群，不是 GT-blind 候选的 precision。

## 3. Scientific roles

`q=JS(ps,pd)/ln2` 是 need，不是 direction；`Delta_sym` 是 shallow/deep 相对裁决；本轮只测试独立问题：两层级都不足时，邻域是否提出可信替代类别。Phase110中q的utility AUC=0.9067、winner AUC=0.5036，不能因为q排名某项好就替换本轮主分数。上述是信号的科学角色，不是本轮把创新模块装入A0。

## 4. Residual universe

`U_R = {m_D=0}` 首先在全部3418×784=2,679,712位置上生成，完全不读GT。all-native U_R有1,925,950位置；事后GT0–3评价人口有1,782,105位置。GT背景4与ignore255只在评价时排除，不进入候选函数、不作为部署可用mask。所有科学precision都是前景评价人口口径；背景候选数量仍必须披露，不能声称全像素部署已被验证。

## 5. ctx replay

保持15×15有效图像内邻域、exclude self；ctx_S/ctx_D各自求概率均值，再ctx_sym=0.5(ctx_S+ctx_D)。

| 检查 | 最大绝对误差 |
| --- | --- |
| 四个support与三个context FP32重放 | 0 |
| raw_logits→ps FP32重放 | 0 |
| q重算（原缓存q保留） | 5.960464478e-08 ≤1e-7 |
| 独立FP64 separable box-filter | 1.569868815e-07 <1e-6 |

FP64交叉检查的舍入差不用于替换冻结context或改变candidate。没有窗口搜索或新教师训练。

## 6. c_s / c_d / c_c

`cs=argmax(ps), cd=argmax(pd), cc=argmax(ctx_sym)`，均沿用first-index tie规则。

| 人口 | raw argmax ties | deep argmax ties | ctx argmax ties |
| --- | --- | --- | --- |
| all_native | 1662 | 1445 | 0 |
| foreground | 1521 | 1280 | 0 |
| background4 | 141 | 165 | 0 |
| ignore255 | 0 | 0 | 0 |

这些ties是人口内的argmax并列位置计数；不把它们误称为strict-margin拒绝数。

## 7. Alternative candidate

```text
a_alt = (cc != cs) AND (cc != cd)
M_alt = ctx(cc) - max(ctx(cs), ctx(cd))
A_alt = (m_D == 0) AND a_alt AND (M_alt > 0)
```

A_alt函数输入只有ps/pd/ctx/mD，不含GT。M_alt==0严格拒绝；本次strict-zero拒绝数为0，与旧“不同于both”提议计数没有tie导致的差异。这里只计算附件规定的诊断候选，未设计额外恢复gate或写回推理。

## 8. M_alt

主分数固定为替代类别相对两个层级候选的context优势。Rescue/gradient任务在A_alt上用原M_alt；Both-Wrong检测在U_R上用`where(A_alt,M_alt,0)`。不加温度、不乘q、不翻符号、不做排名阈值或Top-k。Controls在U_R保留各自原始值，不额外乘候选mask。

## 9. Candidate counts

| 人口 | 全部位置 | U_R | 候选 | 候选/U_R | Strict-zero rejected |
| --- | --- | --- | --- | --- | --- |
| all_native | 2,679,712 | 1,925,950 | 227,781 | 11.8269% | 0 |
| foreground | 2,479,143 | 1,782,105 | 202,678 | 11.3730% | 0 |
| background4 | 200,569 | 143,845 | 25,103 | 17.4514% | 0 |
| ignore255 | 0 | 0 | 0 | NA | 0 |

全部native候选227,781，其中前景202,678、背景25,103、ignore 0。前景候选分布于3,167张图像；所有3,418张均处理，没有因无前景或无候选而丢图。

## 10. Candidate composition

| 前景候选来源 | Count | 占全部前景候选 |
| --- | --- | --- |
| BothWrong_ctx_correct | 108,541 | 53.5534% |
| BothWrong_ctx_wrong | 19,165 | 9.4559% |
| DeepWin_intrusion | 9,966 | 4.9172% |
| ShallowWin_intrusion | 8,529 | 4.2082% |
| StableCorrect_intrusion | 56,477 | 27.8654% |
| other | 0 | 0.0000% |

五个非空组互斥且穷尽，other=0。Stable-Correct误入56,477，占所有候选27.8654%，是不能只报告Both-Wrong precision的直接证据。

## 11. Deployment-style CandidatePrecision

`108541 / 202678 = 53.5534%`，95% CI **[52.2799%, 54.8400%]**。

这项precision是先GT-blind生成候选、再仅在GT0–3评价，不是先用GT筛选Both-Wrong。正类108,541、失败94,137；点值<65%，CI下界也未超过55%，Gate A的precision两项均失败。上轮BW内84.9929%仍可重现，但不能替代本项53.5534%。

## 12. Hard Repair / Harm / WrongToWrong

只做内存诊断：候选处raw→cc，其余位置保留raw；不生成新模型预测文件、不改变官方inference。

| 评价人口 | Repair | Harm | Wrong→Wrong activated | StableCorrect activated | NetRepair | Accuracy delta |
| --- | --- | --- | --- | --- | --- | --- |
| U_R | 108,541 | 65,006 | 29,131 | 0 | 43,535 | 2.4429 pp |
| all_foreground | 108,541 | 65,006 | 29,131 | 0 | 43,535 | 1.7561 pp |

| 评价人口 | Raw accuracy | 诊断后accuracy | Wrong→Wrong full | StableCorrect full |
| --- | --- | --- | --- | --- |
| U_R | 75.5803% | 78.0232% | 326,644 | 1,281,914 |
| all_foreground | 71.4253% | 73.1814% | 599,866 | 1,705,730 |

U_R accuracy delta 95% CI=[2.2128%, 2.6684%]；全部前景 delta CI=[1.5910%, 1.9183%]（对应百分点）。Hard net=43,535，count-equivalent CI=[39434.3395, 47553.6941]。净修复为正不能抵消预注册precision/排名失败；这是native28 hard accuracy，不是mIoU提升。

## 13. Coverage versus 31,266 gap

| 项目 | 结果 |
| --- | --- |
| ThirdRescueCount | 108,541 |
| count-equivalent 95% CI | [104859.9357, 112235.9806] |
| RequiredGap | 31,266 |
| Rescue / gap | 3.4715 |
| 该比值95% CI | [3.3538, 3.5897] |

数量与下界都足够。**Gate A失败不是数量不足，而是其附带的CandidatePrecision不合格。**此处hard-label headroom与旧局部导数BenefitRate不是同一事件；不宣称旧Gate E被补齐、或Full25一定获益。

## 14. M_alt rescue ranking

| 指标 | 结果 |
| --- | --- |
| Image-balanced AUROC | 0.6249 |
| 95% CI | [0.6149, 0.6349] |
| Pooled AUROC | 0.6401 |
| AUPRC (noninterpolated AP) | 0.6796 |
| Positive prevalence | 53.5534% |
| 正/负 | 108,541 / 94,137 |
| Dual-label eligible / 有targets图像 | 2167 / 3167 |
| zero excluded | 0 |

Task B在全部前景候选内区分Third-Rescue vs Alternative-Failure。Image AUC=0.6249，虽下界>0.50，但未达0.65，且95%CI上界约0.6349。不能改用pooled或其它score过门槛。

## 15. Both-Wrong detection

| 指标 | 结果 |
| --- | --- |
| Image-balanced AUROC | 0.7591 |
| 95% CI | [0.7523, 0.7660] |
| Pooled AUROC | 0.6754 |
| AUPRC (noninterpolated AP) | 0.4072 |
| Positive prevalence | 18.0675% |
| 正/负 | 321,981 / 1,460,124 |
| Dual-label eligible / 有targets图像 | 3187 / 3416 |
| zero excluded | 0 |

`P(BW|A_alt)=63.0093%`，高于`P(BW|U_R)=18.0675%`。Gate C PASS。这个结果支持候选富集neither-hierarchy位置，但不能证明context给出的那个替代类别正确；检测Both-Wrong与安全选择其正确第三类是不同任务。

## 16. Controls

C_ctx=max(ctx)，E_ctx=-H(ctx)=sum ctx log(ctx+1e-8)，q、Delta沿用冻结值，D_hier=1-max(max ps,max pd)。所有方向固定。

### rescue

| Score | Pooled AUC | Image AUC | 95% CI | AP |
| --- | --- | --- | --- | --- |
| M_alt | 0.6401 | 0.6249 | [0.6149, 0.6349] | 0.6796 |
| C_ctx | 0.6442 | 0.6221 | [0.6121, 0.6324] | 0.6803 |
| E_ctx | 0.5971 | 0.5628 | [0.5519, 0.5742] | 0.6319 |
| q | 0.5151 | 0.5298 | [0.5207, 0.5387] | 0.5364 |
| Delta_sym | 0.4736 | 0.4665 | [0.4584, 0.4748] | 0.5200 |
| D_hier | 0.5446 | 0.5655 | [0.5566, 0.5745] | 0.5664 |

### bothwrong

| Score | Pooled AUC | Image AUC | 95% CI | AP |
| --- | --- | --- | --- | --- |
| M_alt | 0.6754 | 0.7591 | [0.7523, 0.7660] | 0.4072 |
| C_ctx | 0.2241 | 0.2780 | [0.2722, 0.2839] | 0.1115 |
| E_ctx | 0.2319 | 0.2936 | [0.2879, 0.2993] | 0.1123 |
| q | 0.6452 | 0.6049 | [0.5983, 0.6116] | 0.2402 |
| Delta_sym | 0.2651 | 0.2840 | [0.2779, 0.2900] | 0.1165 |
| D_hier | 0.6288 | 0.6291 | [0.6232, 0.6350] | 0.2583 |

### gradient

| Score | Pooled AUC | Image AUC | 95% CI | AP |
| --- | --- | --- | --- | --- |
| M_alt | 0.6231 | 0.6270 | [0.6175, 0.6364] | 0.7836 |
| C_ctx | 0.5498 | 0.5933 | [0.5836, 0.6031] | 0.7469 |
| E_ctx | 0.4640 | 0.5163 | [0.5060, 0.5270] | 0.6861 |
| q | 0.5997 | 0.5800 | [0.5711, 0.5887] | 0.7511 |
| Delta_sym | 0.4505 | 0.4639 | [0.4559, 0.4720] | 0.6544 |
| D_hier | 0.5903 | 0.5727 | [0.5638, 0.5815] | 0.7415 |

没有仅展示最佳对照，没有翻转低AUC的score，也没有拼接置信度、q与M_alt训练分类器。[完整 CSV](../audit/results/rddr_phase2b111/rddr_phase2b111_score_controls.csv)

## 17. Context KL logit gradient

```text
t = stopgrad(ctx_sym); p = softmax(L_raw); eps = 1e-8
KL = sum_c t_c * [log(t_c+eps) - log(p_c+eps)]
r_c = t_c*p_c/(p_c+eps)
g_c = p_c*sum_j(r_j) - r_c
v = -g
```

逐候选、未加权、未归一化，不乘q、不聚合训练loss。使用冻结FP32 p/t升至FP64计算解析公式；非候选g=0。不是简化的p−t，亦没有调用autograd/backward。GT margin按原始logits的全部max-tied nonGT competitor计算精确方向导数。

| 独立检查 | 误差/结果 |
| --- | --- |
| 显式softmax Jacobian vs闭式公式 | 4.440892099e-16 |
| 固定128候选×4通道FP64 logit有限差分 | 1.064749137e-07 <1e-6 |
| 实测candidate dM符号一致性 | 完全一致，无epsilon重标 |
| 非候选梯度/dM | 精确0 |

真实有限差分从原始logits重算FP64 softmax，只用于验证；没有替换主流程的冻结概率或标签。本轮每像素未加权导数尺度不能与旧q加权/全图归一化loss的dM幅度直接比较。

## 18. Candidate gradient utility

| Group | N | Benefit% | Harm% | Zero% | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- |
| all | 202,678 | 68.4011% | 31.5989% | 0.0000% | 0.240582 | 0.400727 |
| Top20 | 52,574 | 77.5897% | 22.4103% | 0.0000% | 0.314892 | 0.387532 |
| Bottom80 | 150,104 | 65.1828% | 34.8172% | 0.0000% | 0.214555 | 0.408810 |
| Q1 | 22,693 | 55.2770% | 44.7230% | 0.0000% | 0.136322 | 0.680835 |
| Q2 | 36,032 | 62.4861% | 37.5139% | 0.0000% | 0.227250 | 0.581104 |
| Q3 | 44,966 | 65.7986% | 34.2014% | 0.0000% | 0.229634 | 0.461597 |
| Q4 | 55,080 | 70.9259% | 29.0741% | 0.0000% | 0.226979 | 0.327860 |
| Q5 | 43,907 | 79.5363% | 20.4637% | 0.0000% | 0.333684 | 0.394715 |
| class0 | 66,220 | 71.0193% | 28.9807% | 0.0000% | 0.273670 | 0.450079 |
| class1 | 77,382 | 67.1810% | 32.8190% | 0.0000% | 0.270237 | 0.408813 |
| class2 | 41,358 | 57.8147% | 42.1853% | 0.0000% | 0.055179 | 0.249702 |
| class3 | 17,718 | 88.6556% | 11.3444% | 0.0000% | 0.420174 | 0.439086 |
| boundary | 31,846 | 56.2331% | 43.7669% | 0.0000% | 0.049109 | 0.189019 |
| interior | 170,832 | 70.6694% | 29.3306% | 0.0000% | 0.276276 | 0.433791 |
| ThirdRescue | 108,541 | 100.0000% | 0.0000% | 0.0000% | 0.711899 | 0.682686 |
| AlternativeFailure | 94,137 | 31.9672% | 68.0328% | 0.0000% | -0.302852 | -0.368261 |
| BothWrong | 127,706 | 99.3548% | 0.6452% | 0.0000% | 0.669651 | 0.640444 |
| DeepWin_intrusion | 9,966 | 97.0098% | 2.9902% | 0.0000% | 0.400461 | 0.389325 |
| ShallowWin_intrusion | 8,529 | 5.4989% | 94.5011% | 0.0000% | -0.409969 | -0.380891 |
| RawCorrect | 65,006 | 3.2059% | 96.7941% | 0.0000% | -0.626846 | -0.611521 |
| RawWrong | 137,672 | 99.1850% | 0.8150% | 0.0000% | 0.650164 | 0.616915 |
| StableCorrect_intrusion | 56,477 | 2.8596% | 97.1404% | 0.0000% | -0.659599 | -0.660846 |

以上所有分组均在candidate内评价，分母不是原整组。全候选有益138,634、有害64,044、zero0。ThirdRescue全部局部有益；但AlternativeFailure内仍有30,093个局部有益位置，因此hard错误与局部margin下降不是等价标签。Deep-Win intrusion中soft dM大多有益，同时hard替代类别仍全部错误，不能混用两种“安全”概念。

## 19. M_alt gradient-utility ranking

| 指标 | 结果 |
| --- | --- |
| Image-balanced AUROC | 0.6270 |
| 95% CI | [0.6175, 0.6364] |
| Pooled AUROC | 0.6231 |
| AUPRC (noninterpolated AP) | 0.7836 |
| Positive prevalence | 68.4011% |
| 正/负 | 138,634 / 64,044 |
| Dual-label eligible / 有targets图像 | 2135 / 3167 |
| zero excluded | 0 |

全候选Benefit=68.4011% > Harm=31.5989%，Mean dM=0.240582>0，M_alt gradient image AUC=0.6270≥0.60，Gate D PASS。局部方向有效不保证训练稳定、有限步获益或硬预测安全。

## 20. Raw-Correct protection

| Denominator | N | Active | Activation% | Hard harm% | Third proposal% | Rescue% | Candidate precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 1,346,920 | 65,006 | 4.8263% | 4.8263% | 4.8263% | 0.0000% | 0.0000% |
| active_only | 65,006 | 65,006 | 100.0000% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |

| Denominator | Context accuracy | dM Benefit% | dM Harm% | dM Zero% | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 92.9268% | 0.1547% | 4.6715% | 95.1737% | -0.030253 | 0.000000 |
| active_only | 0.0000% | 3.2059% | 96.7941% | 0.0000% | -0.626846 | -0.611521 |

[完整 CSV](../audit/results/rddr_phase2b111/rddr_phase2b111_rawcorrect_protection.csv)

Gate E使用U_R内全部1,346,920个Raw-Correct作分母，未激活位置dM=0。Hard harm=4.8263%≤8%，gradient harm=4.6715%≤15%；对应CI分别[4.5995%, 5.0481%] / [4.4532%, 4.8876%]。**但是，被激活的65,006个Raw-Correct位置hard harm=100%，gradient harm=96.7941%。**这是少量激活下的全分母通过，不是候选对正确语义无害。

## 21. Rejected Deep-Win protection

| Denominator | N | Active | Activation% | Hard harm% | Third proposal% | Rescue% | Candidate precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 113,204 | 9,966 | 8.8036% | 0.0000% | 8.8036% | 0.0000% | 0.0000% |
| active_only | 9,966 | 9,966 | 100.0000% | 0.0000% | 100.0000% | 0.0000% | 0.0000% |

| Denominator | Context accuracy | dM Benefit% | dM Harm% | dM Zero% | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 64.8758% | 8.5403% | 0.2632% | 91.1964% | 0.035255 | 0.000000 |
| active_only | 0.0000% | 97.0098% | 2.9902% | 0.0000% | 0.400461 | 0.389325 |

[完整 CSV](../audit/results/rddr_phase2b111/rddr_phase2b111_deepwin_protection.csv)

全分母第三类误入8.8036%，与Phase110一致且≤12%。9,966次候选激活均未硬救回；raw本来错误，所以raw-correct→wrong定义的hard harm为0，但不能据此说错误第三类安全。Context accuracy 64.8758%是对整个DW人口不加候选限制的诊断；实际候选context accuracy为0。

## 22. Rejected Shallow-Win protection

| Denominator | N | Active | Activation% | Hard harm% | Third proposal% | Rescue% | Candidate precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 144,662 | 8,529 | 5.8958% | 5.8958% | 5.8958% | 0.0000% | 0.0000% |
| active_only | 8,529 | 8,529 | 100.0000% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |

| Denominator | Context accuracy | dM Benefit% | dM Harm% | dM Zero% | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 73.1830% | 0.3242% | 5.5716% | 94.1042% | -0.024171 | 0.000000 |
| active_only | 0.0000% | 5.4989% | 94.5011% | 0.0000% | -0.409969 | -0.380891 |

[完整 CSV](../audit/results/rddr_phase2b111/rddr_phase2b111_shallowwin_protection.csv)

全分母第三类误入5.8958%，与Phase110一致且≤10%。激活的8,529个位置全部损害raw hard预测；active-only gradient harm=94.5011%。未使用GT防止这些位置被激活。

## 23. Rejected Both-Wrong rescue

| Denominator | N | Active | Activation% | Hard harm% | Third proposal% | Rescue% | Candidate precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 321,981 | 127,706 | 39.6626% | 0.0000% | 39.6626% | 33.7104% | 84.9929% |
| active_only | 127,706 | 127,706 | 100.0000% | 0.0000% | 100.0000% | 84.9929% | 84.9929% |

| Denominator | Context accuracy | dM Benefit% | dM Harm% | dM Zero% | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 33.7104% | 39.4067% | 0.2559% | 60.3374% | 0.265601 | 0.000000 |
| active_only | 84.9929% | 99.3548% | 0.6452% | 0.0000% | 0.669651 | 0.640444 |

[完整 CSV](../audit/results/rddr_phase2b111/rddr_phase2b111_bothwrong_rescue.csv)

| Group | 旧提议 | 旧救回 | 旧错误 | 严格候选 | 严格救回 | 零margin拒绝 |
| --- | --- | --- | --- | --- | --- | --- |
| BothWrong | 127,706 | 108,541 | 19,165 | 127,706 | 108,541 | 0 |
| DeepWin | 9,966 | 0 | 9,966 | 9,966 | 0 | 0 |
| ShallowWin | 8,529 | 0 | 8,529 | 8,529 | 0 | 0 |

108,541正确第三类与19,165错误第三类均精确重现。这里的84.9929%仍是BW条件precision，不得用它代替第11节包含one-correct及stable-correct的全候选precision。

## 24. Stable-Correct intrusion

| Denominator | N | Active | Activation% | Hard harm% | Third proposal% | Rescue% | Candidate precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 1,202,258 | 56,477 | 4.6976% | 4.6976% | 4.6976% | 0.0000% | 0.0000% |
| active_only | 56,477 | 56,477 | 100.0000% | 100.0000% | 100.0000% | 0.0000% | 0.0000% |

| Denominator | Context accuracy | dM Benefit% | dM Harm% | dM Zero% | Mean dM | Median dM |
| --- | --- | --- | --- | --- | --- | --- |
| full_residual_group | 95.3024% | 0.1343% | 4.5632% | 95.3024% | -0.030985 | 0.000000 |
| active_only | 0.0000% | 2.8596% | 97.1404% | 0.0000% | -0.659599 | -0.660846 |

[完整 CSV](../audit/results/rddr_phase2b111/rddr_phase2b111_stablecorrect_intrusion.csv)

56,477个双方原本正确的位置被context第三类覆盖，全部hard有害；占全部65,006次hard harm的 86.8797%。本轮没有使用“双方一致时禁用”等事后规则。

## 25. q strata

冻结Q边界：`0.020935675129294395, 0.072734534740448, 0.163648784160614, 0.3369627296924591`，side=left；Top20沿用缓存，不在U_R或candidate中重新计算。

| Group | U_R | Candidate | Activation% | Rescue / failure | Precision | Rescue image AUC [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| Top20 | 276,393 | 52,574 | 19.0215% | 28,054 / 24,520 | 53.3610% | 0.6135 [0.5987, 0.6280] |
| Bottom80 | 1,505,712 | 150,104 | 9.9690% | 80,487 / 69,617 | 53.6208% | 0.6271 [0.6163, 0.6378] |
| Q1 | 471,814 | 22,693 | 4.8097% | 10,741 / 11,952 | 47.3318% | 0.6269 [0.6062, 0.6473] |
| Q2 | 411,242 | 36,032 | 8.7618% | 18,860 / 17,172 | 52.3424% | 0.6167 [0.6024, 0.6311] |
| Q3 | 339,381 | 44,966 | 13.2494% | 24,660 / 20,306 | 54.8414% | 0.6236 [0.6107, 0.6366] |
| Q4 | 290,794 | 55,080 | 18.9412% | 31,289 / 23,791 | 56.8065% | 0.6113 [0.5989, 0.6236] |
| Q5 | 268,874 | 43,907 | 16.3300% | 22,991 / 20,916 | 52.3629% | 0.6066 [0.5902, 0.6232] |

| Group | Dual images / rescue power | Gradient AUC / power | dM Benefit% / Harm% | Raw-Correct hard / dM harm% |
| --- | --- | --- | --- | --- |
| Top20 | 1615 / POWERED | 0.6097 / POWERED | 77.5897% / 22.4103% | 9.8154% / 9.0572% |
| Bottom80 | 2101 / POWERED | 0.6296 / POWERED | 65.1828% / 34.8172% | 4.3210% / 4.2274% |
| Q1 | 1107 / POWERED | 0.6339 / POWERED | 55.2770% / 44.7230% | 2.3266% / 2.3254% |
| Q2 | 1685 / POWERED | 0.6339 / POWERED | 62.4861% / 37.5139% | 3.8499% / 3.8382% |
| Q3 | 1816 / POWERED | 0.6291 / POWERED | 65.7986% / 34.2014% | 5.9684% / 5.9069% |
| Q4 | 1833 / POWERED | 0.6109 / POWERED | 70.9259% / 29.0741% | 9.0681% / 8.5192% |
| Q5 | 1390 / POWERED | 0.5951 / POWERED | 79.5363% / 20.4637% | 7.8981% / 7.1560% |

Top20 precision约53.36%，并未解决可靠性问题；不实施Top20-only或q阈值规则。

## 26. Per-class

| Group | U_R | Candidate | Activation% | Rescue / failure | Precision | Rescue image AUC [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| class0 | 714,244 | 66,220 | 9.2713% | 43,879 / 22,341 | 66.2625% | 0.6194 [0.5992, 0.6396] |
| class1 | 748,575 | 77,382 | 10.3372% | 46,035 / 31,347 | 59.4906% | 0.6274 [0.6089, 0.6457] |
| class2 | 218,686 | 41,358 | 18.9120% | 13,719 / 27,639 | 33.1713% | 0.5334 [0.5057, 0.5614] |
| class3 | 100,600 | 17,718 | 17.6123% | 4,908 / 12,810 | 27.7006% | 0.5454 [0.4979, 0.5938] |

| Group | Dual images / rescue power | Gradient AUC / power | dM Benefit% / Harm% | Raw-Correct hard / dM harm% |
| --- | --- | --- | --- | --- |
| class0 | 800 / POWERED | 0.6663 / POWERED | 71.0193% / 28.9807% | 3.1912% / 3.1544% |
| class1 | 1068 / POWERED | 0.6273 / POWERED | 67.1810% / 32.8190% | 4.3779% / 4.1828% |
| class2 | 438 / POWERED | 0.5297 / POWERED | 57.8147% / 42.1853% | 14.7652% / 14.3732% |
| class3 | 134 / POWERED | 0.5407 / POWERED | 88.6556% / 11.3444% | 8.1355% / 7.1598% |

四类rescue与gradient排名均POWERED（≥500正、≥500负、≥30dual图像）。只有class0/1的rescue image AUC>0.55，class2/3未达标。Class2 Raw-Correct hard harm=14.7652%，class3=8.1355%；虽然全局Gate E通过，不代表每一类都满足全局阈值。这里只披露，不改class规则、不增设事后门槛。

## 27. Boundary / interior

| Group | U_R | Candidate | Activation% | Rescue / failure | Precision | Rescue image AUC [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| boundary | 159,063 | 31,846 | 20.0210% | 12,330 / 19,516 | 38.7176% | 0.5515 [0.5395, 0.5635] |
| interior | 1,623,042 | 170,832 | 10.5254% | 96,211 / 74,621 | 56.3191% | 0.6480 [0.6366, 0.6595] |

| Group | Dual images / rescue power | Gradient AUC / power | dM Benefit% / Harm% | Raw-Correct hard / dM harm% |
| --- | --- | --- | --- | --- |
| boundary | 1553 / POWERED | 0.5542 / POWERED | 56.2331% / 43.7669% | 17.1503% / 16.5680% |
| interior | 2051 / POWERED | 0.6472 / POWERED | 70.6694% / 29.3306% | 4.0210% / 3.8942% |

复用≤7px boundary，未改窗口/边界宽度。Interior rescue image AUC=0.6480通过F的interior子项；但boundary precision仅38.7176%，其Raw-Correct hard harm=17.1503%，反映全局平均之外的风险。不得用interior-only取代global gates。

## 28. 10k bootstrap

10,000次paired image bootstrap，seed42，每次3418张有放回抽样，所有endpoint同一批索引。Image AUC只平均dual-label图像；比例每次重新汇总分子/分母，使用2.5/97.5 percentile CI。

救回count-equivalent=`draw救回数/draw全部Raw-Wrong数 × 固定708407`，与固定31266比较。Hard NetRepair count-equivalent=`draw净修复/draw前景U_R × 固定1782105`；另给出U_R与全部前景归一化accuracy delta。

索引 SHA256：`98e6164a3524dde42fc993cac0b5665076f7ebac7f6a73b7420d20c81022d00b`。独立direct-gather相对主流程bincount聚合的replicate误差 `2.842e-14`。所有endpoint有效重采样数均为10000。

| Endpoint（率0–1；count-equivalent为位置数） | Estimate | 95% CI |
| --- | --- | --- |
| CandidatePrecision | 0.535534 | [0.522799, 0.548400] |
| CandidateRate | 0.113730 | [0.111003, 0.116411] |
| ThirdRescue_count_equivalent | 108541.000000 | [104859.935705, 112235.980572] |
| ThirdRescue_to_gap | 3.471535 | [3.353801, 3.589713] |
| Hard_NetRepair_count_equivalent | 43535.000000 | [39434.339456, 47553.694053] |
| Hard_accuracy_delta_UR | 0.024429 | [0.022128, 0.026684] |
| Hard_accuracy_delta_foreground | 0.017561 | [0.015910, 0.019183] |
| RawCorrect_hard_HarmRate | 0.048263 | [0.045995, 0.050481] |
| RawCorrect_gradient_HarmRate | 0.046715 | [0.044532, 0.048876] |
| Candidate_gradient_BenefitRate | 0.684011 | [0.673955, 0.693795] |
| Candidate_gradient_HarmRate | 0.315989 | [0.306205, 0.326045] |
| Candidate_gradient_Mean_dM | 0.240582 | [0.225551, 0.255411] |
| BW_prevalence_candidate | 0.630093 | [0.619626, 0.640517] |
| BW_prevalence_UR | 0.180675 | [0.175074, 0.186310] |
| DeepWin_intrusion | 0.088036 | [0.081431, 0.094986] |
| ShallowWin_intrusion | 0.058958 | [0.054180, 0.063910] |
| rescue:all:M_alt:image_AUROC | 0.624907 | [0.614887, 0.634926] |
| rescue:all:C_ctx:image_AUROC | 0.622124 | [0.612100, 0.632444] |
| rescue:all:E_ctx:image_AUROC | 0.562799 | [0.551868, 0.574184] |
| rescue:all:q:image_AUROC | 0.529805 | [0.520706, 0.538690] |
| rescue:all:Delta_sym:image_AUROC | 0.466495 | [0.458367, 0.474800] |
| rescue:all:D_hier:image_AUROC | 0.565540 | [0.556618, 0.574460] |
| rescue:Top20:M_alt:image_AUROC | 0.613482 | [0.598697, 0.628047] |
| rescue:Bottom80:M_alt:image_AUROC | 0.627083 | [0.616309, 0.637843] |
| rescue:Q1:M_alt:image_AUROC | 0.626912 | [0.606243, 0.647296] |
| rescue:Q2:M_alt:image_AUROC | 0.616682 | [0.602366, 0.631088] |
| rescue:Q3:M_alt:image_AUROC | 0.623620 | [0.610686, 0.636558] |
| rescue:Q4:M_alt:image_AUROC | 0.611315 | [0.598888, 0.623650] |
| rescue:Q5:M_alt:image_AUROC | 0.606630 | [0.590236, 0.623234] |
| rescue:class0:M_alt:image_AUROC | 0.619354 | [0.599197, 0.639644] |
| rescue:class1:M_alt:image_AUROC | 0.627376 | [0.608860, 0.645747] |
| rescue:class2:M_alt:image_AUROC | 0.533425 | [0.505670, 0.561427] |
| rescue:class3:M_alt:image_AUROC | 0.545380 | [0.497945, 0.593786] |
| rescue:boundary:M_alt:image_AUROC | 0.551539 | [0.539548, 0.563520] |
| rescue:interior:M_alt:image_AUROC | 0.647954 | [0.636636, 0.659523] |
| bothwrong:all:M_alt:image_AUROC | 0.759112 | [0.752274, 0.766014] |
| bothwrong:all:C_ctx:image_AUROC | 0.278026 | [0.272167, 0.283877] |
| bothwrong:all:E_ctx:image_AUROC | 0.293584 | [0.287853, 0.299275] |
| bothwrong:all:q:image_AUROC | 0.604919 | [0.598281, 0.611574] |
| bothwrong:all:Delta_sym:image_AUROC | 0.284020 | [0.277918, 0.290025] |
| bothwrong:all:D_hier:image_AUROC | 0.629082 | [0.623211, 0.635036] |
| bothwrong:Top20:M_alt:image_AUROC | 0.758263 | [0.750392, 0.765870] |
| bothwrong:Bottom80:M_alt:image_AUROC | 0.752886 | [0.745925, 0.759981] |
| bothwrong:Q1:M_alt:image_AUROC | 0.715805 | [0.707286, 0.724571] |
| bothwrong:Q2:M_alt:image_AUROC | 0.740476 | [0.732733, 0.748132] |
| bothwrong:Q3:M_alt:image_AUROC | 0.743052 | [0.735557, 0.750747] |
| bothwrong:Q4:M_alt:image_AUROC | 0.740712 | [0.732944, 0.748398] |
| bothwrong:Q5:M_alt:image_AUROC | 0.761397 | [0.753045, 0.769351] |
| bothwrong:class0:M_alt:image_AUROC | 0.768816 | [0.756632, 0.780697] |
| bothwrong:class1:M_alt:image_AUROC | 0.727470 | [0.715957, 0.739291] |
| bothwrong:class2:M_alt:image_AUROC | 0.528705 | [0.508917, 0.548733] |
| bothwrong:class3:M_alt:image_AUROC | 0.581425 | [0.552978, 0.609728] |
| bothwrong:boundary:M_alt:image_AUROC | 0.567414 | [0.562670, 0.572285] |
| bothwrong:interior:M_alt:image_AUROC | 0.783770 | [0.776797, 0.790730] |
| gradient:all:M_alt:image_AUROC | 0.627023 | [0.617509, 0.636424] |
| gradient:all:C_ctx:image_AUROC | 0.593251 | [0.583617, 0.603114] |
| gradient:all:E_ctx:image_AUROC | 0.516346 | [0.506031, 0.527014] |
| gradient:all:q:image_AUROC | 0.579986 | [0.571122, 0.588746] |
| gradient:all:Delta_sym:image_AUROC | 0.463919 | [0.455885, 0.472048] |
| gradient:all:D_hier:image_AUROC | 0.572742 | [0.563778, 0.581489] |
| gradient:Top20:M_alt:image_AUROC | 0.609694 | [0.595906, 0.623348] |
| gradient:Bottom80:M_alt:image_AUROC | 0.629649 | [0.619182, 0.639996] |
| gradient:Q1:M_alt:image_AUROC | 0.633899 | [0.613170, 0.654498] |
| gradient:Q2:M_alt:image_AUROC | 0.633936 | [0.619198, 0.648289] |
| gradient:Q3:M_alt:image_AUROC | 0.629094 | [0.616242, 0.641801] |
| gradient:Q4:M_alt:image_AUROC | 0.610943 | [0.598570, 0.623119] |
| gradient:Q5:M_alt:image_AUROC | 0.595141 | [0.579786, 0.611156] |
| gradient:class0:M_alt:image_AUROC | 0.666300 | [0.647358, 0.684451] |
| gradient:class1:M_alt:image_AUROC | 0.627338 | [0.611379, 0.643386] |
| gradient:class2:M_alt:image_AUROC | 0.529685 | [0.508740, 0.550341] |
| gradient:class3:M_alt:image_AUROC | 0.540692 | [0.504170, 0.576462] |
| gradient:boundary:M_alt:image_AUROC | 0.554160 | [0.541393, 0.566649] |
| gradient:interior:M_alt:image_AUROC | 0.647191 | [0.636689, 0.657770] |

区间仅表示这个固定checkpoint下的图像抽样不确定性，不是多seed方差；不覆盖长期反复方案选择的适应性偏差。未做多重比较校正，controls仅为诊断。

## 29. Gate A–F

| Gate | 冻结要求 | 实测 | 结论 |
| --- | --- | --- | --- |
| A | 救回≥31266且CI下界≥31266；precision≥.65且lower>.55 | 救回108541/lower104859.94通过；precision.5355/lower.5228失败 | FAIL |
| B | rescue image AUC≥.65，lower>.50 | 0.6249 [0.6149, 0.6349] | FAIL |
| C | BW image AUC≥.65，lower>.50，且BW富集 | 0.7591；63.0093%>18.0675% | PASS |
| D | candidate Benefit>Harm，Mean dM>0，gradient AUC≥.60 | 68.4011%>31.5989%；mean .240582；AUC .6270 | PASS |
| E | RC hard≤8%，RC dM harm≤15%，DW intrusion≤12%，SW intrusion≤10% | 4.8263% / 4.6715% / 8.8036% / 5.8958% | PASS |
| F | interior rescue AUC>.60且≥3个powered class AUC>.55 | interior .6480通过；仅2/4类通过（4/4 powered） | FAIL |

按A→B/C→D→E→F优先级输出，A已经失败；B与F的失败仍完整披露。不存在仅因power不足而暂停判定的情况。

## 30. Secondary flags

`CONTEXT_CONFIDENCE_DIAGNOSTIC_STRONGER = FALSE`。同一candidate rescue任务的C_ctx image AUC=0.6221，略低于M_alt=0.6249；尽管C_ctx的pooled AUC略高，也不能替代预注册image AUC比较。未根据任何control设计新机制。

## 31. STRONG_THIRD_EVIDENCE_SIGNAL

`FALSE`。A/B/F失败；precision .5355<.75、rescue AUC .6249<.75、gradient AUC .6270<.70。即使救回数量≥2gap、全局RC hard harm≤5%，也不满足Strong的合取要求。

## 32. Scientific interpretation / delivery

本轮将问题定位为：**可以识别更可能“两层级都错”的位置，但仍不能足够可靠地选择正确替代语义。**第三证据存在，局部soft方向平均有益，硬净修复也为正；这些事实不能替代候选precision和跨类别可靠性门槛。

关键污染来自双方原本都正确的Stable-Correct位置，以及Shallow-Win位置。全分母保护通过由较低激活率支撑，active-only仍有严重伤害；boundary和class2更明显。上述仅为机制定位，不在本轮加一致性排除、边界规则、类别阈值或新score。

因此不进入Phase2B1.12 gate设计，不训练，不做test/LUAD或其它seed；等待用户下一份独立方案。实验失败不等于所有context思路不可能，只说明这套冻结候选、分数和合同未达标。

交付54项测试、29项独立检查、全部CSV/JSON与bootstrap replicates、输入SHA和命令，[运行说明](README_rddr_phase2b111.md)、[冻结合同](rddr_phase2b111_contract.md)、[交付摘要](rddr_phase2b111_delivery_summary.md)。证据位于`audit/results/rddr_phase2b111/`；大输入及image_statistics NPZ留服务器。没有覆盖旧实验、checkpoint或官方代码。

## 33. Exact final decision

固定决策标签如下。再次强调：**operational headroom不足在这里指Gate A的整体可操作条件不成立，具体短板是precision，不是救回数量。**完整审计结束，停止，不自动追加实验。

DECISION = THIRD_EVIDENCE_OPERATIONAL_HEADROOM_INSUFFICIENT
