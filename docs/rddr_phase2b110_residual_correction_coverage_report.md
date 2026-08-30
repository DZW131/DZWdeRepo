# RDDR Phase-2B1.10 — Residual Correction Coverage & Recoverability Audit

完整实验报告｜BCSS validation-only｜C0 seed42 / final Epoch25｜zero training

结论摘要：潜在有益残余数量充分，但预注册主分数 `S_D_sym` 未通过恢复能力门槛；仅第三证据诊断得到支持。**这不是训练 GO，也不是已经验证的安全选择器。**

## 1. Provenance / SHA / commands

本轮覆盖全部 **3,418 张 validation 图像**，使用既有 native 28×28 冻结观测；不读取训练/test/LUAD split，不加载模型，不执行新的网络 forward/backward。所有位置计数均是 native28 特征位置，不是 224×224 分割像素计数。

- 纯 A0 基线：`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`。
- 分支：`feature/rddr-phase2b110-residual-coverage`；PR 目标 `baseline/official-a0`。
- 主审计运行 commit：`61d1a8afa8a58b5b18087207040e08460078a91b`。
- 独立验证 commit：`d904bf8eeca7f7dc55bf5f243cc6c9c03f8118d6`；后续报告提交不改变上述运行来源。
- 冻结合同 SHA256：`9436874a3b4f029f9dbe576521fd93e3dc0773f84209c792507484b6070c3a9d`。

输入路径与运行前后相同的 SHA256：

- `native`：`/home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz`
  SHA256：`767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a`
- `derived`：`/home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz`
  SHA256：`237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514`
- `observations`：`/home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_observations.npz`
  SHA256：`d4f65c519920c010e307ba8f32fb8e110387e0e14db73baa7c43163072ad0f1a`
- `checkpoint`：`/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
  SHA256：`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- `previous_runtime`：`/home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_runtime.json`
  SHA256：`bb54ae356e1258baabd8795894c51d12d5532137ac352cf1d6d0c4c88d3f48a0`
- `previous_summary`：`/home/duyanhong/experiments/RDDR_PHASE2B19/report_r1/rddr_phase2b19_summary.json`
  SHA256：`84e009170aa335c0f625afdc097f86369e709ea004903881d10b9c264c7a0eb7`
- `previous_identity`：`/home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_identity_audit.json`
  SHA256：`4adffe179ac328db9ce922c7c7ab3f18de759c1cb0cff6bbd09631cde3cd6637`

准确执行命令（历史记录，既有 output 不可覆盖；复跑须使用新目录）：

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b110
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b110_audit.py --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz --observations /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_observations.npz --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --previous-runtime /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_runtime.json --previous-summary /home/duyanhong/experiments/RDDR_PHASE2B19/report_r1/rddr_phase2b19_summary.json --previous-identity /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_identity_audit.json --output /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/verify_rddr_phase2b110.py --run /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1
RDDR_PHASE2B110_RUN=/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1 /home/duyanhong/miniconda3/envs/sshr5090/bin/python -m unittest discover -s tests -p 'test_rddr_phase2b110*.py' -v
```

| 实测项目 | 结果 |
| --- | --- |
| 环境 | NVIDIA GeForce RTX 5090 D v2; PyTorch 2.11.0+cu128; NumPy 1.23.5 |
| 主审计运行时间 | 14.5374 s |
| 概率张量 GPU 重放 | 2.7353 s（不是网络推理） |
| 概率重放峰值 allocated / reserved | 15.0781 / 24.0000 MiB |
| 主流程 FP32 support / context 最大误差 | 0 / 0（四个 support、三个 context 全量重放） |
| q 重算最大误差 | 5.960464478e-08 ≤ 1e-7；保留原缓存 q |
| 独立 FP64 context 对 FP32 缓存 | 1.863757183e-07 < 1e-6；不同归约顺序交叉检查 |
| 独立排名 / context 指标最大误差 | 2.220e-16 / 0.000e+00 |
| 工程验证 | 44/44 unit+integration tests；26/26 独立检查；所有输出有限 |
| 模型 / 训练操作 | model=0, network forward=0, backward=0, optimizer step=0, checkpoint write=0 |

**Identity 证据边界：**本轮新测的是全部输入文件 SHA 前后一致、A0 原始源码不变，以及冻结张量/计数重放。既有 Phase2B1.9 的 state/BN/160-image prediction identity 原样继承，**没有**把它们标作本轮新运行的模型等价测试。checkpoint 只计算 SHA，不实例化/加载网络。

[identity 记录](../audit/results/rddr_phase2b110/rddr_phase2b110_identity_audit.json)；[独立验证](../audit/results/rddr_phase2b110/rddr_phase2b110_verification.json)；[44 项测试日志](../audit/results/rddr_phase2b110/rddr_phase2b110_tests.txt)。

## 2. Phase-2B1.9 frozen status

| Gate | A | B | C | D | E | F | G |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 冻结结果 | PASS | PASS | PASS | PASS | FAIL | PASS | PASS |

冻结决定：`ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE`。本轮不改判旧实验。

旧 Gate E 的 Raw-Wrong beneficial coverage 未达 40%。此前较低 HHCR、较高 ShallowProtection 不能概括为“已经完整安全”：负向 Shallow-Win dM 幅度增加、class3 安全证据不足的限制仍保留。本轮只诊断 residual coverage，不能用新的分组统计倒推旧的安全问题已解决。

## 3. Exact Gate-E deficit

`N_RW = 708407`；`B_ADT = 252097`。

全分母 benefit rate = `0.3558646371365613` = 35.5865%。距离 40% 的连续比例差为 **4.41353629 pp**。以下计数使用精确整数，不使用打印后截断的 0.3558646371。

## 4. Required additional beneficial count

```text
Target = ceil(2 * N_RW / 5)
       = 283363
RequiredAdditionalBenefit = 283363 - 252097 = 31266
```

因此至少需要 **31,266** 个额外有益位置；折算全 Raw-Wrong 分母为 4.41356452 pp。该整数折算率与上一节连续差的微小区别来自 ceil，不是计算冲突。

## 5. Residual population

`m_D = 1[Delta_sym > 0]`；`R = {m_D=0}`；`R_RW = R ∩ {raw wrong}`，诊断人口限定 GT 0–3，背景4/ignore255不参与。既有 m_D 只重放，不构造恢复 gate。

| Population | 位置数 | 有该人口的图像数 |
| --- | --- | --- |
| foreground | 2,479,143 | 3416 |
| Raw_Wrong | 708,407 | 3407 |
| Residual | 1,782,105 | 3416 |
| R_RW | 435,185 | 3194 |
| Rejected_Deep_Win | 113,204 | 2562 |
| Rejected_Both_Wrong | 321,981 | 3187 |
| Rejected_Shallow_Win | 144,662 | 3150 |
| Residual_Beneficial | 177,865 | 2923 |
| Residual_Harmful | 257,316 | 3139 |
| Residual_Zero | 4 | 4 |

3,418 张全部读取；其中 3,416 张在 native28 有前景。人口图像数不同不是静默丢图。

## 6. Rejected Deep-Win / Both-Wrong counts

`R_RW = 113,204 Rejected Deep-Win + 321,981 Rejected Both-Wrong = 435,185`，与冻结记录 exact replay。

Both-Wrong 占 residual Raw-Wrong 的 **73.9872%**。另有 144,662 个 Rejected Shallow-Win，用于 one-correct 对照；它们 raw 正确，不能加进 R_RW 分母。

## 7. Residual headroom

| 量 | 结果 |
| --- | --- |
| Residual beneficial / harmful / zero | 177,865 / 257,316 / 4 |
| CoverageHeadroom = beneficial / ALL Raw-Wrong | 25.1077% |
| 95% CI of CoverageHeadroom | [24.4339%, 25.7795%] |
| 原始 N_RW 下 count-equivalent 95% CI | [173091.4789, 182623.7259] |
| HeadroomOverGap | 146,599 |
| 只从 rejected Deep-Win 补足缺口所需比例 | 27.6192% |
| Residual beneficial prevalence（含 zero 分母） | 40.8711% |
| 该 prevalence 95% CI | [39.9205%, 41.8468%] |

Gate A 的分母始终是所有 Raw-Wrong，不是 residual-only。理想选择器的局部导数有益机会数量充足，但这只是 arithmetic/local-derivative headroom，**不等于**这些位置经有限步训练后一定转对，更不等于 mIoU 增益。

## 8. Primary S_D hypothesis

`S_D_sym = 0.5(T_DS + T_DD)`；`S_S_sym = 0.5(T_SS + T_SD)`；`Delta_sym = S_D_sym - S_S_sym`。主假设是被相对 support 拒绝的位置中，绝对 deep support 越高越能识别有益的 deep-transfer。GT 不进入上述 score。本轮始终使用正向 S_D：没有取反、阈值化、top-k、温度或事后替换。

## 9. Residual beneficial / harmful definition

在 R_RW 上，冻结 `UDT dM > 0` 为 beneficial，`< 0` 为 harmful，`==0` 单列。用冻结 FP32 raw logits/gradient，FP64 累积 `v=-g` 的 GT-vs-max-competitor margin 方向导数；遇到最大 competitor 并列时采用方向导数的 exact max-tie 规则。没有重新 backward。

正类 177,865；负类 257,316；4 个零值仅在二分类排名中排除，二分类 prevalence = 40.8715%，95% CI [39.9208%, 41.8470%]。GT 只参与人口/标签/指标构造；这是一种回顾性诊断，不是可直接部署的 GT-blind 选择人口。

## 10. S_D residual utility AUROC

| 指标 | 结果 |
| --- | --- |
| Primary image-balanced AUROC | 0.5002 |
| 95% CI | [0.4913, 0.5090] |
| Pooled AUROC | 0.4298 |
| Pooled AUPRC (Average Precision) | 0.3444 |
| Positive prevalence | 40.8715% |
| 正 / 负 / binary 总数 | 177,865 / 257,316 / 435,181 |
| Dual-label eligible images / images with targets | 2868 / 3194 |

Gate B FAIL：主 image AUC 约为随机水平，且下界不高于 0.50；不能以 pooled AUC 或其它 score 替代主指标。

## 11. S_D rejected winner AUROC

| 指标 | 结果 |
| --- | --- |
| Primary image-balanced AUROC | 0.6083 |
| 95% CI | [0.5993, 0.6174] |
| Pooled AUROC | 0.5868 |
| Pooled AUPRC (Average Precision) | 0.4637 |
| Positive prevalence | 43.9003% |
| 正 / 负 / binary 总数 | 113,204 / 144,662 / 257,866 |
| Dual-label eligible images / images with targets | 2468 / 3244 |

正类是 Rejected Deep-Win，负类是 Rejected Shallow-Win；仅 exactly-one-correct conflict。下界 >0.50，但 image AUC <0.65，因此 Gate C FAIL，而不是完全没有排名信号。

## 12. Delta / q / confidence / entropy controls

所有方向按合同冻结：Delta 不取绝对值/不翻符号；q 使用原缓存；confidence = max(pd)-max(ps)；entropy = H(ps)-H(pd)，H=-sum p log(p+1e-8)，单位 nats。

### residual_utility

| Score | Pooled AUROC | Image AUROC | Image AUROC 95% CI | AUPRC |
| --- | --- | --- | --- | --- |
| S_D_sym | 0.4298 | 0.5002 | [0.4913, 0.5090] | 0.3444 |
| Delta_sym | 0.4482 | 0.4853 | [0.4776, 0.4932] | 0.3929 |
| q | 0.9166 | 0.9067 | [0.9030, 0.9103] | 0.9179 |
| deep_confidence_advantage | 0.6875 | 0.6653 | [0.6586, 0.6717] | 0.6331 |
| deep_entropy_advantage | 0.6604 | 0.6333 | [0.6265, 0.6400] | 0.6170 |

### rejected_winner

| Score | Pooled AUROC | Image AUROC | Image AUROC 95% CI | AUPRC |
| --- | --- | --- | --- | --- |
| S_D_sym | 0.5868 | 0.6083 | [0.5993, 0.6174] | 0.4637 |
| Delta_sym | 0.6911 | 0.6645 | [0.6558, 0.6735] | 0.5962 |
| q | 0.5757 | 0.5036 | [0.4957, 0.5116] | 0.4629 |
| deep_confidence_advantage | 0.7131 | 0.6542 | [0.6467, 0.6615] | 0.6496 |
| deep_entropy_advantage | 0.7141 | 0.6416 | [0.6340, 0.6490] | 0.6562 |

q 的 utility image AUC 约0.9067，但区分 rejected winner 只有约0.5036。因此“需要纠正”与“应该信任 deep 而不是 shallow”在本次人口中是不同问题；高 utility AUC 不能证明层级安全性。confidence/Delta 等对照也不能取代失败的 S_D 主门槛，没有据此组合新 score 或建议阈值。

[原始 CSV](../audit/results/rddr_phase2b110/rddr_phase2b110_score_controls.csv)

## 13. Beneficial composition

| 来源 | Count | 该 utility 内占比 | Mean q | Mean S_D | Mean Delta | Mean deep confidence |
| --- | --- | --- | --- | --- | --- | --- |
| Rejected_Deep_Win | 113,204 | 63.6460% | 0.4858 | 0.5979 | -0.1314 | 0.9301 |
| Rejected_Both_Wrong | 64,661 | 36.3540% | 0.4452 | 0.5110 | -0.1940 | 0.8455 |

有益残余以 missed Deep-Win 为主（63.6460%），但仍有36.3540%来自 Both-Wrong 的局部 margin 改善。局部 margin 增加并不要求 deep 的最终 argmax 正确。

[原始 CSV](../audit/results/rddr_phase2b110/rddr_phase2b110_beneficial_composition.csv)

## 14. Harmful composition

| 来源 | Count | 该 utility 内占比 | Mean q | Mean S_D | Mean Delta | Mean deep confidence |
| --- | --- | --- | --- | --- | --- | --- |
| Rejected_Deep_Win | 0 | 0.0000% | NA | NA | NA | NA |
| Rejected_Both_Wrong | 257,316 | 100.0000% | 0.1405 | 0.6039 | -0.1229 | 0.9544 |

有害残余全部属于 Both-Wrong；不存在的 harmful Deep-Win 子组均值为 NA，不是0。4 个 zero 也都来自 Both-Wrong，在 zero_composition.csv 保留；没有用 epsilon 重标标签。

[原始 CSV](../audit/results/rddr_phase2b110/rddr_phase2b110_harmful_composition.csv)

## 15. Delta quintiles

在 R_RW 含4个 zero 上计算20/40/60/80分位；线性 quantile，`searchsorted(side=left)` 保留 score ties、不拆 ties。Q1→Q5 为低到高。

冻结诊断切点：`-0.2204952240, -0.1354820728, -0.0810556173, -0.0380561739`。

| Q | N | 有益% | 有害% | Zero | DW% | BW% | Mean S_D | Mean q | Mean Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 87,037 | 53.6439% | 46.3538% | 2 | 25.0560% | 74.9440% | 0.3983 | 0.4146 | -0.3200 |
| 2 | 87,037 | 38.8168% | 61.1820% | 1 | 26.1498% | 73.8502% | 0.5432 | 0.2993 | -0.1738 |
| 3 | 87,037 | 35.1241% | 64.8759% | 0 | 24.8251% | 75.1749% | 0.6158 | 0.2477 | -0.1069 |
| 4 | 87,037 | 36.0169% | 63.9820% | 1 | 26.0131% | 73.9869% | 0.6705 | 0.2185 | -0.0589 |
| 5 | 87,037 | 40.7539% | 59.2461% | 0 | 28.0203% | 71.9797% | 0.7149 | 0.1980 | -0.0188 |

越靠近 Delta=0 并未出现单调增加的有益率；最负的 Q1 反而最高。这些分组仅描述结构，不生成 percentile gate、放宽 sign gate 或事后翻转分数。

[原始 CSV](../audit/results/rddr_phase2b110/rddr_phase2b110_delta_quintiles.csv)

## 16. S_D quintiles

在 R_RW 含4个 zero 上计算20/40/60/80分位；线性 quantile，`searchsorted(side=left)` 保留 score ties、不拆 ties。Q1→Q5 为低到高。

冻结诊断切点：`0.4573622823, 0.5655905962, 0.6439853668, 0.7223184466`。

| Q | N | 有益% | 有害% | Zero | DW% | BW% | Mean S_D | Mean q | Mean Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 87,037 | 43.6987% | 56.2990% | 2 | 15.7783% | 84.2217% | 0.3502 | 0.3507 | -0.2725 |
| 2 | 87,037 | 43.9652% | 56.0325% | 2 | 27.8950% | 72.1050% | 0.5157 | 0.3213 | -0.1728 |
| 3 | 87,037 | 47.7429% | 52.2571% | 0 | 35.6136% | 64.3864% | 0.6058 | 0.3118 | -0.1170 |
| 4 | 87,037 | 45.2084% | 54.7916% | 0 | 35.1586% | 64.8414% | 0.6817 | 0.2592 | -0.0756 |
| 5 | 87,037 | 23.7405% | 76.2595% | 0 | 15.6186% | 84.3814% | 0.7893 | 0.1351 | -0.0404 |

绝对 S_D 最高的 Q5 有益率反而降至23.7405%，不支持“越高越有益”的主假设。这些分组仅描述结构，不生成 percentile gate、放宽 sign gate 或事后翻转分数。

[原始 CSV](../audit/results/rddr_phase2b110/rddr_phase2b110_deep_support_quintiles.csv)

## 17. Per-class

| Group | N | 有益 / 有害 / zero | 有益率 | DW / BW | Dual-label images | Power |
| --- | --- | --- | --- | --- | --- | --- |
| class0 | 110,669 | 26,908 / 83,761 / 0 | 24.3139% | 11,243 / 99,426 | 1474 | POWERED |
| class1 | 146,849 | 55,209 / 91,637 / 3 | 37.5958% | 33,732 / 113,117 | 1999 | POWERED |
| class2 | 99,819 | 44,961 / 54,857 / 1 | 45.0425% | 27,628 / 72,191 | 818 | POWERED |
| class3 | 77,848 | 50,787 / 27,061 / 0 | 65.2387% | 40,601 / 37,247 | 341 | POWERED |

| Group | S_D Image AUC | 95% CI | Pooled AUC | AUPRC | DW% / BW% |
| --- | --- | --- | --- | --- | --- |
| class0 | 0.4943 | [0.4800, 0.5086] | 0.3787 | 0.1873 | 10.1591% / 89.8409% |
| class1 | 0.5643 | [0.5515, 0.5773] | 0.4886 | 0.3432 | 22.9705% / 77.0295% |
| class2 | 0.4096 | [0.3901, 0.4289] | 0.3567 | 0.3519 | 27.6781% / 72.3219% |
| class3 | 0.4857 | [0.4548, 0.5165] | 0.4381 | 0.5785 | 52.1542% / 47.8458% |

Power 固定为至少500正类、500负类、30张 dual-label 图像。本轮四类均 POWERED，但仅 class1 的 image AUC >0.55，未达到至少3类。此处 class3 POWERED 是 **utility 标签**的样本量结论；与旧 Phase2B1.9 稀少 Shallow-Win 的安全性人口不同，不能消除旧 UNDERPOWERED 限制。这里只保留冻结 class0–3 编码，不制定 class-specific rule。

## 18. Boundary / interior

沿用既有 boundary ≤7px / interior >7px mask，不重算边界、不调整宽度。

| Group | N | 有益 / 有害 / zero | 有益率 | DW / BW | Dual-label images | Power |
| --- | --- | --- | --- | --- | --- | --- |
| boundary | 76,452 | 24,865 / 51,586 / 1 | 32.5237% | 15,145 / 61,307 | 1819 | POWERED |
| interior | 358,733 | 153,000 / 205,730 / 3 | 42.6501% | 98,059 / 260,674 | 2836 | POWERED |

| Group | S_D Image AUC | 95% CI | Pooled AUC | AUPRC | DW% / BW% |
| --- | --- | --- | --- | --- | --- |
| boundary | 0.3834 | [0.3728, 0.3941] | 0.3277 | 0.2359 | 19.8098% / 80.1902% |
| interior | 0.5285 | [0.5192, 0.5378] | 0.4542 | 0.3727 | 27.3348% / 72.6652% |

Interior image AUC=0.5285，未达 >0.60；boundary=0.3834。Interior 的相对较好结果不足以解锁训练。

## 19. Top20 / Bottom80

复用 Phase2B1 冻结 q Top20 mask；它不是在本轮 residual 上重新划出的20%。

| Group | N | 有益 / 有害 / zero | 有益率 | DW / BW | Dual-label images | Power |
| --- | --- | --- | --- | --- | --- | --- |
| Top20 | 152,536 | 122,719 / 29,814 / 3 | 80.4525% | 82,555 / 69,981 | 2541 | POWERED |
| Bottom80 | 282,649 | 55,146 / 227,502 / 1 | 19.5104% | 30,649 / 252,000 | 2714 | POWERED |

| Group | S_D Image AUC | 95% CI | Pooled AUC | AUPRC | DW% / BW% |
| --- | --- | --- | --- | --- | --- |
| Top20 | 0.5248 | [0.5144, 0.5355] | 0.5125 | 0.7830 | 54.1216% / 45.8784% |
| Bottom80 | 0.5931 | [0.5841, 0.6020] | 0.5413 | 0.1957 | 10.8435% / 89.1565% |

Top20 的有益 prevalence 较高，但其中 S_D 排名依然弱。这些子群为诊断，不能执行 Top20-only rescue 或利用表格选择 gate。

## 20. Rejected Both-Wrong ctx_sym

`ctx_S/ctx_D` 分别在15×15有效图像内邻域、exclude self 取概率均值；`ctx_sym=0.5(ctx_S+ctx_D)`，argmax 无温度、无阈值。

| 指标 | 结果 |
| --- | --- |
| Population | 321,981 Rejected Both-Wrong |
| Accuracy | 33.7104% |
| Accuracy 95% CI | [32.4235%, 34.9965%] |
| Conditional 4-class mIoU | 17.8948% |
| Conditional 4-class mDice | 30.0397% |
| NLL (nats) | 1.2793 |
| Brier (sum over 4 classes) | 0.7504 |
| class0 / 1 / 2 / 3 IoU | 24.4015% / 20.7260% / 15.4037% / 11.0478% |

指标仅在 GT 定义的 rejected Both-Wrong 子集上计算，native28 四前景 confusion matrix，无背景覆盖修正；union=0 的类别为 NA、从 mean 排除；NLL=-log(p_GT+1e-8)，Brier 不除以4。**不能与官方全 validation/test mIoU 对比，也不是本轮重新评测模型的 segmentation 结果。**

## 21. Third-class rescue

| 事件 | Count / rate |
| --- | --- |
| Context argmax 不同于 raw 和 deep | 127,706 / 39.6626% |
| 第三类且正确 | 108,541 |
| 第三类但错误 | 19,165 |
| Rescue rate（分母全部 Rejected Both-Wrong） | 33.7104% |
| Rescue rate 95% CI | [32.4235%, 34.9965%] |
| Rescue precision（分母不同于 both） | 84.9929% |
| Rescue precision 95% CI | [83.7495%, 86.1850%] |
| Third-class harm / 全部 BW | 5.9522% |

在 Both-Wrong 中，任何正确 context 必然不同于两个错误候选，所以 accuracy 与 rescue rate 是**同一个事件**，不是两条独立证据。Context 来自原模型邻域概率，所谓第三证据是证据形式不同，不代表新训练的独立模型或统计独立样本。

## 22. Third-evidence harm control

| 人口 | N | Context accuracy [95% CI] | Third intrusion [95% CI] | 总错误率 |
| --- | --- | --- | --- | --- |
| Rejected_Deep_Win | 113,204 | 64.8758% [63.1427%, 66.5616%] | 8.8036% [8.1431%, 9.4986%] | 35.1242% |
| Rejected_Shallow_Win | 144,662 | 73.1830% [71.9912%, 74.3556%] | 5.8958% [5.4180%, 6.3910%] | 26.8170% |

对应第三类误入数为 9,966 / 8,529，全部错误，因为 one-correct 人口已经有一个候选正确。此处 third intrusion=third harm，但不是全部 context 错误：context 选错原有候选同样可能造成伤害。**不能把8.80%/5.90%当作总错误率。**

Both-Wrong 子群由 GT 定义，部署时无法直接获得；没有证明如何 GT-blind 地识别该群并避开上述污染。因此本轮不把 ctx_sym 写入 loss、gate 或实际预测。

[原始 CSV](../audit/results/rddr_phase2b110/rddr_phase2b110_third_evidence_harm_control.csv)

## 23. Bootstrap

10000次 paired image-level bootstrap，seed42，每次从全部3418图像有放回抽样3418张，所有 endpoint 使用同一批抽样索引。无 pixel bootstrap。每个 ratio 对抽到的图像重新汇总分子/分母；image-balanced AUC 只对抽中的 dual-label eligible 图像做等权均值。区间为2.5/97.5百分位；本次所有 endpoint 均有10000个有限 replicate。

Gate A 每次计算 `sum(residual beneficial)/sum(ALL Raw-Wrong) * fixed original N_RW`；将 count-equivalent 下界与**固定**原始 gap 比较，不使用 residual prevalence 分母，也不把 bootstrap 样本量变化误称为新增机会。

抽样索引 SHA256：`98e6164a3524dde42fc993cac0b5665076f7ebac7f6a73b7420d20c81022d00b`。独立 direct-gather 与主流程权重矩阵聚合的 replicate 最大误差 `6.439e-15`，区间误差 `2.665e-15`。

| Endpoint（rate用0–1，count-equivalent用位置数） | Estimate | 95% CI |
| --- | --- | --- |
| ResidualBeneficial_prevalence | 0.408711 | [0.399205, 0.418468] |
| ResidualBeneficial_binary_prevalence | 0.408715 | [0.399208, 0.418470] |
| CoverageHeadroom_rate | 0.251077 | [0.244339, 0.257795] |
| ResidualBeneficial_count_equivalent | 177865.000000 | [173091.478863, 182623.725906] |
| residual_utility:all:S_D_sym:image_AUROC | 0.500179 | [0.491341, 0.509048] |
| residual_utility:all:Delta_sym:image_AUROC | 0.485256 | [0.477556, 0.493153] |
| residual_utility:all:q:image_AUROC | 0.906700 | [0.902952, 0.910286] |
| residual_utility:all:deep_confidence_advantage:image_AUROC | 0.665255 | [0.658624, 0.671687] |
| residual_utility:all:deep_entropy_advantage:image_AUROC | 0.633340 | [0.626507, 0.640010] |
| residual_utility:Top20:S_D_sym:image_AUROC | 0.524752 | [0.514398, 0.535470] |
| residual_utility:Bottom80:S_D_sym:image_AUROC | 0.593109 | [0.584051, 0.601997] |
| residual_utility:class0:S_D_sym:image_AUROC | 0.494286 | [0.480041, 0.508570] |
| residual_utility:class1:S_D_sym:image_AUROC | 0.564337 | [0.551533, 0.577321] |
| residual_utility:class2:S_D_sym:image_AUROC | 0.409587 | [0.390058, 0.428942] |
| residual_utility:class3:S_D_sym:image_AUROC | 0.485686 | [0.454772, 0.516457] |
| residual_utility:boundary:S_D_sym:image_AUROC | 0.383379 | [0.372801, 0.394135] |
| residual_utility:interior:S_D_sym:image_AUROC | 0.528499 | [0.519209, 0.537768] |
| rejected_winner:all:S_D_sym:image_AUROC | 0.608274 | [0.599321, 0.617396] |
| rejected_winner:all:Delta_sym:image_AUROC | 0.664510 | [0.655762, 0.673546] |
| rejected_winner:all:q:image_AUROC | 0.503636 | [0.495663, 0.511556] |
| rejected_winner:all:deep_confidence_advantage:image_AUROC | 0.654164 | [0.646730, 0.661496] |
| rejected_winner:all:deep_entropy_advantage:image_AUROC | 0.641551 | [0.634006, 0.649025] |
| rejected_winner:Top20:S_D_sym:image_AUROC | 0.674732 | [0.663750, 0.685660] |
| rejected_winner:Bottom80:S_D_sym:image_AUROC | 0.561724 | [0.552662, 0.570864] |
| rejected_winner:class0:S_D_sym:image_AUROC | 0.632252 | [0.616769, 0.647559] |
| rejected_winner:class1:S_D_sym:image_AUROC | 0.721249 | [0.709260, 0.732709] |
| rejected_winner:class2:S_D_sym:image_AUROC | 0.348128 | [0.326769, 0.369414] |
| rejected_winner:class3:S_D_sym:image_AUROC | 0.304143 | [0.244700, 0.367115] |
| rejected_winner:boundary:S_D_sym:image_AUROC | 0.542580 | [0.530750, 0.554556] |
| rejected_winner:interior:S_D_sym:image_AUROC | 0.618049 | [0.608473, 0.627931] |
| ctx_sym_rejected_BothWrong_accuracy | 0.337104 | [0.324235, 0.349965] |
| ThirdClassRescueRate | 0.337104 | [0.324235, 0.349965] |
| ThirdClassRescuePrecision | 0.849929 | [0.837495, 0.861850] |
| Rejected_Deep_Win:ctx_accuracy | 0.648758 | [0.631427, 0.665616] |
| Rejected_Deep_Win:third_intrusion | 0.088036 | [0.081431, 0.094986] |
| Rejected_Shallow_Win:ctx_accuracy | 0.731830 | [0.719912, 0.743556] |
| Rejected_Shallow_Win:third_intrusion | 0.058958 | [0.054180, 0.063910] |

全量原始 replicates 与 summary CSV 已保存。区间反映固定 seed42/checkpoint 下的图像抽样不确定性，不是训练 seed 方差，也不覆盖后续反复选择方案产生的适应性偏差；对照区间是诊断、未做多重比较校正。

## 24. Gate A / B / C / D

| Gate | 冻结要求 | 实测 | 判定 |
| --- | --- | --- | --- |
| A | beneficial count ≥ gap，且 count-equivalent CI lower ≥ gap | 177,865 ≥ 31,266；lower=173091.48 | PASS |
| B | utility image AUC ≥0.65 且 lower>0.50 | 0.5002 [0.4913, 0.5090] | FAIL |
| C | winner image AUC ≥0.65 且 lower>0.50 | 0.6083 [0.5993, 0.6174] | FAIL |
| D | interior>0.60 且≥3个 powered class>0.55 | interior=0.5285；1/4类通过（4/4 powered） | FAIL |

D 没有把 UNDERPOWERED 自动当 FAIL：本轮四类都有 power，而观测到的 interior 和类数均不达标，故确为 FAIL。主门槛不允许任何 control 替代 S_D。

## 25. RESIDUAL_THIRD_EVIDENCE_SIGNAL

`TRUE`。Rejected Both-Wrong ctx accuracy=33.7104%≥25%，CI lower=32.4235%>20%，rescue rate=33.7104%≥20%。三项按合同满足；accuracy 与 rescue 重复事件的限制如第21节。这是 secondary route diagnosis，不是 GO gate。

## 26. STRONG_RESIDUAL_DEEP_RECOVERY_SIGNAL

`FALSE`。B/C/D 未通过，且两个 S_D 主 image AUC 分别0.5002、0.6083，均低于0.75。不能用 q 的0.9067或某个子群值宣称 Strong signal。

## 27. Route attribution

| 路径 | 支持与限制 | 本轮判断 |
| --- | --- | --- |
| A — missed Deep-Win | 113,204个机会；需理论恢复27.6192%，但 S_D 的utility/winner/cross-stratum门槛失败 | 数量支持，预注册恢复排名不支持 |
| B — Both-Wrong context | 321,981个位置；ctx正确108,541个，条件救回33.7104%；one-correct污染尚未解决 | 仅支持另行第三证据审计 |

Residual Raw-Wrong 的多数人口是 Both-Wrong（约74%），但 beneficial 子集的多数仍是 missed Deep-Win（约64%）。“误差人口构成”与“有益机会构成”不可混为一谈。现有数据也不能证明所有冻结证据都无用：否定的是预注册 S_D 方案在本合同下的能力，不是对一切未来方案作不可能性证明。

## 28. Scientific interpretation / engineering delivery

本轮回答：残余覆盖率不足**不是没有潜在有益位置**，而是预注册的绝对 deep support 尚不能可靠筛选它们；对大部分 neither-hierarchy-correct 的残余，邻域 context 有条件性的额外信息。

这仍不足以安全修复 Gate E：有益标签是局部 logit-margin 方向导数，不等于有限更新的纠错；第三证据救回是在 GT 划分的人口上观察到的，尚缺 GT-blind 人口辨识与 one-correct 保护证据。没有得出“CH必然可以安全救回”或“直接用q训练”的结论。

工程交付包含独立 tools、44项测试、26项独立复核、完整 CSV/JSON、10000 bootstrap replicates、输入SHA与运行记录、可复跑 README。原 `network/`、`tool/`、`train_sshr.py` 与 A0 完全一致。本轮使用既有缓存，不生成新的大特征文件或 checkpoint，不删除旧实验。

[运行说明](README_rddr_phase2b110.md)；[交付摘要](rddr_phase2b110_delivery_summary.md)；[冻结合同](rddr_phase2b110_contract.md)。大输入缓存和 image_statistics NPZ 留服务器，CSV/JSON/报告进入独立分支供 PR 审核。没有新增训练、lambda/threshold搜索、恢复 gate 或新 loss。

## 29. Exact final decision

按已批准的 decision precedence：A PASS，B/C/D 非 PASS，第三证据信号 TRUE，故为第三证据路线支持。只允许在用户下一次确认后另立独立、预注册的 third-evidence rescue audit；**本轮到此停止，不自动设计机制，不训练，不做 Full25，不评 test/LUAD/其它seed。**

DECISION = RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED
