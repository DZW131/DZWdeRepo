# RDDR Phase-2B0 Reliable Relation Feasibility Audit

**最终判定：`RDDR_PHASE2B0_NOGO`。** 本轮是 C0 冻结权重、BCSS validation-only、零训练、零新增参数的关系审计。
Gate A/B/C/D：{'A': False, 'B': False, 'C': True, 'D': True}。AUROC、purity、accuracy、IoU、Dice 用0–1比例，相关差值乘100为pp；Mass/N_eff及人数是计数量，不是比例。

Primary image AUROC=0.6224（门槛0.65）；purity增益=1.1676pp（门槛3pp）。无训练邻居估计的mIoU虽提高7.8379pp，但不等于官方final-CAM收益，也不能补偿A/B失败。

## 1. Provenance / frozen assets

- Pure A0: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Extraction commit: `80fab5a1fe9cb970da5b4e7f02af5a26b1c20237`
- Analysis commit: `44af58a5c47b74c4cbec9a009f86077117d3ffb9`
- Checkpoint: `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Statistics SHA256: `f66b0717a4a0bddeb9bb84699b98f74f8eb5cd52ab88e6ecd0a86dc8a10e04d3`
- 只新增 tools/tests/docs/audit；官方网络、预处理、训练和推理源文件保持 A0 原样。

## 2. Exact commands / environment

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b0-10d4c6f
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b0_relation_audit.py --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --population-cache /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/diagnostics/frozen_phase0_populations --phase0-results /home/duyanhong/experiments/RDDR_PHASE0_586f402/formal --output /home/duyanhong/experiments/RDDR_PHASE2B0/formal_complete_r1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/summarize_rddr_phase2b0.py --input /home/duyanhong/experiments/RDDR_PHASE2B0/formal_complete_r1 --output /home/duyanhong/experiments/RDDR_PHASE2B0/report_final
```

Python 3.10.20; torch 2.11.0+cu128; NumPy 1.23.5; NVIDIA GeForce RTX 5090 D v2。batch1, BF16 forward, FP32 probability/relation；无TTA。benchmark=False, matmul=none, conv=tf32。

## 3. Tensor / forward contract

实际运行未经修改的 `Net.forward`，用只读 hook 取得 HFRM28_1 输入/输出。F28_raw/F28_rect=[1,512,28,28]，Ddeep=[1,4096,28,28]。Ls=ic1(F28_raw)，Ld=fc8(Ddeep)，未做 ReLU/CAM normalize。dropout 在 eval 中关闭。使用 softmax(logits.float())，温度1。所有参数 requires_grad=False，无 optimizer，无 backward。

## 4. GT / frozen population projection

224×224 GT/历史人口 mask nearest 到28×28。类别0–3计入指标，4/255只从指标中排除。预测/关系权重不读取GT，也不依GT删除背景来源。历史 mask 已逐文件 SHA 校验并逐图对照 Phase0 CSV 人数。历史原始逐像素文件未保留；本次复用的是 Phase2A 在原代码/权重/backend 下重建且逐图人数 exact 的 immutable cache，不能声称曾与不存在的历史像素哈希比较。

| Group | 224-grid count | 28-grid count |
|---|---|---|
| all | 158639345 | 2479143 |
| Corrected_by_CH | 19934592 | 305318 |
| Still_Wrong | 24262754 | 378130 |
| Harmed_by_CH | 4443224 | 69617 |
| Stable_Correct | 109998775 | 1726078 |
| Top20 | 31727873 | 485451 |
| Bottom80 | 126911472 | 1993692 |

本轮重新提取 q 与缓存原始算术最大误差=0.0；Torch/NumPy ln2 除法舍入差异最大=5.96046448e-08，单独记录不用于重分组。

## 5. Neighborhood / eligibility contract

15×15、radius7，只保留图内邻居，排除self。角落63、中部224个邻居。Propagation使用全部合法来源；pair/purity只评价FG–FG。实际 Mass/N_eff 分母绝不GT过滤。无前景来源时 purity 未定义，而非设0或完美；无同类来源时oracle未定义且不回退。

## 6. Frozen U / SR / SC / SRSC formulas

```text
q_j=clip(JS(ps_j,pd_j)/ln2,0,1); r_j=1-q_j
c_ij=clip(1-JS(pd_i,ps_j)/ln2,0,1)
U=1; SR=r_j; SC=c_ij; SRSC=r_j*c_ij
p_tilde_i=sum_j(A_ij*ps_j)/(sum_j A_ij+1e-8)
```

自然对数、epsilon=1e-8，epsilon位置与Phase0一致。无receiver reliability、learned层、其他score或搜索。Primary始终SRSC vs U，SR/SC只解释。U是均匀语义概率聚合对照，**不等于训练后CH卷积核的特征输出**。

## 7. Counts / streaming / resources

3418图；FG target=2,479,143；FG–FG directed pair=405,023,844；FG target的实际空间边=416,469,801；无FG邻居target=5。

提取全程 85.3s；forward+relations 12.7s；统计 63.8s；bootstrap/汇总 3.9s。CUDA peak allocated=0.559GiB，reserved=0.631GiB。仅逐图pair临时张量、累计直方图和逐图充分统计量，无全数据集pair缓存、无新模型checkpoint。

## 8. Pair AUROC / AUPRC

| Relation | Image AUROC | Pooled AUROC | AP | Prevalence | AP/prevalence | Eligible images |
|---|---|---|---|---|---|---|
| U | 0.500000 | 0.500000 | 0.890503 | 0.890503 | 1.000000 | 2103 |
| SR | 0.528971 | 0.564314 | 0.910146 | 0.890503 | 1.022057 | 2103 |
| SC | 0.624640 | 0.665475 | 0.935072 | 0.890503 | 1.050049 | 2103 |
| SRSC | 0.622374 | 0.663412 | 0.936164 | 0.890503 | 1.051275 | 2103 |

AUPRC按非插值AP；同分数计tie。4096固定bin，16张等间隔确定性图像全部四配置对照exact排序：最大AUROC误差=0.00168514，最大AP误差=0.00267400。SRSC自己的最大AUROC误差=0.00020229。这是预选subset的误差实测，不是对未校验图像的数学误差上界；没有据此修改bin数或score。详见 histogram_validation.csv。

## 9. Image-balanced AUROC + CI

| AUROC | 95% low | 95% high | Eligible images |
|---|---|---|---|
| 0.622374 | 0.617012 | 0.627688 | 2103 |

缺少正/负pair的图像AUROC为NA，未人为赋0.5；bootstrap对图像采样后忽略该指标NA。

## 10. Weighted neighbor purity

| Group | Relation | Image purity | Pixel purity | Gain vs U | Image N_eff | Same mass | Wrong mass |
|---|---|---|---|---|---|---|---|
| all | U | 0.889752 | 0.890345 | 0.000000 | 167.331017 | 145.483769 | 17.888753 |
| all | SR | 0.890178 | 0.890781 | 0.000426 | 158.926888 | 118.327985 | 13.902702 |
| all | SC | 0.900154 | 0.900747 | 0.010401 | 153.168130 | 110.472228 | 10.810716 |
| all | SRSC | 0.901429 | 0.902023 | 0.011676 | 143.152719 | 93.639039 | 8.526559 |

FG–FG purity完整p25/median/p75见 purity.csv；其百分位采用4096bin，绝对量化误差<=0.5/4096。主image-balanced增益=+1.1676pp，95%CI=[+1.1074,+1.2286]pp。

## 11. Effective neighbors / mass

| Group | Mass mean | Mass p05 | Mass p50 | Mass p95 | N_eff mean | N_eff p05 | N_eff p50 | N_eff p95 |
|---|---|---|---|---|---|---|---|---|
| all | 104.556643 | 33.605469 | 102.894531 | 178.855469 | 143.987904 | 69.808594 | 142.488281 | 214.621094 |
| Corrected_by_CH | 84.860313 | 23.980469 | 81.949219 | 156.542969 | 139.258493 | 66.035156 | 138.441406 | 209.753906 |
| Harmed_by_CH | 85.305173 | 27.480469 | 83.972656 | 147.136719 | 140.620372 | 66.527344 | 142.050781 | 208.714844 |
| Top20 | 70.900203 | 17.636719 | 67.019531 | 139.644531 | 127.027184 | 58.816406 | 124.605469 | 198.378906 |
| Bottom80 | 112.751767 | 48.207031 | 110.714844 | 182.300781 | 148.117729 | 74.183594 | 146.863281 | 215.878906 |

此表包括GT背景来源的实际传播图；FG same/wrong mass另报，不可混用分母。分位数量化误差<=224/(2×4096)=0.027344。

## 12. Corrected / Harmed mechanism

| Group | Relation | Image purity | Pixel purity | Gain vs U | Image N_eff | Same mass | Wrong mass |
|---|---|---|---|---|---|---|---|
| Corrected_by_CH | U | 0.903727 | 0.917359 | 0.000000 | 169.244386 | 155.968885 | 14.258783 |
| Corrected_by_CH | SRSC | 0.911227 | 0.924900 | 0.007500 | 140.248919 | 76.480896 | 6.162518 |
| Harmed_by_CH | U | 0.603875 | 0.642082 | 0.000000 | 165.565742 | 107.762012 | 60.318169 |
| Harmed_by_CH | SRSC | 0.610118 | 0.647149 | 0.006244 | 129.818796 | 51.727223 | 29.593090 |

主paired差只用同时包含两组的 2228 张图。Corrected-Harmed=0.260664，95%CI=[0.249853,0.271912]。上表各组全体image mean与这个配对样本总体不同，不可直接相减替代主检验。历史by-CH命名实际为 raw→完整HFRM（含GSR）；这里是关联证据，不能证明CH单独因果。

| Corrected-positive score | Pooled AUC | Image AUC | 95% low | 95% high | Images |
|---|---|---|---|---|---|
| purity | 0.810673 | 0.766017 | 0.757041 | 0.775215 | 2228 |
| purity_gain | 0.529234 | 0.536126 | 0.525674 | 0.546469 | 2228 |
| negative_wrong_mass | 0.800801 | 0.740003 | 0.731273 | 0.748935 | 2229 |

wrong mass在预注册时即取负号；gain线性映射到[0,1]只为固定hist，并未改变排序。

## 13. Frozen Top20 / Bottom80

| Group | Relation | Image purity | Pixel purity | Gain vs U | Image N_eff | Same mass | Wrong mass |
|---|---|---|---|---|---|---|---|
| Top20 | U | 0.872224 | 0.860262 | 0.000000 | 164.571837 | 139.451700 | 22.431906 |
| Top20 | SR | 0.868023 | 0.855281 | -0.004201 | 153.680850 | 97.626652 | 16.913386 |
| Top20 | SC | 0.877015 | 0.866006 | 0.004791 | 142.416405 | 79.195397 | 12.122571 |
| Top20 | SRSC | 0.874433 | 0.863125 | 0.002209 | 131.516458 | 59.449749 | 9.390777 |
| Bottom80 | U | 0.891429 | 0.897670 | 0.000000 | 167.304914 | 146.952539 | 16.782525 |
| Bottom80 | SR | 0.893323 | 0.899426 | 0.001894 | 159.627945 | 123.368624 | 13.169620 |
| Bottom80 | SC | 0.903326 | 0.909207 | 0.011897 | 155.143974 | 118.087932 | 10.491288 |
| Bottom80 | SRSC | 0.905801 | 0.911494 | 0.014372 | 145.506857 | 101.963908 | 8.316128 |

Top20来自历史224-grid阈值mask，nearest后不强制正好20%，没有按本轮score重新选择。

## 14. Conflict quintiles

固定native q_feature边界：[0.020935675129294395, 0.072734534740448, 0.163648784160614, 0.3369627296924591]。method=higher，tie归较低分位。

| Group | Relation | Image purity | Pixel purity | Gain vs U | Image N_eff | Same mass | Wrong mass |
|---|---|---|---|---|---|---|---|
| Q1 | U | 0.896677 | 0.921979 | 0.000000 | 142.516820 | 132.704312 | 10.992879 |
| Q1 | SR | 0.901239 | 0.925615 | 0.004562 | 136.240203 | 116.768757 | 8.617257 |
| Q1 | SC | 0.914067 | 0.936414 | 0.017391 | 131.467600 | 113.459352 | 6.434518 |
| Q1 | SRSC | 0.919156 | 0.940273 | 0.022479 | 122.980662 | 102.536648 | 5.124795 |
| Q2 | U | 0.893611 | 0.906021 | 0.000000 | 167.025131 | 149.991368 | 15.332311 |
| Q2 | SR | 0.896469 | 0.908542 | 0.002858 | 159.602465 | 127.970869 | 12.050497 |
| Q2 | SC | 0.907284 | 0.918655 | 0.013673 | 155.547018 | 124.585995 | 9.672323 |
| Q2 | SRSC | 0.910705 | 0.921615 | 0.017093 | 146.058357 | 108.936019 | 7.680146 |
| Q3 | U | 0.888064 | 0.890634 | 0.000000 | 176.033550 | 153.794919 | 18.695008 |
| Q3 | SR | 0.889382 | 0.891892 | 0.001318 | 167.975597 | 127.610354 | 14.695227 |
| Q3 | SC | 0.899161 | 0.901614 | 0.011097 | 164.032259 | 122.955483 | 12.019008 |
| Q3 | SRSC | 0.901152 | 0.903511 | 0.013088 | 154.089351 | 104.578650 | 9.515560 |
| Q4 | U | 0.880773 | 0.871121 | 0.000000 | 177.596756 | 151.092988 | 22.247576 |
| Q4 | SR | 0.880231 | 0.870559 | -0.000542 | 168.780474 | 120.456848 | 17.388541 |
| Q4 | SC | 0.889313 | 0.880093 | 0.008540 | 163.212110 | 111.592767 | 13.895580 |
| Q4 | SRSC | 0.889677 | 0.880489 | 0.008904 | 152.695086 | 91.705224 | 10.957836 |
| Q5 | U | 0.874126 | 0.861971 | 0.000000 | 165.019555 | 139.835294 | 22.176017 |
| Q5 | SR | 0.870177 | 0.857299 | -0.003949 | 154.322290 | 98.833084 | 16.762007 |
| Q5 | SC | 0.878050 | 0.866961 | 0.003925 | 143.159174 | 79.767506 | 12.032166 |
| Q5 | SRSC | 0.875588 | 0.864226 | 0.001463 | 132.411690 | 60.438597 | 9.354470 |

## 15. Boundary / interior

| Group | Image AUROC | Pooled AUROC | Purity | Gain | N_eff |
|---|---|---|---|---|---|
| boundary | 0.533284 | 0.538542 | 0.554417 | 0.012260 | 126.742287 |
| interior | 0.643922 | 0.683791 | 0.926165 | 0.012855 | 143.371145 |

边界仅由fullres FG–FG transition的欧氏距离<=7px构建，再投影；不进入score。

## 16. Per-class relation audit

| Class | Same prevalence | AUROC | AP | U purity | SRSC purity | Gain vs U | N_eff |
|---|---|---|---|---|---|---|---|
| class0 | 0.914653 | 0.724397 | 0.961498 | 0.785934 | 0.808287 | 0.022353 | 138.672621 |
| class1 | 0.880770 | 0.674992 | 0.930222 | 0.683315 | 0.706417 | 0.023103 | 133.966320 |
| class2 | 0.843673 | 0.565228 | 0.870610 | 0.652344 | 0.679774 | 0.027430 | 119.837138 |
| class3 | 0.900946 | 0.440853 | 0.871412 | 0.704181 | 0.722899 | 0.018718 | 114.983362 |

四类完整U/SR/SC/SRSC数据均保留，任何单类收益不替代主门槛。

## 17. Training-free neighbor estimator

| Estimator | Accuracy | mIoU | Dice | NLL | Brier | Coverage |
|---|---|---|---|---|---|---|
| U | 0.781395 | 0.489692 | 0.614260 | 0.643973 | 0.336364 | 1.000000 |
| SR | 0.791591 | 0.522219 | 0.654638 | 0.610669 | 0.316704 | 1.000000 |
| SC | 0.793659 | 0.542697 | 0.679856 | 0.600515 | 0.309462 | 1.000000 |
| SRSC | 0.799535 | 0.568071 | 0.709404 | 0.579546 | 0.298289 | 1.000000 |
| raw | 0.714253 | 0.436349 | 0.581730 | 0.784837 | 0.406249 | 1.000000 |
| deep | 0.767429 | 0.566286 | 0.717333 | 1.784933 | 0.409590 | 1.000000 |
| oracle | 0.823474 | 0.544555 | 0.657641 | 0.587205 | 0.296441 | 0.999952 |

原生28-grid四类指标，不能与224-grid/TTA final CAM mIoU直接比较。没有background prediction overwrite；zero-union类别从macro mean排除并保留NA，不设1。Brier是四类平方误差求和再target均值，NLL=-log(p_GT+eps)。raw/deep只是参照，没有参加protocol选择。

## 18. Frozen Top20 repair / harm

| Estimator | Repair n | Harm n | Targets | Repair | Harm | NetRepair |
|---|---|---|---|---|---|---|
| U | 154226 | 50261 | 485451 | 0.317696 | 0.103535 | 0.214162 |
| SR | 168016 | 50938 | 485451 | 0.346103 | 0.104929 | 0.241174 |
| SC | 178431 | 58573 | 485451 | 0.367557 | 0.120657 | 0.246900 |
| SRSC | 190546 | 59675 | 485451 | 0.392513 | 0.122927 | 0.269586 |
| raw | 0 | 0 | 485451 | 0.000000 | 0.000000 | 0.000000 |
| deep | 220561 | 103341 | 485451 | 0.454342 | 0.212876 | 0.241466 |
| oracle | 170605 | 35207 | 485410 | 0.351466 | 0.072530 | 0.278935 |

基准raw=原生28-grid argmax(ps)，不是历史upsampled raw。repair/harm均除以全部eligible Top20。

## 19. Deep-Correct / Deep-Wrong

| Stratum | Relation | Purity | Gain | Neighbor acc | Stratum NetRepair | Top20 NetRepair |
|---|---|---|---|---|---|---|
| Deep_Correct | U | 0.901271 | 0.000000 | 0.878835 | 0.044259 | 0.342390 |
| Deep_Correct | SRSC | 0.924713 | 0.023442 | 0.950839 | 0.116263 | 0.558374 |
| Deep_Wrong | U | 0.823382 | 0.000000 | 0.459863 | 0.142647 | 0.025528 |
| Deep_Wrong | SRSC | 0.802706 | -0.020676 | 0.300266 | -0.016950 | -0.155243 |
| Top20_Deep_Correct | U | 0.883901 | 0.000000 | 0.579198 | 0.342390 | 0.342390 |
| Top20_Deep_Correct | SRSC | 0.909097 | 0.025196 | 0.795182 | 0.558374 | 0.558374 |
| Top20_Deep_Wrong | U | 0.814231 | 0.000000 | 0.551562 | 0.025528 | 0.025528 |
| Top20_Deep_Wrong | SRSC | 0.787155 | -0.027077 | 0.370791 | -0.155243 | -0.155243 |

Deep-Wrong邻居准确率变化=-15.9597pp；其Top20净修复由+2.5528%变为-15.5243%。这不是只有潜在风险：本次实测在deep错误子集发生明显退化。**deep-anchored relation is conditional, not universally safe**。

## 20. Deep-hypothesis echo

| Group | Echo fraction | Non-echo n | SRSC right / deep wrong | SRSC wrong / deep right |
|---|---|---|---|---|
| all | 0.873631 | 313287 | 173126 | 93532 |
| Top20 | 0.675767 | 157399 | 72843 | 59192 |
| Deep_Correct | 0.950839 | 93532 | 0 | 93532 |
| Deep_Wrong | 0.618862 | 219755 | 173126 | 0 |

非echo总量还包括双方均错但预测类别不同的情况，所以末两列不一定加和为非echo总量。

## 21. Oracle diagnostic upper bound

| Estimator | Accuracy | mIoU | Dice | NLL | Brier | Coverage |
|---|---|---|---|---|---|---|
| U | 0.781395 | 0.489692 | 0.614260 | 0.643973 | 0.336364 | 1.000000 |
| SRSC | 0.799535 | 0.568071 | 0.709404 | 0.579546 | 0.298289 | 1.000000 |
| oracle | 0.823474 | 0.544555 | 0.657641 | 0.587205 | 0.296441 | 0.999952 |

Oracle有效targets=2,479,023；实测purity=1.000000000；平均同类邻居数=145.490812。
Oracle仅同类GT邻居；有邻居时purity约1，零同类邻居为未定义，coverage显式报告；没有raw/deep fallback。它是GT关系选择的诊断参照，并非所有预测指标的数学上界（同类source的shallow语义仍可能错）。无oracle调参。

## 22. Paired image bootstrap

| Metric | Observed | 95% low | 95% high | Images |
|---|---|---|---|---|
| SRSC_image_balanced_pair_AUROC | 0.622374 | 0.617012 | 0.627688 | 2103 |
| SRSC_minus_U_purity | 0.011676 | 0.011074 | 0.012286 | 3416 |
| SRSC_Corrected_minus_Harmed_purity | 0.260664 | 0.249853 | 0.271912 | 2228 |
| Harmed_SRSC_minus_U_purity | 0.006244 | 0.003443 | 0.009041 | 2247 |
| SRSC_mean_N_eff | 143.152719 | 142.698204 | 143.610675 | 3416 |
| target_AUROC_purity | 0.766017 | 0.757041 | 0.775215 | 2228 |
| target_AUROC_purity_gain | 0.536126 | 0.525674 | 0.546469 | 2228 |
| target_AUROC_negative_wrong_mass | 0.740003 | 0.731273 | 0.748935 | 2229 |
| SRSC_minus_U_neighbor_accuracy | 0.018140 | 0.014827 | 0.021501 | 3416 |
| SRSC_minus_U_neighbor_mIoU | 0.078379 | 0.070776 | 0.086080 | 3416 |
| Top20_SRSC_minus_U_NetRepair | 0.055425 | 0.044943 | 0.065939 | 3405 |

10000次、seed42、相同图像索引，同一image内所有target/pair保持一起。mIoU重加4×4confusion后重算，非逐图mIoU平均；Top20按sample总repair-harm/sample总Top20。未做pair-level naive bootstrap。完整replicates CSV和充分统计NPZ保留，可独立重算。

## 23. Preregistered gates

| Gate | Result |
|---|---|
| A | FAIL |
| B | FAIL |
| C | PASS |
| D | PASS |

A: image AUROC>=.65且CI下界>.5；B: image purity增益>=.03且CI下界>0、meanN_eff>=5；C: paired Corrected-Harmed>0且CI下界>0、Harmed gain>0；D: neighbor accuracy/mIoU均增、至少一项CI下界>0、Top20净修复增。

## 24. Scientific interpretation / delivery

至少一个关系判别/纯度主门槛失败，当前r×c formulation未达到预注册可行性条件，应停止当前formulation。

观察与推断分开：整体邻居估计accuracy提高1.8140pp、mIoU提高7.8379pp，说明有聚合utility，但Q5的image purity增益仅0.1463pp。SR单独的pair区分力很弱，SC与SRSC接近；这提示‘source shallow/deep一致’不足以保证source正确，属于机制解释而非新公式证据。整体echo约87%而非接近100%，不能称为完全复制deep；同时Deep-Wrong显著损害说明其依赖deep hypothesis。上述utility与风险均不能把预注册NOGO改写为GO。

SR/SC超过primary、某子集改善、oracle headroom均不触发替换primary或posthoc调参。与先前feature/context suppression失败形成的证据链限于：冲突存在≠必须少用context；本轮检验source selection，而非新模型有效性。本轮遵循研究实施交付流程，保留冻结合同、可运行命令、测试、逐图/汇总CSV、JSON、原始充分统计和独立PR；未训练、未访问test/LUAD、未新增权重、未自动merge。此前各实验不删除不覆盖。

复核命令：`python -m unittest discover -s tests -p test_rddr_phase2b0.py -v`；`python tools/verify_rddr_phase2b0_delivery.py --report <report-directory>`。独立核验不import关系实现，读取逐图CSV重新计算confusion、32个bootstrap replicate及全部10000次CI分位点，输出 `rddr_phase2b0_independent_verification.json`；服务器测试日志随交付保留。

## 25. Exact decision / stop

完成报告后停止。即使GO，也不自动启动Phase-2B训练。

DECISION = RDDR_PHASE2B0_NOGO
