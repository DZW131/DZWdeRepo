# PDSR‑VPCA‑HQMR v1 Phase0 Final Report

> **FINAL_DECISION = FULL_MODEL_NOGO**
> HQMR P0 = 65.5727%
> VLM‑Last P1 = 65.5726%
> Static‑PDSR P2 = 65.5731%
> VPCA‑PDSR P3 = 65.5725%
> SSHR = 66.6967%
> P1−P0 = -0.000175 pp; P2−P1 = +0.000519 pp; P3−P2 = -0.000626 pp; P3−P0 = -0.000282 pp
> PDSR_DECISION = NOGO; VPCA_DECISION = NOGO
> Hard‑M1 P0 = 0.00%; P1 = 0.0470%; P2 = 0.0452%; P3 = 0.0435%
> PDSR Hard‑M1 gain = -0.001857 pp; VPCA Hard‑M1 gain = -0.001630 pp
> NCE = 0.976
> CLASS_DAMAGE = False; LAYER_COLLAPSE = False; CONCEPT_COLLAPSE = True; CCRA_COLLAPSE = False; VLM_IGNORED = True
> FULL25_GO = False

## 1. Executive Decision

最终判定 **FULL_MODEL_NOGO**。P3 相对 P0 -0.000282 pp，没有可解释的分割收益；PDSR=NOGO，VPCA=NOGO。P3 γ abs mean=0.000475069<0.001，4 类概念均出现 top‑1 固定，NCE=0.976<1.5。按预注册要求停止，不运行 Full25。

## 2. Frozen Research Context

RACC、gate rescue、dynamic alpha、prototype relabel 与 H5/H4/H3 consensus 路线保持关闭。本轮只检验独立 PLIP 语义源及其 PDSR/VPCA 使用方式。

## 3. Reproduction Gate

冻结 HQMR checkpoint 与 UMRF baseline prediction bank 的原始复现门槛通过；原先封存的 HQMR mIoU=65.5724%，本轮在同一 UMRF prediction bank 与当前 validation masks 上重新汇总 P0=65.5727%，差异 +0.000294 pp。因此本轮 P0 **没有达到与历史标量完全一致的严格数值复现**，这是报告限制；P1/P2/P3 全部与同一个当前 P0 bank 配对比较，不会把该微小差异伪装成模型收益。Checkpoint SHA‑256 `84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb`，三视图 TTA、最终 gate 与 CAM normalization 未更改。

## 4. PLIP Runtime Audit

官方 `vinid/plip@67ade53` 权重 `98a7f8d2a1f4a8fc8f6dedb3a16ff7efbe02a7ef67c93904c80bca9767c69630`；hidden states=13，block 4/8/12 dense token 均为 `[1, 49, 768]`，patch grid=7×7，视觉 hidden=768，text dim=512。

## 5. Class Mapping

BCSS index 固定为 C0 Tumor、C1 Stroma、C2 Lymphocytic/Inflammatory infiltrate、C3 Necrosis；C4 是 evaluation ignore/other。

## 6. Concept Bank

每类 8 个、共 32 个 atomic pathology concepts；bank SHA `5ed9a01e930fe7326a7c36ff572eab139a7ac568f1a21fb13a7901bcf19c48d3`，embedding SHA `1f5a01dd0fa6891b5c9a17a090f3c79dde88e9be327d2314f38d244a07924b34`，全部 L2 normalized，且在读取 validation GT 前冻结。

## 7. Leakage Audit

训练 batch 仅包含 `['image_id', 'raw_rgb', 'image_level_label']`，segmentation GT in train=False，E1–E4 validation access=False。

## 8. Architecture Implementation

P1 使用 block12 dense residual；P2 使用 block4/8/12 static-concept reconstruction；P3 仅将 static prior 替换为 VPCA soft q×rho。旧 HQMR 和 PLIP 参数均冻结。

## 9. Identity Tests

P1/P2/P3 的 γ=0 identity 均通过：{"P1": {"same_argmax": true, "max_abs_drift": 0.0, "gamma_abs_max": 0.0, "pass": true}, "P2": {"same_argmax": true, "max_abs_drift": 0.0, "gamma_abs_max": 0.0, "pass": true}, "P3": {"same_argmax": true, "max_abs_drift": 0.0, "gamma_abs_max": 0.0, "pass": true}}。为满足严格 identity，采用 H5+γZ；协议冲突与梯度链处理记录为：{"identity_preserving_fusion": "H5 + gamma*Z; standard LN would violate the mandatory gamma=0 identity", "gradient_audit": "At exact gamma=0, P1 P_last and P2/P3 Ps are mathematically gradient-blocked on the first backward; require gamma first-backward gradient and projection gradient after first optimizer update."}。

## 10. Gradient Tests

三组 gradient manifests 均验证 HQMR grad=None、PLIP grad=None；γ 首次 backward 有梯度，受 γ=0 数学阻断的投影在首个 optimizer update 后获得非零梯度。

## 11. P0 HQMR

mIoU 65.5727%，mDice 78.9611%；作为全部 paired comparison 的冻结基线。

## 12. P1 VLM‑Last

mIoU 65.5726%（P1−P0 -0.000175 pp），Hard‑M1 corrective rate 0.0470%，WRP 99.9458%，γ abs mean 0.00101633。

## 13. VLM Novel Semantic Evidence

Question A：P1>P0 = False；Hard‑M1 correction 是否非零 = True。这只说明 frozen PLIP last-layer 是否提供可利用的新信息，不归因于 PDSR。

## 14. P2 Static‑PDSR

mIoU 65.5731%（P2−P1 +0.000519 pp），Hard‑M1=0.0452%，M1=0.0647%。

## 15. PDSR Cross‑Layer Reconstruction

β mean=[0.12385726720094681, 0.34774020314216614, 0.5284019708633423]，β>0.9 fractions=[0.0, 0.0, 0.0]，R/U ratios=[0.47836869955062866, 0.31793177127838135, 0.08696253597736359]；per-class、Hard-M1 与 correct-control 分层统计：`{"all": {"pixels": 158639345, "mean": [0.12241910372799862, 0.34982285661816936, 0.5277577897542763], "std": [0.04473952479535627, 0.06233040170917851, 0.07011872348269584], "gt_0_9_fraction": [0.0, 0.0, 0.0]}, "hard_m1": {"pixels": 20225308, "mean": [0.13216268667367106, 0.3369604331314452, 0.5308756824422325], "std": [0.03830661972586753, 0.0641751236328084, 0.07272594458254855], "gt_0_9_fraction": [0.0, 0.0, 0.0]}, "correct_controls": {"pixels": 129949034, "mean": [0.12046542155643031, 0.3526833970695723, 0.5268508399631837], "std": [0.045909467258725724, 0.06160273173068863, 0.07062951100177207], "gt_0_9_fraction": [0.0, 0.0, 0.0]}, "class_0": {"pixels": 61781842, "mean": [0.12723152062174525, 0.36141763645027813, 0.5113501241062297], "std": [0.027188986740319297, 0.05235659456629891, 0.05365622602100626], "gt_0_9_fraction": [0.0, 0.0, 0.0]}, "class_1": {"pixels": 66845621, "mean": [0.09490181862048683, 0.34623043884453203, 0.5588682616405555], "std": [0.034583560789405256, 0.06269979216170615, 0.06414613709375529], "gt_0_9_fraction": [0.0, 0.0, 0.0]}, "class_2": {"pixels": 20708801, "mean": [0.16963983620275927, 0.3641176241546475, 0.46624154569018567], "std": [0.043807014503178324, 0.061332155843838346, 0.07736243267408018], "gt_0_9_fraction": [0.0, 0.0, 0.0]}, "class_3": {"pixels": 9303081, "mean": [0.1830663203582213, 0.2668130272282855, 0.5501190373217631], "std": [0.046056474822752415, 0.05562467149668785, 0.06453822134590183], "gt_0_9_fraction": [0.0, 0.0, 0.0]}}`。

## 16. PDSR Mechanism Metrics

P2−P1 Hard‑M1 gain=-0.001857 pp，M1 gain=+0.003552 pp；PDSR decision=NOGO。

## 17. P3 VPCA‑PDSR

mIoU 65.5725%（P3−P2 -0.000626 pp；P3−P0 -0.000282 pp），Hard‑M1=0.0435%，M1=0.0619%。

## 18. Concept Participation Analysis

P3 concept statistics: `{"0": {"images": 2046, "entropy_mean": 0.36794039607048035, "top1_frequency_max": 1.0, "top2_mass_mean": 0.9470722675323486, "inter_image_js_mean": 0.002476873420896913, "collapse": true}, "1": {"images": 2713, "entropy_mean": 0.2952865958213806, "top1_frequency_max": 1.0, "top2_mass_mean": 0.9586492776870728, "inter_image_js_mean": 0.0014414001468641456, "collapse": true}, "2": {"images": 952, "entropy_mean": 0.4752589762210846, "top1_frequency_max": 1.0, "top2_mass_mean": 0.9229234457015991, "inter_image_js_mean": 0.009503871682590362, "collapse": true}, "3": {"images": 391, "entropy_mean": 0.8234496712684631, "top1_frequency_max": 1.0, "top2_mass_mean": 0.8278236985206604, "inter_image_js_mean": 0.0203082513777563, "collapse": true}}`。VPCA 使用固定 Top20%、τ=0.1、无 hard gate/threshold。

## 19. Concept Diversity / Collapse

CONCEPT_COLLAPSE=True；判据为同类图像 top‑1 concept 频率>80% 且 entropy<0.5·log(8)。

## 20. Hard‑M1 Analysis

冻结 Hard‑M1 count=5037，area=23161610；P1/P2/P3 area‑weighted HMCR=0.0470%/0.0452%/0.0435%。

## 21. M1 Analysis

historical M1=4440，exact active M1=4402；P1/P2/P3 M1CR=0.0612%/0.0647%/0.0619%。

## 22. True‑vs‑Rival Analysis

P2 margin={'mean': 0.08460844879640168, 'median': 0.060195766389369965, 'positive_fraction': 0.5898352193766131}；P3 margin={'mean': 0.10750680730797121, 'median': 0.0, 'positive_fraction': 0.48501091919793526}；positive‑fraction gain=-10.482430 pp。

## 23. TP Safety

P1/P2/P3 NCE=0.958/1.032/0.976；P3 TP harm=0.0080%。

## 24. Per‑Class Results

IoU P0/P1/P2/P3：C0 75.5616%/75.5618%/75.5617%/75.5614%；C1 69.2003%/69.1987%/69.2006%/69.2001%；C2 55.8135%/55.8132%/55.8140%/55.8127%；C3 61.7156%/61.7166%/61.7161%/61.7156%。CLASS_DAMAGE=False。

## 25. CCRA Health

{"effective_query_ratio": 1.0, "responsibility_overlap_change": 0.0, "query_peak_diversity_change": 0.0, "identity_audit": {"sample": "TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0", "stage_max_abs_drift": [0.0, 0.0], "max_abs_drift": 0.0, "exact_upstream_identity": true}, "reason": "CCRA tensors are exactly identical because the frozen CCRA computation is structurally upstream of the only H5 residual injection"}。注入点位于 CCRA 之后，且旧路径冻结，因此 CCRA responsibility 不随 P1/P2/P3 更新。

## 26. Computational Cost

| Variant | BF16 counted GFLOPs/view | Full parameters | Trainable parameters | E5 3-view FPS | Peak VRAM GiB |
|---|---:|---:|---:|---:|---:|
| P0 | 426.03 | 112269530 | 0 | 27.94 | 0.53 |
| P1 | 435.15 | 263743963 | 197120 | 68.59 | 1.83 |
| P2 | 435.28 | 264859611 | 1312768 | 66.46 | 1.84 |
| P3 | 435.32 | 264859611 | 1312768 | 65.91 | 1.94 |

P0 FPS 是单图循环估算，P1–P3 FPS 是 batch=8 的实际 E5 三视图吞吐量，不宜直接横比。GFLOPs 采用同一 BF16 profiler，但只计 torch.profiler 支持的算子；PLIP 虽冻结仍计入完整推理参数和吞吐量。三组训练耗时（秒）：{'P1': 691.3, 'P2': 736.3, 'P3': 746.3}。

## 27. Bootstrap CI

2000 次 paired image bootstrap，seed42。P3−P0 mIoU 95% CI=[-0.001138881787486612, 0.0004935914750051059] pp，包含零；P3−P2 95% CI=[-0.0009500587896982271, -0.00031351073202678936] pp，虽偏负但绝对量级仅约千分之一百分点。全量机器可读统计：`{"miou_P1_P0": {"mean": -0.00019222959130837936, "ci95": [-0.001358380638026846, 0.0008198778762752431]}, "miou_P2_P1": {"mean": 0.0005283291971954485, "ci95": [-0.00021745911658488382, 0.0012808064058139608]}, "miou_P3_P2": {"mean": -0.0006264102058955612, "ci95": [-0.0009500587896982271, -0.00031351073202678936]}, "miou_P3_P0": {"mean": -0.00029031060000849206, "ci95": [-0.001138881787486612, 0.0004935914750051059]}, "hmcr_P1": {"mean": 0.0004705888529996435, "ci95": [0.00043358468803354934, 0.0005119118476350145]}, "m1cr_P1": {"mean": 0.0006126390222219894, "ci95": [0.0005431123227092015, 0.0006842510561698474]}, "tpharm_P1": {"mean": 8.974467409913735e-05, "ci95": [8.298431385322816e-05, 9.714573200877668e-05]}, "hmcr_P2": {"mean": 0.0004522658063883596, "ci95": [0.0004162949429108501, 0.000489650200035876]}, "m1cr_P2": {"mean": 0.0006485611125257904, "ci95": [0.0005756825746468285, 0.0007267416958488228]}, "tpharm_P2": {"mean": 7.769077289252001e-05, "ci95": [7.20312348995779e-05, 8.336239326002045e-05]}, "hmcr_P3": {"mean": 0.0004359235827221811, "ci95": [0.00040164035985215305, 0.00047358812928136496]}, "m1cr_P3": {"mean": 0.000619982356558199, "ci95": [0.0005514287112414656, 0.0006962354762974839]}, "tpharm_P3": {"mean": 8.009187488046761e-05, "ci95": [7.43014104647095e-05, 8.598725806640397e-05]}, "resamples": 2000, "seed": 42, "unit": "paired image"}`。

## 28. Representative Cases

案例库：`{"P1": {"hard_m1_recovered": {"available": 1725, "rendered": 20}, "hard_m1_unrecovered": {"available": 2976, "rendered": 20}, "baseline_correct_harmed": {"available": 2288, "rendered": 20}}, "P2": {"hard_m1_recovered": {"available": 1863, "rendered": 20}, "hard_m1_unrecovered": {"available": 2838, "rendered": 20}, "baseline_correct_harmed": {"available": 2335, "rendered": 20}, "concept_grounding": {"available": 1863, "rendered": 20}}, "P3": {"hard_m1_recovered": {"available": 1808, "rendered": 20}, "hard_m1_unrecovered": {"available": 2893, "rendered": 20}, "baseline_correct_harmed": {"available": 2329, "rendered": 20}, "concept_grounding": {"available": 1808, "rendered": 20}}}`；每组固定 recovered/unrecovered/harmed，P2/P3 另含 concept grounding。

## 29. PDSR GO/NOGO

PDSR=NOGO。门槛：ΔHMCR≥5pp、ΔM1CR≥3pp、3/4 classes non‑negative、无 layer/CCRA collapse。

## 30. VPCA GO/NOGO

VPCA=NOGO。门槛：ΔHMCR≥5pp、positive margin fraction≥5pp、3/4 classes non‑negative、无 concept collapse。

## 31. Full Model GO/NOGO

FINAL_DECISION=FULL_MODEL_NOGO，FULL25_GO=False。需要 P3−P0≥0.50pp、PDSR与VPCA均GO、NCE>1.5且无严重安全失败。

## 32. Exact Next Step

本轮到此停止，不自动执行 Full25。若 FULL25_GO=True，下一步必须从官方 fresh ResNet38 initialization 开始公平的 25‑epoch 训练；否则仅根据报告中失败的因果环节设计下一轮最小实验。

