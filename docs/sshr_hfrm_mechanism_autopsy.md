# SSHR-HMA-v0 — HFRM Mechanism Autopsy

> 本报告是 frozen-checkpoint 机制审计，不是模型改造或调参实验。全程未构造 optimizer、未调用 optimizer.step、未评估 BCSS test/LUAD，也未训练新模型。

## 1. Frozen provenance and safety

- A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Audit commit: `eb0dc3de9a0ab71776549e20fe6fe30a40682b41`
- Checkpoint: `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- BCSS validation: 3418 images / 22 slides
- Fixed gradient audit: 32 logical batches × 20 (microbatch=4 with mean-gradient accumulation), seed=42
- Parameter SHA before/after gradient audit: `d326a905269d5b335802179b1bac4f3ee8375990b7065c23f184d8ad96e1050c` / `d326a905269d5b335802179b1bac4f3ee8375990b7065c23f184d8ad96e1050c`
- Buffer SHA before/after gradient audit: `12bf6484ee3e10df015879ba11bbfa46ac92665f9c72a29817f68b496fe15c1c` / `12bf6484ee3e10df015879ba11bbfa46ac92665f9c72a29817f68b496fe15c1c`
- Runtime: 3.89 min; peak CUDA memory: 2.111 GiB

### Instrumentation hard parity

Decision: **SSHR_HMA_PARITY_PASS**. Compared 480 same-process tensors over 32 images × 3 TTA views; max absolute difference=0.0, final differing pixels=0.

## 2. Executive summary

| Question | Measurement | Result | Evidence Level |
|---|---|---|---|
| GSR truly vetoes? | absent response Δ | median Δlogit=-0.265625; suppressed=90.67% | Direct frozen measurement |
| GSR handles present confusion? | present-confusion net recovery | 957,073/83,057,625 (1.152%) | Direct frozen measurement |
| CH remains low-pass? | kernel FFT / uniform cosine | 56: CH_BEHAVES_AS_HOMOGENIZER, 28_1: CH_BEHAVES_AS_HOMOGENIZER, 28_2: CH_BEHAVES_AS_HOMOGENIZER | Direct parameter measurement |
| CH helps interior? | raw→CH B2 | net=1,780,792; Δacc=1.2220 pp | Direct causal measurement |
| CH hurts boundary? | raw→CH B0 | net=-24,733; Δacc=-0.4773 pp | Direct causal measurement |
| GSR/CH complementary? | unique recover / Jaccard | G unique=72,879; CH unique=3,947,644; J=0.0544 | Direct paired prediction |
| 28_1 or 28_2 dominant? | paired branch-off ΔmIoU | 28_1=+1.1262 pp; 28_2=+0.1576 pp; 28_1 larger | Direct causal measurement |
| deep supervision dominant? | feat_deep gradient norm | deep=0.000658915; shallow sum=2.89426e-06 | Fixed-batch gradient audit |
| class gate contributes strongly? | gated vs ungated | ΔmIoU=+9.3771 pp | Direct pipeline decomposition |
| min-max amplifies response? | raw vs normalized delta | median ratio=0.393; >2×=6.79% | Direct response measurement |

## 3. Learned scalar autopsy

| Stage | gamma_veto | gamma_context | sign_veto | sign_context | |veto/context| |
|---|---:|---:|---|---|---:|
| 56 | 0.796096 | 1.54709 | positive | positive | 0.514575 |
| 28_1 | 0.567998 | 1.5722 | positive | positive | 0.361277 |
| 28_2 | 0.57585 | 1.26884 | positive | positive | 0.453839 |

按公开公式，`gamma_veto > 0` 时 GSR 项是对 gated feature 的加性放大/调制；只有 `gamma_veto < 0` 才是直接 feature attenuation。语义上的 absent-class 抑制需由实测响应另行判断，不能仅凭模块命名断言。

## 4. Context-kernel autopsy

| Stage | Uniform Cosine | Neg. Fraction | DC Gain | HF/LF | Anisotropy | Label |
|---|---:|---:|---:|---:|---:|---|
| 56 | 0.99935 | 0 | 0.799333 | 0.0437655 | 0.00108381 | CH_BEHAVES_AS_HOMOGENIZER |
| 28_1 | 0.999572 | 0 | 0.503578 | 0.0457749 | 0.00147404 | CH_BEHAVES_AS_HOMOGENIZER |
| 28_2 | 0.999951 | 0 | 0.465741 | 0.0475583 | 0.000935898 | CH_BEHAVES_AS_HOMOGENIZER |

K=15 对应约 60 input pixels（F56）和 120 input pixels（F28）；这是尺度结构事实，不构成 K 值优劣的性能证据。

## 5. Same-forward causal CAM audit

| Branch | Raw mIoU | GSR-only | CH-only | Full | GSR Gain | CH Gain | Full Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAM56 | 48.1248 | 50.0818 | 62.0435 | 61.4655 | +1.9570 | +13.9187 | +13.3408 |
| CAM28_1 | 56.8371 | 58.6762 | 67.0218 | 67.0272 | +1.8391 | +10.1847 | +10.1901 |
| CAM28_2 | 64.1385 | 64.5086 | 66.4848 | 66.4956 | +0.3701 | +2.3462 | +2.3571 |
| CAMdeep | 64.9611 | — | — | 64.9611 | — | — | 0.0000 |

CAM56 的结果只描述其 standalone head；released final fusion 不使用 CAM56，而且 frozen ablation 不能识别其历史训练期因果贡献。

### Final official-fusion variants

| Variant | mIoU | Δ vs Full | mDice | C0 | C1 | C2 | C3 | Differing pixels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official Full | 67.3283 | — | 80.2683 | 76.4494 | 70.5721 | 57.8272 | 64.4646 | 0 |
| All HFRM off | 65.0065 | -2.3219 | 78.5061 | 75.1278 | 69.4560 | 54.0668 | 61.3754 | 7,587,750 |
| GSR-only | 65.2366 | -2.0917 | 78.6896 | 75.1966 | 69.4965 | 54.6008 | 61.6526 | 7,359,205 |
| CH-only | 67.3141 | -0.0143 | 80.2578 | 76.4237 | 70.5383 | 57.7663 | 64.5280 | 713,680 |
| 28_1 HFRM off | 66.2021 | -1.1262 | 79.4098 | 75.8934 | 69.9326 | 55.4113 | 63.5712 | 5,343,285 |
| 28_2 HFRM off | 67.1708 | -0.1576 | 80.1494 | 76.3857 | 70.5348 | 57.5627 | 64.1999 | 1,212,689 |
| 28_1 GSR off | 67.3112 | -0.0172 | 80.2559 | 76.4239 | 70.5414 | 57.7854 | 64.4940 | 604,358 |
| 28_1 CH off | 66.3034 | -1.0249 | 79.4907 | 75.9143 | 69.9506 | 55.7129 | 63.6358 | 5,215,753 |
| 28_2 GSR off | 67.3336 | +0.0052 | 80.2719 | 76.4505 | 70.5703 | 57.8105 | 64.5030 | 173,072 |
| 28_2 CH off | 67.1920 | -0.1364 | 80.1659 | 76.3900 | 70.5422 | 57.6233 | 64.2125 | 1,186,446 |

Full vs Raw frozen-checkpoint effect: **+2.3219 mIoU points**. GSR-only gain=+0.2301; CH-only gain=+2.3076.

## 6. GSR, CH, and inference pipeline

- Absent-class response: median GSR Δlogit=-0.265625, mean=-0.291393, suppression fraction=90.67%.
- Present-confusion net recovery: 957,073 pixels from 83,057,625 raw present-confusion errors (1.152%).
- CH raw→CH near-boundary net=-24,733, interior net=1,780,792; paired GSR→Full results are retained in `paired_causal/ch_spatial_effect.json`.
- Complementarity: recover Jaccard=0.0544; GSR-only unique=72,879; CH-only unique=3,947,644; overlap=231,441.
- Official hard gate: Full no-gate=57.9513, Full gated=67.3283, Δ=+9.3771 points.
- Min-max response: median |raw-scale delta|=0.0545341; median normalized/raw amplification ratio=0.393; fraction >2×=6.79%.

## 7. Fixed training-gradient audit

| Loss branch | Weight | Shared Early | Mid | Late | feat_deep | HFRM target |
|---|---:|---:|---:|---:|---:|---:|
| L56 | 0.10 | 0.476107 | 0.0100616 | 0.00890803 | 1.51162e-06 | 0.0269274 |
| L28_1 | 0.15 | 0.355505 | 0.464978 | 0.0010779 | 5.37794e-07 | 0.0307072 |
| L28_2 | 0.25 | 0.670564 | 0.844362 | 0.798769 | 8.44847e-07 | 0.0450779 |
| Ldeep | 0.50 | 1.45718 | 1.76094 | 1.75659 | 0.000658915 | — |

Direct deep-loss feat_deep gradient norm=0.000658915; the three shallow losses' norms sum to 2.89426e-06 (ratio=0.0044). Gradient-cosine matrices are in the JSON audit and `figures/gradient_cosine_matrix.png`; these are fixed-batch observational gradients, not optimizer updates.

## 8. Error taxonomy

| Candidate | Error type | Raw wrong | Candidate wrong | Recovered | Harmed | Net |
|---|---|---:|---:|---:|---:|---:|
| gsr_only | absent_class | 4,365,373 | 4,357,892 | 44,487 | 29,163 | 15,324 |
| gsr_only | present_confusion | 24,466,493 | 24,351,460 | 259,833 | 152,643 | 107,190 |
| gsr_only | boundary | 6,165,204 | 6,162,301 | 38,894 | 35,991 | 2,903 |
| gsr_only | interior | 22,666,662 | 22,547,051 | 265,426 | 145,815 | 119,611 |
| ch_only | absent_class | 4,365,373 | 4,038,116 | 599,064 | 247,900 | 351,164 |
| ch_only | present_confusion | 24,466,493 | 23,115,108 | 3,580,021 | 2,252,543 | 1,327,478 |
| ch_only | boundary | 6,165,204 | 6,267,354 | 520,178 | 622,328 | -102,150 |
| ch_only | interior | 22,666,662 | 20,885,870 | 3,658,907 | 1,878,115 | 1,780,792 |
| official_full | absent_class | 4,365,373 | 4,036,548 | 573,784 | 220,733 | 353,051 |
| official_full | present_confusion | 24,466,493 | 23,084,639 | 3,379,768 | 2,022,140 | 1,357,628 |
| official_full | boundary | 6,165,204 | 6,252,292 | 481,920 | 569,008 | -87,088 |
| official_full | interior | 22,666,662 | 20,868,895 | 3,471,632 | 1,673,865 | 1,797,767 |

在 Full frozen prediction 中，remaining absent-class=4,036,548，present-class confusion=23,084,639；数量更大的类别是 **present-class confusion**。

## 9. Evidence-ranked weakness map

Completion: **HFRM_MECHANISM_MAP_COMPLETE**

Evidence labels: `GSR_IS_GLOBAL_AMPLIFICATION`, `GSR_VETO_SUPPORTED`, `CH_HOMOGENIZER_SUPPORTED`, `CH_BOUNDARY_TRADEOFF_CONFIRMED`, `HFRM28_1_DOMINANT`, `DEEP_SUPERVISION_DOMINANT`, `CLASS_GATE_DOMINANT`

### Tier A — directly supported

- All trained gamma_veto scalars are positive, so the released residual equation performs additive channel amplification/modulation rather than direct feature suppression.
- CH improves interior pixels while causing net near-boundary harm in paired causal predictions.
- The remaining frozen Full errors are dominated by present class confusion (23084639 present-confusion vs 4036548 absent-class pixels).

### Tier B — structurally plausible

- Each GSR gate is global over space: one channel vector is broadcast to every pixel in a stage.
- The same deep feature is directly supervised at weight 0.50 and simultaneously conditions all three HFRMs.
- HFRM56 has a training loss but no direct path to the released final inference fusion or deeper backbone stages.
- A fixed K=15 spans approximately 60 input pixels at F56 and 120 at F28; this is a structural scale mismatch, not proven performance harm.

### Tier C — speculative

- Whether a spatially varying semantic mechanism would improve present-class localization is untested.
- Whether constraining CH to a low-pass family would improve segmentation is untested.
- Whether changing K by stage would improve the smoothing/boundary trade-off is untested.
- The historical training-time contribution of HFRM56 cannot be identified by frozen-checkpoint inference ablation.

## 10. Answers to the 20 frozen questions

1. Gamma values/signs are listed in Section 3; exact values are preserved in `parameter_autopsy/gamma_autopsy.json`.
2. By sign, trained GSR is mathematically additive amplification/modulation; empirical absent response is a separate measurement.
3. Absent classes: median Δlogit=-0.265625, suppressed fraction=90.67%; this meets the frozen veto criterion.
4. Present confusion net recovery=957,073 pixels (1.152% of the raw present-confusion count).
5. Trained CH labels: 56: CH_BEHAVES_AS_HOMOGENIZER, 28_1: CH_BEHAVES_AS_HOMOGENIZER, 28_2: CH_BEHAVES_AS_HOMOGENIZER; the conclusion uses direct spatial weights and zero-padded FFT, not module naming.
6. CH raw→CH boundary Δacc=-0.4773 pp; interior Δacc=+1.2220 pp.
7. Boundary harm is supported by the preregistered net-transition sign (net=-24,733).
8. GSR/CH recover-set Jaccard=0.0544; the frozen label is unclassified: no complementarity/redundancy/conflict rule fired, so the asymmetric/mixed pattern is reported without forcing a label.
9. Direct official-fusion removal costs: 28_1=+1.1262 pp, 28_2=+0.1576 pp; 28_1 is larger.
10. CAM56 standalone Raw/GSR/CH/Full is reported in Section 5. It is excluded from released fusion, and frozen inference cannot establish its training-time causal contribution.
11. Same-checkpoint Full−Raw=+2.3219 mIoU points.
12. Official hard class-gate contribution on Full=+9.3771 mIoU points.
13. Min-max median amplification ratio=0.393; 6.79% of stage/class/image cells exceed 2×.
14. Deep weighted loss feat_deep norm=0.000658915 vs shallow sum=2.89426e-06; dominance is decided from these measured norms, not nominal 0.50 alone.
15. Shallow-to-deep gradient ratio=0.0044; per-branch values and cosines are archived.
16. Remaining error is dominated by present-class confusion: absent=4,036,548, present-confusion=23,084,639.
17. Tier-A weaknesses are exactly those listed in Section 9 and have direct parameter/paired-prediction/gradient evidence.
18. Tier-B items are structural facts or plausible bottlenecks without frozen causal performance proof.
19. Tier-C items are deliberately retained as untested hypotheses and are not promoted to innovation designs.
20. Most important unresolved scientific question: **How can deep semantic guidance resolve spatial present-class confusion while preserving tissue boundaries, and is that limitation causal during retraining?**

## 11. Artifact map

- `provenance/manifest.json` and `provenance/source_contract.json`
- `parameter_autopsy/gamma_autopsy.json`
- `kernels/kernel_channel_metrics.csv` and `kernels/kernel_summary.json`
- `gates/gate_vectors.npz`, gate statistics, semantic separability, and GSR response rows
- `paired_causal/final_variants.json`, CH spatial effects, complementarity, and present-confusion effects
- `standalone_cam/standalone_cam.json`
- `error_taxonomy/error_taxonomy.json`
- `inference_decomposition/pipeline.json`
- `gradient_audit/gradient_rows.csv`, component rows, and summary
- `figures/*.png`

---

**HFRM_MECHANISM_MAP_COMPLETE**

STOP: no new HFRM, gamma/K change, spatial gate, loss, training, test, or LUAD evaluation was executed.
