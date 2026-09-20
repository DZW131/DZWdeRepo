# PCSI-v1 — Pre-CCRA Semantic Injection + VPCA Collapse Audit

> BCSS seed 42 · frozen HQMR epoch 25 · PLIP frozen · Phase0 5 epochs C1/C2 · 3-view TTA · 3,418 validation images

## 首页决策卡

| Field | Result |
|---|---|
| PHASE_A_DECISION | GO |
| PRE_CCRA_SITE | I1_PRE_CONTEXT |
| IDENTITY_PASS | True |
| GRADIENT_PATH_PASS | True |
| VPCA_V1_STATUS | CLOSED |
| VPCA_ROOT_CAUSE | VISUAL_RESPONSE_RANK_LOCK, GLOBAL_CONCEPT_BIAS, COMBINED_FAILURE |
| TEXT_SPACE_COLLAPSE | False |
| VISUAL_RESPONSE_RANK_LOCK | True |
| TOPK_AGGREGATION_LOCK | False |
| TEMPERATURE_SHARPENING | False |
| GLOBAL_CONCEPT_BIAS | True |
| PHASE_C_EXECUTED | True |
| C0_mIoU | 65.5727% |
| C1_mIoU | 65.5727% |
| C2_mIoU | 65.5729% |
| C1-C0 | -0.000064 pp |
| C2-C1 | +0.000253 pp |
| C2-C0 | +0.000189 pp |
| C1_HRCR | 0.0000% |
| C2_HRCR | 0.0000% |
| C1_HardM1CR | 0.0473% |
| C2_HardM1CR | 0.0443% |
| C1_NCE | 0.947106 |
| C2_NCE | 1.01978 |
| PLIP_SEMANTIC_SOURCE | NOGO_BY_C1_GATE |
| PDSR_DECISION | NOGO |
| FINAL_ROUTE | PRE_CCRA_VLM_SEMANTIC_NOGO |
| FULL25_GO | False |

## 关键因果顺序（固定 32 张，E5 后）

`decoder1/query1` 是注入点之前的输入，须不变；HQMR 内部名为 `query0` 的张量在 CCRA2 之后，因此应变化。原方案把两个 q0 混用，不能以 HQMR `query0` 不变作为合法门槛。

| Tensor | C1 RMS / max abs | C2 RMS / max abs |
|---|---:|---:|
| query1 | 0 / 0 | 0 / 0 |
| context_pre | 0 / 0 | 0 / 0 |
| context | 0.000647337 / 0.0062654 | 8.08473e-05 / 0.000752419 |
| ccra_k5 | 0.000403551 / 0.00390625 | 0.000213638 / 0.00390625 |
| ccra_v5 | 0.000472028 / 0.0078125 | 0.000240103 / 0.0078125 |
| ccra2_responsibility | 1.56725e-06 / 0.000120297 | 4.08451e-07 / 5.15431e-05 |
| hqmr_query0 | 0.000544306 / 0.0142748 | 0.00040302 / 0.0139682 |
| hqmr_k5 | 0.00207242 / 0.0427918 | 0.000807694 / 0.040957 |
| hqmr_v5 | 0.00158992 / 0.0318789 | 0.000694951 / 0.0236447 |
| hqmr5_logits | 0.00689876 / 0.0625 | 0.00289362 / 0.0625 |
| hqmr_query5 | 0.000876697 / 0.0137506 | 0.000509959 / 0.0115074 |
| hqmr4_logits | 0.00848468 / 0.09375 | 0.00364491 / 0.09375 |
| hqmr_query4 | 0.000926078 / 0.0130755 | 0.000554757 / 0.0105602 |
| hqmr3_logits | 0.00764503 / 0.0820312 | 0.00271817 / 0.0859375 |
| final | 0.000861781 / 0.00390625 | 0.000330371 / 0.00390625 |

CCRA2 spatial responsibility argmax change: C1 0.1126%, C2 0.0219%. Tensor hashes are saved in `phaseC/posttrain_tensor_hash_audit.json`.

## 1. Executive Decision

**PRE_CCRA_VLM_SEMANTIC_NOGO**。Phase A 证明了注入位置的因果通路，但 E5 的 C1–C0 仅 -0.000064 pp、C2–C0 仅 +0.000189 pp；C2–C0 的配对 95% CI 为 [-0.000737, +0.001015] pp，包含零，且两组 Hard-M1 的 L5 正确责任恢复都是 0。C1 最低门槛=False，C2 PDSR 门槛=False，强门槛=False。本轮不进入 Full25。

## 2. Frozen Context

HQMR checkpoint SHA-256 `84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb`; C0 exact UMRF prediction bank `/home/duyanhong/experiments/UMRF_v1_Upstream_MultiLevel_Responsibility_Formation_Audit_BCSS_Seed42/gt_free_prediction_maps.uint8.npy`. Historical HQMR scalar 65.572444% differs from paired bank ~65.572738% by +0.000294 pp; all present deltas use the single paired bank. Previous post-CCRA P1/P2/P3 were 65.5726/65.5731/65.5725%, essentially tied with P0.

## 3. Why Previous Phase0 Did Not Test Pre-CCRA Hypothesis

The old `h5_residual` is added only after `GCQMNet.forward` has returned; CCRA2 and CCRA3 have already assigned class responsibility. Its identity of upstream CCRA tensors was expected, not evidence that PLIP has no semantics.

## 4. Exact Forward Dependency Graph

Runtime-hooked DAG and edge list: `phaseA_forward/forward_dependency_graph.json` and `.md`. `F5.detach → context_projection → CHPF/context_feature → CCRA2 K/V and responsibility → query2 → CCRA3/HQMR query0`; the same context feature also feeds HQMR scale-5 K/V → L5 → q5 → L4 → q4 → L3.

## 5. CCRA Entry Point

Selected I1: the common 256×28×28 `context_feature` immediately after `f5_chpf`, before CCRA2's memory K/V. No query residual, CCRA weight change, GT mask, or image-label leakage was introduced at inference.

## 6. Injection Candidate Audit

| Site | pre-CCRA | ΔCCRA-K RMS | ΔCCRA-V RMS | ΔL5 RMS | spatial responsibility change | GO |
|---|---|---:|---:|---:|---:|---|
| I0_LATE_H5 | False | 0 | 0 | 0.0303938 | 0.0000% | False |
| I1_PRE_CONTEXT | True | 0.0073319 | 0.00731083 | 0.0314296 | 3.4957% | True |
| I2_CCRA_K_ONLY | True | 0.0166714 | 0 | 0.00202087 | 4.3846% | True |
| I3_CCRA_V_ONLY | True | 0 | 0.0208075 | 0.00648425 | 0.0000% | False |

I0 is a deliberately late negative control; I2 K-only is causal but lacks the common K/V feature source; I3 V-only does not change responsibility.

## 7. Identity Tests

All candidate alpha=0 tests were exact. Both newly instantiated C1/C2 had γ=0 and exact identity for query1, prefeature, CCRA2/3 responsibility, L5/L3 and final output on the fixed 32 images. Positive γ=.1 changed CCRA and L5. Manifest: `phaseC/identity_pretrain.json`.

## 8. Causal Perturbation

I1 at alpha=.10, no training/no segmentation GT: ΔCCRA K/V RMS 0.0073319/0.00731083; spatial responsibility change 3.50%; ΔL5/L3 RMS 0.03143/0.04449. Alpha .05 and .20 are secondary sensitivity checks in `causal_sensitivity.csv`. The probe normalized semantic RMS to the context RMS for a controlled perturbation; Phase C used the preregistered raw `γ·Z` fusion, so alpha=.10 is not an E5 effect-size prediction.

## 9. Gradient Path

Using original HQMR loss, the prefeature, gamma and PDSR projection received nonzero gradients while frozen HQMR/PLIP parameters had grad=None. Diagnostic CCRA responsibility is detached on output, but internal pooled update remains differentiable. `phaseA_forward/gradient_path.json`.

## 10. Phase-A Decision

`GO`. Structural causality, alpha-zero identity, and gradient path all passed; only then were C1/C2 run.

## 11. VPCA Collapse Recap

Previous P3 showed four-class top-1 concept collapse and no segmentation gain. This audit re-used the exact P3 projection, PLIP and concept cache on 23,422 training images without opening segmentation GT.

## 12. Text-Space Geometry

Within-class mean pairwise cosine: C0 0.711, C1 0.645, C2 0.732, C3 0.616. Cross-class confusion table and all 8×8 cosine plots are under `phaseB_vpca/`.

## 13. Concept Effective Rank

Effective rank out of 8: C0 6.911, C1 6.904, C2 6.833, C3 6.903. None met text-space collapse criterion.

## 14. Raw Visual-Concept Ranking

Top-1 concept frequencies among positive training images: C0 concept 2 = 100.0000%, C1 concept 1 = 100.0000%, C2 concept 6 = 100.0000%, C3 concept 3 = 100.0000%. The lock is already present in raw g20, before q/rho.

## 15. Pooling Audit

Mean, Top50, Top20, Top10 and Max all retained ~100% top-1 locks. Thus TopK did not *create* the lock; see `phaseB_vpca/pooling_audit.csv`.

## 16. Temperature Audit

At τ=.1, mean q entropy by class: C0 0.351, C1 0.287, C2 0.469, C3 0.643. Positive-temperature softmax preserves argmax exactly; the plan's proposed criterion 'top1 under 80% before softmax but over 80% afterward' is mathematically impossible. Lower τ sharpens entropy/mass but cannot introduce a different top-1 ranking.

## 17. rho/q Decomposition

ρ max median 0.257; dominant rho-class frequency 41.7086%. Class-level ρ dominance criterion was not met. `phaseB_vpca/rho_audit.csv` separates ρ from within-class q.

## 18. Global Concept Bias

Raw visual response rankings favor one concept in every class across positive images, with mean-pooling lock already present. `phaseB_vpca/concept_bias.csv` contains score levels/gaps; the evidence supports a global concept bias diagnosis, not a causal proof that any one text prompt alone is defective.

## 19. Instance Sensitivity

ISR by class: C0 0.0291, C1 0.0239, C2 0.0661, C3 0.1183. All are <1, consistent with concept-mean differences dominating image-to-image variation.

## 20. VPCA Root-Cause Decision

`VISUAL_RESPONSE_RANK_LOCK, GLOBAL_CONCEPT_BIAS, COMBINED_FAILURE`. TEXT_SPACE_COLLAPSE=False; TOPK_AGGREGATION_LOCK=False; TEMPERATURE_SHARPENING=False; RHO_CLASS_DOMINANCE=False. VPCA-v1 stays CLOSED; no formula repairs or training were performed.

## 21. Corrected Pre-CCRA Experiment Protocol

C0 frozen UMRF bank; C1 frozen PLIP last dense + P_last + zero-initialized channelwise gamma; C2 frozen PLIP multi-level static-concept PDSR + gamma. Frozen HQMR/CCRA/PLIP. Seed42, original PolyOptimizer LR=.1/WD=.0005, BF16, microbatch5×accumulation4, exactly five epochs/5,855 optimizer steps each. No validation E1–E4; E5-only common 3-view evaluation.

## 22. C0

Frozen baseline mIoU=65.5727%, mDice=78.9611%. Prediction maps were not regenerated or tuned.

## 23. C1 Pre-CCRA VLM-Last

mIoU=65.5727%, mDice=78.9610%; gamma abs mean=0.00113827; train time=703.3s, peak VRAM=1.605 GiB, checkpoint SHA-256 `8e8b302c911141a3911efdb39b48933dd5664316a187708c130a64be2e98481e`. Validation/segmentation GT access during training both false.

## 24. C2 Pre-CCRA Static-PDSR

mIoU=65.5729%, mDice=78.9612%; gamma abs mean=0.000396058; train time=753.3s, peak VRAM=1.637 GiB, checkpoint SHA-256 `659d1c15d1c2453df8ce1335a9d121332ba52ad3fc4ea513838bdcf3ba045f27`. Validation/segmentation GT access during training both false.

## 25. L5 Responsibility Changes

Component-weighted RCR (baseline L5 wrong → new L5 true): C1 0.0000% (n=5631), C2 0.0000% (n=5631). Stage5 responsibility is class evidence from frozen HQMR logits5 + detached class weights, 3-view averaged, class-softmaxed, and uniformly pooled over the *exact* baseline connected component.

## 26. Hard-M1 Responsibility Recovery

HRCR: C1 0.0000% (area-weighted 0.0000%), C2 0.0000% (area-weighted 0.0000%). Hard-M1 cohort is frozen UMRF sequential5/4/3-all-wrong, not redefined from new predictions.

## 27. Final Segmentation Recovery

Hard-M1 pixel-area corrective rate: C1 0.0473%, C2 0.0443%; M1 corrective rate: C1 0.0618%, C2 0.0631%. These are distinct from component-weighted HRCR.

## 28. True-vs-Rival

Rival exit RER: C1 0.2184%, C2 0.1191%；离开后进入正确类的条件比例均为 0；Hard-M1 的 L5 true−rival margin 均值分别为 -0.0302336、-0.0302143。RER 的 rival 严格取冻结 L5 的错误预测类别（`baseline_l5`），不是最终分割图上的 baseline 类别。首次汇总误用了后者；最终结果来自修正后的 `evaluate_corrected.log`，原始错误口径日志仅作审计留痕，不参与结论。

## 29. TP Safety

TP harm: C1 0.0091%, C2 0.0078%; NCE=corrected/harmed: C1 0.947106, C2 1.01978. Interpret small absolute changes against the paired C0 mask, not only the ratio.

## 30. Per-Class

| Class | C0 IoU | C1 IoU | C2 IoU | C1−C0 | C2−C1 |
|---|---:|---:|---:|---:|---:|
| C0 | 75.5616% | 75.5617% | 75.5616% | +0.000115 pp | -0.000086 pp |
| C1 | 69.2003% | 69.1984% | 69.2005% | -0.001890 pp | +0.002072 pp |
| C2 | 55.8135% | 55.8137% | 55.8139% | +0.000158 pp | +0.000227 pp |
| C3 | 61.7156% | 61.7169% | 61.7157% | +0.001361 pp | -0.001201 pp |

BCSS C0 Tumor, C1 Stroma, C2 Lymphocytic infiltrate, C3 Necrosis; label 4 ignored in foreground metrics.

## 31. Parameter Audit

Adapter checkpoints contain trainable keys only. Frozen base and PLIP state hashes are identical before/after adapter loading, and runtime gradient manifests show frozen grad=None. C1: base `2a2b578c157c62a1…`, PLIP `7c60e2178b993305…`, trainable params 197,120; C2: base `2a2b578c157c62a1…`, PLIP `7c60e2178b993305…`, trainable params 1,312,768. See `phaseC/parameter_audit.json`.

## 32. Bootstrap

2,000 paired-image bootstrap resamples, seed42; mIoU deltas are pp, rates are fractions.

| Metric | mean | paired-image 95% CI |
|---|---:|---|
| C1-C0_mIoU_pp | -7.84482e-05 | [-0.00124244, 0.00100008] |
| C2-C1_mIoU_pp | 0.000253642 | [-0.000471474, 0.000958082] |
| C2-C0_mIoU_pp | 0.000175194 | [-0.000737148, 0.00101512] |
| C1_HardM1CR | 0.000473627 | [0.000435395, 0.000515556] |
| C2_HardM1CR | 0.000442986 | [0.000407815, 0.000479925] |
| C1_HRCR | 0 | [0, 0] |
| C2_HRCR | 0 | [0, 0] |
| C1_TPHarm | 9.1157e-05 | [8.40428e-05, 9.87543e-05] |
| C2_TPHarm | 7.82915e-05 | [7.26618e-05, 8.4009e-05] |

## 33. Representative Cases

A: L5 repaired + final repaired 0/20; B: L5 repaired but final wrong 0/20; C: baseline correct harmed 20/20. Each panel shows RGB, GT, C0, C1, C2 and frozen component. If fewer than 20 exist, all available distinct images are shown, without fabricating cases. `visualizations/manifest.csv`; VPCA has 10 GT-free training examples per class.

## 34. PLIP Semantic Source Decision

NOGO_BY_C1_GATE. C1 minimum gate requires HRCR≥5%, Hard-M1 final correction≥1%, NCE>1. A failure means the tested frozen PLIP-last route did not meet this gate; it does not prove all VLM semantics absent.

## 35. PDSR Decision

NOGO. C2 gate requires HRCR gain ≥5 pp **or** Hard-M1CR gain ≥3 pp versus C1, plus C2 mIoU≥C1 and ≥3/4 class IoUs nonnegative. Strong architecture gate additionally needs C2−C0 mIoU≥0.30 pp, C2 HRCR≥10%, NCE≥1.5.

## 36. VPCA-v1 Status

CLOSED. This experiment audited the previous collapse only; no VPCA-v1 repair, new formula, or VPCA training was performed.

## 37. Preserved Routes

The frozen HQMR/CCRA/PLIP, same BCSS/UMRF bank, and static-concept PDSR are retained as controlled references. Future routes depend on the explicit gate outcomes above, not an E5 cherry-picked checkpoint.

## 38. Closed Routes

No Full25 in this round; previous late post-CCRA injection and VPCA-v1 are closed as primary candidate mechanisms. Do not infer improvement from a positive causal probe alone.

## 39. Exact Next Step

按预注册门槛停止本轮；关闭当前 PLIP-last / Static-PDSR pre-CCRA 主线及 VPCA-v1，不追加 10/20/25 epochs，也不依据 validation GT 调 γ。下一轮若继续，应另立方案，优先判别“语义源不能改变正确类别排序”与“CCRA/HQMR 接口放大不足”，而非把这次微小的 mIoU 波动当成涨点。

## Reproducibility and limitations

Server output root: `/home/duyanhong/experiments/PCSI_v1_BCSS_Seed42`. Code: `tools/pcsi_v1/README.md` and branch `experiment/pcsi-v1`. Checkpoints: C1 `8e8b302c911141a3911efdb39b48933dd5664316a187708c130a64be2e98481e`, C2 `659d1c15d1c2453df8ce1335a9d121332ba52ad3fc4ea513838bdcf3ba045f27`. Single seed (42), single BCSS split, five-epoch mechanism probe: uncertainty CIs are conditional on this split and do not establish external generalization. The original plan's 'q0 unchanged' only applies to pre-CCRA input query1, not HQMR query0. Temperature top1-change criterion is rank-invariant and therefore not used as a positive test.
