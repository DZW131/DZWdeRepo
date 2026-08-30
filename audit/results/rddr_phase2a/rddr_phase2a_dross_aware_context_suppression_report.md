# RDDR Phase-2A Dross-Aware Context Suppression Report

## 0. 执行结论 / Executive conclusion

- 最终判定：`CONTEXT_SUPPRESSION_SEMANTIC_DAMAGE`。仅评估 BCSS validation，未使用 test。
- 同一官方评估器：C0=67.3363，GS=67.0918，RCS=67.0703 mIoU。
- RCS−C0=-0.2661 pp，95% CI=[-0.4317, -0.1302] pp。
- Gate A/B/C/D：FAIL / FAIL / FAIL / PASS。
- 全部比较使用 Epoch25 FINAL；Epoch1/5/10/15/20 权重仅用于机制轨迹观察，不用于模型选择。
- 本报告结束即停止，不启动 test、LUAD、多 seed、消融或下一种结构。

## 1. Frozen provenance and commands

- Implementation commit: `6f45ac7676b2e7bd7ae21c23db3303de95e02c6c`
- Evaluation commit: `1d6250566f383e924c16ace51c1c156942895e1d`
- Pure A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- C0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- GS checkpoint SHA256: `f748b4290e6cb6eedc6e3372b313bd1f977d41ce48c8fc048949d250f82a7031`
- RCS checkpoint SHA256: `2f9960bfa5bdd61a560e8a60ecd1af139ebbe7594dd1c7b6954bdd034fd333ac`
- Locked JSD helper SHA256: `1142ff8e8f95d3447012af9c4eb8f91eb923a48d5e8f840ea42098cc2f1de58b`
- Locked model source SHA256: `a6f6cf3a82c23d5a7a99c41c6f1348c118428aa6a508ee0dc71d7f44ac9f1f3d`
- Dataset/split: 3418 BCSS validation images; no test or LUAD access.

```bash
bash tools/run_rddr_phase2a_server.sh /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7 /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params /home/duyanhong/reseg-data/raw/BCSS-WSSS/training /home/duyanhong/miniconda3/envs/sshr5090/bin/python
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/analyze_rddr_phase2a.py --c0-checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --gs-dir /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/GS --rcs-dir /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/RCS --phase0-dir /home/duyanhong/experiments/RDDR_PHASE0_586f402/formal --phase1-dir /home/duyanhong/experiments/RDDR_PHASE1_4e08c9d/report --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --frozen-phase0-cache /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/diagnostics/frozen_phase0_populations --smoke-json /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/diagnostics/rddr_phase2a_smoke.json --pretrained /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --python-executable /home/duyanhong/miniconda3/envs/sshr5090/bin/python --output-dir /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/report_final --num-workers 4 --max-images 0 --bootstrap-resamples 10000
```

## 2. Architecture, capacity, and semantic-preservation contract

Only the HFRM28_1 context residual is scaled. The original feature F and semantic veto residual remain untouched.

```text
C0: F' = F + gamma_sem*R_sem + gamma_ctx*R_ctx
GS: F' = F + gamma_sem*R_sem + gamma_ctx*mean(1-q)*R_ctx
RCS: F' = F + gamma_sem*R_sem + gamma_ctx*(1-q_i)*R_ctx
```

- Total parameters (all variants): 112709714
- Additional trainable parameters: 0
- Initial zero-gamma max absolute difference: 0
- Same-checkpoint pre-HFRM feature max difference: 0
- Same-checkpoint pre-HFRM feature cosine: 1.000000000

## 3. Training equivalence

GS and RCS use seed42, batch20, BF16, epoch0→25, official pretrained weights, released augmentation, loss 0.10/0.15/0.25/0.50, released PolyOptimizer/LR schedule, and Epoch-25 FINAL checkpoints. Training never evaluated validation or test.

Training data: 23,422 parsed images, 1,171 batches/epoch with drop_last, 29,275 optimizer steps. Input 224x224; random horizontal/vertical flips; ImageNet mean/std normalization. No learning-rate or momentum correction was introduced. Poly decay exponent remains 0.9.

| Optimizer group | Initial LR | Weight decay | Actual SGD momentum | Parameter tensors |
|---|---:|---:|---:|---:|
| 0 | 0.01 | 0.0005 | 0.0005 | 82 |
| 1 | 0.02 | 0.0 | 0.0005 | 39 |
| 2 | 0.1 | 0.0005 | 0.0005 | 19 |
| 3 | 0.2 | 0.0 | 0.0005 | 3 |

GS and RCS optimizer-group records match exactly; every parameter is grouped once. The released actual momentum=0.0005 is intentionally retained.

## 4. Overall metrics and CAM hierarchy

Official three-view TTA is averaged in the native output dtype before FP32 normalization. BCSS presence thresholds are [0.8,0.9,0.8,0.6], with argmax fallback when none pass; final fusion is 0.6/0.2/0.2 (CAM56 diagnostic only). The initial copied audit helper averaged after FP32 conversion; this was corrected before this evaluation. Original infer()/metric/model files were not changed.

Direct pixel parity against unchanged official infer(): {'C0': {'images': 8, 'mismatched_prediction_pixels': 0}, 'GS': {'images': 8, 'mismatched_prediction_pixels': 0}, 'RCS': {'images': 8, 'mismatched_prediction_pixels': 0}}

Full-split metric parity against unchanged official infer(): {'C0': {'images': 3418, 'mIoU': 0.6733634176835867, 'mDice': 0.8027457407566831, 'max_metric_difference': 0.0}, 'GS': {'images': 3418, 'mIoU': 0.6709182796109426, 'mDice': 0.8008753927808225, 'max_metric_difference': 0.0}, 'RCS': {'images': 3418, 'mIoU': 0.670702799836945, 'mDice': 0.8007490713402556, 'max_metric_difference': 0.0}}

Historical audit C0=67.3104; current same-evaluator C0 difference=+0.0259 pp. The old audit used a different TTA reduction order; native BF16 GPU execution and benchmark algorithm choices can also affect exact values. We do not tune backend settings to recover a historical rounded number. All reported deltas use the C0 evaluated alongside GS/RCS, with full official-function parity.

Metric retains official GT-background overwrite; foreground classes 0–3 enter the mean. Absent-class IoU is NaN/excluded; absent-class Dice is 0. Boundary masks include foreground-to-foreground transitions only. Size bins use per-class 8-connected GT-component area q25/q75; recall is pixel-weighted and size mIoU is mask-restricted, not instance IoU.

| Variant | CAM56 mIoU | CAM28_1 mIoU | CAM28_2 mIoU | CAMdeep mIoU | Final mIoU | Final mDice |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 61.4715 | 67.0349 | 66.5082 | 64.9678 | 67.3363 | 80.2746 |
| GS | 61.3794 | 66.7797 | 66.2851 | 64.7479 | 67.0918 | 80.0875 |
| RCS | 61.3986 | 66.5179 | 66.3369 | 64.8019 | 67.0703 | 80.0749 |

## 5. Boundary, interior, and object size

| Variant | Boundary acc | Boundary mIoU | Interior acc | Interior mIoU | Small recall/mIoU | Medium recall/mIoU | Large recall/mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51.5621 | 31.7292 | 85.6835 | 71.5077 | 35.0735/17.9432 | 68.4054/45.9480 | 89.4318/78.5788 |
| GS | 51.6525 | 31.5530 | 85.6210 | 71.2418 | 35.3621/18.1643 | 68.2576/45.5501 | 89.4218/78.4098 |
| RCS | 51.7174 | 31.5978 | 85.5628 | 71.2100 | 35.7852/18.4049 | 68.2042/45.5527 | 89.3730/78.3968 |

## 6. Per-class IoU

| Variant | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|---:|
| C0 | 76.4484 | 70.5763 | 57.8542 | 64.4664 |
| GS | 76.4236 | 70.6046 | 57.4823 | 63.8568 |
| RCS | 76.3191 | 70.5236 | 57.4569 | 63.9815 |

## 7. q dynamics

q is JS/ln(2), computed at 28x28; these dynamics include all grid positions. Phase1-DD rows are imported observations, not re-trained models.

| Source | Epoch | Mean | Std | Min | p05 | p25 | p50 | p75 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reconstructed-init | 0 | 0.068370 | 0.053279 | 0.000001 | 0.010737 | 0.031040 | 0.055002 | 0.090555 | 0.172147 | 0.601937 |
| Phase0-C0 | 25 | 0.192099 | 0.206927 | 0.000000 | 0.002367 | 0.033056 | 0.115839 | 0.283482 | 0.651629 | 0.999801 |
| GS | 1 | 0.324066 | 0.150833 | 0.000008 | 0.091033 | 0.252536 | 0.311371 | 0.365050 | 0.685674 | 0.971978 |
| GS | 5 | 0.178140 | 0.182068 | 0.000000 | 0.004911 | 0.038916 | 0.114281 | 0.259966 | 0.572782 | 0.998001 |
| GS | 10 | 0.182054 | 0.182386 | 0.000000 | 0.004227 | 0.040675 | 0.121549 | 0.268204 | 0.570508 | 0.998671 |
| GS | 15 | 0.179674 | 0.179882 | 0.000000 | 0.003575 | 0.038458 | 0.120371 | 0.268552 | 0.555339 | 0.997815 |
| GS | 20 | 0.180423 | 0.186824 | 0.000000 | 0.002270 | 0.033519 | 0.116639 | 0.271371 | 0.573725 | 0.998571 |
| GS | 25 | 0.181434 | 0.196241 | 0.000000 | 0.001417 | 0.027441 | 0.110042 | 0.274934 | 0.600073 | 0.999372 |
| RCS | 1 | 0.323979 | 0.150920 | 0.000015 | 0.090637 | 0.252432 | 0.311277 | 0.365047 | 0.686228 | 0.971887 |
| RCS | 5 | 0.178860 | 0.182069 | 0.000000 | 0.005067 | 0.039490 | 0.115335 | 0.260762 | 0.573391 | 0.997839 |
| RCS | 10 | 0.182147 | 0.182038 | 0.000000 | 0.004299 | 0.040949 | 0.122052 | 0.268168 | 0.569486 | 0.998559 |
| RCS | 15 | 0.179122 | 0.180201 | 0.000000 | 0.003529 | 0.038097 | 0.119194 | 0.267394 | 0.556263 | 0.997984 |
| RCS | 20 | 0.179906 | 0.187583 | 0.000000 | 0.002214 | 0.032868 | 0.115215 | 0.270254 | 0.575933 | 0.998543 |
| RCS | 25 | 0.180756 | 0.197189 | 0.000000 | 0.001367 | 0.026681 | 0.108230 | 0.273331 | 0.603208 | 0.999382 |
| Phase1-DD | 1 | 0.328794 | 0.141606 | 0.000064 | 0.097272 | 0.264186 | 0.322489 | 0.370905 | 0.651642 | 0.957629 |
| Phase1-DD | 5 | 0.201243 | 0.219697 | 0.000000 | 0.007639 | 0.046728 | 0.120846 | 0.264012 | 0.757411 | 0.999290 |
| Phase1-DD | 10 | 0.204167 | 0.217057 | 0.000000 | 0.006921 | 0.048268 | 0.126847 | 0.274284 | 0.740190 | 0.998972 |
| Phase1-DD | 15 | 0.209221 | 0.234235 | 0.000000 | 0.005429 | 0.043500 | 0.121273 | 0.276480 | 0.798546 | 0.999740 |
| Phase1-DD | 20 | 0.206919 | 0.233196 | 0.000000 | 0.003955 | 0.039354 | 0.118877 | 0.280349 | 0.789096 | 0.999911 |
| Phase1-DD | 25 | 0.206797 | 0.240753 | 0.000000 | 0.002583 | 0.032111 | 0.111486 | 0.285042 | 0.799433 | 0.999941 |

## 8. Effective context strength

| Variant | Mean reliability | Mean suppression | r p05/p25/p50/p75/p95 | Context RMS before | after | ratio |
|---|---:|---:|---|---:|---:|---:|
| GS | 0.818566 | 0.181434 | 0.721296/0.778046/0.817191/0.861884/0.916722 | 0.447179 | 0.372075 | 0.832050 |
| RCS | 0.819247 | 0.180753 | 0.396780/0.726664/0.891775/0.973322/0.998634 | 0.440834 | 0.379386 | 0.860608 |

## 9. gamma dynamics and compensation

| Variant | Epoch | gamma_context | gamma_veto | Mean r | EffectiveContextScale |
|---|---:|---:|---:|---:|---:|
| Phase0-C0 | 25 | +1.572197 | +0.567998 | 1.000000 | 1.572197 |
| GS | 1 | +0.203889 | +0.165168 | 0.675934 | 0.137815 |
| GS | 5 | +1.354367 | +0.708207 | 0.821860 | 1.113100 |
| GS | 10 | +1.547089 | +0.746523 | 0.817945 | 1.265434 |
| GS | 15 | +1.561313 | +0.733230 | 0.820326 | 1.280786 |
| GS | 20 | +1.561190 | +0.724867 | 0.819577 | 1.279516 |
| GS | 25 | +1.553347 | +0.720936 | 0.818566 | 1.271517 |
| RCS | 1 | +0.203044 | +0.164931 | 0.676021 | 0.137262 |
| RCS | 5 | +1.339128 | +0.711007 | 0.821140 | 1.099612 |
| RCS | 10 | +1.529943 | +0.761328 | 0.817853 | 1.251269 |
| RCS | 15 | +1.541970 | +0.753959 | 0.820878 | 1.265769 |
| RCS | 20 | +1.540607 | +0.748476 | 0.820094 | 1.263443 |
| RCS | 25 | +1.532349 | +0.745409 | 0.819245 | 1.255369 |

## 10. Frozen Phase-0 Top20 / Bottom80

The first evaluation pass was retained for audit but superseded: recomputing Phase-0 populations under the current benchmark/TF32 settings caused small count drift. The final pass reads an immutable cache replayed in a fresh process from original Phase-0 commit 586f402. Every image's four CH-group counts and Top20 count exactly match the archived Phase-0 CSV. No candidate q is used to redefine these groups.

Frozen population verification: `{'Corrected_by_CH': 19934592, 'Still_Wrong': 24262754, 'Harmed_by_CH': 4443224, 'Stable_Correct': 109998775, 'Top20': 31727873}`.

| Variant | Top20 repair/harm/net | Bottom80 repair/harm/net |
|---|---:|---:|
| GS | 0.6597/0.8090/-0.1493 pp | 0.2803/0.3055/-0.0253 pp |
| RCS | 0.9751/1.2571/-0.2820 pp | 0.3312/0.3836/-0.0523 pp |

## 11. Frozen C0 CH-transition groups

Historical group names are retained. Their definition compares raw CAM28_1 to post-HFRM CAM28_1 (CH plus semantic veto), not a CH-only intervention. Repair/harm below compare candidate FINAL predictions against C0 FINAL predictions within these fixed groups; they do not prove isolated CH causality.

| Variant/group | Repair | Harm | Net change |
|---|---:|---:|---:|
| GS/Corrected_by_CH | 0.2737 pp | 1.1072 pp | -0.8334 pp |
| GS/Still_Wrong | 0.5398 pp | 1.1997 pp | -0.6599 pp |
| GS/Harmed_by_CH | 2.7527 pp | 0.4639 pp | +2.2888 pp |
| GS/Stable_Correct | 0.2338 pp | 0.1018 pp | +0.1319 pp |
| RCS/Corrected_by_CH | 0.2946 pp | 1.7367 pp | -1.4420 pp |
| RCS/Still_Wrong | 0.8584 pp | 1.3062 pp | -0.4478 pp |
| RCS/Harmed_by_CH | 4.2323 pp | 0.6285 pp | +3.6038 pp |
| RCS/Stable_Correct | 0.2497 pp | 0.1769 pp | +0.0728 pp |

## 12. Frozen-C0 q-quintile selectivity

All bins are defined from the locked C0, never from candidate q. Prediction bins use full-resolution foreground q; context bins use 28x28 C0 q with nearest-resized foreground masks and separately computed quintiles. They are resolution-specific populations, not identical pixels. Exact thresholds and counts are in the JSON/CSV.

| Variant | Quintile | Mean r | Context RMS before | after | ratio | Accuracy delta vs C0 |
|---|---|---:|---:|---:|---:|---:|
| GS | Q1 | 0.849485 | 0.475080 | 0.408866 | 0.860625 | -0.0142 pp |
| GS | Q2 | 0.835347 | 0.466119 | 0.394934 | 0.847281 | -0.0200 pp |
| GS | Q3 | 0.823234 | 0.452808 | 0.377869 | 0.834502 | -0.0238 pp |
| GS | Q4 | 0.808775 | 0.437197 | 0.358175 | 0.819252 | -0.0430 pp |
| GS | Q5 | 0.783504 | 0.414908 | 0.330244 | 0.795945 | -0.1493 pp |
| RCS | Q1 | 0.992712 | 0.471459 | 0.468202 | 0.993092 | -0.0193 pp |
| RCS | Q2 | 0.956283 | 0.459385 | 0.441249 | 0.960521 | -0.0347 pp |
| RCS | Q3 | 0.884904 | 0.445279 | 0.397753 | 0.893267 | -0.0468 pp |
| RCS | Q4 | 0.758147 | 0.429597 | 0.332004 | 0.772826 | -0.1085 pp |
| RCS | Q5 | 0.517564 | 0.407875 | 0.224888 | 0.551365 | -0.2820 pp |

## 13. Paired image-level bootstrap

10,000 paired image resamples, seed42; sum each resample's confusion matrices and recompute mIoU. This measures validation-image uncertainty conditional on these checkpoints, not uncertainty over training seeds.

| Comparison | Observed delta mIoU | Bootstrap mean | 95% CI |
|---|---:|---:|---:|
| RCS-C0 | -0.2661 pp | -0.2668 pp | [-0.4317, -0.1302] pp |
| RCS-GS | -0.0215 pp | -0.0220 pp | [-0.1091, +0.0823] pp |
| GS-C0 | -0.2445 pp | -0.2448 pp | [-0.4270, -0.0881] pp |

## 14. Preregistered gates

| Gate | Requirement | Result | Pass |
|---|---|---|:---:|
| A | RCS mIoU > C0 and RCS-C0 CI low >= 0 | delta=-0.002661, low=-0.004317 | False |
| B | RCS > GS with nonnegative CI low or CAM28_1+Top20 fallback | delta=-0.000215, low=-0.001091, fallback=False | False |
| C | RCS CAM28_1 >= C0, interior >= -0.10 pp, large mIoU >= -0.20 pp | CAM=-0.005170, interior=-0.001207, large=-0.001820 | False |
| D | RCS Harmed-by-CH > 0 and > GS; Stable-Correct >= -0.10 pp | RCS_harmed=+0.036038, GS_harmed=+0.022888, stable=+0.000728 | True |

## 15. Scientific interpretation

The semantic-safety gate fails: even receiver-only context suppression damages CAM28_1, interior, or large-region behavior under the frozen thresholds. Failed gates: A, B, C. No post-hoc transformation or tuning is permitted.

### 机制解释与边界

RCS 的 CAM28_1 相对 C0 改变 -0.5170 pp；interior accuracy 改变 -0.1207 pp；large-object mIoU 改变 -0.1820 pp。三项门槛分别为 ≥0、≥−0.10、≥−0.20 pp；不能因为其他局部指标改善而放宽。

高风险 Top20 的净修复：RCS -0.2820 pp，GS -0.1493 pp。需结合 Harmed-by-CH 与 Corrected-by-CH 两类的收益/代价理解，不能只报告前者。

空间选择性是否被实际执行，可以由 Q1→Q5 的 reliability 与 context RMS 比值判断；存在明显的空间抑制不等于存在 segmentation utility。原始 F 路径完全保留，也不保证经过完整训练后语义表现不会下降。

GS 与 RCS 只有在相同 q 上才严格均值匹配。独立训练后 q 和 context 能量分布会变化，因此须同时查看 mean(r)、RMS ratio 和 gamma，而不能把两组称为最终能量严格匹配。EffectiveContextScale 是 abs(gamma_context)×mean(r) 的标量代理，不是实际残差 RMS。

判定顺序保持预注册语义安全优先：若 C FAIL，即便 D PASS，也输出 CONTEXT_SUPPRESSION_SEMANTIC_DAMAGE，不改判为可推进的局部成功。本轮不会提出或运行事后 alpha、阈值、温度、kernel 或 stage 搜索。

## 16. Engineering and artifact record

- Main final-checkpoint evaluation: 2.95 min; complete evaluation including dynamics/bootstrap: 6.88 min; peak CUDA memory 3.155 GiB.
- GS/RCS training runtime: 45.21 / 45.28 min.
- All required curves, q/context/gamma, fixed-strata, CH, quintile, per-class, bootstrap, optimizer, per-image, and summary artifacts were generated.
- No BCSS test, LUAD, best-epoch selection, or post-hoc tuning was used.
- Runtime environment: {'python': '3.10.20 (main, Jun 11 2026, 15:17:37) [GCC 14.3.0]', 'torch': '2.11.0+cu128', 'numpy': '1.23.5', 'gpu': 'NVIDIA GeForce RTX 5090 D v2'}

## 17. Epoch0 initialization observation

Retrospective seed42/pretrained reconstruction, eval mode, validation images, batch20 BF16; zero training steps

This reconstructs initialization after training has finished; it is not a contemporaneous training log. Shared raw features and q are computed once, then the frozen GS/RCS context scaling is applied. Initial gammas are zero, so attenuated context does not yet contribute to the output.

| Variant | Mean r | Mean suppression | Context RMS before | after | ratio |
|---|---:|---:|---:|---:|---:|
| GS | 0.931630 | 0.068370 | 0.030606 | 0.028517 | 0.931764 |
| RCS | 0.931630 | 0.068370 | 0.030606 | 0.028479 | 0.930509 |

DECISION = CONTEXT_SUPPRESSION_SEMANTIC_DAMAGE
