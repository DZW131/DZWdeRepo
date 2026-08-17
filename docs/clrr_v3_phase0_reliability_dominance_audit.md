# CLRR-v3 Phase-0 Reliability-Dominance Audit

## 1. Executive conclusion

The final frozen-signal audit completed on all 3,418 BCSS validation images.
No training was performed and the test split was not accessed.

Reliability dominance fixes the localized teacher-eligibility failure observed
in CLRR-v2: all three stages have positive teacher advantage on active pixels,
all three have negative mean consensus-CE deltas, all three stage CAM mIoUs
increase, and all three stage predictions have positive net correction.
However, this mechanism does not produce a material improvement in the exact
official fused prediction. Fused mIoU changes from 67.3279% to 67.3321%, only
**+0.0042 percentage points**, below the frozen **+0.05 pp** hard gate.

The preregistered decision is therefore **NOGO**. No CLRR training module,
training-readiness run, test evaluation, or v4 reliability-gate patch is
authorized. The hierarchy-consensus backprojection route stops here.

## 2. Frozen scope and reproducibility

- Repository base: official SSHR A0 commit
  `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- Dataset/split: BCSS validation only, 3,418 images and 3,418 masks.
- A0 seed42 FINAL checkpoint:
  `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`.
- Checkpoint SHA256:
  `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Checkpoint size: 451,130,207 bytes.
- Fixed eta: 0.05; the tool exposes no eta argument.
- Precision: official BF16 model forward; detached FP32 analytical feedback.
- Official inference: unchanged TTA, presence thresholds, CAM normalization,
  `0.6 CAM28_1 + 0.2 CAM28_2 + 0.2 CAMdeep`, and official `iouutils`.
- Test evaluated: false. Training performed: false.

Exact audit command:

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python -u \
  tools/audit_clrr_v3_phase0.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --output-json /home/duyanhong/experiments/CLRR_V3_PHASE0_A0_SEED42/clrr_v3_phase0_summary.json \
  --num-workers 4
```

The same A0 forward produces Pass0, v2, and v3 probes. Consequently, the
v2/v3 comparison cannot be affected by checkpoint, data-order, TTA, class
presence, or hardware differences.

## 3. Formula and isolation audit

The v2 consensus, semantic residual, classifier backprojection, RMS
normalization, feature scale, maturity gate, and fixed eta remain unchanged.
The only v3 mechanism change is:

```text
r_i   = 1 - H(P_i) / log(C)
rho_i = 1 - H(Pbar_i) / log(C)
d_i   = relu(rho_i - r_i)

DeltaH_i_v2 = kappa_i * rho_i * m_i * s_i * Bhat_i
DeltaH_i_v3 = kappa_i * d_i   * m_i * s_i * Bhat_i
```

`rho_i` is not multiplied into the v3 correction again. Unit tests explicitly
verify the exact gate equation, the zero-update behavior when `rho_i <= r_i`,
the absence of an extra rho multiplier, detached state, the 5% update bound,
the official metric algebra, and the frozen decision thresholds.

No file under `network/` and no training, inference, loss, optimizer, or
metric source was modified. The branch contains only audit tools, tests,
evidence, and documentation on top of official A0.

## 4. Official Pass0 parity

| Evaluation path | mIoU (%) | mDice (%) |
|---|---:|---:|
| Released official `infer()` | 67.3279 | 80.2680 |
| Independent joint-audit Pass0 | 67.3279 | 80.2680 |
| Absolute difference | **0.0000** | **0.0000** |

Strict parity passed exactly.

## 5. Phase-0A — reliability-dominance validity

Teacher outcomes are evaluated at native stage resolution on foreground
pixels for which `d_i > 0`.

| Stage | Active pixels | Active foreground | d mean | d std | d p50 | d p90 | Teacher win | Teacher loss | Net | Net rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F56 | 62.8963% | 63.2619% | 0.1825 | 0.2021 | 0.0993 | 0.4938 | 890,675 | 240,610 | +650,065 | +10.3633% |
| F28_1 | 15.4562% | 15.2364% | 0.1177 | 0.1278 | 0.0718 | 0.3086 | 35,507 | 31,327 | +4,180 | +1.1066% |
| F28_2 | 11.5998% | 10.9178% | 0.1282 | 0.1352 | 0.0812 | 0.3309 | 32,175 | 22,614 | +9,561 | +3.5324% |

Teacher win exceeds teacher loss for **3/3 stages**. No stage has negative
overall teacher advantage, so the Phase-0A gate passes.

Additional outcome counts:

| Stage | Both correct | Both wrong |
|---|---:|---:|
| F56 | 4,416,206 | 725,255 |
| F28_1 | 211,928 | 98,969 |
| F28_2 | 155,085 | 60,795 |

Per-class teacher advantage is diagnostic only:

| Stage | Class | Active | Win | Loss | Net | Net rate |
|---|---:|---:|---:|---:|---:|---:|
| F56 | 0 | 2,402,930 | 147,495 | 103,662 | +43,833 | +1.8241% |
| F56 | 1 | 2,631,217 | 458,246 | 82,883 | +375,363 | +14.2658% |
| F56 | 2 | 790,885 | 104,377 | 43,357 | +61,020 | +7.7154% |
| F56 | 3 | 447,714 | 180,557 | 10,708 | +169,849 | +37.9369% |
| F28_1 | 0 | 135,786 | 8,961 | 10,161 | -1,200 | -0.8837% |
| F28_1 | 1 | 150,909 | 12,790 | 15,777 | -2,987 | -1.9793% |
| F28_1 | 2 | 58,470 | 7,410 | 4,196 | +3,214 | +5.4968% |
| F28_1 | 3 | 32,566 | 6,346 | 1,193 | +5,153 | +15.8233% |
| F28_2 | 0 | 116,973 | 17,459 | 4,067 | +13,392 | +11.4488% |
| F28_2 | 1 | 98,814 | 9,313 | 12,679 | -3,366 | -3.4064% |
| F28_2 | 2 | 42,280 | 4,865 | 3,309 | +1,556 | +3.6802% |
| F28_2 | 3 | 12,602 | 538 | 2,559 | -2,021 | -16.0371% |

The per-class negatives show that entropy dominance is not uniformly
class-calibrated, but no class-specific rule was introduced or selected.

## 6. Phase-0B — backprojection direction sanity

The decision statistics use active pixels with nontrivial mismatch.

| Stage | Active nontrivial pixels | Mean CE delta | Median CE delta | CE-decrease fraction |
|---|---:|---:|---:|---:|
| F56 | 6,711,843 | -0.061690 | -0.000738 | 84.3756% |
| F28_1 | 411,397 | -0.021788 | -0.000990 | 76.6287% |
| F28_2 | 309,523 | -0.030316 | -0.001662 | 83.5602% |

All three mean CE deltas are negative and all three CE-decrease fractions
exceed 70%; the Phase-0B gate passes.

Inactive/nontrivial pixels are reported as a numerical control:

| Stage | Stratum | Pixels | Mean CE delta | Median | CE-decrease fraction |
|---|---|---:|---:|---:|---:|
| F56 | active | 6,711,843 | -0.061690 | -0.000738 | 84.3756% |
| F56 | inactive | 3,962,092 | +0.001868 | +0.000123 | 29.2178% |
| F28_1 | active | 411,397 | -0.021788 | -0.000990 | 76.6287% |
| F28_1 | inactive | 2,253,173 | -0.000181 | 0.000000 | 45.8343% |
| F28_2 | active | 309,523 | -0.030316 | -0.001662 | 83.5602% |
| F28_2 | inactive | 2,355,590 | +0.000724 | +0.000014 | 27.0129% |

The tiny inactive changes arise because the analytical probe re-evaluates the
classifier in FP32 while Pass0 logits originate from the official BF16
autocast forward. They are not semantic feedback and are not used by the
direction gate. This is also why the final +0.0042 pp fused change should not
be interpreted as a meaningful improvement.

## 7. Phase-0C — frozen virtual segmentation utility

| Output | Pass0 mIoU | v3 mIoU | Delta pp | Pass0 mDice | v3 mDice | Delta pp |
|---|---:|---:|---:|---:|---:|---:|
| CAM56 | 61.4651 | 62.2583 | +0.7932 | 75.7237 | 76.3721 | +0.6484 |
| CAM28_1 | 67.0276 | 67.0451 | +0.0175 | 80.0461 | 80.0591 | +0.0130 |
| CAM28_2 | 66.4982 | 66.5239 | +0.0257 | 79.6982 | 79.7169 | +0.0187 |
| CAMdeep | 64.9608 | 64.9608 | 0.0000 | 78.5478 | 78.5478 | 0.0000 |
| Official fused | 67.3279 | 67.3321 | **+0.0042** | 80.2680 | 80.2712 | +0.0032 |

Stage utility is positive for 3/3 target stages. Prediction-change counts are:

| Output | Corrected | Harmed | Net | Prediction change |
|---|---:|---:|---:|---:|
| CAM56 | 710,426 | 196,578 | +513,848 | 0.6205% |
| CAM28_1 | 62,891 | 49,447 | +13,444 | 0.0767% |
| CAM28_2 | 67,173 | 31,120 | +36,053 | 0.0677% |
| Official fused | 32,296 | 28,859 | **+3,437** | 0.0422% |

All three target stages have positive net correction, but almost none of this
stage-level gain survives the frozen official normalization and fusion.

Per-class IoU comparison:

| Output | Class | Pass0 IoU | v3 IoU | Delta pp |
|---|---:|---:|---:|---:|
| CAM56 | 0 | 73.0720 | 73.5626 | +0.4907 |
| CAM56 | 1 | 67.9895 | 68.3013 | +0.3118 |
| CAM56 | 2 | 53.8312 | 54.0279 | +0.1966 |
| CAM56 | 3 | 50.9676 | 53.1412 | +2.1736 |
| CAM28_1 | 0 | 76.2781 | 76.2894 | +0.0113 |
| CAM28_1 | 1 | 70.4882 | 70.4959 | +0.0078 |
| CAM28_1 | 2 | 57.5511 | 57.5780 | +0.0269 |
| CAM28_1 | 3 | 63.7928 | 63.8168 | +0.0240 |
| CAM28_2 | 0 | 74.7899 | 74.8206 | +0.0307 |
| CAM28_2 | 1 | 69.4874 | 69.5228 | +0.0354 |
| CAM28_2 | 2 | 57.4119 | 57.4518 | +0.0399 |
| CAM28_2 | 3 | 64.3036 | 64.3004 | -0.0031 |
| CAMdeep | 0 | 73.6674 | 73.6674 | 0.0000 |
| CAMdeep | 1 | 68.5058 | 68.5058 | 0.0000 |
| CAMdeep | 2 | 55.3393 | 55.3393 | 0.0000 |
| CAMdeep | 3 | 62.3307 | 62.3307 | 0.0000 |
| Official fused | 0 | 76.4491 | 76.4497 | +0.0006 |
| Official fused | 1 | 70.5718 | 70.5744 | +0.0026 |
| Official fused | 2 | 57.8268 | 57.8385 | +0.0117 |
| Official fused | 3 | 64.4640 | 64.4658 | +0.0018 |

## 8. Direct CLRR-v2 versus CLRR-v3 comparison

The joint audit exactly reproduces the previous frozen v2 deltas, including
the -0.007525 pp fused result and -13,716 fused net correction.

| Metric | CLRR-v2 | CLRR-v3 |
|---|---:|---:|
| Active correction fraction F56 | 100.0000% | 62.8963% |
| Active correction fraction F28_1 | 100.0000% | 15.4562% |
| Active correction fraction F28_2 | 100.0000% | 11.5998% |
| F56 delta mIoU | +1.6237 pp | +0.7932 pp |
| F28_1 delta mIoU | +0.0337 pp | +0.0175 pp |
| F28_2 delta mIoU | +0.0830 pp | +0.0257 pp |
| Fused delta mIoU | -0.0075 pp | **+0.0042 pp** |
| Fused corrected - harmed | -13,716 | **+3,437** |
| CE-decrease fraction F56 | 86.2820% | 84.3756% |
| CE-decrease fraction F28_1 | 76.5278% | 76.6287% |
| CE-decrease fraction F28_2 | 68.5309% | 83.5602% |

Reliability dominance removes many low-authority corrections and changes the
fused correction sign from negative to positive. This is useful mechanism
evidence, but the resulting fused magnitude is approximately twelve times
smaller than the preregistered +0.05 pp minimum.

## 9. Stability and resources

| Stage | Mean update/feature RMS | p99 | Maximum | Bound |
|---|---:|---:|---:|---:|
| F56 | 0.2358% | 2.4756% | 4.2749% | pass |
| F28_1 | 0.0289% | 0.7837% | 3.8259% | pass |
| F28_2 | 0.0245% | 0.7513% | 3.4227% | pass |

- All tensors are finite.
- Maximum local update ratio is 4.2749%, below 5% plus numerical tolerance.
- Feedback tensors are detached FP32.
- Classifier structure and the official model source are unchanged.
- Python 3.10.20; PyTorch 2.11.0+cu128; CUDA 12.8; cuDNN 9.19.0.
- GPU: NVIDIA GeForce RTX 5090 D v2.
- Official inference runtime: 49.34 seconds.
- Joint v2/v3 audit runtime: 226.90 seconds.
- Peak CUDA allocated/reserved: 0.820/0.895 GiB.
- Local tests: 10/10 passed. Server tests: 10/10 passed.

## 10. Frozen Go/No-Go decision

| Gate | Requirement | Result | Status |
|---|---|---:|---|
| Phase-0A | teacher win > loss in at least 2/3 stages | 3/3 | pass |
| Phase-0B mean | negative mean CE delta in 3/3 | 3/3 | pass |
| Phase-0B fraction | at least 70% in at least 2/3 | 3/3 | pass |
| C1 stage utility | positive delta mIoU in at least 2/3 | 3/3 | pass |
| C2 stage net correction | positive in at least 2/3 | 3/3 | pass |
| C3 official fused | delta mIoU at least +0.05 pp | **+0.0042 pp** | **fail** |
| Stability | finite and max update ratio at most 5% | 4.2749% | pass |

The only failed hard gate is the decisive official-fusion requirement. Under
the frozen specification, that failure is sufficient for NOGO. Validation was
not used to tune eta, consensus, temperature, thresholds, class rules, stage
rules, or fusion.

CLRR_V3_SIGNAL_NOGO
