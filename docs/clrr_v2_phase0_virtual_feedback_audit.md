# CLRR-v2 Phase-0 Virtual Feedback Audit

## 1. Executive conclusion

The frozen A0 virtual-feedback audit completed on all 3,418 BCSS validation
images with no training and no test-set access. The analytical correction is
finite, bounded, and a valid classifier-consistent descent direction. It also
improves all three individual stage CAM mIoUs. However, the preregistered
Phase-0A consensus-recoverability target fails: `recoverable > harmful` holds
for only one of three target stages, rather than at least two.

Accordingly, the overall frozen decision is **NOGO**. Per the technical
specification, no CLRR-v2 training-model code, 20-step readiness run, or
25-epoch experiment is authorized.

## 2. Isolation and protocol

- Base: official SSHR A0 commit
  `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9` (PR #1).
- Branch base: `baseline/official-a0`; the branch does not contain HST,
  FA-MPR, SC-MPR, or CDSR.
- Dataset/split: BCSS validation only, 3,418 images and 3,418 masks.
- A0 seed42 FINAL checkpoint:
  `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`.
- Checkpoint SHA256:
  `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Feedback: detached FP32, foreground softmax, reliability-weighted
  leave-one-out consensus, fixed `eta=0.05`.
- Official prediction: unchanged BCSS thresholds, three-way TTA, fusion
  `0.6 CAM28_1 + 0.2 CAM28_2 + 0.2 CAMdeep`, and `iouutils` metric.
- Environment: Python 3.10.20, PyTorch 2.11.0+cu128, CUDA 12.8,
  cuDNN 9.19.0, NVIDIA GeForce RTX 5090 D v2.
- Training performed: none. Test evaluated: false.

Exact audit command:

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python -u \
  tools/audit_clrr_v2_phase0.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --output-json /home/duyanhong/experiments/CLRR_V2_PHASE0_A0_SEED42_RETRY2/phase0_result.json \
  --num-workers 4
```

The tool accepts no test-root argument and exposes no eta option.

## 3. Phase -1 classifier linearity proof

| Target state | Reused classifier | Weight shape | Bias | Intervening trainable nonlinearity |
|---|---|---|---|---|
| F56 HFRM output | `ic_56` | `[4,256,1,1]` | `[4]` | none |
| F28_1 HFRM output | `ic1` | `[4,512,1,1]` | `[4]` | none |
| F28_2 HFRM output | `ic2` | `[4,1024,1,1]` | `[4]` | none |

Each raw stage logit is exactly `A_i = W_i H_i + b_i`. A direct server-side
comparison showed that the audit extractor and official `forward_cam()`
produce elementwise-identical Pass0 CAM tensors. The post-classifier ReLU is
used only for the released nonnegative inference CAM and is not part of the
feedback direction.

## 4. Official Pass0 parity

| Evaluation path | mIoU (%) | mDice (%) |
|---|---:|---:|
| Released `infer()` | 67.3279 | 80.2680 |
| Independent Phase-0 Pass0 reconstruction | 67.3279 | 80.2680 |
| Absolute difference | **0.0000** | **0.0000** |

Strict parity passed. The direct checkpoint score is 0.0177 mIoU points above
the separately recorded A0 reference 67.3102; all virtual deltas below use the
same-checkpoint Pass0 score, so this small reference difference cannot bias
the Phase-0 comparison.

## 5. Phase 0A — consensus recoverability

Diagnostics use the identity view at each stage's native resolution and only
pixels whose resized GT is one of the four foreground classes.

| Stage | Recoverable | Harmful | Net | Net rate | Stage foreground mIoU | Consensus foreground mIoU | Target pass? |
|---|---:|---:|---:|---:|---:|---:|---|
| F56 | 1,281,326 | 556,259 | +725,067 | +7.3124% | 50.0263% | 63.4638% | yes |
| F28_1 | 85,955 | 117,560 | -31,605 | -1.2748% | 63.7895% | 62.1274% | no |
| F28_2 | 97,476 | 97,938 | -462 | -0.0186% | 63.5753% | 62.5693% | no |

Only 1/3 stages has `recoverable > harmful`; the preregistered target requires
at least 2/3. F28_2 is nearly neutral, but it is still not positive, while the
F28_1 teacher consensus is materially worse than the current stage.

Per-class recoverable/harmful counts:

| Stage | Class | Recoverable | Harmful | Net |
|---|---:|---:|---:|---:|
| F56 | 0 | 196,247 | 292,736 | -96,489 |
| F56 | 1 | 702,954 | 157,180 | +545,774 |
| F56 | 2 | 152,420 | 91,270 | +61,150 |
| F56 | 3 | 229,705 | 15,073 | +214,632 |
| F28_1 | 0 | 19,272 | 42,346 | -23,074 |
| F28_1 | 1 | 36,257 | 55,596 | -19,339 |
| F28_1 | 2 | 16,313 | 15,973 | +340 |
| F28_1 | 3 | 14,113 | 3,645 | +10,468 |
| F28_2 | 0 | 46,781 | 20,826 | +25,955 |
| F28_2 | 1 | 34,080 | 49,803 | -15,723 |
| F28_2 | 2 | 14,507 | 15,873 | -1,366 |
| F28_2 | 3 | 2,108 | 11,436 | -9,328 |

## 6. Phase 0B — backprojection direction sanity

| Stage | Mean CE delta | Median CE delta | CE-decrease fraction | Direction gate |
|---|---:|---:|---:|---|
| F56 | -0.151206 | -0.004311 | 86.2820% | pass |
| F28_1 | -0.035048 | -0.001190 | 76.5278% | pass |
| F28_2 | -0.043358 | -0.000924 | 68.5309% | mean pass; fraction below 70% |

All three mean CE deltas are negative, and 2/3 stages exceed the required 70%
CE-decrease fraction. The directional gate passes.

The unit test independently compares `W^T(Pbar-P)` with the negative autograd
gradient of consensus CE on a toy tensor and requires cosine similarity above
0.999999.

## 7. Phase 0C — virtual segmentation utility

| Prediction | Pass0 mIoU | Virtual mIoU | Delta mIoU | Pass0 mDice | Virtual mDice | Delta mDice |
|---|---:|---:|---:|---:|---:|---:|
| CAM56 | 61.4651 | 63.0888 | **+1.6237** | 75.7237 | 77.0298 | +1.3061 |
| CAM28_1 | 67.0276 | 67.0613 | **+0.0337** | 80.0461 | 80.0707 | +0.0246 |
| CAM28_2 | 66.4982 | 66.5812 | **+0.0830** | 79.6982 | 79.7572 | +0.0589 |
| CAMdeep | 64.9608 | 64.9608 | 0.0000 | 78.5478 | 78.5478 | 0.0000 |
| Official fused | 67.3279 | 67.3204 | **-0.0075** | 80.2680 | 80.2622 | -0.0057 |

All 3/3 target-stage mIoUs are nondecreasing. The fused delta is above the
specified harm floor of -0.05 percentage points, but it is not a gain.

Per-class IoU change in percentage points:

| Prediction | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|---:|
| CAM56 | +1.0767 | +0.7303 | +0.2916 | +4.3961 |
| CAM28_1 | +0.0330 | -0.0188 | +0.0025 | +0.1182 |
| CAM28_2 | +0.1244 | +0.1312 | +0.0971 | -0.0206 |
| Official fused | -0.0006 | -0.0198 | -0.0285 | +0.0189 |

Corrected/harmed foreground pixels after official TTA and mask conversion:

| Prediction | Corrected | Harmed | Net corrected | Net rate | Prediction-change rate |
|---|---:|---:|---:|---:|---:|
| CAM56 | 1,768,665 | 664,672 | +1,103,993 | +0.6959% | 1.6647% |
| CAM28_1 | 326,939 | 317,200 | +9,739 | +0.0061% | 0.4335% |
| CAM28_2 | 339,314 | 212,184 | +127,130 | +0.0801% | 0.3687% |
| Official fused | 192,025 | 205,741 | -13,716 | -0.0086% | 0.2675% |

All three target stages have positive net correction. The official fusion has
a small negative net correction, consistent with its -0.0075 mIoU delta.

## 8. Stability, bounds, and resources

| Stage | Mean update/feature RMS | P99 | Maximum | Mean abs probability change | Argmax change |
|---|---:|---:|---:|---:|---:|
| F56 | 0.8355% | 4.6623% | 5.0000% | 1.6684% | 4.6246% |
| F28_1 | 0.3353% | 2.5821% | 4.9872% | 0.4175% | 1.0411% |
| F28_2 | 0.3424% | 2.6029% | 4.9937% | 0.4260% | 1.0183% |

- Every probability, consensus, residual, backprojection, update, and virtual
  output was finite.
- The maximum local update ratio remained at or below the fixed 5% bound for
  every stage.
- Official inference: 49.31 s; virtual audit: 134.02 s.
- Peak CUDA allocated/reserved: 0.78/0.85 GiB.
- Local tests: 6/6 passed; server tests: 6/6 passed.

## 9. Frozen gate decision

| Preregistered check | Required | Observed | Result |
|---|---|---|---|
| Consensus recoverability | positive in at least 2/3 stages | positive in 1/3 | **fail** |
| Mean consensus CE delta | negative in all stages | negative in 3/3 | pass |
| CE-decrease fraction | at least 70% in at least 2/3 | 2/3 | pass |
| Stage CAM mIoU | nondecreasing in at least 2/3 | 3/3 | pass |
| Stage net correction | positive in at least 2/3 | 3/3 | pass |
| Fused mIoU harm floor | greater than -0.05 pp | -0.0075 pp | pass |
| Finite/update bound | all finite and at most 5% | passed | pass |

The specification separately names the Phase-0A target
`recoverable > harmful in at least 2/3 stages` and forbids changing eta,
reliability, consensus weights, or normalization to rescue Phase 0. That target
is therefore treated as a frozen gate rather than discarded after observing
the otherwise favorable virtual-update results.

## 10. Delivery state

Implemented and retained:

- isolated Phase -1 structure report;
- frozen analytical audit functions under `tools/`;
- six unit/integration tests;
- full BCSS validation raw evidence under `audit/results/`.

Intentionally not implemented:

- no `network/clrr/` training modules;
- no CLRR CLI changes to `train_sshr.py`;
- no loss, optimizer, inference, threshold, or metric changes;
- no 20-step training readiness;
- no 25-epoch run, test evaluation, LUAD run, or ablation.

CLRR_V2_SIGNAL_NOGO
