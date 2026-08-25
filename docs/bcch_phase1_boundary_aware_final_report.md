# BCCH Phase-1 Boundary-Aware CH Mechanism Validation

## 1. Frozen protocol

- Experiment: `EXP-BCCH-001`; BCSS seed42; Epoch20 common state → Epoch21–25 matched continuation.
- C0/W1 are reused SHA-locked matched artifacts; only BC-CH is newly trained.
- Only HFRM28_1 CH changes; original CH15 parameter and optimizer state restore exactly.
- `E_HF=sqrt(LH²+HL²+HH²)` → channel mean → per-image spatial min-max → bilinear upsample; `B` is detached and `alpha=1-B`.
- No new trainable parameter, classifier, GSR change, loss, contrastive objective, inference change or metric change.
- Batch20, 224×224, BF16, same batch/augmentation/model seeds, official optimizer/poly schedule and Epoch25 FINAL only.
- No test, LUAD, other seed, best-checkpoint selection or validation tuning.

## 2. Overall, spatial and object-size results

The preregistered gates use the exact earlier WD-CH boundary/interior pixel-accuracy definition (`≤7 px` / `≥8 px`). Zone-restricted mIoU is additionally reported so it is not mislabeled as accuracy.

| Variant | mIoU | Δ pp | mDice | Boundary acc. | Δ pp | Interior acc. | Δ pp | Boundary mIoU | Interior mIoU | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 66.8555 | +0.0000 | 79.9194 | 51.6894 | +0.0000 | 85.5057 | +0.0000 | 31.9958 | 70.9137 | 36.6083 | 68.5264 | 89.1509 |
| W1 | 66.8522 | -0.0034 | 79.9200 | 52.1305 | +0.4411 | 85.4218 | -0.0839 | 32.2440 | 70.8667 | 38.0735 | 68.4510 | 89.1155 |
| BC-CH | 66.8429 | -0.0127 | 79.9135 | 51.8666 | +0.1772 | 85.4398 | -0.0658 | 32.2430 | 70.8697 | 38.0645 | 68.2999 | 89.1736 |

## 3. CAM hierarchy

| Variant | CAM56 | CAM28_1 | Δ CAM28_1 pp | CAM28_2 | Deep | Final |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 61.0919 | 66.4431 | +0.0000 | 66.2999 | 64.5274 | 66.8555 |
| W1 | 61.2428 | 65.8860 | -0.5571 | 66.4122 | 64.5719 | 66.8522 |
| BC-CH | 61.0962 | 66.1422 | -0.3009 | 66.3233 | 64.5275 | 66.8429 |

## 4. Per-class final IoU

| Variant | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|---:|
| C0 | 76.5672 | 70.1931 | 57.9787 | 62.6833 |
| W1 | 76.4294 | 70.1589 | 57.8720 | 62.9485 |
| BC-CH | 76.4602 | 70.1480 | 57.9712 | 62.7921 |

## 5. Feature mechanism statistics

Residual RMS values are normalized by input-feature RMS. Boundary/interior residuals use continuous `B` / `1-B` weighting and therefore require no post-hoc threshold.

- Raw CH residual RMS: 1.214070 ± 0.179403.
- Selected BC-CH residual RMS: 0.905173 ± 0.170195.
- Boundary selected residual RMS: 0.770730 ± 0.121719; retention=0.619338.
- Interior selected residual RMS: 0.950100 ± 0.183195; retention=0.786463.
- Boundary map mean/std: 0.270733 / 0.176620.
- Alpha mean/std: 0.729267 / 0.176620.
- Final gamma_context / gamma_veto: 1.57423520 / 0.60462594.

## 6. Preregistered gates

| Gate | Observed Δ | Criterion | Result |
|---|---:|---:|:---:|
| A_boundary | +0.1772 pp | > 0.0 pp | PASS |
| B_interior | -0.0658 pp | > -0.2 pp | PASS |
| C_overall | -0.0127 pp | > 0.0 pp | FAIL |

## 7. Engineering and reproducibility evidence

- Preflight: `BCCH_PREFLIGHT_PASS`; real batch=20, BF16, official loss, no optimizer step.
- Training implementation commit: `f2a4c14af3657ea2b8842479effc6a12b85d3864`.
- Common Epoch20 SHA256: `2aae7e7c83373a4bb8865084ede86ba91a79ae3788b732b19fa478ee6c4311fb`.
- Schedule SHA256: `fa648405f40852e98f3d73776b7feee904bd59309ea1df2a97255650b0d00eea`.
- C0 Epoch25 SHA256: `44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8`.
- W1 Epoch25 SHA256: `31976d27e5670256bd08565bb8b34efb510442e4afc0651a7797fee68d88b7fc`.
- BC-CH Epoch25 SHA256: `959ba77c16e440a8e69ece8740238f03f2711f4ea0faca737c33c4b92131f2ad`.
- Prediction order and GT masks are byte-equal across C0/W1/BC-CH.

## 8. Mechanism interpretation

- The router is active: boundary-weighted CH residual retention is 0.6193, versus 0.7865 in interiors. It therefore suppresses contextual mixing more strongly where HF energy is high.
- BC-CH recovers 0.2562 pp (45.99%) of W1's CAM28_1 loss, but CAM28_1 remains -0.3009 pp below C0.
- Boundary accuracy improves +0.1772 pp and zone-restricted Boundary mIoU improves +0.2472 pp, so the structural signal is consistent across both spatial definitions.
- Object-size deltas versus C0 are Small +1.4562 pp, Medium -0.2265 pp and Large +0.0227 pp. The gain remains concentrated in small structures.
- Final mIoU is -0.0127 pp versus C0. Thus the fixed router validates selective boundary suppression, but does not supply enough semantic discrimination for positive overall utility.

## 9. Decision

BC-CH preserves a boundary mechanism signal without positive overall utility.

NEXT_STEP: Semantic discrimination is still required; design contrastive affinity learning as a new experiment.

DECISION = NEXT_STEP

STOP.
