# FDHR Phase-3 Cross-Band Interaction Utility Gate

## 1. Frozen protocol

- Experiment: `EXP-FDHR-003`.
- BCSS seed42; every model starts from the identical locked SSHR Epoch 20 state and follows the identical Epoch 21–25 schedule.
- Batch20, 224×224, BF16, official loss/optimizer/poly schedule, augmentation, inference and metric.
- C0 and W1 are reused SHA-locked matched artifacts; only A/B/C were newly continued.
- Primary result: Epoch 25 FINAL only. No test, LUAD, other seed, best-checkpoint selection or tuning.
- Only HFRM28_1 changes. Fixed strengths are α=β=γ=0.1 and are non-trainable buffers.
- Variant C resolves `Pool(HF)` as arithmetic mean over the LH/HL/HH band axis. The Haar bands already have LL spatial resolution, so no second spatial downsampling is applied.

## 2. Overall, spatial and object-size metrics

| Variant | mIoU | Δ mIoU pp | mDice | Boundary | Δ Boundary pp | Interior | Δ Interior pp | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 66.8555 | +0.0000 | 79.9194 | 51.6894 | +0.0000 | 85.5057 | +0.0000 | 36.6083 | 68.5264 | 89.1509 |
| W1 | 66.8522 | -0.0034 | 79.9200 | 52.1305 | +0.4411 | 85.4218 | -0.0839 | 38.0735 | 68.4510 | 89.1155 |
| A | 66.8649 | +0.0093 | 79.9296 | 52.1391 | +0.4496 | 85.4179 | -0.0877 | 38.1745 | 68.4474 | 89.1124 |
| B | 66.8925 | +0.0370 | 79.9493 | 52.1387 | +0.4493 | 85.4470 | -0.0586 | 38.0321 | 68.5225 | 89.1193 |
| C | 66.8246 | -0.0309 | 79.8992 | 52.1279 | +0.4385 | 85.4174 | -0.0883 | 37.9847 | 68.4592 | 89.1064 |

## 3. CAM hierarchy

| Variant | CAM56 | CAM28_1 | Δ CAM28_1 pp | CAM28_2 | Deep CAM | Final |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 61.0919 | 66.4431 | +0.0000 | 66.2999 | 64.5274 | 66.8555 |
| W1 | 61.2428 | 65.8860 | -0.5571 | 66.4122 | 64.5719 | 66.8522 |
| A | 61.2503 | 65.8606 | -0.5825 | 66.4293 | 64.5996 | 66.8649 |
| B | 61.2479 | 65.9664 | -0.4767 | 66.4225 | 64.5989 | 66.8925 |
| C | 61.2212 | 65.8672 | -0.5759 | 66.3694 | 64.5474 | 66.8246 |

## 4. Per-class final IoU

| Variant | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|---:|
| C0 | 76.5672 | 70.1931 | 57.9787 | 62.6833 |
| W1 | 76.4294 | 70.1589 | 57.8720 | 62.9485 |
| A | 76.4202 | 70.1547 | 57.8481 | 63.0365 |
| B | 76.4649 | 70.1866 | 57.9031 | 63.0155 |
| C | 76.4363 | 70.1522 | 57.8834 | 62.8265 |

## 5. Frequency and interaction statistics

Definitions: `E_LL = mean(LL²)`, `E_HF = mean(LH²+HL²+HH²)`, and interaction magnitude is the reconstructed feature-domain RMS of the added cross-band term.

| Variant | E_LL mean±std | E_HF mean±std | Interaction RMS mean±std | Interaction/Input RMS mean±std |
|---|---:|---:|---:|---:|
| A | 0.434141±0.120198 | 0.062767±0.010453 | 0.012690±0.001054 | 0.036525±0.002669 |
| B | 0.425459±0.116854 | 0.062084±0.010283 | 0.016662±0.007981 | 0.047159±0.018996 |
| C | 0.433829±0.120190 | 0.062738±0.010452 | 0.004230±0.000380 | 0.012176±0.000932 |

## 6. Preregistered utility gates

Strict criteria: CAM28_1 Δ > −0.28 pp; Boundary Δ > +0.20 pp; final mIoU Δ > +0.10 pp. A variant succeeds only if all three pass.

| Variant | Semantic Δ | Gate A | Boundary Δ | Gate B | mIoU Δ | Gate C | Variant success |
|---|---:|:---:|---:|:---:|---:|:---:|:---:|
| A | -0.5825 | FAIL | +0.4496 | PASS | +0.0093 | FAIL | FAIL |
| B | -0.4767 | FAIL | +0.4493 | PASS | +0.0370 | FAIL | FAIL |
| C | -0.5759 | FAIL | +0.4385 | PASS | -0.0309 | FAIL | FAIL |

## 7. Reproducibility

- Training implementation commit: `3f23c5f58eba9d7bf9e72815ca6c024261fc66ed`.
- Common Epoch 20 checkpoint SHA256: `2aae7e7c83373a4bb8865084ede86ba91a79ae3788b732b19fa478ee6c4311fb`.
- Schedule SHA256: `fa648405f40852e98f3d73776b7feee904bd59309ea1df2a97255650b0d00eea`.
- C0 Epoch 25 checkpoint SHA256: `44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8`.
- W1 Epoch 25 checkpoint SHA256: `31976d27e5670256bd08565bb8b34efb510442e4afc0651a7797fee68d88b7fc`.
- A Epoch 25 checkpoint SHA256: `3f07f15ffb684f7c4695033bfce842212f56a2baf14c887baab2704c451c2e61`.
- B Epoch 25 checkpoint SHA256: `ac38765c013fdef56360504c09d9e9102f38be3dceb7785edf7d492c263e6e73`.
- C Epoch 25 checkpoint SHA256: `521ea8b7184d16c91de289dc62f71ea9166480d9d947b5264dfeaefb27cf0c49`.
- Validation prediction order and GT masks were byte-equal across all five variants.

## 8. Core scientific interpretation

- Best semantic result: Variant B, CAM28_1 Δ -0.4767 pp. It recovers only 0.0804 pp (14.44%) of W1's 0.5571 pp loss, below the required 50%.
- Best overall result: Variant B, final mIoU Δ +0.0370 pp, below the +0.10 pp utility threshold.
- W1 already gives Boundary Δ +0.4411 pp. A/B/C give +0.4496/+0.4493/+0.4385 pp, so the fixed interactions mostly preserve rather than create W1's structural gain.
- Variant B has the largest measured interaction RMS but still fails semantic recovery; within these frozen mechanisms, the failure is not explained by an interaction that is merely too small.
- Answer to the core question: these three fixed minimal cross-band interactions do not solve the semantic degradation caused by frequency decoupling.

## 9. Decision

No variant solved all three gates, although at least one preregistered signal remained positive.

NEXT_STEP: Reformulate frequency interaction from the partial gate evidence; do not tune these fixed variants.

DECISION = NEXT_STEP

STOP.
