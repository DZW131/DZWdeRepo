# CBCCH Phase-2 Contrastive Boundary Affinity Learning

## 1. Frozen protocol

- Experiment: `EXP-CBCCH-002`; BCSS seed42; locked common Epoch20 → Epoch21–25 matched continuation.
- C0, W1 and Phase-1 BC-CH are reused SHA-locked artifacts; A2 and A3 are newly trained.
- A2: local semantic-affinity propagation at every pixel; every valid pixel participates in contrastive learning.
- A3: `Y=(1-B)P_affinity+B*F`; only exact top-20% detached-B anchors participate in contrastive learning.
- Local neighborhood=15×15; `z_s` reuses `ic1`; `z_h` is mean(|LH|), mean(|HL|), mean(|HH|); one deterministic positive/negative; τ=0.07.
- `L=L_official+0.1 L_con`; no new trainable parameters. Same schedule, batches, augmentation/model seeds, optimizer, BF16 and Epoch25 FINAL rule.
- No test, LUAD, alternate seed, best-checkpoint selection or validation tuning.

## 2. Overall and spatial validation results

| Variant | mIoU | Δ pp | mDice | Boundary acc. | Δ pp | Interior acc. | Δ pp | Boundary mIoU | Interior mIoU | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 66.8555 | +0.0000 | 79.9194 | 51.6894 | +0.0000 | 85.5057 | +0.0000 | 31.9958 | 70.9137 | 36.6083 | 68.5264 | 89.1509 |
| W1 | 66.8522 | -0.0034 | 79.9200 | 52.1305 | +0.4411 | 85.4218 | -0.0839 | 32.2440 | 70.8667 | 38.0735 | 68.4510 | 89.1155 |
| BC-CH | 66.8429 | -0.0127 | 79.9135 | 51.8666 | +0.1772 | 85.4398 | -0.0658 | 32.2430 | 70.8697 | 38.0645 | 68.2999 | 89.1736 |
| A2 | 66.7650 | -0.0905 | 79.8531 | 52.1963 | +0.5068 | 85.4009 | -0.1048 | 32.2234 | 70.7680 | 38.0366 | 68.6022 | 89.0308 |
| A3 | 66.7300 | -0.1255 | 79.8305 | 52.2525 | +0.5630 | 85.3165 | -0.1892 | 32.3609 | 70.7101 | 39.1432 | 68.4812 | 88.9715 |

## 3. CAM hierarchy

| Variant | CAM56 | CAM28_1 | Δ CAM28_1 pp | CAM28_2 | Deep | Final |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 61.0919 | 66.4431 | +0.0000 | 66.2999 | 64.5274 | 66.8555 |
| W1 | 61.2428 | 65.8860 | -0.5571 | 66.4122 | 64.5719 | 66.8522 |
| BC-CH | 61.0962 | 66.1422 | -0.3009 | 66.3233 | 64.5275 | 66.8429 |
| A2 | 61.2563 | 65.6903 | -0.7528 | 66.4385 | 64.5896 | 66.7650 |
| A3 | 61.2682 | 65.3638 | -1.0793 | 66.4488 | 64.5907 | 66.7300 |

## 4. Contrastive and affinity mechanism

### A2

- Epoch25 contrastive loss=0.044018; valid-anchor fraction=0.740938.
- Positive/negative semantic similarity=0.936682/0.063950; margin=0.872731.
- Affinity entropy=0.992839±0.003843; max=0.008250; self=0.008248; effective neighbors=162.911.
- Boundary/interior propagation RMS=0.791350/0.598769; global residual=0.657132.
- Final gamma_context/gamma_veto=1.65727818/0.66987377.

### A3

- Epoch25 contrastive loss=0.041221; valid-anchor fraction=0.143794.
- Positive/negative semantic similarity=0.941380/0.090425; margin=0.850954.
- Affinity entropy=0.992634±0.003858; max=0.008289; self=0.008287; effective neighbors=162.749.
- Boundary/interior propagation RMS=0.792895/0.600302; global residual=0.658645.
- Final gamma_context/gamma_veto=1.66036737/0.66717768.

## 5. Preregistered gates

| Gate | Margin | Criterion | Result |
|---|---:|---|:---:|
| A_CAM28_1_recovery | -0.9793 pp | A3 CAM28_1 > C0 CAM28_1 - 0.1 pp | FAIL |
| B_boundary_accuracy | +0.5630 pp | A3 boundary accuracy > C0 | PASS |
| C_final_mIoU | -0.1255 pp | A3 final mIoU > C0 | FAIL |

## 6. Reproducibility and resource evidence

- Preflight: `CBCCH_PREFLIGHT_PASS`; real batch20 BF16; official+contrastive loss; no optimizer step.
- Common Epoch20 SHA256: `2aae7e7c83373a4bb8865084ede86ba91a79ae3788b732b19fa478ee6c4311fb`.
- Schedule SHA256: `fa648405f40852e98f3d73776b7feee904bd59309ea1df2a97255650b0d00eea`.
- C0 Epoch25 SHA256: `44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8`.
- W1 Epoch25 SHA256: `31976d27e5670256bd08565bb8b34efb510442e4afc0651a7797fee68d88b7fc`.
- BC-CH Epoch25 SHA256: `959ba77c16e440a8e69ece8740238f03f2711f4ea0faca737c33c4b92131f2ad`.
- A2 Epoch25 SHA256: `f305d9b4f9fa0d9c478837f1206fbd2b4f0e2e3a4afd64a985f6ab44dab4dc21`.
- A3 Epoch25 SHA256: `2a128636fba2417342e130787c88cd9d30410702f0797ad93b908173bc70cc4e`.
- A2: continuation runtime=0.379 h; peak CUDA memory=7.633 GiB.
- A3: continuation runtime=0.378 h; peak CUDA memory=7.634 GiB.
- Prediction order and validation ground truth are byte-equal across all five variants.
- Trainable parameter count is exactly C0 for A2/A3; legacy CH15 state is restored for parity but intentionally dormant under the frozen Phase-2 aggregation equation.

## 7. Decision

Boundary behavior improves, but contrastive affinity does not deliver positive overall utility.

DECISION = BOUNDARY_ONLY

STOP.
