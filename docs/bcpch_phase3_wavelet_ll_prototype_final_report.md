# BCP-CH Phase-3 Wavelet Low-frequency Semantic Prototype Recovery

## 1. Frozen protocol

- Experiment: `EXP-BCPCH-003`; BCSS seed42; locked public Epoch20 → Epoch21–25 matched continuation.
- Only BCP-CH is newly trained. C0, BC-CH and CBCCH are reused SHA-locked Epoch25 FINAL references.
- LL embedding is exactly `z=L2(IDWT(LL,0,0,0))`; CAM selection is per-class spatial min-max `ReLU(ic1(F))>0.70` with detached mask.
- Existing `fc8(feat_deep)` and official BCSS thresholds determine image presence; no new classifier, projection or trainable parameter.
- `Y=(1-B)(0.5P_affinity+0.5P_prototype)+BF`; no-valid-prototype fallback is exactly `P_prototype=P_affinity`.
- CBCCH local15, top-20% boundary anchors, τ=0.07 and `L=L_official+0.1L_con` remain unchanged.
- Same optimizer, LR schedule, batch/augmentation/model seeds, BF16 and Epoch25 FINAL selection. No test, LUAD, other seed or tuning.

## 2. Overall and boundary validation results

| Variant | mIoU | Δ vs C0 pp | mDice | Boundary acc. | Δ vs C0 pp | Boundary mIoU | Interior acc. | Interior mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 66.8555 | +0.0000 | 79.9194 | 51.6894 | +0.0000 | 31.9958 | 85.5057 | 70.9137 |
| BC-CH | 66.8429 | -0.0127 | 79.9135 | 51.8666 | +0.1772 | 32.2430 | 85.4398 | 70.8697 |
| CBCCH | 66.7300 | -0.1255 | 79.8305 | 52.2525 | +0.5630 | 32.3609 | 85.3165 | 70.7101 |
| BCP-CH | 66.7354 | -0.1202 | 79.8392 | 52.2509 | +0.5614 | 32.4202 | 85.2612 | 70.7066 |

## 3. CAM hierarchy

| Variant | CAM56 | CAM28_1 | Δ CAM28_1 vs C0 pp | CAM28_2 | Deep | Final |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 61.0919 | 66.4431 | +0.0000 | 66.2999 | 64.5274 | 66.8555 |
| BC-CH | 61.0962 | 66.1422 | -0.3009 | 66.3233 | 64.5275 | 66.8429 |
| CBCCH | 61.2682 | 65.3638 | -1.0793 | 66.4488 | 64.5907 | 66.7300 |
| BCP-CH | 61.3447 | 65.1441 | -1.2990 | 66.5124 | 64.6392 | 66.7354 |

## 4. Prototype and propagation diagnostics

- GT-category boundary-to-prototype cosine: Epoch20=0.582804 over 9781836 pixels; Epoch25=0.616266 over 9932085 pixels; Δ=+0.033462.
- CAM confidence fraction=0.008710; predicted presence/image=1.4116; valid prototypes/image=1.4116; fallback fraction=0.000000.
- LL reconstruction/input RMS=0.933041; affinity output/input RMS=0.690575; prototype output/input RMS=0.119015.
- Boundary/interior context residual RMS=0.472262/0.462555.
- Final gamma_context/gamma_veto=1.68286335/0.69763798.

## 5. Preregistered gates

| Gate | Observed | Margin/status | Criterion | Result |
|---|---:|---:|---|:---:|
| A Boundary accuracy | 52.2509 | -0.0016 pp | > 52.2525% (CBCCH) | FAIL |
| B CAM28_1 recovery | 65.1441 | FAIL | COMPLETE >=66.4431%; PARTIAL >=65.9035% | FAIL |
| C Final mIoU | 66.7354 | -0.1201 pp | > 66.8555% (C0) | FAIL |

## 6. Training and resource evidence

- Preflight: `BCPCH_PREFLIGHT_PASS`; real batch20 BF16; official+contrastive loss; no optimizer step.
- Epoch21→25 official loss: 0.187271 → 0.171774; contrastive loss: 0.046182 → 0.040714.
- Continuation runtime=0.389 h; peak CUDA memory=7.631 GiB.
- Common Epoch20 SHA256: `2aae7e7c83373a4bb8865084ede86ba91a79ae3788b732b19fa478ee6c4311fb`.
- Schedule SHA256: `fa648405f40852e98f3d73776b7feee904bd59309ea1df2a97255650b0d00eea`.
- C0 Epoch25 SHA256: `44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8`.
- BC-CH Epoch25 SHA256: `959ba77c16e440a8e69ece8740238f03f2711f4ea0faca737c33c4b92131f2ad`.
- CBCCH Epoch25 SHA256: `2a128636fba2417342e130787c88cd9d30410702f0797ad93b908173bc70cc4e`.
- BCP-CH Epoch25 SHA256: `eb0a8acf09ebe5004193dd682f80686c6351c47a22b0efaf8999b6cac9b7c629`.
- Validation image order and GT masks are byte-equal across all four variants.
- BCP-CH trainable parameter count equals C0; the legacy CH15 parameter restores for parity but is dormant under the frozen Phase-3 equation.

## 7. Scientific interpretation

The frozen LL prototype construction does not validate the proposed semantic-recovery mechanism.

DECISION = NO_GO

STOP.
