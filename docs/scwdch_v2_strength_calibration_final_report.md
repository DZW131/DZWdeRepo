# SC-WDCH Strength Calibration Final Report

## 1. Frozen protocol and provenance

- Experiment: `EXP-WDCH-002`; dataset: BCSS; seed: 42.
- Calibration used the full training split and common epoch20 only; validation/test were forbidden.
- C0/W1 are the hash-verified matched v1 continuations; only W2 was newly trained from the same common state.
- Epoch21-25, batch20, 224x224, BF16, identical optimizer/poly schedule/augmentations/batch order.
- Epoch25 FINAL only; validation was observation-only and test was not run.
- Common checkpoint SHA256: `2aae7e7c83373a4bb8865084ede86ba91a79ae3788b732b19fa478ee6c4311fb`
- Schedule SHA256: `fa648405f40852e98f3d73776b7feee904bd59309ea1df2a97255650b0d00eea`

## 2. Training-only strength calibration

- R_CH: 0.30895151
- R_WD: 0.17027384
- Fixed scale s=R_CH/R_WD: 1.81443911
- Direct initial SC-WD rectification RMS: 0.30861387
- Initial strength ratio: 0.998907

## 3. Final validation performance

| Variant | mIoU | mDice | C0 IoU | C1 IoU | C2 IoU | C3 IoU |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 66.8555 | 79.9194 | 76.5672 | 70.1931 | 57.9787 | 62.6833 |
| W1 | 66.8522 | 79.9200 | 76.4294 | 70.1589 | 57.8720 | 62.9485 |
| W2 | 66.7537 | 79.8406 | 76.4418 | 70.1936 | 57.5312 | 62.8483 |

- W2-C0: mIoU -0.1018 pp; mDice -0.0788 pp.
- W2-W1: mIoU -0.0985 pp.
- Per-class W2-C0 IoU: C0 -0.1254 pp, C1 +0.0005 pp, C2 -0.4475 pp, C3 +0.1650 pp

## 4. Boundary and interior

| Variant | Boundary accuracy | Interior accuracy |
|---|---:|---:|
| C0 | 51.6894 | 85.5057 |
| W1 | 52.1305 | 85.4218 |
| W2 | 51.3833 | 85.4663 |

- W2-C0: Boundary -0.3061 pp; Interior -0.0394 pp.
- W2-W1: Boundary -0.7472 pp; Interior +0.0446 pp.

## 5. Component-size analysis

| Variant | Small | Medium | Large |
|---|---:|---:|---:|
| C0 | 36.6083 | 68.5264 | 89.1509 |
| W1 | 38.0735 | 68.4510 | 89.1155 |
| W2 | 34.8998 | 68.0158 | 89.2931 |

## 6. CAM hierarchy

| Stage | C0 mIoU | W1 mIoU | W2 mIoU | W2-C0 | W2-W1 |
|---|---:|---:|---:|---:|---:|
| 56 | 61.0919 | 61.2428 | 61.1886 | +0.0966 | -0.0542 |
| 28_1 | 66.4431 | 65.8860 | 65.7846 | -0.6585 | -0.1015 |
| 28_2 | 66.2999 | 66.4122 | 66.3258 | +0.0259 | -0.0864 |
| deep | 64.5274 | 64.5719 | 64.5288 | +0.0014 | -0.0431 |
| final | 66.8555 | 66.8522 | 66.7537 | -0.1018 | -0.0985 |

## 7. Feature magnitude audit

| Variant/operator | Input RMS | Context output RMS | Rectification RMS | Output/Input |
|---|---:|---:|---:|---:|
| C0/CH15 | 0.287665 | 0.377646 | 0.342855 | 1.305678 |
| W1/WDCH7 | 0.349709 | 0.241556 | 0.229870 | 0.689437 |
| W2/SC-WDCH7(s=1.81443906) | 0.339424 | 0.310156 | 0.408864 | 0.912264 |

- Final SC-WD/CH rectification-strength ratio: 1.192527.

## 8. Preregistered gates

- Gate A — Strength Recovery: FAIL (1.192527; required 0.9-1.1).
- Gate B — Mechanism Preservation: FAIL (Boundary -0.3061 pp; Interior -0.0394 pp).
- Gate C — Model Improvement: FAIL (mIoU -0.1018 pp; required +0.10 pp).
- CAM28_1 recovery versus W1: NO (-0.1015 pp).

## 9. Interpretation

Final strength recovery failed; close the fixed strength-calibration hypothesis.

## 10. Checkpoints

- C0: `44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8`
- W1: `31976d27e5670256bd08565bb8b34efb510442e4afc0651a7797fee68d88b7fc`
- W2: `1fcfd37f2b21715b519a74054099e94ad3060fd8dfd0963b4707871f01f6b412`

No test, LUAD, other seed, checkpoint selection, scalar tuning or additional model variant was used.

DECISION = STOP
