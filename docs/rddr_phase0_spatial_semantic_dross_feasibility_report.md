# RDDR Phase-0 Spatial-Semantic Dross Feasibility Report

## 1. Commit and frozen provenance

- Audit commit: `586f402a30f446c409c625b55953e329cc041dcc`
- Pure A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Model: evaluation-only; no optimizer, gradient, training, or new checkpoint.

## 2. Exact command

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase0_dross_audit.py --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --output-dir /home/duyanhong/experiments/RDDR_PHASE0_586f402/formal --num-workers 4 --bootstrap-resamples 10000
```

## 3. Tensor contract

# RDDR Phase-0 Tensor Contract

This contract is audited against pure A0 commit `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`. The diagnostic runner repeats the frozen `Net.forward` operations without changing `network/resnet38_cls.py`; a unit test requires the default and diagnostic output tuples to be numerically identical in evaluation mode.

| Audit name | A0 source | 224-input shape | Spatial / channels | Position | Classifier |
|---|---|---:|---|---|---|
| `F28_raw` | `network/resnet38_cls.py:100`, `feat_28_1` | `[B,512,28,28]` | 28×28 / 512 | HFRM28_1 input, before rectification | diagnostic `ic1(F28_raw)` |
| `F28_rect` | `network/resnet38_cls.py:110`, `feat_28_1_rectified` | `[B,512,28,28]` | 28×28 / 512 | HFRM28_1 output, after rectification | existing `ic1` |
| `Ddeep` | `network/resnet38_cls.py:106`, `feat_deep` | `[B,4096,28,28]` | 28×28 / 4096 | deepest semantic feature, before dropout | existing `fc8` |
| `CAM28_raw` | diagnostic-only `ic1(F28_raw)` using existing layer declared at `network/resnet38_cls.py:75` | `[B,4,28,28]` | 28×28 / 4 | pre-HFRM class logits; not used by SSHR inference | `ic1` |
| `CAM28_rect` | `network/resnet38_cls.py:115` | `[B,4,28,28]` | 28×28 / 4 | normal post-HFRM class logits; official CAM applies ReLU | `ic1` |
| `CAMdeep` | `network/resnet38_cls.py:119` (training) and `Net_CAM.forward_cam` (inference) | `[B,4,28,28]` | 28×28 / 4 | deep class logits; official CAM applies ReLU | `fc8` |

Primary `p_s` and `p_d` are softmaxes of the pre-threshold, pre-ReLU class logits `ic1(F28_raw)` and `fc8(Ddeep)`, upsampled bilinearly to the 224×224 mask with `align_corners=False`. The BCSS classifier has four foreground classes. Ground-truth label 4 is the evaluator-overwritten background and is excluded from the foreground error-detection population; any other out-of-range label is recorded as excluded/ignore.

## 4. Eligible population

- Images: `3418`
- Eligible foreground pixels: `158639345`
- Excluded evaluator-background pixels: `12862223`
- Excluded invalid/ignore pixels: `0`

## 5. Frozen Primary S_JS

`S_JS = JS(softmax(ic1(F28_raw)), softmax(upsample(fc8(Ddeep))))`, using natural log, epsilon 1e-8 and temperature 1.0. No GT, boundary, edge, threshold, or learned parameter enters the score.

## 6. Primary error prevalence

Raw shallow foreground error prevalence: `27.8603%`.

## 7. AUROC / AUPRC

- Image-balanced AUROC: `0.764969`; 95% CI `[0.760472, 0.769470]`.
- Pixel-weighted AUROC/AUPRC: `0.772858` / `0.580023`.
- Image-balanced AUPRC: `0.574627`; 95% CI `[0.566858, 0.582607]`.
- Pixel AUPRC/prevalence ratio: `2.081901`.

## 8. Fixed Top-k enrichment

| Quantile | Coverage | Enrichment | Net correction |
|---:|---:|---:|---:|
| Top 5% | 5.000% | 2.6159 | +29.8794 pp |
| Top 10% | 10.000% | 2.6330 | +31.5963 pp |
| Top 20% | 20.000% | 2.3517 | +25.0584 pp |
| Top 30% | 30.000% | 1.9768 | +17.0072 pp |

Primary Top20 enrichment bootstrap 95% CI: `[2.3019, 2.4046]`.

## 9. Deep correction potential

- Top20 correction rate among shallow-wrong pixels: `74.7875%`.
- Top20 harm rate among shallow-correct pixels: `69.4384%`.
- NetCorrection20: `+25.0584 pp`; 95% CI `[+23.0874, +27.0009] pp`.

## 10. Oracle Top20 swap diagnostic

Raw shallow mIoU/accuracy: `44.2554` / `72.1397`. Hybrid-Top20: `57.0076` / `77.1514`.

## 11. Boundary / interior

| Stratum | AUROC [95% CI] | Enrichment20 [95% CI] | NetCorrection20 [95% CI] |
|---|---:|---:|---:|
| boundary | 0.580668 [0.576023, 0.585435] | 1.1953 [1.1856, 1.2053] | -0.5588 [-1.7470, +0.6240] pp |
| interior | 0.781542 [0.776941, 0.786053] | 2.5563 [2.4961, 2.6205] | +28.2228 [+26.0939, +30.2863] pp |

## 12. Per-class analysis

| Class | Error prevalence | AUROC [95% CI] | AUPRC | Enrichment20 [95% CI] | NetCorrection20 [95% CI] |
|---:|---:|---:|---:|---:|---:|
| 0 | 15.5516% | 0.748581 [0.739716, 0.757549] | 0.485552 | 2.7593 [2.6325, 2.9008] | -22.1595 [-24.3218, -19.9453] pp |
| 1 | 23.3921% | 0.670266 [0.661040, 0.679626] | 0.565211 | 2.5007 [2.4170, 2.5901] | +13.1326 [+10.8824, +15.3513] pp |
| 2 | 54.0625% | 0.710706 [0.696033, 0.724735] | 0.833298 | 1.5575 [1.5121, 1.6042] | +54.6419 [+52.3309, +56.8370] pp |
| 3 | 83.3805% | 0.688770 [0.663727, 0.713988] | 0.968058 | 1.0619 [1.0468, 1.0783] | +77.7724 [+75.8086, +79.5455] pp |

Macro AUROC `0.704581`; minimum class AUROC `0.670266`.

## 13. Correct/error score distributions

- Correct S_JS: mean ± std `0.089087 ± 0.104995`, median `0.052095`, p25/p75 `0.014506/0.126607`.
- Error S_JS: mean ± std `0.229342 ± 0.163489`, median `0.208248`, p25/p75 `0.085147/0.349371`.
- Error-minus-correct mean difference `0.140256`; 95% CI `[0.136632, 0.143911]`.
- Cohen's d `1.130241`; Cliff's delta `0.545716`.

## 14. CH transition analysis

| Group | Pixels | Mean S_JS |
|---|---:|---:|
| Corrected_by_CH | 19934592 | 0.291117 [0.287006, 0.295117] |
| Still_Wrong | 24262754 | 0.178587 [0.175825, 0.181409] |
| Harmed_by_CH | 4443224 | 0.235441 [0.231920, 0.239062] |
| Stable_Correct | 109998775 | 0.083175 [0.082041, 0.084311] |

Harmed-by-CH vs Stable-Correct image-balanced AUROC: `0.816309`.

Pairwise image-bootstrap mean differences:

| Left | Right | Difference [95% CI] |
|---|---|---:|
| Corrected_by_CH | Still_Wrong | +0.112530 [+0.108421, +0.116477] |
| Corrected_by_CH | Harmed_by_CH | +0.055676 [+0.050520, +0.060708] |
| Corrected_by_CH | Stable_Correct | +0.207942 [+0.203808, +0.212008] |
| Still_Wrong | Harmed_by_CH | -0.056854 [-0.060955, -0.052851] |
| Still_Wrong | Stable_Correct | +0.095413 [+0.092558, +0.098380] |
| Harmed_by_CH | Stable_Correct | +0.152266 [+0.148751, +0.155916] |

## 15. Uncertainty baseline comparison

| Score | Pixel AUROC | Pixel AUPRC | Enrichment20 | NetCorrection20 |
|---|---:|---:|---:|---:|
| S_JS | 0.772858 | 0.580023 | 2.3517 | +25.0584 pp |
| S_entropy | 0.738251 | 0.509269 | 1.9486 | +16.7590 pp |
| S_lowconf | 0.747511 | 0.517745 | 2.0004 | +18.2216 pp |
| S_cos | 0.798326 | 0.595479 | 2.4859 | +24.4464 pp |
| S_hard | 0.730696 | 0.506449 | 2.4570 | +21.6320 pp |

## 16. Secondary error targets

S_JS image-balanced AUROC for rectified CAM28_1 error: `0.609883`; for canonical final error: `0.597526`.

## 17. Bootstrap contract

All reported CIs use 10,000 image-level resamples with seed 42. Dataset-level Top20 thresholds remain fixed during resampling.

## 18. Four preregistered gates

| Gate | Requirement | Result | Pass |
|---|---|---:|---:|
| A | AUROC >= 0.58 and bootstrap lower > 0.50 | AUROC=0.7650, low=0.7605 | True |
| B | Enrichment20 >= 1.40 and bootstrap lower > 1.20 | enrichment=2.3517, low=2.3019 | True |
| C | NetCorrection20 > 0 and bootstrap lower > 0 | net=0.250584, low=0.230874 | True |
| D | Interior AUROC > 0.52 and Enrichment20 > 1.10 | AUROC=0.7815, enrichment=2.5563 | True |

## 19. Scientific interpretation

All preregistered links are supported: hierarchical disagreement detects local errors, high-disagreement pixels retain positive deep-repair potential, and the signal remains informative in interiors.

## 20. Final decision

Decision: `RDDR_PHASE0_GO`.

DECISION = RDDR_PHASE0_GO
