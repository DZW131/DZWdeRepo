# WSA-CH EXP001 Semantic Assignment Feasibility Audit

## 1. Executive conclusion

**Decision: `WSA_CH_ASSIGNMENT_EXISTS_NO_REFINEMENT_GAIN`.**

Semantic assignment exists, but CBCCH boundary refinement does not improve it over raw F.

This is a validation-only diagnostic. No WSA-CH model was trained and the oracle ceiling cannot unlock a future run.

## 2. Frozen protocol and provenance

- Implementation commit: `03408eb84ca5076396a5c04cf8f0af8cee7fbdbb`.
- BCSS validation only; canonical unflipped 224×224 view; BF16 inference.
- Same-space HFRM28_1 representations: `F`, `F_CH=CH_C0(F)`, and `F_b=P_affinity(F)`.
- Primary `G_c`: normalized CH feature weighted by `softmax(ReLU(ic1_C0(F_CH)))`, restricted only by image-level foreground labels.
- Oracle `G_c`: GT-interior-weighted CH feature; observation only.
- Boundary: foreground-class transition distance ≤7 px; hardest wrong group defines the margin.
- No training, test, LUAD, threshold, temperature, checkpoint selection, or parameter tuning.
- Exact command: `tools/run_wsa_ch_exp001.py --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --c0-dir /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/matched/C0 --cbcch-dir /home/duyanhong/experiments/EXP-CBCCH-002-8057faa/matched/A3 --output-dir /home/duyanhong/experiments/WSA-CH-EXP001-03408eb/full --num-workers 4 --bootstrap-resamples 10000`

| Artifact | SHA256 | Locked val mIoU | Locked val mDice |
|---|---|---:|---:|
| C0 | `44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8` | 66.8555 | 79.9194 |
| CBCCH-A3 | `2a128636fba2417342e130787c88cd9d30410702f0797ad93b908173bc70cc4e` | 66.7300 | 79.8305 |

## 3. Primary automatic-group assignment

| Query | Image-balanced margin | Image-balanced acc. | Pixel margin | Pixel acc. | Same sim. | Hardest-wrong sim. |
|---|---:|---:|---:|---:|---:|---:|
| Raw F | 0.005551 | 0.504489 | 0.006166 | 0.516757 | 0.496553 | 0.490387 |
| CH(F) | -0.001083 | 0.479330 | -0.002236 | 0.486501 | 0.923179 | 0.925416 |
| CBCCH F_b | 0.002009 | 0.488621 | 0.001654 | 0.495801 | 0.697219 | 0.695565 |

- Image-balanced chance accuracy: 0.459418; pixel-weighted chance: 0.443227.
- Eligible images/pixels: 2152 / 12907279.
- Excluded: 1265 single-class images and 1 multi-class images without eligible GT boundary pixels.

## 4. Preregistered gates

| Gate | Bootstrap estimate [95% CI] | Result |
|---|---:|:---:|
| F_b margin > 0 | 0.002009 [0.000913, 0.003110] | PASS |
| F_b accuracy − 1/K > 0 | 0.029203 [0.024334, 0.034065] | PASS |
| F_b accuracy − Raw F accuracy > 0 | -0.015868 [-0.019080, -0.012638] | FAIL |

## 5. Boundary difficulty analysis

Easy/hard is frozen by the raw-F assignment, not by post-hoc margin quantiles.

| Query | Raw-easy pixels | Raw-hard pixels | Hard correction rate | Easy harm rate | Acc. on raw-hard |
|---|---:|---:|---:|---:|---:|
| CH(F) | 6669924 | 6237355 | 0.268321 | 0.309468 | 0.268321 |
| CBCCH F_b | 6669924 | 6237355 | 0.169553 | 0.199110 | 0.169553 |

## 6. Per-class primary assignment

| Class | Raw margin / acc. | CH margin / acc. | F_b margin / acc. |
|---:|---:|---:|---:|
| 0 | 0.021814 / 0.577804 | -0.025427 / 0.421004 | 0.037540 / 0.638433 |
| 1 | 0.009375 / 0.542444 | 0.008463 / 0.513059 | -0.003831 / 0.498183 |
| 2 | -0.019829 / 0.398461 | 0.004806 / 0.516546 | -0.029717 / 0.329028 |
| 3 | -0.020277 / 0.356998 | 0.014499 / 0.528936 | -0.048172 / 0.235587 |

## 7. GT-interior oracle ceiling

Eligible oracle images/pixels: 2042 / 12787928; excluded for insufficient interior prototypes or boundary pixels: 110.

| Query | Image-balanced margin | Image-balanced acc. | Pixel margin | Pixel acc. |
|---|---:|---:|---:|---:|
| Raw F | 0.009972 | 0.538105 | 0.009046 | 0.542516 |
| CH(F) | 0.008169 | 0.535688 | 0.006482 | 0.538192 |
| CBCCH F_b | 0.008140 | 0.533344 | 0.006436 | 0.530610 |

The oracle uses segmentation GT and therefore measures representational capacity only; it is not a deployable assignment result.

## 8. Semantic-group diagnostics

- CAM spatial entropy: 0.861022 ± 0.063291.
- Mean maximum spatial weight: 0.016400.
- Effective weighted locations: 374.158828.
- Inter-group prototype cosine: 0.872589 ± 0.132433.

## 9. Scientific interpretation

CBCCH F_b has positive semantic margin (0.002009) and exceeds chance (0.029203), but its paired accuracy gain over raw F is -0.015868 with 95% CI [-0.019080, -0.012638]. Thus the CH groups contain assignment structure, while the frozen CBCCH refinement supplies no validated incremental assignment signal. Full WSA-CH training is not unlocked under this contract.

## 10. Validation evidence and artifacts

- Processed 3418 images in 25.29 s (0.0074 s/image).
- Peak CUDA allocated memory: 0.900 GiB.
- Tests: numerical margin/assignment, CAM prototype construction, oracle construction, bootstrap reproducibility, and difficulty accounting.
- Machine-readable outputs: `wsa_ch_exp001_summary.json`, `wsa_ch_exp001_per_image.csv`, and `wsa_ch_exp001_per_class.csv`.

STOP. No full WSA-CH implementation or training was started.
