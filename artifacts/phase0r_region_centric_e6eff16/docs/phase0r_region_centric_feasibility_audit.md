# SSHR Phase-0R Region-Centric Representation Feasibility Audit

## 1. Executive decision

**REGION_REP_REVIEW**

Evidence is intermediate and does not satisfy a preregistered Go or No-Go boundary; no model implementation is authorized.

## 2. Frozen protocol and parity

- Audit source commit: `e6eff16c2847410a1b60c8f56e136882f5ac175a`
- Frozen A0 source commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Checkpoint: `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Validation images/slides: 3418 / 22
- Differing pixels versus released inference: 0
- Maximum metric difference: 0.000e+00

## 3. Baseline and region oracle

| Protocol | mIoU | mDice | ΔmIoU |
|---|---:|---:|---:|
| Official A0 final | 67.3279% | 80.2680% | — |
| Shape-preserving majority oracle | 78.7157% | 88.0358% | +11.3878 pp |

Recoverable error fraction: -0.6243.

| Class | A0 IoU | Oracle IoU | ΔIoU |
|---:|---:|---:|---:|
| 0 | 76.4491% | 84.3590% | +7.9099 pp |
| 1 | 70.5718% | 78.1274% | +7.5556 pp |
| 2 | 57.8268% | 73.2343% | +15.4075 pp |
| 3 | 64.4640% | 79.1422% | +14.6782 pp |

## 4. Region purity and error taxonomy

Primary area ≥ 8 regions: 8445. Pure-region fraction: 0.7288.

| Taxonomy | Regions | Wrong-pixel mass | Fraction |
|---|---:|---:|---:|
| A_correct_pure | 3795 | 4285024 | 0.1072 |
| B_misclassified_pure | 2044 | 7462392 | 0.1866 |
| C_false_positive_pure | 424 | 2718372 | 0.0680 |
| D_mixed_boundary | 2290 | 25518000 | 0.6382 |

See `tables/region_purity_by_class.csv` and `tables/taxonomy_error_mass.csv` for the complete distribution.

## 5. Frozen feature probes

| Representation | Accuracy | Macro-F1 | Balanced accuracy | Segmentation ΔmIoU |
|---|---:|---:|---:|---:|
| centroid | 0.5613 | 0.4782 | 0.4889 | -15.7269 pp |
| bbox | 0.5844 | 0.5045 | 0.5174 | -8.1397 pp |
| region | 0.5989 | 0.5176 | 0.5282 | -7.1092 pp |
| geometry | 0.2755 | 0.2335 | 0.2526 | -52.2753 pp |
| region_geometry | 0.5979 | 0.5145 | 0.5243 | -7.3248 pp |

Region−BBox macro-F1: +0.0131; Region−Centroid macro-F1: +0.0395.

Positive slide-held-out folds for Region relabeling: 0/5.

## 6. Area sensitivity and representation geometry

The fixed area thresholds 1, 8, and 32 are reported in `tables/probe_results.csv`. Silhouette, Davies–Bouldin, and between/within scatter diagnostics are in `tables/representation_cluster_metrics.csv`.

## 7. Qualitative audit

The 32 automatically selected cases and four fixed-category panels are recorded in `tables/qualitative_selection.csv` and `figures/`. No examples were hand-picked.

## 8. Decision evidence

- Oracle gain: +11.3878 pp
- Region probe gain over BBox: +0.0131
- Region probe gain over Centroid: +0.0395
- Region relabeling gain: -7.1092 pp
- Oracle recovery fraction: -0.6243
- Mixed-boundary share of error mass: 0.6382
- Region+Geo relabeling gain: -7.3248 pp
- Geometry macro-F1 increment: -0.0031
- Geometry segmentation increment: -0.2155 pp
- Geometry adds value flag: False

## 9. Scope guard

This was a validation-only diagnostic. It did not train a model, inspect test data, change inference, or alter the frozen SSHR A0 architecture.
