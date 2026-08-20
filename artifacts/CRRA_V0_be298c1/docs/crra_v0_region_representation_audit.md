# CRRA-v0 Core-aware Region Representation Audit

## 1. Executive conclusion

**CRRA_V0_NOGO**

Representation flag: **REGION_REPRESENTATION_ROUTE_CLOSED**.

The highest frozen, slide-held-out representation is Core+Rim, with a Macro-F1 delta of -0.0051 and a Type-B accuracy delta of -0.0136 versus WholeToken.

This is a representation audit only. It does not establish a WSSS segmentation gain.

## 2. Frozen protocol and provenance

- A0 source commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Audit commit: `be298c1704ffdb684f779d09e943fd495adf6f14`
- A0 checkpoint: `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Exact command: `/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/audit_crra_v0.py --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --output-dir /home/duyanhong/experiments/CRRA_V0_be298c1 --audit-commit be298c1704ffdb684f779d09e943fd495adf6f14 --batch-size 20 --num-workers 8 --amp-dtype bf16`
- Precision: bf16
- Validation images/slides: 3418 / 22
- Training: none. Test/LUAD: not accessed.

## 3. Common-support and exclusion audit

- Raw connected components: 9199
- Area-1 components rejected by the frozen minimum-area rule: 719
- Total proposed regions (area >= 2): 8480
- Empty-core regions: 1526
- Empty-rim regions: 0
- Excluded regions: 1526
- Common-support regions: 6954
- Common-support fraction: 82.0047%
- Coverage review: COVERAGE_PASS

Per-class exclusions are recorded in `diagnostics/exclusion_by_predicted_class.csv` and `diagnostics/exclusion_by_gt_majority_class.csv`.

## 4. Required executive table

| Representation | Dim | OOF Macro-F1 | Delta vs Whole | Type-B Acc | DeltaB vs Whole | Type-A Acc |
|---|---:|---:|---:|---:|---:|---:|
| Whole | 512 | 0.6799 | — | 0.3498 | — | 0.8570 |
| Core | 512 | 0.6565 | -0.0234 | 0.3140 | -0.0358 | 0.8489 |
| Core+Rim | 1024 | 0.6748 | -0.0051 | 0.3362 | -0.0136 | 0.8681 |

OOF Macro-F1 is the mean over foreground classes C0-C3. The fixed multinomial probe is trained on every pure common-support region, including pure background-majority false-positive regions as class 4; accuracy and balanced accuracy therefore cover all observed labels.

## 5. Required fold table

| Fold | Whole F1 | Core F1 | Core-Whole | Core+Rim F1 | CR-Whole |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7331 | 0.6805 | -0.0526 | 0.7119 | -0.0213 |
| 2 | 0.6642 | 0.6439 | -0.0204 | 0.6765 | +0.0123 |
| 3 | 0.6302 | 0.5993 | -0.0310 | 0.6237 | -0.0065 |
| 4 | 0.5608 | 0.5584 | -0.0024 | 0.5786 | +0.0178 |
| 5 | 0.6037 | 0.6202 | +0.0164 | 0.6030 | -0.0007 |

Best-candidate positive folds: 2/5; mean/min/max fold delta: +0.0003 / -0.0213 / +0.0178.

## 6. Required per-class table

| Class | Whole F1 | Core F1 | DeltaCore | Core+Rim F1 | DeltaCR |
|---|---:|---:|---:|---:|---:|
| C0 | 0.8075 | 0.8006 | -0.0069 | 0.8160 | +0.0086 |
| C1 | 0.7406 | 0.7282 | -0.0124 | 0.7495 | +0.0089 |
| C2 | 0.6626 | 0.6389 | -0.0237 | 0.6584 | -0.0042 |
| C3 | 0.5089 | 0.4582 | -0.0507 | 0.4754 | -0.0335 |

Complete accuracy, balanced accuracy, confusion matrices, OOF predictions, and fold manifests are stored under `probes/` and `folds/`.

## 7. Core/rim diagnostics

The table reports mean dispersion and median Core-Rim discrepancy.

| Group | Whole Dispersion | Core Dispersion | Rim Dispersion | Core-Rim Discrepancy |
|---|---:|---:|---:|---:|
| Type-A | 0.069294 | 0.045841 | 0.095469 | 0.030499 |
| Type-B | 0.067242 | 0.039068 | 0.075218 | 0.028006 |
| Mixed | 0.082505 | 0.056682 | 0.104461 | 0.028651 |

Type-B vs Type-A discrepancy Mann-Whitney U: 2047147.00; two-sided p=0.000000; rank-biserial=-0.122053.

## 8. Slide bootstrap uncertainty

- Core-Whole: mean=-0.0213, 95% CI [-0.0444, +0.0066] (5000 slide bootstrap samples, seed=42).
- Core+Rim-Whole: mean=-0.0038, 95% CI [-0.0164, +0.0131] (5000 slide bootstrap samples, seed=42).

## 9. Answers to the preregistered questions

1. CoreToken vs WholeToken: delta Macro-F1=-0.0234; this is not a GO-scale improvement.
2. Core+Rim vs Core: delta Macro-F1=+0.0184; Type-B additional accuracy=+0.0222.
3. Highest OOF Macro-F1: Core+Rim (0.6748).
4. Fold stability: 2/5 best-candidate folds are positive.
5. Type-B separability: best gain=-0.0136.
6. Type-A preservation: best-candidate drop=-0.0111; review threshold exceeded=False.
7. C2 benefit: Core=-0.0237, Core+Rim=-0.0042 versus Whole.
8. Core dispersion is lower than Whole in all three taxonomies: True.
9. Type-B median Core-Rim discrepancy exceeds Type-A: False.
10. Common-support coverage=82.0047%; sufficient=True.
11. Representation recommendation: REGION_REPRESENTATION_ROUTE_CLOSED.
12. Region-centric route decision: CRRA_V0_NOGO.

## 10. Decision trace and stop boundary

- Best candidate: Core+Rim
- Best delta Macro-F1: -0.0051
- Best Type-B accuracy delta: -0.0136
- Positive folds: 2/5
- Non-negative/positive foreground classes: 2/4 / 2/4
- Type-A drop: -0.0111
- Hard NOGO conditions: `{'best_delta_below_0.01': True, 'type_b_does_not_improve': True, 'at_most_2_positive_folds': True, 'both_candidates_worse_than_whole': True}`
- REVIEW conditions: `{'best_bootstrap_ci_includes_zero': True, 'core_and_dual_sign_conflict': False, 'common_support_below_0.70': False, 'type_a_drop_above_0.02': False}`

Final decision: **CRRA_V0_NOGO**
Final flag: **REGION_REPRESENTATION_ROUTE_CLOSED**

The audit stops here. CRSR training, segmentation training, test, LUAD, graph, prototype, attention pooling, and any fourth representation were not run.
