# SSHR Phase-0B Routing Signal Learnability Audit

## 1. Executive conclusion

The preregistered MLP-C OOF router changes BCSS validation mIoU by -0.7158 pp, recovers -34.19% of the safe-image oracle gap, has 0/5 positive folds, and a slide-bootstrap 95% CI of [-1.0178, -0.4312] pp. Under the frozen decision logic, the result is `ROUTING_SIGNAL_NOGO`.

Final frozen decision: **ROUTING_SIGNAL_NOGO**.

## 2. Frozen contract

- Phase-0B parent commit: `f1a95059cd7914e9d6b72e08ec135c4c8ea32c06`.
- Baseline commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- Phase-0B audit commit: `76ffbf4d61f77dd6ef04946c0ffa16643b9acd86`.
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Dataset/split: BCSS validation only (3418 images, 22 source slides).
- SSHR training: false. Test evaluation: false. LUAD evaluation: false.
- Network, released inference, thresholds, TTA, and metric are unchanged.

Exact command:

```bash
python -u tools/audit_routing_signal_learnability.py \
  --phase0-dir /home/duyanhong/experiments/SSHR_DECISION_BOTTLENECK_PHASE0_f82eb0e \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --output-dir /home/duyanhong/experiments/SSHR_ROUTING_SIGNAL_PHASE0B \
  --num-workers 4
```

## 3. Exact parity

- Released-vs-Phase-0B differing prediction pixels: 0.
- Phase-0 released differing pixels: 0.
- mIoU absolute difference: 0.
- mDice absolute difference: 0.
- Parity gate: True.

## 4. Reproduction of Phase-0 references

Official fusion is 67.3279 mIoU / 80.2680 mDice. The frozen Phase-0 global-fusion, branch-oracle, pixel-oracle, and OOF class-probe references remain unchanged in the parent audit artifacts.

## 5. Safe Image Candidate Oracle

| Method | mIoU | mDice | Delta mIoU |
|---|---:|---:|---:|
| official_fusion | 67.3279 | 80.2680 | +0.0000 |
| safe_image_candidate_oracle | 69.4215 | 81.7516 | +2.0936 |

Official fusion is included as the first, tie-preferred safety candidate.

## 6. Slide-Level Safe Oracle

| Method | mIoU | mDice | Delta mIoU |
|---|---:|---:|---:|
| official_fusion | 67.3279 | 80.2680 | +0.0000 |
| slide_safe_oracle | 67.3704 | 80.2883 | +0.0425 |

Slide recovery ratio: 0.0203; slide-context flag: False.

## 7. Image-Level Fusion Oracle

| Method | mIoU | mDice | Delta mIoU |
|---|---:|---:|---:|
| official_fusion | 67.3279 | 80.2680 | +0.0000 |
| image_fusion_oracle | 69.9578 | 82.1257 | +2.6299 |

Frozen grid size: 286; soft gain beyond safe hard selection: +0.5363 pp; phenotype: `SOFT_MIXTURE_FAVORED`.

## 8. Exact Local Image×Class Oracle

| Method | mIoU | mDice | Delta mIoU |
|---|---:|---:|---:|
| official_fusion | 67.3279 | 80.2680 | +0.0000 |
| exact_local_imageclass_oracle | 71.6633 | 83.3251 | +4.3354 |

This is an exact per-image local diagnostic ceiling over 625 combinations, not a dataset-additive bound. Mean/median local q: 0.6288/0.5771; phenotype: `CLASS_CONDITIONAL_SIGNAL`.

## 9. Routing signal definitions

Signal A contains 153 aggregated-CAM/confidence/morphology/disagreement scalars per candidate. Signal B adds 7 aligned three-view TTA reliability scalars. Signal C adds 12 frozen feature statistics and four train-fold-only 16-D PCA contexts. No GT, slide ID, filename, patient ID, error mask, or validation-fitted calibration enters any probe input.

## 10. Signal-target correlations

| Set | Signal | Spearman(relative utility) | Spearman(absolute utility) |
|---|---|---:|---:|
| A | pred_class0_boundary_density | -0.2415 | -0.2205 |
| B | pred_class0_boundary_density | -0.2415 | -0.2205 |
| C | pred_class0_boundary_density | -0.2415 | -0.2205 |
| A | pred_class1_boundary_density | -0.2152 | -0.2118 |
| B | pred_class1_boundary_density | -0.2152 | -0.2118 |
| C | pred_class1_boundary_density | -0.2152 | -0.2118 |
| A | pred_class1_component_count | -0.1968 | -0.1558 |
| B | pred_class1_component_count | -0.1968 | -0.1558 |
| C | pred_class1_component_count | -0.1968 | -0.1558 |
| A | present_cam_p99_std | -0.1930 | -0.2467 |
| B | present_cam_p99_std | -0.1930 | -0.2467 |
| C | present_cam_p99_std | -0.1930 | -0.2467 |
| A | present_cam_spatial_entropy_std | -0.1899 | -0.2307 |
| B | present_cam_spatial_entropy_std | -0.1899 | -0.2307 |
| C | present_cam_spatial_entropy_std | -0.1899 | -0.2307 |

These correlations are diagnostic only; no signal was selected or removed from a probe after inspection.

## 11. Linear-A/B/C

| Method | mIoU | mDice | Delta mIoU |
|---|---:|---:|---:|
| Linear-A | 67.0206 | 80.0491 | -0.3073 |
| Linear-B | 67.0055 | 80.0388 | -0.3224 |
| Linear-C | 67.0661 | 80.0841 | -0.2618 |

## 12. MLP-A/B/C

| Method | mIoU | mDice | Delta mIoU |
|---|---:|---:|---:|
| MLP-A | 66.6262 | 79.7716 | -0.7017 |
| MLP-B | 66.2549 | 79.4961 | -1.0730 |
| MLP-C | 66.6121 | 79.7647 | -0.7158 |

## 13. Formal MLP-C OOF segmentation result

| Method | mIoU | mDice | Delta mIoU |
|---|---:|---:|---:|
| official_fusion | 67.3279 | 80.2680 | +0.0000 |
| MLP-C | 66.6121 | 79.7647 | -0.7158 |

Oracle recovery ratio: -0.3419. Only this preregistered primary probe determines GO/NOGO.

## 14. Override diagnostics

- override_rate: 0.780281
- oracle_override_opportunity: 0.477472
- override_precision: 0.239595
- harmful_override_rate: 0.348742
- mean_positive_override_gain: 0.030890
- mean_harmful_override_loss: 0.027597
- best_branch_top1_accuracy: 0.268578
- best_branch_top2_accuracy: 0.529257
- pairwise_ranking_accuracy: 0.380778
- relative_utility_mae: 0.048279
- predicted_true_spearman: 0.086271

## 15. Fold stability

| Fold | Official | Router | Delta | Override rate | Override precision |
|---:|---:|---:|---:|---:|---:|
| 0 | 64.8375 | 64.1709 | -0.6666 | 0.7525 | 0.1973 |
| 1 | 59.8077 | 58.5672 | -1.2405 | 0.8790 | 0.2139 |
| 2 | 61.9958 | 61.2380 | -0.7578 | 0.7073 | 0.2353 |
| 3 | 82.1834 | 81.7793 | -0.4041 | 0.8106 | 0.2065 |
| 4 | 63.2301 | 62.4697 | -0.7604 | 0.7511 | 0.3507 |

Positive held-out folds: 0/5.

## 16. Slide-level paired bootstrap

The paired grouped bootstrap uses 2000 replicates over 22 source slides (seed 20260817). Mean/median delta: -0.7144/-0.7162 pp; 95% CI [-1.0178, -0.4312].

## 17. Oracle recovery ratio

MLP-C recovers -34.19% of the safe-image oracle gap. Negative recovery is retained rather than clipped.

## 18. Routing phenotype flags

- `SOFT_MIXTURE_FAVORED`
- `CLASS_CONDITIONAL_SIGNAL`

## 19. Qualitative routing cases

32 cases were selected automatically: eight successful overrides, eight harmful overrides, eight missed opportunities, and eight correct fallbacks. See `figures/qualitative/` and `tables/qualitative_manifest.csv`.

## 20. Scientific interpretation

MLP-C gains less than +0.10 mIoU under the frozen OOF protocol. The secondary probes, oracle phenotypes, and correlations are explanatory only and cannot replace MLP-C as the primary decision probe.

## 21. Final frozen decision

MLP-C gains less than +0.10 mIoU under the frozen OOF protocol.

This audit now stops. It does not authorize a formal router, SSHR changes, test/LUAD/other-seed runs, feature additions, or hyperparameter/threshold tuning.

ROUTING_SIGNAL_NOGO
