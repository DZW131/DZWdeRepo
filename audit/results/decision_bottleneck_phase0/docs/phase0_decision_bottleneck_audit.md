# SSHR Phase-0 Decision Bottleneck Audit

## 1. Executive conclusion

Using the same frozen A0 checkpoint, the formal five-fold OOF class probe changes mIoU by -0.1603 pp, the image-class oracle by +1.6013 pp, and the pixel oracle by +6.7134 pp. Under the frozen hierarchy, the outcome is `NONLINEAR_ROUTING_REVIEW`. This is a scientific diagnosis only and does not make any oracle or validation-GT-fitted weight a deployable weakly supervised method.

Final frozen decision: **NONLINEAR_ROUTING_REVIEW**.

No SSHR training, test-set access, threshold tuning, or model change was performed. Validation GT was used only for diagnosis, oracle ceilings, and held-out-fold probe fitting.

## 2. Frozen protocol and parity

- Base commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- BCSS validation images/masks: 3418/3418.
- TTA: official three-way identity/horizontal/vertical.
- Official fusion: `0 CAM56 + 0.6 CAM28_1 + 0.2 CAM28_2 + 0.2 CAMdeep`.
- Class-presence thresholds: `0.8 / 0.9 / 0.8 / 0.6`.
- Main metric: released `tool.iouutils.scores()`.
- Released/audit differing prediction pixels: 0.
- mIoU absolute difference: 0.
- mDice absolute difference: 0.

Exact command:

```bash
python -u tools/audit_decision_bottleneck.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --output-dir /home/duyanhong/experiments/SSHR_DECISION_BOTTLENECK_PHASE0_f82eb0e \
  --num-workers 4
```

## 3. Individual hierarchy quality

| Prediction | mIoU | mDice | C0 | C1 | C2 | C3 |
|---|---|---|---|---|---|---|
| cam56 | 61.4651 | 75.7237 | 73.0720 | 67.9895 | 53.8312 | 50.9676 |
| cam28_1 | 67.0276 | 80.0461 | 76.2781 | 70.4882 | 57.5511 | 63.7928 |
| cam28_2 | 66.4982 | 79.6982 | 74.7899 | 69.4874 | 57.4119 | 64.3036 |
| camdeep | 64.9608 | 78.5478 | 73.6674 | 68.5058 | 55.3393 | 62.3307 |
| official_fusion | 67.3279 | 80.2680 | 76.4491 | 70.5718 | 57.8268 | 64.4640 |

## 4. Global static fusion

The best frozen grid point is `0.20/0.45/0.30/0.05` with mIoU 67.4909, a +0.1630 pp change from the same-run official baseline.

| Rank | w56 | w28_1 | w28_2 | wdeep | mIoU | Delta |
|---|---|---|---|---|---|---|
| 1 | 0.20 | 0.45 | 0.30 | 0.05 | 67.4909 | +0.1630 |
| 2 | 0.20 | 0.40 | 0.35 | 0.05 | 67.4890 | +0.1610 |
| 3 | 0.20 | 0.45 | 0.35 | 0.00 | 67.4876 | +0.1597 |
| 4 | 0.20 | 0.40 | 0.30 | 0.10 | 67.4874 | +0.1595 |
| 5 | 0.20 | 0.45 | 0.25 | 0.10 | 67.4870 | +0.1591 |
| 6 | 0.15 | 0.50 | 0.25 | 0.10 | 67.4867 | +0.1588 |
| 7 | 0.15 | 0.50 | 0.30 | 0.05 | 67.4866 | +0.1586 |
| 8 | 0.15 | 0.45 | 0.30 | 0.10 | 67.4864 | +0.1585 |
| 9 | 0.20 | 0.50 | 0.25 | 0.05 | 67.4854 | +0.1575 |
| 10 | 0.15 | 0.45 | 0.25 | 0.15 | 67.4841 | +0.1562 |

## 5. Class preference and unique evidence

| Class | cam56 | cam28_1 | cam28_2 | camdeep | Best | Gap vs CAM28_1 |
|---|---|---|---|---|---|---|
| 0 | 73.0720 | 76.2781 | 74.7899 | 73.6674 | cam28_1 | +0.0000 |
| 1 | 67.9895 | 70.4882 | 69.4874 | 68.5058 | cam28_1 | +0.0000 |
| 2 | 53.8312 | 57.5511 | 57.4119 | 55.3393 | cam28_1 | +0.0000 |
| 3 | 50.9676 | 63.7928 | 64.3036 | 62.3307 | cam28_2 | +0.5107 |

Different classes prefer different hierarchies: **True**. CAM56 contributes 2,956,418 unique-correct foreground pixels (1.8636%). Against official fusion it has 4,871,914 recoverable and 8,699,810 harmful pixels (net -3,827,896).

Full pairwise, per-class unique, and recoverability tables are stored under `tables/`.

## 6. Error-set geometry

The mean off-diagonal foreground error-set Jaccard is 0.6710 (moderate overlap).

| Branch i | Branch j | Jaccard |
|---|---|---|
| cam56 | cam28_1 | 0.6334 |
| cam56 | cam28_2 | 0.5678 |
| cam56 | camdeep | 0.5706 |
| cam28_1 | cam28_2 | 0.7608 |
| cam28_1 | camdeep | 0.7064 |
| cam28_2 | camdeep | 0.7870 |

## 7. Oracle ceilings

| Method | mIoU | mDice | Delta |
|---|---|---|---|
| official_fusion | 67.3279 | 80.2680 | +0.0000 |
| image_oracle | 69.3340 | 81.6916 | +2.0061 |
| image_class_oracle | 68.9292 | 81.4087 | +1.6013 |
| pixel_oracle | 74.0414 | 84.9193 | +6.7134 |

Pixel-oracle coverage is 87.5626% with 19,730,576 unrecoverable foreground pixels.

## 8. Five-fold class-conditioned linear probe

The formal GroupKFold OOF probe uses 16 scalars, 22 source slides, Adam lr=0.05, 500 fixed steps, and image batch=16.

| Method | mIoU | mDice | Delta mIoU |
|---|---|---|---|
| Official | 67.3279 | 80.2680 | — |
| 5-fold OOF probe | 67.1676 | 80.1471 | -0.1603 |

Every image appears exactly once out-of-fold and no source group crosses a fold boundary.

## 9. Calibration and evidence comparability

| Branch | Mean entropy | Mean max confidence | Foreground coverage |
|---|---|---|---|
| cam56 | 1.3583 | 0.3419 | 100.00% |
| cam28_1 | 1.3576 | 0.3447 | 100.00% |
| cam28_2 | 1.3582 | 0.3409 | 100.00% |
| camdeep | 1.3729 | 0.3051 | 100.00% |

No temperature, ECE, Platt, or isotonic calibration was fitted.

## 10. Automatically selected qualitative evidence

24 validation panels were selected solely by frozen recoverable-error counts across Types A–D; none were hand-picked. See `figures/qualitative/` and `tables/qualitative_manifest.csv`.

## 11. Answers to the ten preregistered questions

1. Individual hierarchy mIoUs are listed in Section 3; official is 67.3279%.
2. Per-class best hierarchies are: C0=cam28_1, C1=cam28_1, C2=cam28_1, C3=cam28_2.
3. Different classes have different scale preferences: True.
4. CAM56 unique evidence: 2,956,418 pixels; net vs official -3,827,896.
5. Branch error sets have moderate overlap (mean Jaccard 0.6710).
6. Best global static fusion gain: +0.1630 pp.
7. Image oracle gain: +2.0061 pp.
8. Image-class oracle gain: +1.6013 pp.
9. Pixel oracle gain: +6.7134 pp.
10. Held-out 5-fold probe gain: -0.1603 pp.

## 12. Frozen decision

The image-class oracle is large but the frozen 16-scalar OOF probe is weak; only a human review of low-capacity nonlinear routing is permitted.

This audit stops here. It does not authorize UCER, a nonlinear router, test evaluation, LUAD, threshold changes, or any model implementation without human review.

NONLINEAR_ROUTING_REVIEW
