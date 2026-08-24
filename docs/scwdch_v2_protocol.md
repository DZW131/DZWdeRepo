# EXP-WDCH-002: SC-WDCH Strength Calibration

This branch implements the frozen strength-calibration experiment on top of
WD-CH v1.  Only HFRM28_1 changes:

```text
WD = IDWT(CH(LL), LH, HL, HH)
SC-WD = F + s * (WD - F)
```

The fixed scalar is calculated once from the full BCSS training split and the
common seed42 epoch20 state:

```text
s = mean RMS(CH15(F)-F) / mean RMS(WDCH(F)-F)
```

Validation and test are forbidden during calibration.  The scalar is stored as
a non-learnable persistent buffer.  W2 resumes epochs 21-25 from the same model,
optimizer, scheduler, RNG and batch/augmentation schedule as the hash-verified
C0/W1 controls.  Epoch25 FINAL is the only primary checkpoint.

Server runner:

```bash
bash tools/run_scwdch_v2.sh \
  TRAIN_ROOT VAL_ROOT COMMON_EPOCH20 SCHEDULE PHASE0_SUMMARY \
  C0_DIR W1_DIR EXPERIMENT_DIR NUM_WORKERS
```

Primary output:

```text
reports/scwdch_v2_strength_calibration_final_report.md
```
