# WD-CH Utility Gate Protocol

This branch starts from the isolated official A0 baseline
`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`. The released model, loss,
optimizer, preprocessing, inference and metric files remain unchanged.

The only candidate model change is at HFRM28_1:

```text
CH15(F) -> IDWT(CH_k*(LL), LH, HL, HH)
```

The Haar filters are fixed orthonormal buffers. `k*` is selected once from
5/7/9 by an impulse-response receptive-field distance in Phase 0 and cannot be
selected by segmentation performance. HFRM56, HFRM28_2, GSR, CAM heads, loss,
inference and metrics are untouched.

## Formal execution

```bash
bash tools/run_wdch_utility_gate.sh \
  /path/to/BCSS-WSSS/training \
  /path/to/BCSS-WSSS/val \
  /path/to/a0_seed42_final/stage1_last.pth \
  /path/to/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  /path/to/WDCH_UTILITY_GATE_<commit> \
  4
```

The runner stops immediately if Phase 0 fails or if the frozen W1 intervention
meets the preregistered catastrophic-failure condition. Otherwise it creates a
fresh seed42 C0 state through epoch20, records model/optimizer/RNG/sampler
state, and runs C0 and W1 on the same epoch21–25 schedule. The primary result
is epoch25 FINAL only.

## Output index

- `reports/wdch_phase0_engineering_audit.md`
- `reports/wdch_reconstruction_metrics.csv`
- `reports/wdch_band_energy.csv`
- `reports/wdch_receptive_field_matching.csv`
- `reports/wdch_feature_statistics.csv`
- `reports/wdch_frozen_intervention_report.md`
- `matched/common/common_epoch20.pth`
- `matched/C0/checkpoints/epoch25_final.pth`
- `matched/W1/checkpoints/epoch25_final.pth`
- `reports/wdch_utility_gate_final_report.md`

No test set, LUAD, other seed, best-epoch selection, alternate wavelet, learned
gate, attention, new loss or inference change is authorized in this gate.
