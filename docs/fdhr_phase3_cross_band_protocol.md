# FDHR Phase-3 Cross-Band Interaction Protocol

`EXP-FDHR-003` asks only whether a minimal cross-band interaction repairs the
semantic degradation observed in WD-CH W1. It starts from the SHA-locked common
SSHR seed42 Epoch 20 checkpoint and reuses the locked C0/W1 matched artifacts.

At HFRM28_1 only, all candidates use the Phase-0-locked Haar transform and
depthwise `CH7(LL)`:

- A: `IDWT(CH7(LL), 1.1 LH, 1.1 HL, 1.1 HH)`.
- B: `LL' = CH7(LL) * (1 + 0.1 (|LH|+|HL|+|HH|))`.
- C: `LL' = CH7(LL) + 0.1 mean(LH,HL,HH)`.

For C, `Pool(HF)` is fixed as arithmetic mean pooling over the three Haar-band
axis. Haar high-frequency coefficients already have LL spatial resolution;
therefore no additional spatial pooling, interpolation or resize is introduced.
The 0.1 strengths are fixed buffers. There are no new trainable parameters
relative to W1.

All candidates continue through Epoch 21–25 with the same frozen batch order,
augmentation seeds, model seeds, optimizer state, poly schedule, BF16 protocol,
loss, inference and official metric. Only the Epoch 25 FINAL checkpoint is a
primary result. Test, LUAD, other seeds, tuning and best-checkpoint selection are
forbidden.

```bash
bash tools/run_fdhr_phase3.sh \
  /path/to/BCSS-WSSS/training \
  /path/to/BCSS-WSSS/val \
  /path/to/WDCH_UTILITY_GATE_a00fb90 \
  /path/to/EXP-FDHR-003_<commit> \
  4
```

The runner verifies the locked common state, schedule, C0 and W1 SHA256 values,
runs a batch20 BF16 no-step preflight, trains A/B/C sequentially, evaluates the
final validation predictions, and emits
`reports/fdhr_phase3_cross_band_final_report.md`.
