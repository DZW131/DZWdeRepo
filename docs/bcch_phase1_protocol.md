# BCCH Phase-1 Frozen Protocol

`EXP-BCCH-001` tests whether fixed wavelet-derived boundary energy can identify
locations where original SSHR CH15 should be weakened. It is a mechanism gate,
not a contrastive-learning experiment.

Only HFRM28_1 changes. For its input feature `F`:

```text
(LL,LH,HL,HH) = HaarDWT(F)
E_HF = sqrt(LH^2 + HL^2 + HH^2)
B = bilinear_upsample(spatial_minmax(channel_mean(E_HF)))
alpha = 1 - detach(B)
BCCH(F) = alpha * CH15(F) + (1-alpha) * F
```

Normalization is per image over the half-resolution spatial grid. Bilinear
upsampling uses `align_corners=False`. No alpha floor, scale, threshold or
validation-selected setting is permitted. The original CH15 parameter name,
value and optimizer state are preserved exactly. Haar filters are fixed buffers;
there are no new trainable parameters.

C0 and W1 reuse the SHA-locked prior matched artifacts. BC-CH starts from the
same seed42 common Epoch20 checkpoint, restores the complete optimizer state,
and follows the same Epoch21–25 batch order, augmentation seeds, model seeds,
BF16 precision, official loss/optimizer/poly schedule and final-checkpoint-only
validation rule.

```bash
bash tools/run_bcch_phase1.sh \
  /path/to/BCSS-WSSS/training \
  /path/to/BCSS-WSSS/val \
  /path/to/WDCH_UTILITY_GATE_a00fb90 \
  /path/to/EXP-BCCH-001_<commit> \
  4
```

The runner verifies all locked hashes, performs a real batch20 BF16 no-step
preflight, trains BC-CH once, evaluates Epoch25 FINAL, and generates
`reports/bcch_phase1_boundary_aware_final_report.md`. Test, LUAD, other seeds,
contrastive loss, new classifiers, GSR changes, inference changes and tuning are
forbidden.
