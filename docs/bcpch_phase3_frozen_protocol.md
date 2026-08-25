# BCP-CH Phase-3 Frozen Protocol

`EXP-BCPCH-003` tests whether a low-frequency semantic prototype can pull boundary features toward a valid class manifold while CBCCH contrastive learning pushes apart incorrect local neighbors.

## Architecture contract

- Only HFRM28_1 changes relative to the locked CBCCH implementation.
- The feature embedding is exactly `z=L2(IDWT(LL,0,0,0))`.
- Confidence masks use per-class spatial min-max normalized `ReLU(ic1(F)) > 0.70`; the Boolean selection is detached.
- Image presence uses the existing `fc8(feat_deep)` and official BCSS thresholds. If none passes, the maximum-probability class is retained.
- Each valid class prototype is `normalize(mean(z_i))` over its confident pixels.
- `P_prototype(i)=sum_c softmax(z_i^T P_c)P_c` over valid present classes.
- If an image has no valid prototype, `P_prototype=P_affinity` exactly.
- Final propagation is `Y=(1-B)(0.5P_affinity+0.5P_prototype)+BF`.
- Existing local15 affinity, top-20% detached-B anchors, τ=0.07 and `L=L_official+0.1L_con` remain unchanged.
- No new classifier, projection head, trainable parameter, loss, inference setting or metric is introduced.

## Matched continuation and gates

Only BCP-CH is trained from the SHA-locked public Epoch20 state through Epoch21–25. It consumes the same seed42 batch, augmentation and model-seed schedule, optimizer, poly LR schedule and BF16 protocol. The only selected checkpoint is Epoch25 FINAL.

- Gate A: boundary accuracy > 52.2525%.
- Gate B complete: CAM28_1 >= 66.4431%; partial: CAM28_1 >= 65.9035%.
- Gate C: final mIoU > 66.8555%.

The full validation split is used to compare GT-category boundary-to-prototype cosine similarity at initial Epoch20 and final Epoch25. This is observation-only. Test, LUAD, other seeds, tuning and best-checkpoint selection are forbidden.

## Command

```bash
bash tools/run_bcpch_phase3.sh \
  /path/to/BCSS-WSSS/training \
  /path/to/BCSS-WSSS/val \
  /path/to/WDCH_UTILITY_GATE_a00fb90 \
  /path/to/EXP-BCCH-001-f2a4c14 \
  /path/to/EXP-CBCCH-002-8057faa \
  /path/to/EXP-BCPCH-003_<commit> \
  4
```
