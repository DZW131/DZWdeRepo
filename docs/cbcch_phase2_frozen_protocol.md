# CBCCH Phase-2 Frozen Protocol

`EXP-CBCCH-002` tests whether local semantic affinity can solve the propagation-direction limitation observed in BC-CH.

## Confirmed implementation contract

- Only HFRM28_1 changes. C0, W1 and Phase-1 BC-CH are locked references.
- A2 uses `Y=P_affinity` at every pixel and contrastive supervision on every valid anchor.
- A3 uses `Y=(1-B)P_affinity+B*F` and contrastive supervision only on the exact top-20% detached `B` anchors.
- The local neighborhood is fixed at 15×15.
- Semantic feature `z_s` is the existing `ic1(F)` CAM probe after ReLU, image-label filtering for contrastive mining, and L2 normalization.
- Structural feature `z_h` is the L2-normalized three-vector of channel-mean `|LH|`, `|HL|`, `|HH|` responses.
- Each valid anchor selects one same-predicted-class maximum-similarity positive and one different-class maximum-dissimilarity negative.
- InfoNCE temperature is 0.07. Total loss is `L_official + 0.1 L_con`.
- There is no new classifier, projection head, learned affinity parameter, inference setting or metric.

## Matched continuation

Both A2 and A3 start independently from the SHA-locked common Epoch20 model, optimizer and RNG state and consume the same locked Epoch21–25 batch, augmentation and model-seed schedule. Evaluation uses only BCSS validation and only the Epoch25 FINAL checkpoint.

## Preregistered gates

- Gate A: A3 CAM28_1 mIoU must be greater than C0 CAM28_1 mIoU minus 0.1 percentage point.
- Gate B: A3 boundary accuracy must be greater than C0.
- Gate C: A3 final mIoU must be greater than C0.

No test set, LUAD, other seed, hyperparameter sweep, validation tuning or best-checkpoint selection is permitted.

## Reproduction command

```bash
bash tools/run_cbcch_phase2.sh \
  /path/to/BCSS-WSSS/training \
  /path/to/BCSS-WSSS/val \
  /path/to/WDCH_UTILITY_GATE_a00fb90 \
  /path/to/EXP-BCCH-001-f2a4c14 \
  /path/to/EXP-CBCCH-002_<commit> \
  4
```
