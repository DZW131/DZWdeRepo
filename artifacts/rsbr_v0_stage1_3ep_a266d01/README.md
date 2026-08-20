# RSBR-v0 Stage-1 Three-Epoch Pilot Artifact Manifest

- Experiment code commit: `a266d0129edd52e537807ce45bf6ab58f34a9e29`
- Report clarification commit: `55d3c30911006ac8e855267f324430954a164bb7`
- Server output root: `/home/duyanhong/experiments/RSBR_V0_PILOT_3EP_a266d01`
- Final decision: `RSBR_V0_PILOT_REVIEW`
- Secondary flag: `REGION_SEMANTIC_SIGNAL`
- Best epoch: 1
- Best paired delta: `+0.02711616` mIoU percentage points
- Epoch-3 paired delta: `+0.02279312` mIoU percentage points
- Test, LUAD, additional seeds, or further training: not run

`REVIEW` is a mechanism-only exception, not a performance GO. The best paired
delta is below the ordinary `+0.05 pp` review threshold, while the frozen
mechanism rule is met because Core-only is positive, Full does not improve on
Core-only, and Type-B errors decrease.

The complete report is in `docs/rsbr_v0_3epoch_pilot_report.md`. Compact
configuration, training summaries, paired validation summaries, and the
epoch-3 ablation are tracked here. Per-step JSONL logs and the four checkpoint
files remain under the server output root.

Checkpoint SHA256 values:

- `epoch1_rsbr.pth`: `c530486b00683ff9dc24fe4d19ce4d79300cf6301172f9b4dbeb81c875c395ca`
- `epoch2_rsbr.pth`: `ecd0d67df5fb2a91164d7c8314daf749378182801ee29601c7bf29c8500bb846`
- `epoch3_rsbr.pth`: `d39358b46de304dfeba052c0b6ddcc22b24fe60af897210903ec6b52dc8bbe36`
- `best_val_rsbr.pth`: identical to Epoch 1 (`c530486b...`).
