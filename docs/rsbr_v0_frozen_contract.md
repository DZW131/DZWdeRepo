# RSBR-v0 Frozen Experimental Contract

RSBR-v0 is implemented from `baseline/official-a0` at commit
`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`. It is isolated from all archived
FAMPR, SC-MPR, CDSR, CLRR, and OSMF branches.

The model refines only `CAM28_1`. A detached released-fusion proposal is split
into 8-connected foreground components. Area-1 components are transition-only;
area >=2 components use the mean post-HFRM H28_1 feature as their 512-D region
token. The transition zone is the union of a 3x3 outer boundary and the fixed
top 25% feature-to-token cosine-deviation pixels, with flattened index used as
the deterministic tie-break.

Trainable additions are exactly:

- a zero-initialized `Linear(512, 4)` region semantic residual head;
- a `Conv1d(1541, 128, 1) -> ReLU -> Conv1d(128, 4, 1)` transition residual
  head whose final convolution is zero-initialized.

The original SSHR classification loss is retained, with the CAM28_1 slot using
the refined map. Region MIL has fixed weight 0.05 and residual L1 regularization
has fixed weight 0.01. No segmentation mask, region annotation, boundary
annotation, pseudo-mask, new teacher, threshold, prototype, or decoder enters
training.

Execution is hard-gated:

1. Full BCSS validation zero-init parity.
2. Exactly 32 real BCSS batches with A0 frozen and RSBR heads trainable.
3. Only `RSBR_V0_READINESS_PASS` unlocks a fresh-restart, frozen-A0, 3-epoch
   BCSS pilot with full validation after every epoch.

Test, LUAD, and 25-epoch training are outside this experiment.

