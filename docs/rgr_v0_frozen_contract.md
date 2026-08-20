# RGR-v0 frozen contract

RGR-v0 starts from the official A0 commit `4e9a288` and A0 seed-42 final
checkpoint. It never loads an RSBR-trained checkpoint.

The only new trainable module is a one-layer complete directed graph over
coarse predicted semantic regions:

- node token: mean H28_1 feature over each region;
- edge feature: cosine similarity, normalized centroid distance, 3x3-dilation
  contact, and coarse-class agreement;
- relation gate: `Linear(4,16)-ReLU-Linear(16,1)-Sigmoid`;
- message: gated normalized neighbor aggregation;
- residuals: zero-initialized isolated `Linear(512,4)` plus graph
  `Linear(128,4)`;
- broadcast: semantic core only; transition pixels remain A0;
- `N=1`: message and graph residual are exactly zero.

The official loss, optimizer family, augmentation, BF16 precision, class
thresholds, TTA, fusion, and metric remain unchanged. The executor is hard
limited to parity, 32 real readiness batches, and (only after PASS) three
epochs on BCSS seed 42. It has no test, LUAD, other-seed, or 25-epoch path.
