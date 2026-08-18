# RSBR-v0 Stage -1R Frozen Contract

Stage -1R is an audit-harness diagnosis based on RSBR-v0 commit `98e15df` and
the frozen SSHR A0 checkpoint. It does not modify the RSBR model or its zero
initialization, region extraction, transition masks, fusion, class-presence
logic, thresholds, TTA, or metric.

The audit performs:

1. exact same-process structural identity on a fixed 32-image subset;
2. three independent full BCSS-validation A0 inference processes;
3. three independent full BCSS-validation zero-init RSBR processes;
4. preregistered paired cross-model comparisons;
5. production, deterministic-algorithm, FP32, and TF32-off diagnostics on the
   same fixed subset;
6. residual-merge tensor hash, dtype, zero, and contiguity checks.

The audit contains no optimizer and performs no training. BCSS test, LUAD,
Stage 0, and the three-epoch pilot are forbidden. It ends with exactly one of
the four decisions specified in the Stage -1R technical protocol.

