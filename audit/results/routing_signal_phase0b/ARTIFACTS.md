# Phase-0B artifact manifest

This directory is the lightweight, reviewable archive of the frozen-model
BCSS validation-only Phase-0B routing-signal learnability audit.

- Audit source commit: `76ffbf4d61f77dd6ef04946c0ffa16643b9acd86`
- Frozen parent commit: `f1a95059cd7914e9d6b72e08ec135c4c8ea32c06`
- Frozen A0 baseline commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Delivery bundle SHA256: `93d7f98d18dee8a688ed6e3c16544485d2e0cd4c52008e1b1d0d3b7e510ce951`

Included:

- the final report and machine-readable summary;
- all 28 CSV evidence tables;
- all 39 deterministic figures, including 32 preregistered qualitative panels;
- the frozen contract, environment record, signal manifests, and feature names;
- the formal run log (`run_retry1.log`);
- the retained first startup log (`run.log`), which records a pre-audit Python
  import-path failure. No model forward, oracle, probe fitting, or evaluation
  occurred in that failed attempt.

Excluded from Git:

- the approximately 1.1 GB array cache and reconstructed prediction masks;
- the BCSS images and masks;
- the frozen model checkpoint.

Those large artifacts remain on the 5090 server under
`/home/duyanhong/experiments/SSHR_ROUTING_SIGNAL_PHASE0B`. Their provenance and
integrity-critical metadata are captured by `cache/signal_manifest.json`,
`config/frozen_contract.json`, `summary.json`, and the raw CSV tables.

The formal run evaluated BCSS validation only. It performed no SSHR training,
test-set evaluation, LUAD evaluation, threshold search, or inference change.
