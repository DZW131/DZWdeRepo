# RSBR-v0 Parity R1 and 32-Batch Readiness Artifact Manifest

- Experiment code commit: `7cbe5aa0ad73d7e6827962f832bd50d6050d0b73`
- Server output root: `/home/duyanhong/experiments/RSBR_V0_PARITY_R1_AND_READINESS_7cbe5aa`
- Corrected parity decision: `RSBR_V0_PARITY_R1_PASS`
- Readiness decision: `RSBR_V0_READINESS_PASS`
- Three-epoch pilot started: no
- BCSS test or LUAD accessed: no

The complete human-readable delivery report is in
`docs/rsbr_v0_parity_r1_and_readiness_delivery.md`. JSON summaries, the
frozen contract, and driver logs are retained in this directory.

Full-validation prediction NPZ files remain under the server output root and
are intentionally not tracked in Git because of their size. Their SHA256
digests are recorded in `parity_r1/summary.json`.

The region head is initialized to exact zero, so its reported relative
movement uses an epsilon denominator and is not interpretable as a percentage.
The absolute movement norm (`2.328310e-02`) is the relevant evidence that the
head updated. The transition head has a nonzero initial first-layer norm, so
its relative movement (`8.707890e-04`) is directly interpretable.
