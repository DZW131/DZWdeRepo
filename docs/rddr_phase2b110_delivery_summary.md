# Phase-2B1.10 Project Delivery Summary

## What Was Implemented

- Dataset changes: none. All 3418 validation image identities replayed from immutable caches; no split files accessed.
- Model changes: none. Pure A0 source remains unchanged; no innovation module imported.
- Training engineering changes: none. Zero optimizer steps, no model loading, no checkpoint creation.
- Evaluation: standalone residual headroom, S_D primary ranking, four fixed controls, composition/strata, frozen context rescue/harm and 10k paired-image bootstrap.
- Verification: independent implementation of ranking, context, margin derivative, counts, bootstrap denominator and decisions, plus 44 tests.
- Visualization: report tables only. No model visualizations or new image assets required.
- Documentation: frozen contract, full 29-section report, runnable README, deterministic renderer and SHA manifest.

## Validation Evidence

- Server main run: `61d1a8afa8a58b5b18087207040e08460078a91b`; verifier: `d904bf8eeca7f7dc55bf5f243cc6c9c03f8118d6`.
- 44 tests pass, no skips; 26 independent checks pass.
- Support and context exact FP32 replay error 0; q recomputation error 5.96046448e-8, inherited q retained.
- Independent FP64 context error 1.86375718e-7; independent rank error 2.22044605e-16; bootstrap interval error 2.66453526e-15.
- Main audit 14.5374 seconds; probability-only GPU replay peak allocated 15.0781 MiB, reserved 24 MiB. These are not training or network-forward resource measurements.
- New input/checkpoint/source identity checks pass. Prior state/BN/prediction identity is explicitly inherited rather than claimed as new testing.
- Training results: not applicable. Segmentation performance evaluation: not rerun; conditional native28 mechanism metrics only.

## Result

Gate A/B/C/D = PASS/FAIL/FAIL/FAIL. Theoretical missing beneficial count is 31,266; residual beneficial count is 177,865. S_D utility image AUC=0.50017874 and winner AUC=0.60827404 do not meet preregistered gates. Rejected Both-Wrong context accuracy/rescue=33.710374%, supporting only a separate third-evidence audit. One-correct context intrusion and total error remain important limitations.

`DECISION = RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED`

## Remaining Items

- Nothing remains to run for this phase.
- User PR review remains; do not auto-merge.
- Intentionally deferred/prohibited: recovery gate construction, training, test/LUAD/extra seed, threshold/lambda search, new context loss.
- Future work is not unlocked as training: the GT-defined Both-Wrong group needs independent GT-blind identification and one-correct safety evidence under a new contract.

## Final Command Index

- Training: prohibited/not applicable.
- Inference: prohibited/not applicable; only frozen probability replay.
- Evaluation/tests/report: exact runnable commands in [README_rddr_phase2b110.md](README_rddr_phase2b110.md).
- Visualization: read the tables in [the full report](rddr_phase2b110_residual_correction_coverage_report.md).

## Artifact Locations

- Server checkout: `/home/duyanhong/DZWdeRepo-rddr-phase2b110`.
- Server immutable run: `/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1`.
- Logs: `/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1.log` and `verify_r1.log`.
- Local checkout: `G:/05_科研工作/SSHR/DZWdeRepo-rddr-phase2b110`.
- Metrics: `audit/results/rddr_phase2b110/` with CSV/JSON, bootstrap replicates, tests and manifest.
- Existing checkpoint and inputs: exact paths/SHAs in runtime JSON and report section 1; originals preserved.

## Handoff Notes

Use a fresh directory to rerun. Do not overwrite completed `formal_r1` or mutate pinned hashes. Report bytes can be regenerated without reading the checkpoint/cache, using only small committed evidence. A q control outperforming the S_D primary does not authorize score substitution. A positive local derivative or GT-conditioned context rescue is not a proven finite-training improvement.
