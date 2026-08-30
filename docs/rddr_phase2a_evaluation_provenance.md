# Phase-2A evaluation provenance

## Immutable training

GS and RCS were trained at implementation commit `6f45ac7`, each from official
pretrained weights for 25 epochs. This delivery did not resume or retrain either
model. Their final checkpoints, earlier diagnostic milestones, source hashes,
and optimizer records remain unchanged.

## Evaluation corrections made before delivery

1. The copied audit helper converted each TTA output to FP32 before averaging.
   The official implementation averages the native tensor dtype first. The
   audit helper was corrected to match the unchanged official function; no
   threshold, fusion weight, normalization, model or metric was changed.
2. The first complete evaluation (`123a19a`, server `report/`) used current
   evaluation backend settings to regenerate Phase-0 groups. Independent count
   checks found small drift. Those results are preserved, but superseded.
3. The original Phase-0 code (`586f402`), original C0 checkpoint and fresh-process
   backend settings were replayed. All 3,418 images match their historical CSV
   counts for each of four CH groups and Top20. The resulting immutable cache
   has a SHA256 manifest. Original full pixel masks were not retained by
   Phase-0; the available historical verification is per-image count equality,
   not comparison against historical mask-file hashes.
4. The intermediate corrected-population evaluation (`af9cc6d`, server
   `report_verified/`) stopped on a check against the historical rounded C0
   score, before issuing a report. That number came from the older audit
   reduction order; it is not the authoritative numeric reference for native
   official inference. The tolerance was not widened to obtain a desired
   result. It was replaced with full-split equality against unchanged official
   `infer()` for **each** of C0, GS and RCS (`1d62505`, server `report_final/`).

All three final models use the same official evaluation pipeline. Historical
score differences are reported rather than tuned away. Fixed risk/CH groups
come only from the historical C0 replay, never from a candidate's q. No gate,
checkpoint choice, training setting or scientific hypothesis was changed in
response to results.

## Locations

- Experiment root: `/home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7`
- Official training source: `/home/duyanhong/DZWdeRepo-rddr-phase2a-6f45ac7`
- Final evaluator: `/home/duyanhong/DZWdeRepo-rddr-phase2a-final-eval`
- Population cache: experiment root plus `diagnostics/frozen_phase0_populations`
- Final outputs: experiment root plus `report_final/`
- Local canonical delivery: `audit/results/rddr_phase2a/` and the report in `docs/`

The cache is approximately 593 MiB and is not committed to Git. No checkpoints
or previous outputs were deleted or overwritten in this evaluation.
