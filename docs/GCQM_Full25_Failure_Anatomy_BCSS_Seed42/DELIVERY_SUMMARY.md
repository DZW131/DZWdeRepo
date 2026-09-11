# GCQM Full25 Failure Anatomy Delivery Summary

## What Was Implemented

- Dataset changes: none; the audit reads only the frozen 3,418-image BCSS validation split.
- Model changes: none; the sealed GCQM E25 and frozen SSHR B0 checkpoints are read-only.
- Training engineering changes: none; no optimizer, backward pass, loss change, parameter update, threshold tuning, or checkpoint selection exists in this workflow.
- Inference and evaluation changes: added H1–H7 paired failure anatomy, fixed bootstrap statistics, global/pixel/top-k/oracle diagnostics, Phase-0 proxy-validity audit, and descriptive clustering.
- Visualization changes: generated nearest-centroid cluster representatives plus fixed wins/similar/failures with predictions, errors, boundary/contact bands, weights, bases, contributions, and top-k panels.
- README and documentation changes: added `docs/GCQM_FAILURE_ANATOMY_README.md`, the 28-section audit report, and this delivery summary.

## Validation Evidence

- Local checks: complete repository suite passed, 139/139.
- Server-side checks: focused Full25/anatomy suite passed, 12/12; one-image end-to-end smoke passed.
- Reproduction gate: exact GCQM mIoU 64.3543%, SSHR B0 mIoU 66.6967%, delta -2.3424 pp on 3,418 paired images.
- Frozen identities: GCQM SHA256 `6e1b909fc86a870e652213831521e8ff552371a083f85faad7dac3a21d969d0f`; SSHR SHA256 `b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70`.
- Main diagnosis: `MISSING_SPATIAL_COHERENCE`, confidence `HIGH`.
- Independent coherence evidence: GCQM minus SSHR component count +0.2912, hole count +0.2448, and compactness +0.4788; all three worsen in 4/4 classes.
- Supporting evidence: normalized FN delta +0.0227 with 95% CI [0.0179, 0.0277]; r=3 interior loss +0.0241 versus boundary loss +0.0114.
- Counter-evidence retained: contact-specific failure is false; basis purity is mixed; fixed top-k truncation does not improve the full-196 result.

## Remaining Items

- What still needs to be run: nothing for this frozen failure-anatomy decision.
- What still needs verification: Phase-0 proxy-to-validation predictive validity remains undetermined because no legitimate per-validation-image Phase-0 proxy records exist.
- What is intentionally deferred: all new architecture design and training. This audit authorizes only the next mechanism target, not GCQM-v2.

## Final Command Index

- Training: prohibited for this audit.
- Inference and evaluation: see `../GCQM_FAILURE_ANATOMY_README.md` and `tools/audit_gcqm_full25_failure_anatomy.py`.
- Result finalization: `python tools/finalize_gcqm_failure_anatomy.py --output-dir <audit-output>`.
- Visualization: generated automatically by the audit command.

## Artifact Locations

- Checkpoints: no new checkpoint was created; frozen checkpoint identities are recorded under `provenance/`.
- Logs: `provenance/failure_anatomy_run.log`.
- Metrics: `paired/`, `error_decomposition/`, `boundary_interior/`, `contact_region/`, `weight_diffuseness/`, `basis_purity/`, `spatial_property/`, and `correlation/`.
- Visual outputs: `visualizations/`.
- Final report: `report/GCQM_Full25_Failure_Anatomy_Decoder_Bottleneck_Audit_Report.md`.

## Handoff Notes

- Important assumption: GT is used only for post-hoc diagnosis and oracle ceilings after the formal Full25 decision.
- Common failure point: high weight diffuseness is correlated with worse images, but top-k truncation is uniformly inferior to full-196; do not reinterpret this audit as evidence that simple sparsification will improve the model.
- How to resume work: target region homogeneity/coherence while preserving CCRA, then write and freeze a separate experimental plan before any new training.
