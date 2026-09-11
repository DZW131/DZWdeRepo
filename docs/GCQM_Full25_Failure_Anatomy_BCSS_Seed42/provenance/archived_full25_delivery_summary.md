# GCQM Final Full25 Project Delivery Summary

## What Was Implemented

- Dataset changes: no dataset content was changed; BCSS training and validation remained separated.
- Model changes: the frozen GCQM Phase-0 architecture was trained without adding parameters or auxiliary losses.
- Training engineering changes: added the reproducible Seed42 Full25 runner, checkpoint sealing, provenance capture, and train-only mechanism monitoring.
- Inference or evaluation changes: added same-machine, same-evaluator SSHR B0 reproduction, three-view TTA, fixed class thresholds, paired per-image comparison, and 10,000-sample bootstrap confidence intervals.
- Visualization changes: generated automatically selected five wins, five similar cases, and five failures, with predictions, error maps, query weights, and mask bases.
- README and documentation changes: added `docs/GCQM_FULL25_README.md`, the generated final validation report, and this delivery summary.

## Validation Evidence

- Local checks: focused GCQM Full25 tests passed, 6/6.
- Server-side checks: focused GCQM Full25 tests passed, 6/6; the earlier complete suite passed, 133/133.
- Training results: 25 epochs, 29,275 steps, all finite, 4,958.90 seconds, peak training VRAM 5.785 GiB.
- Evaluation results: protocol audit `COMPARABLE`; B0 archived metrics and per-image results were reproduced exactly on 3,418 validation images.
- Performance result: GCQM mIoU 64.3543% and mDice 78.0756%; SSHR B0 mIoU 66.6967% and mDice 79.7977%.
- Statistical result: GCQM minus B0 mIoU was -2.3424 pp, paired 95% CI [-3.0121, -1.6730] pp.
- Final decision: `GCQM_FULL25_NOGO`.
- Visualization artifacts: 176 files under `visualizations/`.

## Remaining Items

- What still needs to be run: nothing required for this Seed42 Full25 decision.
- What still needs verification: no mandatory verification remains; FLOPs were intentionally not reported because no validated FLOPs tool was available in this environment.
- What is intentionally deferred: multi-seed runs (11/17/42) are not started because the pre-registered Seed42 decision is NOGO.

## Final Command Index

- Training: see `../../GCQM_FULL25_README.md` and `tools/run_gcqm_full25_bcss_seed42.py`.
- Inference and evaluation: see `../../GCQM_FULL25_README.md` and `tools/eval_gcqm_full25_bcss_seed42.py`.
- Visualization: generated automatically by the evaluator; selected cases are under `visualizations/qualitative/`.

## Artifact Locations

- Server checkpoint: `/home/duyanhong/experiments/GCQM_Final_Full25_BCSS_Seed42/checkpoints/gcqm_full25_epoch25_final.pth`.
- Checkpoint SHA256: `6e1b909fc86a870e652213831521e8ff552371a083f85faad7dac3a21d969d0f`.
- Local report: `report/GCQM_BCSS_Seed42_Full25_Final_Validation_Report.md`.
- Local metrics: `evaluation/` and `metrics/`.
- Local mechanism records: `mechanism/`.
- Local visual outputs: `visualizations/`.
- Local provenance: `provenance/`.

## Handoff Notes

- Important assumptions: performance is decided only by the sealed E25 checkpoint and the paired BCSS validation comparison.
- Common failure points: do not select another epoch after seeing validation metrics, change fixed inference thresholds, or compare against a baseline trained/evaluated under another protocol.
- How to resume work: use the NOGO failure cases and per-class/per-image deltas to design a new mechanism; do not continue this GCQM branch into multi-seed confirmation.
