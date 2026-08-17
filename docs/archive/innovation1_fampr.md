# Archived Innovation 1: FA-MPR

## Status

Frequency-Adaptive Morphology-Preserving Rectification (FA-MPR) was archived
on 2026-08-16. It remains selectable as `--context-mode fampr` only so the
experiment can be reproduced. It is not an active development or training
path, and no source, test, checkpoint reference, or report was deleted.

## Evidence for archival

The frozen BCSS seed-42 final-checkpoint comparison showed no stable aggregate
improvement. On the complete validation split, A0 achieved 67.3102 mIoU and
80.2563 mDice, while Full FA-MPR achieved 66.8062 mIoU and 79.8850 mDice:

| Metric | A0 | Full FA-MPR | Delta (pp) |
|---|---:|---:|---:|
| validation mIoU | 67.3102 | 66.8062 | -0.5040 |
| validation mDice | 80.2563 | 79.8850 | -0.3713 |

The final-checkpoint test mIoU change was only +0.0596 pp and therefore
inconsistent with validation. The test-only class-3 gain (+0.9843 IoU pp) also
reversed on validation (-1.1524 IoU pp). The defensible conclusion is that the
effect was split-sensitive rather than a stable semantic improvement.

The class-response diagnosis did identify a useful research clue: shallow
CAM56 recovered some class-3 boundaries and small components, but that signal
did not propagate reliably to the official fused CAM and incurred losses in
large/interior regions. That observation motivated semantic verification in
the later SC-MPR candidate; FA-MPR itself is not being tuned further.

## Preserved record

- implementation: `network/fampr/`
- controls: `tests/test_fampr.py`
- smoke/profile tools: `tools/smoke_fampr.py`, `tools/profile_fampr.py`
- original implementation report: `docs/fampr_implementation_report.md`
- frozen class-response diagnosis: GitHub PR #5,
  <https://github.com/DZW131/DZWdeRepo/pull/5>
- final implementation source: `main@e4b7b6c`

The archived HST candidate is independently preserved in
`docs/innovation1_hst_migration.md` and remains incompatible with alternative
HFRM context modes.
