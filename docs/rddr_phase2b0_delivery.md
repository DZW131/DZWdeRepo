# Phase-2B0 delivery / verification record

Completed 2026-08-30. Decision: **RDDR_PHASE2B0_NOGO** (A/B FAIL; C/D PASS).

## What was implemented

- Independent branch from pure A0 `4e9a288`; no network, optimizer, preprocessing,
  loss, inference or official metric source changes.
- GT-independent U/SR/SC/SRSC relation and neighbor-probability audit.
- Existing immutable populations reused, SHA/count checked on all 3418 images.
- Streaming foreground pair/purity diagnostics, actual-neighborhood Mass/N_eff,
  deep/Top20 intersections, oracle coverage, fixed-bin error validation.
- Paired 10,000-image-bootstrap statistics and complete 25-section Markdown.

## Validation evidence

- 21/21 server unit tests PASS; log in
  `audit/results/rddr_phase2b0/tests_final_20260830.txt`.
- CUDA BF16 two-image smoke passed. Two initial failed smoke outputs were
  preserved: A0 eval() return-value compatibility and NumPy/Torch q-division
  arithmetic were corrected in the audit wrapper, not in A0 or score equations.
- Complete 3418-image extraction: cached native q exact, model-state digest
  unchanged, strict checkpoint load, input checkpoint SHA unchanged.
- An initial extraction omitted Top20/deep intersections; the complete extraction
  added these observations plus explicit oracle purity, without changing any
  relation equation. The initial output remains preserved, is not used for gates.
- Independent NumPy/CSV verification passed on the server and local Windows:
  population partitions, 4x4 confusion matrices, 32 independently recomputed
  bootstrap replicates, all 10,000-replicate percentile CI endpoints, gate logic.
- Intermediate report outputs were preserved. Final report uses unchanged
  complete sufficient statistics; an NPZ read-performance fix and report-label
  corrections did not change estimates or decisions.
- Report SHA256 matches server/local:
  `a9bda67a2b74bc489cd119aa488383f97ac9c39d79833195821cfcac6c43e860`.

## Primary result

| Check | Observed | Frozen threshold | Result |
|---|---:|---:|---|
| Image-balanced pair AUROC | 0.622374 | >=0.65 | FAIL |
| Image-balanced purity gain | +1.167624 pp | >=+3 pp | FAIL |
| Corrected–Harmed mechanism | positive CI / Harmed gain | all positive | PASS |
| Training-free neighbor utility | accuracy/mIoU/Top20 improve | all required | PASS |

The native-grid neighbor mIoU gain of +7.837882 pp is **not** a final SSHR
inference gain. Deep-Wrong targets lose substantial accuracy; conditional
safety and weak Q5 purity gain are disclosed. Secondary utility cannot override
the preregistered A/B gates. No score or hyperparameter was changed after results.

## Artifact locations

- Canonical report: `docs/rddr_phase2b0_reliable_relation_feasibility_report.md`.
- Full machine-readable delivery: `audit/results/rddr_phase2b0/`.
- Server final report/CSV:
  `/home/duyanhong/experiments/RDDR_PHASE2B0/report_final/`.
- Raw sufficient-statistic NPZ, exact-bin checks and extraction runtime:
  `/home/duyanhong/experiments/RDDR_PHASE2B0/formal_complete_r1/`.
- NPZ SHA256:
  `f66b0717a4a0bddeb9bb84699b98f74f8eb5cd52ab88e6ecd0a86dc8a10e04d3`.
- Server checkout:
  `/home/duyanhong/DZWdeRepo-rddr-phase2b0-10d4c6f`.
- Local intermediate report remains under ignored `audit/cache/`; nothing from
  prior experiments, baseline checkpoints or dataset assets was removed.

## Final command index and remaining items

Runnable commands/environment/data contract: `docs/README_rddr_phase2b0.md`.
The complete report also records exact extraction and analysis commands.
Training and model visualization are intentionally not part of this audit.
No additional requested audit work remains. Do not launch Phase-2B training,
test, LUAD, other seeds or posthoc searches. PR is for review, not auto-merge.
