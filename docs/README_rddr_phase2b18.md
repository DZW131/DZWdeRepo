# Phase-2B1.8 pre-rectification guidance audit

## Purpose / outcome

Audit whether the frozen symmetric teacher can supervise raw pre-HFRM28_1 semantics safely.
Full [38-section Chinese report](rddr_phase2b18_prerectification_teacher_guidance_report.md).
Approved [pre-outcome contract](rddr_phase2b18_contract.md).

A/B/C/D/E = PASS/PASS/FAIL/FAIL/PASS.
`DECISION = TEACHER_BETTER_THAN_RAW_BUT_GRADIENT_UNSAFE`.
No Full25, optimizer, checkpoint writes, test, LUAD or hyperparameter search.

## Environment

Existing server: `duyanhong@10.15.20.77`, RTX5090 D v2.
Python: `/home/duyanhong/miniconda3/envs/sshr5090/bin/python`.
PyTorch2.11.0+cu128, NumPy1.23.5, original A0 torchvision/Pillow dependencies.
No environment upgrades/installations were needed. Use the existing environment and runner backend assertions.
BF16 network forward/backward; FP32 loss/logit/q derivative; FP64 diagnostic accumulators.
The report renderer uses only the Python standard library and can run locally without torch.

## Repository structure

- `tools/rddr_phase2b16_common.py`: frozen audit math and utility functions, not an innovation model.
- `tools/rddr_phase2b18_common.py`: frozen-head logits, Uraw/FAraw/PRG, q derivative, strata and decision.
- `tools/run_rddr_phase2b18_audit.py`: real3418 replay/backward, shared-head diagnostic, fixed160 identity and batch20.
- `tools/analyze_rddr_phase2b18.py`: complete metrics, native confusion matrices, image bootstrap and gates.
- `tools/verify_rddr_phase2b18.py`: independent implementation, no imports from primary audit/analysis.
- `tools/render_rddr_phase2b18_report.py`: deterministic38-section report and artifact hash manifest.
- `tests/test_rddr_phase2b18.py`: 13 mathematical/guardrail tests.
- `tests/test_rddr_phase2b18_artifacts.py`: 24 required real-run integration checks.
- `audit/results/rddr_phase2b18/`: small CSV/JSON, all10000 bootstrap replicates, per-image losses, tests and hashes.
- Original `network/`, `tool/`, `train_sshr.py`: unchanged from pure A0 `4e9a288`.

## Data / immutable assets

All3418 sorted BCSS validation images must match the Phase2B1 native cache, Phase2B1.5 symmetric cache and
Phase2B1.6 rect cache. Paths and SHA256 are in the report and `rddr_phase2b18_runtime.json`.
Checkpoint is C0 Full25 seed42 final, SHA256 `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
No raw-logit cache existed in the original native archive: compare real original/frozen-head logits exactly,
then verify their softmax against old ps. Do not invert ps with log to manufacture historical logits.

Loss covers all784 positions per image, including background/ignore; GT only enters diagnostics.
Native28 metrics use foreground0–3, excluding bg4/ignore255, absent-union class NA.
No official background correction in diagnostic confusion matrices; original official inference is untouched.

## Actual completed run

- Worktree: `/home/duyanhong/DZWdeRepo-rddr-phase2b18`.
- GPU artifacts: `/home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1`.
- GPU log: `/home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1.log`.
- Canonical statistics and verification: `/home/duyanhong/experiments/RDDR_PHASE2B18/report_r1`.
- Large observation NPZ (server-only) stores raw logits, three gradients, q derivatives, feature reductions,
  per-image parameter energies and loss records, not multi-GB full feature gradients.

## Command index

**Training: none; prohibited.** The commands below document an optional zero-update replay, not an instruction
to start another run. Use NEW output directories; do not overwrite the completed experiment.

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b18
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
$PY tools/run_rddr_phase2b18_audit.py \
  --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz \
  --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz \
  --previous /home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1/rddr_phase2b16_gradient_observations.npz \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --val-images /home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img \
  --output /home/duyanhong/experiments/RDDR_PHASE2B18/replay_001

$PY tools/analyze_rddr_phase2b18.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B18/replay_001 \
  --output /home/duyanhong/experiments/RDDR_PHASE2B18/replay_report_001

$PY tools/verify_rddr_phase2b18.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B18/replay_001 \
  --report /home/duyanhong/experiments/RDDR_PHASE2B18/replay_report_001
```

Original infer is exercised only by the runner's fixed160 before/after identity test; no additional segmentation
evaluation is required. The hash is taken before the official background overwrite.

### Tests against completed artifacts

```bash
RDDR_PHASE2B18_RUN=/home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1 \
RDDR_PHASE2B18_REPORT=/home/duyanhong/experiments/RDDR_PHASE2B18/report_r1 \
  /home/duyanhong/miniconda3/envs/sshr5090/bin/python \
  -m unittest discover -s tests -p 'test_rddr_phase2b18*.py' -v
```

37 tests PASS, zero skips. Without these environment variables the24 artifact tests skip; this is not equivalent
to a validated full run. Independent verifier has29 PASS checks, exact FP32 loss/g_s/g_q replay and FP64 analytic proofs.

### Report / visualization

```bash
python tools/render_rddr_phase2b18_report.py \
  --results audit/results/rddr_phase2b18 \
  --output audit/cache/rddr_phase2b18_report_replay.md
```

The renderer refuses overwrites and requires successful verification/test artifacts.
The report uses source-backed tables; no additional plots or mask exports are required.
`artifact_manifest.json` stores byte-exact artifact hashes and report SHA256.

## Results and interpretation

| Evidence | Result | Contract |
| --- | --- | --- |
| Teacher−raw native28 mIoU | +15.6822pp | semantic superiority PASS |
| PRG all Benefit/Harm | 78.9280% / 21.0718% | global utility PASS |
| PRG Raw-Wrong Benefit | 60.6942% | requires≥70%, FAIL |
| Shallow-Win HHCR | 96.2258% | requires≤30%, FAIL |
| Shallow-Win teacher accuracy | 37.2862% | requires≥60%, FAIL |
| Shared head energy fraction | 0.99212% | absorption flag FALSE |

At the frozen point and neglecting tiny epsilon terms, the symmetric mixture's raw-KL gradient is proportional
to `wD*(ps-pd)`: contextual weights scale a deep-directed correction rather than guarantee protection of useful
shallow dissent. This is a local mathematical interpretation, not a new deep-teacher experiment.
Temporary BN affine gradients do not change future A0 training freeze rules. Selected b4-path batch20 memory
is not a full-unfrozen training-memory guarantee. Local dM/dQ risks do not measure actual long-term training collapse.

STOP for review. Branch `feature/rddr-phase2b18-prerect-guidance`, PR against `baseline/official-a0`, no auto-merge.
