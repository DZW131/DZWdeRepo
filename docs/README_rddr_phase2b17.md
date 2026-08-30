# RDDR Phase-2B1.7 reproducibility guide

## Purpose and result

Frozen contextual correction acceptance audit on all 3,418 BCSS validation images.
This is **not training**. Pure A0 source, architecture, inference, metric and weights remain unchanged.
The scientific decision is `CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED` (A/B/C/D all FAIL; engineering PASS).
Do not start Full25 from this result.

Read the complete [Chinese experiment report](rddr_phase2b17_contextual_correction_acceptance_report.md)
and the pre-outcome [approved contract](rddr_phase2b17_contract.md).

## Environment

Actual server: `duyanhong@10.15.20.77`, RTX 5090 D v2.
Existing Python: `/home/duyanhong/miniconda3/envs/sshr5090/bin/python`.
PyTorch `2.11.0+cu128`, NumPy `1.23.5`; torchvision, Pillow, SciPy and original A0 dependencies
are already installed. No environment upgrade or installation was performed for the audit.
Use the existing environment; its backend settings must match the runner's assertions.
The runner uses BF16 network forward and FP32 probabilities/support/loss.
The supplementary batch20 memory result only covers the approved seven student parameter tensors.

## Repository layout

- `tools/rddr_phase2b16_common.py`: frozen prior audit math, not an innovation model.
- `tools/rddr_phase2b17_common.py`: GT-blind support/HA/SA, frozen strata, ranks and decisions.
- `tools/run_rddr_phase2b17_acceptance_audit.py`: frozen parity, 3,418 real forwards/backwards, identity and batch20.
- `tools/analyze_rddr_phase2b17.py`: primary metrics, image bootstrap, fixed gates.
- `tools/verify_rddr_phase2b17.py`: independently implemented support, gradients, ranks, bootstrap and decision.
- `tools/render_rddr_phase2b17_report.py`: standard-library-only report renderer, no GPU/dataset access.
- `tests/test_rddr_phase2b17.py`: 8 mathematical/guardrail tests.
- `tests/test_rddr_phase2b17_artifacts.py`: 21 real-artifact integration checks.
- `audit/results/rddr_phase2b17/`: CSV/JSON, 10,000 bootstrap replicates, 47,852 parameter-gradient records,
  test transcript and SHA256 manifest. No checkpoint or large feature cache is committed.

## Data and immutable assets

Source hashes and absolute paths are recorded in `rddr_phase2b17_runtime.json` and the report.
Only the following are used:

- C0 BCSS seed42 Full25 final checkpoint, SHA256 `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Phase2B1 native observations; Phase2B1.5 symmetric teacher observations; Phase2B1.6 logits/gradients.
- BCSS validation images under `/home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img`.
- Validation masks are available to the original inference identity probe; diagnostic GT/strata come from frozen caches.

All 3,418 sorted filenames must match every cache exactly. Background=4 and ignore=255 are excluded only
from foreground diagnostics, not the 784-position loss denominator. No test, training split, LUAD or other seed is opened.

## Completed runs and commands

Completed GPU run: `/home/duyanhong/experiments/RDDR_PHASE2B17/formal_r1`.
Canonical analysis/verification: `/home/duyanhong/experiments/RDDR_PHASE2B17/report_r3`.
Exact original commands are also recorded in runtime/summary/verification JSON.
The following **replay examples are not an instruction to launch another experiment**.
Use new output directories: all existing outputs must remain intact.

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b17
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python

# Zero-update forward/backward audit; never an optimizer step.
$PY tools/run_rddr_phase2b17_acceptance_audit.py \
  --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz \
  --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz \
  --previous /home/duyanhong/experiments/RDDR_PHASE2B16/formal_r1/rddr_phase2b16_gradient_observations.npz \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --val-images /home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img \
  --output /home/duyanhong/experiments/RDDR_PHASE2B17/replay_001

$PY tools/analyze_rddr_phase2b17.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B17/replay_001 \
  --output /home/duyanhong/experiments/RDDR_PHASE2B17/replay_report_001

$PY tools/verify_rddr_phase2b17.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B17/replay_001 \
  --report /home/duyanhong/experiments/RDDR_PHASE2B17/replay_report_001
```

Tests against the already completed run (29 PASS, zero skips):

```bash
RDDR_PHASE2B17_RUN=/home/duyanhong/experiments/RDDR_PHASE2B17/formal_r1 \
RDDR_PHASE2B17_REPORT=/home/duyanhong/experiments/RDDR_PHASE2B17/report_r3 \
  /home/duyanhong/miniconda3/envs/sshr5090/bin/python \
  -m unittest discover -s tests -p 'test_rddr_phase2b17*.py' -v
```

Without these two environment variables, artifact tests skip; do not present a skipped local suite as real integration evidence.

## Report / visualization

The complete report uses tables rather than additional plots. No learned visualization model or predicted-mask export is needed.
From the repository root, regenerate a new Markdown from the committed small artifacts using ordinary Python3:

```bash
python tools/render_rddr_phase2b17_report.py \
  --results audit/results/rddr_phase2b17 \
  --output audit/cache/rddr_phase2b17_report_replay.md
```

The renderer requires the actual PASS verification and 29-test transcript and refuses to overwrite outputs.
`artifact_manifest.json` records original artifact bytes/SHA256 and the published report SHA256.

## Training, inference, evaluation policy

Training command: **none, prohibited**. No optimizer is constructed; checkpoint writes are blocked.
Inference: existing official path, used on fixed160 validation images for before/after prediction identity only.
Evaluation: native CAM28_1 correction/gradient diagnostics, not a newly reported full segmentation score.
No test-set evaluation, no full validation mIoU rerun, no final model selection, no lambda/threshold/class-rule search.

| Run | Split | Winner image AUC | Gradient image AUC | Accepted teacher−rect accuracy | A/B/C/D | Engineering |
| --- | --- | --- | --- | --- | --- | --- |
| Phase2B1.7 formal_r1 + report_r3 | BCSS val 3418 | 0.619465 | 0.523684 | −1.2850 pp | FAIL/FAIL/FAIL/FAIL | PASS |

## Known numerical and interpretation limitations

Prior q reconstruction differs by at most 5.96e-8; the frozen cached q is retained.
Independent support summation differs at 3 near-zero Δ signs; primary Δ remains unchanged.
The first FP64-vs-FP32 absolute-only gradient verifier failed. Final verification instead proves exact independent
FP32 replay plus FP64-autograd/analytic agreement; see the full incident disclosure in report section25.
`report_r1/r2` were preserved. They must not be relabeled as successful original verifications.

Negative GT-margin derivative is not itself a prediction flip or an observed future training mIoU loss.
The secondary soft flag is TRUE but does not unlock training. STOP for review; do not automatically merge the PR.
