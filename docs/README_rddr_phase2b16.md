# RDDR Phase-2B1.6 reproducible audit

This branch is based directly on pure official A0, not on any archived innovation branch.
It adds training-time-only **audit helpers**, not a training implementation. No optimizer, checkpoint write, architecture change, or formal training is authorized here.

## Environment

Validated server: `/home/duyanhong/DZWdeRepo-rddr-phase2b16` on RTX 5090 D v2.
Existing Python: `/home/duyanhong/miniconda3/envs/sshr5090/bin/python` (PyTorch 2.11.0+cu128, NumPy 1.23.5).
Use the existing official environment and weights; no package upgrade is required.
Ordinary training/inference instructions in the repository's original README remain unchanged and are **not** this audit's next action.

## Layout

- `docs/rddr_phase2b16_contract.md`: approved, pre-outcome contract and ambiguity resolutions.
- `tools/rddr_phase2b16_common.py`: GT-blind detached teacher, three loss probes, diagnostic helpers.
- `tools/run_rddr_phase2b16_trainability_audit.py`: exact cache replay, real 3418-image backward audit, fixed160 identity, batch20 BF16 smoke.
- `tools/analyze_rddr_phase2b16.py`: native28 pooled metrics, gradient utility and 10k paired image bootstrap.
- `tools/verify_rddr_phase2b16.py`: independent NumPy equations/gradient/metrics/bootstrap verification.
- `tools/report_rddr_phase2b16.py`: deterministic Markdown renderer (standard library only).
- `tests/test_rddr_phase2b16*.py`: synthetic math/safety tests and separate assertions over real integration evidence.
- `audit/results/rddr_phase2b16/`: CSV/JSON, test evidence and manifests; no checkpoints or raw images.

## Frozen data

Only `/home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img` and the corresponding `val/mask` are read.
The Dataset class is the original official `Stage1_InferDataset`; sorted filenames must match all 3418 frozen cache IDs.
Source probabilities/GT/q/Top20/boundary are in the frozen Phase2B1 cache, symmetric supports in Phase2B1.5.
All SHA256 values and exact full commands are in the final report/runtime JSON.

## Run commands

The following commands create **new** directories. Existing directories are rejected. Choose new names for any deliberate replay; do not overwrite formal_r1/report_r1.

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b16
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python

$PY -m unittest discover -s tests -p test_rddr_phase2b16.py -v

$PY tools/run_rddr_phase2b16_trainability_audit.py \
  --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz \
  --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --val-images /home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img \
  --output /home/duyanhong/experiments/RDDR_PHASE2B16/formal_replay

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY tools/analyze_rddr_phase2b16.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B16/formal_replay \
  --output /home/duyanhong/experiments/RDDR_PHASE2B16/report_replay

OPENBLAS_NUM_THREADS=1 $PY tools/verify_rddr_phase2b16.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B16/formal_replay \
  --report /home/duyanhong/experiments/RDDR_PHASE2B16/report_replay

RDDR_PHASE2B16_RUN=/home/duyanhong/experiments/RDDR_PHASE2B16/formal_replay \
RDDR_PHASE2B16_REPORT=/home/duyanhong/experiments/RDDR_PHASE2B16/report_replay \
$PY -m unittest discover -s tests -p 'test_rddr_phase2b16*.py' -v
```

The original inference function is invoked internally on the frozen160 set both before and after backward. It uses original TTA, thresholds, CAM fusion and decoding. No standalone inference modification is needed.

## Report and tabular visualization

Download the run's JSON/CSV and analysis JSON/CSV into a new local artifact directory. The existing committed results directory contains this completed run.

```bash
python tools/report_rddr_phase2b16.py \
  --results audit/results/rddr_phase2b16 \
  --output docs/rddr_phase2b16_report_replay.md
```

This renders all principal evidence as Markdown tables; no plotting/UI or new analysis threshold is required. Raw gradients remain in a 148MiB compressed server NPZ and are identified by SHA256, not committed to Git.

## Result / next-action boundary

| Audit | Dataset | Teacher vs FixedAvg native mIoU | A/B/C/D | Decision |
| --- | --- | --- | --- | --- |
| Phase2B16 | BCSS validation, 3418 images | +1.9651 pp | PASS/PASS/FAIL/PASS | TEACHER_SIGNAL_PRESENT_GRADIENT_UNSAFE |

These are native28 diagnostic metrics, not official fused segmentation benchmark scores.
No Full25, test, LUAD, other seed, lambda selection or parameter update was run. The report is the terminal deliverable; any next hypothesis requires separate user review.
