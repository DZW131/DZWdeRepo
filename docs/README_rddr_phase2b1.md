# RDDR Phase-2B1 — reproducible audit guide

## Purpose and frozen result

This is a zero-training, validation-only audit of whether shallow neighborhood
evidence can adjudicate shallow versus deep semantic hypotheses. It is an
independent branch from pure A0
`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`; no official model, training,
inference, or metric source is modified.

**Result: `RDDR_PHASE2B1_NOGO`.** Gates A/B/C/D: PASS / FAIL / FAIL / PASS.
This result does not authorize Phase-2B2 or any test evaluation.

| Frozen endpoint | Result |
|---|---:|
| Image-balanced adjudication AUROC | 0.734850 |
| AUROC 95% paired image-bootstrap CI | [0.726086, 0.743701] |
| Pooled sign balanced accuracy | 0.593973 |
| Deep-Win / Shallow-Win recall | 0.261653 / 0.926293 |
| Anchor minus FixedAvg accuracy | +0.520502 percentage points |
| Anchor minus FixedAvg mIoU | -0.443308 percentage points |
| Deep-Wrong accuracy difference | +7.494762 percentage points |
| Top20 Deep-Wrong accuracy difference | +17.929479 percentage points |

These are native 28-grid, four-class foreground diagnostic metrics, **not**
official final-CAM test mIoU. See the complete report for denominators,
uncertainty, class-specific failures, and nonlinear mIoU aggregation.

## Environment

The executed environment is the existing server environment, not a new install:

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b1
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
"$PY" -c "import torch,numpy,scipy,PIL; print(torch.__version__, numpy.__version__, scipy.__version__, PIL.__version__)"
"$PY" -m unittest discover -s tests -p test_rddr_phase2b1.py -v
```

Executed with Python 3.10.20, PyTorch 2.11.0+cu128, NumPy 1.23.5 and an
RTX 5090 D v2. SciPy and Pillow are required for independent rank checks and
image handling. Do not upgrade the frozen environment to rerun this audit.
Forward is batch1/BF16, with FP32 probabilities and support calculations;
the recorded Phase-0 backend is restored by the extractor.

## Code and artifact map

| Path relative to repository | Role |
|---|---|
| `docs/rddr_phase2b1_contract.md` | Approved preregistration and decision precedence |
| `tools/rddr_phase2b1_common.py` | Fixed JSD, support, and metric helpers |
| `tools/run_rddr_phase2b1_dual_hypothesis_audit.py` | Unmodified A0 forward plus read-only hook; observations |
| `tools/run_rddr_phase2b1_server.sh` | Pinned server inputs and test/extraction entrypoint |
| `tools/analyze_rddr_phase2b1.py` | Exact ranks, frozen strata, 10,000 paired bootstrap replicates |
| `tools/verify_rddr_phase2b1.py` | Independent NumPy/SciPy verification, without shared audit helpers |
| `tools/report_rddr_phase2b1.py` | Complete Markdown report renderer |
| `tests/test_rddr_phase2b1.py` | 21 safety, formula, population, and decision tests |
| `audit/results/rddr_phase2b1/` | CSV/JSON evidence, all bootstrap replicates, test log, report |
| `docs/rddr_phase2b1_dual_hypothesis_context_adjudication_report.md` | Reviewed complete report |
| `docs/rddr_phase2b1_delivery.md` | Validation evidence, hashes, and handoff limitations |

## Frozen data organization

```text
/home/duyanhong/reseg-data/raw/BCSS-WSSS/val/
  img/       3418 PNG images
  mask/      3418 matched PNG masks
/home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/diagnostics/
  frozen_phase0_populations/    immutable replay cache, 3418 NPZ + manifest
/home/duyanhong/experiments/RDDR_PHASE0_586f402/formal/
  ... original population-count evidence
```

Checkpoint:
`/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`.
SHA256:
`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.

Historical `by_CH` labels describe raw-to-full-HFRM transitions, **not a
CH-only intervention**. The reused cache matches original per-image counts;
historical original pixel-level hashes were not available. GT only creates
audit targets and strata, never support scores, weights, or anchor probabilities.

## Training

**Disabled and out of scope.** There is no training command for Phase-2B1.
Do not create an optimizer, call backward, save a model checkpoint, or proceed
to another experiment automatically.

## Forward extraction, evaluation, and report commands

The actual completed commands and code versions are recorded in
`audit/results/rddr_phase2b1/rddr_phase2b1_runtime.json`. Existing output folders
are never overwritten. The following optional reproduction commands use new
names; they are documentation, not an instruction to rerun now:

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b1
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
AUDIT=/home/duyanhong/experiments/RDDR_PHASE2B1

# Optional 2-real-image safety smoke; unique output name required.
bash tools/run_rddr_phase2b1_server.sh smoke_reproduction_01 2

# All 3418 validation images; no test or train split is read.
bash tools/run_rddr_phase2b1_server.sh formal_reproduction_01

# Offline analysis: does not repeat the model forward.
"$PY" tools/analyze_rddr_phase2b1.py \
  --input "$AUDIT/formal_reproduction_01" \
  --output "$AUDIT/report_reproduction_01"

# Independent verification must pass before rendering the report.
"$PY" tools/verify_rddr_phase2b1.py \
  --report "$AUDIT/report_reproduction_01" \
  --native "$AUDIT/formal_reproduction_01/rddr_phase2b1_native_observations.npz"
"$PY" tools/report_rddr_phase2b1.py \
  --report-dir "$AUDIT/report_reproduction_01"
```

No threshold/temperature/window/fusion sweep is supported. `Delta > 0` is the
frozen deep decision, with zero ties assigned to shallow. The ratio-based soft
anchor is distinct from this hard decision.

## Visualization and interpretation

The deliverable is a full Markdown report with quantitative tables; no chart or
qualitative-image selection was required. Open the report in a Markdown viewer.
Machine-readable per-image, per-class, fixed-stratum, support, calibration,
confusion, and paired-bootstrap evidence is in `audit/results/rddr_phase2b1/`.
Do not average subgroup mIoUs to reconstruct global mIoU; add their confusion
matrices first. Independent verification checks this explicitly.

## Server outputs and retention

- `formal_r1/`: immutable full-validation native observations and extraction metadata.
- `report_r1/`: original complete statistical outputs and first report draft.
- `delivery/`: reviewed report and enhanced independent-verification JSON.
- `tests_final.txt`: 21/21 passing tests.

All four are under `/home/duyanhong/experiments/RDDR_PHASE2B1`.
Local canonical report is the reviewed `delivery/` version, byte-for-byte.
First local drafts were retained in ignored `audit/cache/`; no earlier
experiment or baseline was deleted. Large NPZ observations/statistics are not
committed to Git and remain at the server paths above. The code, CSV/JSON,
replicates and report are versioned on the independent feature branch.

The approved workflow ends with this report and an unmerged PR. No further run
is pending or required to establish this NOGO decision.
