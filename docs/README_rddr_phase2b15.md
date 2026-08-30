# Phase-2B1.5 reproducible audit guide

## Purpose / final result

This is a validation-only, zero-training, zero-new-parameter and zero-search
diagnostic audit of representation-family bias and contextual third evidence.
It starts from pure A0 `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`, reuses
Phase-2B1's frozen probability cache, and does not load or forward a network.

**Decision: `SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED`.**
Gates A/B/D PASS; C UNDERPOWERED (class2 passes the frozen 0.45 point threshold;
class3 has only 418 Shallow-Win targets). Both strong-signal flags are true but
do not override class evidence or authorize training.

| Endpoint | Old | Symmetric |
|---|---:|---:|
| Image-balanced AUROC | 0.734850 | 0.784842 |
| Pooled zero-sign balanced accuracy | 0.593973 | 0.715627 |
| Deep-Win recall | 0.261653 | 0.640314 |
| Shallow-Win recall | 0.926293 | 0.790939 |
| All-FG mean Delta | -0.124286 | -0.031619 |
| Anchor native-grid mIoU | 0.569087 | 0.593171 |

Symmetric anchor minus FixedAvg mIoU: **+1.965066 percentage points**.
These are native 28-grid diagnostic metrics, not official final-CAM scores.
The original Phase-2B1 NOGO is unchanged.

## Environment

The run used the existing environment, without upgrading dependencies:

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b15
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
"$PY" -c "import torch,numpy,scipy; print(torch.__version__,numpy.__version__,scipy.__version__)"
"$PY" -m unittest discover -s tests -p test_rddr_phase2b15.py -v
```

Python 3.10.20 / PyTorch 2.11.0+cu128 / NumPy 1.23.5 / SciPy, RTX 5090 D v2.
Cached probabilities are FP32 from the frozen original BF16 forward; probe
arithmetic is FP32 with batch1. CPU analysis uses two BLAS/OMP threads.
The report renderer uses only Python's standard library and writes UTF-8/LF.

## Repository structure

| Path | Role |
|---|---|
| `docs/rddr_phase2b15_contract.md` | User-confirmed protocol and completed gate precedence |
| `tools/rddr_phase2b1_common.py` | Unchanged frozen Phase-2B1 helpers; no old innovation model |
| `tools/rddr_phase2b15_common.py` | GT-blind four-way support and separate GT-only diagnostics |
| `tools/run_rddr_phase2b15_bias_decomposition_audit.py` | Input hashes, exact old parity, cache-only probes |
| `tools/analyze_rddr_phase2b15.py` | 45 fixed groups, 12 ordered pairs, native metrics, bootstrap |
| `tools/verify_rddr_phase2b15.py` | Independent NumPy/SciPy verifier, no shared-helper imports |
| `tools/report_rddr_phase2b15.py` | Complete 30-section Markdown report |
| `tests/test_rddr_phase2b15.py` | 23 equation/safety/population/decision tests |
| `audit/results/rddr_phase2b15/` | Full CSV/JSON evidence, 10k replicates, tests |
| `docs/rddr_phase2b15_adjudication_bias_decomposition_report.md` | Canonical final report |
| `docs/rddr_phase2b15_delivery.md` | Evidence, retention, hashes and handoff |

Official `network/`, `tool/` and `train_sshr.py` are unchanged.

## Data and artifact organization

No image dataset is reopened. The frozen input is:

```text
/home/duyanhong/experiments/RDDR_PHASE2B1/
  formal_r1/rddr_phase2b1_native_observations.npz
  report_r1/rddr_phase2b1_summary.json
  report_r1/rddr_phase2b1_per_image.csv
```

The NPZ has all 3418 validation image IDs, ps/pd, old supports/anchor/context,
native GT, q, and frozen grouping masks. Checkpoint access is SHA256 read-only:
`/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`.

New server outputs under `/home/duyanhong/experiments/RDDR_PHASE2B15/`:

```text
smoke_r1/     two-real-image smoke outputs
formal_r1/    derived observations and probe runtime/provenance
report_r1/    complete statistics, all 10k replicates, independent verification
delivery/     canonical reviewed Markdown report
tests_final.txt
formal_r1.log
analysis_r1.log
verify_r1.log
```

## Training

**Not permitted for this phase. No training command is provided.** Do not
create an optimizer, run backward, save a model checkpoint, or launch Phase-2B2.

## Forward / offline evaluation / report commands

The following are optional reproduction instructions, not additional queued
runs. Outputs must be new paths: existing artifacts are not overwritten.
The completed run's exact commands/commits are recorded in `runtime.json`.

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b15
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
AUDIT=/home/duyanhong/experiments/RDDR_PHASE2B15
NATIVE=/home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz
PREVIOUS=/home/duyanhong/experiments/RDDR_PHASE2B1/report_r1
CHECKPOINT=/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth

# This reads cached probabilities only, not a network forward.
# For a separate smoke use --smoke-images 2 and a different output directory.
"$PY" tools/run_rddr_phase2b15_bias_decomposition_audit.py \
  --native "$NATIVE" --previous-report "$PREVIOUS" \
  --checkpoint "$CHECKPOINT" --output "$AUDIT/formal_reproduction_01"

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PY" tools/analyze_rddr_phase2b15.py \
  --native "$NATIVE" --derived "$AUDIT/formal_reproduction_01" \
  --output "$AUDIT/report_reproduction_01"

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PY" tools/verify_rddr_phase2b15.py \
  --native "$NATIVE" \
  --derived "$AUDIT/formal_reproduction_01/rddr_phase2b15_derived_observations.npz" \
  --report "$AUDIT/report_reproduction_01"

"$PY" tools/report_rddr_phase2b15.py \
  --report-dir "$AUDIT/report_reproduction_01" \
  --output "$AUDIT/report_reproduction_01/rddr_phase2b15_adjudication_bias_decomposition_report.md"
```

## Visualization / interpretation

Open the complete Markdown report in a Markdown viewer. It includes support
quantiles, bias CIs, all 12 ordered pairs, class2/class3 root-cause tables,
confidence/mass/GT composition, rescue/harm, four semantic states, safety,
boundary/interior, Q1-Q5, bootstrap, and gates. No selected qualitative images
or extra figure-generation command is needed.

Do not average subgroup mIoUs to infer global mIoU. Class3's low sample count
and class2's sub-0.5 AUROC remain limitations despite aggregate strong flags.
Both-Wrong rescue equals context accuracy by definition; one-correct intrusion
equals harm. These are checked identities, not independent supporting evidence.

## Stop and retention

All computation and independent verification is complete. The report and
independent PR are the final deliverables. No training, test, model design,
source selection, calibration, or follow-up run is authorized automatically.
Large NPZ files stay on the server and are ignored by Git. No baseline or
previous experiment was deleted or overwritten.
