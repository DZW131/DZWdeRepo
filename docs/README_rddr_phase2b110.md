# RDDR Phase-2B1.10 — runnable audit guide

This branch contains a **cache-only residual correction coverage audit**, based directly on pure official A0 `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`. No innovation architecture is imported. Original `network/`, `tool/`, and `train_sshr.py` are unchanged. This README is an audit command index, not permission to start training.

## Result and scope

| Item | Result |
| --- | --- |
| Data | All 3418 frozen BCSS validation images, native28 observations |
| Checkpoint provenance | C0 seed42 final Epoch25, SHA256 pinned in contract/runtime |
| Required additional beneficial count | 31,266 |
| Residual beneficial count | 177,865 |
| S_D residual utility image AUC | 0.50017874 [0.49134083, 0.50904762] |
| S_D rejected winner image AUC | 0.60827404 [0.59932098, 0.61739637] |
| Context rejected Both-Wrong accuracy/rescue | 0.33710374 [0.32423468, 0.34996457] |
| Gates A/B/C/D | PASS / FAIL / FAIL / FAIL |
| Decision | RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED |
| Validation | 44 tests, 26 independent checks passed |

These are conditional native28 mechanism metrics, **not full segmentation mIoU or training results**. GT defines audit populations and labels, never scores. The outcome supports only a separately preregistered third-evidence audit, not a safe selector or training GO. See the [complete 29-section report](rddr_phase2b110_residual_correction_coverage_report.md).

## Environment

The completed server run used `/home/duyanhong/miniconda3/envs/sshr5090/bin/python`, PyTorch `2.11.0+cu128`, NumPy `1.23.5`, SciPy, NVIDIA RTX 5090 D v2. Reuse that environment; no package upgrades are required. The main audit needs CUDA only for exact frozen FP32 probability-neighborhood replay (not a network forward). The verifier uses CPU NumPy/SciPy. The report renderer uses the Python standard library only.

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b110
/home/duyanhong/miniconda3/envs/sshr5090/bin/python -c 'import torch,numpy,scipy; print(torch.__version__, numpy.__version__, scipy.__version__, torch.cuda.is_available())'
git diff 4e9a2887b220d17e27649d72a3d13f32b7ebe8f9 -- network tool train_sshr.py
```

The final command must print nothing. SHA mismatches are blockers; do not regenerate inputs or edit pinned hashes merely to make checks pass.

## Files and data organization

| File / location | Purpose |
| --- | --- |
| `docs/rddr_phase2b110_contract.md` | Approved pre-outcome definitions and decision precedence |
| `tools/rddr_phase2b110_common.py` | Frozen scores, GT-only labels, tied AUC/AP, context metrics, decisions |
| `tools/run_rddr_phase2b110_audit.py` | Full cached population, probability replay, decomposition and 10k bootstrap |
| `tools/verify_rddr_phase2b110.py` | Independent ranking/context/bootstrap/decision verifier |
| `tools/render_rddr_phase2b110_report.py` | Deterministic report and SHA manifest from evidence, no model work |
| `tests/test_rddr_phase2b110*.py` | 17 unit + 27 real-artifact integration tests |
| `audit/results/rddr_phase2b110/` | CSV/JSON, all bootstrap replicates, tests and manifest; no training weights |
| `/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1` | Immutable completed server artifacts, including image-statistics NPZ |
| `/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1.log` | Main audit log |
| `/home/duyanhong/experiments/RDDR_PHASE2B110/verify_r1.log` | Independent verifier log |

The seven exact input paths and SHAs are in the runtime JSON and report §1. The three previous observation NPZ files remain in their original Phase2B1/2B15/2B19 experiment directories. The original checkpoint is hash-read only. No dataset split directory is needed by this audit; the cache already contains all validation probabilities, masks and frozen gradients. Large input NPZ files and checkpoints are not committed.

## Training / inference commands

**Not applicable and prohibited for this phase.** There is no new model, loss, recovery gate, optimizer, checkpoint write or network inference. Do not invoke `train_sshr.py`, test, LUAD, another seed or Full25 as part of this audit. Existing A0 training instructions elsewhere in the repository are unrelated to this phase.

## Reproduce the audit and evaluation

The completed run used `formal_r1`. For a deliberate rerun, select a fresh output such as the following `replay_r1`; the runner refuses to overwrite any existing output directory. Do not run the verifier against `formal_r1` because it emits its verification JSON into the selected run. Use the newly created replay directory instead.

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b110
RDDR110_REPLAY=/home/duyanhong/experiments/RDDR_PHASE2B110/replay_r1
test ! -e "$RDDR110_REPLAY" || exit 1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b110_audit.py \
  --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz \
  --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz \
  --observations /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_observations.npz \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --previous-runtime /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_runtime.json \
  --previous-summary /home/duyanhong/experiments/RDDR_PHASE2B19/report_r1/rddr_phase2b19_summary.json \
  --previous-identity /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_identity_audit.json \
  --output "$RDDR110_REPLAY" && \
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/verify_rddr_phase2b110.py --run "$RDDR110_REPLAY" && \
RDDR_PHASE2B110_RUN="$RDDR110_REPLAY" /home/duyanhong/miniconda3/envs/sshr5090/bin/python \
  -m unittest discover -s tests -p 'test_rddr_phase2b110*.py' -v \
  > "$RDDR110_REPLAY/rddr_phase2b110_tests.txt" 2>&1
```

Without `RDDR_PHASE2B110_RUN`, artifact tests skip; that is not a completed readiness check. The delivered log has 44 passed, zero skipped. SHA/file equality is measured this phase; prior model/BN/fixed160 prediction identity is **inherited**, not freshly rerun.

## Render report / visualize results

No images, model visualizations or additional GPU experiments are needed. The complete Markdown provides evidence tables. To independently reproduce its bytes from the committed CSV/JSON, use a fresh ignored output path:

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b110
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/render_rddr_phase2b110_report.py \
  --run audit/results/rddr_phase2b110 \
  --report audit/cache/rddr_phase2b110_report_replay.md \
  --manifest audit/cache/rddr_phase2b110_manifest_replay.json
cmp docs/rddr_phase2b110_residual_correction_coverage_report.md audit/cache/rddr_phase2b110_report_replay.md
```

`cmp` must be silent. Renderer refuses report/manifest overwrites. CSVs and source evidence are never rewritten by the renderer. The manifest records exact bytes of all delivered scientific CSV/JSON/TXT plus the report; Git attributes preserve those bytes across platforms.

## Stop condition

Everything required for this audit has run. There is no pending training or validation evaluation. Do not automatically construct an S_D/q/Delta recovery threshold, use GT-based groups as deployment masks, add a context loss, or alter Phase2B1.9's conclusion. Any next audit needs a separate approved contract. PR review is the only remaining user action; no automatic merge.
