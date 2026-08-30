# Phase-2B1.11 — runnable audit guide

Standalone, zero-training **Neither-Hierarchy / Third-Evidence Feasibility Audit**. Pure official A0 `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`; no innovation model code imported. Original `network/`, `tool/`, and `train_sshr.py` are unchanged.

## Result

| Item | Result |
| --- | --- |
| Inputs | 3418 frozen BCSS validation images, native28, C0 seed42 final25 |
| Foreground GT-blind candidates | 202,678 |
| Rescue / failure | 108,541 / 94,137 |
| Candidate precision | 53.5534%, 95% CI [52.2799%, 54.8400%] |
| M_alt rescue image AUC | 0.62490654 |
| Neither-hierarchy image AUC | 0.75911247 |
| M_alt gradient-utility image AUC | 0.62702312 |
| Gates A/B/C/D/E/F | FAIL / FAIL / PASS / PASS / PASS / FAIL |
| Decision | THIRD_EVIDENCE_OPERATIONAL_HEADROOM_INSUFFICIENT |

**Gate A fails its precision condition, not its rescue-count condition.** The label must not be read as saying third evidence does not exist. Both-Wrong detection passes, but safe alternative selection does not. No follow-on gate design or training is unlocked. Read the [33-section full report](rddr_phase2b111_neither_hierarchy_third_evidence_report.md).

## Environment and setup

The completed server run uses `/home/duyanhong/miniconda3/envs/sshr5090/bin/python`, PyTorch `2.11.0+cu128`, NumPy `1.23.5`, SciPy, NVIDIA RTX 5090 D v2. Reuse the existing environment; do not upgrade packages for this audit.

Main runner uses CUDA solely for FP32 **probability-only replay**; no model is instantiated. The independent verifier uses CPU NumPy/SciPy. Report renderer requires only Python stdlib.

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b111
/home/duyanhong/miniconda3/envs/sshr5090/bin/python -c 'import torch,numpy,scipy; print(torch.__version__,numpy.__version__,scipy.__version__,torch.cuda.is_available())'
git diff 4e9a2887b220d17e27649d72a3d13f32b7ebe8f9 -- network tool train_sshr.py
```

The diff must be empty. Hash mismatch is a blocker, not permission to replace cached observations or alter pinned constants.

## Repository and data organization

| File | Purpose |
| --- | --- |
| `docs/rddr_phase2b111_contract.md` | Approved pre-outcome equations, denominators, gates, stop conditions |
| `tools/rddr_cache_metrics.py` | Reused verified scalar/tied-ranking helpers; no model code |
| `tools/rddr_probability_replay.py` | FP32 support/context/q/raw probability replay; forbidden graph operations guarded |
| `tools/rddr_phase2b111_common.py` | GT-blind candidate, controls, analytic epsilon-KL derivative, GT-only metrics |
| `tools/run_rddr_phase2b111_audit.py` | Full population, ranking, stratification, hard/soft effects, paired bootstrap |
| `tools/verify_rddr_phase2b111.py` | Independent context/Jacobian/finite-difference/ranking/bootstrap verification |
| `tools/render_rddr_phase2b111_report.py` | Deterministic report and artifact SHA manifest |
| `tests/test_rddr_phase2b111*.py` | 21 unit + 33 real-artifact integration tests |
| `audit/results/rddr_phase2b111/` | All CSV/JSON, 10k replicates, test log and manifest |

Exact eight input paths and SHA256 values are in runtime JSON and report §1. Caches remain in their original Phase2B1/2B15/2B19 directories, previous evidence in Phase2B110, and checkpoint in the existing official baseline run. No dataset split files are opened. Dataset masks already exist in immutable caches and are used only after candidate construction.

Completed run: `/home/duyanhong/experiments/RDDR_PHASE2B111/formal_r1`. Large original input NPZ/checkpoint and small per-image-statistics NPZ remain on server, not in Git. New logs: `/home/duyanhong/experiments/RDDR_PHASE2B111/formal_r1.log` and `verify_r1.log`.

## Training / inference

**Not applicable and prohibited in this phase.** No model loading, forward, backward, autograd, optimizer, checkpoint write, test/LUAD/other seed, learned gate, threshold or score search. Historical baseline training commands elsewhere in this repository do not authorize another run.

## Reproduce audit / evaluation

Use a new output directory. Do not rerun the verifier against completed `formal_r1`: the verifier refuses to overwrite its existing JSON. The following example creates an independent `replay_r1`; it must not already exist.

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b111
RDDR111_REPLAY=/home/duyanhong/experiments/RDDR_PHASE2B111/replay_r1
test ! -e "$RDDR111_REPLAY" || exit 1
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b111_audit.py \
  --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz \
  --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz \
  --observations /home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1/rddr_phase2b19_observations.npz \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --previous-summary /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_summary.json \
  --previous-runtime /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_runtime.json \
  --previous-identity /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_identity_audit.json \
  --previous-verification /home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1/rddr_phase2b110_verification.json \
  --output "$RDDR111_REPLAY" && \
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/verify_rddr_phase2b111.py --run "$RDDR111_REPLAY" && \
RDDR_PHASE2B111_RUN="$RDDR111_REPLAY" /home/duyanhong/miniconda3/envs/sshr5090/bin/python \
  -m unittest discover -s tests -p 'test_rddr_phase2b111*.py' -v \
  > "$RDDR111_REPLAY/rddr_phase2b111_tests.txt" 2>&1
```

Without `RDDR_PHASE2B111_RUN`, artifact tests skip: that is not sufficient validation. Delivered results contain 54 passes, zero skips, and 29 independent verification checks. Main audit took 15.9748s; peak process RSS 2.4279 GiB and probability-replay peak GPU allocation 15.0903 MiB. These are not training costs.

## Render report / visualization

Report tables are the visualization; no extra image or model inference job is needed. To reproduce delivered report bytes using only committed evidence:

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b111
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/render_rddr_phase2b111_report.py \
  --run audit/results/rddr_phase2b111 \
  --report audit/cache/rddr_phase2b111_report_replay.md \
  --manifest audit/cache/rddr_phase2b111_manifest_replay.json
cmp docs/rddr_phase2b111_neither_hierarchy_third_evidence_report.md audit/cache/rddr_phase2b111_report_replay.md
```

Use fresh paths; existing report/manifest paths are rejected. `cmp` must print nothing. The manifest records exact hashes/sizes for all scientific CSV/JSON/TXT and report; Git attributes preserve bytes across platforms.

## Interpretation and stop

GT-blind construction does not imply GT-free evaluation: foreground inclusion and rescue labels are retrospective. Both-Wrong detection, context hard correctness, and local gradient utility are separate questions. Global protection passes because activation is low; active Raw-Correct hard harm is 100% and gradient harm 96.7941%, with important boundary/class variation. Do not claim a safe trained mechanism or full segmentation improvement.

This phase is complete. No remaining training/evaluation is pending. PR review only; no auto-merge, no Phase2B1.12 design or training until a new approved request. Preserve baseline and all prior outputs.
