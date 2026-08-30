# Phase-2B1.9 Directional Transfer Audit

Full [41-section Chinese report](rddr_phase2b19_directional_transfer_report.md) and [approved contract](rddr_phase2b19_contract.md).

Result: `ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE`.
A/B/C/D/E/F/G = PASS/PASS/PASS/PASS/FAIL/PASS/PASS.
No model training, optimizer, checkpoint writes, test, LUAD or searches occurred.

## Environment / data

Existing server duyanhong@10.15.20.77, RTX5090 D v2.
Python `/home/duyanhong/miniconda3/envs/sshr5090/bin/python`, PyTorch2.11.0+cu128, NumPy1.23.5.
Use the existing A0 dependencies; no upgrades or installations were performed.
Original network is BF16; loss/logit/q derivatives FP32, diagnostic sums FP64.

Validation images: `/home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img`.
Dataset order is sorted and must match all3418 names in frozen NPZ archives. The dataset factory only opens validation image data;
GT masks come from the original immutable native28 archive, only for diagnostic strata, never loss/gating.
The three NPZ paths, C0 checkpoint path and SHA256 are enforced in the runner and listed in the report/runtime JSON.
GT0-3 counts enter diagnostic denominators; bg4/ignore255 excluded. All784 positions enter each batch1 loss denominator.

## Repository structure

- `tools/rddr_phase2b16_common.py`, `tools/rddr_phase2b18_common.py`: unchanged inherited audit math/helpers, not imported innovation models.
- `tools/rddr_phase2b19_common.py`: detached support, exact gate, GT-blind rate-matched RG, UDT/RG/ADT/SDT, frozen decision.
- `tools/run_rddr_phase2b19_audit.py`: original-model replay, full3418 gradients, fixed160 inference identity and batch20 BF16.
- `tools/analyze_rddr_phase2b19.py`: strata, historical comparison, 10000 paired image bootstrap and preregistered gates.
- `tools/verify_rddr_phase2b19.py`: independent implementations of derivatives/AUROC/statistics/bootstrap/decision.
- `tools/render_rddr_phase2b19_report.py`: stdlib-only deterministic41-section report and manifest.
- `tests/test_rddr_phase2b19*.py`: 20 unit and32 integration tests.
- `audit/results/rddr_phase2b19/`: required CSV/JSON, all10000 bootstrap replicates, per-image losses, test evidence and SHA manifest.
- Original `network/`, `tool/`, `train_sshr.py`: unchanged from A0 `4e9a288`.

## Command index

Training: **none, prohibited**. The following commands are optional zero-update replays. Use NEW output directories;
the runner/analyzer/verifier refuse to overwrite their output targets. Do not rerun completed commands with `formal_r1` paths.

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b19
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
$PY tools/run_rddr_phase2b19_audit.py \
  --native /home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz \
  --derived /home/duyanhong/experiments/RDDR_PHASE2B15/formal_r1/rddr_phase2b15_derived_observations.npz \
  --previous /home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1/rddr_phase2b18_observations.npz \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --val-images /home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img \
  --output /home/duyanhong/experiments/RDDR_PHASE2B19/replay_001
$PY tools/analyze_rddr_phase2b19.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B19/replay_001 \
  --output /home/duyanhong/experiments/RDDR_PHASE2B19/replay_report_001
$PY tools/verify_rddr_phase2b19.py \
  --run /home/duyanhong/experiments/RDDR_PHASE2B19/replay_001 \
  --report /home/duyanhong/experiments/RDDR_PHASE2B19/replay_report_001
```

Inference: runner exercises the unchanged official inference on fixed32+random128 validation images before/after.
No additional test/evaluation command is authorized. The prediction hash is taken before official background overwrite.

### Verify completed artifacts

```bash
RDDR_PHASE2B19_RUN=/home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1 \
RDDR_PHASE2B19_REPORT=/home/duyanhong/experiments/RDDR_PHASE2B19/report_r1 \
  /home/duyanhong/miniconda3/envs/sshr5090/bin/python \
  -m unittest discover -s tests -p 'test_rddr_phase2b19*.py' -v
```

52 tests PASS, zero skips; independent verifier33 checks PASS. Missing run/report environment variables skip32 integration tests,
which is NOT a validated full audit. Required33 test names are covered by the combined test suite.

### Report / visualization

```bash
python tools/render_rddr_phase2b19_report.py \
  --results audit/results/rddr_phase2b19 \
  --output audit/cache/rddr_phase2b19_report_replay.md
```

The report uses source-backed tables; no new mask exports or plots are needed. Optional `--manifest` only for a NEW result directory.
The committed artifact_manifest.json stores byte-exact hashes for small evidence files and the delivered Markdown.
Large observation/derived-stat NPZ archives remain server-only and are not checked into Git.

## Result record

| Metric | ADT | Contract |
| --- | --- | --- |
| Deep-Win BRR | 64.0311% | >=60%, PASS |
| Shallow-Win HHCR | 20.1598% | <=30%, PASS |
| Raw-Correct Harm | 3.8456% | <=30%, PASS |
| Raw-Wrong Benefit | 35.5865% [34.6906%,36.4944%] | >=40%, FAIL |
| Foreground active fraction | 28.1161% | >=10%, PASS |
| ADT-RG all Mean dM | +0.0003893938 | favorable95%CI, PASS |

No new Dice/IoU training result exists: no weights were updated. Phase18 raw/teacher native metrics are historical, not this audit's final segmentation performance.
Important limitations: Raw-Wrong active coverage38.5685% already bounds its all-denominator Benefit below40%; class3 has only418 Shallow-Win pixels (UNDERPOWERED).
Although Shallow-Win harm incidence drops, its mean negative dM magnitude increases versus PRG. SDT promising flag does not override ADT failure.
Local logit derivatives do not predict actual shared-parameter updates or long-term segmentation gains.

## Completed artifacts / stop

- Worktree `/home/duyanhong/DZWdeRepo-rddr-phase2b19`.
- GPU data `/home/duyanhong/experiments/RDDR_PHASE2B19/formal_r1` and log `formal_r1.log`.
- Analysis `/home/duyanhong/experiments/RDDR_PHASE2B19/report_r1` and logs `report_r1.log`, `verify_r1.log` in its parent.
- CPU/GPU audit total68.146s, actual3418 backprop32.976s; batch20 reserved2.5430GiB. Not a full-training estimate.
- Independent branch `feature/rddr-phase2b19-directional-transfer`, PR against `baseline/official-a0`; no auto-merge.

STOP for review. No Full25, no lambda or gate modifications.
