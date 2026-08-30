# RDDR Phase-2B0 reproducible audit

This independent A0 branch implements the user-approved zero-training,
validation-only reliable-relation audit. Read `rddr_phase2b0_contract.md` first.
No official network/training/inference files are modified.

## Environment and data

Use the existing 5090 server environment (no package upgrade):
`/home/duyanhong/miniconda3/envs/sshr5090/bin/python`.
Dependencies: PyTorch/CUDA, torchvision, NumPy, scipy, Pillow. Exact versions
and backend settings are recorded in `rddr_phase2b0_runtime.json`.

- Validation images and masks: `BCSS-WSSS/val/{img,mask}/*.png`, 3418 each.
- Input C0 Full25 seed42 checkpoint; SHA pinned in the contract and code.
- Immutable Phase-0 raw/rect/Top20/q cache + manifest from Phase-2A replay.
- Original Phase-0 per-image CSV is required to verify historical group counts.
- No access to training/test/LUAD data, no checkpoint generation.

## Files and commands

- `tools/rddr_phase2b0_common.py`: GT-blind relation builder, metric helpers.
- `tools/run_rddr_phase2b0_relation_audit.py`: unchanged A0 forward with read-only
  hook, frozen-cache parity, streaming summaries/histograms, resource audit.
- `tools/summarize_rddr_phase2b0.py`: offline paired bootstrap, CSV/JSON/report.
- `tests/test_rddr_phase2b0.py`: mathematical/eligibility/safety regression tests.
- `tools/run_rddr_phase2b0_server.sh`: exact server paths and run launcher.

```bash
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
$PY -m unittest discover -s tests -p test_rddr_phase2b0.py -v
# Output name MUST be new. Existing outputs are never overwritten.
bash tools/run_rddr_phase2b0_server.sh smoke_unique_name 2
bash tools/run_rddr_phase2b0_server.sh formal_unique_name
$PY tools/summarize_rddr_phase2b0.py \
  --input /home/duyanhong/experiments/RDDR_PHASE2B0/formal_unique_name \
  --output /home/duyanhong/experiments/RDDR_PHASE2B0/report_unique_name
```

Training: intentionally unavailable. Inference and evaluation are the audit
commands above, not official final-CAM evaluation. Visualization: the Markdown
tables are sufficient for the preregistered comparisons; no visual cherry-pick.

## Outputs and verification

All required CSV/JSON files plus complete Markdown are emitted by the summary
command. Raw sufficient-statistic NPZ stays beside extraction outputs; it is
small but ignored by default Git rules (not a model). Per-image CSV contains
the primary confusion matrices and bootstrap inputs for independent checking.
The report records metrics, eligibility, quantization errors, resource usage,
exact commands/SHAs and the four preregistered gates. No test unlock or training
is automatic, including a GO result. Prior runs and failed smoke outputs remain
untouched. User review and PR merge are separate from execution.

## Delivery record

| Phase | Split | Model | New training | Output |
|---|---|---|---|---|
| 2B0 | BCSS validation | Frozen A0/C0 Full25 | None | Reliable-relation feasibility report |
