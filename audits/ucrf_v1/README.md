# UCRF-v1 frozen HQMR audit

This directory implements the no-training Upstream Class-Responsibility
Formation audit for the frozen BCSS Seed42 HQMR-v1 checkpoint.

## Environment

- Existing `sshr5090` conda environment
- PyTorch/CUDA versions are recorded by `gate.py`
- BCSS validation images and masks in `VAL/img` and `VAL/mask`
- Frozen HQMR checkpoint
- Frozen M1 component table from the previous target-representation audit

No command in this directory calls `backward()` or `optimizer.step()`.

## Reproduction

```bash
python audits/ucrf_v1/gate.py \
  --checkpoint "$CHECKPOINT" --val-root "$VAL" --output "$OUT"

python audits/ucrf_v1/extract.py \
  --checkpoint "$CHECKPOINT" --val-root "$VAL" \
  --prior-target "$PRIOR_TARGET_PARQUET" --output "$OUT"

python audits/ucrf_v1/analyze.py --output "$OUT"

python audits/ucrf_v1/visualize.py \
  --checkpoint "$CHECKPOINT" --val-root "$VAL" --output "$OUT"

python audits/ucrf_v1/generate_report.py --output "$OUT"
```

The first command must reproduce mIoU `0.6557244403737567`, 4,440 M1
components, and 8,750,254 M1 pixels before downstream extraction is allowed.

## Mathematical boundary

HQMR tensors `logits5`, `direct4`, `logits4`, and `logits3` are query-mask
logits, not four-class tissue logits. Actual class evidence is the frozen
query-to-class weighted mixture of sigmoid query logits. Therefore:

- query-logit identities are checked exactly;
- stage class competition uses the actual frozen nonlinear mixture;
- the direct4 class-margin effect is `margin(C4)-margin(C_U5)`;
- a standalone direct4 mixture is reported only as an auxiliary proxy;
- query4 is isolated by replacing q4 with q5 in the frozen Stage3 affinity
  expression, while holding k3 and class weights fixed.

See `forward_formula_manifest.md` for the complete formula.

## Outputs

- `00_reproduction_gate.json` and `hook_manifest.json`
- component-stage parquet/CSV tables and 2,000-resample bootstrap intervals
- matched TP control tables
- 120 replay-validated case visualizations in six categories
- raw five-stage tensors for the selected images
- `UCRF_v1_Upstream_Class_Responsibility_Formation_Audit_Report.md`

The report generator validates the frozen cohort size, visualization count,
and replay consistency before writing the final 25-section report.
