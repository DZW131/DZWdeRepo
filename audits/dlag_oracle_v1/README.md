# DLAG Oracle v1

Frozen, no-training ceiling audit for two HQMR-v1 failure paths:

1. component-level deep/local arbitration through the real query-logit
   equation `L4(alpha)=U(L5)+alpha*D4`;
2. targeted preservation of a true class excluded by the final presence gate.

GT is used only to select one complete output from a pre-registered alpha bank
or to enable an already-existing class at the final binary gate. No GT label is
written into a prediction. `parameter_updates=0` throughout.

## Environment and data

- Conda environment: `/home/duyanhong/miniconda3/envs/sshr5090`
- Frozen checkpoint: `CCRA_HQMR_Full25_BCSS_Seed42/checkpoints/hqmr_epoch25_final.pth`
- Checkpoint SHA256: `84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb`
- BCSS validation root: `/home/duyanhong/reseg-data/raw/BCSS-WSSS/val`
- Frozen UCRF output: `UCRF_v1_Upstream_Class_Responsibility_Formation_Audit_BCSS_Seed42`
- Seed: 42; TTA: identity/horizontal/vertical

The exact PyTorch, CUDA, GPU, commit, and config hashes are written to
`00_reproduction_gate.json` at runtime.

## Files

- `gate.py`: fresh HQMR/M1/UCRF-anchor reproduction gate.
- `oracle_bank.py`: eight-alpha real-forward bank and A1/A2/B1/B2/AB outputs.
- `analyze.py`: mIoU, per-class, safety, overlap, 2,000-resample bootstrap,
  and pre-registered decision.
- `visualize.py`: four aggregate figures and 5×20 replay-validated cases.
- `generate_report.py`: validated 25-section Chinese report.
- `forward_formula_manifest.md`, `gate_formula_manifest.md`: exact frozen math.
- `configs/audits/dlag_oracle_v1.yaml`: pre-registered grid and thresholds.

## Commands

```bash
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
REPO=/home/duyanhong/DZWdeRepo-gcqm-phase0
CHECKPOINT=/home/duyanhong/experiments/CCRA_HQMR_Full25_BCSS_Seed42/checkpoints/hqmr_epoch25_final.pth
VAL=/home/duyanhong/reseg-data/raw/BCSS-WSSS/val
UCRF=/home/duyanhong/experiments/UCRF_v1_Upstream_Class_Responsibility_Formation_Audit_BCSS_Seed42
OUT=/home/duyanhong/experiments/DLAG_Oracle_v1_BCSS_Seed42

cd "$REPO"
$PY audits/dlag_oracle_v1/gate.py \
  --checkpoint "$CHECKPOINT" --val-root "$VAL" \
  --ucrf-output "$UCRF" --output "$OUT"
$PY audits/dlag_oracle_v1/oracle_bank.py \
  --checkpoint "$CHECKPOINT" --val-root "$VAL" \
  --ucrf-output "$UCRF" --output "$OUT"
$PY audits/dlag_oracle_v1/analyze.py --output "$OUT"
$PY audits/dlag_oracle_v1/visualize.py \
  --checkpoint "$CHECKPOINT" --val-root "$VAL" \
  --ucrf-output "$UCRF" --output "$OUT"
$PY audits/dlag_oracle_v1/generate_report.py --output "$OUT"
```

There is deliberately no training command: training, backward passes, optimizer
steps, new modules, and threshold search are outside this audit.

## Results

| Run | mIoU | Delta vs frozen HQMR | Purpose |
|---|---:|---:|---|
| Frozen HQMR | 65.5724% | — | Reference |
| SSHR | 66.6967% | +1.1243 pp | Target reference |
| Oracle A1 | 66.3801% | +0.8076 pp | Component-support arbitration |
| Oracle B1 | 67.9824% | +2.4100 pp | Targeted true-class force-on |
| Oracle A+B | 69.1632% | +3.5908 pp | Primary joint ceiling |

Decision: `STRONG_ARCHITECTURE_GO` with `HIGH` confidence. Oracle A+B is
`+2.4665 pp` above SSHR. The four-class gains are distributed rather than
concentrated in one class. A1 alone does not reach the 1.50 pp architecture
threshold; B1 is a first-order mechanism, and the joint ceiling has positive
synergy (`+0.3732 pp`).

## Output structure

- `counterfactuals/alpha_*/predictions.npz`: complete validation predictions.
- `arbitration_oracle/`: best alpha, response curve, A1/A2 predictions/metrics.
- `gate_oracle/`: gate manifest, B1/B2 predictions/metrics.
- `joint_oracle/`: AB predictions and component recoverability.
- `metrics/`: confusion matrices, global curve, TP safety, stratification,
  bootstrap intervals, synergy, and decision.
- `visualizations/`: four aggregate figures plus 100 case panels.
- `DLAG_v1_Oracle_Ceiling_Audit_Report.md`: final report.

## Important caveats

- A/B/AB are GT-assisted ceilings, not deployable inference procedures.
- The sealed gate masks absent classes to zero, not negative infinity. Exact
  zero ties can still be won by a low-index absent class; this explains why B2
  is diagnostic rather than the primary gate ceiling.
- Alpha=1 directly reuses the canonical output from the same model forward.
  Final resize is outside BF16 autocast, matching the sealed evaluator.
- The independent bank process differed from the frozen reference by only
  `+0.000294 pp`; point deltas use the frozen reference, while paired bootstrap
  uses the same-process bank baseline.
