# PSCR-v1 Audit

PSCR-v1 is a zero-training audit of whether frozen weak-supervision evidence can identify and correct HQMR prototype-source class labels. It does not modify CCRA, HQMR, the backbone, decoder, or checkpoints.

## Environment and data

- Python environment used on the server: `/home/duyanhong/miniconda3/envs/sshr5090`
- BCSS-WSSS train images: `/home/duyanhong/reseg-data/raw/BCSS-WSSS/training`
- BCSS validation: `/home/duyanhong/reseg-data/raw/BCSS-WSSS/val`
- Reconstructed TRAIN oracle masks are evaluation-only and must never be read by `extract`.
- HQMR and SSHR checkpoint hashes are enforced in `run_pscr_v1.py`.

## Files

- `run_pscr_v1.py`: extract, source-evaluate, target-evaluate, report, and visualization pipeline.
- `../../configs/audits/pscr_v1.yaml`: preregistered evidence and policy rules.
- `../../tests/test_pscr_v1.py`: policy, gate, ablation, and decision-boundary tests.

## Execution order

All commands use the same arguments and output directory. Run modes strictly in this order:

```bash
python audits/pscr_v1/run_pscr_v1.py --mode extract [common arguments]
python audits/pscr_v1/run_pscr_v1.py --mode source-evaluate [common arguments]
python audits/pscr_v1/run_pscr_v1.py --mode target [common arguments]
python audits/pscr_v1/run_pscr_v1.py --mode visualize [common arguments]
```

Common required arguments are `--trainroot`, `--train-gt-root`, `--train-gt-audit`, `--val-root`, `--hqmr-checkpoint`, `--sshr-checkpoint`, `--phase0-output`, `--anatomy-output`, and `--output-dir`.

`extract` is the only permitted decision-path feature extraction stage. It must finish with `gt_accessed=false` and exact Bank-A source/embedding reproduction. `source-evaluate` freezes all policy banks before opening GT. `target` is the only validation evaluation stage. `visualize` is post-hoc and cannot change metrics or decisions.

## Evaluation and outputs

Primary metric: area-weighted max-cosine accuracy on the exact 4,440 M1 false components. The decision is based on recovery of the frozen Bank-A to oracle Bank-B gap.

Main artifacts:

- `00_reproduction_gate.json`
- `metrics/bank_b_gap_recovery.json`
- `pscr_v1_result.json`
- `PSCR_v1_Prototype_Source_Class_Reliability_Audit_Report.md`
- `visualizations/visualization_manifest.csv`

## Final recorded result

| Run | Dataset | Training | Bank A | Best bank | Gap recovery | Decision |
|---|---|---:|---:|---:|---:|---|
| PSCR-v1 | BCSS Seed42 | 0 updates | 7.58% | P3 7.66% | 0.42% | NOGO |

The static prototype-source verification family is therefore frozen. Do not tune thresholds, train a verifier, or launch Full25 from this result.
