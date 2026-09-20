# PDSR‑VPCA‑HQMR v1 Phase0

This branch runs the preregistered frozen-HQMR/frozen-PLIP mechanism-isolation experiment on BCSS Seed42. It trains only the new semantic projections and channel-wise `gamma`, evaluates segmentation once after fixed Epoch 5, writes the 32-section report, and never launches Full25 automatically.

## Environment and data

- Python environment: `/home/duyanhong/miniconda3/envs/sshr5090`
- GPU: RTX 4090D with BF16
- train: `/home/duyanhong/reseg-data/raw/BCSS-WSSS/training` (23,422 image-level-labelled patches)
- val: `/home/duyanhong/reseg-data/raw/BCSS-WSSS/val` (3,418 images/masks; E5 only)
- frozen HQMR: `CCRA_HQMR_Full25_BCSS_Seed42/checkpoints/hqmr_epoch25_final.pth`
- frozen local PLIP: `/home/duyanhong/models/vinid-plip`

## Repository structure

- `network/plip_adapter.py`: local frozen PLIP, official preprocessing, dense-token safety.
- `network/pdsr.py`: three-level concept reconstruction and JS layer consensus.
- `network/vpca.py`: static and visual soft concept participation.
- `network/pdsr_vpca_hqmr.py`: P1/P2/P3 adapters and identity-initialized H5 residual.
- `tools/pdsr_vpca_phase0/prepare.py`: P0/PLIP/concept/identity/leakage gates.
- `train.py`, `evaluate.py`, `visualize.py`, `generate_report.py`: fixed execution chain.

## Commands

Use the paths above and output `/home/duyanhong/experiments/PDSR_VPCA_Phase0_BCSS_Seed42`.

1. Run `prepare.py` with `--checkpoint`, `--plip`, `--train-root`, `--concepts configs/concepts/bcss_vpca_concepts_v1.yaml`, `--output`, and the passed Oracle reproduction gate as `--source-gate`.
2. Run `train.py --variant P1`, then P2, then P3. Each fixed run uses micro-batch 5 × accumulation 4 = effective batch 20 and exactly 5,855 optimizer steps.
3. Run `evaluate.py` once after all three E5 runtime manifests exist, passing BCSS val and the frozen UMRF output.
4. Run `visualize.py`, then `generate_report.py`.

Training never receives a validation or mask path. Evaluation refuses to run until all three E5 manifests report 5,855 steps.

## Result table

| Run | Frozen backbone/VLM | Trainable path | Epoch | mIoU | mDice | Decision |
|---|---|---|---:|---:|---:|---|
| P0 | HQMR | none | 0 | 65.5727% | 78.9611% | paired frozen prediction bank |
| P1 | HQMR + PLIP | P_last + gamma | 5 | 65.5726% | 78.9609% | no material gain |
| P2 | HQMR + PLIP | P4/P8/P12/Ps + gamma | 5 | 65.5731% | 78.9613% | PDSR NOGO |
| P3 | HQMR + PLIP | same as P2 | 5 | 65.5725% | 78.9608% | VPCA NOGO / Full Model NOGO |

All three runs finished 5,855 optimizer steps without training-time validation access. The final report is [PDSR_VPCA_HQMR_v1_Phase0_Final_Report.md](PDSR_VPCA_HQMR_v1_Phase0_Final_Report.md). Full machine-readable metrics, E5 prediction/mechanism banks, 220 case visualizations, checkpoints, and logs remain under `/home/duyanhong/experiments/PDSR_VPCA_Phase0_BCSS_Seed42` on the server. The historical HQMR scalar (65.572444%) and the recomputed paired-bank P0 (65.572738%) differ by 0.000294 pp; the report preserves this caveat.

## Resume and failure handling

Formal variant directories are write-once. A failed run is diagnosed from its launcher log; do not overwrite an E5 checkpoint or use intermediate validation to select an epoch. Smoke tests use a separate `_Smoke` output. Full25 is intentionally deferred even if the final gate passes.
