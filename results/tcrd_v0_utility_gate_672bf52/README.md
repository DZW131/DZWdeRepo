# TCRD-v0 Utility Gate — Archived Result

## Decision

`ROUTE_E_CLOSE`

The frozen matched five-epoch utility gate completed all four branches and all
24 preregistered BCSS validation points. No test split, LUAD run, other seed,
fresh 25-epoch training, checkpoint selection, or hyperparameter tuning was
performed.

| Branch | Epoch-5 mIoU | Delta vs C0 | CAM28_1 delta | Decision |
|---|---:|---:|---:|---|
| C0 | 67.2312 | — | — | Reference |
| D | 67.2485 | +0.0174 pp | +0.0344 pp | SPED_UTILITY_NOGO |
| R | 67.2983 | +0.0672 pp | +0.1022 pp | TCER_UTILITY_REVIEW |
| DR | 67.2880 | +0.0568 pp | +0.1035 pp | TCRD_FULL_REVIEW |

SPED generated a measurable update but did not learn the preregistered
same/cross-tissue selectivity (`1.0062 < 1.05`) and increased boundary errors.
TCER reduced present-class confusion by only `0.1427%`, below the `0.5%` PASS
threshold. The combined branch did not provide synergy over R. Therefore no
candidate qualifies for automatic fresh 25-epoch training.

## Frozen provenance

- Implementation commit: `672bf5222dbe992f401c9c1c6106dc3b8d90f290`
- A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Shared schedule SHA256: `3998960668cc91161150b7561e12194b15f89639e3283de6fa0d720432288e41`
- Train/validation: `23,422 / 3,418`
- Effective batch: `20`; precision: BF16; seed: `42`

## Artifact map

- `docs/tcrd_v0_utility_gate_report.md`: complete human-readable report
- `comparison/route_decision.json`: preregistered gate decisions and deltas
- `comparison/epoch_metrics.csv`: all 24 fixed validation measurements
- `comparison/mechanism_metrics.csv`: reaction/diffusion diagnostics
- `comparison/error_taxonomy.json`: paired boundary and present-confusion audit
- `figures/`: five required plots
- `preflight/`: real batch20 BF16 readiness evidence
- `C0_control/`, `D_diffusion/`, `R_reaction/`, `DR_full/`: histories,
  provenance, training summaries, and completion records
- `utility_schedule.npz` and `.json`: exact shared deterministic schedule
- `run.log`: complete sequential execution log

The four final checkpoints and prediction arrays remain on the server under
`/home/duyanhong/experiments/TCRD_V0_UTILITY_GATE_672bf52/`; they are excluded
from Git because the full experiment directory is approximately 1.7 GB.

STOP: this archive does not authorize any follow-up training.
