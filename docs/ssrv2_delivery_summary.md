# SSR-v2 Full-Model Delivery Summary

## Delivered change

- Implemented SSR-v2 PCSD and PTCR on the clean official SSHR A0 baseline.
- Added exactly one learnable scalar (`beta_spatial`) and preserved the frozen
  training, HFRM, inference and metric protocols outside SSR-v2.
- Added isolated preflight, 25-epoch training and final-checkpoint evaluation
  entry points, plus focused unit and integration tests.
- Archived the BCSS seed42 epoch25 FINAL validation evidence and figures.

## Validation performed

- Local test suite: 9 focused SSR-v2 tests passed.
- Server preflight: passed, including A0 identity, detach boundaries, optimizer
  coverage, BF16 execution, pretrained compatibility and zero-residual cases.
- Formal training: 25/25 epochs completed on BCSS seed42 with no validation
  checkpoint selection.
- Formal evaluation: epoch25 FINAL evaluated on BCSS validation only for both A0
  and SSR-v2.

## Result and decision

| Model | mIoU | mDice |
|---|---:|---:|
| SSHR A0 epoch25 FINAL | 67.3283 | 80.2683 |
| SSR-v2 epoch25 FINAL | 66.8575 | 79.9354 |

SSR-v2 changed mIoU by -0.4708 pp and mDice by -0.3328 pp. The largest
class-level regression was C3 at -1.3625 pp. This triggers
`SSRV2_FULLMODEL_NO_CLEAR_GAIN`, `SSRV2_CLASS_REGRESSION_REVIEW`, and the
preregistered hard stop.

## Reproduction and artifacts

The exact commands and server paths are recorded in
[`ssrv2_fullmodel_README.md`](ssrv2_fullmodel_README.md). The complete result is
reported in [`ssrv2_full_25ep_report.md`](ssrv2_full_25ep_report.md), with
machine-readable evidence in
[`results/ssrv2_full_25ep/`](../results/ssrv2_full_25ep/).

The primary checkpoint remains on the server at:

`/home/duyanhong/experiments/SSRV2_FULL_25EP_SEED42_04e4631/checkpoints/epoch25_final.pth`

SHA256:
`34265e42164f85dc5a59dcadaf56685bd1c34a89ab71300005dbc6d51c4ea6c3`

## Deferred by the frozen protocol

No BCSS test evaluation, LUAD experiment, seeds 11/17, ablation, schedule or
coefficient sweep, or SSR-v3 modification was performed. Any such work requires
a separate approved experiment.
