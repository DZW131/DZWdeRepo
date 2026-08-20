# RGR-v0 experiment artifact manifest

## Outcome

- Final decision: `RGR_V0_PILOT_NOGO`
- Formal experiment commit: `8a7081f5fb3ac1ffb4037e6e60d0716e6363e2c0`
- Base: official A0 commit `4e9a288`, BCSS seed 42 final checkpoint
- Parity: `RGR_V0_PARITY_PASS`
- Readiness: `RGR_V0_READINESS_PASS`
- Pilot: exactly 3 epochs / 3,513 optimizer updates
- Test, LUAD, other seeds, 25 epochs: not run

The pilot NOGO is a performance/mechanism result, not a safety failure. The
isolated branch improved validation mIoU by `+0.0537`, `+0.0676`, and
`+0.0699` pp, but Full minus Isolated was negative in all epochs (`-0.0443`,
`-0.0660`, `-0.0731` pp). Epoch-3 Full minus Base was `-0.0032` pp.

## Formal server artifacts

- Full output: `/home/duyanhong/experiments/RGR_V0_8a7081f`
- Disposable preflight: `/home/duyanhong/experiments/RGR_V0_PREFLIGHT_6f7b41d`
- Full output size: 9.0 MB
- Compact archive SHA256:
  `b4e03877cc910aa2e9034079d9f47d887274133ea4203e6fa4f19d1af8342fc1`

The repository archives compact JSON summaries and the report. Per-step JSONL
logs and checkpoints remain on the server.

## Checkpoints

| Epoch | Server file | SHA256 |
|---:|---|---|
| 1 | `checkpoints/epoch1_rgr.pth` | `bf55ba192b4320747f3b783a67e28c16c58460f635417108956689cff180c843` |
| 2 | `checkpoints/epoch2_rgr.pth` | `8d79fe7a85a541b587e43e86d015a5b62332be853a9f3f393fbe2f44f53e7bff` |
| 3 | `checkpoints/epoch3_rgr.pth` | `a74eaeb06adf19e887f3d18311614331a4511ac535006564882fd26211457b58` |

## Audit history

Two preserved engineering-only failed starts preceded the formal run:

- `6f7b41d`: pre-validation evaluator setup failed before any validation image
  or training step because official `eval()` is non-chainable;
- `df47edc`: RGR same-forward parity completed, then independent A0 setup failed
  before the A0 comparison and before training for the same chaining reason.

Both issues were fixed and covered by regression tests. The only formal result
is the clean `8a7081f` run above.

## Files

- `contract.json`: frozen command and source hashes
- `parity/summary.json`: same-process and production parity
- `readiness_32b/summary.json`: 32-batch gradient/state/resource audit
- `preflight/preflight_6f7b41d.json`: disposable real-CUDA smoke
- `pilot_3ep/summary.json`: consolidated pilot metrics and safety state
- `pilot_3ep/validation_epoch*.json`: full paired validation diagnostics
- `pilot_3ep/train/epoch*_summary.json`: training aggregate statistics
- `docs/rgr_v0_delivery.md`: complete experiment report
