# CRRA-v0 formal artifact manifest

This directory is the compact, reviewable archive of the formal BCSS
validation-only CRRA-v0 audit run from code commit
`be298c1704ffdb684f779d09e943fd495adf6f14`.

## Decision

- Final decision: `CRRA_V0_NOGO`
- Representation flag: `REGION_REPRESENTATION_ROUTE_CLOSED`
- Test, LUAD, CRSR, segmentation training, and additional representations were
  not run.

## Provenance

- A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Checkpoint SHA256:
  `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Server output: `/home/duyanhong/experiments/CRRA_V0_be298c1`
- Formal report: `docs/crra_v0_region_representation_audit.md`
- Machine-readable result: `summary.json`

## Integrity

- `summary.json` SHA256:
  `342a3c58ec0de03d50c923649f74250f6bfbb6e64c135b9267e8a2bf39281e99`
- `regions/metadata.csv` SHA256:
  `d410e588d16d33d1f4c3ede4f326ae5a50eb4a31f16ff4157ed71573a75e47fa`
- Full archive SHA256:
  `987500e598fe47c9b9b85667928c77cfec2d150e5bf22296dde320b2882dca8e`
- Compact archive SHA256:
  `007b711fe0c400b68b3a6ef552ff406bdecc8c77c8b845af6bc2f33481ad4e5c`

The 73,767,825-byte `regions/features.npz` is intentionally omitted from Git.
Its SHA256 is
`375cd92cd12ee2cdd3d4d80755f5d0e7fb1f799d3a079fc56b340a072e52da8e`.
It remains in the server output and in the local full archive
`G:/05_科研工作/SSHR/_codex_ops/CRRA_V0_be298c1_full.tar.gz`.
