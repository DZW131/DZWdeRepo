# OSMF-v1.0 Phase -1 artifact manifest

This directory archives the successful frozen-model BCSS validation-only
initialization-parity run.

- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Executed OSMF commit: `5eb7b258f0cdeb4fa8779b65e716c105c9541f9a`
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Source archive SHA256: `bab1248e6ed0071d4b7b5f09df7d6432b5ca4206ce8a31011d66f479a9c29a1d`

Final server locations:

- Source: `/home/duyanhong/DZWdeRepo-osmf-v1-5eb7b25`
- Results: `/home/duyanhong/experiments/OSMF_V1_PHASE_MINUS1_5eb7b25`
- Run log: `/home/duyanhong/experiments/OSMF_V1_PHASE_MINUS1_5eb7b25_run.log`

Archived file hashes:

- `osmf_phase_minus1_readiness_report.md`:
  `2782d2ffacbd65fc7fc2db840ba5c31b3ce6fb996f99e96aa122bc9f06b1d542`
- `parity.csv`:
  `6e9bfba5c64a91b16711182537764349950aa1a00b50ed98673130391816d504`
- `summary.json` and `run.log`:
  `d229a3b841d69f7878e1d84ba455c7756beaae4d0fd847680dc476356573c921`

The machine-readable metrics in `summary.json` use fractions; multiply by 100
for percentages. Thus `0.6732791669541228` is `67.32791670%` mIoU.

The run performed no optimization step, backward pass, SSHR training, test-set
evaluation, or LUAD evaluation. Phase 0 has not started.
