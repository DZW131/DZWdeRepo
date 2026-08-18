# OSMF-v1.3 Phase -1.1 Parity Recheck

## Decision

**OSMF_V13_PARITY_PASS**

No optimizer or training step was executed.

## Frozen provenance

- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- OSMF-v1.3 commit: `5a19675e76e60a020892be934936aa19f31b03fa`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Exact command: `tools/audit_osmf_v13_parity.py --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --output-dir /home/duyanhong/experiments/OSMF_V13_LOCAL_STRUCTURAL_5a19675/parity --osmf-v13-commit 5a19675e76e60a020892be934936aa19f31b03fa --num-workers 4`
- Precision: official BF16 with A0 TF32; OSMF identity projections use local IEEE FP32.

## Tensor parity

### random_input

- baseline_vs_osmf_hfrm_max_abs: `0`
- osmf_input_vs_reconstruction_max_abs: `0`
- baseline_vs_osmf_reconstruction_max_abs: `0`
- cam56_max_abs: `0`
- cam28_1_max_abs: `0`
- cam28_2_max_abs: `0`
- camdeep_max_abs: `0`
- classification_probability_max_abs: `0`

### real_validation_input

- baseline_vs_osmf_hfrm_max_abs: `0`
- osmf_input_vs_reconstruction_max_abs: `0`
- baseline_vs_osmf_reconstruction_max_abs: `0`
- cam56_max_abs: `0`
- cam28_1_max_abs: `0`
- cam28_2_max_abs: `0`
- camdeep_max_abs: `0`
- classification_probability_max_abs: `0`

## Full BCSS validation parity

- Images: 3418
- Differing prediction pixels: 0
- A0/v1.3 mIoU: 0.6732793717 / 0.6732793717
- Absolute mIoU difference: 0
- A0/v1.3 mDice: 0.8026798070 / 0.8026798070
- Absolute mDice difference: 0

## Parameter compatibility

- New trainable tensors: 4
- Parameter delta: 524,288
- Overhead: 0.465167%
- Missing keys are exactly p_sem/p_morph/u_sem/u_morph weights.
- No semantic auxiliary classifier exists.

## Boundary

Only a PASS authorizes the separate 8-real-batch readiness audit. No 128-batch audit or Phase 1 was started by this tool.

OSMF_V13_PARITY_PASS
