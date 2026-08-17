# OSMF-v1.0 Phase -1 Implementation Parity Report

## 1. Decision

Final decision: **OSMF_PHASE_MINUS1_PASS**.

This phase implemented OSMF-v1.0 at post-HFRM H28_1 and performed initialization parity only. No optimization step or SSHR training was run.

## 2. Frozen contract

- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- OSMF audit commit: `5eb7b258f0cdeb4fa8779b65e716c105c9541f9a`.
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Dataset/split: BCSS validation only.
- CAM56, CAM28_2, CAMdeep, official fusion, thresholds, TTA, and metric: unchanged.
- Test evaluated: false. LUAD evaluated: false. Training performed: false.
- Exact command: `tools/audit_osmf_phase_minus1.py --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --output-dir /home/duyanhong/experiments/OSMF_V1_PHASE_MINUS1_5eb7b25 --osmf-commit 5eb7b258f0cdeb4fa8779b65e716c105c9541f9a --num-workers 4 --amp-dtype bf16`.

## 3. Implementation

- Factorization point: 512-channel post-HFRM H28_1.
- Semantic/morphology channels: 256/256.
- New inference path: P_sem/P_morph then U_sem/U_morph before the original ic1 head.
- Complementary channel-selection/placement initialization reconstructs the identity.
- The semantic auxiliary head and all specialization losses are training-only.

## 4. Feature and CAM parity

### Random Input

- baseline_vs_osmf_hfrm_max_abs: `0`
- osmf_input_vs_reconstruction_max_abs: `0`
- baseline_vs_osmf_reconstruction_max_abs: `0`
- cam56_max_abs: `0`
- cam28_1_max_abs: `0`
- cam28_2_max_abs: `0`
- camdeep_max_abs: `0`
- classification_probability_max_abs: `0`

### Real Validation Input

- baseline_vs_osmf_hfrm_max_abs: `0`
- osmf_input_vs_reconstruction_max_abs: `0`
- baseline_vs_osmf_reconstruction_max_abs: `0`
- cam56_max_abs: `0`
- cam28_1_max_abs: `0`
- cam28_2_max_abs: `0`
- camdeep_max_abs: `0`
- classification_probability_max_abs: `0`

## 5. Full validation released-inference parity

- Images: 3418.
- Differing pixels: 0.
- A0 mIoU: 0.6732791670.
- OSMF-init mIoU: 0.6732791670.
- Absolute mIoU difference: 0.
- A0 mDice: 0.8026795301.
- OSMF-init mDice: 0.8026795301.
- Absolute mDice difference: 0.

## 6. Pretrained compatibility

The baseline checkpoint loaded strictly into A0. Every frozen A0 tensor loaded bit-exactly into OSMF. Missing keys were restricted to the newly introduced factorizer and semantic auxiliary classifier:

- `osmf_28_1.p_morph.weight`
- `osmf_28_1.p_sem.weight`
- `osmf_28_1.semantic_classifier.bias`
- `osmf_28_1.semantic_classifier.weight`
- `osmf_28_1.u_morph.weight`
- `osmf_28_1.u_sem.weight`

Unexpected keys: `[]`.

## 7. Cost at initialization

- A0 parameters: 112,709,714.
- OSMF parameters: 113,235,030.
- Parameter delta: 525,316 (0.4661%).

## 8. Phase boundary

Phase 0 (128 real BCSS training batches) has **not** been started. The loss weights, 128-batch gradient audit, 3-epoch pilot, and formal training remain gated by human review.

OSMF_PHASE_MINUS1_PASS
