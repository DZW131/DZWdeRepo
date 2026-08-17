# OSMF-v1.0 Phase -1 Delivery Summary

## What Was Implemented

- Added `OSMFFactorizer` at the 512-channel post-HFRM H28_1 representation.
- Added fixed 256/256 semantic and morphology projections and reconstructions.
- Added complementary channel-partition initialization with exact identity
  reconstruction under the official CUDA BF16/TF32 environment.
- Reused the original CAM28_1 head; CAM56, CAM28_2, and CAMdeep remain unchanged.
- Implemented the frozen semantic, equivariance, orthogonality, and detached
  reconstruction objectives for later gated phases.
- Added an OSMF-only training entry point without a test-evaluation path.
- Added a validation-only released-inference parity audit.

## Validation Evidence

- Local: 25 tests passed and one CUDA-only test skipped.
- Server: all 26 tests passed on the RTX 5090 D v2.
- Fixed random input reconstruction maximum absolute error: 0.
- Real BCSS input reconstruction maximum absolute error: 0.
- All five released forward outputs have maximum absolute error 0.
- Full 3418-image BCSS validation predictions have 0 differing pixels.
- A0 and OSMF-init validation mIoU: 67.32791670%.
- A0 and OSMF-init validation mDice: 80.26795301%.
- Parameter overhead: 525,316 parameters, or 0.4661%.
- Frozen A0 checkpoint tensors load bit-exactly; only six new OSMF keys are
  missing and there are no unexpected keys.

## Remaining Items

- Phase 0: exactly 128 real BCSS training batches and the preregistered gradient
  ratio/finite/collapse audit.
- Phase 1: the 3-epoch mechanism pilot, only after Phase 0 human approval.
- Phase 2: the 25-epoch formal experiment, only after mechanism GO and explicit
  authorization.
- Test, LUAD, additional seeds, ablations, and validation tuning are
  intentionally deferred.

## Final Command Index

Phase -1 parity command:

```bash
python -u tools/audit_osmf_phase_minus1.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --output-dir /home/duyanhong/experiments/OSMF_V1_PHASE_MINUS1_5eb7b25 \
  --osmf-commit 5eb7b258f0cdeb4fa8779b65e716c105c9541f9a \
  --num-workers 4 \
  --amp-dtype bf16
```

The future training entry point is `train_osmf.py`; it is documented but must
not be launched before the next gated audit is approved.

## Artifact Locations

- Report: `audit/results/osmf_phase_minus1/osmf_phase_minus1_readiness_report.md`
- Machine summary: `audit/results/osmf_phase_minus1/summary.json`
- Parity table: `audit/results/osmf_phase_minus1/parity.csv`
- Run log: `audit/results/osmf_phase_minus1/run.log`
- Technical specification: `docs/specs/osmf_v1_technical_spec.md`

## Handoff Notes

- The frozen A0 code files were not modified; OSMF uses a separate network
  module and training entry point.
- OSMF projection/reconstruction convolutions run in FP32 with TF32 disabled
  locally because the released post-HFRM tensor remains FP32. This is required
  for exact identity parity; the rest of SSHR retains the official BF16/TF32
  behavior.
- Resume only with the Phase 0 structural sanity audit. Do not jump directly to
  the 3-epoch or 25-epoch runs.
