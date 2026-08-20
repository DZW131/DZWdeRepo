# RSBR-v0 Corrected Parity R1 Audit

## Decision

**RSBR_V0_PARITY_R1_PASS**

## Layer 1: same-process hard identity

- Fixed validation images: 32
- Maximum CAM28_1 difference: 0.000e+00
- Delta-core exact zero: True
- Delta-transition exact zero: True
- Base/refined differing pixels: 0
- Production flags unchanged: True

## Layer 2: frozen production numerical envelope

- A0 mIoU / mDice: 67.325154 / 80.267174
- RSBR-zero mIoU / mDice: 67.336008 / 80.274349
- Cross-process mIoU difference: 0.01085458 pp (allowance 0.01379944 pp)
- Cross-process differing pixels: 61,173 (allowance 87,808)
- Production flags unchanged: True


## Provenance and scope

- A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Audit commit: `7cbe5aa0ad73d7e6827962f832bd50d6050d0b73`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- RSBR model hashes unchanged: True
- Evaluation-only audit: true
- Test accessed: false
- LUAD accessed: false

Exact command:

```bash
tools/audit_rsbr_v0_parity_r1.py --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --output-dir /home/duyanhong/experiments/RSBR_V0_PARITY_R1_AND_READINESS_7cbe5aa/parity_r1 --audit-commit 7cbe5aa0ad73d7e6827962f832bd50d6050d0b73 --num-workers 4
```
