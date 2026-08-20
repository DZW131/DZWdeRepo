# RSBR-v0 Parity R1 and 32-Batch Readiness Delivery

## 1. Executive conclusion

- Corrected parity: **RSBR_V0_PARITY_R1_PASS**
- Stage-0 readiness: **RSBR_V0_READINESS_PASS**
- 3-epoch pilot started: **false**
- Test/LUAD accessed: **false / false**

## 2. Corrected two-layer parity

Layer 1 used the fixed 32-image BCSS validation subset in one RSBR process:

- Maximum CAM difference: 0.000e+00
- Delta-core exact zero: True
- Delta-transition exact zero: True
- Base/refined differing pixels: 0

Layer 2 used independent production BF16 A0 and RSBR-zero validation runs:

- mIoU difference: 0.01085458 pp / allowance 0.01379944 pp
- Differing pixels: 61,173 / allowance 87,808
- Production flags unchanged: True

## 3. Frozen Stage-0 control

- Dataset / seed: BCSS train / 42
- Parsed samples: 23,422
- Real batches / batch size / image size: 32 / 20 / 224
- Precision: BF16; production cuDNN benchmark and TF32 enabled
- Fresh A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Loss: `L_SSHR_refined28_1 + 0.05 L_region + 0.01 L_res`
- SSHR parameters frozen and absent from the update groups: True
- RSBR model source hashes unchanged: True

## 4. Connectivity and dynamics

| Step | Total loss | Region MIL | Region grad from L_region | Transition grad from refined cls | Region total grad | Transition total grad | T/R grad | Residual ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.172483 | 0.293671 | 9.253e-01 | 2.293e-03 | 7.089e-02 | 2.293e-03 | 0.0323 | 0.000000 |
| 1 | 0.175334 | 0.307513 | 9.127e-01 | 2.281e-03 | 3.012e-02 | 2.949e-03 | 0.0979 | 0.006629 |
| 2 | 0.268310 | 0.509318 | 8.522e-01 | 4.455e-03 | 4.339e-02 | 2.580e-03 | 0.0595 | 0.007280 |
| 4 | 0.153993 | 0.344424 | 6.994e-01 | 3.054e-03 | 4.114e-02 | 2.242e-03 | 0.0545 | 0.007542 |
| 8 | 0.183767 | 0.379550 | 7.885e-01 | 2.184e-03 | 5.643e-02 | 4.113e-03 | 0.0729 | 0.013138 |
| 16 | 0.198189 | 0.203831 | 8.062e-01 | 3.302e-03 | 4.451e-02 | 1.938e-03 | 0.0436 | 0.012775 |
| 24 | 0.135653 | 0.168010 | 6.621e-01 | 1.871e-03 | 3.930e-02 | 2.801e-03 | 0.0713 | 0.011138 |
| 32 | 0.248941 | 0.456557 | 1.172e+00 | 6.485e-03 | 1.052e-01 | 5.132e-03 | 0.0488 | 0.013763 |

At step 32:

- Region movement absolute / relative: 2.328310e-02 / 2.328310e+10
- Transition movement absolute / relative: 5.688145e-03 / 8.707890e-04
- Region valid images: 20/20
- Region logit mean/std/min/max: 0.252361 / 5.070564 / -13.125000 / 20.250000
- Mean region token norm: 17.030181

## 5. Region and mask statistics

- Mean / median regions per image: 3.6219 / 3.0000
- Mean valid-token regions per image: 3.3297
- Mean core / transition fraction: 0.677282 / 0.322718
- Last-8 transition fraction: 0.328731
- Mean tiny / no-region fraction: 0.080169 / 0.000000
- Deterministic extraction repeat: {'structures_identical': True, 'component_counts_identical': True}

## 6. Residual and safety

- End residual ratio: 0.013763
- End core / transition RMS: 0.084505 / 0.000132
- End maximum absolute core / transition delta: 0.287109 / 0.001022
- All finite: True
- Peak CUDA memory: 1.724 GiB
- Runtime: 84.07 s

## 7. Decision evidence

- Hard failures: none
- Review triggers: none
- Rapid residual growth: False

Final decision: **RSBR_V0_READINESS_PASS**.

The protocol requires a stop after this report even when readiness passes.
No 3-epoch pilot or formal experiment was launched.

## 8. Exact commands

Parity:

```bash
tools/audit_rsbr_v0_parity_r1.py --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --output-dir /home/duyanhong/experiments/RSBR_V0_PARITY_R1_AND_READINESS_7cbe5aa/parity_r1 --audit-commit 7cbe5aa0ad73d7e6827962f832bd50d6050d0b73 --num-workers 4
```

Readiness:

```bash
tools/run_rsbr_v0_readiness_32b.py --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --output-dir /home/duyanhong/experiments/RSBR_V0_PARITY_R1_AND_READINESS_7cbe5aa --audit-commit 7cbe5aa0ad73d7e6827962f832bd50d6050d0b73 --batch-size 20 --img-size 224 --num-workers 4 --lr 0.01 --wt-dec 0.0005
```
