# CDSR-v2 Hierarchy-Shared Alpha Readiness Report

## 1. Executive decision

**CDSR_V2_READINESS_PASS**

Both hierarchy-shared alpha logits diverge measurably from
their matched weight-decay-only shadows after exactly 20 real
BCSS training steps. The simplified architecture passes the
frozen engineering-readiness gate. Per instruction, no 25-epoch
experiment is started in this phase.

## 2. Frozen CDSR-v2 architecture

```text
N_i = R_i * (1 - (1-D_i) * (1-U_i))
G_sem_i = 1 - alpha_sem * (1 - N_i)
G_ctx_i = 1 - alpha_ctx * (1 - N_i)
F_R_i = F_i + gamma_sem_i * G_sem_i * F_sem_i
              + gamma_ctx_i * G_ctx_i * F_CH15_i
```

- `alpha_sem` and `alpha_ctx` are the only two new learnable scalars
- the same two parameter objects are used at F56, F28_1, and F28_2
- N remains independently computed and spatially varying per stage
- alpha initializes to 0.10 and every gate remains <=1
- alpha->0 exactly recovers original SSHR
- original GSR, CH15, CAM probes, detached FP32 Need, loss, optimizer,
  inference, and metrics are unchanged
- there is no new classifier, learned uncertainty head, or spatial
  policy

## 3. Test and compatibility evidence

- local full suite: **67 passed**
- RTX 5090 server full suite: **67 passed**
- every trainable parameter is covered exactly once by the optimizer
- both shared alpha logits are in the original from-scratch group
- exact A0 uniform strict load: **True**
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- CDSR-v2 missing/unexpected keys: 2/0
- additional parameters: 2
- A0 compatibility: **PASS**

The two missing keys are exactly the shared semantic/context alpha
logits; backbone-pretrained loading differs from A0 by only those
same two expected keys.

## 4. Frozen real-step protocol

- parsed BCSS training samples: 23422
- batch / image size: 20 / 224
- steps / seed: 20 / 42
- precision: bf16
- classification-loss weights: [0.1, 0.15, 0.25, 0.5]
- optimizer: official PolyOptimizer/SGD
- base LR / 25-epoch max-step: 0.01 / 29275
- momentum / weight decay: 0.0005 / 0.0005
- PyTorch / CUDA: 2.11.0+cu128 / 12.8
- GPU: NVIDIA GeForce RTX 5090 D v2
- step-1 / step-20 loss: 0.872995 / 0.628621
- all finite: **True**
- shadow LR matched at every step: **True**

The measurable-movement criterion is unchanged from v1: final
task-excess logit movement must be at least one float32 ULP at the
initial logit. This tests optimizer-visible movement, not merely a
nonzero mathematical gradient.

## 5. Every-step shared-alpha audit

Movement columns are cumulative from initialization.

| step | alpha | task grad | actual movement | WD-only movement | task-excess | alpha value |
|---:|---|---:|---:|---:|---:|---:|
| 1 | shared.alpha_sem | 0.000e+00 | 1.099e-04 | 1.099e-04 | 0.000e+00 | 0.1000098884 |
| 1 | shared.alpha_ctx | 0.000e+00 | 1.099e-04 | 1.099e-04 | 0.000e+00 | 0.1000098884 |
| 2 | shared.alpha_sem | 1.437e-09 | 2.198e-04 | 2.198e-04 | 0.000e+00 | 0.1000197828 |
| 2 | shared.alpha_ctx | 3.314e-09 | 2.198e-04 | 2.198e-04 | 0.000e+00 | 0.1000197828 |
| 3 | shared.alpha_sem | 3.602e-09 | 3.297e-04 | 3.297e-04 | 0.000e+00 | 0.1000296772 |
| 3 | shared.alpha_ctx | 1.001e-08 | 3.297e-04 | 3.297e-04 | 0.000e+00 | 0.1000296772 |
| 4 | shared.alpha_sem | -5.631e-09 | 4.396e-04 | 4.396e-04 | 0.000e+00 | 0.1000395715 |
| 4 | shared.alpha_ctx | -1.101e-08 | 4.396e-04 | 4.396e-04 | 0.000e+00 | 0.1000395715 |
| 5 | shared.alpha_sem | 9.325e-09 | 5.496e-04 | 5.496e-04 | 0.000e+00 | 0.1000494659 |
| 5 | shared.alpha_ctx | 3.919e-08 | 5.496e-04 | 5.496e-04 | 0.000e+00 | 0.1000494659 |
| 6 | shared.alpha_sem | 8.832e-09 | 6.595e-04 | 6.595e-04 | 0.000e+00 | 0.1000593603 |
| 6 | shared.alpha_ctx | 2.991e-08 | 6.595e-04 | 6.595e-04 | 0.000e+00 | 0.1000593603 |
| 7 | shared.alpha_sem | 2.929e-08 | 7.694e-04 | 7.694e-04 | 0.000e+00 | 0.1000692621 |
| 7 | shared.alpha_ctx | 9.834e-08 | 7.694e-04 | 7.694e-04 | 0.000e+00 | 0.1000692621 |
| 8 | shared.alpha_sem | 5.695e-08 | 8.793e-04 | 8.793e-04 | 0.000e+00 | 0.1000791639 |
| 8 | shared.alpha_ctx | 1.829e-07 | 8.793e-04 | 8.793e-04 | 0.000e+00 | 0.1000791639 |
| 9 | shared.alpha_sem | 5.548e-08 | 9.892e-04 | 9.892e-04 | 0.000e+00 | 0.1000890583 |
| 9 | shared.alpha_ctx | 1.733e-07 | 9.892e-04 | 9.892e-04 | 0.000e+00 | 0.1000890583 |
| 10 | shared.alpha_sem | 8.736e-08 | 1.099e-03 | 1.099e-03 | 0.000e+00 | 0.1000989527 |
| 10 | shared.alpha_ctx | 2.810e-07 | 1.099e-03 | 1.099e-03 | 0.000e+00 | 0.1000989527 |
| 11 | shared.alpha_sem | 4.163e-08 | 1.209e-03 | 1.209e-03 | 0.000e+00 | 0.1001088545 |
| 11 | shared.alpha_ctx | 1.277e-07 | 1.209e-03 | 1.209e-03 | 0.000e+00 | 0.1001088545 |
| 12 | shared.alpha_sem | 1.125e-07 | 1.319e-03 | 1.319e-03 | 0.000e+00 | 0.1001187563 |
| 12 | shared.alpha_ctx | 3.635e-07 | 1.319e-03 | 1.319e-03 | -2.384e-07 | 0.1001187414 |
| 13 | shared.alpha_sem | 2.628e-08 | 1.429e-03 | 1.429e-03 | 0.000e+00 | 0.1001286656 |
| 13 | shared.alpha_ctx | 7.927e-08 | 1.429e-03 | 1.429e-03 | -2.384e-07 | 0.1001286432 |
| 14 | shared.alpha_sem | 1.631e-07 | 1.539e-03 | 1.539e-03 | -2.384e-07 | 0.1001385525 |
| 14 | shared.alpha_ctx | 5.225e-07 | 1.538e-03 | 1.539e-03 | -4.768e-07 | 0.1001385227 |
| 15 | shared.alpha_sem | 1.290e-07 | 1.648e-03 | 1.649e-03 | -4.768e-07 | 0.1001484320 |
| 15 | shared.alpha_ctx | 4.078e-07 | 1.648e-03 | 1.649e-03 | -7.153e-07 | 0.1001484096 |
| 16 | shared.alpha_sem | 2.049e-07 | 1.758e-03 | 1.758e-03 | -4.768e-07 | 0.1001583114 |
| 16 | shared.alpha_ctx | 6.467e-07 | 1.758e-03 | 1.758e-03 | -7.153e-07 | 0.1001582965 |
| 17 | shared.alpha_sem | 1.565e-07 | 1.868e-03 | 1.868e-03 | -4.768e-07 | 0.1001681983 |
| 17 | shared.alpha_ctx | 4.789e-07 | 1.867e-03 | 1.868e-03 | -7.153e-07 | 0.1001681760 |
| 18 | shared.alpha_sem | 2.277e-07 | 1.977e-03 | 1.978e-03 | -4.768e-07 | 0.1001780778 |
| 18 | shared.alpha_ctx | 7.415e-07 | 1.977e-03 | 1.978e-03 | -7.153e-07 | 0.1001780629 |
| 19 | shared.alpha_sem | 2.050e-07 | 2.087e-03 | 2.087e-03 | -4.768e-07 | 0.1001879722 |
| 19 | shared.alpha_ctx | 6.800e-07 | 2.087e-03 | 2.087e-03 | -7.153e-07 | 0.1001879498 |
| 20 | shared.alpha_sem | 2.402e-07 | 2.197e-03 | 2.197e-03 | -4.768e-07 | 0.1001978591 |
| 20 | shared.alpha_ctx | 7.670e-07 | 2.196e-03 | 2.197e-03 | -7.153e-07 | 0.1001978368 |

Final shadow decision:

| alpha | final task-excess logit | task-excess alpha | ULP | measurable |
|---|---:|---:|---:|---|
| shared.alpha_sem | -4.768e-07 | -4.470e-08 | 2.384e-07 | True |
| shared.alpha_ctx | -7.153e-07 | -6.706e-08 | 2.384e-07 | True |

## 6. Every-step per-stage mechanism audit

| step | stage | N mean | N std | G_sem mean | G_sem std | G_ctx mean | G_ctx std | gamma_sem | gamma_ctx | finite |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | stage1 | 0.1645 | 0.1616 | 0.9164 | 0.0162 | 0.9164 | 0.0162 | 0.000000 | 0.000000 | True |
| 1 | stage2 | 0.1886 | 0.1817 | 0.9189 | 0.0182 | 0.9189 | 0.0182 | 0.000000 | 0.000000 | True |
| 1 | stage3 | 0.1886 | 0.1817 | 0.9189 | 0.0182 | 0.9189 | 0.0182 | 0.000000 | 0.000000 | True |
| 2 | stage1 | 0.6252 | 0.1640 | 0.9625 | 0.0164 | 0.9625 | 0.0164 | -0.000002 | 0.000012 | True |
| 2 | stage2 | 0.6491 | 0.1703 | 0.9649 | 0.0170 | 0.9649 | 0.0170 | 0.000009 | 0.000014 | True |
| 2 | stage3 | 0.6490 | 0.1703 | 0.9649 | 0.0170 | 0.9649 | 0.0170 | -0.000054 | -0.000096 | True |
| 3 | stage1 | 0.6961 | 0.1819 | 0.9696 | 0.0182 | 0.9696 | 0.0182 | -0.000103 | -0.000154 | True |
| 3 | stage2 | 0.7184 | 0.1888 | 0.9718 | 0.0189 | 0.9718 | 0.0189 | 0.000003 | 0.000007 | True |
| 3 | stage3 | 0.7186 | 0.1889 | 0.9719 | 0.0189 | 0.9719 | 0.0189 | -0.000138 | -0.000232 | True |
| 4 | stage1 | 0.2899 | 0.1182 | 0.9290 | 0.0118 | 0.9290 | 0.0118 | -0.000106 | -0.000151 | True |
| 4 | stage2 | 0.3085 | 0.1297 | 0.9308 | 0.0130 | 0.9308 | 0.0130 | 0.000032 | 0.000050 | True |
| 4 | stage3 | 0.3087 | 0.1298 | 0.9308 | 0.0130 | 0.9308 | 0.0130 | -0.000221 | -0.000364 | True |
| 5 | stage1 | 0.3222 | 0.1494 | 0.9322 | 0.0149 | 0.9322 | 0.0149 | 0.000033 | 0.000107 | True |
| 5 | stage2 | 0.3413 | 0.1598 | 0.9341 | 0.0160 | 0.9341 | 0.0160 | 0.000047 | 0.000075 | True |
| 5 | stage3 | 0.3415 | 0.1599 | 0.9341 | 0.0160 | 0.9341 | 0.0160 | -0.000256 | -0.000433 | True |
| 6 | stage1 | 0.4722 | 0.1750 | 0.9472 | 0.0175 | 0.9472 | 0.0175 | 0.000244 | 0.000481 | True |
| 6 | stage2 | 0.4993 | 0.1873 | 0.9499 | 0.0187 | 0.9499 | 0.0187 | 0.000061 | 0.000095 | True |
| 6 | stage3 | 0.4997 | 0.1874 | 0.9499 | 0.0188 | 0.9499 | 0.0188 | -0.000291 | -0.000494 | True |
| 7 | stage1 | 0.5083 | 0.1928 | 0.9508 | 0.0193 | 0.9508 | 0.0193 | 0.000294 | 0.000571 | True |
| 7 | stage2 | 0.5318 | 0.2016 | 0.9532 | 0.0202 | 0.9532 | 0.0202 | 0.000114 | 0.000175 | True |
| 7 | stage3 | 0.5322 | 0.2017 | 0.9532 | 0.0202 | 0.9532 | 0.0202 | -0.000315 | -0.000540 | True |
| 8 | stage1 | 0.3723 | 0.1093 | 0.9372 | 0.0109 | 0.9372 | 0.0109 | 0.000454 | 0.000862 | True |
| 8 | stage2 | 0.3954 | 0.1190 | 0.9395 | 0.0119 | 0.9395 | 0.0119 | 0.000161 | 0.000245 | True |
| 8 | stage3 | 0.3957 | 0.1190 | 0.9395 | 0.0119 | 0.9395 | 0.0119 | -0.000358 | -0.000615 | True |
| 9 | stage1 | 0.4075 | 0.1570 | 0.9407 | 0.0157 | 0.9407 | 0.0157 | 0.000642 | 0.001186 | True |
| 9 | stage2 | 0.4336 | 0.1696 | 0.9433 | 0.0170 | 0.9433 | 0.0170 | 0.000187 | 0.000283 | True |
| 9 | stage3 | 0.4341 | 0.1697 | 0.9434 | 0.0170 | 0.9434 | 0.0170 | -0.000381 | -0.000656 | True |
| 10 | stage1 | 0.3510 | 0.1235 | 0.9350 | 0.0124 | 0.9350 | 0.0124 | 0.000772 | 0.001420 | True |
| 10 | stage2 | 0.3762 | 0.1355 | 0.9376 | 0.0136 | 0.9376 | 0.0136 | 0.000245 | 0.000372 | True |
| 10 | stage3 | 0.3766 | 0.1356 | 0.9376 | 0.0136 | 0.9376 | 0.0136 | -0.000423 | -0.000722 | True |
| 11 | stage1 | 0.3979 | 0.1090 | 0.9397 | 0.0109 | 0.9397 | 0.0109 | 0.000944 | 0.001724 | True |
| 11 | stage2 | 0.4260 | 0.1182 | 0.9425 | 0.0118 | 0.9425 | 0.0118 | 0.000280 | 0.000421 | True |
| 11 | stage3 | 0.4265 | 0.1183 | 0.9426 | 0.0118 | 0.9426 | 0.0118 | -0.000439 | -0.000761 | True |
| 12 | stage1 | 0.2472 | 0.1514 | 0.9246 | 0.0152 | 0.9246 | 0.0152 | 0.001000 | 0.001823 | True |
| 12 | stage2 | 0.2668 | 0.1648 | 0.9266 | 0.0165 | 0.9266 | 0.0165 | 0.000327 | 0.000489 | True |
| 12 | stage3 | 0.2670 | 0.1649 | 0.9266 | 0.0165 | 0.9266 | 0.0165 | -0.000459 | -0.000805 | True |
| 13 | stage1 | 0.3147 | 0.1456 | 0.9314 | 0.0146 | 0.9314 | 0.0146 | 0.001126 | 0.002048 | True |
| 13 | stage2 | 0.3399 | 0.1595 | 0.9339 | 0.0160 | 0.9339 | 0.0160 | 0.000367 | 0.000546 | True |
| 13 | stage3 | 0.3403 | 0.1596 | 0.9339 | 0.0160 | 0.9339 | 0.0160 | -0.000449 | -0.000801 | True |
| 14 | stage1 | 0.2768 | 0.1454 | 0.9276 | 0.0146 | 0.9276 | 0.0146 | 0.001147 | 0.002083 | True |
| 14 | stage2 | 0.3001 | 0.1602 | 0.9299 | 0.0160 | 0.9299 | 0.0160 | 0.000404 | 0.000603 | True |
| 14 | stage3 | 0.3004 | 0.1604 | 0.9300 | 0.0161 | 0.9300 | 0.0161 | -0.000420 | -0.000766 | True |
| 15 | stage1 | 0.2862 | 0.1669 | 0.9285 | 0.0167 | 0.9285 | 0.0167 | 0.001349 | 0.002433 | True |
| 15 | stage2 | 0.3091 | 0.1791 | 0.9308 | 0.0179 | 0.9308 | 0.0179 | 0.000440 | 0.000658 | True |
| 15 | stage3 | 0.3095 | 0.1792 | 0.9309 | 0.0179 | 0.9309 | 0.0179 | -0.000383 | -0.000729 | True |
| 16 | stage1 | 0.2954 | 0.1675 | 0.9294 | 0.0168 | 0.9294 | 0.0168 | 0.001470 | 0.002650 | True |
| 16 | stage2 | 0.3191 | 0.1807 | 0.9318 | 0.0181 | 0.9318 | 0.0181 | 0.000493 | 0.000736 | True |
| 16 | stage3 | 0.3194 | 0.1808 | 0.9318 | 0.0181 | 0.9318 | 0.0181 | -0.000368 | -0.000722 | True |
| 17 | stage1 | 0.3538 | 0.1237 | 0.9353 | 0.0124 | 0.9353 | 0.0124 | 0.001666 | 0.002997 | True |
| 17 | stage2 | 0.3845 | 0.1365 | 0.9384 | 0.0137 | 0.9384 | 0.0137 | 0.000538 | 0.000804 | True |
| 17 | stage3 | 0.3849 | 0.1365 | 0.9384 | 0.0137 | 0.9384 | 0.0137 | -0.000333 | -0.000680 | True |
| 18 | stage1 | 0.3518 | 0.1627 | 0.9351 | 0.0163 | 0.9351 | 0.0163 | 0.001809 | 0.003246 | True |
| 18 | stage2 | 0.3824 | 0.1786 | 0.9381 | 0.0179 | 0.9381 | 0.0179 | 0.000578 | 0.000863 | True |
| 18 | stage3 | 0.3828 | 0.1787 | 0.9382 | 0.0179 | 0.9382 | 0.0179 | -0.000314 | -0.000661 | True |
| 19 | stage1 | 0.3302 | 0.1717 | 0.9329 | 0.0172 | 0.9329 | 0.0172 | 0.002031 | 0.003647 | True |
| 19 | stage2 | 0.3584 | 0.1862 | 0.9357 | 0.0186 | 0.9357 | 0.0186 | 0.000620 | 0.000924 | True |
| 19 | stage3 | 0.3588 | 0.1863 | 0.9358 | 0.0187 | 0.9358 | 0.0187 | -0.000264 | -0.000604 | True |
| 20 | stage1 | 0.3241 | 0.1801 | 0.9323 | 0.0180 | 0.9323 | 0.0180 | 0.002169 | 0.003910 | True |
| 20 | stage2 | 0.3527 | 0.1952 | 0.9351 | 0.0196 | 0.9351 | 0.0196 | 0.000669 | 0.000997 | True |
| 20 | stage3 | 0.3530 | 0.1954 | 0.9352 | 0.0196 | 0.9352 | 0.0196 | -0.000229 | -0.000563 | True |

## 7. Resource profile

RTX 5090, batch20, 224x224, BF16, three warmups and ten
synchronized measured iterations per mode.

| quantity | CDSR-v2 vs A0 | budget | result |
|---|---:|---:|---|
| parameters | +2 (+0.0000%) | exactly +2 | PASS |
| estimated FLOPs | +0.0232% | <+0.1% | PASS |
| forward median latency | +2.5080% | <+5% | PASS |
| train median latency | +1.4378% | <+10% | PASS |
| forward peak memory | +1.9655% | reported | — |
| train peak memory | +1.4128% | reported | — |

## 8. Readiness matrix

| check | result |
|---|---|
| Need formula unchanged | PASS |
| exactly two shared parameter objects | PASS |
| stage-specific N retained | PASS |
| uniform exact A0 | PASS |
| all local/server tests | PASS |
| batch20 BF16 finite | PASS |
| pretrained/A0 compatibility | PASS |
| optimizer coverage | PASS |
| matched shadow LR | PASS |
| shared alpha_sem measurable task-excess | PASS |
| shared alpha_ctx measurable task-excess | PASS |
| overall | **PASS** |

## 9. Stop decision

CDSR-v2 is engineering-ready under the frozen gate. The next
possible action is the controlled 25-epoch BCSS seed-42 run,
but it is intentionally not started here and requires review.

CDSR_V2_READINESS_PASS
