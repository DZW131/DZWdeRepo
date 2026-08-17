# CDSR Implementation Readiness Report

## 1. Executive decision

**CDSR_IMPLEMENTATION_READINESS_FAIL**

The frozen Full CDSR architecture is structurally correct, finite,
A0-compatible, and within every resource budget. However, the mandatory
20-real-step weight-decay-shadow audit passes for only 4/6 alpha scalars.
Both F28_2 alpha logits remain bitwise identical to their matched
weight-decay-only shadows. Therefore the frozen readiness gate fails and
the 25-epoch BCSS experiment must not start.

This report does not change the Need formula, remove F28_2, create
class-specific behavior, tune D/U/R, alter alpha initialization, or
modify the loss/optimizer.

## 2. Frozen implementation

```text
N = R * (1 - (1-D) * (1-U))
G_sem = 1 - alpha_sem * (1 - N)
G_ctx = 1 - alpha_ctx * (1 - N)
F_R = F + gamma_sem * G_sem * F_sem + gamma_ctx * G_ctx * F_CH15
```

- original GSR semantic content is unchanged
- original CH15 context is unchanged
- raw probes reuse `ic_56`, `ic1`, `ic2`, and `fc8` under no-grad
- all D/U/R/N math is detached FP32 and exactly matches Phase 0
- there is no new classifier, spatial predictor, uncertainty head, or loss
- three stages each add semantic/context alpha logits: six scalars total
- alpha initializes to 0.10; initial gates lie in [0.9, 1.0]
- `uniform` remains the default exact SSHR path
- CDSR rejects FA-MPR and archived HST combinations

## 3. Test and integration evidence

- local full suite: **66 passed**
- RTX 5090 server full suite: **66 passed**
- batch20 / 224 / BF16 real-data forward-backward: finite for 20/20 steps
- optimizer coverage: every trainable parameter exactly once; all six
  alpha logits are in the original from-scratch weight group
- tested: uniform equivalence, frozen formula equality, JSD symmetry and
  bounds, entropy/reliability/Need bounds, detached probe, shared CAM
  heads, six-scalar count, alpha initialization, gate bounds, alpha=0
  exact HFRM fallback, shapes, finite CAMs, CLI isolation, and optimizer
  coverage

## 4. A0 and pretrained compatibility

- A0 checkpoint: `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- exact uniform strict load: **True**
- CDSR unexpected keys: 0
- CDSR-only missing keys: 6
- additional parameters: 6
- compatibility audit: **PASS**

The six CDSR-only missing keys are exactly the six new alpha logits;
the released backbone-pretrained audit has the same missing/unexpected
keys as A0 plus only these six expected scalars.

## 5. Real BCSS 20-step smoke

- parsed training samples: 23422
- batch / image size: 20 / 224
- seed / steps: 42 / 20
- precision: bf16
- official loss weights: [0.1, 0.15, 0.25, 0.5]
- optimizer: official PolyOptimizer/SGD
- base LR / max-step: 0.01 / 29275
- momentum / weight decay: 0.0005 / 0.0005
- PyTorch / CUDA: 2.11.0+cu128 / 12.8
- GPU: NVIDIA GeForce RTX 5090 D v2
- step-1 / step-20 loss: 0.872995 / 0.628621
- minimum observed loss: 0.566187
- all outputs/losses/alpha gradients finite: **True**

Step-20 mechanism state:

| Stage | N mean | N std | G_sem mean | G_ctx mean | gamma_sem | gamma_ctx | finite |
|---|---:|---:|---:|---:|---:|---:|---|
| stage1 | 0.3241 | 0.1801 | 0.9323 | 0.9323 | 0.002169 | 0.003910 | True |
| stage2 | 0.3527 | 0.1952 | 0.9351 | 0.9351 | 0.000669 | 0.000997 | True |
| stage3 | 0.3530 | 0.1954 | 0.9352 | 0.9352 | -0.000229 | -0.000563 | True |

## 6. Matched weight-decay-only shadow audit

A shadow copy of every alpha logit used the same group-2 LR, poly
schedule, momentum, and weight decay, but received zero task gradient.
The preregistered numerical criterion is divergence from the shadow
by at least one float32 ULP at the initial logit. LR equality held at
all 20 steps.

| Alpha | step-20 task grad | actual logit movement | WD-only movement | task-excess logit | ULP | measurable |
|---|---:|---:|---:|---:|---:|---|
| stage1.alpha_ctx | 7.606e-07 | 2.196e-03 | 2.197e-03 | -7.153e-07 | 2.384e-07 | True |
| stage1.alpha_sem | 2.339e-07 | 2.197e-03 | 2.197e-03 | -4.768e-07 | 2.384e-07 | True |
| stage2.alpha_ctx | 3.314e-08 | 2.197e-03 | 2.197e-03 | -2.384e-07 | 2.384e-07 | True |
| stage2.alpha_sem | 1.501e-08 | 2.197e-03 | 2.197e-03 | -2.384e-07 | 2.384e-07 | True |
| stage3.alpha_ctx | -2.673e-08 | 2.197e-03 | 2.197e-03 | 0.000e+00 | 2.384e-07 | False |
| stage3.alpha_sem | -8.694e-09 | 2.197e-03 | 2.197e-03 | 0.000e+00 | 2.384e-07 | False |

Result: **4/6** alpha scalars show measurable
task-excess movement. F28_2 semantic/context gradients are nonzero
mathematically, but their optimizer updates remain below float32
resolution and are erased by rounding at every observed step.
Even the four measurable scalars are only 1-3 ULPs from their
shadows; task-excess is 0.011% to
0.033% of the matched weight-decay-only
movement. This is weak partial activation, not a healthy readiness
pass.

## 7. Resource profile

Measured on the same RTX 5090 with batch20, 224x224, BF16, three
warmups and ten synchronized measured iterations per mode.

| Quantity | CDSR vs A0 | frozen budget | result |
|---|---:|---:|---|
| parameters | +6 (+0.0000%) | exactly +6 | PASS |
| estimated FLOPs | +0.0232% | <+0.1% | PASS |
| forward median latency | +2.5811% | <+5% | PASS |
| train median latency | +1.8183% | <+10% | PASS |
| forward peak memory | +1.9657% | reported | — |
| train peak memory | +1.4961% | reported | — |

FLOPs combine exact Conv2d/Linear multiply-add counts with the
explicit analytical-operation estimate recorded in the profile JSON.

## 8. Readiness matrix

| Check | Result |
|---|---|
| frozen formula unchanged | PASS |
| A0 uniform compatibility | PASS |
| six alpha scalars only | PASS |
| all local/server tests | PASS |
| batch20 BF16 finite | PASS |
| pretrained audit | PASS |
| optimizer coverage | PASS |
| matched shadow LR | PASS |
| resource budget | PASS |
| measurable task-excess for all six alphas | **FAIL (4/6)** |
| overall readiness | **FAIL** |

## 9. Stop decision

Per the frozen specification, engineering readiness is not granted.
Do not start the 25-epoch BCSS experiment. No architecture or
hyperparameter remedy is proposed or applied in this branch; the
result is stopped for review exactly at the readiness gate.

## 10. Reproduction commands

```bash
python -m pytest -q
python tools/check_cdsr_a0_compatibility.py --checkpoint <A0.pth> \
  --output-json audit/results/cdsr_a0_compatibility.json
python tools/smoke_cdsr.py --train-root <BCSS-training> \
  --weights <ResNet38.params> --dataset bcss --batch-size 20 \
  --steps 20 --formal-epochs 25 --image-size 224 --seed 42 \
  --output-json audit/results/cdsr_readiness_smoke.json
python tools/profile_cdsr.py --batch-size 20 --image-size 224 \
  --warmup 3 --iterations 10 \
  --output-json audit/results/cdsr_resource_profile.json
```

CDSR_IMPLEMENTATION_READINESS_FAIL
