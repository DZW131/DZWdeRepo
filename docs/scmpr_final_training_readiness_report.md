# SC-MPR Final Training Readiness Report

## 1. Executive decision

**Do not start the 25-epoch BCSS run.** The only approved model change was
implemented exactly: the final shared policy `Conv1x1(16 -> 2)` now uses
`nn.init.xavier_uniform_(weight, gain=0.01)`, while its bias remains
`logit(0.1)`.

This change solves the 12-13-order relative gradient attenuation: by step 10,
the policy-input/output gradient ratio is `4.93e-2`, the shared-deep/output
ratio is `1.98e-3`, and stage-target/output ratios are
`1.05e-4-1.82e-3`. The full graph is active by step 2.

However, final training readiness still fails for two measured reasons:

1. the small Xavier perturbation is visible in FP32, but BF16 rounds every
   initial gate to the same `0.099609375` on both fixed-random and real BCSS
   inputs;
2. after 10 official optimizer steps, every tracked parameter is exactly
   equal to a weight-decay-only counterfactual. The apparent parameter
   movement is entirely optimizer weight decay; loss-driven movement is zero
   at stored FP32 parameter precision.

All outputs and gradients remain finite, batch 20 fits, `forward_cam` is
finite, A0 behavior is unchanged, and all safety tests pass. These properties
are necessary but do not override the failed nonconstant-BF16-gate and
measurable-learning criteria.

## 2. Frozen control and exact change

No SC-MPR forward equation was changed in this review. The following remain
frozen: `R_fine`, `R_morph`, `Q_fine/Q_morph`, semantic confidence,
uncertainty, variation, feature compatibility, the policy architecture,
`beta`, CH15 anchor, GSR, loss, optimizer, inference, and metric.

The complete model change is:

```python
# Before
nn.init.zeros_(final_layer.weight)
nn.init.constant_(final_layer.bias, logit(0.1))

# After
nn.init.xavier_uniform_(final_layer.weight, gain=0.01)
nn.init.constant_(final_layer.bias, logit(0.1))
```

`beta_init=0.10`, `beta_max=0.50`, and zero-initialized `gamma_context` are
unchanged. No auxiliary loss, surrogate gradient, new module, class identity,
or validation-selected setting was introduced.

## 3. Initialization safety

Seed 42 was used throughout. Fixed-random statistics use a batch of 20
normalized `224x224` tensors. Real-input statistics use the first four
lexicographically sorted images under the read-only BCSS training root. Gate
cells show `mean / std / min / max`; drift is
`||Y_SC - Y_CH||_2 / ||Y_CH||_2`.

### 3.1 FP32 initialization

| Input | Stage | `G_fine` | `G_morph` | context drift |
|---|---|---|---|---:|
| fixed random | 56 | `0.099941 / 3.52e-5 / 0.099718 / 0.100120` | `0.099914 / 4.42e-5 / 0.099756 / 0.100173` | 1.1075% |
| fixed random | 28_1 | `0.099944 / 4.43e-5 / 0.099754 / 0.100117` | `0.099910 / 5.46e-5 / 0.099784 / 0.100170` | 0.8403% |
| fixed random | 28_2 | `0.099944 / 4.34e-5 / 0.099765 / 0.100113` | `0.099909 / 5.49e-5 / 0.099788 / 0.100155` | 0.7086% |
| real BCSS | 56 | `0.099928 / 7.91e-5 / 0.099376 / 0.100138` | `0.099931 / 1.06e-4 / 0.099527 / 0.100662` | 0.4933% |
| real BCSS | 28_1 | `0.099937 / 6.57e-5 / 0.099689 / 0.100148` | `0.099916 / 9.17e-5 / 0.099550 / 0.100263` | 0.5480% |
| real BCSS | 28_2 | `0.099939 / 6.00e-5 / 0.099720 / 0.100141` | `0.099915 / 8.13e-5 / 0.099612 / 0.100206` | 0.5085% |

In FP32, both gates are nonconstant, remain centered near 0.1, and retain the
same sub-1.2% context-drift regime as the preceding implementation.

### 3.2 BF16 initialization

| Input | Stage | `G_fine` mean/std/min/max | `G_morph` mean/std/min/max | context drift |
|---|---|---|---|---:|
| fixed random | 56 | `0.099609 / 0 / 0.099609 / 0.099609` | `0.099609 / 0 / 0.099609 / 0.099609` | 1.1133% |
| fixed random | 28_1 | `0.099609 / 0 / 0.099609 / 0.099609` | `0.099609 / 0 / 0.099609 / 0.099609` | 0.8513% |
| fixed random | 28_2 | `0.099609 / 0 / 0.099609 / 0.099609` | `0.099609 / 0 / 0.099609 / 0.099609` | 0.7227% |
| real BCSS | 56 | `0.099609 / 0 / 0.099609 / 0.099609` | `0.099609 / 0 / 0.099609 / 0.099609` | 0.5104% |
| real BCSS | 28_1 | `0.099609 / 0 / 0.099609 / 0.099609` | `0.099609 / 0 / 0.099609 / 0.099609` | 0.5689% |
| real BCSS | 28_2 | `0.099609 / 0 / 0.099609 / 0.099609` | `0.099609 / 0 / 0.099609 / 0.099609` | 0.5295% |

The mean and drift safety requirements pass, but the requested nonconstant
initial policy does not survive the actual BF16 forward path. The exact zero
standard deviation in all six cases is consistent with the `gain=0.01`
pre-sigmoid perturbation being below BF16 resolution around the
`logit(0.1)` bias.

## 4. Ten-step official-loss gradient audit

The audit used batch 20, `224x224`, BF16, the unchanged SSHR four-scale
classification loss, official parameter groups, official `PolyOptimizer`, and
the official poly schedule. Vectors are ordered `56 / 28_1 / 28_2`. L2 norms
are accumulated in FP64 by diagnostics to avoid false zero from reduction
underflow.

| Step | `gamma_context` grad | beta grad | policy output grad | policy input grad | target-projector grad | deep-projector grad |
|---:|---|---|---:|---:|---|---:|
| 1 | `4.25e-4 / 5.74e-4 / 9.30e-5` | `0 / 0 / 0` | `0` | `0` | `0 / 0 / 0` | `0` |
| 2 | `4.10e-4 / 5.63e-4 / 8.23e-5` | `4.53e-12 / 8.59e-12 / 6.28e-13` | `1.20e-11` | `5.92e-13` | `1.97e-14 / 1.63e-14 / 1.48e-15` | `2.15e-14` |
| 3 | `3.94e-4 / 5.55e-4 / 7.56e-5` | `8.19e-12 / 1.57e-11 / 1.05e-12` | `2.18e-11` | `1.08e-12` | `3.67e-14 / 3.04e-14 / 2.58e-15` | `4.01e-14` |
| 4 | `3.81e-4 / 5.48e-4 / 7.03e-5` | `1.11e-11 / 2.18e-11 / 1.36e-12` | `3.01e-11` | `1.48e-12` | `5.13e-14 / 4.25e-14 / 3.47e-15` | `5.60e-14` |
| 5 | `3.70e-4 / 5.43e-4 / 6.47e-5` | `1.34e-11 / 2.69e-11 / 1.60e-12` | `3.70e-11` | `1.82e-12` | `6.37e-14 / 5.33e-14 / 4.16e-15` | `6.97e-14` |
| 6 | `3.63e-4 / 5.38e-4 / 6.00e-5` | `1.53e-11 / 3.10e-11 / 1.74e-12` | `4.23e-11` | `2.09e-12` | `7.45e-14 / 6.18e-14 / 4.65e-15` | `8.06e-14` |
| 7 | `3.54e-4 / 5.34e-4 / 5.64e-5` | `1.67e-11 / 3.41e-11 / 1.82e-12` | `4.64e-11` | `2.28e-12` | `8.31e-14 / 6.88e-14 / 5.07e-15` | `9.04e-14` |
| 8 | `3.48e-4 / 5.30e-4 / 5.35e-5` | `1.78e-11 / 3.65e-11 / 1.88e-12` | `4.95e-11` | `2.44e-12` | `8.97e-14 / 7.40e-14 / 5.34e-15` | `9.72e-14` |
| 9 | `3.43e-4 / 5.27e-4 / 5.15e-5` | `1.88e-11 / 3.84e-11 / 1.90e-12` | `5.22e-11` | `2.57e-12` | `9.51e-14 / 7.87e-14 / 5.52e-15` | `1.03e-13` |
| 10 | `3.40e-4 / 5.25e-4 / 4.99e-5` | `1.91e-11 / 3.99e-11 / 1.91e-12` | `5.41e-11` | `2.67e-12` | `9.85e-14 / 8.25e-14 / 5.68e-15` | `1.07e-13` |

All recorded losses, outputs, diagnostics, and gradients are finite. The full
semantic-conditioning graph becomes nonzero at step 2.

## 5. Gradient-ratio audit

| Step | policy input/output | deep projector/output | target 56/output | target 28_1/output | target 28_2/output |
|---:|---:|---:|---:|---:|---:|
| 2 | `4.92e-2` | `1.79e-3` | `1.64e-3` | `1.36e-3` | `1.23e-4` |
| 3 | `4.93e-2` | `1.84e-3` | `1.68e-3` | `1.39e-3` | `1.18e-4` |
| 4 | `4.92e-2` | `1.86e-3` | `1.70e-3` | `1.41e-3` | `1.15e-4` |
| 5 | `4.92e-2` | `1.89e-3` | `1.72e-3` | `1.44e-3` | `1.12e-4` |
| 6 | `4.93e-2` | `1.90e-3` | `1.76e-3` | `1.46e-3` | `1.10e-4` |
| 7 | `4.92e-2` | `1.95e-3` | `1.79e-3` | `1.48e-3` | `1.09e-4` |
| 8 | `4.93e-2` | `1.96e-3` | `1.81e-3` | `1.49e-3` | `1.08e-4` |
| 9 | `4.93e-2` | `1.98e-3` | `1.82e-3` | `1.51e-3` | `1.06e-4` |
| 10 | `4.93e-2` | `1.98e-3` | `1.82e-3` | `1.53e-3` | `1.05e-4` |

The approved initialization therefore removes the previous relative
12-13-order attenuation. The smallest step-10 ratio is about `1.05e-4`, not
`1e-12`. This ratio-level criterion passes. Absolute gradients nevertheless
remain small enough that the resulting loss-driven optimizer increments do
not alter stored parameters within 10 steps.

## 6. Parameter-update and movement audit

### 6.1 Actual optimizer update norms per step

| Step | policy input | policy output | deep projector | target-projector range | beta (each stage) |
|---:|---:|---:|---:|---:|---:|
| 1 | `1.141e-4` | `9.529e-7` | `1.631e-4` | `1.631e-4-1.641e-4` | `6.926e-5` |
| 2 | `1.038e-4` | `8.671e-7` | `1.484e-4` | `1.484e-4-1.493e-4` | `6.306e-5` |
| 3 | `9.336e-5` | `7.798e-7` | `1.335e-4` | `1.335e-4-1.343e-4` | `5.674e-5` |
| 4 | `8.279e-5` | `6.914e-7` | `1.184e-4` | `1.184e-4-1.191e-4` | `5.031e-5` |
| 5 | `7.206e-5` | `6.018e-7` | `1.030e-4` | `1.030e-4-1.036e-4` | `4.375e-5` |
| 6 | `6.116e-5` | `5.108e-7` | `8.743e-5` | `8.745e-5-8.794e-5` | `3.719e-5` |
| 7 | `5.003e-5` | `4.177e-7` | `7.152e-5` | `7.154e-5-7.194e-5` | `3.040e-5` |
| 8 | `3.861e-5` | `3.225e-7` | `5.520e-5` | `5.521e-5-5.553e-5` | `2.348e-5` |
| 9 | `2.681e-5` | `2.238e-7` | `3.833e-5` | `3.833e-5-3.855e-5` | `1.633e-5` |
| 10 | `1.436e-5` | `1.200e-7` | `2.054e-5` | `2.054e-5-2.066e-5` | `8.702e-6` |

These nonzero update norms alone are misleading: the unchanged official
optimizer applies weight decay to these parameters even when their loss
gradient is zero.

### 6.2 Ten-step relative movement and weight-decay counterfactual

A matched shadow started from the same tracked parameters and applied the
same per-group learning rates, weight decay, momentum, and poly schedule, but
zero task gradients. `Task excess` is the L2 distance between the real
parameter and this weight-decay-only shadow after step 10.

| Parameter group | `||theta10-theta0|| / ||theta0||` | actual delta | decay-only delta | task-excess delta |
|---|---:|---:|---:|---:|
| policy input layer | `2.850e-4` | `6.571e-4` | `6.571e-4` | `0` |
| policy output layer | `1.766e-6` | `5.488e-6` | `5.488e-6` | `0` |
| shared deep projector | `2.880e-4` | `9.394e-4` | `9.394e-4` | `0` |
| target projector 56 | `2.874e-4` | `9.449e-4` | `9.449e-4` | `0` |
| target projector 28_1 | `2.877e-4` | `9.428e-4` | `9.428e-4` | `0` |
| target projector 28_2 | `2.879e-4` | `9.396e-4` | `9.396e-4` | `0` |
| beta 56 | `2.880e-4` | `3.992e-4` | `3.992e-4` | `0` |
| beta 28_1 | `2.880e-4` | `3.992e-4` | `3.992e-4` | `0` |
| beta 28_2 | `2.880e-4` | `3.992e-4` | `3.992e-4` | `0` |

The real and decay-only parameters are identical at FP32 storage precision.
Therefore the requirement for measurable loss-driven movement fails, despite
the much healthier gradient ratios.

## 7. Safety and isolation regression

On the RTX 5090 (`PyTorch 2.11.0+cu128`), all 21 SC-MPR controls pass:

- constant-input DC safety and exact `Y_SC=Y_CH`;
- exact default A0 versus explicit `context_mode=ch` equivalence;
- semantic-input stop-gradient with trainable projectors;
- one shared policy object;
- exact beta-zero CH fallback and unchanged beta bounds;
- batch-20, `224x224`, BF16 forward/loss/backward/step;
- finite `forward_cam`;
- exact optimizer coverage without duplication;
- unchanged backbone/BN freezing;
- released MXNet pretrained conversion with no new missing backbone keys.

The full local suite reports 64 tests total: 61 pass and the three
CUDA/pretrained tests are skipped because those resources are absent on the
Windows host. No behavior outside `context_mode=sc-mpr` changed.

The final 10-step smoke allocated 5,171,324,416 CUDA bytes (4.82 GiB),
including the co-resident inference model used for the final CAM check. It did
not OOM.

## 8. Resource profile

Paired measurements used batch 20, `224x224`, BF16, three warmups, and ten
measured iterations on the same RTX 5090. The initialization-only change adds
no parameters or operations.

| Measure | A0 CH | SC-MPR | Delta |
|---|---:|---:|---:|
| parameters | 112,709,714 | 112,899,175 | +0.1681% |
| estimated FLOPs/image | 200.488 GFLOPs | 201.361 GFLOPs | +0.4356% |
| forward median/batch | 30.34 ms | 34.86 ms | +14.88% |
| train-step median/batch | 93.74 ms | 122.92 ms | +31.12% |
| forward peak allocated | 1.221 GiB | 1.275 GiB | +0.055 GiB |
| train peak allocated | 3.667 GiB | 4.106 GiB | +0.440 GiB |

Parameter, estimated-FLOP, and preferred forward-latency gates pass.

## 9. Final recommendation

The final readiness checklist is:

| Criterion | Result |
|---|---|
| gate mean remains approximately 0.1 | Pass |
| initial context drift remains small | Pass |
| policy/projector relative gradients are practically active | Pass |
| parameters measurably move from the task loss over 10 steps | **Fail** |
| BF16 gates are initially spatially/semantically nonconstant | **Fail** |
| all outputs and gradients remain finite | Pass |
| A0 and non-SC-MPR behavior remain unchanged | Pass |

**Final recommendation: not ready for full BCSS seed-42 training.** The
approved Xavier `gain=0.01` change improves backward gradient ratios but does
not create a nonconstant BF16 forward gate and does not produce any stored
task-driven parameter movement in the 10-step audit. No BCSS training,
validation tuning, inference change, or unapproved model modification was
performed. Further model decisions require a new explicit review; this audit
stops here.
