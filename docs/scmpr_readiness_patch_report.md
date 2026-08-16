# SC-MPR Readiness Patch Report

## 1. Why exact spatial de-meaning was invalid

The reviewed implementation formed a per-channel zero-mean residual

```text
Delta_zero_mean = Delta - mean_HW(Delta).
```

SSHR trains each scale through a spatially shared `1x1` CAM convolution,
global-average pooling (GAP), and the image-level classification loss. If `W`
is the CAM convolution, linearity gives

```text
GAP(W * Delta_zero_mean)
= W * GAP(Delta_zero_mean)
= 0.
```

Consequently, exact spatial de-meaning made the SC-MPR correction invisible
to the only training objective. This was an algebraic optimization blocker,
not a random-seed, hardware, or mixed-precision effect.

## 2. Exact approved change

The patch removes only the exact spatial de-meaning operation:

```text
Before: Delta_SC = Delta - mean_HW(Delta)
After:  Delta_SC = Delta

Y_SC = Y_CH + beta_i * Delta_SC
```

The safety description is correspondingly changed from exact zero-mean
preservation to **DC-safe / amplitude-controlled residual**.

## 3. Frozen equations and controls

Everything else remains frozen:

```text
R_fine  = F - LP3(F)
R_morph = LP3(F) - LP15(F)
Delta   = G_fine * R_fine + G_morph * R_morph
Y_SC    = Y_CH + beta_i * Delta
F_R     = F + gamma_sem * F_sem_original + gamma_context * Y_SC
```

The patch does not change the fixed unit-sum LP3/LP15 filters, replicate
padding, original CH15 anchor, GSR, detached deep semantic inputs, one shared
semantic-frequency policy, target/deep projectors, 0.10 gate initialization,
`beta_init=0.10`, `beta_max=0.50`, backbone, CAM heads, loss, optimizer,
schedule, inference, or metrics. It adds no module, class identity, auxiliary
loss, surrogate gradient, or validation-selected parameter.

## 4. Constant-input and DC-safe control

The obsolete zero-mean assertion was replaced by a stronger constant-input
control. With spatially constant `F`, the test verifies exactly that

```text
R_fine = R_morph = Delta_SC = 0
Y_SC = Y_CH.
```

The FP32 test uses exact tensor equality and passes. Thus fixed unit-sum
filters still reject DC input, while a non-constant gated residual may retain
a nonzero spatial mean and remain visible to the classification objective.

## 5. Classification-loss visibility

The regression test now constructs the actual learning path:

```text
SC-MPR context -> 1x1 CAM -> GAP -> multilabel classification loss.
```

After `gamma_context` opens, finite nonzero gradients reach `beta`, the
policy output and input layers, every stage target compatibility projector,
and the shared deep compatibility projector. A separate full-model smoke uses
the training class `Net.forward()` rather than `Net_CAM.forward()`; the latter
intentionally returns only the deep CAM and is not the formal training entry
point.

On the RTX 5090 environment (`PyTorch 2.11.0+cu128`), all 21 SC-MPR tests pass,
including CUDA, batch-20 BF16, and released-pretrained conversion controls.
The full local repository suite passes 64 tests, with only the three
CUDA/pretrained tests skipped on the Windows host.

## 6. Five-step official-loss gradient audit

The full model used seed 42, batch 20, `224x224`, BF16, the unchanged four
classification-loss weights, and the repository's official optimizer and
poly schedule. Values are shown as `parameter value / gradient L2 norm`.
Very small gradient norms are accumulated in float64 by the diagnostic tool
to prevent false zeros from float32 square-and-sum underflow.

| Step | Stage | `gamma_sem` value / grad | `gamma_context` value / grad | `beta` value / grad | target-projector grad |
|---:|---|---:|---:|---:|---:|
| 1 | 56 | `0 / 2.744e-5` | `0 / 4.626e-5` | `0.100000 / 0` | `0` |
| 1 | 28_1 | `0 / 7.374e-5` | `0 / 1.111e-4` | `0.100000 / 0` | `0` |
| 1 | 28_2 | `0 / 3.424e-5` | `0 / 5.175e-5` | `0.100000 / 0` | `0` |
| 2 | 56 | `-2.744e-6 / 2.144e-5` | `-4.626e-6 / 3.565e-5` | `0.100006 / 5.344e-14` | `0` |
| 2 | 28_1 | `-7.374e-6 / 6.949e-5` | `-1.111e-5 / 1.046e-4` | `0.100006 / 2.956e-13` | `0` |
| 2 | 28_2 | `3.424e-6 / 3.888e-5` | `5.175e-6 / 5.898e-5` | `0.100006 / 4.843e-13` | `0` |
| 3 | 56 | `-4.498e-6 / 1.555e-5` | `-7.544e-6 / 2.525e-5` | `0.100010 / 6.452e-14` | `2.078e-27` |
| 3 | 28_1 | `-1.306e-5 / 6.593e-5` | `-1.967e-5 / 9.904e-5` | `0.100010 / 6.094e-13` | `5.296e-27` |
| 3 | 28_2 | `6.605e-6 / 4.276e-5` | `1.000e-5 / 6.501e-5` | `0.100010 / 9.641e-13` | `3.730e-27` |
| 4 | 56 | `-5.480e-6 / 1.226e-5` | `-9.139e-6 / 1.941e-5` | `0.100014 / 5.713e-14` | `5.804e-27` |
| 4 | 28_1 | `-1.723e-5 / 6.328e-5` | `-2.593e-5 / 9.495e-5` | `0.100014 / 8.596e-13` | `1.670e-26` |
| 4 | 28_2 | `9.306e-6 / 4.521e-5` | `1.411e-5 / 6.881e-5` | `0.100014 / 1.383e-12` | `1.235e-26` |
| 5 | 56 | `-6.018e-6 / 1.013e-5` | `-9.991e-6 / 1.567e-5` | `0.100016 / 5.429e-14` | `9.558e-27` |
| 5 | 28_1 | `-2.000e-5 / 6.044e-5` | `-3.009e-5 / 9.060e-5` | `0.100016 / 1.046e-12` | `3.062e-26` |
| 5 | 28_2 | `1.129e-5 / 4.674e-5` | `1.713e-5 / 7.119e-5` | `0.100016 / 1.710e-12` | `2.354e-26` |

Shared-path gradient norms were:

| Step | policy output | policy input | shared deep projector | Full core path active? |
|---:|---:|---:|---:|---|
| 1 | `0` | `0` | `0` | No; zero-initialized gammas open first |
| 2 | `2.422e-13` | `0` | `0` | Partial |
| 3 | `4.214e-13` | `4.046e-26` | `5.356e-27` | **Yes** |
| 4 | `5.680e-13` | `1.285e-25` | `1.689e-26` | Yes |
| 5 | `7.140e-13` | `2.468e-25` | `3.092e-26` | Yes |

All losses, activations, diagnostics, and gradients were finite at every
step. Loss decreased from `0.693152` to `0.692776`. The full transition path
became nonzero at step 3, so the stipulated step-5 stop condition was not
triggered. Projector and policy-input signals are numerically small and should
therefore remain explicit health diagnostics during any later full run, but
they are no longer identically cancelled.

Across the five steps, the amplitude diagnostics remained bounded:

| Stage | residual mean ratio | scaled residual RMS ratio | context shift ratio | signed residual mean | gate means |
|---|---:|---:|---:|---:|---:|
| 56 | `0.002152-0.002153` | `0.011011-0.011014` | `0.011133-0.011134` | about `-5.06e-6` | `0.099609 / 0.099609` |
| 28_1 | `0.003708-0.003710` | `0.008353-0.008354` | `0.008511-0.008513` | about `5.19e-5` | `0.099609 / 0.099609` |
| 28_2 | `0.003454-0.003455` | `0.007043-0.007044` | `0.007224-0.007226` | about `2.77e-5` | `0.099609 / 0.099609` |

The two gate means are `G_fine / G_morph`; BF16 rounds the frozen 0.10
initialization to `0.099609`. `beta` stayed between `0.100000` and `0.100016`.
No amplitude diagnostic is used as a loss.

## 7. Batch-20 BF16 official-path smoke

The required `batch=20`, `224x224`, BF16 sequence completed on an NVIDIA
GeForce RTX 5090 D v2:

- five `Net.forward -> classification loss -> backward -> optimizer.step`
  iterations completed without OOM;
- all SC-MPR core parameters were gradient-connected by step 3;
- a state-equivalent `Net_CAM.forward_cam` pass was finite;
- peak allocated CUDA memory for the combined smoke was 5,174,427,136 bytes
  (4.82 GiB), including the co-resident inference model used for the final
  `forward_cam` check.

## 8. Patched resource profile

Paired measurements used the same RTX 5090, batch 20, `224x224`, BF16, three
warmups, and ten measured iterations. Parameter count is unchanged from the
reviewed SC-MPR implementation, as required.

| Measure | A0 CH | Patched SC-MPR | Delta |
|---|---:|---:|---:|
| parameters | 112,709,714 | 112,899,175 | +189,461 / +0.1681% |
| estimated FLOPs/image | 200.488 GFLOPs | 201.361 GFLOPs | +0.4356% |
| forward median/batch | 30.57 ms | 35.06 ms | +14.70% |
| train-step median/batch | 94.17 ms | 123.32 ms | +30.95% |
| forward peak allocated | 1.221 GiB | 1.275 GiB | +0.055 GiB |
| train peak allocated | 3.667 GiB | 4.106 GiB | +0.440 GiB |

The frozen `<1%` parameter and estimated-FLOP gates pass. The preferred
`<15%` forward-latency gate also passes in this paired run. Latencies are
engineering measurements rather than isolated hardware benchmarks.

## 9. Readiness recommendation

**Recommendation: the patched SC-MPR is engineering-ready for one controlled
BCSS seed-42 validation experiment after code review/merge and explicit run
approval.** The exact optimization blocker is removed, all requested controls
pass, the full path becomes nonzero by step 3, and batch-20 BF16 fits the
available GPU.

This recommendation is not a claim of segmentation improvement. The very
small downstream projector gradients make per-epoch gradient, gate, beta,
residual-amplitude, and finite diagnostics mandatory during the first run. No
BCSS/LUAD training, seed sweep, validation tuning, or inference change was
performed as part of this readiness patch.
