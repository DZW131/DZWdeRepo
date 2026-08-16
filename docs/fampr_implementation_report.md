# FA-MPR Implementation Report

## 1. Executive conclusion

Full Frequency-Adaptive Morphology-Preserving Rectification (FA-MPR) v1.0 is
implemented on `feature/innovation1-fampr`, based directly on merged
`main@2923da2`. The implementation changes only the Contextual Homogenization
(CH) signal inside the three original HFRM stages. The backbone, Global
Semantic Rectification path, CAM heads, four-scale classification loss,
optimizer, learning-rate schedule, inference, and metrics are unchanged.

The implementation-readiness gate passes:

- the exact A0 default remains tensor-identical to the pre-FA-MPR code;
- all 43 local and CUDA tests pass;
- all FA-MPR paths receive finite nonzero gradients by optimization step 2;
- a batch-20, 224x224, BF16 training step and `forward_cam` pass on the 5090;
- the parameter increase is 279,855 parameters, or 0.2483%, below the fixed
  10% budget.

No formal BCSS training has been started. The next allowed experiment after PR
review and merge is one controlled BCSS seed-42, 25-epoch, final-checkpoint run.

## 2. Experimental control and isolation

The public selector is explicit:

| Command path | Constructed model |
|---|---|
| default or `--rectifier hfrm --context-mode ch` | exact A0 HFRM + CH15 |
| `--rectifier hfrm --context-mode fampr` | HFRM + Full FA-MPR |
| `--rectifier hst --hst-variant a1\|a2\|a3` | archived HST |

`--rectifier hst --context-mode fampr` raises an error. In a Full FA-MPR model,
there is no `hst_rectifier` instance and no parameter name containing `hst`.
The original CH helper was copied into the neutral `network/context.py` module,
so A0 and FA-MPR no longer import CH through the archived HST namespace. HST
keeps its own existing helper and remains reproducible, but is not on the active
model path.

The strongest A0 compatibility test still reports:

- total parameters: 112,709,714;
- state-key SHA256:
  `23038075b660d3f97ada855a9c138cda6f82711902214c2aa70e6a394e45b796`;
- every public forward output is exactly equal, tensor by tensor, to the
  pre-refactor HFRM computation.

## 3. Implemented Full FA-MPR architecture

For a stage feature `F`, fixed replicate-padded average pools form the
telescoping bands:

```text
B0 = F - LP3(F)
B1 = LP3(F) - LP7(F)
B2 = LP7(F) - LP15(F)
B3 = LP15(F)
```

Their sum reconstructs `F`. Channel-mean absolute band energies drive
`Conv3x3(4->16) -> GELU -> Conv1x1(16->4)`. The final convolution is
zero-initialized and the weights are `2 * sigmoid(logit)`, hence every initial
band weight is exactly one. A residual formulation makes frequency selection
an exact tensor identity at initialization.

The morphology sensitivity and dilation are:

```text
M = (A0 E0 + A1 E1) / (sum_i Ai Ei + 1e-6)
M = replicate_avg_pool3(M)
D = 1 + (1 - M) * 6
```

Thus `M` is in `[0, 1]`, `D` is in `[1, 7]`, and more morphology-sensitive
locations use smaller dilation. A pure-PyTorch vectorized sampler packs all
nine 3x3 positions into one `[B, H, W*9, 2]` grid and makes one
`grid_sample` call with `padding_mode=border`, `align_corners=True`, and
internal FP32 sampling.

Each stage owns a learnable depthwise Gaussian 3x3 base kernel. It is split
into its spatial mean and residual high-frequency component. Channel gates use
`Linear(C->max(C/16,16)) -> GELU -> Linear(hidden->2C)`; the last layer is
zero-initialized, so both `2 * sigmoid` gates start at one. The adaptive output
is anchored to original CH15:

```text
Y_FA  = g_low * Y_low + g_high * Y_high
Y_MPR = Y_CH + sigmoid(anchor_logit) * (Y_FA - Y_CH)
```

The anchor starts at 0.25 and remains learnable. The original HFRM semantic and
context residual gammas remain zero-initialized.

## 4. Files and interfaces

| File | Responsibility |
|---|---|
| `network/context.py` | neutral exact CH15 primitive |
| `network/fampr/frequency_selection.py` | fixed multiband decomposition, weights, morphology |
| `network/fampr/adaptive_sampler.py` | one-call vectorized spatial sampler |
| `network/fampr/adaptive_kernel.py` | Gaussian depthwise kernel spectrum and neutral gates |
| `network/fampr/fampr_context.py` | frozen configuration, full branch, CH anchor, diagnostics |
| `network/resnet38_cls.py` | A0/FA-MPR resolver and optimizer grouping |
| `train_sshr.py` | CLI resolver, run manifest, first-batch epoch diagnostics |
| `tools/smoke_fampr.py` | optimization/CAM/BF16 smoke |
| `tools/profile_fampr.py` | paired A0/FA-MPR resource profile |
| `tools/check_pretrained_load.py` | FA-MPR-aware pretrained audit |
| `tests/test_fampr.py` | component, isolation, integration, and gradient controls |

Each training directory records `experiment_config.json` with the commit,
environment, resolved configuration, dataset size, parameter counts, optimizer
groups, and pretrained-load result. FA-MPR runs additionally append one
first-training-batch diagnostic record per epoch to
`fampr_diagnostics.jsonl`. These records observe the model and do not add a
loss or alter optimization.

## 5. Verification

### 5.1 Unit and integration tests

`python -m unittest discover -s tests -v` passes 43/43 tests on both the local
environment and the 5090 environment. FA-MPR-specific coverage includes:

- exact telescoping-band reconstruction;
- exact frequency-selector identity and unit gates at initialization;
- morphology/dilation bounds and inverse ordering;
- adaptive integer sampling versus replicate-padded depthwise convolution,
  with maximum error below `1e-4`;
- exact kernel low/high reconstruction and unit channel gates;
- exact original-CH fallback when the anchor is manually set to zero;
- real SSHR stage channel/shape preservation;
- finite public forward diagnostics and finite `forward_cam`;
- every trainable parameter appears in exactly one optimizer group;
- no HST object or parameter enters Full FA-MPR;
- the frequency/adaptive path opens by step 2.

### 5.2 CUDA optimization smoke

Environment:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 D v2 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| image size | 224x224 |
| precision | BF16 autocast; adaptive sampling internally FP32 |

The five-step, batch-2 smoke passed all finite checks on every step. At step 1,
the zero-initialized HFRM context gamma received a nonzero gradient. By step 2,
all three stages had finite nonzero gradients in the band predictor, adaptive
base kernel, kernel-gate predictor, and anchor. `forward_cam` remained finite.
Peak allocated CUDA memory was 2,060,074,496 bytes (1.92 GiB).

The official-shape batch-20 one-step smoke also passed the forward, loss,
backward, gradients, optimizer step, and single-image `forward_cam` checks.
Peak allocated CUDA memory was 8,849,874,432 bytes (8.24 GiB). This one-step
control verifies feasibility; path-opening is established by the separate
five-step smoke.

### 5.3 Pretrained loading

The released MXNet file audit reports:

| Item | Value |
|---|---|
| size | 436,873,620 bytes |
| SHA256 | `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16` |
| converted keys | 191 |
| unexpected keys | 0 |

There are 60 missing model keys. Fifty-two are expected new HFRM/CAM/FA-MPR
keys. The remaining eight are the already-known `bn45`/`bn52` affine and
running-stat keys absent from the released conversion. No additional backbone
key is missing because of FA-MPR.

## 6. Resource profile

The paired profile used the same 5090 process, batch 20, 224x224, BF16, three
warmups, and ten measured iterations per model. A concurrent external GPU
workload was present, so latency values are paired engineering measurements,
not pristine benchmark claims.

| Measure | A0 CH | Full FA-MPR | Delta |
|---|---:|---:|---:|
| total parameters | 112,709,714 | 112,989,569 | +0.2483% |
| estimated FLOPs/image | 200.488 GFLOPs | 201.324 GFLOPs | +0.4168% |
| forward median/batch | 34.04 ms | 53.04 ms | +55.81% |
| train-step median/batch | 105.17 ms | 189.76 ms | +80.43% |
| forward peak allocated | 1.22 GiB | 2.30 GiB | +1.08 GiB |
| train peak allocated | 3.67 GiB | 7.72 GiB | +4.05 GiB |

FLOPs are exact for Conv2d/Linear multiply-and-adds plus an explicit estimate
for FA-MPR functional pooling, grid interpolation, kernel reductions, gates,
and anchoring. The sampler is memory/bandwidth intensive, so measured latency
and memory rise much more than arithmetic FLOPs.

## 7. FADC relationship and clean-room boundary

The design was checked against the official Frequency-Adaptive Dilated
Convolution (FADC) paper and repository. The useful transferable principles are
neutral `2*sigmoid` frequency weights, zero-initialized predictors, adaptive
dilation, and low/high kernel-spectrum modulation. The SSHR implementation is a
clean-room pure-PyTorch design specialized to morphology preservation; no FADC
source code was copied and no `mmcv` deformable-convolution dependency was
introduced.

- FADC paper: <https://arxiv.org/html/2403.05369>
- official FADC repository: <https://github.com/Linwei-Chen/FADC>

## 8. Remaining risks and next action

Implementation correctness and optimization connectivity are established, but
they do not demonstrate segmentation improvement. The main engineering cost is
the vectorized FP32 nine-point sample tensor, reflected in the batch-20 latency
and memory increase. No architectural or hyperparameter tuning should occur
before the controlled result is known.

After review and merge, run exactly one BCSS seed-42 experiment for 25 epochs
with final-checkpoint evaluation and the existing A0 protocol, adding only:

```bash
--rectifier hfrm --context-mode fampr
```

Then compare the final checkpoint directly with the frozen A0 seed-42 result.
Do not start LUAD, additional seeds, component ablations, or parameter searches
until that go/no-go result is reviewed.
