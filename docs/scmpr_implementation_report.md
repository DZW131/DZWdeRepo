# SC-MPR Initial Implementation Review (Archived Blocker Record)

> Status update (2026-08-16): the exact de-meaning blocker documented below
> was removed by the explicitly approved minimal patch. The patched design,
> CUDA gradient audit, batch-20 BF16 smoke, and current readiness decision are
> recorded in [`scmpr_readiness_patch_report.md`](scmpr_readiness_patch_report.md).
> This file is retained as the evidence that motivated that correction; its
> pre-patch formulas and stop decision are historical, not the current design.

## 1. Executive conclusion

Semantic-Conditioned Morphology-Preserving Rectification (SC-MPR) v1.0 has
been implemented on `feature/innovation1-scmpr`, based on the latest merged
`main@e4b7b6c`. The implementation follows the frozen forward equations,
preserves exact A0 CH behavior, keeps FA-MPR and HST isolated, passes all
component/integration/resource/pretrained controls, and fits batch 20 BF16 on
the RTX 5090.

However, the formal training readiness gate **does not pass**. The required
per-channel spatial de-meaning makes the SC-MPR residual mathematically
invisible to SSHR's only training objective. A five-step official-loss CUDA
smoke confirms that `gamma_context` opens but the policy input layer and both
compatibility projectors remain at zero-gradient scale. In accordance with the
frozen specification, no 25-epoch training was started.

## 2. Experimental control and isolation

The public selector is explicit:

| Command path | Constructed model |
|---|---|
| default or `--rectifier hfrm --context-mode ch` | exact A0 HFRM + CH15 |
| `--rectifier hfrm --context-mode sc-mpr` | HFRM + CH15 + SC-MPR residual |
| `--rectifier hfrm --context-mode fampr` | archived Full FA-MPR |
| `--rectifier hst` | archived HST |

Alternative HFRM contexts are rejected when `rectifier_type=hst`. The default
`ch` model remains tensor-identical to the explicit A0 model. SC-MPR does not
instantiate or call FA-MPR, HST, adaptive dilation, `grid_sample`, AdaKern,
learned smoothing kernels, or a global CH-vs-adaptive scalar blend.

The backbone, original GSR computation, trainable CH15, zero-initialized HFRM
semantic/context gammas, CAM heads, four classification-loss weights,
optimizer, poly schedule, augmentation, inference, CAM fusion, thresholds, and
metric are unchanged.

## 3. Implemented SC-MPR forward

Each target stage uses fixed replicate-padded unit-sum filters:

```text
R_fine  = F - LP3(F)
R_morph = LP3(F) - LP15(F)
Q       = clamp(mean_channel(abs(R)) / (spatial_mean + 1e-6), 0, 5)
```

The deep CAM logits are computed without dropout only for semantic
conditioning and detached immediately. Their softmax confidence, normalized
entropy, local probability variation, and cosine compatibility are combined
with `Q_fine/Q_morph`. Target projectors are stage-specific 1x1 convolutions
to 32 dimensions. One deep projector and one `6 -> 16 -> 2` policy object are
registered once and shared by all stages. Shared deep probabilities and the
deep projection are computed once per forward and resized per stage.

The final policy convolution is zero-initialized with bias `logit(0.1)`, so
both spatial gates start at 0.1. Each stage owns:

```text
beta = 0.5 * sigmoid(beta_logit), beta_init = 0.1
Delta = G_fine * R_fine + G_morph * R_morph
Delta_zero_mean = Delta - spatial_mean(Delta)
Y_SC = Y_CH + beta * Delta_zero_mean
F_R = F + gamma_sem * F_sem + gamma_context * Y_SC
```

Semantic inputs are stop-gradient, while the target/deep projectors remain
trainable in the graph. Diagnostics expose every proposal, semantic map, gate,
beta, residual, original CH, SC-MPR context, gradient, quantile, amplitude
ratio, and finite check required by the specification.

## 4. Verification summary

### 4.1 Unit and integration controls

On the RTX 5090 environment (`PyTorch 2.11.0+cu128`), 21/21 SC-MPR controls
pass, including the 20 requested controls:

- exact A0 default equivalence and constant-preserving LP3/LP15;
- finite normalized proposals and bounded semantic maps;
- semantic-input stop-gradient with learnable projectors;
- 0.1 gate initialization, shared policy identity, and beta controls;
- per-channel residual mean below `1e-6` in FP32;
- exact `beta=0 -> Y_SC=Y_CH` control;
- real hierarchy shapes and finite CAMs;
- optimizer coverage exactly once and unchanged frozen BN/backbone behavior;
- five-step CUDA component connectivity;
- batch-20, 224x224 BF16 official-path forward/backward/step;
- released MXNet pretrained conversion audit.

The additional 21st regression test proves the optimization blocker: a
per-channel zero-mean tensor has exactly zero effect through a shared 1x1 CAM
head followed by spatial global-average pooling.

The full repository suite passes locally: 64 tests total, with only the three
CUDA/pretrained controls skipped because the Windows environment lacks those
resources. The same three controls pass on the 5090.

### 4.2 Amplitude

At initialization under batch-2 BF16 official shapes:

| Stage | `||beta * Delta_zero_mean|| / ||Y_CH||` |
|---|---:|
| 56 | 0.011052 |
| 28_1 | 0.008361 |
| 28_2 | 0.007048 |

The initial context drift is about 0.7-1.1%, rather than the 24-28% drift
observed in the archived FA-MPR design. Constant inputs produce zero frequency
residuals, and the de-meaned residual is numerically centered.

## 5. Optimization blocker

Let `W` denote the shared spatial 1x1 CAM transform and `GAP` spatial global
average pooling. The SC-MPR contribution to a training logit is:

```text
GAP(W(gamma_context * beta * Delta_zero_mean))
= gamma_context * beta * W(GAP(Delta_zero_mean))
= 0
```

because `GAP(Delta_zero_mean)=0` independently for every feature channel. Thus
the classification logits and loss are invariant to `beta`, both gates, the
policy, and compatibility projectors. This is an exact algebraic cancellation,
not a hardware, random-seed, or mixed-precision effect.

The full-model five-step BF16 CUDA smoke observed:

| Step | `gamma_context(56)` grad norm | beta grad norm | policy output grad norm | policy input grad norm | target/deep projector grad norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.500e-3 | 0 | 0 | 0 | 0 |
| 2 | 4.274e-3 | 1.592e-12 | 3.063e-13 | 0 | 0 |
| 3 | 4.088e-3 | 2.831e-13 | 5.579e-13 | 0 | 0 |
| 4 | 3.946e-3 | 4.252e-13 | 1.078e-12 | 0 | 0 |
| 5 | 3.845e-3 | 9.314e-12 | 9.201e-14 | 0 | 0 |

The tiny nonzero beta/output-layer values are floating-point cancellation
residue and are not useful learning signals. Repeating in FP32 produced the
same conclusion (approximately `1e-17` policy-output gradients and zero
projector gradients). `forward_cam` remained finite and peak allocated memory
was 1.765 GiB for this batch-2 smoke.

## 6. Resource profile

Paired RTX 5090 measurements used batch 20, 224x224, BF16, three warmups and
ten measured iterations. Two unrelated GPU processes occupied about 3.9 GiB,
so latency is an engineering estimate rather than an isolated benchmark.

| Measure | A0 CH | SC-MPR | Delta |
|---|---:|---:|---:|
| parameters | 112,709,714 | 112,899,175 | +0.1681% |
| estimated FLOPs/image | 200.488 GFLOPs | 201.365 GFLOPs | +0.4376% |
| forward median/batch | 34.02 ms | 39.14 ms | +15.06% |
| train-step median/batch | 105.11 ms | 151.69 ms | +44.31% |
| forward peak allocated | 1.221 GiB | 1.275 GiB | +0.054 GiB |
| train peak allocated | 3.666 GiB | 4.111 GiB | +0.445 GiB |

Parameter and FLOP budgets (<1%) pass. Batch 20 does not OOM. Forward latency
is effectively at the preferred 15% boundary and below the 20% explanation
threshold. Training overhead is larger because gradients traverse the fixed
LP3/LP15 operations, even though their arithmetic count is small relative to
the backbone.

## 7. Pretrained loading

The released MXNet weight is unchanged:

| Item | Value |
|---|---|
| size | 436,873,620 bytes |
| SHA256 | `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16` |
| converted keys | 191 |
| unexpected keys | 0 |

All new SC-MPR/HFRM/CAM keys are expected missing keys. The only missing
backbone entries are the eight previously audited `bn45`/`bn52` affine and
running-stat keys; SC-MPR introduces no new missing backbone key.

## 8. Source relationship

The design was checked against the public primary sources. AFRDA refines
high-resolution features using low-resolution semantic priors, high-frequency
detail, and uncertainty. FASA motivates explicit semantic-frequency alignment
across scales. SC-MPR adopts these high-level principles but not their domain
adaptation objectives, decoders, DCT modules, contrastive losses, prototypes,
or source code.

- AFRDA paper: <https://arxiv.org/abs/2507.17957>
- AFRDA repository: <https://github.com/Masrur02/AFRDA>
- FASA paper: <https://arxiv.org/abs/2604.12341>

## 9. Decision required before training

No implementation can simultaneously keep all four of the following:

1. exact per-channel spatial de-meaning;
2. the unchanged linear CAM plus GAP training head;
3. no new spatial/auxiliary loss;
4. trainable SC-MPR gates/projectors from the classification objective.

At least one constraint must change. The smallest transparent option is to
remove exact spatial de-meaning and retain the existing bounded gate/beta
initialization. Other options are an explicit surrogate-gradient estimator,
a new structure-sensitive auxiliary loss, or relocating rectification before a
downstream nonlinear block; each is a larger methodological decision.

The current branch intentionally stops at this engineering-review gate. Do not
run BCSS/LUAD, additional seeds, validation sweeps, or ablations until one
option is explicitly approved and re-smoked.
