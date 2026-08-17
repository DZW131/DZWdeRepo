# OSMF-v1.0 Phase-0 Delivery

## Outcome

**`OSMF_PHASE0_NOGO`**

The preregistered audit stopped after 2/128 real BCSS training batches because
the weighted semantic-objective gradient ratio at post-HFRM `H28_1` exceeded
`0.50` at two consecutive audit points:

| Audit step | r_sem | r_eq | r_orth | r_rec |
|---:|---:|---:|---:|---:|
| 1 | 4.106567 | 0.332644 | 0.053213 | 0.00000002 |
| 2 | 2.480527 | 0.318972 | 0.058430 | 0.012912 |

The exact hard-stop reason is
`PERSISTENT_SEM_GRADIENT_RATIO_GT_0_50`. Per the frozen specification, the run
was stopped immediately; no lambda, architecture, optimizer, or initialization
was changed in response, and Phase 1 was not started.

## Scope and provenance

- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- OSMF Phase -1 parent: `5eb7b258f0cdeb4fa8779b65e716c105c9541f9a`
- Executed Phase-0 audit commit: `39fdf788aed6d0e31bd42108d87fc502a37d591a`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Dataset: BCSS training split only, parsed count 23,422
- Seed / batch / image / precision: `20260817 / 20 / 224 / BF16`
- Fixed loss weights: semantic `0.20`, morphology `0.20`, orthogonality `0.05`, reconstruction `0.10`
- Released optimizer behavior: PolyOptimizer, momentum `0.0005`, LR power `0.9`
- Environment: PyTorch `2.11.0+cu128`, CUDA `12.8`, RTX 5090 D v2

No validation metric, test data, LUAD data, segmentation ground truth, 3-epoch
pilot, or 25-epoch training was used.

## Preflight and start-state safety

The server test suite passed `43/43`, including batch-20 CUDA BF16 exact
identity under explicitly enabled A0 TF32. The formal start state was:

- `max|H_hat - H| = 0`
- reconstruction cosine `= 1.0`
- all outputs and losses finite
- only the six expected OSMF keys missing from the A0 checkpoint

An earlier preflight attempt at commit `8ae91a6` stopped before any optimizer
step because PyTorch 2.11's new per-operator TF32 API was not overridden by the
legacy cuDNN context. The compatibility patch temporarily selects IEEE FP32
only for the four OSMF projection convolutions and restores A0 TF32 immediately
afterward. This enforces the already-frozen exact-identity contract without
changing the model equations, parameters, losses, or A0 operators.

## Mechanism observations before the hard stop

- All six OSMF parameters received finite, nonzero task gradients and measurable updates.
- Reconstruction cosine remained `0.996959` at step 2.
- Semantic/morphology RMS ratio remained healthy: `1.3063 -> 1.1199`.
- Cross-covariance changed `0.0158448 -> 0.0153385` without branch collapse.
- Morphology equivariance gradients were connected and finite.
- No NaN, Inf, SSHR-loss explosion, or dead path was observed.

The failure is therefore localized: with the frozen coefficient `0.20`, the
randomly initialized auxiliary semantic classifier dominates the original SSHR
gradient at `H28_1` immediately. It is not a failure of connectivity,
reconstruction identity, or numerical stability.

## Parameter movement at step 2

| Parameter | Absolute cumulative update | Relative update |
|---|---:|---:|
| `p_sem.weight` | 0.118780 | 0.007424 |
| `p_morph.weight` | 0.005070 | 0.000317 |
| `u_sem.weight` | 0.006175 | 0.000386 |
| `u_morph.weight` | 0.005511 | 0.000344 |
| `semantic_classifier.weight` | 0.072546 | 0.025565 |
| `semantic_classifier.bias` | 0.011196 | not meaningful from zero initialization |

The raw generated relative-update figure is visually dominated by the
zero-initialized classifier bias because its relative denominator is epsilon.
The underlying absolute and relative values remain available in CSV. The report
generator has been corrected to plot absolute cumulative movement in future
runs; the downloaded raw artifact is preserved unchanged.

## Resource observation

Peak training-step allocation was 3.687 GiB. Runtime overhead cannot be
estimated from this run because the hard stop occurred before the first
scheduled equivariance training step at step 4. The raw two-step timing rows are
retained but must not be interpreted as an OSMF overhead estimate.

## Decision boundary

The frozen OSMF-v1.0 Phase-0 criteria do not permit Phase 1. Any future change
to the semantic loss scale, auxiliary head initialization, or objective design
would constitute a new, separately preregistered OSMF version—not a continuation
of this run.

Raw report, tables, figures, environment capture, command, and log are archived
under [`audit/results/OSMF_V1_PHASE0_128B_39fdf78`](../audit/results/OSMF_V1_PHASE0_128B_39fdf78/ARTIFACTS.md).

