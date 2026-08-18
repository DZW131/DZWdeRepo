# OSMF-v1.3 Local Structural Morphology Learning — Delivery

## 1. Final decision

**OSMF_V13_READINESS_NOGO**

Stage A exact parity passed. Stage B completed exactly eight real BCSS training batches and failed one preregistered hard connectivity requirement: `u_morph.weight` receives no direct gradient from `L_struct`. Consequently, Stage C (fresh 128-batch Phase-0S) was not unlocked and was not run.

## 2. Provenance and frozen boundary

- Executed commit: `5a19675e76e60a020892be934936aa19f31b03fa`
- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Dataset: BCSS only
- Seed/batch/image/precision: `20260817 / 20 / 224 / BF16`
- Objective weights: semantic/structural/orthogonality/reconstruction = `0.05/0.05/0.05/0.10`
- Structural interval: 4 optimizer steps
- Local loss: masked SmoothL1, `beta=1.0`, direction-aware inverse alignment
- Tests on the 5090 server: `115 passed`
- No checkpoint was saved. Test, LUAD, segmentation GT, Phase-0S, pilot, and full training were not run.

The preliminary `c3f40a5` readiness output is excluded from formal evidence because its SSHR-loss stability diagnostic compared unmatched shuffled batches with an invalid relative heuristic. Commit `5a19675` corrected only the audit logic, added direct semantic-gradient evidence, and reran parity/readiness from scratch.

## 3. Stage A — exact parity

**OSMF_V13_PARITY_PASS**

- Full BCSS validation images: 3418
- Differing segmentation pixels: 0
- mIoU absolute difference: 0
- mDice absolute difference: 0
- Random-input and real-input maximum differences for reconstructed H28_1, all four CAMs, and classification probability: 0
- New trainable tensors: exactly four frozen-factorizer projection weights
- Parameter overhead: 524,288 parameters (0.465167%)

The absolute A0 validation score is not used for selection or model assessment here; parity only tests exact equality between A0 and v1.3 in the same run.

## 4. Stage B — eight-batch readiness

### 4.1 Structural causal signal

| Step | StructErr before | StructErr after | Delta |
|---:|---:|---:|---:|
| 4 | 0.000156335605 | 0.000071610652 | -0.000084724954 |
| 8 | 0.000500571914 | 0.000361059094 | -0.000139512820 |

- Improved/harmed/neutral: `2 / 0 / 0`
- Improved fraction: `1.0`
- Mean causal delta: `-0.000112118887`

The short same-pair causal check is favorable.

### 4.2 Gradient budgets

| Objective | Mean | Max | P95 |
|---|---:|---:|---:|
| semantic preservation | 0.141535 | 0.219207 | 0.211888 |
| structural affinity | 0.002988 | 0.007145 | 0.006340 |
| orthogonality | 0.053258 | 0.054173 | 0.054096 |
| reconstruction | 0.005386 | 0.016878 | 0.014793 |

Both preregistered semantic and structural budgets pass comfortably.

### 4.3 Direct objective connectivity

| Step | `p_morph` from `L_struct` | `u_morph` from `L_struct` | `p_sem` from `L_sem` | `u_sem` from `L_sem` |
|---:|---:|---:|---:|---:|
| 4 | 8.296409e-4 | 0 | 0.334831 | 0.327774 |
| 8 | 2.560043e-3 | 0 | 0.217932 | 0.216239 |

All four projection tensors do receive finite gradients and measurable updates from the **joint total loss**. However, the technical specification explicitly requires both morphology tensors to receive a finite non-zero gradient directly from `L_struct` at steps 4 and 8. `u_morph` fails that condition exactly.

### 4.4 Representation safety

- All tensors, losses, and gradients: finite
- Semantic path: active
- SSHR loss: stable (no numerical explosion)
- SemAgree: `0.856729 -> 0.936311`
- Reconstruction cosine: `1.000000 -> 0.999103`
- CrossCov: `0.125875 -> 0.088994`
- No semantic, morphology, response, or reconstruction collapse

## 5. Root-cause interpretation

This failure follows directly from the frozen computation graph rather than weak numerical scale:

```text
H --P_morph--> M --U_morph--> H_morph
                 |
                 +--> A_M --> L_struct
```

The specified `L_struct` is computed on the local affinities of latent morphology `M`. Therefore `L_struct` is downstream of `P_morph` but upstream of, and independent from, `U_morph`:

```text
d L_struct / d P_morph != 0
d L_struct / d U_morph  = 0
```

Making the second derivative non-zero would require changing the frozen objective to act on a quantity downstream of `U_morph` (for example the reconstructed morphology contribution), or changing the architecture. Neither change is authorized by the v1.3 specification, so the readiness gate cannot be repaired by an implementation-only adjustment.

## 6. Stop boundary

The exact gate rule has been enforced:

- `OSMF_V13_READINESS_NOGO`
- no fresh 128-batch Phase-0S
- no fixed-probe Phase-0S decision
- no validation-based tuning
- no v1.4 proposal or implementation
- no 3-epoch/25-epoch experiment

The next action requires human scientific review of the inconsistency between the frozen `L_struct(A_M)` equation and the requirement that `u_morph` receive a direct `L_struct` gradient.

## 7. Evidence locations

- `artifacts/osmf_v13/5a19675/parity/summary.json`
- `artifacts/osmf_v13/5a19675/readiness_8b/summary.json`
- `artifacts/osmf_v13/5a19675/readiness_8b/tables/morphology_structural_gradients.csv`
- `artifacts/osmf_v13/5a19675/readiness_8b/tables/semantic_preservation_gradients.csv`
- `artifacts/osmf_v13/5a19675/readiness_8b/tables/same_pair_causal.csv`
- `artifacts/osmf_v13/5a19675/readiness_8b/tables/gradient_ratios.csv`

Server archive: `/home/duyanhong/experiments/OSMF_V13_LOCAL_STRUCTURAL_5a19675`.
