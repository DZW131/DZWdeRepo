# OSMF-v1.3-R1 Graph-Corrected Audit — Project Delivery Summary

## What Was Implemented

- Model changes: none. OSMF-v1.3 factorization, local-affinity loss, all objective weights, optimizer, schedule, and augmentation remain byte-for-byte unchanged.
- Audit changes: corrected the morphology graph contract so `grad(L_struct,u_morph)=0` is expected, while `grad(L_total,u_morph)>0` and measurable parameter movement remain mandatory.
- Stage A: combined real-BCSS graph audit and full 3418-image validation parity.
- Stage B: fresh 8-batch readiness with exact same-pair step-4/8 causal checks.
- Stage C: gated fresh 128-batch Phase-0S with all 32 active-step causal checks and a fixed GT-free 64-image probe.
- Decision changes: versioned outputs `OSMF_V13R1_*`; fixed morphology probe must improve rather than inherit an unrelated absolute tolerance.
- Documentation: archived the supplied R1 plan, executable commands, raw CSV/JSON evidence, and the final route-level stop decision.

## Final Decision

**OSMF_V13R1_PHASE0S_NOGO**

The corrected connectivity gate passes. The local structural objective nevertheless fails to generalize: fixed-probe morphology affinity error rises from `0.01048904` to `0.01329310`, a relative worsening of `26.7333%`. Per the frozen R1 stop rule, the OSMF local-structural morphology-specialization line ends here; no v1.4/v1.5 or pilot is authorized.

## Provenance

- Executed commit: `ce97889049d4212bd6c95e355642f5dee277c45d`
- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Dataset: BCSS only
- Seed/batch/image/precision: `20260817 / 20 / 224 / BF16`
- Objective weights: semantic/structural/orthogonality/reconstruction = `0.05/0.05/0.05/0.10`
- Structural interval: 4
- SmoothL1 beta: 1.0
- Server: RTX 5090 D v2, PyTorch 2.11.0+cu128, CUDA 12.8

The earlier `d631917` Phase-0S output is excluded from formal evidence because its audit counted seven Phase-0S gradient observations using an 8-batch-only `len == 2` condition and inherited a non-R1 fixed-probe tolerance. No scientific/model setting changed in `ce97889`; all stages were rerun from scratch under the corrected same-commit gate.

## Validation Evidence

### Local and server tests

- Local final tests: `118 passed, 3 skipped` (CUDA-only tests skipped locally).
- RTX 5090 final tests: `121 passed`.
- All formal tensors, losses, gradients, and outputs are finite.
- No checkpoint was saved.

### Stage A — Graph + Parity

Decision: **OSMF_V13R1_GRAPH_PARITY_PASS**

| Required relationship | Gradient norm | Result |
|---|---:|---|
| `grad(L_struct,p_morph)>0` | 0.000551470 | PASS |
| `grad(L_struct,u_morph)=0` | 0 | PASS (expected by graph) |
| `grad(L_total,p_morph)>0` | 0.035933793 | PASS |
| `grad(L_total,u_morph)>0` | 0.040237200 | PASS |

- Full BCSS validation images: 3418
- Differing prediction pixels: 0
- mIoU absolute difference: 0
- mDice absolute difference: 0
- Random and real-input reconstructed features, four CAMs, and classification probabilities: exact equality

### Stage B — Fresh 8-Batch Readiness

Decision: **OSMF_V13R1_READINESS_PASS**

- Step 4/8 causal deltas: `-8.50338e-5`, `-1.39321e-4`
- Improved fraction: `2/2 = 1.0`
- Mean causal delta: `-1.12177e-4`
- `r_sem` mean/max: `0.141721 / 0.219207`
- `r_struct` mean/max: `0.002991 / 0.007158`
- All four projection tensors: finite total-loss gradients and measurable parameter updates
- Reconstruction cosine end: `0.999100`
- SemAgree end: `0.936161`
- CrossCov: `0.125875 -> 0.089090`
- No collapse or SSHR-loss explosion

### Stage C — Fresh 128-Batch Phase-0S

Decision: **OSMF_V13R1_PHASE0S_NOGO**

| Same-pair result | Count / value |
|---|---:|
| Improved | 16 |
| Harmed | 15 |
| Neutral | 1 |
| Improved fraction | 0.500000 |
| Mean delta | -0.0000075251 |
| Median delta | -0.0000011348 |

The mean same-pair delta is slightly favorable, but the improved fraction misses the 0.75 GO threshold and only reaches the REVIEW boundary.

| Fixed 64-image probe | Step 0 | Step 128 | Relative improvement |
|---|---:|---:|---:|
| AffinityEqErr morphology | 0.01048904 | 0.01329310 | -26.7333% |
| AffinityEqErr semantic control | 0.00893960 | 0.01049240 | -17.3699% |
| Raw EqErr morphology | 0.06337955 | 0.07856521 | -23.9600% |

Negative improvement means worsening. The primary fixed morphology probe fails decisively and triggers the formal NOGO.

### Phase-0S health controls

- Graph expectation: PASS at all recorded audit points
- `p_morph` direct structural gradient: finite/non-zero at steps 4, 8, 16, 32, 64, 96, 128
- `u_morph` direct structural gradient: exactly zero at all those steps, as expected
- All four factorization tensors: measurable cumulative updates
- `r_sem` mean/max/p95: `0.161645 / 0.274865 / 0.254330`
- `r_struct` mean/max/p95: `0.003073 / 0.007145 / 0.005761`
- SemAgree: `0.856729 -> 0.987612`
- Reconstruction cosine: `1.000000 -> 0.998205`
- CrossCov: `0.125875 -> 0.114262`
- SSHR loss: stable

These controls rule out dead gradients, auxiliary domination, semantic collapse, reconstruction failure, and numerical instability as the primary reason for NOGO.

## Final Command Index

Run from the repository root with the final executed commit checked out.

### Graph + parity

```bash
python tools/audit_osmf_v13r1_graph_parity.py \
  --train-root /path/to/BCSS-WSSS/training \
  --val-root /path/to/BCSS-WSSS/val \
  --checkpoint /path/to/a0/stage1_last.pth \
  --output-dir /path/to/output/graph_parity \
  --osmf-v13r1-commit ce97889049d4212bd6c95e355642f5dee277c45d \
  --num-workers 4
```

### Fresh 8-batch readiness

```bash
python tools/audit_osmf_v13r1_gradient_gate.py \
  --gate readiness \
  --train-root /path/to/BCSS-WSSS/training \
  --checkpoint /path/to/a0/stage1_last.pth \
  --parity-summary /path/to/output/graph_parity/summary.json \
  --output-dir /path/to/output/readiness_8b \
  --audit-commit ce97889049d4212bd6c95e355642f5dee277c45d \
  --num-workers 4
```

### Fresh 128-batch Phase-0S

```bash
python tools/audit_osmf_v13r1_gradient_gate.py \
  --gate phase0s \
  --train-root /path/to/BCSS-WSSS/training \
  --checkpoint /path/to/a0/stage1_last.pth \
  --parity-summary /path/to/output/graph_parity/summary.json \
  --readiness-summary /path/to/output/readiness_8b/summary.json \
  --output-dir /path/to/output/phase0s_128b \
  --audit-commit ce97889049d4212bd6c95e355642f5dee277c45d \
  --num-workers 4
```

The gate tools reject test/LUAD paths and expose no epoch or checkpoint-save option.

## Artifact Locations

- Formal repository archive: `artifacts/osmf_v13r1/ce97889/`
- Stage A summary: `artifacts/osmf_v13r1/ce97889/graph_parity/summary.json`
- Stage B summary: `artifacts/osmf_v13r1/ce97889/readiness_8b/summary.json`
- Stage C summary: `artifacts/osmf_v13r1/ce97889/phase0s_128b/summary.json`
- Full causal table: `artifacts/osmf_v13r1/ce97889/phase0s_128b/tables/same_pair_causal.csv`
- Fixed probe trajectory and manifest: `artifacts/osmf_v13r1/ce97889/phase0s_128b/tables/fixed_probe.csv`, `fixed_probe_manifest.csv`
- Gradient and update tables: `artifacts/osmf_v13r1/ce97889/phase0s_128b/tables/`
- Server archive: `/home/duyanhong/experiments/OSMF_V13R1_GRAPH_CORRECTED_ce97889`

## Remaining Items

- Nothing remains within the R1 authorization.
- Test, LUAD, 3-epoch pilot, 25-epoch training, and checkpoint creation were intentionally not run.
- Per the explicit R1 stop rule, OSMF local-structural morphology specialization is closed after this NOGO; do not continue with v1.4/v1.5 patches.
