# OSMF-v1.1 Readiness Audit

## 1. Decision

**OSMF_V11_SEMANTIC_READINESS_REVIEW**

Processed real BCSS batches: 8/8.
Decision reasons: `['SEMANTIC_RATIO_REMAINS_ABOVE_PASS_RANGE']`.
Flags: `[]`.

## 2. Frozen contract

- Audit commit: `35591791e0bd81edaf53183afbf319358ccb7b81`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Parity proof SHA256: `12ffd37b13203e59e8012683108673fc41302b43fa919b0f4642cd126a4b15a4`
- BCSS train only; seed 20260817; batch 20; 224x224; BF16.
- Fixed objective weights: 0.20/0.20/0.05/0.10.
- Fresh A0 checkpoint and optimizer state; no continuation from another audit.
- Exact command: `tools/audit_osmf_v11_gradient_gate.py --gate readiness --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --parity-summary /home/duyanhong/experiments/OSMF_V11_SEMANTIC_PRESERVATION_3559179/parity/summary.json --output-dir /home/duyanhong/experiments/OSMF_V11_SEMANTIC_PRESERVATION_3559179/readiness_8b --audit-commit 35591791e0bd81edaf53183afbf319358ccb7b81 --num-workers 4`

## 3. Main mechanism table

| Metric | Start | Mean | End | Min | Max |
|---|---:|---:|---:|---:|---:|
| L_SSHR | 0.169584 | 0.315096 | 0.445859 | 0.169584 | 0.551597 |
| L_sem_pres | 0.143288 | 0.0874802 | 0.0609632 | 0.0346736 | 0.16492 |
| L_eq | 0.0625734 | 0.088104 | 0.129739 | 0.0625734 | 0.129739 |
| L_orth | 0.0158448 | 0.0138864 | 0.00811768 | 0.00811768 | 0.0173229 |
| L_rec | 0 | 0.00048752 | 0.00161344 | 0 | 0.00161344 |
| r_sem_pres | 0.876828 | 0.598807 | 0.715869 | 0.380616 | 0.876828 |
| cos(base,sem_pres) | 0.116062 | 0.0393146 | 0.0110015 | -0.0770865 | 0.116062 |
| r_eq | 0.332644 | 0.38835 | 0.599952 | 0.289275 | 0.599952 |
| cos(base,eq) | 0.00561597 | -0.000480343 | -0.00822942 | -0.00822942 | 0.00561597 |
| r_orth | 0.053213 | 0.0512966 | 0.0500481 | 0.0472149 | 0.0547103 |
| cos(base,orth) | -0.0152036 | -0.0125238 | -0.00983088 | -0.0252554 | 0.000194852 |
| r_rec | 2.11865e-08 | 0.00623983 | 0.0162712 | 2.11865e-08 | 0.0162712 |
| cos(base,rec) | 0 | 0.09459 | 0.43261 | -0.0718894 | 0.43261 |
| RMS(H) | 0.81863 | 0.754926 | 0.56913 | 0.56913 | 0.81863 |
| RMS(S) | 0.919284 | 0.840142 | 0.648868 | 0.648868 | 0.919284 |
| RMS(M) | 0.703722 | 0.655601 | 0.473301 | 0.473301 | 0.708199 |
| RMS(H_hat) | 0.81863 | 0.753245 | 0.567153 | 0.567153 | 0.81863 |
| RMS(S)/RMS(M) | 1.30632 | 1.28709 | 1.37094 | 1.22695 | 1.37094 |
| Cos(H,H_hat) | 1 | 0.999601 | 0.998917 | 0.998917 | 1 |
| ResidualRatio | 0 | 0.023155 | 0.0458155 | 0 | 0.0458155 |
| CrossCov(S,M) | 0.0158448 | 0.0147268 | 0.0100741 | 0.0100741 | 0.0166061 |
| EqErr(M) | 0.0625734 | 0.0687909 | 0.0993292 | 0.0595025 | 0.0993292 |
| EqErr(S) | 0.0456588 | 0.0535085 | 0.0757599 | 0.0450091 | 0.0757599 |
| RMS(Z_S) | 4.76613 | 4.31902 | 3.54678 | 3.54678 | 4.76613 |
| RMS(Z_H) | 5.80145 | 5.42155 | 4.66729 | 4.66729 | 5.81716 |
| R_Z | 0.821541 | 0.79534 | 0.759923 | 0.759923 | 0.827436 |
| SemAgree | 0.856712 | 0.9063 | 0.978579 | 0.856712 | 0.978579 |

## 4. Parameter health

| Parameter | Grad nonzero | Mean grad norm | Absolute update | Relative update |
|---|---:|---:|---:|---:|
| `p_sem.weight` | True | 0.102299 | 0.0509464 | 0.00318415 |
| `p_morph.weight` | True | 0.0648009 | 0.0288839 | 0.00180524 |
| `u_sem.weight` | True | 0.107887 | 0.0460467 | 0.00287792 |
| `u_morph.weight` | True | 0.0518108 | 0.0308643 | 0.00192902 |

## 5. Semantic preservation answers

1. v1.0 r_sem range was 2.480527–4.106567; v1.1 r_sem_pres range is 0.380616–0.876828, mean 0.598807.
2. p_sem/u_sem active: True.
3. Semantic response non-degenerate: True; end R_Z=0.759923, SemAgree=0.978579.
4. End reconstruction cosine: 0.99891675.
5. Next gate authorized: none.

## 6. Morphology and reconstruction

- Morphology objective gradient active: True.
- EqErr(M) start/end: 0.0625734 / 0.0993292.
- S/M RMS ratio end: 1.37094.
- CrossCov start/end: 0.0158448 / 0.0100741.

## 7. Safety and boundary

- All finite: True.
- Auxiliary semantic ic1 gradient-free: True.
- Original SSHR path updated ic1: True.
- Validation performance evaluated: false.
- Test/LUAD/segmentation GT accessed: false.
- Checkpoint saved for continuation: false.
- Phase 1 started: false.

The run stops at this gate. Even a Phase-0 GO requires separate human authorization for a 3-epoch pilot.

OSMF_V11_SEMANTIC_READINESS_REVIEW
