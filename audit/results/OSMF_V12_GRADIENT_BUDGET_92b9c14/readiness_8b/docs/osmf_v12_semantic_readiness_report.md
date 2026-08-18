# OSMF-v1.2 Readiness Audit

## 1. Decision

**OSMF_V12_READINESS_PASS**

Processed real BCSS batches: 8/8.
Decision reasons: `[]`.
Flags: `[]`.

## 2. Frozen contract

- Audit commit: `92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Parity proof SHA256: `e2585306d7f43be4842ea39168ff2e5c7f2d64cf86e6332c1e11d8eccfef0dcf`
- BCSS train only; seed 20260817; batch 20; 224x224; BF16.
- Fixed objective weights: 0.05/0.05/0.05/0.10.
- Fresh A0 checkpoint and optimizer state; no continuation from another audit.
- Exact command: `tools/audit_osmf_v12_gradient_gate.py --gate readiness --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --parity-summary /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/parity/summary.json --output-dir /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/readiness_8b --audit-commit 92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4 --num-workers 4`

## 3. Main mechanism table

| Metric | Start | Mean | End | Min | Max |
|---|---:|---:|---:|---:|---:|
| L_total | 0.17754 | 0.286647 | 0.30432 | 0.17754 | 0.469526 |
| L_SSHR | 0.169584 | 0.280132 | 0.296363 | 0.169584 | 0.465097 |
| L_sem_pres | 0.143288 | 0.0963815 | 0.0405204 | 0.0393335 | 0.189816 |
| L_eq | 0.0625734 | 0.0804161 | 0.107091 | 0.0625734 | 0.107091 |
| L_orth | 0.0158448 | 0.013707 | 0.0105487 | 0.00897158 | 0.016645 |
| L_rec | 0 | 0.000174416 | 0.000495315 | 0 | 0.000616431 |
| r_sem_pres | 0.219207 | 0.136138 | 0.12305 | 0.0977935 | 0.219207 |
| cos(base,sem_pres) | 0.116062 | 0.100995 | 0.192291 | -0.0545924 | 0.192291 |
| r_eq | 0.0831609 | 0.0921116 | 0.13127 | 0.0725833 | 0.13127 |
| cos(base,eq) | 0.00561597 | 0.00573643 | 0.0131349 | 0.000184424 | 0.0131349 |
| r_orth | 0.053213 | 0.0563164 | 0.0718695 | 0.0452439 | 0.0718695 |
| cos(base,orth) | -0.0152036 | -0.013118 | -0.0123439 | -0.0261308 | 0.00120644 |
| r_rec | 2.11865e-08 | 0.00307699 | 0.0086317 | 2.11865e-08 | 0.0086317 |
| cos(base,rec) | 0 | 0.0989313 | 0.433922 | -0.0666469 | 0.433922 |
| RMS(H) | 0.81863 | 0.760341 | 0.620253 | 0.620253 | 0.81863 |
| RMS(S) | 0.919284 | 0.85329 | 0.733636 | 0.733636 | 0.919284 |
| RMS(M) | 0.703722 | 0.651618 | 0.479132 | 0.479132 | 0.705545 |
| RMS(H_hat) | 0.81863 | 0.759245 | 0.619504 | 0.619504 | 0.81863 |
| RMS(S)/RMS(M) | 1.30632 | 1.32299 | 1.53118 | 1.23443 | 1.53118 |
| Cos(H,H_hat) | 1 | 0.999895 | 0.999692 | 0.999692 | 1 |
| ResidualRatio | 0 | 0.0118637 | 0.0245555 | 0 | 0.0245555 |
| CrossCov(S,M) | 0.0158448 | 0.0144546 | 0.0108729 | 0.0108729 | 0.0162203 |
| EqErr(M) | 0.0625734 | 0.0700656 | 0.108276 | 0.057512 | 0.108276 |
| EqErr(S) | 0.0456588 | 0.0509019 | 0.0640675 | 0.0448072 | 0.0640675 |
| RMS(Z_S) | 4.76613 | 4.39368 | 3.9852 | 3.9852 | 4.76613 |
| RMS(Z_H) | 5.80145 | 5.43977 | 4.89985 | 4.89985 | 5.83127 |
| R_Z | 0.821541 | 0.807841 | 0.813332 | 0.759089 | 0.833364 |
| SemAgree | 0.856712 | 0.890554 | 0.94826 | 0.842838 | 0.94826 |

## 4. Parameter health

| Parameter | Grad nonzero | Mean grad norm | Absolute update | Relative update |
|---|---:|---:|---:|---:|
| `p_sem.weight` | True | 0.0762807 | 0.0238071 | 0.00148794 |
| `p_morph.weight` | True | 0.0595028 | 0.0164651 | 0.00102907 |
| `u_sem.weight` | True | 0.0831685 | 0.0236327 | 0.00147704 |
| `u_morph.weight` | True | 0.0506937 | 0.0170614 | 0.00106634 |

## 5. Gradient-budget gate answers

1. Semantic gradient budget: max=0.219207, mean=0.136138, p95=0.204784.
2. Morphology gradient budget: max=0.13127, mean=0.0921116, p95=0.124054.
3. Both semantic parameters active: True; morphology objective active: True.
4. Semantic response non-degenerate: True; end R_Z=0.813332, SemAgree=0.94826.
5. End reconstruction cosine: 0.99969238.
6. End S/M RMS ratio: 1.53118.
7. Next gate authorized: phase0_128b.

## 6. Morphology and reconstruction

- Morphology objective gradient active: True.
- EqErr(M) start/end: 0.0625734 / 0.108276.
- S/M RMS ratio end: 1.53118.
- CrossCov start/end: 0.0158448 / 0.0108729.

## 7. Safety and boundary

- All finite: True.
- Auxiliary semantic ic1 gradient-free: True.
- Original SSHR path updated ic1: True.
- Validation performance evaluated: false.
- Test/LUAD/segmentation GT accessed: false.
- Checkpoint saved for continuation: false.
- Phase 1 started: false.

The run stops at this gate. Even a Phase-0 GO requires separate human authorization for a 3-epoch pilot.

OSMF_V12_READINESS_PASS
