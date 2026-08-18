# OSMF-v1.2 Phase0 Audit

## 1. Decision

**OSMF_V12_PHASE0_REVIEW**

Processed real BCSS batches: 128/128.
Decision reasons: `['MORPHOLOGY_EQUIVARIANCE_NO_FAVORABLE_TREND']`.
Flags: `['GENUINE_DECORRELATION_SIGNAL']`.

## 2. Frozen contract

- Audit commit: `92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Parity proof SHA256: `e2585306d7f43be4842ea39168ff2e5c7f2d64cf86e6332c1e11d8eccfef0dcf`
- BCSS train only; seed 20260817; batch 20; 224x224; BF16.
- Fixed objective weights: 0.05/0.05/0.05/0.10.
- Fresh A0 checkpoint and optimizer state; no continuation from another audit.
- Exact command: `tools/audit_osmf_v12_gradient_gate.py --gate phase0 --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --parity-summary /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/parity/summary.json --readiness-summary /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/readiness_8b/summary.json --output-dir /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/phase0_128b --audit-commit 92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4 --num-workers 4`

## 3. Main mechanism table

| Metric | Start | Mean | End | Min | Max |
|---|---:|---:|---:|---:|---:|
| L_total | 0.17754 | 0.244589 | 0.228737 | 0.107359 | 0.68014 |
| L_SSHR | 0.169584 | 0.240269 | 0.223103 | 0.104699 | 0.676317 |
| L_sem_pres | 0.143288 | 0.0508827 | 0.0119578 | 0.00897086 | 0.222073 |
| L_eq | 0.0625734 | 0.0810696 | 0.0836677 | 0.0490376 | 0.106756 |
| L_orth | 0.0158448 | 0.0115838 | 0.0125472 | 0.00778851 | 0.0166394 |
| L_rec | 0 | 0.00184057 | 0.00225449 | 0 | 0.00296688 |
| r_sem_pres | 0.219207 | 0.162896 | 0.0907168 | 0.0907168 | 0.277304 |
| cos(base,sem_pres) | 0.116062 | 0.0720311 | 0.0331646 | -0.0546452 | 0.192572 |
| r_eq | 0.0831609 | 0.107691 | 0.106941 | 0.0725845 | 0.132036 |
| cos(base,eq) | 0.00561597 | 0.00554026 | -0.000911872 | -0.000911872 | 0.0137368 |
| r_orth | 0.053213 | 0.0522638 | 0.0421884 | 0.0343702 | 0.0783389 |
| cos(base,orth) | -0.0152036 | -0.00866134 | -0.0107562 | -0.0262549 | 0.00139324 |
| r_rec | 2.11865e-08 | 0.0119495 | 0.0196349 | 2.11865e-08 | 0.0232821 |
| cos(base,rec) | 0 | 0.0403547 | 0.164764 | -0.271826 | 0.431624 |
| RMS(H) | 0.81863 | 0.68365 | 0.67314 | 0.554687 | 0.81863 |
| RMS(S) | 0.919284 | 0.761102 | 0.76444 | 0.538112 | 0.919284 |
| RMS(M) | 0.703722 | 0.58986 | 0.563898 | 0.457443 | 0.705636 |
| RMS(H_hat) | 0.81863 | 0.681281 | 0.672084 | 0.554333 | 0.81863 |
| RMS(S)/RMS(M) | 1.30632 | 1.29861 | 1.35563 | 0.946972 | 1.52899 |
| Cos(H,H_hat) | 1 | 0.998976 | 0.998094 | 0.997527 | 1 |
| ResidualRatio | 0 | 0.0366549 | 0.0612173 | 0 | 0.0692717 |
| CrossCov(S,M) | 0.0158448 | 0.012745 | 0.0129976 | 0.00959711 | 0.0162261 |
| EqErr(M) | 0.0625734 | 0.078236 | 0.0811646 | 0.0572567 | 0.108125 |
| EqErr(S) | 0.0456588 | 0.0619012 | 0.0654105 | 0.0448022 | 0.0915611 |
| RMS(Z_S) | 4.76613 | 3.95073 | 4.27204 | 2.49318 | 4.76613 |
| RMS(Z_H) | 5.80145 | 5.03757 | 5.54779 | 3.75726 | 5.83149 |
| R_Z | 0.821541 | 0.780268 | 0.770044 | 0.663564 | 0.833363 |
| SemAgree | 0.856712 | 0.915084 | 0.986955 | 0.842552 | 0.986955 |

## 4. Parameter health

| Parameter | Grad nonzero | Mean grad norm | Absolute update | Relative update |
|---|---:|---:|---:|---:|
| `p_sem.weight` | True | 0.0624583 | 0.172839 | 0.0108024 |
| `p_morph.weight` | True | 0.0478357 | 0.114626 | 0.0071641 |
| `u_sem.weight` | True | 0.0649526 | 0.145084 | 0.00906775 |
| `u_morph.weight` | True | 0.0426396 | 0.115653 | 0.00722833 |

## 5. Gradient-budget gate answers

1. Semantic gradient budget: max=0.277304, mean=0.162896, p95=0.25495.
2. Morphology gradient budget: max=0.132036, mean=0.107691, p95=0.131882.
3. Both semantic parameters active: True; morphology objective active: True.
4. Semantic response non-degenerate: True; end R_Z=0.770044, SemAgree=0.986955.
5. End reconstruction cosine: 0.99809361.
6. End S/M RMS ratio: 1.35563.
7. Next gate authorized: none.

## 6. Morphology and reconstruction

- Morphology objective gradient active: True.
- EqErr(M) start/end: 0.0625734 / 0.0811646.
- S/M RMS ratio end: 1.35563.
- CrossCov start/end: 0.0158448 / 0.0129976.

## 7. Safety and boundary

- All finite: True.
- Auxiliary semantic ic1 gradient-free: True.
- Original SSHR path updated ic1: True.
- Validation performance evaluated: false.
- Test/LUAD/segmentation GT accessed: false.
- Checkpoint saved for continuation: false.
- Phase 1 started: false.

The run stops at this gate. Even a Phase-0 GO requires separate human authorization for a 3-epoch pilot.

OSMF_V12_PHASE0_REVIEW
