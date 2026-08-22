# TCRD-v0 Matched 5-Epoch Utility Gate

## 1. Executive conclusion

- Final route: **ROUTE_E_CLOSE**
- SPED: **SPED_UTILITY_NOGO**
- TCER: **TCER_UTILITY_REVIEW**
- Full TCRD: **TCRD_FULL_REVIEW**
- Primary comparisons use epoch5 candidate minus epoch5 matched C0; best epochs did not select the gate result.

## 2. Matched experimental control

- All four branches started from the same SHA-locked SSHR A0 seed42 epoch25 FINAL checkpoint.
- All reused the same 5×1171×20 schedule of indices, per-sample augmentation seeds and per-step model seeds.
- All original SSHR parameters remained trainable under the same derived epoch20/25 tail-replay PolyOptimizer schedule.
- Effective batch20, 224×224, BF16 and the official four classification weights were frozen.

## 3. Required executive table

| Branch | Step0 mIoU | Epoch5 mIoU | Δ vs C0 | CAM28_1 Δ | Decision |
|---|---:|---:|---:|---:|---|
| C0 Control | 67.3283 | 67.2312 | — | — | Reference |
| D SPED | 67.3262 | 67.2485 | +0.0174 | +0.0344 | SPED_UTILITY_NOGO |
| R TCER | 67.3683 | 67.2983 | +0.0672 | +0.1022 | TCER_UTILITY_REVIEW |
| DR Full | 67.3675 | 67.2880 | +0.0568 | +0.1035 | TCRD_FULL_REVIEW |

## 4. Epoch-by-epoch validation

| Point | C0 | D | R | DR |
|---|---:|---:|---:|---:|
| step0 | 67.3283 | 67.3262 | 67.3683 | 67.3675 |
| epoch1 | 67.0284 | 67.0283 | 67.0435 | 67.0580 |
| epoch2 | 67.0338 | 67.0487 | 67.1009 | 67.0894 |
| epoch3 | 67.2603 | 67.2540 | 67.2839 | 67.2533 |
| epoch4 | 67.0583 | 67.0471 | 67.0840 | 67.0800 |
| epoch5 | 67.2312 | 67.2485 | 67.2983 | 67.2880 |

## 5. Required mechanism table

| Branch | Update RMS/Z0 RMS | Main mechanism metric | Boundary Δ | Present-confusion Δ |
|---|---:|---:|---:|---:|
| D | 0.022147 | same/cross=1.006246 | -0.0359 pp | — |
| R | 0.010022 | ηR=0.102128 | — | +0.1427% |
| DR | D=0.022319; R=0.010039 | same/cross=1.006252 | -0.0229 pp | +0.1103% |

## 6. SPED finding

- Same/cross conductance: 0.13202364 / 0.13120420 (ratio 1.006246).
- Diffusion update RMS/Z0 RMS: 0.022147.
- B0 boundary recovered/harmed/net: 828/1024/-196.

## 7. TCER finding

- Reaction update RMS/Z0 RMS: 0.010022.
- Present-confusion wrong pixels C0/R: 23121359/23088357 (+0.1427%).
- Present entropy Z0→ZT: 0.157057→0.143586.
- Top1–top2 margin Z0→ZT: 0.866988→0.880491.
- REACTION_OVERCONFIDENCE_REVIEW: False.

## 8. Best epochs (observation only)

- C0: step0 mIoU=67.3283; not used by the gate.
- D: step0 mIoU=67.3262; not used by the gate.
- R: step0 mIoU=67.3683; not used by the gate.
- DR: step0 mIoU=67.3675; not used by the gate.

## 9. Provenance and artifacts

- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Validation pairs: 3,418; official TTA, predicted presence, hard gate, min-max, 0/0.6/0.2/0.2 fusion and released metric.
- Machine-readable metrics are under `comparison/`; branch histories and final checkpoints remain in their branch directories.

## 10. Interpretation limit

This mature-checkpoint continuation gate evaluates whether a mechanism can improve a mature SSHR representation. It does not establish fresh-25-epoch performance, multi-seed stability, LUAD generalization or a publication claim.

## 11. Stop boundary

No test, LUAD, other seed, fresh 25-epoch training, hierarchy expansion, T/eta/formula change or auxiliary loss was run.

**ROUTE_E_CLOSE**

STOP.
