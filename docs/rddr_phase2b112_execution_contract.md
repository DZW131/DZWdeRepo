# Phase2B1.12 approved execution contract

User approved 2026-08-31. This file records interpretation BEFORE outcomes.

- Pure A0 base: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- C0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Three identical clones B (official loss), A (ADT), R (random gate), BCSS seed42, batch20 BF16, exactly500 steps. No test/LUAD/other seeds/full25.
- Fresh identical optimizer states: original model-only checkpoint has no momentum buffers. Original SGD momentum .0005, WD groups [.0005,0,.0005,0], poly exponent .9; reconstruct global_step=max_step=29275 and retain last-used LR [9.55328615544644e-7,1.910657231089288e-6,9.55328615544644e-6,1.910657231089288e-5]. Original PolyOptimizer retains these LRs beyond max_step. No restart or LR search.
- Main loss uses original train/freeze path. Auxiliary-only gradients may update all39 tensors in b4..b4_5/bn45, including BN affine, while all BN running statistics remain frozen. B leaves all BN frozen. A/R BN-affine changes are part of the treatment, not claimed absent; original group WD applies whenever an auxiliary gradient is supplied.
- q=JS/ln2 is need, not direction; Delta_sym and 15x15 exclude-self support stay frozen. Detach feat56, ic1, deep target, q/Delta/gates. No third evidence.
- Random gate matches A-arm current per-image active count, with independent seed42 RNG; R uses its own current predictions/q.
- Same transformed batches and same main-network RNG across arms. One32-batch train-only no-step calibration: lambda=.1*median(Gmain/(GADT+1e-8)); restore unchanged step0 state and RNG. No validation tuning.
- Snapshots0/50/100/250/500, primary500 only. Save only0/250/500 checkpoints (shared step0 storage allowed). 10k paired image bootstrap seed42 recomputes dataset-confusion-matrix metrics.
- Gate E margin: margin_A-margin_B >= -.05*abs(step0 mean margin).
- Gate G: conservative check of all4 classes; no post-hoc class exclusion.
- Gate H gradient ratio median: all500 A training steps, separately report R.
- Decision priority: provenance blocked; engineering H fail; A or B nonpass; semantic safety D/E/F/G fail; C fail; all pass GO. Weak-positive A is not PASS.
- No architecture changes or hyperparameter rescue. No automatic next phase.

## Task ledger

| Owner | Scope | Output | Status |
|---|---|---|---|
| Lead | immutable provenance, implementation, paired training and evaluation | runner/common, server evidence | in progress |
| Statistics worker | snapshot-only analysis and report renderer | analysis/report module | pending |
| Independent verifier | tests and adversarial code/artifact checks | verification findings | pending |

Final report: `docs/rddr_phase2b112_short_horizon_optimization_report.md`.
Original specification: `D:/work/RDDR_Phase2B1_12_Short_Horizon_ADT_Optimization_Dynamics_Audit_v1.0.md` (user-supplied; not an instruction to alter the approved contract).
