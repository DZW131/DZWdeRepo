# RDDR Phase-2B1 delivery evidence

## What was implemented

- Dataset: no data edits; frozen 3418-image BCSS validation population only.
- Model: no model edits or learnable additions; pure official A0 plus read-only hook.
- Training: no training, optimizer, gradients, or model checkpoint writes.
- Evaluation: isolated dual-hypothesis support audit, fixed hard sign and soft
  anchor, exact AUROC/AP, frozen strata, paired image bootstrap and gate logic.
- Verification: independent NumPy/SciPy calculations, separate from audit helpers.
- Documentation: frozen contract, complete report, runnable README, evidence CSV/JSON.
- Visualization: report tables; no posthoc qualitative selection or figures.

## Validation evidence

| Check | Result / scope |
|---|---|
| Local Python compilation | All Phase-2B1 Python source compiled |
| Server unit tests | 21/21 PASS; saved `tests_final.txt` |
| Real-image BF16 smoke | 2 images PASS |
| Formal extraction | All 3418 validation images completed |
| Strict checkpoint load | No missing/unexpected keys |
| Model/checkpoint immutability | Before/after model digest equal; input checkpoint SHA unchanged |
| Frozen population replay | All cache SHAs and original per-image counts verified |
| Native conflict replay | Maximum q error exactly 0 |
| CUDA-to-CPU observation replay | wD/anchor/FixedAvg maximum errors exactly 0 |
| Exact ranks | All per-image AUROCs independently recomputed with SciPy rankdata |
| Pooled AUROC/AP | Independent exact ranks and descending tied-score AP matched |
| Proper scores/confusions | Native arrays independently reconstruct NLL/Brier and counts |
| Real support checks | 9 fixed positions explicitly enumerated; maximum error 1.1920929e-7 |
| Bootstrap | 32 independent complete replicates matched exactly; all 10,000 CI quantiles matched |
| Safety | All 15 preregistered Deep-Wrong aggregate strata independently checked |
| Nonlinear aggregation | Top20/Bottom80 and q-quintile native confusion matrices sum exactly to global |
| Decision | A PASS, B FAIL, C FAIL, D PASS; STRONG_SIGNAL false; NOGO |

No test/LUAD/other seeds were evaluated, and no Phase-2B2 run was launched.

## Results and interpretation

Image-balanced AUROC is 0.734850 [0.726086, 0.743701], but the pooled zero-sign
decision has BA 0.593973 and Deep-Win recall 0.261653. Soft anchor accuracy
improves +0.520502 percentage points over FixedAvg while mIoU changes
-0.443308 percentage points [ -0.829455, -0.037244 ]. Deep-Wrong safety passes.

This is **`RDDR_PHASE2B1_NOGO`**, not GO on the strength of AUROC or selected
subsets. The main report discloses class2/class3 reversed ranking, class3's
limited Shallow-Win population, and the distinction between pooled,
image-balanced, hard-decision, soft-fusion and subgroup-aggregation results.

## Source and artifact provenance

- Pure A0: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- Extraction version: `82e10afe85af1bda69a1f0e0f8de003110178d08`.
- Analysis version: `abc8ff28471aae94fda3a422932b7032f3b4ef9d`.
- Enhanced verifier/report version: `e3164c4`.
- Branch: `feature/rddr-phase2b1-dual-hypothesis`.
- PR target: `baseline/official-a0`; do not merge automatically.

| Artifact | SHA256 |
|---|---|
| C0 final checkpoint | `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579` |
| Native observations NPZ | `767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a` |
| Sufficient-statistics NPZ | `56eed74d93172456a1149a06691ce7eb37e2aec1cc2acf1366e0cc236fd1671e` |
| Reviewed final Markdown | `d007ce5ce736bd69e1599418e66f139cb4db8083d01aae3e29d79515ed38c126` |
| Enhanced independent-verification JSON | `568e24371a576950d11b117c4c36946a757d04dfb467e3fbbd9597e2c928af99` |

The canonical report in `docs/` and its artifact copy are byte-identical to
the server `delivery/` report. The checked-in verification JSON is the enhanced
version that explicitly verifies nonlinear partition mIoU. First drafts are
preserved on the server and in ignored local cache; no user baseline or prior
experiment was overwritten or removed.

## Final command index

See `docs/README_rddr_phase2b1.md` for exact path-aware commands.

- Training: prohibited for this phase.
- Forward: `bash tools/run_rddr_phase2b1_server.sh UNIQUE_RUN_NAME`.
- Offline statistics: `tools/analyze_rddr_phase2b1.py --input ... --output ...`.
- Independent verification: `tools/verify_rddr_phase2b1.py --report ... --native ...`.
- Report: `tools/report_rddr_phase2b1.py --report-dir ...`.
- Visualization: open the delivered Markdown tables; no extra rendering needed.

## Artifact locations

- Server checkout: `/home/duyanhong/DZWdeRepo-rddr-phase2b1`.
- Server experiment root: `/home/duyanhong/experiments/RDDR_PHASE2B1`.
- Native probabilities: `formal_r1/rddr_phase2b1_native_observations.npz`.
- Statistical artifacts: `report_r1/` under the experiment root.
- Reviewed outputs: `delivery/` under the experiment root.
- Input checkpoint: `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`.
- Local report: `G:/05_科研工作/SSHR/DZWdeRepo-rddr-phase2b1/docs/rddr_phase2b1_dual_hypothesis_context_adjudication_report.md`.
- Local/versioned statistics: `audit/results/rddr_phase2b1/` in this checkout.

## Remaining items / handoff

No additional extraction, statistical run, training, or evaluation is needed.
The independent PR is for human review only. Further experiment design requires
new user direction; do not optimize thresholds or weaken gates to rescue this
run. Large NPZ artifacts are intentionally excluded from Git and retained on
the server. If rerunning, choose new output names: tools refuse overwrites.

This delivery followed the research-project-implementation-delivery workflow:
isolated implementation, recorded server validation, reproducible commands,
complete report and explicit stop boundary.
