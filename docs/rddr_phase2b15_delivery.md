# Phase-2B1.5 delivery evidence

## What was implemented

- Dataset changes: none; immutable previous validation probability cache reused.
- Model changes: none; official A0 network/training/inference/metrics unchanged.
- Training engineering: none; no optimizer, backward, checkpoint loading or writes.
- Audit: four-way supports, source reversal, symmetric adjudication/anchor,
  class/pair/confidence/mass/GT decomposition and fixed third-evidence probes.
- Statistics: 45 groups, 12 ordered prediction pairs, exact AUROC/AP, pooled
  confusion metrics, all requested CSV/JSON, 10,000 paired image bootstrap.
- Verification: independent NumPy/SciPy script with no shared audit helper imports.
- Documentation: frozen contract, 30-section report, README and this evidence note.
- Visualization: complete report tables; no posthoc image/pair/source selection.

## Validation evidence

| Check | Result |
|---|---|
| Local compilation | All new Python files compile |
| Server tests | 23/23 PASS, complete log delivered |
| Smoke | Two real cached images PASS |
| Formal coverage | All 3418 validation images |
| Old SS/SD/Delta/ctx parity | All maximum absolute differences exactly 0 |
| Frozen masks and counts | All checked per image, unchanged |
| Input immutability | Cache/checkpoint SHA before and after match |
| Independent group counts | All 45 groups exact |
| Independent native confusions | All groups and eight estimators exact |
| Independent ranks | 116,212 image-score combinations checked with SciPy rankdata |
| Pooled ranks / AP | Exact independent tied-score calculations agree |
| Support distribution statistics | Means/std/quantiles checked independently |
| Explicit neighbor checks | Nine fixed real positions; support error <=9.012e-8 |
| Context / GT composition errors | <=6.232e-8 / <=2.906e-8 |
| Rescue/harm identities | Both exact identities verified |
| Independent bootstrap | 32 replicates x159 columns; max difference 1.333e-15 |
| CI quantiles | All 10,000-replicate percentile CIs checked |
| Gates and flags | Independently reconstructed; identical |

Extraction/probes including compression: 18.34 seconds; statistics/bootstrap:
34.12 seconds. Peak CUDA allocated 15.57 MiB, reserved 26.00 MiB. New derived
observations are 287.62 MiB. No additional training or evaluation is pending.

## Result

`SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED`.
A/B/D PASS; C UNDERPOWERED. Both strong-signal flags are true.

Symmetric image AUROC 0.784842 [0.777130, 0.792815]; BA 0.715627;
Deep-Win/Shallow-Win recalls 0.640314/0.790939. Mean-bias shrinkage 74.56%.
Symmetric anchor minus FixedAvg native-grid mIoU +1.965066 percentage points.
ctx_sym Both-Wrong accuracy/rescue 0.336057 [0.323216, 0.348766];
one-correct third-class harm 0.045261 [0.042627, 0.047964].

Class2's image AUROC 0.451437 only narrowly clears the preregistered 0.45 point
threshold, with its CI entirely below0.5. Class3's image AUROC 0.340851 remains
low and is UNDERPOWERED because Shallow-Win count is418. No claim of fully
resolved class inversion, official mIoU improvement, or readiness to train.

## Provenance and hashes

- A0: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- Probe run: `ec7abb6a2d889b6dad7f20b4539806395e93d37b`.
- Statistical run: `a09b51de5eef82973f908dd68fd0ef84cb933b6a`.
- Independent verification: `0a53acd`.
- Feature branch: `feature/rddr-phase2b15-bias-decomposition`.
- PR target: `baseline/official-a0`; human review only, no automatic merge.

| Artifact | SHA256 |
|---|---|
| C0 input checkpoint | `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579` |
| Frozen Phase-2B1 native cache | `767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a` |
| New derived observations | `237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514` |
| Sufficient statistics | `acf2f821b7d161a3166d4bd9885b617d4dfc3d316cbdf95723e5680d673ffdd9` |
| Canonical UTF-8/LF Markdown report | `35a0d75746c9ebf996c1ce87322a6699541640f822ebcb12a0159e530a079213` |

Report formatting was performed locally from verified server CSV/JSON; all
scientific computation used the recorded server environment. The renderer
explicitly emits LF for deterministic report bytes across Windows/Linux.

## Final command index

See `docs/README_rddr_phase2b15.md` for full path-aware commands.

- Training: prohibited for this phase.
- Network inference: unnecessary; frozen cache is reused.
- Probability probes: `tools/run_rddr_phase2b15_bias_decomposition_audit.py`.
- Statistics: `tools/analyze_rddr_phase2b15.py`.
- Independent checks: `tools/verify_rddr_phase2b15.py`.
- Markdown: `tools/report_rddr_phase2b15.py`.
- Visualization: open the complete report tables.

## Artifact locations and handoff

- Server checkout: `/home/duyanhong/DZWdeRepo-rddr-phase2b15`.
- Experiment root: `/home/duyanhong/experiments/RDDR_PHASE2B15`.
- Probe output: `formal_r1/`; full statistics: `report_r1/`; reviewed report: `delivery/`.
- Local report: `G:/05_科研工作/SSHR/DZWdeRepo-rddr-phase2b15/docs/rddr_phase2b15_adjudication_bias_decomposition_report.md`.
- Versioned statistics: `audit/results/rddr_phase2b15/` in this checkout.
- Large NPZ files: retained on the server and ignored by Git.

No old experiment or baseline was deleted. The first local CRLF rendering is
preserved in ignored cache; the canonical report uses LF. The implementation
delivery skill guided isolation, evidence recording, reproducible commands and
the explicit stop boundary. All required work is complete; next model design,
training or test evaluation requires new user direction.
