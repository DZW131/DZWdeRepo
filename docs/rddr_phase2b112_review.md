# Phase2B1.12 independent implementation review

Scope: approved execution contract and the supplied v1.0 specification; only the
Phase2B1.12 test file and this review are verifier-owned. Source, training,
optimizer, analysis, Git state, and server jobs are lead-owned.

## Outcome

Independent local run on 2026-08-31: **53 tests passed**, 0 failures, 0 errors,
0 skips; unittest reported 9.298 seconds. Environment: Windows Python 3.13.5,
NumPy 2.2.6, PyTorch 2.12.1+cpu, CUDA unavailable. The verifier did not run any
server command or allocate CUDA memory.

Command: `python -m unittest discover -s tests -p test_rddr_phase2b112.py -v`.

No unresolved implementation defect was found in the final reviewed code.
This is implementation-level verification, not a completed scientific audit.

## Findings and disposition

| Severity | Finding | Status |
| --- | --- | --- |
| P1 | A forward-pre-hook on `ResBlock` does not execute: the immutable block overrides `__call__`, bypassing PyTorch hook dispatch. | Closed. The lead changed capture to the first block's `bn_branch2a`, common module line 169. Explicit hook and end-to-end thin-fixture regressions pass. |
| P2 | The training loader hardcoded four workers, but initial preflight did not assert that setting and seed42 against the frozen launcher/status. | Closed. Runner lines 64 and 68 now require the launcher worker count and BCSS seed42 status. |
| P2 | Report manifest filtering omitted actual `calibration`/`training` keys and would inline approximately ten thousand image records. | Closed. Analyzer lines 771–780 summarize the actual arrays and link/hash the full manifest. |

## Checks performed

- Independent NumPy loss oracle; q normalized JS; strict-positive Delta gate;
  scalar 15x15/radius7/exclude-self support oracle at corners/interior; swap/tie
  semantics; detached target/q/gate/Delta; finite zero-gate loss and gradients.
- Original `PolyOptimizer` last-used LR reconstruction, 500 tail steps with no
  LR restart, and manual float64 SGD updates including group weight decay and
  the official momentum 0.0005; fresh identical three-arm optimizer states.
- Original residual block, BN, HFRM, and freeze implementations in a small-channel
  39-tensor topology. B gradients equal a separate main-objective backward;
  A changes no gradients outside approved tensors; main BN affine remains
  requires-grad false/grad None in B, while the auxiliary leaves supply affine
  gradients in A; feature input and ic1 are detached; no parameter or BN-buffer
  update occurs in gradient helpers. Bitwise local-forward parity is checked.
- Per-image random-gate count equality, independent seed42 stream and repeated
  draws, R using supplied A counts instead of its own gate counts, and identical
  main dropout outputs across arms under shared RNG.
- Executed extracted loader/augmentation-observer/data-access functions, plus
  AST checks of one fixed 32-batch calibration, frozen lambda, common transformed
  batch/RNG reuse, 1..500 loop, snapshots, checkpoint schedule, no early-stop
  branch, and no search/other-seed CLI.
- Read canonical `infer_fun.py` and `iouutils.py` against the complete snapshot
  observer. Tests compare original `scores()` with both observer and analyzer
  confusion metrics, including background and absent-class behavior. Static
  checks require the canonical inference call/default parameters and unchanged
  output object; representation selection is independently reproduced.
- Actual 10,000-replicate seed42 paired bootstrap compared with a separate
  direct-image-index NumPy oracle recomputing pooled confusion metrics. Tests
  cover frozen quintile ties, Gate E margin/strict-CI boundaries, all-four-class
  Gate G, all500-step Gate H ratio median, decision priority, weak-positive
  nonpass, and pending evidence never becoming GO.

## Evidence boundary

The independent tests use CPU formula/optimizer or small-channel fixtures built
from original residual-block, BN, HFRM, and train/freeze implementations. They do
not constitute a full original-model batch20/BF16/CUDA smoke test, exact C0
checkpoint identity, real data-stream replay, 32-batch calibration, 500-step
execution, official evaluation, bootstrap artifact verification, or an approval
to occupy a GPU. Those require separately recorded runtime evidence. In
particular, source/AST checks do not prove that real data were consumed in that
order, and synthetic identity does not prove that the actual C0 clones are
identical. The lead reported a server CPU provenance check, but the verifier did
not independently execute or inspect that server artifact in this subtask.

No calibrated lambda, real validation improvement, A–H pass, Full25 readiness,
or final scientific decision is asserted here. The full original-network
batch20/BF16/CUDA admission and runtime checks remain outstanding.

## Reviewed/tested file fingerprints

SHA256 immediately after the passing run (the lead may subsequently commit the
same files; any further content edit warrants rerunning relevant tests):

| File | SHA256 |
| --- | --- |
| `tools/rddr_phase2b112_common.py` | `640c41a9d21bd2666c2dbb9b1fdb40cdc895250ab41fc985d28a0fa71864611a` |
| `tools/run_rddr_phase2b112.py` | `afccf324db1e62d9e1c8599dec942efbef1f95632b3dd5dd05bcf16086225617` |
| `tools/rddr_phase2b112_evaluation.py` | `fba1009f287af5e50f365f434fea2bfcba71c4441798e3d0221c74aa171af66d` |
| `tools/analyze_rddr_phase2b112.py` | `77761b04e4a4494e993f483a05374a3d12c0400400e06c91ba4fb77eb40ad8e4` |
| `tests/test_rddr_phase2b112.py` | `1d5bb7f51d5ad9f2a58b4a371e5fff50aa6f96c50892b917c4f811792b09bfe0` |
