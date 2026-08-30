# Phase-2B1.11 Delivery Summary

## What Was Implemented

- Dataset: unchanged. Read frozen caches of all3418 validation images; GT used only for evaluation, never candidate/score construction.
- Model: unchanged pure A0; no innovation network imports.
- Training: none. No model forward, backward, autograd, optimizer, step, BN update or checkpoint write.
- Audit: prescribed A_alt/M_alt, three ranking tasks, five frozen controls, full candidate composition, hard effects, analytic epsilon-KL gradient, protection, strata and 10k paired image bootstrap.
- Verification: independent context box filter, softmax Jacobian, fixed128 real-candidate finite differences, tied ROC/AP, direct-gather bootstrap and gates; 54 tests.
- Visualization: quantitative tables in the full33-section Markdown report.
- Documentation: approved contract, runnable README, deterministic renderer, SHA manifest and full report.

## Validation Evidence

- Main run/verifier commit: `57c1c2da9541abd9f70105ebe4902ddf6d7643a3`.
- 54 tests passed, zero skips; 29 independent checks passed.
- FP32 support/context and raw-probability replay maximum error0; q rounding error5.96046448e-8, stored q retained.
- Independent FP64 context error1.56986881e-7; explicit Jacobian vs closed form4.44089210e-16; real128 finite difference1.06474914e-7.
- All candidate gradient label signs identical across independent algebra; no epsilon-based relabel.
- Bootstrap replicate maximum discrepancy2.84217094e-14.
- Input/checkpoint hashes and original A0 sources unchanged. Previous model/BN/prediction identity remains inherited, not newly rerun.
- Main run15.9748 seconds. GPU use was probability arithmetic only; process peak RSS2545848KiB.

## Scientific Result

Gate A/B/C/D/E/F = FAIL/FAIL/PASS/PASS/PASS/FAIL.

Candidate precision53.5534% is insufficient despite108,541 rescues (3.4715× the31,266 reference gap). M_alt rescue imageAUC0.6249 also fails; Both-Wrong detection and average local gradient utility pass. Stable-Correct intrusion56,477 exposes a major contamination source. Full-denominator protection passes but does not prove safe active candidates or class/boundary safety.

`DECISION = THIRD_EVIDENCE_OPERATIONAL_HEADROOM_INSUFFICIENT`

The fixed decision name describes failure of Gate A's combined operational conditions, specifically precision here; it does not mean raw rescue-count headroom is insufficient.

## Remaining Items

- No required computation remains.
- PR review remains, with no automatic merge.
- Deferred/prohibited: threshold/gate design, another teacher, score fusion, class/boundary/Top20 exclusion rules, training, Full25, test/LUAD/otherseed.

## Command Index

- Training / network inference: not applicable, prohibited.
- Audit, verifier, tests: [README](README_rddr_phase2b111.md).
- Report generation / visualization: deterministic Markdown renderer; see README.

## Artifact Locations

- Local: `G:/05_科研工作/SSHR/DZWdeRepo-rddr-phase2b111`.
- Server code: `/home/duyanhong/DZWdeRepo-rddr-phase2b111`.
- Completed run: `/home/duyanhong/experiments/RDDR_PHASE2B111/formal_r1`.
- Logs: `/home/duyanhong/experiments/RDDR_PHASE2B111/formal_r1.log`, `verify_r1.log`.
- Small evidence: `audit/results/rddr_phase2b111/`; all bootstrap replicates retained.
- Checkpoint/input caches: original paths and SHA values in report §1; no deletion or replacement.
- Full report: [rddr_phase2b111_neither_hierarchy_third_evidence_report.md](rddr_phase2b111_neither_hierarchy_third_evidence_report.md).

## Handoff Notes

Use new directories for a deliberate replay. Do not modify input hashes or previous results. This is a native28 frozen-mechanism audit, not a new segmentation evaluation or training experiment. Local margin utility, conditional Both-Wrong accuracy, deployable candidate precision and full-model performance must remain distinct.
