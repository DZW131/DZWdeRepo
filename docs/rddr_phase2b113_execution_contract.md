# RDDR Phase-2B1.13 Execution Contract

## Scope

This branch implements the frozen Context-Specific Parameter-Gradient Attribution Audit. It is a zero-step diagnostic: it does not train, tune, select a checkpoint, write a checkpoint, or access BCSS test/LUAD data.

The scientific question is whether the contextual ADT gate remains distinguishable from the rate-matched random gate after aggregation through the shared `b4..bn45` Jacobian, and whether the contextual parameter gradient is more aligned with a GT-only raw-shallow semantic oracle.

## Frozen provenance

- Pure A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- C0 SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Phase-2B1.12 lambda: `0.027074256246554088`
- Training-stream population: Phase-2B1.12 manifest step 1–128, exact transformed-tensor SHA replay
- Validation population: all 3418 BCSS validation images
- Bootstrap: 10,000 paired minibatch resamples, seed 42
- Approved space: exactly 39 parameters / 27,275,776 scalars in `b4`, `b4_1`, `b4_2`, `b4_3`, `b4_4`, `b4_5`, and `bn45`

Historical Phase-2B1.9–1.12 decisions are immutable and are copied into provenance artifacts and the final report.

## Exact semantic definitions

Track T recomputes the unchanged Phase-2B1.12 ADT and seed42 rate-matched random gate on each frozen training minibatch. Every primary gradient starts from C0. The formal optimizer is constructed only to lock group provenance; it is never stepped.

The virtual optimizer transform reproduces the exact effective FP32 displacement of a fresh-state PyTorch SGD update, including weight decay, group LR, `grad is None` skipping, and parameter-add rounding. Numerical parity is checked on isolated cloned tensors.

Track V uses the immutable Phase-2B1.12 step0 `q`, `Delta>0` gate, deep probability, populations, and native28 truth. The gate and random control are GT-blind. GT classes 0–3 are used only for the raw-shallow oracle cross-entropy and post-hoc population attribution; background 4 and ignore 255 are excluded from the oracle.

BCSS validation filenames do not contain the training split's weak image-level labels. Track V therefore does not fabricate a main classification target from segmentation GT. Its `u_A/u_R` endpoint is the exact fresh-SGD transform of `lambda * g_ctx/g_rnd` plus the frozen per-group weight-decay behavior. Track T remains the primary endpoint for interaction with the real official main loss.

Population decomposition uses the unnormalized auxiliary numerator on the exhaustive foreground partition `Deep-Win_0`, `Shallow-Win_0`, `Both-Wrong_0`, and `Stable-Correct_0`. Direct foreground-numerator gradients independently verify the group-sum identity.

## Command

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b113
nohup bash tools/execute_rddr_phase2b113.sh \
  /home/duyanhong/experiments/RDDR_PHASE2B113/formal_4090_r1 \
  > /home/duyanhong/experiments/RDDR_PHASE2B113/formal_4090_r1.log 2>&1 &
```

The wrapper runs 38 CPU/control tests, the raw CUDA audit, independent verification, report generation, and post-analysis verification in that order.

## Prohibitions enforced by code/tests

- no formal `optimizer.step()` or parameter update
- no checkpoint write
- no lambda/LR/gate/window/seed sweep
- no new loss, head, backbone, or third evidence
- no test/LUAD access
- no mutation of `network/`, `tool/`, or `train_sshr.py`

## Artifacts

Raw CSV/JSON artifacts are written beneath the supplied output directory. The final report is written to:

`docs/rddr_phase2b113_parameter_gradient_attribution_report.md`

The final report must end with exactly one preregistered `DIAGNOSIS = ...` line.
