# RDDR Phase-2A Architecture Contract

## Provenance

- Pure official A0 base: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Dataset: BCSS-WSSS
- Development seed: 42
- Training: epoch 0 to 25, BF16, batch 20, FINAL checkpoint only
- Evaluation: BCSS validation only

Phase-2A does not inherit or instantiate the Phase-1 Dross Disposal Adapter.
The feature entering `HFRM28_1` is always the original `F`.

## Frozen score

For raw shallow and deep logits at 28x28,

```text
p_s = softmax(ic1(F28_raw))
p_d = softmax(fc8(Ddeep))
q   = detach(clip(JS(p_s, p_d) / ln(2), 0, 1))
```

The JSD uses natural logarithms, epsilon `1e-8`, and temperature `1.0`.

## Modes

```text
none:
  F' = F + gamma_sem * R_sem + gamma_ctx * R_ctx

global:
  r_bar = mean_H,W(1 - q)
  F' = F + gamma_sem * R_sem + gamma_ctx * r_bar * R_ctx

receiver:
  r_i = 1 - q_i
  F' = F + gamma_sem * R_sem + gamma_ctx * r_i * R_ctx
```

Only `HFRM28_1` receives a context scale. `HFRM56`, `HFRM28_2`, the semantic
veto branch, backbone, CAM heads, loss, optimizer, scheduler, augmentation,
inference fusion, thresholds, and metric remain unchanged.

## Capacity and gradient contract

- Additional trainable parameters: 0
- `q` is analytical and detached.
- No feature subtraction, projection, replacement, threshold, temperature,
  learned alpha, or new loss is present.
- `mode=none` preserves the original SSHR state-dict and forward result.
- GS and RCS use identical mean per-image reliability; only RCS retains
  spatial selectivity.

## Required run order

```bash
bash tools/run_rddr_phase2a_server.sh \
  /path/to/RDDR_PHASE2A_RUN \
  /path/to/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  /path/to/BCSS-WSSS/training \
  /path/to/python
```

The runner executes GS Full25 followed by RCS Full25 and never evaluates
validation, test, or LUAD during training.
