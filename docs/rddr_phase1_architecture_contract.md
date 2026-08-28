# RDDR Phase-1 Architecture Contract

This implementation is anchored to pure SSHR A0 commit
`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`. The model implementation used
for training is locked at
`4e08c9d228ee269f7754c5f6b78ca734cd165c61`.

Only the input to `hfrm_28_1` may change. `hfrm_56`, `hfrm_28_2`, the deep
branch, all four CAM heads, all four classification losses, optimizer,
schedule, inference fusion, thresholds, and metric remain the released A0
implementation.

For the raw 28×28 feature `F` and deep feature `Ddeep`, DD computes

```text
Ls = ic1(F)
Ld = fc8(Ddeep)
q  = detach(clip(JS(softmax(Ls), softmax(Ld)) / ln(2), 0, 1))
```

using natural logarithms, epsilon `1e-8`, and temperature `1.0`. The Dross
Disposal Adapter is

```text
Conv1x1(512 -> 128) -> GELU
-> depthwise Conv3x3(128 -> 128) -> GELU
-> Conv1x1(128 -> 512)
```

The final projection has exactly zero weight and bias at initialization.
Consequently, both variants are identity-preserving:

```text
UC: Fclean = F - DDA(F)
DD: Fclean = F - q * DDA(F)
```

UC and DD have identical adapter parameters and initialization. Both pass
`Fclean` into the unchanged original `hfrm_28_1`. Default mode `none` does
not instantiate the adapter and retains the original A0 state-dict and
forward behavior.

The adapter contributes 132,992 parameters and 103,663,616 convolution MACs
at 28×28. All adapter weights and biases enter the existing scratch optimizer
groups exactly once; there is no special learning rate, weight decay, warmup,
loss, threshold, exponent, temperature, or scalar strength.
