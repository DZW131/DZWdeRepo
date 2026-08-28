# RDDR Phase-0 Tensor Contract

This contract is audited against pure A0 commit `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`. The diagnostic runner repeats the frozen `Net.forward` operations without changing `network/resnet38_cls.py`; a unit test requires the default and diagnostic output tuples to be numerically identical in evaluation mode.

| Audit name | A0 source | 224-input shape | Spatial / channels | Position | Classifier |
|---|---|---:|---|---|---|
| `F28_raw` | `network/resnet38_cls.py:100`, `feat_28_1` | `[B,512,28,28]` | 28×28 / 512 | HFRM28_1 input, before rectification | diagnostic `ic1(F28_raw)` |
| `F28_rect` | `network/resnet38_cls.py:110`, `feat_28_1_rectified` | `[B,512,28,28]` | 28×28 / 512 | HFRM28_1 output, after rectification | existing `ic1` |
| `Ddeep` | `network/resnet38_cls.py:106`, `feat_deep` | `[B,4096,28,28]` | 28×28 / 4096 | deepest semantic feature, before dropout | existing `fc8` |
| `CAM28_raw` | diagnostic-only `ic1(F28_raw)` using existing layer declared at `network/resnet38_cls.py:75` | `[B,4,28,28]` | 28×28 / 4 | pre-HFRM class logits; not used by SSHR inference | `ic1` |
| `CAM28_rect` | `network/resnet38_cls.py:115` | `[B,4,28,28]` | 28×28 / 4 | normal post-HFRM class logits; official CAM applies ReLU | `ic1` |
| `CAMdeep` | `network/resnet38_cls.py:119` (training) and `Net_CAM.forward_cam` (inference) | `[B,4,28,28]` | 28×28 / 4 | deep class logits; official CAM applies ReLU | `fc8` |

Primary `p_s` and `p_d` are softmaxes of the pre-threshold, pre-ReLU class logits `ic1(F28_raw)` and `fc8(Ddeep)`, upsampled bilinearly to the 224×224 mask with `align_corners=False`. The BCSS classifier has four foreground classes. Ground-truth label 4 is the evaluator-overwritten background and is excluded from the foreground error-detection population; any other out-of-range label is recorded as excluded/ignore.
