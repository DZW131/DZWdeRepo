# Frozen HQMR-v1 presence-gate formula (DLAG Oracle v1)

Sources: `network/gcqm_net.py` and
`tools/eval_gcqm_full25_bcss_seed42.py` on `audit/dlag-oracle-v1`.

1. The frozen ResNet38 deep head produces spatial logits
   `deep_cam_logits[B,4,H,W]`.
2. Image-level logits are global-average pooled:
   `deep_logits = adaptive_avg_pool2d(deep_cam_logits,1).flatten(1)`.
3. Gate scores are `deep_gate = sigmoid(deep_logits)`.
4. For each of the three TTA views (identity, horizontal flip, vertical
   flip), the model independently produces `deep_gate`. The final gate score
   is their arithmetic mean. Thus gate aggregation occurs **after TTA model
   forward**, independently of spatial CAM unflipping.
5. Frozen class thresholds are `[0.8, 0.9, 0.8, 0.6]` and the decision is
   `label[c] = 1[mean_deep_gate[c] > threshold[c]]`. If all labels are zero,
   the single class with maximal mean score is enabled.
6. Independently, each TTA Stage3 HQMR class mixture is resized to the input
   size, unflipped, and averaged. Each class map is then min-max normalized
   over its spatial extent.
7. The binary presence label is applied **after** TTA averaging and **after**
   per-class CAM min-max normalization. Absent classes receive zero score;
   the final prediction is the per-pixel argmax over the remaining classes.
   The sealed implementation uses zero rather than negative infinity, so an
   absent low-index class can still win an exact all-zero tie. The audit keeps
   this behavior unchanged; it is especially relevant when interpreting B2.

Oracle B changes only the binary label after steps 1–6. It does not change
the deep logits, gate scores, backbone features, query logits, class maps,
TTA averaging, or CAM normalization. B1 enables only true classes belonging
to frozen M1 components in that image whose baseline gate excluded them.
B2 replaces the label with exact image-level GT presence and is secondary.
