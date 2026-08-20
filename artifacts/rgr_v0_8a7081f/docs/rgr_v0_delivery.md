# RGR-v0 Minimal Region Graph Reasoning Delivery

## 1. Executive conclusion

Final decision: `RGR_V0_PILOT_NOGO`

The run followed the gated sequence: zero-init parity, fresh 32-batch
readiness, and only after readiness PASS a fresh three-epoch frozen-SSHR pilot.
No RSBR trained weight, dense label, test split, LUAD split, other seed, or
25-epoch continuation was used.

## 2. Frozen implementation

- Base commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Experiment commit: `8a7081f5fb3ac1ffb4037e6e60d0716e6363e2c0`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Seed / batch / image / precision: 42 / 20 / 224 / BF16
- Loss coefficients: region=0.05, residual=0.01
- RGR parameters: 101225
- Parameter overhead: 0.089810%
- Source hashes: `{"network/resnet38_cls_rgr.py": "4ea34afa1457de8d0df9f4decf61677d31f53bec789fa86493c8157fa7713abd", "network/rgr_v0.py": "0568042112201ecfd603a473342794caf35f57c83688d81ea0ad98db25821da6", "tool/infer_rgr_v0_paired.py": "9c8ff4bb55bd41457aaa6756cc00850db1abbc6196fb9478dc34439263f01c96", "tools/run_rgr_v0.py": "eb28b3023f83a6d1e8609d738782f5ddb633fe61ac5a99ad688537cb0167f4fb"}`

## 3. Stage -1 parity

- Decision: `RGR_V0_PARITY_PASS`
- Same-process exact: True
- Independent mIoU delta: +0.00000000 pp
- Independent differing pixels: 0
- Corrected envelope pass: True

## 4. Stage 0 readiness

- Decision: `RGR_V0_READINESS_PASS`
- Failures: []
- Step-1 isolated/graph heads active: True
- Upstream active by step 8: {'node_projection': True, 'edge_gate': True, 'value_projection': True, 'message_projection': True}
- Upstream active by step 32: {'node_projection': True, 'edge_gate': True, 'value_projection': True, 'message_projection': True}
- Final residual ratio: 0.01301461
- Graph/isolated residual RMS: 0.17319162
- Frozen SSHR unchanged: True
- All finite: True

## 5. Three-epoch paired validation

| Epoch | Base | Isolated | Graph-only | Full | Full-Base | Full-Isolated |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 67.3175 | same | same | same | 0 | 0 |
| 1 | 67.3175 | 67.3711 | 67.2782 | 67.3268 | +0.0094 | -0.0443 |
| 2 | 67.3175 | 67.3851 | 67.2130 | 67.3190 | +0.0016 | -0.0660 |
| 3 | 67.3175 | 67.3874 | 67.1968 | 67.3143 | -0.0032 | -0.0731 |


## 6. Per-class changes

| Epoch | Variant | C0 delta | C1 delta | C2 delta | C3 delta |
|---:|---|---:|---:|---:|---:|
| 1 | Isolated | +0.0191 | +0.0458 | +0.1079 | +0.0418 |
| 1 | Full | -0.0123 | -0.0260 | +0.0985 | -0.0227 |
| 2 | Isolated | +0.0268 | +0.0571 | +0.1426 | +0.0439 |
| 2 | Full | -0.0224 | -0.0373 | +0.0904 | -0.0245 |
| 3 | Isolated | +0.0281 | +0.0587 | +0.1475 | +0.0453 |
| 3 | Full | -0.0237 | -0.0358 | +0.0712 | -0.0243 |


## 7. Graph diagnostics

| Epoch | Touch gate | Non-touch gate | Same-class gate | Diff-class gate | Graph RMS |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.213529 | 0.175589 | 0.171125 | 0.206317 | 0.491575 |
| 2 | 0.214090 | 0.176098 | 0.171616 | 0.206865 | 0.577849 |
| 3 | 0.214303 | 0.176254 | 0.171757 | 0.207077 | 0.585926 |


| Epoch | Type-B recovery | Type-D recovery | Graph/isolated RMS |
|---:|---:|---:|---:|
| 1 | +44,538 | +28,622 | 0.485997 |
| 2 | +55,653 | +36,711 | 0.509789 |
| 3 | +50,804 | +35,082 | 0.510815 |


## 8. Node-count stratification at epoch 3

| Region count | Images | Full-Base | Full-Isolated |
|---|---:|---:|---:|
| N=1 | 1865 | -0.0001 | +0.0004 |
| N=2 | 286 | -0.0529 | -0.0900 |
| N=3-4 | 678 | -0.0467 | -0.1272 |
| N>=5 | 589 | +0.0101 | -0.2112 |
| N>=2_all | 1553 | -0.0227 | -0.1347 |

The node-count bin is assigned from the unflipped TTA view. A nominal `N=1`
image can have more than one proposal in another TTA view, so its tiny nonzero
paired delta is a binning/TTA effect; the module-level `N=1` graph message and
graph residual remain exactly zero by construction and unit test. `N>=2_all`
is the primary relational diagnostic.


## 9. Training and resources

| Epoch | Train seconds | Validation seconds | Mean loss | Isolated grad | Graph grad |
|---:|---:|---:|---:|---:|---:|
| 1 | 96.85 | 78.06 | 0.175974 | 5.415212e-02 | 1.648942e-02 |
| 2 | 95.89 | 78.49 | 0.173742 | 4.923062e-02 | 1.938645e-02 |
| 3 | 95.99 | 78.22 | 0.173228 | 4.769144e-02 | 1.924282e-02 |


- Pilot peak CUDA memory: 1.079 GiB
- Parameter overhead below 1%: True
- Runtime review: True
- Epoch-3 mean region extraction: 0.5479 ms/image/view
- Epoch-3 mean graph construction: 0.4485 ms/image/view
- Epoch-3 mean message passing: 0.2980 ms/image/view

## 10. Required scientific answers

1. Full improves A0: True.
2. Full improves isolated correction: False.
3. Graph-only positive at any epoch: False.
4. Maximum graph increment: -0.0443 pp.
5. Multi-node relational signal: False.
6. Edge gates are reported descriptively above; no gate statistic was tuned.
7. Per-class beneficiaries are reported above.
8. Type-B recovery at epoch 3: 50804 pixels.
9. Graph context reduces isolated errors: False.
10. Parameter/runtime overhead are reported above.
11. A 25-epoch experiment is not automatically authorized; scientific review is required.
12. Transition-aware graph edges are not authorized by this experiment.

## 11. Commands and artifacts

```bash
tools/run_rgr_v0.py --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --rsbr-pilot-summary /home/duyanhong/experiments/RSBR_V0_PILOT_3EP_a266d01/summary.json --output-dir /home/duyanhong/experiments/RGR_V0_8a7081f --experiment-commit 8a7081f5fb3ac1ffb4037e6e60d0716e6363e2c0 --num-workers 4
```

- Parity: `parity/summary.json`
- Readiness: `readiness_32b/summary.json`
- Pilot: `pilot_3ep/summary.json`
- Checkpoints: `checkpoints/epoch1_rgr.pth` through `epoch3_rgr.pth`
- Checkpoint SHA256: `{"epoch1_rgr.pth": "bf55ba192b4320747f3b783a67e28c16c58460f635417108956689cff180c843", "epoch2_rgr.pth": "8d79fe7a85a541b587e43e86d015a5b62332be853a9f3f393fbe2f44f53e7bff", "epoch3_rgr.pth": "a74eaeb06adf19e887f3d18311614331a4511ac535006564882fd26211457b58"}`

## 12. STOP boundary

Execution stops after this report regardless of decision. No test, LUAD,
additional seed, 25-epoch run, deeper GNN, transition head, GAT, Transformer,
prototype, topology change, edge-feature change, or tuning is performed.
