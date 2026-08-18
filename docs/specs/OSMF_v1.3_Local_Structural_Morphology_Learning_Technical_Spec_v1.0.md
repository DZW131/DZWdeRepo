# OSMF-v1.3：Local Structural Morphology Learning
## 局部结构形态专门化执行方案 v1.0

**项目：** 基于 SSHR 的 Objective-Induced Semantic–Morphology Factorization（OSMF）  
**当前新版本：** OSMF-v1.3  
**前置版本：** OSMF-v1.2 Conservative Gradient-Budgeted Specialization  
**前置结论：** `MORPH_EQ_OBJECTIVE_INVALID`  
**本版本性质：** 新版本重新预注册；不是对 v1.2 的续跑、加训或调参  
**核心变化：**

\[
\boxed{
L_{eq}^{pointwise}
\;\longrightarrow\;
L_{struct}^{local\ affinity}
}
\]

即：

> 不再要求同一位置的 morphology feature vector 在变换前后保持相同，而是要求局部邻域关系结构在变换前后保持一致。

**本版本最多授权：**

1. exact parity；
2. 8-batch structural-gradient readiness；
3. 若 PASS，再执行 fresh 128-batch Phase-0S structural/causal audit；
4. 无论结果如何均 STOP；
5. 不自动进入 3-epoch pilot。

---

# 1. v1.2 / Phase-0M 已确认的科学事实

OSMF-v1.2 已经证明：

## 1.1 Gradient budget 已健康

\[
Mean(r_{sem})=0.162896
\]

\[
Mean(r_{morph})=0.107691
\]

且：

\[
P95(r_{sem})=0.254950
\]

\[
P95(r_{morph})=0.131882.
\]

说明 auxiliary objectives 已经是：

\[
\boxed{
\text{auxiliary but non-dominant}
}
\]

而不是主导 SSHR。

---

## 1.2 Semantic preservation 已表现健康

\[
SemAgree:
0.856712\rightarrow0.986955
\]

semantic response 无 collapse。

因此：

\[
\boxed{
\text{pretrained-semantic preservation path should be retained.}
}
\]

---

## 1.3 Factorization / reconstruction 已健康

\[
Cos(H,\hat H):
1.000000\rightarrow0.998094
\]

\[
RMS(S)/RMS(M):
1.306320\rightarrow1.355634
\]

无 branch collapse。

---

## 1.4 Cross-subspace redundancy 有下降信号

\[
CrossCov:
0.015845\rightarrow0.012998.
\]

且不是由某一路 collapse 导致。

---

## 1.5 Pointwise morphology equivariance 已被因果否定

Phase-0M 在完全冻结 v1.2 的条件下得到：

### Same-pair causal test

32 个 morphology-active steps：

```text
improved = 16
harmed   = 16
neutral  = 0
improved_fraction = 0.50
mean causal delta = +0.00004601
```

因此：

\[
\boxed{
\text{pointwise }L_{eq}\text{ has no net causal benefit.}
}
\]

### Fixed probe

Raw EqErr(M)：

\[
0.063379\rightarrow0.069951
\]

AffinityEqErr(M)：

\[
0.010489\rightarrow0.012005.
\]

二者均没有改善。

### Gradient conflict

\[
\cos(g_{eq},g_{SSHR})=-0.005286.
\]

不存在足以解释失败的强主任务冲突。

因此正式结论：

```text
MORPH_EQ_OBJECTIVE_INVALID
```

当前 pointwise feature-equivariance objective 必须停止。

---

# 2. v1.3 的核心科学假设

v1.2 假设：

\[
M_a(p)\approx M_b(Tp)
\]

即 morphology role 由 **absolute pointwise feature identity** 定义。

Phase-0M 已表明这一假设不成立。

v1.3 改为：

\[
\boxed{
\mathcal G(M_a,p)
\approx
\mathcal G(T^{-1}M_b,p)
}
\]

其中：

\[
\mathcal G
\]

不是 feature 本身，而是该位置与局部邻域之间的关系结构。

本版本提出：

\[
\boxed{
\text{Morphology should be encoded as local relational geometry,
not pointwise feature identity.}
}
\]

---

# 3. v1.3 与 v1.2 的唯一核心变化

## 保留不变

以下全部继承 OSMF-v1.2：

```text
factorization point = post-HFRM H28_1
channel split = 256 / 256

P_sem
P_morph
U_sem
U_morph

semantic preservation objective
orthogonality objective
reconstruction objective

lambda_sem  = 0.05
lambda_orth = 0.05
lambda_rec  = 0.10

optimizer
scheduler
augmentation
TTA
fusion
thresholds
metric
dataset split
```

---

## 删除

彻底删除训练中的：

```text
pointwise morphology equivariance loss
L_eq_point
```

删除其作为训练 objective 的全部使用。

可以保留原 raw EqErr 作为 diagnostics，但不能继续参与训练。

---

## 新增

加入：

```text
local structural affinity equivariance loss
L_struct
```

并固定：

\[
\boxed{
\lambda_{struct}=0.05
}
\]

原因：

- 继承 v1.2 已验证健康的 morphology gradient budget；
- 不使用 validation performance 调整；
- 新版本首先检验 objective formulation，而不是增大作用强度。

---

# 4. OSMF-v1.3 总 Loss

\[
L
=
L_{SSHR}
+
0.05L_{sem-pres}
+
0.05L_{struct}
+
0.05L_{orth}
+
0.10L_{rec}.
\]

禁止修改任何系数。

---

# 5. Local Structural Representation

设 morphology feature：

\[
M\in\mathbb R^{B\times C_m\times h\times w}.
\]

对每个 spatial location：

\[
p=(x,y)
\]

取固定 8-neighborhood：

\[
\mathcal N_8(p)
\]

对应 offset：

```text
(-1,-1), (-1,0), (-1,+1)
( 0,-1),          ( 0,+1)
(+1,-1), (+1,0), (+1,+1)
```

禁止：

```text
learnable neighborhood
dilated neighborhood search
attention neighborhood
multi-radius neighborhood
graph construction
```

v1.3 只验证最小 8-neighbor local structure。

---

# 6. Local Affinity Definition

先对 morphology feature 做 channel normalization：

\[
\bar M_p
=
\frac{
M_p
}{
\|M_p\|_2+\epsilon
}.
\]

对每个：

\[
q\in\mathcal N_8(p)
\]

定义 cosine affinity：

\[
A_M(p,q)
=
\langle
\bar M_p,\bar M_q
\rangle.
\]

于是得到：

\[
A_M
\in
\mathbb R^{B\times8\times h\times w}.
\]

边界位置使用固定 valid-neighbor mask：

\[
V(p,q)\in\{0,1\}.
\]

禁止 padding 产生伪邻居参与 loss。

---

# 7. Two-View Construction

继续使用 v1.2 的 two-view training framework。

view A：

\[
x_a=A_p^{(a)}(x)
\]

view B：

\[
x_b=A_p^{(b)}(T_g(x)).
\]

其中 geometric transform 仍然只允许：

```text
horizontal flip
vertical flip
```

保持：

```text
structural_interval = 4
```

即每四个 optimizer steps 执行一次 second-view structural objective。

禁止修改 augmentation family。

---

# 8. Direction-Aware Affinity Alignment

这是 v1.3 最重要的实现细节之一。

对于 raw feature map，horizontal/vertical flip 只需要 inverse spatial alignment。

但 8-neighbor affinity map 还存在：

\[
\boxed{
\text{neighbor direction permutation}
}
\]

问题。

例如 horizontal flip 后：

```text
left  <-> right
upper-left <-> upper-right
lower-left <-> lower-right
up/down unchanged
```

vertical flip 后：

```text
up <-> down
upper-left <-> lower-left
upper-right <-> lower-right
left/right unchanged
```

因此必须同时执行：

1. spatial inverse alignment；
2. neighbor-direction channel inverse permutation。

定义：

\[
\tilde A_b
=
T_{dir}^{-1}
\left(
T_{spatial}^{-1}
(A_b)
\right).
\]

Phase-0M 已经有 direction-aware affinity alignment 单元测试，可复用其正确实现，但不得直接复用旧训练 objective。

---

# 9. Structural Affinity Loss

定义：

\[
A_a=A_M(x_a)
\]

\[
A_b=A_M(x_b)
\]

inverse-align 后：

\[
\tilde A_b.
\]

v1.3 使用最简单的 masked Smooth-L1：

\[
\boxed{
L_{struct}
=
\frac{
\sum V\cdot SmoothL1(A_a,\tilde A_b)
}{
\sum V+\epsilon
}
}
\]

固定：

```text
SmoothL1 beta = framework default
```

如果现有 PyTorch API 需要显式 beta，则固定：

```text
beta = 1.0
```

禁止搜索 beta。

---

# 10. 为什么不用 cosine(Aa,Ab)

Affinity 本身已经是：

\[
[-1,1]
\]

范围的 relational scalar。

使用：

\[
|A_a-\tilde A_b|
\]

或 Smooth-L1 更直接地表达：

> 邻域关系强度在变换前后应保持一致。

因此 v1.3 不再对 affinity vector 做二次 cosine normalization。

---

# 11. 为什么不用 pseudo-boundary / GT

v1.3 必须保持原 WSSS supervision setting。

禁止：

```text
segmentation GT
GT boundary
pseudo boundary from val GT
offline pseudo-mask
foundation-model mask
SAM mask
external morphology annotation
```

local affinity supervision 完全来自：

\[
\boxed{
\text{self-supervised two-view structural consistency}
}
\]

---

# 12. 为什么不直接做 Region / Graph

本版本只验证：

\[
\boxed{
\text{local relational morphology supervision}
}
\]

是否比已失败的：

\[
\text{pointwise feature equivariance}
\]

更符合 OSMF 的 morphology role。

因此暂不引入：

- region tokens；
- superpixels；
- GNN；
- topology；
- graph diffusion；
- object slots。

如果 8-neighbor affinity objective 都无法通过因果 gate，则没有理由继续扩大结构。

---

# 13. Stage A — Exact Parity

由于训练 loss 改变，但 inference architecture 不变，初始化 inference 仍应严格等价 A0。

必须：

\[
\max|\hat H-H|<10^{-6}.
\]

检查：

```text
CAM56 diff = 0
CAM28_1 diff = 0
CAM28_2 diff = 0
CAMdeep diff = 0
classification probability diff = 0
```

Full BCSS validation：

```text
differing pixels = 0
mIoU diff < 1e-7
mDice diff < 1e-7
```

失败：

```text
OSMF_V13_PARITY_NOGO
```

STOP。

---

# 14. Stage B — 8-Batch Structural Gradient Readiness

从 A0 fresh restart：

```text
8 real BCSS training batches
seed = 20260817
batch_size = 20
image_size = 224
precision = BF16
structural_interval = 4
```

不得继续 v1.2 或其它 checkpoint。

不得执行第 9 batch。

---

# 15. 8-Batch Audit Points

记录：

```text
0
1
2
4
8
```

---

# 16. Weighted Gradient Ratios

定义：

\[
r_j
=
\frac{
\lambda_j
\|\nabla_HL_j\|_2
}{
\|\nabla_HL_{SSHR}\|_2+\epsilon
}.
\]

记录：

```text
r_sem
r_struct
r_orth
r_rec
```

---

# 17. Structural Gradient Budget

PASS target：

\[
Mean(r_{struct})\le0.20
\]

且：

\[
Max(r_{struct})\le0.30.
\]

Semantic 继续：

\[
Mean(r_{sem})\le0.20
\]

\[
Max(r_{sem})\le0.30.
\]

---

# 18. Structural Objective Connectivity

必须确认：

```text
p_morph.weight gradient finite and non-zero
u_morph.weight gradient finite and non-zero
```

在 structural-active steps：

```text
4
8
```

都应获得来自：

\[
L_{struct}
\]

的有效 gradient。

如果：

```text
structural objective inactive
```

则：

```text
OSMF_V13_READINESS_NOGO
```

---

# 19. 8-Batch Same-Pair Structural Causal Test

v1.3 在 readiness 阶段就必须做小型因果检查。

在 structural-active：

```text
step 4
step 8
```

固定当前 exact two-view pair。

optimizer update 前计算：

\[
AffinityEqErr^{before}.
\]

执行正常 full v1.3 joint optimizer step。

对完全相同 pair re-forward：

\[
AffinityEqErr^{after}.
\]

定义：

\[
\Delta StructErr
=
After-Before.
\]

PASS readiness 要求：

```text
step4 delta < 0
OR
step8 delta < 0
```

且两步平均：

\[
Mean(\Delta StructErr)<0.
\]

如果两步都：

\[
>0
\]

则：

```text
OSMF_V13_READINESS_NOGO
```

不进入 128 batches。

---

# 20. 8-Batch Representation Safety

继续记录：

```text
SemAgree
semantic response RMS ratio
Cos(H,H_hat)
RMS(S)
RMS(M)
RMS(S)/RMS(M)
CrossCov(S,M)
```

PASS 要求：

\[
Cos(H,\hat H)\ge0.95.
\]

\[
0.05<RMS(S)/RMS(M)<20.
\]

semantic response 不 collapse。

---

# 21. OSMF-v1.3 Readiness PASS

输出：

```text
OSMF_V13_READINESS_PASS
```

要求：

1. parity PASS
2. all finite
3. semantic path active
4. structural morphology path active
5. mean/max gradient budgets within target
6. no branch collapse
7. reconstruction ≥0.95
8. semantic response healthy
9. two-step structural causal mean <0
10. no SSHR-loss instability

只有 PASS 才允许进入 128-batch Phase-0S。

---

# 22. Readiness REVIEW

若：

- `r_struct` 进入 0.30–0.50；
- small causal test 一正一负且 mean 接近 0；
- representation health 尚可但 structural signal 不够明确；

输出：

```text
OSMF_V13_READINESS_REVIEW
```

STOP。

---

# 23. Readiness NOGO

若：

- persistent ratio >0.50；
- two structural-active steps 均使 same-pair affinity error 变差；
- morphology path inactive；
- semantic path collapse；
- branch collapse；
- reconstruction <0.90；
- NaN/Inf；
- SSHR loss explosion；

输出：

```text
OSMF_V13_READINESS_NOGO
```

STOP。

---

# 24. Stage C — Fresh 128-Batch Phase-0S

仅在：

```text
OSMF_V13_READINESS_PASS
```

后允许。

必须重新从同一个 A0 checkpoint fresh restart。

不得从 8-batch readiness 权重继续。

运行：

```text
128 real BCSS training batches
seed = 20260817
structural_interval = 4
```

---

# 25. Phase-0S 的四个核心问题

128-batch 只回答：

1. `L_struct` 是否对 same-pair local structure 有稳定因果改善？
2. fixed unseen probe 上 structural consistency 是否真正改善？
3. semantic / structural specialization 是否同时保持？
4. factorization / reconstruction / decorrelation 是否继续健康？

---

# 26. Same-Pair Structural Causal Audit

对所有 structural-active steps：

```text
4,8,12,...,128
```

共：

\[
32
\]

个 steps。

每个 step：

1. freeze exact pair + augmentation metadata；
2. before update：
   \[
   AffinityEqErr^{before}
   \]
3. normal full joint optimizer step；
4. exact same-pair after：
   \[
   AffinityEqErr^{after}
   \]
5. 计算：
   \[
   \Delta StructErr=after-before.
   \]

---

# 27. Primary Causal Gate

定义：

```text
improved_fraction =
count(delta < -1e-6) / 32
```

```text
harmed_fraction =
count(delta > +1e-6) / 32
```

```text
neutral =
abs(delta) <= 1e-6
```

## Strong causal validity

要求：

\[
ImprovedFraction\ge0.75
\]

且：

\[
Mean(\Delta StructErr)<0.
\]

---

# 28. Fixed Structural Probe

固定：

```text
64 BCSS training images
seed = 20260817
```

使用与 Phase-0M 相同思想：

- image IDs 固定；
- photometric transform 固定；
- geometric flip 固定；
- 不使用 GT；
- probe 不参与 optimizer update。

Audit：

```text
0
4
8
16
32
64
96
128
```

---

# 29. Fixed Probe Metrics

必须计算：

## Primary

\[
AffinityEqErr_M^{fixed}
\]

## Semantic control

\[
AffinityEqErr_S^{fixed}
\]

## Secondary raw diagnostic

\[
RawEqErr_M^{fixed}
\]

\[
RawEqErr_S^{fixed}
\]

raw pointwise EqErr 不再作为 v1.3 训练目标或主要成功判据。

---

# 30. Fixed Probe Success

Primary requirement：

\[
\boxed{
AffinityEqErr_M^{fixed}(128)
<
AffinityEqErr_M^{fixed}(0)
}
\]

相对改善定义：

\[
StructImprove
=
\frac{
Err_0-Err_{128}
}{
Err_0+\epsilon
}.
\]

Phase-0S GO target：

\[
\boxed{
StructImprove\ge5\%
}
\]

10% 以上视为 strong structural signal，但不要求。

---

# 31. Morphology-vs-Semantic Specialization

希望：

\[
StructImprove_M
>
StructImprove_S.
\]

定义：

\[
SpecificityGap
=
StructImprove_M-StructImprove_S.
\]

如果：

\[
SpecificityGap>0
\]

标记：

```text
MORPHOLOGY_SPECIFIC_STRUCTURAL_GAIN
```

如果：

\[
M
\]

和：

\[
S
\]

几乎同等改善，

则：

```text
STRUCTURAL_GAIN_NOT_MORPHOLOGY_SPECIFIC
```

这不是自动 NOGO，但会降低 factorization claim 强度。

---

# 32. Semantic Preservation

继续监控：

\[
SemAgree.
\]

Phase-0S GO 要求：

\[
\boxed{
SemAgree_{128}\ge0.90
}
\]

同时：

\[
RMS(Z_S)/RMS(Z_H)>0.05.
\]

---

# 33. Reconstruction

继续：

\[
Cos(H,\hat H).
\]

GO：

\[
\boxed{
CosRec_{128}\ge0.95
}
\]

Review：

\[
0.90\le CosRec<0.95.
\]

NOGO：

\[
CosRec<0.90.
\]

---

# 34. Branch Health

继续：

\[
R_{SM}=RMS(S)/RMS(M).
\]

GO-compatible：

\[
0.05<R_{SM}<20.
\]

---

# 35. Cross-Subspace Redundancy

继续：

\[
CrossCov(S,M).
\]

期望：

\[
CrossCov_{128}<CrossCov_0.
\]

若下降且两路健康：

```text
GENUINE_DECORRELATION_SIGNAL
```

---

# 36. Structural Gradient Competition

在 morphology parameters：

```text
p_morph
u_morph
```

上计算：

\[
\cos(g_{struct},g_{SSHR})
\]

\[
\cos(g_{struct},g_{sem})
\]

\[
\cos(g_{struct},g_{orth})
\]

\[
\cos(g_{struct},g_{rec}).
\]

Audit：

```text
4,8,16,32,64,96,128
```

不做 gradient surgery。

---

# 37. Gradient Budget 继续检查

128-batch target：

\[
Mean(r_{struct})\le0.20
\]

\[
P95(r_{struct})\le0.30.
\]

semantic 同样：

\[
Mean(r_{sem})\le0.20
\]

\[
P95(r_{sem})\le0.30.
\]

持续：

\[
>0.50
\]

直接 NOGO。

---

# 38. Primary Phase-0S Decision

最终只能输出：

```text
OSMF_V13_PHASE0S_GO
OSMF_V13_PHASE0S_REVIEW
OSMF_V13_PHASE0S_NOGO
```

---

# 39. OSMF_V13_PHASE0S_GO

必须满足：

1. readiness PASS
2. 128 batches all finite
3. same-pair improved fraction ≥0.75
4. mean causal structural delta <0
5. fixed-probe morphology affinity error improves ≥5%
6. morphology branch active
7. semantic branch active
8. SemAgree ≥0.90
9. CosRec ≥0.95
10. no branch collapse
11. CrossCov finite
12. specialization gradients remain auxiliary
13. SSHR base loss stable

若同时：

\[
SpecificityGap>0
\]

额外标记：

```text
MORPHOLOGY_SPECIFIC_STRUCTURAL_GAIN
```

这是最理想结果。

GO 只表示：

> 可以人工 review 是否进入 3-epoch mechanism pilot。

---

# 40. OSMF_V13_PHASE0S_REVIEW

适用于：

- same-pair causal validity 成立，但 fixed probe 改善 <5%；
- fixed probe 改善，但 improved fraction 仅 0.50–0.75；
- structural gain 同时发生在 S 和 M，缺少 specialization specificity；
- reconstruction / semantic health 进入 review 区；
- objective 有信号但还不够支持正式 pilot。

STOP。

---

# 41. OSMF_V13_PHASE0S_NOGO

若：

- same-pair improved fraction <0.50；
- mean structural causal delta ≥0；
- fixed-probe morphology AffinityEqErr 明显变差；
- structural objective inactive；
- branch collapse；
- semantic collapse；
- reconstruction <0.90；
- NaN / Inf；
- SSHR loss explosion；
- persistent structural gradient domination >0.50；

输出：

```text
OSMF_V13_PHASE0S_NOGO
```

STOP。

---

# 42. 重要止损规则

如果 v1.3 最终：

```text
OSMF_V13_PHASE0S_NOGO
```

则：

\[
\boxed{
\text{停止 OSMF morphology-specialization objective line}
}
\]

禁止自动开发：

```text
v1.4
multi-scale affinity
graph affinity
topology affinity
region-token rescue
boundary rescue
prototype rescue
```

下一步应转向完全不同的：

```text
region-centric / object-centric representation
```

而不是无限微调 OSMF morphology objective。

---

# 43. 如果 v1.3 GO

才允许设计：

# OSMF-v1.3 Phase 1 — 3-Epoch Mechanism Pilot

该阶段需重新预注册，并首次开始看：

- CAM28_1 validation mIoU；
- official fusion validation mIoU；
- semantic preservation；
- morphology structural consistency；
- CrossCov；
- reconstruction；
- 是否真正将 structural specialization 转化为 segmentation gain。

本执行方案不授权 Phase 1。

---

# 44. 不使用 Validation Performance 做 Phase-0S 决策

Phase-0S 禁止：

```text
validation mIoU tuning
validation mDice tuning
per-class IoU selection
threshold tuning
```

可以做 exact parity，但不能用 validation segmentation metric 选择或调试 v1.3。

Phase-0S 的 primary evidence 是：

\[
\boxed{
\text{causal structural learning + fixed-probe generalization}
}
\]

---

# 45. 输出目录

建议：

```text
/home/duyanhong/experiments/
OSMF_V13_LOCAL_STRUCTURAL_<commit>/
```

结构：

```text
parity/
  summary.json
  report.md

readiness_8b/
  gradient_ratio.csv
  causal_structural.csv
  representation_health.csv
  parameter_health.csv
  report.md

phase0s_128b/
  structural_causal_steps.csv
  structural_causal_summary.csv
  fixed_probe_manifest.csv
  fixed_probe_affinity.csv
  fixed_probe_raw_eq.csv
  gradient_ratio.csv
  morphology_gradient_cosine.csv
  semantic_health.csv
  representation_health.csv
  redundancy.csv
  compute_cost.csv
  figures/
  report.md

docs/
  osmf_v13_local_structural_delivery.md
```

---

# 46. Required Causal Table

| Step | StructErr Before | StructErr After | Delta | Improved |
|---:|---:|---:|---:|---:|
| 4 | | | | |
| 8 | | | | |
| ... | | | | |
| 128 | | | | |

必须给：

```text
improved fraction
harmed fraction
neutral fraction
mean delta
median delta
P25
P75
min
max
```

---

# 47. Required Fixed-Probe Table

| Step | AffinityEqErr_M | AffinityEqErr_S | RawEqErr_M | RawEqErr_S |
|---:|---:|---:|---:|---:|
| 0 | | | | |
| 4 | | | | |
| 8 | | | | |
| 16 | | | | |
| 32 | | | | |
| 64 | | | | |
| 96 | | | | |
| 128 | | | | |

---

# 48. Required Specialization Summary

必须报告：

```text
SemAgree start/end
semantic response RMS ratio start/end

AffinityEqErr_M start/end
AffinityEqErr_S start/end
StructImprove_M
StructImprove_S
SpecificityGap

CrossCov start/end
S/M RMS ratio start/end
CosRec start/end
```

---

# 49. Required Gradient Table

| Objective | Mean | Max | P95 |
|---|---:|---:|---:|
| semantic preservation | | | |
| structural affinity | | | |
| orthogonality | | | |
| reconstruction | | | |

以及 morphology-parameter gradient cosine：

| Pair | Mean cosine |
|---|---:|
| struct vs SSHR | |
| struct vs semantic | |
| struct vs orth | |
| struct vs rec | |

---

# 50. Codex Master Prompt

你现在实现并执行一个新的独立预注册版本：

**OSMF-v1.3 — Local Structural Morphology Learning**

## Scientific reason

OSMF-v1.2 Phase-0M proved that the previous pointwise morphology feature-equivariance objective is invalid:

```text
same-pair improved = 16/32
same-pair harmed = 16/32
mean causal delta = +0.00004601

fixed raw EqErr_M:
0.063379 -> 0.069951

fixed AffinityEqErr_M:
0.010489 -> 0.012005

cos(eq, SSHR) on morphology parameters = -0.005286
```

Therefore the failure is not due to random probes, strong task-gradient conflict, or a raw-vs-affinity metric mismatch.

The pointwise objective must be removed.

## New hypothesis

Morphology should be represented by local relational geometry rather than absolute pointwise feature identity.

## Keep frozen from v1.2

```text
factorization point = post-HFRM H28_1
split = 256/256

P_sem
P_morph
U_sem
U_morph

semantic preservation unchanged
orthogonality unchanged
reconstruction unchanged

lambda_sem    = 0.05
lambda_struct = 0.05
lambda_orth   = 0.05
lambda_rec    = 0.10

optimizer
scheduler
augmentation
fusion
thresholds
TTA
metric
```

## Remove

Do not train with pointwise `L_eq`.

## New morphology objective

Normalize morphology feature channel-wise.

For each pixel, compute cosine affinity to the fixed 8-neighborhood.

Produce:

```text
A_M: [B,8,H,W]
```

Use valid-neighbor masks at borders.

For transformed view:

1. inverse spatially align the affinity map;
2. inverse-permute neighbor-direction channels.

Use:

```text
L_struct = masked SmoothL1(A_a, A_b_aligned)
```

with fixed default beta / beta=1.0 if explicit.

Do not add other structural losses.

## Stage A — parity

Require exact A0 inference parity.

If fail:

```text
OSMF_V13_PARITY_NOGO
```

STOP.

## Stage B — 8-batch readiness

Fresh from A0.

Run exactly 8 BCSS train batches.

Audit:

```text
0,1,2,4,8
```

Gradient targets:

```text
mean r_sem <= 0.20
max r_sem <= 0.30

mean r_struct <= 0.20
max r_struct <= 0.30
```

At structural-active step 4 and 8 perform exact same-pair before/after affinity causal measurement.

Require at least one improving step and mean structural causal delta <0.

Also require:

```text
all finite
semantic path active
morphology path active
all four factorization tensors update
CosRec >=0.95
no branch collapse
semantic response non-collapsed
```

Return exactly one:

```text
OSMF_V13_READINESS_PASS
OSMF_V13_READINESS_REVIEW
OSMF_V13_READINESS_NOGO
```

Only PASS proceeds.

## Stage C — fresh 128-batch Phase-0S

Restart from A0.

Do NOT continue readiness weights.

Run exactly 128 batches.

At every structural-active step:

```text
4,8,12,...,128
```

perform exact same-pair structural causal before/after test.

Primary causal GO target:

```text
improved_fraction >= 0.75
mean causal delta < 0
```

## Fixed 64-image probe

Use fixed GT-free 64-image training probe.

At:

```text
0,4,8,16,32,64,96,128
```

compute:

```text
AffinityEqErr_M
AffinityEqErr_S
RawEqErr_M
RawEqErr_S
```

Primary fixed-probe target:

```text
AffinityEqErr_M_end < AffinityEqErr_M_start
relative improvement >= 5%
```

Also calculate:

```text
StructImprove_M
StructImprove_S
SpecificityGap
```

## Continue safety diagnostics

```text
SemAgree
semantic response RMS ratio
Cos(H,H_hat)
RMS(S)/RMS(M)
CrossCov
r_sem
r_struct
r_orth
r_rec
L_SSHR
```

GO requires:

```text
SemAgree_128 >=0.90
CosRec_128 >=0.95
0.05 < S/M ratio <20
no collapse
no persistent gradient domination
```

## Primary decision

Return exactly one:

```text
OSMF_V13_PHASE0S_GO
OSMF_V13_PHASE0S_REVIEW
OSMF_V13_PHASE0S_NOGO
```

### GO

Require:

```text
same-pair improved_fraction >=0.75
mean causal delta <0
fixed AffinityEqErr_M relative improvement >=5%
healthy semantic path
healthy morphology path
SemAgree >=0.90
CosRec >=0.95
no collapse
stable SSHR loss
```

### REVIEW

Use when structural objective shows some causal/generalization signal but misses one GO threshold.

### NOGO

Use when:

```text
same-pair improved_fraction <0.50
OR mean causal delta >=0
OR fixed AffinityEqErr_M clearly worsens
OR structural path inactive
OR collapse/instability occurs
```

## Stop rule

Even if GO:

STOP.

Do not run 3 epochs.
Do not run 25 epochs.
Do not run BCSS test.
Do not run LUAD.
Do not add region tokens.
Do not add graph modules.
Do not tune lambda.
Do not implement v1.4.

Wait for human scientific review.


