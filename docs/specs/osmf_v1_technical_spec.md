# OSMF-v1.0：Objective-Induced Semantic–Morphology Factorization
## 基于 SSHR 的目标诱导语义–形态互补表征学习最终技术方案

**状态：** Research Candidate / Minimum Falsifiable Version  
**Baseline：** Frozen SSHR experimental protocol  
**首个实验数据集：** BCSS  
**首个修改位置：** `CAM28_1` 对应的 post-HFRM representation  
**核心目标：**

\[
\boxed{
\text{Do not route accidental complementarity;
create structured complementarity by design.}
}
\]

即：

> 不再从 SSHR 偶然形成的多尺度互补结果中事后寻找“谁更可靠”，而是在训练过程中，通过不同学习目标主动形成具有明确职责的 semantic representation 与 morphology representation。

---

# 1. 研究背景

Phase-0 / Phase-0B 已经得到三个关键事实。

第一，SSHR hierarchy 中确实存在明显互补信息：

\[
\Delta ImageOracle \approx +2.1
\]

\[
\Delta ImageSoftFusionOracle \approx +2.63
\]

\[
\Delta LocalImageClassOracle \approx +4.34
\]

\[
\Delta PixelOracle \approx +6.71.
\]

但第二，当前 representation 并没有形成可以稳定预测这种互补性的 inference signal。

正式 MLP-C：

\[
67.3279
\rightarrow
66.6121
\]

即：

\[
\boxed{-0.7158\text{ pp}}.
\]

第三，当前相对有信息量的 signal 反而主要来自：

- boundary density；
- component count；
- spatial entropy；
- activation geometry。

因此当前科学判断是：

\[
\boxed{
\text{SSHR does not mainly lack a stronger router;
it lacks structured complementary representations.}
}
\]

---

# 2. SSHR 当前潜在 representation 问题

对于某一 hierarchy feature：

\[
H_i
\]

它同时承担：

\[
\text{tissue semantic discrimination}
\]

+

\[
\text{local morphology preservation}
\]

+

\[
\text{boundary localization}
\]

+

\[
\text{deep-semantic rectification}.
\]

这相当于要求一个 latent representation 同时满足：

\[
\text{semantic invariance}
\]

和：

\[
\text{spatial sensitivity}.
\]

这两个目标并不完全一致。

对 semantic representation：

\[
T(x)
\]

经过旋转、翻转或颜色变化以后：

\[
Semantic(T(x))
\approx
Semantic(x).
\]

但对 morphology representation：

\[
Morph(T(x))
\]

应该满足：

\[
Morph(T(x))
\approx
T(Morph(x)).
\]

前者是：

\[
\boxed{\text{invariance}}
\]

后者是：

\[
\boxed{\text{equivariance}}.
\]

因此把二者压在同一 feature 中，可能导致：

\[
\boxed{
\text{semantic–morphology entanglement}.
}
\]

---

# 3. 核心研究假设

本工作提出：

\[
\boxed{
H_i
\longrightarrow
\left(
H_i^{sem},
H_i^{morph}
\right)
}
\]

其中：

### Semantic representation

负责：

\[
\boxed{
\text{What tissue is present?}
}
\]

主要学习：

- tissue identity；
- image-level class semantics；
- global discriminative information；
- augmentation-invariant semantics。

### Morphology representation

负责：

\[
\boxed{
\text{How and where is the tissue spatially organized?}
}
\]

主要学习：

- local spatial organization；
- region continuity；
- components；
- fine morphology；
- boundaries / interfaces；
- transformation-equivariant structure。

二者共同形成：

\[
\boxed{
\text{structured complementarity}.
}
\]

---

# 4. OSMF-v1.0 为什么只改 CAM28_1

第一阶段不修改整个 SSHR。

只修改：

\[
\boxed{H_{28_1}}
\]

即 CAM28_1 前面的 post-HFRM representation。

原因：

1. Phase-0 中 CAM28_1 是当前 strongest individual hierarchy；
2. official fusion 中 CAM28_1 权重最高；
3. 改一个 hierarchy 最容易做 attribution；
4. 避免同时改变多个 branch；
5. 如果这一位置都无法证明 factorization 有效，没有必要扩展到其它 stage。

所以第一阶段：

```text
CAM56      unchanged
CAM28_1    OSMF
CAM28_2    unchanged
CAMdeep    unchanged
fusion     unchanged
```

---

# 5. 总体结构

原 SSHR：

```text
H28_1
  ↓
CAM Head
  ↓
CAM28_1
```

OSMF：

```text
                     ┌→ Semantic Encoder → S
H28_1 → Factorizer ──┤
                     └→ Morphology Encoder → M

S + M
  ↓
Reconstruction
  ↓
H28_1_OSMF
  ↓
Original CAM Head
  ↓
CAM28_1
```

其它 SSHR 完全不变。

---

# 6. 最小因子分解结构

设：

\[
H
=
H_{28_1}
\in
\mathbb R^{C\times h\times w}.
\]

拆成：

\[
S=P_s(H)
\]

和：

\[
M=P_m(H).
\]

其中：

\[
S\in\mathbb R^{C_s\times h\times w}
\]

\[
M\in\mathbb R^{C_m\times h\times w}
\]

并：

\[
C_s+C_m=C.
\]

首版固定：

\[
C_s=\lceil C/2\rceil
\]

\[
C_m=\lfloor C/2\rfloor.
\]

不搜索 ratio。

---

# 7. Projection / Reconstruction

使用：

\[
1\times1 Conv
\]

实现：

\[
P_s,\quad P_m.
\]

对应 reconstruction：

\[
\hat H
=
U_s(S)+U_m(M).
\]

其中：

\[
U_s:
C_s\rightarrow C
\]

\[
U_m:
C_m\rightarrow C.
\]

所以：

\[
\boxed{
\hat H
=
U_s(P_s(H))
+
U_m(P_m(H)).
}
\]

最终原 SSHR CAM head 接收：

\[
\hat H.
\]

---

# 8. 极其重要：Baseline-Preserving Initialization

新结构必须满足：

\[
\boxed{
\hat H^{(0)}=H.
}
\]

即模型刚插入 OSMF、尚未训练时：

\[
CAM^{OSMF}_{28_1}
=
CAM^{A0}_{28_1}.
\]

实现方式：

将原通道划成两个互补 partition。

Semantic projection 初始选择：

```text
channels [0 : Cs]
```

Morph projection 初始选择：

```text
channels [Cs : C]
```

对应 reconstruction conv 初始化成反向 channel placement。

因此：

\[
U_sP_s+U_mP_m=I.
\]

要求 initial parity：

\[
\max|\hat H-H|<10^{-6}.
\]

并要求：

\[
mIoU_{OSMF-init}
=
mIoU_{A0}
\]

在同 checkpoint / 同 inference 下完全一致或数值误差：

\[
<10^{-7}.
\]

若失败：

\[
\boxed{STOP}
\]

不得训练。

---

# 9. Loss 总览

保持原 SSHR loss：

\[
L_{SSHR}
\]

完全不变。

加入：

\[
L_{OSMF}
=
\lambda_{sem}L_{sem}
+
\lambda_{morph}L_{eq}
+
\lambda_{orth}L_{orth}
+
\lambda_{rec}L_{rec}.
\]

最终：

\[
\boxed{
L
=
L_{SSHR}
+
L_{OSMF}.
}
\]

v1.0 不加入：

- pseudo mask；
- boundary GT；
- prototype；
- frequency；
- uncertainty；
- router；
- contrastive memory bank；
- topology loss；
- MoE。

---

# 10. Semantic Objective

Semantic path 要回答：

\[
\boxed{\text{What tissue is present?}}
\]

因此：

\[
z_s=GAP(S).
\]

接一个轻量 auxiliary classification head：

\[
\hat y_s=W_sz_s.
\]

使用与 SSHR image-level supervision 相同的：

\[
\boxed{L_{sem}=L_{cls}(\hat y_s,y)}.
\]

这里只允许：

```text
GAP
Linear
```

禁止：

```text
MLP
attention
transformer
prototype classifier
```

目的是让：

\[
S
\]

明确承担：

\[
\boxed{\text{global tissue semantics}}.
\]

---

# 11. Morphology Objective

Morphology branch 不直接使用 segmentation GT。

也不把：

\[
high\ frequency
\]

定义成 morphology。

而通过：

\[
\boxed{\text{spatial equivariance}}
\]

学习 morphology。

---

# 12. 两视图生成

训练中构造：

\[
x_a=A_p^{(a)}(x)
\]

和：

\[
x_b=A_p^{(b)}(T_g(x)).
\]

其中：

### Photometric augmentation

可使用现有训练 pipeline 中已经允许的：

- color jitter；
- brightness；
- contrast；
- stain-like intensity variation。

禁止新增极端 augmentation。

### Geometric transform

v1.0 固定从：

```text
horizontal flip
vertical flip
```

中选择。

不使用：

```text
random arbitrary rotation
elastic deformation
crop-resize warping
```

避免增加额外变量。

---

# 13. Morphology Equivariance

得到：

\[
M_a=M(x_a)
\]

和：

\[
M_b=M(x_b).
\]

将：

\[
M_b
\]

inverse-align：

\[
\tilde M_b=T_g^{-1}(M_b).
\]

对每个 spatial location 做 channel L2 normalization：

\[
\bar M(x)
=
\frac{M(x)}
{\|M(x)\|_2+\epsilon}.
\]

定义：

\[
\boxed{
L_{eq}
=
\frac1{hw}
\sum_x
\left[
1-
\cos(
\bar M_a(x),
\bar{\tilde M}_b(x)
)
\right].
}
\]

它强迫：

\[
M
\]

在颜色变化下保持 spatial correspondence，

同时在 geometric transform 下正确跟随空间位置。

---

# 14. 控制训练开销

第二 view 不需要每个 iteration 都计算。

固定：

```text
equivariance_interval = 4
```

即：

每四个 training steps，

仅一个 step 计算第二 view。

所以理论增加 backbone forward：

\[
\approx25\%.
\]

禁止根据运行速度改变 interval。

第一版 training-time overhead 若：

\[
>40\%
\]

必须记录并人工 review。

---

# 15. Orthogonality / Decorrelation

仅仅：

\[
S\neq M
\]

还不够。

需要减少两者 redundancy。

首先对：

\[
S,M
\]

分别在 batch × spatial token 维做标准化。

令：

\[
\tilde S\in\mathbb R^{N\times C_s}
\]

\[
\tilde M\in\mathbb R^{N\times C_m}
\]

其中：

\[
N=Bhw.
\]

cross-covariance：

\[
C_{SM}
=
\frac1N
\tilde S^T\tilde M.
\]

定义：

\[
\boxed{
L_{orth}
=
\frac{
\|C_{SM}\|_F^2
}{
C_sC_m
}.
}
\]

目的：

\[
\boxed{
\text{reduce redundancy}
}
\]

而不是直接定义 semantic / morphology。

必须明确：

> role 由不同 objective 定义，orthogonality 只用于阻止两条路径学习完全相同的东西。

---

# 16. Reconstruction Constraint

为了防止 decomposition 破坏 SSHR 已经有效的 representation：

\[
\hat H
=
U_s(S)+U_m(M).
\]

原 feature 作为 stop-gradient target：

\[
H^{sg}=stopgrad(H).
\]

定义 normalized reconstruction：

\[
L_{rec}
=
1-
\frac{
\langle
\hat H,H^{sg}
\rangle
}{
\|\hat H\|_2
\|H^{sg}\|_2+\epsilon
}.
\]

或者代码中等价实现 stable cosine reconstruction loss。

禁止 reconstruction loss 反向改变：

\[
H
\]

本身。

即：

```python
target = H.detach()
```

这样 OSMF 学习：

> 如何重新组织已有 HFRM information

而不是：

> 强迫整个 SSHR backbone 去迎合 reconstruction。

---

# 17. v1.0 固定 Loss Weights

第一版固定：

\[
\boxed{
\lambda_{sem}=0.20
}
\]

\[
\boxed{
\lambda_{morph}=0.20
}
\]

\[
\boxed{
\lambda_{orth}=0.05
}
\]

\[
\boxed{
\lambda_{rec}=0.10
}
\]

这些是 **OSMF-v1.0 工程默认值，不是文献或 Phase-0B 给出的经验最优值。**

因此：

\[
\boxed{\text{禁止 validation tuning}}
\]

第一枪只测试一个 configuration。

---

# 18. Pre-Training Gradient Audit

正式训练前，对固定：

```text
128 training batches
seed = 20260817
```

统计各 loss 对：

\[
H_{28_1}
\]

的 gradient norm：

\[
g_{base}
\]

\[
g_{sem}
\]

\[
g_{eq}
\]

\[
g_{orth}
\]

\[
g_{rec}.
\]

报告：

\[
r_j
=
\frac{
\lambda_jg_j
}{
g_{base}+\epsilon
}.
\]

推荐安全区：

\[
0.02
\le r_j
\le0.30.
\]

如果任意 auxiliary objective：

\[
r_j>0.50
\]

则：

\[
\boxed{STOP\ FOR\ REVIEW}
\]

不得自行调 lambda。

这是 sanity gate，

不是 tuning procedure。

---

# 19. 为什么 Morphology Branch 不加普通 classification loss

如果给：

\[
M
\]

也加：

\[
L_{cls}
\]

那么最容易出现：

\[
S\approx M.
\]

两个 branch 都重新学习：

\[
tissue\ classification.
\]

这样：

\[
\boxed{
\text{multiple branches}
\neq
\text{specialized branches}.
}
\]

所以 v1.0：

Semantic：

\[
L_{cls}
\]

Morphology：

\[
L_{eq}.
\]

Joint reconstructed representation：

\[
L_{SSHR}.
\]

这三个 objective 分工明确。

---

# 20. 为什么 Morphology Branch 不使用 Frequency

禁止：

\[
FFT
\]

禁止：

\[
HighPass
\]

禁止：

\[
LowPass
\]

禁止：

\[
HighFrequency=Morphology.
\]

原因：

FA-MPR 已经说明：

\[
\boxed{
HighFrequency
\neq
UsefulMorphology.
}
\]

OSMF 的核心升级就是：

\[
\boxed{
\text{signal-defined decomposition}
\rightarrow
\text{objective-induced decomposition}.
}
\]

---

# 21. 为什么不直接加入 Boundary Loss

因为 BCSS segmentation GT 不能进入训练。

否则：

\[
\boxed{
\text{WSSS setting changes}.
}
\]

GT boundary 只允许在 validation diagnostics 中使用。

训练 morphology path 必须继续：

\[
\boxed{\text{image-level labels only}}
\]

加 self-supervised spatial objective。

---

# 22. Architecture v1.0

最终首版结构：

```text
Image
  ↓
ResNet38
  ↓
Original SSHR hierarchy
  ↓
Original HFRM
  ↓
        H28_1
          │
          ├─────────────────────────────┐
          ↓                             ↓
    Semantic Projector           Morph Projector
          ↓                             ↓
          S                             M
          │                             │
     GAP + classifier            Equivariance loss
          │                             │
      L_sem                         L_eq
          │                             │
          └──── Cross-covariance ──────┘
                     ↓
                  L_orth

          S                             M
          ↓                             ↓
         U_s                           U_m
          └────────────┬────────────────┘
                       ↓
                     H_hat
                       ↓
              reconstruction loss
                       ↓
              Original CAM28_1 Head
                       ↓
                    CAM28_1

Other three SSHR CAMs:
unchanged

Final official fusion:
unchanged
```

---

# 23. Inference

训练完成以后 inference 只需要：

\[
H
\rightarrow
P_s/P_m
\rightarrow
U_s/U_m
\rightarrow
\hat H.
\]

不需要：

- second view；
- semantic auxiliary classifier；
- equivariance loss；
- orthogonality computation；
- reconstruction target。

因此：

\[
\boxed{
training-only specialization objectives
}
\]

不会全部进入 inference cost。

---

# 24. Phase -1：Implementation Parity

新 branch：

```text
research/osmf-v1
```

禁止修改：

```text
dataset split
metric
threshold
TTA
official fusion
other CAM branches
```

首先只实现 OSMF，

不训练。

要求：

### Feature parity

\[
\max|\hat H-H|<10^{-6}.
\]

### Prediction parity

同一 baseline checkpoint：

```text
baseline prediction
OSMF-init prediction
```

differing pixels：

\[
0.
\]

mIoU difference：

\[
<10^{-7}.
\]

若失败：

\[
\boxed{STOP}
\]

---

# 25. Phase 0：128-Batch Structural Sanity Audit

训练正式开始前运行：

```text
128 batches only
```

检查：

### Finite

```text
S finite
M finite
H_hat finite
all losses finite
all gradients finite
```

### Reconstruction

初始：

\[
Cos(H,\hat H)>0.99999.
\]

### Branch norm

记录：

\[
R_{SM}
=
\frac{RMS(S)}{RMS(M)+\epsilon}.
\]

不要求二者一样，

但如果：

\[
R_{SM}>20
\]

或：

\[
R_{SM}<0.05
\]

则判定 collapse risk。

### Gradient ratio

按 Section 18。

完成：

\[
\boxed{STOP\ FOR\ REVIEW}
\]

确认没有工程问题后才能正式训练。

---

# 26. Phase 1：3-Epoch Mechanism Pilot

目的不是看最终 mIoU。

只检查：

\[
\boxed{
\text{specialization mechanism actually moves in the intended direction}.
}
\]

固定：

```text
BCSS
seed = 42
3 epochs
same optimizer/scheduler as SSHR
```

报告每 epoch：

```text
L_SSHR
L_sem
L_eq
L_orth
L_rec
RMS(S)
RMS(M)
Cos(H,H_hat)
CAM28_1 mIoU
official fusion mIoU
```

---

# 27. Phase 1 Mechanism GO

要求至少：

### Semantic learning

\[
L_{sem}^{epoch3}
<
L_{sem}^{epoch1}.
\]

### Morphology learning

\[
L_{eq}^{epoch3}
<
L_{eq}^{epoch1}.
\]

至少相对下降：

\[
10\%.
\]

### Redundancy reduction

\[
L_{orth}^{epoch3}
<
L_{orth}^{epoch1}.
\]

至少下降：

\[
10\%.
\]

### Reconstruction stable

\[
Cos(H,\hat H)\ge0.95.
\]

### No collapse

\[
0.05<R_{SM}<20.
\]

全部满足：

```text
OSMF_MECHANISM_GO
```

若明显违反：

```text
OSMF_MECHANISM_NOGO
```

不进入 25-epoch 正式训练。

---

# 28. Phase 2：正式 BCSS seed42 训练

Mechanism GO 后：

按冻结 SSHR protocol 完整训练：

\[
25\ epochs.
\]

要求：

- 相同 data；
- 相同 augmentations；
- 相同 optimizer；
- 相同 LR；
- 相同 scheduler；
- 相同 seed42；
- 相同 inference；
- 相同 validation protocol；
- 相同 checkpoint selection protocol；
- 相同 official metric。

唯一变量：

\[
\boxed{OSMF@H28_1}.
\]

---

# 29. Formal Primary Metrics

主指标：

\[
mIoU
\]

和：

\[
mDice.
\]

同时报告：

\[
IoU_{C0-C3}
\]

以及所有与 baseline 的：

\[
\Delta.
\]

不得因为 OSMF 对某个 diagnostic metric 有改善就替代正式 mIoU。

---

# 30. Specialization Diagnostics

这是 OSMF 特别重要的一部分。

只有涨点，没有证明 specialization：

\[
\boxed{\text{不够。}}
\]

需要至少验证下面四类诊断。

---

# 31. D1 — Semantic Validation

对 semantic auxiliary head 报告：

```text
image-level classification loss
per-class classification accuracy/AUC if existing pipeline supports it
```

不新增 metric library 时，

至少报告：

\[
L_{sem,val}.
\]

目标：

\[
S
\]

确实保持 class semantics。

---

# 32. D2 — Morphological Equivariance

在 validation 上固定：

```text
identity
horizontal flip
vertical flip
```

计算：

\[
EqErr(S)
\]

和：

\[
EqErr(M).
\]

我们希望：

\[
\boxed{
EqErr(M)<EqErr(S)
}
\]

或者至少：

\[
M
\]

相对训练初始产生显著更好的 spatial correspondence。

---

# 33. D3 — Cross-Subspace Redundancy

记录：

\[
CrossCov(S,M)
\]

baseline initialization vs final。

要求最终：

\[
CrossCov_{final}
<
CrossCov_{initial}.
\]

若表现提升但：

\[
S,M
\]

仍高度同质，

需要在最终报告中标记：

```text
PERFORMANCE_GAIN_WITHOUT_CLEAR_SPECIALIZATION
```

不能直接声称 factorization 成功。

---

# 34. D4 — Boundary / Interior Diagnostic

仅 validation GT。

绝对不进入训练。

构造固定：

\[
3\text{-pixel boundary band}.
\]

计算：

```text
Boundary accuracy / IoU
Interior accuracy / IoU
```

并计算 feature spatial-gradient energy：

\[
G_F(x)
=
\|\nabla F(x)\|_2.
\]

定义：

\[
BSR(F)
=
\frac{
Mean_{boundary}(G_F)
}{
Mean_{interior}(G_F)+\epsilon
}.
\]

分别计算：

\[
BSR(S)
\]

和：

\[
BSR(M).
\]

期望 morphology：

\[
\boxed{
BSR(M)>BSR(S).
}
\]

这只作为 role diagnostic，

不是 GO/NOGO 的唯一性能依据。

---

# 35. OSMF-v1.0 的核心成功条件

真正理想结果不是简单：

\[
mIoU\uparrow.
\]

而是同时出现：

\[
\boxed{
Semantic\ discrimination\uparrow/\text{stable}
}
\]

\[
\boxed{
Morphological\ equivariance\uparrow
}
\]

\[
\boxed{
Semantic-Morphology\ redundancy\downarrow
}
\]

以及：

\[
\boxed{
Final\ segmentation\uparrow.
}
\]

这样我们才能说：

> improvement comes together with structured specialization.

---

# 36. Phase 2 Formal Decision

所有 delta 相对于同 protocol 的 SSHR seed42 baseline。

---

## STRONG GO

满足：

\[
\boxed{
\Delta mIoU\ge+0.50\text{ pp}
}
\]

且：

- mDice positive；
- 至少 3/4 classes non-negative；
- specialization diagnostics 明确成立；
- training overhead ≤40%；
- no instability。

输出：

```text
OSMF_STRONG_GO
```

下一步允许扩展到其它 hierarchy。

---

## GO

若：

\[
\boxed{
\Delta mIoU\ge+0.30\text{ pp}
}
\]

且 specialization diagnostics 至少三项方向正确，

输出：

```text
OSMF_GO
```

下一步补 3-seed。

---

## WEAK REVIEW

若：

\[
+0.10
\le
\Delta mIoU
<
+0.30
\]

但：

\[
specialization
\]

非常清楚，

输出：

```text
OSMF_WEAK_REVIEW
```

不自动调 lambda。

人工判断是否值得进一步做 representation-level refinement。

---

## MECHANISM-ONLY

如果：

\[
\Delta mIoU<+0.10
\]

但 specialization 指标非常明显：

```text
OSMF_MECHANISM_ONLY
```

说明：

> representation 确实分开了，但这种分工暂时没有转化为最终 segmentation performance。

不能因为机制漂亮继续无条件扩展。

需要人工判断 fusion/readout 是否成为新的 bottleneck。

---

## NOGO

如果：

\[
\Delta mIoU<+0.10
\]

且：

- specialization weak；
- 或 collapse；
- 或 reconstruction unstable；
- 或训练明显恶化；

输出：

```text
OSMF_NOGO
```

结束 OSMF-v1 路线。

禁止立即上：

```text
Region Tokens
MoE
Graph
Prototype
Attention
更复杂 morphology loss
```

去救它。

---

# 37. 如果 Seed42 GO

才允许进入：

\[
\boxed{\text{3-seed confirmation}}
\]

沿用 SSHR 当前正式 seeds/protocol。

最终比较：

\[
SSHR
\]

vs：

\[
SSHR+OSMF.
\]

要求：

\[
\Delta mean\ mIoU\ge+0.30
\]

且：

\[
2/3
\]

以上 seeds 为正。

理想：

\[
3/3
\]

正。

---

# 38. 如果 OSMF 最终成功，下一步才考虑的 Innovation 2

禁止现在实现。

下一阶段候选：

# Cross-Hierarchy Shared–Unique Representation

从：

\[
S_i,M_i
\]

进一步发展：

\[
Shared_i
\]

和：

\[
Unique_i.
\]

目标：

\[
Shared_i
\approx Shared_j
\]

学习跨 hierarchy 的共同 tissue semantics。

而：

\[
Unique_i
\]

保留：

\[
\boxed{
\text{task-relevant hierarchy-specific morphology}.
}
\]

最终：

\[
F_i
=
Shared_i
+
Unique_i.
\]

这是 Phase 2，

不是 OSMF-v1.0。

---

# 39. 如果继续成功，模型才可能进一步发展成

```text
Image
  ↓
Shared Backbone
  ↓
Hierarchical Features
  ↓
Semantic Rectification
  ↓
Objective-Induced Factorization
  ├── Semantic Representation
  └── Morphology Representation
            ↓
     Region Morphology Tokens
            ↓
Structured Complementary Fusion
            ↓
Segmentation
```

这时：

SSHR 的 HFRM 只是：

\[
\boxed{
semantic rectification component
}
\]

而整个：

\[
\boxed{
Structured Complementary Representation Framework
}
\]

才是我们的模型主体。

---

# 40. 代码范围

建议新建：

```text
research/osmf-v1
```

主要新增：

```text
network/
└── osmf.py
```

核心类：

```python
class OSMFFactorizer(nn.Module):
    ...
```

推荐接口：

```python
h_hat, aux = osmf(h28_1)
```

其中：

```python
aux = {
    "semantic": S,
    "morphology": M,
    "reconstruction": h_hat
}
```

训练阶段额外返回 auxiliary tensors。

inference：

只需要：

```python
h_hat = osmf.forward_inference(h28_1)
```

---

# 41. 日志要求

每 epoch 保存：

```text
loss_total
loss_sshr
loss_sem
loss_eq
loss_orth
loss_rec

semantic_rms
morphology_rms
reconstruction_cosine
cross_covariance

CAM28_1 val mIoU
official fusion val mIoU
per-class IoU
```

每个正式 run 生成：

```text
osmf_training_report.md
```

---

# 42. 必须保存的模型

至少：

```text
last.pth
best_val.pth
```

选择规则完全沿用当前 frozen SSHR research protocol。

禁止：

> 根据 specialization diagnostic 选择 checkpoint。

checkpoint selection 仍只由预先冻结的 validation protocol 决定。

---

# 43. 必须的最小 ablation

只在 OSMF seed42 获得 GO 后进行。

否则不跑。

正式 ablation：

### A0

```text
SSHR
```

### A1

```text
SSHR + factorization + reconstruction
```

### A2

```text
A1 + semantic objective
```

### A3

```text
A2 + morphology equivariance
```

### A4

```text
A3 + orthogonality
= Full OSMF
```

目的：

判断真正有效的是：

- mere extra parameters；
- semantic specialization；
- morphology specialization；
- redundancy suppression；

中的哪一部分。

---

# 44. 参数量与运行成本

报告：

```text
baseline parameters
OSMF parameters
parameter delta
baseline FLOPs if available
OSMF FLOPs
training time / epoch
inference time / image
GPU peak memory
```

第一版目标：

\[
\boxed{
parameter\ overhead<5\%
}
\]

\[
\boxed{
inference\ overhead<10\%
}
\]

训练因 second-view：

\[
\boxed{
training\ overhead\le40\%
}
\]

超过目标不自动判死刑，

但必须人工 review。

---

# 45. 研究论文层面的最终动机

如果 OSMF 成功，

论文问题不应该写成：

> We add an orthogonal branch to SSHR.

而应该写：

> Existing hierarchical rectification methods implicitly assume that feature disagreement primarily reflects semantic noise. However, shallow and intermediate representations also contain morphology-specific information that is not predictable from deep semantics. Forcing these heterogeneous signals into a single latent representation may entangle semantic discrimination with spatial morphology modeling.

然后提出：

\[
\boxed{
\text{Objective-Induced Semantic–Morphology Factorization}
}
\]

其核心主张：

> Semantic identity and local morphology should not merely be extracted at different depths; they should be explicitly shaped by different learning objectives.

---

# 46. 本方案最重要的 falsifiable hypothesis

OSMF-v1.0 必须回答：

\[
\boxed{
\textbf{
Does objective-induced specialization produce
better segmentation than an entangled SSHR representation?
}
}
\]

只有：

\[
YES
\]

才有资格继续发展：

- shared/unique decomposition；
- region tokens；
- geometry experts；
- structured fusion。

如果：

\[
NO
\]

则停止这条 representation factorization 路线，

重新寻找完全不同的模型原则。

---

# 47. Codex Master Prompt

现在实现一个新的研究候选：

**OSMF-v1.0 — Objective-Induced Semantic–Morphology Factorization**

目标不是立即开发完整新模型。

目标是严格验证：

> 将 SSHR 的单一 entangled representation，通过不同训练目标显式分解为 semantic representation 和 morphology representation，是否能够产生可观测 specialization 并提高 segmentation。

## Scope

只修改：

```text
post-HFRM H28_1
```

其它：

```text
CAM56
CAM28_2
CAMdeep
official fusion
thresholds
metric
TTA
dataset
optimizer
scheduler
```

全部保持 frozen protocol。

不要实现 router。

不要实现 prototype。

不要实现 frequency module。

不要实现 uncertainty。

不要实现 graph。

不要实现 region token。

不要使用 segmentation GT 训练。

## Factorization

Given:

```text
H = post-HFRM H28_1
```

construct:

```text
S = P_sem(H)
M = P_morph(H)

H_hat =
U_sem(S) +
U_morph(M)
```

Use complementary channel-partition initialization such that:

```text
H_hat == H
```

at initialization.

Require:

```text
max_abs(H_hat - H) < 1e-6
```

and exact inference parity before training.

## Semantic objective

```text
z_sem = GAP(S)
logits_sem = Linear(z_sem)
L_sem = original image-level classification loss
```

No MLP.

## Morphology objective

On every fourth training step:

```text
view_a = photometric_aug(x)
view_b = photometric_aug(geometric_flip(x))
```

Run both through the same model.

Extract morphology representations:

```text
M_a
M_b
```

inverse-align `M_b`.

L2-normalize features channel-wise.

Use spatial cosine equivariance:

```text
L_eq =
mean(
    1 - cosine(
        M_a,
        inverse_aligned(M_b)
    )
)
```

Geometric transform is only:

```text
horizontal flip
vertical flip
```

No arbitrary rotation/crop/elastic deformation.

## Orthogonality

Standardize S and M over batch × spatial tokens.

Compute:

```text
C_sm = S.T @ M / N
```

and:

```text
L_orth =
||C_sm||_F^2 / (Cs * Cm)
```

## Reconstruction

```text
target = H.detach()
```

Use normalized cosine reconstruction:

```text
L_rec =
1 - cosine(H_hat, target)
```

The reconstruction target must not propagate gradient into H.

## Total loss

Keep original SSHR loss unchanged.

Add:

```text
lambda_sem   = 0.20
lambda_morph = 0.20
lambda_orth  = 0.05
lambda_rec   = 0.10
```

No validation tuning.

## Phase -1

Implement only.

Verify exact baseline parity.

If parity fails:

STOP.

## Phase 0

Run 128 training batches.

Audit:

```text
all tensors finite
all losses finite
all gradients finite
gradient ratios
semantic/morph RMS
reconstruction cosine
cross covariance
```

If any weighted auxiliary gradient exceeds:

```text
0.50 * baseline gradient norm
```

STOP FOR REVIEW.

Do not tune lambda automatically.

## Phase 1

Run:

```text
BCSS
seed42
3 epochs
```

Only mechanism diagnosis.

Require:

```text
L_sem decreases
L_eq decreases >=10%
L_orth decreases >=10%
reconstruction cosine >=0.95
no representation collapse
```

If failed:

```text
OSMF_MECHANISM_NOGO
```

STOP.

If passed:

```text
OSMF_MECHANISM_GO
```

STOP and wait for review before formal 25-epoch training.

## Formal 25-epoch experiment

Only after human authorization.

Use exactly the existing SSHR experiment protocol.

Primary metric:

```text
validation mIoU
```

plus:

```text
mDice
per-class IoU
```

## Specialization diagnostics

Validation GT may be used for diagnosis only.

Report:

```text
semantic classification behavior
semantic/morph equivariance
cross-subspace covariance
3-pixel boundary-band metrics
interior metrics
boundary sensitivity ratio
```

Do not train on any segmentation GT.

## Decision

STRONG GO:

```text
mIoU >= baseline +0.50 pp
clear specialization
>=3/4 classes non-negative
stable training
```

GO:

```text
mIoU >= baseline +0.30 pp
at least 3 specialization diagnostics correct
```

WEAK REVIEW:

```text
+0.10 <= mIoU delta < +0.30
with clear specialization
```

MECHANISM ONLY:

```text
mIoU delta < +0.10
but specialization is clear
```

NOGO:

```text
mIoU delta < +0.10
and specialization is weak/unstable
```

After every phase:

STOP.

Do not automatically continue to the next stage.

Do not run test.

Do not run LUAD.

Do not add new modules.

Wait for human scientific review.