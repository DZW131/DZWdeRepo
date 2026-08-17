# OSMF-v1.0 Phase 0：128-Batch Structural & Gradient Audit
## 冻结结构与梯度可行性审查最终技术方案 v1.0

**项目：** 基于 SSHR 的 Objective-Induced Semantic–Morphology Factorization（OSMF）  
**阶段：** Phase 0 — Structural / Gradient / Early-Specialization Audit  
**前置状态：** `OSMF_PHASE_MINUS1_PASS`  
**本阶段性质：** 仅 128 个真实 BCSS training batches 的机制审查，不做正式训练，不评价最终性能  
**允许的数据：** BCSS train；仅在需要 parity/reference 时读取 BCSS validation，禁止 test / LUAD  
**核心原则：**

\[
\boxed{
\text{先证明 OSMF 在真实训练梯度下“安全且会动”，再允许进入 3-epoch pilot}
}
\]

---

# 1. 前置事实与阶段目标

Phase -1 已完成并冻结：

- OSMF 插入位置：post-HFRM `H28_1`
- 输入通道：512
- semantic / morphology：256 / 256
- 初始化满足严格 identity reconstruction
- random input 与 real validation input 上：
  - `H_hat == H`
  - 四个 CAM 完全一致
  - classification probability 完全一致
- BCSS validation 3418 张：
  - differing pixels = 0
  - mIoU difference = 0
  - mDice difference = 0
- 参数增量仅约 `+0.4661%`
- 未执行任何 optimization step

因此 Phase 0 不再检查“代码能否插进去”，而要回答：

> **当真实训练梯度开始作用时，semantic objective、morphology equivariance、orthogonality 和 reconstruction 是否能够在不破坏 SSHR 原表示的前提下，共同驱动两个子空间形成健康的早期 specialization？**

---

# 2. 本阶段只回答五个问题

## Q1 — Auxiliary gradients 是否处于安全量级？

对于：

\[
L_{sem},\quad L_{eq},\quad L_{orth},\quad L_{rec}
\]

其加权梯度相对于原 SSHR 主损失：

\[
L_{SSHR}
\]

是否过强、过弱或数值异常？

## Q2 — Semantic / Morphology 两条路径是否都真正获得有效梯度？

必须排除：

- semantic branch 无梯度；
- morphology branch 无梯度；
- 某个 projector / reconstructor 没有参数更新；
- morphology loss 只在 tensor 上变化但不能影响模型。

## Q3 — 两个子空间是否出现 collapse？

必须监控：

\[
RMS(S),\quad RMS(M)
\]

避免：

\[
S\rightarrow0
\]

或：

\[
M\rightarrow0.
\]

## Q4 — Decorrelation 是否是真 specialization，而不是 branch death？

若：

\[
L_{orth}\downarrow
\]

但同时：

\[
RMS(M)\rightarrow0,
\]

则这是伪成功。

必须联合观察：

\[
CrossCov(S,M)
\]

与：

\[
RMS(S),RMS(M).
\]

## Q5 — OSMF 是否在保持原 SSHR 表示稳定的同时开始形成 objective-induced role separation？

重点观察：

\[
Cos(H,\hat H)
\]

与 morphology equivariance：

\[
EqErr(M).
\]

理想状态：

\[
Cos(H,\hat H)\text{ 仍然高}
\]

同时：

\[
EqErr(M)\downarrow.
\]

---

# 3. Frozen Contract

## 3.1 基线

继续使用 Phase -1 已验证的：

```text
baseline commit:
4e9a2887b220d17e27649d72a3d13f32b7ebe8f9
```

OSMF implementation commit：

```text
5eb7b258f0cdeb4fa8779b65e716c105c9541f9a
```

checkpoint：

```text
/home/duyanhong/sshr-official-25ep-final-retry2-20260815/
runs/bcss_seed42/checkpoints/stage1_last.pth
```

SHA256：

```text
509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579
```

## 3.2 建议新分支

```text
research/osmf-v1-phase0
```

从 Phase -1 已通过 commit 创建。

## 3.3 绝对禁止

本阶段禁止：

```text
NO BCSS test
NO LUAD
NO 3-epoch pilot
NO 25-epoch training
NO other seeds

NO threshold changes
NO TTA changes
NO metric changes
NO fusion changes
NO HFRM changes
NO CAM-head changes

NO lambda tuning
NO learning-rate tuning
NO architecture tuning
NO channel-ratio tuning
NO equivariance-interval tuning

NO prototype
NO router
NO attention
NO frequency branch
NO uncertainty gate
NO region token
NO graph module
NO boundary supervision
NO segmentation GT in training
```

本阶段只能：

```text
run exactly 128 real BCSS training batches
collect losses
collect gradients
collect representation statistics
collect parameter-update statistics
write report
STOP
```

---

# 4. OSMF 结构冻结

输入：

\[
H=H_{28_1}^{post-HFRM}
\in\mathbb R^{512\times h\times w}
\]

分解：

\[
S=P_{sem}(H),\qquad M=P_{morph}(H)
\]

其中：

\[
C_s=C_m=256.
\]

重构：

\[
\hat H
=
U_{sem}(S)
+
U_{morph}(M).
\]

原 SSHR `ic1` CAM head 继续接收：

\[
\hat H.
\]

其它三个 hierarchy 完全不变。

---

# 5. Loss 冻结

总 loss：

\[
L
=
L_{SSHR}
+
\lambda_{sem}L_{sem}
+
\lambda_{morph}L_{eq}
+
\lambda_{orth}L_{orth}
+
\lambda_{rec}L_{rec}.
\]

固定：

\[
\lambda_{sem}=0.20,\quad
\lambda_{morph}=0.20,\quad
\lambda_{orth}=0.05,\quad
\lambda_{rec}=0.10.
\]

禁止根据本阶段结果自动调整。

---

# 6. Morphology Equivariance 频率冻结

固定：

```text
equivariance_interval = 4
```

即每四个 training steps，仅一个 step 计算 second-view morphology equivariance。

Geometric transforms 仅允许：

```text
horizontal flip
vertical flip
```

禁止：

```text
arbitrary rotation
crop-resize warp
elastic deformation
```

---

# 7. 训练批次数与随机性

固定：

```text
num_real_batches = 128
seed = 20260817
```

使用原 SSHR train dataloader / optimizer / scheduler 逻辑。

不得为了 Phase 0 单独设计 optimizer。

若原 protocol 需要恢复 optimizer state，则按现有 frozen research protocol执行；若 Phase 0 以 A0 checkpoint 为初始化进行新实验训练，则必须在报告中明确记录 optimizer 初始化方式。

---

# 8. Phase 0A — Start-State Exact Check

正式 batch 1 前重新检查：

\[
\max |\hat H-H|<10^{-6}.
\]

并抽取至少一个真实 train batch：

```text
all tensors finite
all CAMs finite
all losses finite
```

如果 start-state parity 已丢失：

```text
OSMF_PHASE0_NOGO
```

立即 STOP。

---

# 9. Phase 0B — Gradient Decomposition Audit

这是本阶段最重要的部分。

对固定 audit steps：

```text
steps = 1, 2, 4, 8, 16, 32, 64, 96, 128
```

分别测量各 objective 对目标 representation / OSMF 参数的梯度。

---

# 10. 基础梯度

定义：

\[
g_{base}
=
\left\|
\nabla_{H}
L_{SSHR}
\right\|_2.
\]

同时建议额外记录 OSMF 参数层面：

\[
g_{base}^{\theta}
=
\left\|
\nabla_{\theta_{OSMF}}
L_{SSHR}
\right\|_2.
\]

---

# 11. Auxiliary Gradient Norms

分别独立 backward / `torch.autograd.grad` 计算：

\[
g_{sem}
=
\left\|
\nabla_H L_{sem}
\right\|_2
\]

\[
g_{eq}
=
\left\|
\nabla_H L_{eq}
\right\|_2
\]

\[
g_{orth}
=
\left\|
\nabla_H L_{orth}
\right\|_2
\]

\[
g_{rec}
=
\left\|
\nabla_H L_{rec}
\right\|_2.
\]

然后计算加权 ratio：

\[
r_{sem}
=
\frac{
\lambda_{sem}g_{sem}
}{
g_{base}+\epsilon
}
\]

\[
r_{eq}
=
\frac{
\lambda_{morph}g_{eq}
}{
g_{base}+\epsilon
}
\]

\[
r_{orth}
=
\frac{
\lambda_{orth}g_{orth}
}{
g_{base}+\epsilon
}
\]

\[
r_{rec}
=
\frac{
\lambda_{rec}g_{rec}
}{
g_{base}+\epsilon
}.
\]

---

# 12. 梯度安全区

推荐健康区：

\[
0.02
\le
r_j
\le
0.30.
\]

不是所有 loss 都必须严格落在此区间，但需要综合解释。

### Hard Review Zone

若：

\[
0.30<r_j\le0.50
\]

持续出现：

```text
OSMF_PHASE0_REVIEW
```

### Hard Stop

若：

\[
r_j>0.50
\]

持续出现或伴随明显训练破坏：

```text
OSMF_PHASE0_NOGO
```

立即 STOP。

不得自动降低 lambda。

---

# 13. Gradient Direction Audit

除了 norm，还需要看 auxiliary gradient 是否与主任务强冲突。

对：

\[
g_{base}
\]

和每个：

\[
g_j
\]

计算 cosine：

\[
c_j
=
\frac{
\langle g_{base},g_j\rangle
}{
\|g_{base}\|
\|g_j\|+\epsilon
}.
\]

报告：

```text
cos(base, sem)
cos(base, eq)
cos(base, orth)
cos(base, rec)
```

解释：

- \(c>0\)：方向大体协同
- \(c\approx0\)：近似独立
- \(c<0\)：存在冲突

本阶段不因单次负 cosine 直接 NOGO。

但若：

\[
Mean(c_j)<-0.5
\]

且：

\[
r_j>0.30,
\]

则标记：

```text
STRONG_GRADIENT_CONFLICT
```

并输出：

```text
OSMF_PHASE0_REVIEW
```

或更严重时 NOGO。

禁止使用 PCGrad 等方法自动修复。

---

# 14. Parameter-Level Gradient Coverage

对 OSMF 新参数分别记录：

```text
p_sem.weight
p_morph.weight
u_sem.weight
u_morph.weight
semantic_classifier.weight
semantic_classifier.bias
```

统计：

```text
grad_norm
nonzero_grad_fraction
parameter_update_norm
relative_update_norm
```

定义：

\[
RelativeUpdate
=
\frac{
\|\theta_{128}-\theta_0\|_2
}{
\|\theta_0\|_2+\epsilon
}.
\]

必须确认：

- semantic projector 在动；
- morphology projector 在动；
- semantic classifier 在动；
- reconstruction projections 在动。

若某个关键路径 128 batches 后仍近似零更新：

```text
DEAD_PATH_WARNING
```

若 semantic 或 morphology 主路径完全无有效梯度：

```text
OSMF_PHASE0_NOGO
```

---

# 15. Phase 0C — Representation Health Audit

每个 audit step 记录：

\[
RMS(S),\quad RMS(M),\quad RMS(\hat H),\quad RMS(H).
\]

定义：

\[
R_{SM}
=
\frac{
RMS(S)
}{
RMS(M)+\epsilon
}.
\]

---

# 16. Branch Collapse 判据

健康区不要求：

\[
R_{SM}=1.
\]

### Review

如果：

\[
R_{SM}>10
\]

或：

\[
R_{SM}<0.10
\]

持续多个 audit step：

```text
BRANCH_IMBALANCE_WARNING
OSMF_PHASE0_REVIEW
```

### NOGO

如果：

\[
R_{SM}>20
\]

或：

\[
R_{SM}<0.05
\]

并持续或伴随某一路 RMS 持续下降：

```text
BRANCH_COLLAPSE
OSMF_PHASE0_NOGO
```

---

# 17. Reconstruction Stability

记录：

\[
CosRec
=
\cos(H,\hat H).
\]

初始：

\[
CosRec=1.
\]

### GO target

128 batches 结束时：

\[
\boxed{
CosRec\ge0.95
}
\]

### REVIEW

若：

\[
0.90\le CosRec<0.95
\]

输出：

```text
OSMF_PHASE0_REVIEW
```

### NOGO

若：

\[
CosRec<0.90
\]

或出现持续快速下降：

```text
RECONSTRUCTION_DESTABILIZED
OSMF_PHASE0_NOGO
```

---

# 18. Reconstruction Residual Magnitude

额外计算：

\[
Residual
=
\hat H-H.
\]

报告：

\[
RR
=
\frac{
RMS(\hat H-H)
}{
RMS(H)+\epsilon
}.
\]

记录：

```text
start
mean
end
max
```

不设独立 GO threshold，但必须和 `CosRec` 联合解释。

---

# 19. Phase 0D — Cross-Subspace Redundancy Audit

对 standardized：

\[
S,M
\]

计算：

\[
C_{SM}
=
\frac1N\tilde S^T\tilde M.
\]

定义：

\[
CrossCov
=
\frac{
\|C_{SM}\|_F^2
}{
C_sC_m
}.
\]

记录：

```text
CrossCov_start
CrossCov_mean
CrossCov_end
CrossCov_min
CrossCov_max
```

目标不是要求 128 batches 内大幅下降。

只要求：

- 不爆炸；
- 不靠 branch collapse 下降；
- 有合理下降趋势则记为 positive evidence。

---

# 20. Decorrelation Validity

若：

\[
CrossCov\downarrow
\]

且：

\[
RMS(S),RMS(M)
\]

保持健康：

```text
GENUINE_DECORRELATION_SIGNAL
```

若 CrossCov 下降但某一路 collapse：

```text
FALSE_DECORRELATION_BY_COLLAPSE
OSMF_PHASE0_NOGO
```

---

# 21. Phase 0E — Early Equivariance Audit

定义 inverse-aligned morphology equivariance error：

\[
EqErr_M
=
Mean_x
\left[
1-
\cos(
\bar M(x),
T^{-1}\bar M(Tx)
)
\right].
\]

每个 equivariance audit step 记录。

建议同时计算一个**不参与训练**的 semantic reference：

\[
EqErr_S.
\]

`EqErr_S` 只能诊断，不能作为 loss。

---

# 22. 早期 specialization 迹象

理想趋势：

\[
EqErr_M^{end}
<
EqErr_M^{start}.
\]

本 Phase 0 不要求达到 10% 降幅。

Phase 0 只要求：

- `L_eq` finite；
- `EqErr_M` 能响应训练；
- morphology parameters 获得有效梯度；
- 无数值 collapse。

如果：

\[
EqErr_M
\]

完全不动，且 morphology gradient 近零：

```text
MORPHOLOGY_OBJECTIVE_INACTIVE
OSMF_PHASE0_NOGO
```

---

# 23. Semantic Branch Early Audit

记录：

\[
L_{sem}.
\]

同时记录 semantic classifier：

```text
logit mean
logit std
probability mean
classification gradient norm
```

不要求 128 batches 内 validation classification 提升。

只要求：

- finite；
- classifier 不 collapse；
- `L_sem` 有正常波动/初步下降；
- semantic projector 获得有效梯度。

若 semantic auxiliary classifier 输出几乎常数，且梯度长期近零：

```text
SEMANTIC_PATH_INACTIVE
OSMF_PHASE0_NOGO
```

---

# 24. Main SSHR Loss Stability

必须记录：

\[
L_{SSHR}
\]

的：

```text
start
mean
end
min
max
```

如：

- `L_SSHR` 爆炸；
- NaN；
- 连续异常上升；
- CAM logits 数值失控；

直接：

```text
OSMF_PHASE0_NOGO
```

---

# 25. Training-Time Cost

记录：

```text
mean iteration time
equivariance iteration time
non-equivariance iteration time
peak GPU memory
```

估算相对原 SSHR：

\[
TrainingOverhead.
\]

目标：

\[
\le40\%.
\]

如果：

\[
>40\%
\]

只标记：

```text
COST_REVIEW
```

不单独 NOGO。

不得改变 `equivariance_interval=4`。

---

# 26. Audit Sampling

详细记录：

```text
0
1
2
4
8
16
32
64
96
128
```

每个 audit point 保存：

```text
losses
weighted gradient ratios
gradient cosines
branch RMS
reconstruction cosine
residual ratio
cross covariance
EqErr_M
EqErr_S
parameter grad norms
parameter relative updates
iteration time
GPU memory
```

---

# 27. 输出目录

```text
/home/duyanhong/experiments/
OSMF_V1_PHASE0_128B_<commit>/
```

结构：

```text
config/
  frozen_contract.json
  environment.json

tables/
  loss_trace.csv
  gradient_ratio.csv
  gradient_cosine.csv
  parameter_gradient_coverage.csv
  parameter_update.csv
  representation_health.csv
  reconstruction.csv
  redundancy.csv
  equivariance.csv
  compute_cost.csv

figures/
  loss_curves.png
  gradient_ratio_curves.png
  gradient_cosine_curves.png
  branch_rms.png
  reconstruction_cosine.png
  cross_covariance.png
  equivariance_error.png
  parameter_update.png

docs/
  osmf_phase0_128batch_audit.md
```

---

# 28. Required Main Table

最终报告必须包含：

| Metric | Start | Mean | End | Min | Max | Status |
|---|---:|---:|---:|---:|---:|---|
| \(L_{SSHR}\) | | | | | | |
| \(L_{sem}\) | | | | | | |
| \(L_{eq}\) | | | | | | |
| \(L_{orth}\) | | | | | | |
| \(L_{rec}\) | | | | | | |
| \(r_{sem}\) | | | | | | |
| \(r_{eq}\) | | | | | | |
| \(r_{orth}\) | | | | | | |
| \(r_{rec}\) | | | | | | |
| cos(base, sem) | | | | | | |
| cos(base, eq) | | | | | | |
| cos(base, orth) | | | | | | |
| cos(base, rec) | | | | | | |
| RMS(S) | | | | | | |
| RMS(M) | | | | | | |
| RMS(S)/RMS(M) | | | | | | |
| Cos(H,H_hat) | | | | | | |
| Residual Ratio | | | | | | |
| CrossCov(S,M) | | | | | | |
| EqErr(M) | | | | | | |
| EqErr(S) | | | | | | |

---

# 29. Parameter Health Table

| Parameter | Grad nonzero? | Mean grad norm | End relative update | Status |
|---|---:|---:|---:|---|
| `p_sem.weight` | | | | |
| `p_morph.weight` | | | | |
| `u_sem.weight` | | | | |
| `u_morph.weight` | | | | |
| `semantic_classifier.weight` | | | | |
| `semantic_classifier.bias` | | | | |

---

# 30. GO 判定

输出：

```text
OSMF_PHASE0_GO
```

必须满足全部硬条件：

1. 所有 tensor finite；
2. 所有 loss finite；
3. 所有关键 gradient finite；
4. semantic path 有有效梯度；
5. morphology path 有有效梯度；
6. factorizer / reconstructor 参数均发生有效更新；
7. 任一 weighted auxiliary gradient ratio 无持续：
   \[
   >0.50
   \]
8. 128 batch 结束：
   \[
   Cos(H,\hat H)\ge0.95
   \]
9. 无 branch collapse：
   \[
   0.05<RMS(S)/RMS(M)<20
   \]
10. CrossCov 无数值爆炸；
11. `L_eq` / `EqErr_M` 对训练有响应；
12. 原 `L_SSHR` 无明显异常破坏。

GO 只意味着：

> **可以人工授权进入 3-epoch mechanism pilot。**

不授权自动继续。

---

# 31. REVIEW 判定

输出：

```text
OSMF_PHASE0_REVIEW
```

若出现任一：

- 持续：
  \[
  0.30<r_j\le0.50
  \]
- 128 batch 末：
  \[
  0.90\le Cos(H,\hat H)<0.95
  \]
- 持续 branch imbalance：
  \[
  10<R_{SM}<20
  \]
  或
  \[
  0.05<R_{SM}<0.10
  \]
- strong gradient conflict：
  \[
  Mean[\cos(g_{base},g_j)]<-0.5
  \]
  且：
  \[
  r_j>0.30
  \]
- training overhead >40%
- objective 在变化但对应路径几乎没有有效 parameter update。

REVIEW 后：

\[
\boxed{STOP}
\]

禁止 Codex 自动调参数。

---

# 32. NOGO 判定

输出：

```text
OSMF_PHASE0_NOGO
```

若出现：

- NaN / Inf；
- weighted auxiliary gradient 持续 \(>0.50\times\) base；
- semantic path dead；
- morphology path dead；
- 任一主 factorization branch collapse；
- `Cos(H,H_hat) < 0.90`；
- false decorrelation by branch collapse；
- 原 SSHR loss 明显爆炸；
- CAM / feature 数值失控；
- OSMF 参数关键路径没有有效梯度；
- second-view equivariance 根本不能反向更新 morphology path。

---

# 33. 本阶段禁止使用 mIoU 做 GO/NOGO

Phase 0 只有：

\[
128\ batches.
\]

因此：

\[
\boxed{
\text{不要根据 validation mIoU 判断机制是否成功。}
}
\]

可以在报告中附：

```text
diagnostic-only current val mIoU
```

但：

- 不要求跑完整 validation；
- 不用作 GO/NOGO；
- 不根据它调整任何参数。

本阶段核心是：

\[
\boxed{
\text{mechanism safety + gradient health + early role emergence}
}
\]

而不是性能。

---

# 34. Phase 0 通过后的下一步

只有人工确认：

```text
OSMF_PHASE0_GO
```

后，才允许进入：

# OSMF Phase 1 — 3-Epoch Mechanism Pilot

Phase 1 才正式要求：

\[
L_{sem}\downarrow
\]

\[
L_{eq}\downarrow\ge10\%
\]

\[
L_{orth}\downarrow\ge10\%
\]

\[
Cos(H,\hat H)\ge0.95
\]

并开始观察：

\[
CAM28_1
\]

和：

\[
official fusion
\]

的 validation behavior。

本 Phase 0 不得自动进入 Phase 1。

---

# 35. Codex Master Prompt

你现在执行：

**OSMF-v1.0 Phase 0 — 128-Batch Structural & Gradient Audit**

这是一个机制安全审查，不是正式训练。

前置状态：

```text
OSMF_PHASE_MINUS1_PASS
```

OSMF 已在 `post-HFRM H28_1` 完成 exact identity initialization，并在 3418 张 BCSS validation 上实现：

```text
differing pixels = 0
mIoU difference = 0
mDice difference = 0
```

本阶段唯一目标：

> 验证真实训练梯度下，semantic / morphology specialization objectives 是否能够安全、有效地作用于 OSMF，而不导致梯度支配、representation collapse 或 SSHR 表示快速破坏。

## Fixed

```text
num_real_batches = 128
seed = 20260817

lambda_sem   = 0.20
lambda_morph = 0.20
lambda_orth  = 0.05
lambda_rec   = 0.10

equivariance_interval = 4
```

不得调整。

## Run only 128 real BCSS training batches

Do NOT:

```text
run test
run LUAD
run 3 epochs
run 25 epochs
change any lambda
change lr
change architecture
change channel split
change fusion
change thresholds
change TTA
add new module
```

## Audit steps

Collect detailed diagnostics at:

```text
0,1,2,4,8,16,32,64,96,128
```

## Losses

Record:

```text
L_SSHR
L_sem
L_eq
L_orth
L_rec
L_total
```

All must remain finite.

## Weighted gradient ratios

For each auxiliary objective compute gradient norm with respect to `H28_1` and preferably also OSMF parameters.

Compute:

```text
r_sem
r_eq
r_orth
r_rec
```

where:

```text
r_j =
lambda_j * ||grad_j||
/
(||grad_SSHR|| + eps)
```

Healthy reference:

```text
0.02 <= r_j <= 0.30
```

Review:

```text
0.30 < r_j <= 0.50
```

Hard stop:

```text
r_j > 0.50
```

if persistent or destabilizing.

Do not tune lambda automatically.

## Gradient direction

Compute cosine similarity between the original SSHR gradient and each auxiliary gradient:

```text
cos(base, sem)
cos(base, eq)
cos(base, orth)
cos(base, rec)
```

If mean cosine is below -0.5 while the same auxiliary ratio exceeds 0.30, mark strong gradient conflict and return REVIEW unless clear instability requires NOGO.

Do not add PCGrad or any gradient surgery.

## Parameter health

For:

```text
p_sem.weight
p_morph.weight
u_sem.weight
u_morph.weight
semantic_classifier.weight
semantic_classifier.bias
```

report:

```text
grad norm
non-zero gradient fraction
absolute update norm
relative update norm
```

Semantic and morphology paths must both receive effective gradients.

## Representation health

Report:

```text
RMS(H)
RMS(S)
RMS(M)
RMS(H_hat)
RMS(S)/RMS(M)
```

NOGO if branch collapse is clear:

```text
ratio > 20
or
ratio < 0.05
```

Review for sustained:

```text
ratio > 10
or
ratio < 0.10
```

## Reconstruction

Report:

```text
cos(H, H_hat)
RMS(H_hat - H) / RMS(H)
```

GO target at batch 128:

```text
cos(H,H_hat) >= 0.95
```

Review:

```text
0.90 <= cosine < 0.95
```

NOGO:

```text
cosine < 0.90
```

## Redundancy

Compute standardized cross-covariance:

```text
CrossCov(S,M)
```

Track start/mean/end/min/max.

If CrossCov decreases while both branch RMS values remain healthy:

```text
GENUINE_DECORRELATION_SIGNAL
```

If CrossCov decreases because one branch collapses:

```text
FALSE_DECORRELATION_BY_COLLAPSE
OSMF_PHASE0_NOGO
```

## Equivariance

Track morphology inverse-aligned equivariance error:

```text
EqErr_M
```

Also compute semantic `EqErr_S` for diagnosis only.

Phase 0 does not require a 10% drop yet.

It only requires:

```text
L_eq finite
EqErr_M responsive
morphology path receives gradient
```

If morphology objective cannot affect the morphology path:

```text
MORPHOLOGY_OBJECTIVE_INACTIVE
OSMF_PHASE0_NOGO
```

## Decision

Return exactly one:

```text
OSMF_PHASE0_GO
OSMF_PHASE0_REVIEW
OSMF_PHASE0_NOGO
```

GO means only：

> safe to request human authorization for the 3-epoch mechanism pilot.

After producing the report:

STOP.

Do not automatically start Phase 1.
