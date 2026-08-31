# RDDR Phase2B1.12：5090 → 4090 迁移交付

日期：2026-08-31。状态：**迁移完成，零步 CUDA/BF16 检查通过，正式实验尚未启动。**

本报告是迁移与工程兼容性记录，不是三臂 500-step 实验报告，不产生科学 GO/NO-GO 结论。

## 1. 已确认的边界

- 源服务器：`duyanhong@10.15.20.77:22`。
- 目标服务器：`duyanhong@10.15.20.149:54268`，RTX 4090 D。
- 大文件由源服务器 rsync/SSH 直接发送至目标服务器；本地只负责控制与接收小型验证记录。
- 源端数据、权重、环境、代码和历史实验全部保留，不停止其他用户的进程。
- 只搬当前 Phase2B1.12 所需资产，不迁移 BCSS test、LUAD 数据或其他旧实验。
- 不更改网络、训练、推理、指标、batch、随机种子、优化器、LR 或校准规则。
- 不执行正式训练、32-batch lambda 校准、完整 validation 或 test 评估。

## 2. 迁移清单与完整性

所有目录均保留原绝对路径。复制前逐项确认目标路径不存在；rsync 未使用删除选项，并设置不覆盖已有文件。

| 资产 | 目标路径（与原端一致） |
|---|---|
| 当前实验代码与 Git 历史 | `/home/duyanhong/DZWdeRepo-rddr-phase2b112` |
| Python 环境 | `/home/duyanhong/miniconda3/envs/sshr5090` |
| BCSS training / val | `/home/duyanhong/reseg-data/raw/BCSS-WSSS/{training,val}` |
| C0 权重 | `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth` |
| C0 来源记录 | 上述 run 根目录的 `environment.tsv`，以及 seed42 的 `status.tsv`、`train.log` |
| 原始启动脚本（仅作来源证据） | `/home/duyanhong/run_official_25ep_retry2.sh` |
| 冻结 native 缓存 | `/home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz` |

- 直接传输了 **78,484 个普通文件**；rsync 列表共 84,425 项，含目录、符号链接。
- 文件内容约 **11.49 GB**（十进制，rsync 显示），耗时 **296.87 秒**。
- 复制后执行第二次 `rsync --checksum --dry-run --itemize-changes`：**无差异项，退出码 0**，耗时 **23.13 秒**。此校验在目标端运行 Python 检查、添加迁移文档之前完成。
- 两端环境中的失效符号链接数量均为 **0**。
- 按真实 Dataset 解析：training **23,422**；val image/mask **3,418 / 3,418**，文件名逐一匹配。
- 冻结 native 缓存中的 3,418 个 image name 与 validation 名单完全匹配。
- 迁移结束后 `/home` 仍有约 **204 GB** 可用空间。

C0 SHA256：

`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`

Native 缓存 SHA256：

`767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a`

## 3. 环境与可复现性边界

| 项目 | 4090 实测 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D |
| 驱动 | 550.120，未升级 |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| NumPy | 1.23.5 |
| cuDNN | 91900（PyTorch 返回值） |
| CUDA 可用 / BF16 支持 | True / True |
| 唯一包名及版本清单 | 两端 111 项完全一致 |

环境是原前缀的完整复制，不是重新安装或重新解析依赖；**没有复制 Conda base 管理器**。环境名保留 `sshr5090` 只是为了保持绝对路径，不表示仍在 5090 运行。直接调用该环境的 Python，无需 `conda activate`。

元数据细节：原端枚举得到 112 条 distribution 记录，其中 `reseg-net==0.1.0` 重复；目标端 111 条。按唯一 `(name, version)` 比较完全一致，并非包版本不一致。

**数据内容及标签分布一致，但原 Dataset 使用未排序的文件系统遍历，两端初始遍历顺序不同。** 本次未修改原 Dataset 来强行排序。因此不宣称 seed42 的逐 batch 顺序或未来数值与 5090 位级一致。后续 B/A/R 都应在这台 4090 上执行，由原冻结 runner 保证三臂共享相同转换后 batch 与主网络 RNG；不能把一个臂放回 5090 后当成同环境因果对照。

这次实际通过所需 CUDA/BF16 路径，未据驱动版本号直接假定兼容；也不把单批次通过泛化为所有 CUDA 软件均兼容。

## 4. 验证结果

| 检查 | 结果 |
|---|---|
| 本地原 53 项 CPU 测试 | 53/53 PASS，12.364 秒 |
| 4090 环境重跑同一测试集 | 53/53 PASS，5.420 秒 |
| C0 来源与 SHA256 | PASS |
| C0 strict load | missing_keys=[]；unexpected_keys=[] |
| B/A/R 初始 CPU 模型状态 | 三臂位级相同 |
| 优化器来源 | 原 momentum=.0005；原四组最终 LR/WD；fresh、空 buffer |
| 显存准入 | PASS；保留原 18 GiB 门槛 |
| BF16 矩阵计算及反向 | 有限 |
| 真实 batch20、224×224、forward_cam | 四个 CAM 和 classification 输出均为 BF16 且有限 |
| 同一真实 batch 的 B/A/R 梯度路径 | 全部有限；三个 main loss 完全相同 |
| 权重 / BN 运行统计 | smoke 前后完全不变 |
| 真实模型 optimizer.step 次数 | **0** |
| 校准 / 正式训练 / validation 指标 / test | **均未执行** |

Smoke 使用一个临时模型依次检查 B/A/R backward，不是三个正式训练臂。它将辅助注入系数设为零，同时仍实际构建 ADT/RG 图并求出辅助梯度；该零值**不是实验 lambda**，没有替代或绕过未来的 32-batch 校准。

| 路径 | Main loss | 原始辅助 loss | 原始辅助梯度范数 | Gate fraction |
|---|---:|---:|---:|---:|
| B | 0.2256799042 | 0 | 0 | 0.3153698980（仅诊断） |
| A | 0.2256799042 | 1.2245320082 | 7.3503158214 | 0.3153698980 |
| R | 0.2256799042 | 1.0397270918 | 6.4631993533 | 0.3153698980 |

A/R 均实际求得指定 39 张量的辅助梯度；R 的数量由 A 提供。B 没有施加 gate 或辅助损失。表中的梯度只是兼容性证据，不是学习收益证据。

临时模型前后状态 SHA256 一致：

`c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5`

## 5. 资源实测

- 完整 smoke 用时 **11.09 秒**。
- Peak allocated：**3,494,405,120 bytes，约 3.25 GiB**。
- Peak reserved：**3,934,257,152 bytes，约 3.66 GiB**。
- 检查退出后 GPU 空闲显存 **24,200 MiB**，无计算进程。

这些是单批次、零 optimizer-step 检查的峰值，不能当成完整 500-step 运行峰值或耗时预测；正式运行仍按既有资源检查执行，不下调准入门槛。

## 6. 代码管理

- 冻结实验起点：`8a850a19e9cfb24e9230a2df083de20439f1e0f8`，来自待审查 PR #50。
- 本次独立分支：`feature/rddr-phase2b112-4090-migration`。
- 仅新增独立 migration smoke 工具、迁移证据和说明；原 `network/`、`tool/`、`train_sshr.py` 及全部既有 Phase2B1.12 执行/统计模块未修改。
- 本次 PR 以 `feature/rddr-phase2b112-short-horizon` 为 base，使差异只包含迁移交付。未自动合并任何 PR。
- 5090 工作区仍保留冻结提交；目标 4090 工作区切换至本次迁移分支。

## 7. 证据与命令

服务器工程检查目录：

`/home/duyanhong/experiments/MIGRATION_4090_20260831`

本地/仓库证据：

`audit/results/rddr_phase2b112_4090_migration/`

其中包括 `cuda_bf16_smoke.json`、CPU/preflight 记录、包与数据清单比较、直传和 checksum 摘要。完整直传进度日志保留在本地忽略目录 `audit/cache/migration_4090/`。

连接：

```bash
ssh -p 54268 duyanhong@10.15.20.149
```

Python 入口：

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python
```

再次执行兼容性检查时使用**新**输出文件：

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b112
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/verify_rddr_phase2b112_migration.py \
  --output /home/duyanhong/experiments/MIGRATION_4090_20260831/cuda_bf16_smoke_r2.json
```

**仅在后续获得启动授权后**，运行原冻结有限任务：

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b112
bash tools/execute_rddr_phase2b112.sh \
  /home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r1
```

该命令包括原定 CPU 检查、来源预检、一次校准、三臂各 500 步、固定 validation 快照、统计与最终实验报告。本次没有执行它，也未安排后台训练、GPU 轮询或自动重试。

## 8. 最终交付状态

**迁移完成，工程检查通过，等待用户决定何时启动正式三臂实验。**

尚未产出本轮 mIoU、置信区间、A–H 判定或科学 GO/NO-GO。此前 5090 的 RESOURCE_BLOCKED 记录保持原样，4090 的迁移 PASS 不等于实验假设 PASS。
