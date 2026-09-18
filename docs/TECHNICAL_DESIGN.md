# 技术设计说明

## 任务理解

赛道二要求对带噪三维点云进行去噪。输入点云包含局部扰动、非均匀采样与复杂几何细节，输出需要在保持原始形状结构的同时贴近真实表面。比赛指标同时关注 CD / Chamfer Distance 与 P2S / point-to-surface。CD 更关注整体几何距离，P2S 更关注点到连续表面的贴合程度，两者并不完全一致。

## 模型总体结构

最终版本为 **B24 / CAVR-v2**。结构可以概括为：

```text
noisy patch
    ├── frozen whole parent / historical backbone
    ├── frozen teacher head
    └── trainable student head

student_raw - teacher
    -> bounded residual by local r32
    -> teacher + bounded residual
    -> patch fusion / orthogonal inference
    -> denoised point cloud
```

对应源码：

- `code/b24_model.py`：B24 模型定义、teacher/student 构造、bounded residual。
- `code/train_b24.py`：训练目标、三阶段训练计划、候选 checkpoint 保存。
- `code/b24_cagrad.py`：CD/P2S 双目标梯度冲突处理。
- `code/orthogonal_backbones_v62_v65.py`：patch-based orthogonal inference 与融合。
- `code/validate_b24.py`：固定验证集上筛选 checkpoint 与推理强度。
- `code/infer_b24.py`：B 榜 200 个测试样本推理。

## 关键创新点

### 1. CAVR：Constrained Anchor Vector Residual

B24 不直接让 student 替代 teacher，而是让 student 学习 teacher 周围的局部残差：

```text
candidate = teacher + clip(student_raw - teacher, cap)
cap = 0.10 * r32
```

其中 `r32` 来自局部邻域尺度。这样每个点允许移动的幅度随局部密度与形状尺度变化，能避免隐藏测试集上出现大幅错误位移。

### 2. Teacher-student 双路径蒸馏

`b24_model.py` 中构造 frozen teacher 与 trainable student。teacher 权重固定，只作为锚点与训练参照；student 继承历史强模型初始化，在 bounded residual 空间内学习。

### 3. CD/P2S 冲突梯度处理

`train_b24.py` 中将训练目标拆成 `cd_task` 和 `surface_task`，再在 `b24_cagrad.py` 中通过 CAGrad 风格的梯度合成更新参数。它不是简单加权两个 loss，而是在梯度层面减少冲突。

### 4. 固定验证与选择锁

B24 在推理测试集前会产生 `training_complete.json`、`eligible_checkpoints.json`、`selection_locked.json` 和 `test_access_allowed.json`。`infer_b24.py` 会检查这些 gate 文件与 checkpoint hash，保证最终 `result.zip` 来自可复现流水线。

### 5. Patch-based orthogonal inference

最终推理不是一次性处理整云，而是局部 patch 推理后再融合。关键参数包括 `patch_size=1000`、`seed_k=6`、`beta=12`、`patch_batch=6`。

## 与 A榜算法的关系

B24 继承了 A榜 V65 的强基座与推理经验，但针对 B榜加入了 CAVR-v2 bounded residual、B榜固定验证、双目标约束与更严格的推理隔离。