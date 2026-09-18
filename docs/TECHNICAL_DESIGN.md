# 技术设计

## 任务说明

赛道二关注三维点云去噪。模型输入为带噪点云，输出为更贴近物体真实表面的去噪点云。

评测同时关注两个指标：

- **CD / Chamfer Distance**：衡量输出点云与目标表面的整体点集距离。
- **P2S / point-to-surface distance**：衡量输出点到连续表面的局部贴合质量。

这两个目标相关但并不完全一致。过强的点位移可能改善一个指标，却损害另一个指标。因此 B24 / CAVR-v2 的设计重点是：在稳定 teacher 的几何锚点附近进行局部修正，而不是让 student 自由产生大幅移动。

## 模型结构

B24 / CAVR-v2 由冻结 teacher 分支和可训练 student 分支组成：

```text
noisy patch
    ├── frozen whole-shape parent
    ├── frozen teacher head
    └── trainable student head

student_raw - teacher
    -> local bounded residual
    -> teacher + residual
    -> patch fusion
    -> denoised point cloud
```

主要源码：

- `code/b24_model.py`：模型定义与 bounded residual 推理。
- `code/train_b24.py`：训练阶段与任务损失。
- `code/b24_cagrad.py`：多目标梯度处理。
- `code/orthogonal_backbones_v62_v65.py`：patch 推理与融合。
- `code/validate_b24.py`：固定验证与 checkpoint 选择。
- `code/infer_b24.py`：测试集推理。

## CAVR：受约束锚点残差

最终预测由 teacher 输出加上受约束的 student 残差得到：

```text
candidate = teacher + clip(student_raw - teacher, cap)
cap = 0.10 * r32
```

其中 `r32` 表示局部邻域尺度。残差上限随局部点云密度与几何尺度变化，使稀疏区域和细节区域拥有不同的修正范围。

这一设计有三个作用：

- 保留强 teacher 的稳定性。
- 允许 student 修正局部几何细节。
- 降低分布变化下的不稳定点位移。

## 多目标训练

训练目标包含两个主要任务：

- `cd_task`：面向 CD 的相对改善。
- `surface_task`：面向表面贴合质量的相对改善。

B24 使用 CAGrad 风格的梯度合成来缓解 CD 与 P2S 优化方向的冲突，而不是只依赖固定 loss 加权。

## 验证与推理锁定

流水线会记录明确的训练和选择状态：

- `training_complete.json`
- `eligible_checkpoints.json`
- `selection_locked.json`
- `test_access_allowed.json`

`infer_b24.py` 在测试集推理前会检查这些文件，并校验 checkpoint hash。这使最终预测路径可复现、可追踪。

## Patch 推理

完整点云通过局部 patch 推理后融合。默认推理参数：

| 参数 | 取值 | 含义 |
| --- | ---: | --- |
| `patch_size` | 1000 | 每个局部 patch 的点数 |
| `seed_k` | 6 | patch seed 数量 |
| `beta` | 12 | 融合权重温度 |
| `patch_batch` | 6 | 推理时 patch batch size |

## 与历史版本的关系

B24 继承了 V12、V29、V33、V41、V45、V65 等历史高分版本的组件与 checkpoint。这些版本提供稳定的 parent 表示和结构基础。B24 在此基础上加入 CAVR-v2 残差约束、B 榜适配训练和更严格的复现检查。