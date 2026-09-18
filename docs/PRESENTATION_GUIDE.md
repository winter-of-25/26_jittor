# 展示说明

本文档用于概括项目展示时可以强调的核心内容。

## 简要介绍

本项目解决三维点云去噪任务。给定带噪点云，系统预测更贴近真实表面的去噪点云。最终版本 **B24 / CAVR-v2** 基于 Jittor 实现，结合冻结 teacher、可训练 residual student、局部受约束修正和 CD/P2S 多目标优化。

## 任务难点

- 点云是无序且不规则采样的数据。
- 不同形状和局部区域的噪声强度不同。
- Chamfer Distance 与 point-to-surface 质量相关但不完全一致。
- 隐藏测试分布可能与公开验证分布不同。

## 主要方法

核心预测规则为：

```text
candidate = teacher + clip(student_raw - teacher, 0.10 * r32)
```

teacher 提供稳定的去噪锚点。student 分支学习局部修正，但每个修正都受到局部邻域尺度约束。这样可以保留 teacher 的稳定性，同时继续恢复局部几何细节。

## 优化方式

训练目标包含两个主要部分：

- 面向全局点集准确性的 CD 损失。
- 面向表面贴合质量的 surface / P2S 损失。

模型使用 CAGrad 风格的梯度处理方式，缓解两个目标之间的优化冲突。

## 可复现性

流水线使用固定数据列表、checkpoint hash、预检脚本、固定验证选择和提交包校验，保证训练、选择、推理和打包过程可追踪。

## 展示流程建议

1. 展示仓库结构。
2. 展示 `requirements.txt` 或 `environment.yaml` 中的环境配置。
3. 说明 `code/run_b24.sh` 是完整流水线入口。
4. 说明 `code/b24_model.py` 中的 bounded residual 公式。
5. 展示 `code/validate_submission.py` 如何检查提交结果完整性。

## 结果

| 榜单 | 版本 | 总分 | CD_score | P2S_score |
| --- | --- | ---: | ---: | ---: |
| A榜 | V65 | 83.13 | 73.21 | 93.06 |
| B榜 | B24 / CAVR-v2 | 81.44 | 70.15 | 92.73 |