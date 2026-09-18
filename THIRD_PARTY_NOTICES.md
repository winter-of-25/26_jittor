# 第三方依赖说明

本项目基于 Jittor 框架实现，不包含第三方库源码。依赖版本见 `requirements.txt` 和 `environment.yaml`。

| 依赖 | 用途 |
| --- | --- |
| Jittor | 模型定义、自动微分、GPU 训练与推理 |
| NumPy | 点云数组处理、随机采样、指标辅助计算 |
| SciPy | 空间计算与数值工具 |
| Trimesh | 读取 mesh、采样表面点与法向 |
| PyYAML / OmegaConf | 配置解析 |
| tqdm | 进度输出 |
| Pillow | 图像/辅助 I/O 依赖 |

使用者应遵守上述项目各自的开源许可证。本仓库代码以 `LICENSE` 中声明的许可证发布。

数据集、比赛平台、榜单与评测系统属于竞赛组织方；本仓库不再分发数据集。