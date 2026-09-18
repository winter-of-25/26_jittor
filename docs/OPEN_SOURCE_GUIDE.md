# 开源说明

本仓库围绕可复现、可审阅和可继续开发进行组织。

## 文档结构

- `README.md`：项目概览、成绩和快速开始。
- `docs/TECHNICAL_DESIGN.md`：模型设计与核心思路。
- `docs/REPRODUCTION.md`：环境配置与复现流程。
- `docs/CODE_STRUCTURE.md`：源码文件职责说明。
- `docs/EXPERIMENT_HISTORY.md`：模型演化概述。
- `docs/PERFORMANCE.md`：榜单指标与代表性版本。
- `docs/PRESENTATION_GUIDE.md`：项目展示说明。
- `CONTRIBUTING.md`：贡献流程。
- `THIRD_PARTY_NOTICES.md`：第三方依赖说明。

## 协作流程

推荐流程：

1. 非简单修改先创建 issue，说明目标和验证方式。
2. 使用独立分支，例如 `fix/...`、`docs/...` 或 `exp/...`。
3. 涉及训练或推理的 PR 需要说明数据列表、checkpoint、超参数和验证指标。
4. 提交前运行基础检查。

## 依赖管理

项目依赖 Jittor、NumPy、SciPy、Trimesh、PyYAML、OmegaConf 等常见 Python 包，具体版本见 `requirements.txt` 和 `environment.yaml`。

比赛数据和生成的 checkpoint 不存放在 Git 仓库中，应放在外部存储，并通过文档约定路径引用。

## 复现记录

可复现运行建议保留：

- 固定数据列表。
- checkpoint hash。
- 训练日志。
- 验证记录。
- 最终提交包校验结果。

B24 流水线已经包含预检、checkpoint 选择记录和提交包完整性检查。