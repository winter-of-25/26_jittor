# 开源协作与合规说明

评分中开源协作占 20 分。本仓库围绕“可读、可跑、可检查、可继续开发”建设。

## 仓库建设

- `README.md`：项目入口、成绩、快速运行方式。
- `docs/TECHNICAL_DESIGN.md`：模型结构与创新点。
- `docs/REPRODUCTION.md`：从环境到 `result.zip` 的完整复现流程。
- `docs/CODE_STRUCTURE.md`：源码文件职责说明。
- `docs/EXPERIMENT_HISTORY.md`：从 baseline 到 B24 的实验历程。
- `docs/PERFORMANCE.md`：关键版本分数和指标解释。
- `docs/DEFENSE_GUIDE.md`：现场答辩提纲和常见问答。
- `CONTRIBUTING.md`：协作、分支、代码风格和 PR 要求。
- `THIRD_PARTY_NOTICES.md`：第三方库与许可证说明。
- `.github/ISSUE_TEMPLATE/` 与 `.github/PULL_REQUEST_TEMPLATE.md`：规范 issue 与 PR 信息。

## 协作方式

建议按以下流程协作：

1. 新想法先开 issue，说明目标、预期收益、风险和验证方式。
2. 新版本使用独立分支，例如 `exp/b25-new-router`。
3. 每个 PR 必须写清楚训练数据、参数、checkpoint、验证结果和是否影响推理格式。
4. 合并前至少运行 `python code/validate_submission.py --help`、`python code/preflight_b24.py --help`、`python code/infer_b24.py --help`。

## 第三方库合规

本项目主要依赖 Jittor、NumPy、SciPy、Trimesh、PyYAML、OmegaConf 等。详见 `requirements.txt`、`environment.yaml` 和 `THIRD_PARTY_NOTICES.md`。本仓库不包含比赛数据集，也不包含第三方库源码。

## 后续可改进方向

- 增加轻量 demo 数据，用于无比赛数据时快速展示推理链路。
- 补充 noisy / denoised 可视化结果图。
- 将历史实验记录整理为结构化表格或网页。
- 增加 GitHub Actions 的静态检查；Jittor GPU 训练仍需本地或服务器完成。