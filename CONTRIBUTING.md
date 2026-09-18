# 贡献指南

欢迎围绕复现、文档、模型改进和工程稳定性提交贡献。

## 分支建议

- `main`：稳定版本，对应最终 B24 复现代码。
- `docs/*`：文档更新。
- `fix/*`：bug 修复。
- `exp/*`：实验性模型或训练策略。

## 提交前检查

至少确认：

```bash
python code/validate_submission.py --help
python code/preflight_b24.py --help
python code/infer_b24.py --help
```

如果改动训练、验证或推理逻辑，需要说明：

- 使用的数据列表；
- checkpoint 来源；
- 关键超参数；
- 本地验证指标；
- 是否改变最终 `result.zip` 的目录结构。

## 代码风格

- Python 代码保持清晰命名，避免无关重构。
- 只在复杂逻辑处添加必要注释。
- 不在仓库中提交数据集、临时日志、大型 checkpoint 或 `result.zip`。
- 依赖变更必须同步更新 `requirements.txt` 或 `environment.yaml`。

## PR 内容

PR 描述建议包含：目的、主要改动、验证方式、性能变化和风险。涉及竞赛成绩的 PR 应附上版本号、指标和提交结果。