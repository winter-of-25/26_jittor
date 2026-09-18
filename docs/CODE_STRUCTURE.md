# 代码结构说明

## 顶层入口

| 文件 | 功能 |
| --- | --- |
| `code/launch_b24.sh` | 后台启动 B24 完整流水线，避免终端断开影响训练 |
| `code/run_b24.sh` | B24 训练、验证、推理、打包总入口 |
| `code/train_b24.py` | B24 三阶段训练主程序 |
| `code/validate_b24.py` | 固定验证、候选 checkpoint 排名与 selection lock |
| `code/infer_b24.py` | 使用锁定 checkpoint 对 B 榜测试集推理 |
| `code/validate_submission.py` | 检查 `result.zip` 路径、数量、shape 与 zip 完整性 |

## B24 核心模块

| 文件 | 功能 |
| --- | --- |
| `code/b24_model.py` | CAVR-v2 模型定义；构造 teacher/student；执行 bounded residual |
| `code/b24_cagrad.py` | 双任务梯度处理，解决 CD/P2S 梯度冲突 |
| `code/preflight_b24.py` | 权重 hash、Jittor autograd、CAGrad solver、smoke test 与复现检查 |
| `code/benchmark_b24.py` | 根据显存与速度探测可用 batch size |
| `code/select_batch_b24.py` | 根据 batch probe 结果选择训练 batch size |
| `code/fixed_validation_b24.py` | B24 固定验证套件 |

## 数据与指标模块

| 文件 | 功能 |
| --- | --- |
| `code/jittor_port_data_base.py` | 数据读取、归一化、比赛预测数据集 |
| `code/robust_surface_data.py` | mesh 表面采样、法向与 surface 处理 |
| `code/aligned_ot_data_v29.py` | 对齐 patch 数据集与噪声模拟 |
| `code/train_joint_point_normal.py` | surface term、CD/P2S 代理损失相关函数 |
| `code/b20_utils.py` | JSON、hash、调度、参数统计等通用工具 |

## 历史版本与继承模块

| 文件 | 版本/方向 |
| --- | --- |
| `code/iterativepfn_v12.py` | V12 稳定基座 |
| `code/pseudo_query_corrector_v29.py` | V29 pseudo query 修正 |
| `code/cross_patch_consensus_v33.py` | V33 cross-patch consensus |
| `code/neural_laplacian_flow_v34.py` | V34 neural Laplacian flow |
| `code/competitive_residual_v39_42.py` | V39-V42 competitive residual |
| `code/hierarchical_residual_v43_46.py` | V43-V46 hierarchical residual |
| `code/joint_point_normal_v60.py` | V60 point-normal joint path |
| `code/stable_expert_moe_v61.py` | V61 stable expert MoE |
| `code/orthogonal_backbones_v62_v65.py` | V62-V65 orthogonal backbone / patch inference |
| `code/noise_conditioned_dual_path_v67.py` | V67 noise-conditioned dual path |
| `code/b20_model.py` | B20/B榜 parent model |

## 运行产物

完整运行后会生成 `logs/`、`checkpoints/`、`validation/`、`results/dataset_test_noisy/`、`result.zip`、`result.zip.sha256`、`selection_locked.json` 和 `test_access_allowed.json`。