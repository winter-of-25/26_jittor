# 第六届计图挑战赛赛道二：三维点云去噪

[![Framework](https://img.shields.io/badge/framework-Jittor-red)](https://cg.cs.tsinghua.edu.cn/jittor/)
[![Python](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)
[![Task](https://img.shields.io/badge/task-point--cloud--denoising-orange)](#)

本仓库是队伍 **try一次** 参加第六届计图挑战赛赛道二的开源代码。任务目标是从带噪三维点云中恢复干净表面点云，最终提交版本为 **B24 / CAVR-v2**。

## 成绩

| 榜单 | 最优版本 | 排名 | 总分 | CD_score | P2S_score |
| --- | --- | ---: | ---: | ---: | ---: |
| B榜 | B24 | 第6名 | 81.44 | 70.15 | 92.73 |
| A榜 | V65 | 第12名 | 83.13 | 73.21 | 93.06 |

## 总体设计

B24 不是单次调参得到的模型，而是从 baseline 逐步演化出的 Jittor 点云去噪流水线：

1. **局部 patch 去噪主干**：沿用 IterativePFN / StraightPCF 思路，面向 50k 点云执行 patch 级局部表面恢复。
2. **多阶段历史基座**：保留 V12、V29、V33、V41、V45、V65 等关键阶段的结构经验，用它们初始化或约束后续模型。
3. **CAVR-v2 核心结构**：以稳定 teacher 为锚点，训练 student residual head，只允许有限幅度的局部修正，降低 B 榜分布迁移时的过拟合风险。
4. **CD/P2S 双目标优化**：使用 CAGrad 风格的冲突梯度处理，同时优化 Chamfer Distance 与 point-to-surface 代理目标。
5. **固定验证与隔离推理**：训练、候选选择、测试推理之间用 manifest、hash 与 gate 文件隔离，减少手工选择带来的不可复现风险。

更完整的技术说明见 [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md)。

## 仓库结构

```text
.
├── code/
│   ├── launch_b24.sh                 # 后台启动完整 B24 流水线
│   ├── run_b24.sh                    # B24 训练、验证、推理、打包总入口
│   ├── train_b24.py                  # CAVR-v2 训练主脚本
│   ├── validate_b24.py               # 固定验证与候选选择
│   ├── infer_b24.py                  # B 榜测试集推理
│   ├── validate_submission.py        # result.zip 完整性检查
│   ├── b24_model.py                  # B24 模型定义
│   ├── b24_cagrad.py                 # CD/P2S 多目标梯度处理
│   └── *_v*.py                       # 历史版本结构与复现模块
├── docs/
│   ├── TECHNICAL_DESIGN.md           # 技术创新与模型细节
│   ├── REPRODUCTION.md               # 从环境到 result.zip 的复现流程
│   ├── CODE_STRUCTURE.md             # 代码文件职责说明
│   ├── EXPERIMENT_HISTORY.md         # 从 baseline 到 B24 的实验历程
│   ├── PERFORMANCE.md                # A/B 榜指标与版本对比
│   ├── DEFENSE_GUIDE.md              # 现场答辩提纲
│   └── OPEN_SOURCE_GUIDE.md          # 开源协作与合规说明
├── requirements.txt
├── environment.yaml
├── CONTRIBUTING.md
├── THIRD_PARTY_NOTICES.md
└── LICENSE
```

## 环境配置

推荐环境：

- Ubuntu 22.04
- NVIDIA RTX 3090/4090 或同等显存 GPU
- CUDA 12.4 兼容环境
- Python 3.9
- Jittor 1.3.11.0
- GCC/G++ 10

```bash
conda env create -f environment.yaml
conda activate jittor
python -m jittor_utils.install_cuda
```

或者手动安装：

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install -r requirements.txt
python -m jittor_utils.install_cuda
```

## 数据与权重约定

代码默认使用比赛服务器上的路径：

```text
/root/dataset_train
/root/dataset_test_noisy
/root/datalist/train_b.txt
/root/datalist/validate_b.txt
/root/datalist/test_b.txt
```

B24 复现还需要历史基座权重。出于仓库体积与比赛提交规范考虑，数据集不放入 Git 仓库；权重按比赛提交包或本地归档提供。推荐放置为：

```text
/root/b24_parents/checkpoints/v12/iterativepfn_best.pkl
/root/b24_parents/checkpoints/v12/args.json
/root/b24_parents/checkpoints/v29/v29_best.pkl
/root/b24_parents/checkpoints/v33/v33_best.pkl
/root/b24_parents/checkpoints/v41/v41_best_active.pkl
/root/b24_parents/checkpoints/v45/v45_best_active.pkl
/root/b24_parents/checkpoints/v65/v65_best_active.pkl
/root/b24_parents/checkpoints/b20/b20_c4_teacher.pkl
```

## 快速运行

```bash
cd /root/26_jittor/code
export B24_PARENT_ROOT=/root/b24_parents
export B24_TEACHER_CKPT=/root/b24_parents/checkpoints/b20/b20_c4_teacher.pkl
export B24_RUN_ROOT=/root/26_jittor/code
bash launch_b24.sh
```

完整流水线会依次执行：

1. batch size 探测；
2. 权重 hash 与预检；
3. B24 三阶段训练；
4. 固定验证与候选选择；
5. B 榜测试集推理；
6. `result.zip` 生成与完整性校验。

复现细节、参数解释和常见问题见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。

## 答辩要点

本项目的答辩重点建议围绕：

- 为什么点云去噪需要同时关注 CD 与 P2S；
- 为什么单纯扩大模型或继续调参容易过拟合；
- B24 如何用 teacher-student、bounded residual 与 CAGrad 平衡性能和稳定性；
- 如何保证测试集推理前的候选选择可复现；
- 开源仓库如何支持复现、协作与合规检查。

答辩提纲见 [docs/DEFENSE_GUIDE.md](docs/DEFENSE_GUIDE.md)。

## 开源协作

仓库包含贡献说明、第三方依赖说明、issue/PR 模板和代码结构文档。第三方依赖主要为 Jittor、NumPy、SciPy、Trimesh、PyYAML、OmegaConf 等，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可证

本仓库代码以 MIT License 开源，详见 [LICENSE](LICENSE)。