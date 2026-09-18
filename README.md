# 26_jittor：基于 Jittor 的三维点云去噪

[![Framework](https://img.shields.io/badge/framework-Jittor-red)](https://cg.cs.tsinghua.edu.cn/jittor/)
[![Python](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)
[![Task](https://img.shields.io/badge/task-point--cloud--denoising-orange)](#)

本仓库为 **第六届计图挑战赛赛道二：三维点云去噪** 的参赛代码。最终提交系统为 **B24 / CAVR-v2**，基于 Jittor 实现，面向带噪三维点云进行局部表面恢复与稳定去噪。

队伍名称：**try一次**

## 成绩

| 榜单 | 版本 | 排名 | 总分 | CD_score | P2S_score |
| --- | --- | ---: | ---: | ---: | ---: |
| B榜 | B24 / CAVR-v2 | 第6名 | 81.44 | 70.15 | 92.73 |
| A榜 | V65 | 第12名 | 83.13 | 73.21 | 93.06 |

## 方法概述

B24 / CAVR-v2 的核心思想是：保留一个稳定的冻结 teacher 作为几何锚点，同时训练 student 分支只预测受约束的局部残差。这样可以在改善局部几何细节的同时，减少隐藏测试分布上不稳定的大幅点位移。

主要组件：

- **patch 级点云去噪**：面向 50k 点云进行局部 patch 推理与融合。
- **teacher-student 残差细化**：冻结 teacher，训练 student head。
- **局部尺度约束残差**：根据邻域半径限制每个点的最大修正幅度。
- **CD/P2S 多目标训练**：使用 CAGrad 风格的梯度冲突处理，同时兼顾 Chamfer Distance 与 point-to-surface 质量。
- **可复现的验证与推理流程**：使用 checkpoint hash、固定验证、选择锁和提交包校验保证结果一致性。

技术细节见 [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md)。

## 仓库结构

```text
.
├── code/
│   ├── launch_b24.sh                 # 后台启动完整 B24 流水线
│   ├── run_b24.sh                    # B24 训练、验证、推理与打包入口
│   ├── train_b24.py                  # B24 训练主程序
│   ├── validate_b24.py               # 固定验证与 checkpoint 选择
│   ├── infer_b24.py                  # 测试集推理
│   ├── validate_submission.py        # 提交包完整性校验
│   ├── b24_model.py                  # CAVR-v2 模型定义
│   ├── b24_cagrad.py                 # 多目标梯度处理
│   └── *_v*.py                       # B24 使用到的历史模型组件
├── docs/
│   ├── TECHNICAL_DESIGN.md
│   ├── REPRODUCTION.md
│   ├── CODE_STRUCTURE.md
│   ├── EXPERIMENT_HISTORY.md
│   ├── PERFORMANCE.md
│   ├── PRESENTATION_GUIDE.md
│   └── OPEN_SOURCE_GUIDE.md
├── requirements.txt
├── environment.yaml
├── CONTRIBUTING.md
├── THIRD_PARTY_NOTICES.md
└── LICENSE
```

## 环境配置

推荐环境：

- Ubuntu 22.04
- CUDA 12.4 兼容 NVIDIA GPU
- Python 3.9
- Jittor 1.3.11.0
- GCC/G++ 10

```bash
conda env create -f environment.yaml
conda activate jittor
python -m jittor_utils.install_cuda
```

也可以手动安装：

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install -r requirements.txt
python -m jittor_utils.install_cuda
```

## 数据与权重

比赛数据和大体积 checkpoint 不包含在本 Git 仓库中。代码默认使用以下目录，可在 `code/run_b24.sh` 中调整：

```text
/root/dataset_train
/root/dataset_test_noisy
/root/datalist/train_b.txt
/root/datalist/validate_b.txt
/root/datalist/test_b.txt
```

B24 还依赖若干历史 parent checkpoint。推荐目录形式如下：

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

## 快速开始

```bash
cd /root/26_jittor/code
export B24_PARENT_ROOT=/root/b24_parents
export B24_TEACHER_CKPT=/root/b24_parents/checkpoints/b20/b20_c4_teacher.pkl
export B24_RUN_ROOT=/root/26_jittor/code
bash launch_b24.sh
```

完整流水线会依次执行 batch size 探测、预检、训练、固定验证、测试集推理和 `result.zip` 打包。

完整复现说明见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。

## 文档

- [技术设计](docs/TECHNICAL_DESIGN.md)
- [复现说明](docs/REPRODUCTION.md)
- [代码结构](docs/CODE_STRUCTURE.md)
- [实验历程](docs/EXPERIMENT_HISTORY.md)
- [性能记录](docs/PERFORMANCE.md)
- [展示说明](docs/PRESENTATION_GUIDE.md)
- [开源说明](docs/OPEN_SOURCE_GUIDE.md)

## 许可证

本仓库代码以 MIT License 开源，详见 [LICENSE](LICENSE)。