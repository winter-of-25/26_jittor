# 26_jittor: Point Cloud Denoising with Jittor

[![Framework](https://img.shields.io/badge/framework-Jittor-red)](https://cg.cs.tsinghua.edu.cn/jittor/)
[![Python](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)
[![Task](https://img.shields.io/badge/task-point--cloud--denoising-orange)](#)

This repository contains the code for **Track 2: 3D Point Cloud Denoising** in the 6th Jittor Challenge. The final submitted system is **B24 / CAVR-v2**, a Jittor-based patch denoising pipeline designed for robust surface recovery under noisy point-cloud observations.

队伍名称：**try一次**

## Results

| Leaderboard | Version | Rank | Score | CD_score | P2S_score |
| --- | --- | ---: | ---: | ---: | ---: |
| B榜 | B24 / CAVR-v2 | 6 | 81.44 | 70.15 | 92.73 |
| A榜 | V65 | 12 | 83.13 | 73.21 | 93.06 |

## Method Overview

B24 / CAVR-v2 is built around a conservative but effective idea: keep a strong frozen teacher as the geometric anchor, and train a student branch to predict only a bounded local residual. This improves local detail while reducing unstable point movements on shifted test distributions.

Core components:

- **Patch-based denoising** for 50k-point shapes.
- **Teacher-student residual refinement** with a frozen teacher and trainable student head.
- **Locally bounded residuals** scaled by neighborhood radius.
- **CD/P2S multi-objective training** with CAGrad-style gradient conflict handling.
- **Locked validation and inference pipeline** with checkpoint hash checks and submission validation.

More details are available in [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md).

## Repository Structure

```text
.
├── code/
│   ├── launch_b24.sh                 # Launch the full B24 pipeline in background
│   ├── run_b24.sh                    # Train, validate, infer and package B24
│   ├── train_b24.py                  # B24 training entry point
│   ├── validate_b24.py               # Fixed validation and checkpoint selection
│   ├── infer_b24.py                  # Test-set inference
│   ├── validate_submission.py        # Submission zip integrity check
│   ├── b24_model.py                  # CAVR-v2 model definition
│   ├── b24_cagrad.py                 # Multi-objective gradient handling
│   └── *_v*.py                       # Historical model components used by B24
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

## Environment

Recommended environment:

- Ubuntu 22.04
- CUDA 12.4 compatible NVIDIA GPU
- Python 3.9
- Jittor 1.3.11.0
- GCC/G++ 10

```bash
conda env create -f environment.yaml
conda activate jittor
python -m jittor_utils.install_cuda
```

Manual installation:

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install -r requirements.txt
python -m jittor_utils.install_cuda
```

## Data and Checkpoints

Competition data and large checkpoints are not included in this repository. The code expects the following default layout, which can be adjusted in `code/run_b24.sh`:

```text
/root/dataset_train
/root/dataset_test_noisy
/root/datalist/train_b.txt
/root/datalist/validate_b.txt
/root/datalist/test_b.txt
```

B24 also uses historical parent checkpoints. A typical layout is:

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

## Quick Start

```bash
cd /root/26_jittor/code
export B24_PARENT_ROOT=/root/b24_parents
export B24_TEACHER_CKPT=/root/b24_parents/checkpoints/b20/b20_c4_teacher.pkl
export B24_RUN_ROOT=/root/26_jittor/code
bash launch_b24.sh
```

The pipeline performs batch-size probing, preflight checks, training, fixed validation, test inference and `result.zip` packaging.

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for full reproduction instructions.

## Documentation

- [Technical Design](docs/TECHNICAL_DESIGN.md)
- [Reproduction Guide](docs/REPRODUCTION.md)
- [Code Structure](docs/CODE_STRUCTURE.md)
- [Experiment History](docs/EXPERIMENT_HISTORY.md)
- [Performance](docs/PERFORMANCE.md)
- [Presentation Guide](docs/PRESENTATION_GUIDE.md)
- [Open Source Guide](docs/OPEN_SOURCE_GUIDE.md)

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).