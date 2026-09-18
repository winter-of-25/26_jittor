# 复现说明

本文档说明如何从原始数据、历史权重和本仓库代码复现 B24 训练、验证、推理和 `result.zip`。

## 1. 环境准备

推荐环境：Ubuntu 22.04、CUDA 12.4 compatible、Python 3.9、Jittor 1.3.11.0、GCC/G++ 10、单卡 24GB 显存或更高。

```bash
conda env create -f environment.yaml
conda activate jittor
python -m jittor_utils.install_cuda
```

手动安装：

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install -r requirements.txt
python -m jittor_utils.install_cuda
```

## 2. 数据目录

默认路径：

```text
/root/dataset_train
/root/dataset_test_noisy
/root/datalist/train_b.txt
/root/datalist/validate_b.txt
/root/datalist/test_b.txt
```

`test_b.txt` 在 B24 推理时要求正好 200 个样本。

## 3. 历史权重目录

B24 是从历史版本演化而来，需要外部 parent 权重。推荐放置：

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

`preflight_b24.py` 会检查关键权重 hash。若权重不一致，程序会主动报错。

## 4. 启动完整流水线

```bash
cd /root/26_jittor/code
export B24_PARENT_ROOT=/root/b24_parents
export B24_TEACHER_CKPT=/root/b24_parents/checkpoints/b20/b20_c4_teacher.pkl
export B24_RUN_ROOT=/root/26_jittor/code
bash launch_b24.sh
```

日志：

```bash
tail -f /root/26_jittor/code/pipeline.log
tail -f /root/26_jittor/code/logs/train.log
tail -f /root/26_jittor/code/logs/validation.log
tail -f /root/26_jittor/code/logs/predict.log
```

## 5. 流水线阶段

1. `benchmark_b24.py` 探测 batch size。
2. `preflight_b24.py` 检查权重 hash、CAGrad solver、Jittor autograd 与 smoke test。
3. `train_b24.py` 执行 Stage A/B/C 三阶段训练，默认纯训练目标约 13.5 小时。
4. `validate_b24.py` 对 C1-C4 候选 checkpoint 和多个 residual strength 做固定验证。
5. `infer_b24.py` 推理 200 个测试样本。
6. `validate_submission.py` 校验输出并生成 `result.zip`。

## 6. 关键训练参数

| 参数 | 默认值 |
| --- | ---: |
| `num_points` | 32768 |
| `patch_size` | 1000 |
| `patch_ratio` | 1.2 |
| `alignment_k` | 32 |
| `batch_size` | 自动选择，通常 6 |
| `num_workers` | 8 |
| `seed` | 8242401 |
| Stage A | 2.5h, `4e-5 -> 5e-6` |
| Stage B | 8.5h, `2.5e-5 -> 5e-7` |
| Stage C | 2.5h, `7e-6 -> 2e-7` |

## 7. 提交校验

```bash
python validate_submission.py \
  --data_root /root \
  --test_list /root/datalist/test_b.txt \
  --output_root /root/26_jittor/code/results \
  --zip_path /root/26_jittor/code/result.zip
```

常见问题包括测试列表数量不对、zip 内路径多了一层目录、某些 `denoised.npy` 缺失或 shape 不为 `(50000, 3)`。