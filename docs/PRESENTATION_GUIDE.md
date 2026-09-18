# Presentation Guide

This document summarizes the project in a concise and reproducible way for public presentation.

## Short Introduction

This project addresses 3D point cloud denoising. Given a noisy point cloud, the system predicts a denoised point cloud that better fits the underlying surface. The final version, **B24 / CAVR-v2**, is implemented with Jittor and combines a frozen teacher model, a trainable residual student branch, bounded local correction and CD/P2S multi-objective optimization.

## Task Difficulty

- Point clouds are unordered and irregularly sampled.
- Local noise strength varies across shapes and regions.
- Chamfer Distance and point-to-surface quality are related but not identical.
- Hidden test distributions may differ from public validation distributions.

## Main Method

The core prediction rule is:

```text
candidate = teacher + clip(student_raw - teacher, 0.10 * r32)
```

The teacher provides a stable denoising anchor. The student branch learns local corrections, but each correction is bounded by the local neighborhood scale. This preserves teacher robustness while allowing additional detail recovery.

## Optimization

The training objective contains two main components:

- CD-oriented loss for global point-set accuracy.
- Surface-oriented loss for point-to-surface quality.

CAGrad-style gradient handling is used to reduce conflict between the two objectives.

## Reproducibility

The pipeline uses fixed data lists, checkpoint hash checks, preflight scripts, locked validation selection and submission zip validation.

## Suggested Demo Flow

1. Show the repository structure.
2. Show environment setup from `requirements.txt` or `environment.yaml`.
3. Explain `code/run_b24.sh` as the full pipeline entry point.
4. Explain `code/b24_model.py` and the bounded residual formula.
5. Show `code/validate_submission.py` for result integrity checking.

## Results

| Leaderboard | Version | Score | CD_score | P2S_score |
| --- | --- | ---: | ---: | ---: |
| A榜 | V65 | 83.13 | 73.21 | 93.06 |
| B榜 | B24 / CAVR-v2 | 81.44 | 70.15 | 92.73 |