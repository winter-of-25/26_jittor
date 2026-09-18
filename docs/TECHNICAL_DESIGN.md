# Technical Design

## Task

Track 2 of the Jittor Challenge focuses on 3D point cloud denoising. Given a noisy point cloud, the model predicts a denoised point cloud that better matches the underlying object surface.

The evaluation considers both:

- **CD / Chamfer Distance**: global point-set distance.
- **P2S / point-to-surface distance**: local surface fitting quality.

These two objectives are correlated but not identical. A model that moves points aggressively may improve one metric while degrading the other. B24 / CAVR-v2 is designed to improve local geometry while keeping the prediction anchored to a stable denoising baseline.

## Architecture

B24 / CAVR-v2 combines a frozen teacher branch and a trainable student branch:

```text
noisy patch
    ├── frozen whole-shape parent
    ├── frozen teacher head
    └── trainable student head

student_raw - teacher
    -> local bounded residual
    -> teacher + residual
    -> patch fusion
    -> denoised point cloud
```

Main implementation files:

- `code/b24_model.py`: model definition and bounded residual inference.
- `code/train_b24.py`: training stages and task losses.
- `code/b24_cagrad.py`: multi-objective gradient handling.
- `code/orthogonal_backbones_v62_v65.py`: patch-based inference and fusion.
- `code/validate_b24.py`: fixed validation and checkpoint selection.
- `code/infer_b24.py`: test-set inference.

## CAVR: Constrained Anchor Vector Residual

The final prediction is produced by adding a bounded student residual to the teacher output:

```text
candidate = teacher + clip(student_raw - teacher, cap)
cap = 0.10 * r32
```

Here `r32` is a local neighborhood scale. The residual bound is adaptive: sparse or large-scale regions allow different movement ranges than dense fine-detail regions.

This design has three practical benefits:

- It preserves the robustness of a strong frozen teacher.
- It allows the student to refine local geometric details.
- It reduces unstable point displacement under distribution shift.

## Multi-objective Training

The training objective contains two task losses:

- `cd_task`: relative CD improvement.
- `surface_task`: relative surface-fitting improvement.

Instead of using a fixed weighted sum only, B24 uses CAGrad-style gradient composition to reduce gradient conflict between CD and P2S optimization.

## Validation and Inference Locking

The pipeline records explicit training and selection states:

- `training_complete.json`
- `eligible_checkpoints.json`
- `selection_locked.json`
- `test_access_allowed.json`

`infer_b24.py` checks these files and validates checkpoint hashes before test-set inference. This makes the final prediction path deterministic and auditable.

## Patch-based Inference

Full point clouds are processed with patch-based inference and then fused. Default inference parameters:

| Parameter | Value | Description |
| --- | ---: | --- |
| `patch_size` | 1000 | Points per local patch |
| `seed_k` | 6 | Number of patch seeds |
| `beta` | 12 | Fusion temperature |
| `patch_batch` | 6 | Patch batch size |

## Relationship to Earlier Versions

B24 inherits components and checkpoints from earlier high-performing versions, including V12, V29, V33, V41, V45 and V65. These versions provide stable parent representations and historical architectural components. B24 adds the CAVR-v2 residual constraint, B-leaderboard-oriented training and stricter reproducibility gates.