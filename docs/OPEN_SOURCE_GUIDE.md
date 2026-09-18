# Open Source Guide

This repository is organized for reproducibility, review and future development.

## Documentation

- `README.md`: project overview, results and quick start.
- `docs/TECHNICAL_DESIGN.md`: model design and key ideas.
- `docs/REPRODUCTION.md`: environment setup and reproduction workflow.
- `docs/CODE_STRUCTURE.md`: source file responsibilities.
- `docs/EXPERIMENT_HISTORY.md`: public model evolution summary.
- `docs/PERFORMANCE.md`: leaderboard metrics and representative milestones.
- `docs/PRESENTATION_GUIDE.md`: concise presentation notes.
- `CONTRIBUTING.md`: contribution workflow.
- `THIRD_PARTY_NOTICES.md`: third-party dependency information.

## Contribution Workflow

Recommended workflow:

1. Open an issue for non-trivial changes.
2. Use a dedicated branch such as `fix/...`, `docs/...` or `exp/...`.
3. Describe data lists, checkpoints, hyperparameters and validation metrics in pull requests that affect training or inference.
4. Run basic checks before opening a pull request.

## Dependency Policy

The project depends on Jittor, NumPy, SciPy, Trimesh, PyYAML, OmegaConf and other standard Python packages listed in `requirements.txt` and `environment.yaml`.

Large competition data and generated checkpoints are intentionally not stored in the Git repository. They should be kept in external storage and referenced through documented paths.

## Reproducibility Policy

Reproducible runs should keep:

- Fixed data lists.
- Checkpoint hashes.
- Training logs.
- Validation records.
- Final submission validation output.

The B24 pipeline already includes preflight checks, checkpoint selection records and submission integrity checks.