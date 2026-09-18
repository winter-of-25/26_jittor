# Experiment History

This page gives a high-level, public-facing overview of how the final system evolved.

## Reproduction Baseline

The project started from a StraightPCF / IterativePFN-style point cloud denoising baseline. The first reproduced system reached about 73.61 on the public leaderboard, establishing a reliable implementation and evaluation pipeline.

## Stable Parent Model

V12 became the first strong and stable parent model, reaching about 80.70. It confirmed that patch-based local denoising was a good fit for the dataset and later served as an important initialization point.

## Architecture Improvement

Subsequent versions explored residual refinement, cross-patch consensus, hierarchical residual modeling and noise-conditioned dual-path designs. Representative milestones include:

- V29: first clear architecture-level improvement.
- V33 / V34: improved residual refinement and local consistency.
- V40 / V41: better balance between CD and P2S.
- V45: stable late-stage parent model.
- V65: final A-leaderboard model.

## B-leaderboard Adaptation

After the B-leaderboard data became available, the system was adapted to the new distribution. Early B versions transferred the A-leaderboard parent models and then refined data usage, validation and residual strength selection.

B20 provided a strong B-leaderboard baseline. B24 further introduced the CAVR-v2 constrained residual design and stricter validation/inference locking, producing the final B-leaderboard submission.

## Final Design Choice

The final design prioritizes robustness and reproducibility. Rather than replacing the parent model with an unconstrained new branch, B24 uses a bounded residual student around a frozen teacher. This keeps the model stable while still allowing local geometric improvement.