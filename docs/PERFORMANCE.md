# Performance

## Final Results

| Leaderboard | Version | Rank | Score | CD_score | P2S_score |
| --- | --- | ---: | ---: | ---: | ---: |
| B榜 | B24 / CAVR-v2 | 6 | 81.44 | 70.15 | 92.73 |
| A榜 | V65 | 12 | 83.13 | 73.21 | 93.06 |

## Selected Model Evolution

The final system was developed through a sequence of increasingly stable denoising models. The table below lists representative milestones.

| Version | Score | CD_score | P2S_score | Note |
| --- | ---: | ---: | ---: | --- |
| StraightPCF reproduction | 73.61 | - | - | Initial reproduction baseline |
| V12 | 80.70 | 68.68 | 92.73 | Stable early parent model |
| V29 | 81.37 | 70.83 | 91.91 | Effective architecture upgrade |
| V33 | 82.53 | 72.10 | 92.95 | Improved residual refinement |
| V40 | 82.89 | 72.78 | 93.01 | Strong CD/P2S balance |
| V45 | 83.00 | 72.94 | 93.06 | Stable late A-leaderboard parent |
| V65 | 83.13 | 73.21 | 93.06 | Final A-leaderboard model |
| B20 | 81.41 | 70.10 | 92.71 | Strong B-leaderboard adaptation |
| B24 | 81.44 | 70.15 | 92.73 | Final B-leaderboard model |

## B-leaderboard Adaptation

The B-leaderboard data distribution differs from the A-leaderboard distribution. Directly transferring an A-leaderboard model is a useful starting point, but B24 further adapts the model through:

- B-leaderboard data lists and validation protocol.
- Teacher-anchored residual refinement.
- CD/P2S gradient conflict handling.
- Fixed checkpoint and inference-strength selection.

## Ablation Insights

Experiments with larger model changes and routing-style combinations showed that local complementarity does not always translate into robust leaderboard improvement. B24 therefore favors a stable residual design with explicit movement bounds and reproducible model selection.