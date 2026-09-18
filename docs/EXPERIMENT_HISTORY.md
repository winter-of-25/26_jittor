# 实验历程

本文档从公开展示角度概述最终系统的演化过程。

## 复现基线

项目最初从 StraightPCF / IterativePFN 风格的点云去噪基线出发。初始复现系统在公开榜单上达到约 73.61，为后续模型改进建立了稳定的实现与评估流程。

## 稳定 parent 模型

V12 是第一个表现稳定的强 parent 模型，分数约为 80.70。该版本验证了 patch 级局部去噪路线适合赛题数据，并成为后续多个版本的重要初始化基础。

## 架构改进

后续版本围绕 residual refinement、cross-patch consensus、hierarchical residual modeling 和 noise-conditioned dual-path 等方向展开。代表性节点包括：

- V29：首次体现出明确的架构级提升。
- V33 / V34：增强 residual refinement 与局部一致性。
- V40 / V41：改善 CD 与 P2S 的平衡。
- V45：A 榜后期稳定 parent 模型。
- V65：A 榜最终模型。

## B 榜适配

B 榜数据开放后，系统针对新分布进行适配。早期 B 版本继承 A 榜 parent 模型，并逐步调整数据使用、固定验证和 residual strength 选择。

B20 提供了较强的 B 榜基线。B24 在此基础上引入 CAVR-v2 受约束残差设计，并加强验证与推理锁定流程，形成最终 B 榜提交系统。

## 最终设计选择

最终方案优先考虑稳定性与可复现性。B24 没有用完全自由的新分支替代 parent 模型，而是在冻结 teacher 周围训练受约束的 student residual。该设计能够在保持 teacher 稳定性的同时，继续改善局部几何细节。