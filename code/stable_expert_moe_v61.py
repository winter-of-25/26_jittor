import math

import jittor as jt
import numpy as np
from jittor import nn

from competitive_residual_v39_42 import (
    GeometryState,
    confidence_stitch,
    vector_norm,
)
from iterativepfn_v12 import batched_index
from jittor_port_models import (
    farthest_point_sampling_np,
    knn_points_np,
)
from pseudo_query_corrector_v29 import (
    InvariantTokenContext,
    LinearBNReLU,
    SparseGeoAttention,
)


MODE = "v61_stable_expert_moe"
EXPERT_NAMES = ("v45", "v54", "v60")


def _cosine(first, second):
    return (first * second).sum(-1, keepdims=True) / (
        vector_norm(first) * vector_norm(second) + 1e-8
    )


def _entropy(probability):
    count = max(2, int(probability.shape[-1]))
    return -(
        probability * jt.log(probability + 1e-8)
    ).sum(-1, keepdims=True) / math.log(float(count))


def _edge_statistics(points, indices, count):
    count = min(int(count), int(indices.shape[2]))
    selected = indices[:, :, :count]
    relative = (
        batched_index(points, selected) - points.unsqueeze(2)
    )
    distance = vector_norm(relative)
    mean = distance.mean(dim=2)
    variance = (
        (distance - mean.unsqueeze(2)) ** 2
    ).mean(dim=2)
    return mean, jt.sqrt(variance + 1e-8)


class StableExpertRouter(nn.Module):
    FEATURE_DIM = 72

    def __init__(
        self,
        max_k=32,
        channels=128,
        smooth_k=12,
        smooth_ratio=0.30,
    ):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.smooth_k = int(smooth_k)
        self.smooth_ratio = float(smooth_ratio)
        self.input = LinearBNReLU(
            self.FEATURE_DIM, self.channels
        )
        self.local1 = SparseGeoAttention(
            self.channels,
            self.channels,
            geometry_dim=GeometryState.GEOMETRY_DIM,
            key_dim=32,
        )
        self.local2 = SparseGeoAttention(
            self.channels,
            self.channels,
            geometry_dim=GeometryState.GEOMETRY_DIM,
            key_dim=32,
        )
        self.context = InvariantTokenContext(
            channels=self.channels,
            num_tokens=8,
            key_dim=32,
        )
        self.fuse = LinearBNReLU(
            3 * self.channels, self.channels
        )
        self.local_logits = nn.Linear(self.channels, 3)
        self.global_hidden = LinearBNReLU(
            self.channels, self.channels
        )
        self.global_logits = nn.Linear(self.channels, 3)
        self.confidence = nn.Linear(self.channels, 1)

        self.local_logits.weight.assign(
            jt.zeros_like(self.local_logits.weight)
        )
        self.local_logits.bias.assign(
            jt.zeros_like(self.local_logits.bias)
        )
        self.global_logits.weight.assign(
            jt.zeros_like(self.global_logits.weight)
        )
        initial_prior = np.log(
            np.asarray([0.01, 0.30, 0.69], dtype=np.float32)
        )
        self.global_logits.bias.assign(jt.array(initial_prior))
        self.confidence.weight.assign(
            jt.zeros_like(self.confidence.weight)
        )
        self.confidence.bias.assign(jt.array([3.0]))

    def _feature_tensor(
        self,
        state,
        noisy,
        candidates,
        v45_auxiliary,
        v54_auxiliary,
        v60_auxiliary,
    ):
        v45 = candidates[:, :, 0]
        v54 = candidates[:, :, 1]
        v60 = candidates[:, :, 2]
        total45 = v45 - noisy
        total54 = v54 - noisy
        total60 = v60 - noisy
        delta54 = v54 - v45
        delta60 = v60 - v45
        delta6054 = v60 - v54

        features = [
            state.initial,
            vector_norm(total45),
            vector_norm(total54),
            vector_norm(total60),
            vector_norm(delta54),
            vector_norm(delta60),
            vector_norm(delta6054),
            _cosine(delta54, delta60),
            _cosine(delta54, total45),
            _cosine(delta60, total45),
        ]

        for count in (8, 16):
            base_mean, base_std = _edge_statistics(
                v45, state.indices, count
            )
            for candidate in (v54, v60):
                mean, std = _edge_statistics(
                    candidate, state.indices, count
                )
                features.extend(
                    [
                        mean / (base_mean + 1e-8),
                        std / (base_std + 1e-8),
                    ]
                )

        features.extend(
            [
                v45_auxiliary["route"],
                v45_auxiliary["trust"],
                v45_auxiliary["agreement"],
                v54_auxiliary["transport_entropy"],
                v54_auxiliary["transport_gate"],
                jt.abs(v54_auxiliary["column_mass"] - 1.0),
                vector_norm(v54_auxiliary["update"]),
                jt.abs(
                    (
                        v60_auxiliary["raw_normal"]
                        * v60_auxiliary["filtered_normal"]
                    ).sum(-1, keepdims=True)
                ),
                _entropy(v60_auxiliary["candidate_weights"]),
                vector_norm(v60_auxiliary["update"]),
                v60_auxiliary["gate"].mean(dim=2),
                _entropy(v60_auxiliary["route"]).mean(dim=2),
            ]
        )
        tensor = jt.concat(features, dim=-1)
        if int(tensor.shape[-1]) != self.FEATURE_DIM:
            raise ValueError(
                "router feature mismatch: %d"
                % int(tensor.shape[-1])
            )
        return tensor

    def _spatial_smooth(
        self, probability, state, filtered_normal
    ):
        count = min(self.smooth_k, int(state.indices.shape[2]))
        indices = state.indices[:, :, :count]
        neighbor_probability = batched_index(
            probability, indices
        )
        neighbor_normal = batched_index(
            filtered_normal, indices
        )
        normal_similarity = jt.abs(
            (
                neighbor_normal
                * filtered_normal.unsqueeze(2)
            ).sum(-1, keepdims=True)
        )
        distance = state.base_distance[:, :, :count]
        radius = distance.mean(dim=2, keepdims=True)
        distance_weight = jt.exp(
            -distance / (radius + 1e-8)
        )
        weight = distance_weight * (
            0.10 + 0.90 * normal_similarity ** 2
        )
        neighbor_mean = (
            neighbor_probability * weight
        ).sum(dim=2) / (weight.sum(dim=2) + 1e-8)
        ratio = self.smooth_ratio
        smoothed = (
            (1.0 - ratio) * probability
            + ratio * neighbor_mean
        )
        return smoothed / (
            smoothed.sum(dim=-1, keepdims=True) + 1e-8
        )

    def execute(
        self,
        state,
        noisy,
        candidates,
        v45_auxiliary,
        v54_auxiliary,
        v60_auxiliary,
    ):
        features = self._feature_tensor(
            state,
            noisy,
            candidates,
            v45_auxiliary,
            v54_auxiliary,
            v60_auxiliary,
        )
        batch, num_points, _ = features.shape
        embedded = self.input(
            features.reshape(-1, self.FEATURE_DIM)
        ).reshape(batch, num_points, self.channels)
        indices, geometry = state.prefix(self.max_k)
        local1 = self.local1(
            embedded, indices, geometry
        )
        local2 = self.local2(
            local1, indices, geometry
        )
        context = self.context(local2)
        fused = self.fuse(
            jt.concat(
                [embedded, local2, context], dim=-1
            ).reshape(-1, 3 * self.channels)
        ).reshape(batch, num_points, self.channels)

        patch_feature = fused.mean(dim=1)
        patch_hidden = self.global_hidden(patch_feature)
        patch_logits = self.global_logits(patch_hidden)
        patch_prior = nn.softmax(patch_logits, dim=-1)
        local_logits = self.local_logits(
            fused.reshape(-1, self.channels)
        ).reshape(batch, num_points, 3)
        raw_route = nn.softmax(
            local_logits + patch_logits.unsqueeze(1),
            dim=-1,
        )
        route = self._spatial_smooth(
            raw_route,
            state,
            v60_auxiliary["filtered_normal"],
        )
        confidence = jt.sigmoid(
            self.confidence(
                fused.reshape(-1, self.channels)
            ).reshape(batch, num_points, 1)
        )
        return {
            "route": route,
            "raw_route": raw_route,
            "patch_prior": patch_prior,
            "confidence": confidence,
            "features": features,
            "indices": indices,
            "filtered_normal": v60_auxiliary[
                "filtered_normal"
            ],
        }


class FrozenStableExpertMoE(nn.Module):
    def __init__(
        self,
        v60,
        v54,
        v54_strength=0.75,
        v60_strength=1.25,
        max_k=32,
        channels=128,
        smooth_k=12,
        smooth_ratio=0.30,
    ):
        super().__init__()
        if v54.v45 is not v60.v45:
            raise ValueError("V54 and V60 must share one V45")
        self.v60 = v60
        self.v54 = v54
        self.v45 = v60.v45
        self.v54_strength = float(v54_strength)
        self.v60_strength = float(v60_strength)
        self.mode = MODE
        for parameter in self.v60.parameters():
            parameter.stop_grad()
        for parameter in self.v54.parameters():
            parameter.stop_grad()
        self.router = StableExpertRouter(
            max_k=max_k,
            channels=channels,
            smooth_k=smooth_k,
            smooth_ratio=smooth_ratio,
        )

    def trainable_parameters(self):
        return list(self.router.parameters())

    def set_frozen_eval(self):
        self.v60.eval()
        self.v60.set_frozen_eval()
        self.v54.eval()
        self.v54.set_frozen_eval()

    def frozen_experts(self, noisy):
        self.set_frozen_eval()
        with jt.no_grad():
            shared = self.v60.frozen_outputs(noisy)
            v60_output = self.v60(
                noisy,
                refinement_strength=1.0,
                frozen_outputs=shared,
            )
            v54_output = self.v54(
                noisy,
                refinement_strength=self.v54_strength,
                frozen_outputs=shared,
            )
            v45 = v60_output[1]
            v60 = v45 + self.v60_strength * (
                v60_output[4] - v45
            )
            candidates = jt.stack(
                [v45, v54_output[0], v60], dim=2
            )
        return {
            "shared": shared,
            "v60_output": v60_output,
            "v54_output": v54_output,
            "candidates": candidates,
        }

    def execute(
        self,
        noisy,
        refinement_strength=1.0,
        expert_bundle=None,
    ):
        if expert_bundle is None:
            expert_bundle = self.frozen_experts(noisy)
        shared = expert_bundle["shared"]
        candidates = expert_bundle["candidates"]
        v60_output = expert_bundle["v60_output"]
        v54_output = expert_bundle["v54_output"]
        v45 = candidates[:, :, 0]
        v29_query = shared[2]
        v12_query = shared[3]
        v45_auxiliary = shared[5]
        v29_outputs = shared[7]
        state = GeometryState(
            noisy,
            v12_query,
            v29_query,
            v45,
            v29_outputs[2],
            v29_outputs[3],
            self.router.max_k,
        )
        route_output = self.router(
            state,
            noisy,
            candidates,
            v45_auxiliary,
            v54_output[5],
            v60_output[5],
        )
        mixture = (
            candidates
            * route_output["route"].unsqueeze(-1)
        ).sum(dim=2)
        active = v45 + route_output["confidence"] * (
            mixture - v45
        )
        strength = float(refinement_strength)
        if abs(strength) < 1e-12:
            prediction = v45
        else:
            prediction = v45 + strength * (
                active - v45
            )
        auxiliary = dict(route_output)
        auxiliary.update(
            {
                "mixture": mixture,
                "active": active,
                "candidates": candidates,
                "stitch_confidence": v45_auxiliary[
                    "stitch_confidence"
                ],
                "parent_stitch_confidence": v45_auxiliary[
                    "stitch_confidence"
                ],
                "v54_auxiliary": v54_output[5],
                "v60_auxiliary": v60_output[5],
            }
        )
        return (
            prediction,
            v45,
            v29_query,
            v12_query,
            active,
            auxiliary,
            expert_bundle,
            v29_outputs,
        )


def patch_based_v61_pair(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    beta=12.0,
    patch_batch=6,
):
    original = np.asarray(points, dtype=np.float32)
    num_points = original.shape[0]
    num_patches = max(
        1, int(seed_k * num_points / patch_size)
    )
    seed_indices = farthest_point_sampling_np(
        original, num_patches
    )
    seeds = original[seed_indices]
    patch_distances, point_indices, patches = knn_points_np(
        seeds, original, min(patch_size, num_points)
    )
    centered = patches - seeds[:, None, :]
    normalized_distances = patch_distances / (
        patch_distances[:, -1:] + 1e-8
    )
    base_predictions = []
    active_predictions = []
    confidences = []
    with jt.no_grad():
        for start in range(0, num_patches, int(patch_batch)):
            batch = jt.array(
                centered[
                    start : start + int(patch_batch)
                ].astype(np.float32)
            )
            output = model(
                batch, refinement_strength=1.0
            )
            base_predictions.append(output[1].numpy())
            active_predictions.append(output[4].numpy())
            confidences.append(
                output[5]["stitch_confidence"].numpy()
            )
    base_predictions = (
        np.concatenate(base_predictions, axis=0)
        + seeds[:, None, :]
    )
    active_predictions = (
        np.concatenate(active_predictions, axis=0)
        + seeds[:, None, :]
    )
    confidences = np.concatenate(confidences, axis=0)
    base = confidence_stitch(
        base_predictions,
        confidences,
        point_indices,
        normalized_distances,
        original,
        beta,
    )
    active = confidence_stitch(
        active_predictions,
        confidences,
        point_indices,
        normalized_distances,
        original,
        beta,
    )
    return base, active


def patch_based_v61(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    beta=12.0,
    patch_batch=6,
    refinement_strength=1.0,
):
    base, active = patch_based_v61_pair(
        model,
        points,
        patch_size=patch_size,
        seed_k=seed_k,
        beta=beta,
        patch_batch=patch_batch,
    )
    return base + float(refinement_strength) * (
        active - base
    )
