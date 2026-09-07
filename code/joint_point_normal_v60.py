import math

import jittor as jt
import numpy as np
from jittor import nn

from competitive_residual_v39_42 import (
    GeometryState,
    bounded_vector,
    confidence_stitch,
    vector_norm,
)
from iterativepfn_v12 import batched_index, knn_indices
from jittor_port_models import (
    farthest_point_sampling_np,
    knn_points_np,
)
from pseudo_query_corrector_v29 import LinearBNReLU


MODE = "v60_joint_point_normal_filter"


def _cross(first, second):
    return jt.stack(
        [
            first[..., 1] * second[..., 2]
            - first[..., 2] * second[..., 1],
            first[..., 2] * second[..., 0]
            - first[..., 0] * second[..., 2],
            first[..., 0] * second[..., 1]
            - first[..., 1] * second[..., 0],
        ],
        dim=-1,
    )


def _unit(value):
    return value / (vector_norm(value) + 1e-8)


def _cosine(first, second):
    return (first * second).sum(-1, keepdims=True) / (
        vector_norm(first) * vector_norm(second) + 1e-8
    )


def _align_unoriented(neighbor, center):
    dot = (neighbor * center).sum(-1, keepdims=True)
    sign = jt.where(
        dot >= 0.0,
        jt.ones_like(dot),
        -jt.ones_like(dot),
    )
    return neighbor * sign


class CandidateFrameEstimator(nn.Module):
    """Predict an unoriented local normal from equivariant cross bases."""

    def __init__(self, feature_dim=42, hidden=80):
        super().__init__()
        self.hidden = LinearBNReLU(feature_dim + 4, hidden)
        self.score = nn.Linear(hidden, 1)

    def execute(self, state):
        offsets = state.base_relative
        k = int(offsets.shape[2])
        pairs = (
            (1, max(3, k // 4)),
            (2, max(5, k // 2)),
            (3, max(7, 3 * k // 4)),
            (4, max(9, k - 1)),
            (max(2, k // 5), max(6, 2 * k // 3)),
            (max(3, k // 3), max(8, 5 * k // 6)),
        )
        candidates = []
        invariants = []
        for first_index, second_index in pairs:
            first_index = min(first_index, k - 1)
            second_index = min(second_index, k - 1)
            first = offsets[:, :, first_index]
            second = offsets[:, :, second_index]
            cross = _cross(first, second)
            first_norm = vector_norm(first)
            second_norm = vector_norm(second)
            cross_norm = vector_norm(cross)
            cosine = jt.abs(_cosine(first, second))
            candidates.append(_unit(cross))
            invariants.append(
                jt.concat(
                    [
                        first_norm / (state.base_scale[:, :, 0] + 1e-8),
                        second_norm / (state.base_scale[:, :, 0] + 1e-8),
                        cross_norm
                        / (first_norm * second_norm + 1e-8),
                        cosine,
                    ],
                    dim=-1,
                )
            )
        candidates = jt.stack(candidates, dim=2)
        reference = candidates[:, :, 0:1]
        candidates = _align_unoriented(candidates, reference)
        invariant = jt.stack(invariants, dim=2)
        batch, num_points, count = invariant.shape[:3]
        initial = state.initial.unsqueeze(2).broadcast(
            (batch, num_points, count, state.initial.shape[-1])
        )
        score_input = jt.concat([initial, invariant], dim=-1)
        logits = self.score(
            self.hidden(
                score_input.reshape(-1, score_input.shape[-1])
            )
        ).reshape(batch, num_points, count)
        weights = nn.softmax(logits, dim=2)
        raw_normal = _unit(
            (weights.unsqueeze(-1) * candidates).sum(dim=2)
        )
        return raw_normal, weights, candidates


class NormalFilterBranch(nn.Module):
    def __init__(self, channels=112, normal_k=24):
        super().__init__()
        self.channels = int(channels)
        self.normal_k = int(normal_k)
        self.input_proj = LinearBNReLU(42, self.channels)
        self.edge_hidden = LinearBNReLU(
            2 * self.channels + 5, self.channels
        )
        self.edge_value = nn.Linear(self.channels, self.channels)
        self.edge_score = nn.Linear(self.channels, 1)
        self.fuse = LinearBNReLU(
            2 * self.channels, self.channels
        )

    def execute(self, state, raw_normal):
        count = min(self.normal_k, state.max_k)
        indices = state.indices[:, :, :count]
        relative = state.base_relative[:, :, :count]
        distance = state.base_distance[:, :, :count]
        batch, num_points = state.base_query.shape[:2]
        features = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, num_points, self.channels)
        neighbors = batched_index(features, indices)
        centers = features.unsqueeze(2).broadcast(
            (
                batch,
                num_points,
                count,
                self.channels,
            )
        )
        neighbor_normal = batched_index(raw_normal, indices)
        center_normal = raw_normal.unsqueeze(2).broadcast(
            (batch, num_points, count, 3)
        )
        aligned_normal = _align_unoriented(
            neighbor_normal, center_normal
        )
        direction = relative / (distance + 1e-8)
        normal_dot = jt.abs(
            (center_normal * aligned_normal).sum(
                -1, keepdims=True
            )
        )
        center_incidence = jt.abs(
            (direction * center_normal).sum(
                -1, keepdims=True
            )
        )
        neighbor_incidence = jt.abs(
            (direction * aligned_normal).sum(
                -1, keepdims=True
            )
        )
        edge_geo = jt.concat(
            [
                distance / (state.base_scale + 1e-8),
                normal_dot,
                center_incidence,
                neighbor_incidence,
                1.0 - normal_dot,
            ],
            dim=-1,
        )
        edge = jt.concat(
            [centers, neighbors - centers, edge_geo],
            dim=-1,
        )
        hidden = self.edge_hidden(
            edge.reshape(-1, edge.shape[-1])
        ).reshape(
            batch, num_points, count, self.channels
        )
        attention = nn.softmax(
            self.edge_score(
                hidden.reshape(-1, self.channels)
            ).reshape(batch, num_points, count),
            dim=2,
        )
        filtered_normal = _unit(
            (
                attention.unsqueeze(-1) * aligned_normal
            ).sum(dim=2)
        )
        context = (
            attention.unsqueeze(-1)
            * self.edge_value(
                hidden.reshape(-1, self.channels)
            ).reshape(
                batch, num_points, count, self.channels
            )
        ).sum(dim=2)
        filtered_features = self.fuse(
            jt.concat([features, context], dim=-1).reshape(
                -1, 2 * self.channels
            )
        ).reshape(batch, num_points, self.channels)
        return filtered_normal, filtered_features, attention


def _frame_geometry(state, normal, count, current=None):
    count = min(int(count), state.max_k)
    indices = state.indices[:, :, :count]
    if current is None:
        relative = state.base_relative[:, :, :count]
    else:
        relative = (
            batched_index(current, indices)
            - current.unsqueeze(2)
        )
    distance = vector_norm(relative)
    direction = relative / (distance + 1e-8)
    neighbor_normal = batched_index(normal, indices)
    center_normal = normal.unsqueeze(2).broadcast(
        (
            normal.shape[0],
            normal.shape[1],
            count,
            3,
        )
    )
    aligned_normal = _align_unoriented(
        neighbor_normal, center_normal
    )
    normal_dot = jt.abs(
        (center_normal * aligned_normal).sum(
            -1, keepdims=True
        )
    )
    center_incidence = jt.abs(
        (direction * center_normal).sum(
            -1, keepdims=True
        )
    )
    neighbor_incidence = jt.abs(
        (direction * aligned_normal).sum(
            -1, keepdims=True
        )
    )
    return jt.concat(
        [
            state.geometry[:, :, :count],
            normal_dot,
            center_incidence,
            neighbor_incidence,
            1.0 - normal_dot,
        ],
        dim=-1,
    )


class FrameAwareMessage(nn.Module):
    def __init__(self, channels, geometry_dim=13):
        super().__init__()
        self.channels = int(channels)
        self.edge_hidden = LinearBNReLU(
            2 * self.channels + int(geometry_dim),
            self.channels,
        )
        self.edge_value = nn.Linear(self.channels, self.channels)
        self.edge_score = nn.Linear(self.channels, 1)
        self.fuse = LinearBNReLU(
            2 * self.channels, self.channels
        )

    def execute(self, features, indices, geometry):
        batch, num_points, channels = features.shape
        count = int(indices.shape[2])
        neighbors = batched_index(features, indices)
        centers = features.unsqueeze(2).broadcast(
            (batch, num_points, count, channels)
        )
        edge = jt.concat(
            [centers, neighbors - centers, geometry],
            dim=-1,
        )
        hidden = self.edge_hidden(
            edge.reshape(-1, edge.shape[-1])
        ).reshape(batch, num_points, count, channels)
        attention = nn.softmax(
            self.edge_score(
                hidden.reshape(-1, channels)
            ).reshape(batch, num_points, count),
            dim=2,
        )
        context = (
            attention.unsqueeze(-1)
            * self.edge_value(
                hidden.reshape(-1, channels)
            ).reshape(batch, num_points, count, channels)
        ).sum(dim=2)
        return self.fuse(
            jt.concat([features, context], dim=-1).reshape(
                -1, 2 * channels
            )
        ).reshape(batch, num_points, channels), attention


class NormalGuidedField(nn.Module):
    def __init__(self, channels, geometry_dim=13):
        super().__init__()
        self.channels = int(channels)
        self.edge_hidden = LinearBNReLU(
            2 * self.channels + int(geometry_dim),
            self.channels,
        )
        self.edge_output = nn.Linear(self.channels, 8)
        self.edge_output.weight.assign(
            jt.zeros_like(self.edge_output.weight)
        )
        self.edge_output.bias.assign(
            jt.zeros_like(self.edge_output.bias)
        )

    def execute(
        self,
        features,
        state,
        normal,
        current,
        count,
    ):
        count = min(int(count), state.max_k)
        indices = state.indices[:, :, :count]
        batch, num_points, channels = features.shape
        neighbors = batched_index(features, indices)
        centers = features.unsqueeze(2).broadcast(
            (batch, num_points, count, channels)
        )
        geometry = _frame_geometry(
            state, normal, count, current=current
        )
        hidden = self.edge_hidden(
            jt.concat(
                [centers, neighbors - centers, geometry],
                dim=-1,
            ).reshape(-1, 2 * channels + 13)
        ).reshape(batch, num_points, count, channels)
        output = self.edge_output(
            hidden.reshape(-1, channels)
        ).reshape(batch, num_points, count, 8)
        coefficients = jt.tanh(output[..., :7])
        importance = nn.softmax(output[..., 7], dim=2)
        current_relative = (
            batched_index(current, indices)
            - current.unsqueeze(2)
        )
        center_normal = normal.unsqueeze(2).broadcast(
            (batch, num_points, count, 3)
        )
        normal_relative = (
            current_relative * center_normal
        ).sum(-1, keepdims=True) * center_normal
        tangent_relative = current_relative - normal_relative
        noisy_relative = (
            batched_index(state.noisy, indices)
            - state.noisy.unsqueeze(2)
        )
        v29_relative = (
            batched_index(state.v29_query, indices)
            - state.v29_query.unsqueeze(2)
        )

        def broadcast_center(value):
            return value.unsqueeze(2).broadcast(
                (batch, num_points, count, 3)
            )

        bases = jt.stack(
            [
                normal_relative,
                tangent_relative,
                noisy_relative - current_relative,
                v29_relative - current_relative,
                broadcast_center(
                    state.base_query - state.noisy
                ),
                broadcast_center(state.base_delta),
                broadcast_center(state.parent_update),
            ],
            dim=3,
        )
        edge_field = (
            coefficients.unsqueeze(-1) * bases
        ).sum(dim=3)
        field = (
            importance.unsqueeze(-1) * edge_field
        ).sum(dim=2)
        return field, coefficients, importance


class JointPointNormalHead(nn.Module):
    def __init__(
        self,
        max_k=40,
        channels=144,
        normal_channels=112,
        num_steps=2,
    ):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        self.frame = CandidateFrameEstimator()
        self.normal_filter = NormalFilterBranch(
            channels=normal_channels,
            normal_k=24,
        )
        self.input_proj = LinearBNReLU(
            42 + normal_channels + 3,
            self.channels,
        )
        self.message1 = FrameAwareMessage(self.channels)
        self.message2 = FrameAwareMessage(self.channels)
        self.step_condition = nn.Linear(1, self.channels)
        self.fields = nn.ModuleList(
            [
                NormalGuidedField(self.channels)
                for _ in (12, 24, self.max_k)
            ]
        )
        self.route = nn.Linear(self.channels, 4)
        self.route.weight.assign(jt.zeros_like(self.route.weight))
        self.route.bias.assign(jt.zeros_like(self.route.bias))

    def execute(self, state):
        batch, num_points = state.base_query.shape[:2]
        raw_normal, candidate_weights, candidates = self.frame(
            state
        )
        (
            filtered_normal,
            normal_features,
            normal_attention,
        ) = self.normal_filter(state, raw_normal)
        frame_agreement = jt.abs(
            (raw_normal * filtered_normal).sum(
                -1, keepdims=True
            )
        )
        curvature = (
            jt.abs(
                state.base_relative[:, :, :16]
                * filtered_normal.unsqueeze(2)
            ).sum(-1)
            / (
                state.base_distance[:, :, :16, 0]
                + 1e-8
            )
        ).mean(dim=2, keepdims=True)
        normal_entropy = -(
            candidate_weights
            * jt.log(candidate_weights + 1e-8)
        ).sum(dim=2, keepdims=True) / math.log(
            float(candidate_weights.shape[2])
        )
        features = self.input_proj(
            jt.concat(
                [
                    state.initial,
                    normal_features,
                    frame_agreement,
                    curvature,
                    normal_entropy,
                ],
                dim=-1,
            ).reshape(
                -1, 42 + normal_features.shape[-1] + 3
            )
        ).reshape(batch, num_points, self.channels)
        geometry = _frame_geometry(
            state, filtered_normal, self.max_k
        )
        features, attention1 = self.message1(
            features, state.indices, geometry
        )
        features, attention2 = self.message2(
            features, state.indices, geometry
        )
        current = state.base_query
        step_predictions = []
        step_routes = []
        step_gates = []
        branch_fields = []
        for step in range(self.num_steps):
            step_value = jt.ones(
                (batch, num_points, 1)
            ) * float(step + 1) / float(self.num_steps)
            conditioned = features + self.step_condition(
                step_value.reshape(-1, 1)
            ).reshape(batch, num_points, self.channels)
            scales = []
            for field, count in zip(
                self.fields, (12, 24, self.max_k)
            ):
                value, _, _ = field(
                    conditioned,
                    state,
                    filtered_normal,
                    current,
                    count,
                )
                scales.append(value)
            route_output = self.route(
                conditioned.reshape(-1, self.channels)
            ).reshape(batch, num_points, 4)
            route = nn.softmax(route_output[..., :3], dim=-1)
            gate = jt.sigmoid(route_output[..., 3:4])
            mixed = (
                jt.stack(scales, dim=2)
                * route.unsqueeze(-1)
            ).sum(dim=2)
            radius = state.base_distance[
                :, :, :16
            ].mean(dim=2)
            cap = (
                0.18 * radius
                + 0.12 * vector_norm(state.parent_update)
                + 0.03
                * vector_norm(
                    state.base_query - state.noisy
                )
                + 1e-5
            ) / float(self.num_steps)
            update = bounded_vector(
                gate * mixed / float(self.num_steps),
                cap,
            )
            current = current + update
            step_predictions.append(current)
            step_routes.append(route)
            step_gates.append(gate)
            branch_fields.append(jt.stack(scales, dim=2))
        return current, {
            "raw_normal": raw_normal,
            "filtered_normal": filtered_normal,
            "candidate_normals": candidates,
            "candidate_weights": candidate_weights,
            "normal_attention": normal_attention,
            "message_attention1": attention1,
            "message_attention2": attention2,
            "step_predictions": step_predictions,
            "route": jt.stack(step_routes, dim=2),
            "gate": jt.stack(step_gates, dim=2),
            "branch_fields": jt.stack(branch_fields, dim=2),
            "update": current - state.base_query,
        }


class FrozenJointPointNormalDenoiser(nn.Module):
    def __init__(
        self,
        v45_model,
        v45_strength=1.25,
        max_k=40,
        channels=144,
        normal_channels=112,
        num_steps=2,
    ):
        super().__init__()
        self.v45 = v45_model
        self.v45_strength = float(v45_strength)
        self.mode = MODE
        for parameter in self.v45.parameters():
            parameter.stop_grad()
        self.head = JointPointNormalHead(
            max_k=max_k,
            channels=channels,
            normal_channels=normal_channels,
            num_steps=num_steps,
        )

    def trainable_parameters(self):
        return list(self.head.parameters())

    def set_frozen_eval(self):
        self.v45.eval()
        self.v45.set_frozen_eval()

    def frozen_outputs(self, noisy):
        self.set_frozen_eval()
        with jt.no_grad():
            lower = self.v45.frozen_outputs(noisy)
            output = self.v45(
                noisy,
                refinement_strength=self.v45_strength,
                frozen_outputs=lower,
            )
        return output

    def execute(
        self,
        noisy,
        refinement_strength=1.0,
        frozen_outputs=None,
    ):
        if frozen_outputs is None:
            frozen_outputs = self.frozen_outputs(noisy)
        v45_outputs = frozen_outputs
        base_query = v45_outputs[0]
        v41_query = v45_outputs[1]
        v29_query = v45_outputs[2]
        v12_query = v45_outputs[3]
        v45_auxiliary = v45_outputs[5]
        v29_outputs = v45_outputs[7]
        state = GeometryState(
            noisy,
            v12_query,
            v29_query,
            base_query,
            v29_outputs[2],
            v29_outputs[3],
            self.head.max_k,
        )
        state.parent_update = base_query - v41_query
        state.parent_auxiliary = v45_auxiliary
        refined, auxiliary = self.head(state)
        strength = float(refinement_strength)
        if abs(strength) < 1e-12:
            prediction = base_query
        else:
            prediction = base_query + strength * (
                refined - base_query
            )
        stitch_confidence = v45_auxiliary[
            "stitch_confidence"
        ]
        auxiliary["stitch_confidence"] = stitch_confidence
        auxiliary["parent_stitch_confidence"] = (
            stitch_confidence
        )
        return (
            prediction,
            base_query,
            v29_query,
            v12_query,
            refined,
            auxiliary,
            v45_outputs,
            v29_outputs,
        )


def patch_based_v60_pair(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    beta=12.0,
    patch_batch=8,
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
            output = model(batch, refinement_strength=1.0)
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


def patch_based_v60(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    beta=12.0,
    patch_batch=8,
    refinement_strength=1.0,
):
    base, active = patch_based_v60_pair(
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
