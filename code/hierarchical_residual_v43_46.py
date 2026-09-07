import math

import jittor as jt
import numpy as np
from jittor import nn

from competitive_residual_v39_42 import (
    DenseEquivariantField,
    FrozenBaseCompetitor,
    GeometryState,
    LocalCrossAttention,
    bounded_vector,
    confidence_stitch,
    vector_norm,
)
from iterativepfn_v12 import batched_index, knn_indices
from jittor_port_models import farthest_point_sampling_np, knn_points_np
from pseudo_query_corrector_v29 import (
    InvariantTokenContext,
    LinearBNReLU,
    SparseGeoAttention,
)


SUPPORTED_MODES = (
    "v43_robust_spectral_shrinkage",
    "v44_redescending_conservative_flux",
    "v45_dual_topology_consensus",
    "v46_reversible_dynamic_bridge",
)


def _cosine(first, second):
    return (first * second).sum(-1, keepdims=True) / (
        vector_norm(first) * vector_norm(second) + 1e-8
    )


def _graph_fields(state, current, indices):
    current_relative = (
        batched_index(current, indices) - current.unsqueeze(2)
    )
    noisy_relative = (
        batched_index(state.noisy, indices)
        - state.noisy.unsqueeze(2)
    )
    v29_relative = (
        batched_index(state.v29_query, indices)
        - state.v29_query.unsqueeze(2)
    )
    base_delta_relative = (
        batched_index(state.base_delta, indices)
        - state.base_delta.unsqueeze(2)
    )
    correction_relative = (
        batched_index(state.correction, indices)
        - state.correction.unsqueeze(2)
    )
    current_distance = vector_norm(current_relative)
    noisy_distance = vector_norm(noisy_relative)
    v29_distance = vector_norm(v29_relative)
    current_scale = current_distance.mean(dim=2, keepdims=True)
    noisy_scale = noisy_distance.mean(dim=2, keepdims=True)
    v29_scale = v29_distance.mean(dim=2, keepdims=True)
    strain_noisy = (
        current_distance - noisy_distance
    ) / (noisy_scale + 1e-8)
    strain_v29 = (
        current_distance - v29_distance
    ) / (v29_scale + 1e-8)
    geometry = jt.concat(
        [
            current_distance / (current_scale + 1e-8),
            noisy_distance / (noisy_scale + 1e-8),
            v29_distance / (v29_scale + 1e-8),
            strain_noisy,
            strain_v29,
            _cosine(current_relative, noisy_relative),
            _cosine(current_relative, v29_relative),
            vector_norm(base_delta_relative)
            / (current_scale + 1e-8),
            vector_norm(correction_relative)
            / (current_scale + 1e-8),
        ],
        dim=-1,
    )
    return {
        "current_relative": current_relative,
        "noisy_relative": noisy_relative,
        "v29_relative": v29_relative,
        "base_delta_relative": base_delta_relative,
        "correction_relative": correction_relative,
        "base_delta": state.base_delta,
        "distance": current_distance,
        "base_distance": current_distance,
        "geometry": geometry,
        "indices": indices,
    }


def _weighted_laplacian(points, indices):
    relative = batched_index(points, indices) - points.unsqueeze(2)
    distance = vector_norm(relative)
    radius = distance.mean(dim=2, keepdims=True)
    weights = jt.exp(-distance / (radius + 1e-6))
    weights = weights / (
        weights.sum(dim=2, keepdims=True) + 1e-8
    )
    return (weights * relative).sum(dim=2)


def _smooth_features(features, indices, distance):
    neighbors = batched_index(features, indices)
    radius = distance.mean(dim=2, keepdims=True)
    weights = jt.exp(-distance / (radius + 1e-6))
    weights = weights / (
        weights.sum(dim=2, keepdims=True) + 1e-8
    )
    return (weights * neighbors).sum(dim=2)


class RobustSpectralShrinkageHead(nn.Module):
    def __init__(self, max_k=32, channels=112):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.input_proj = LinearBNReLU(42, 72)
        self.encoder = SparseGeoAttention(
            72, self.channels, geometry_dim=9
        )
        self.band_fuse = LinearBNReLU(
            4 * self.channels + 42, self.channels
        )
        self.context = InvariantTokenContext(
            self.channels, num_tokens=6
        )
        self.output = nn.Linear(
            2 * self.channels, 10
        )
        self.output.weight.assign(
            jt.zeros_like(self.output.weight)
        )
        self.output.bias.assign(
            jt.zeros_like(self.output.bias)
        )

    def execute(self, state):
        batch, num_points = state.base_query.shape[:2]
        indices, geometry = state.prefix(self.max_k)
        projected = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, num_points, 72)
        encoded = self.encoder(projected, indices, geometry)
        distance = state.base_distance[:, :, : self.max_k]
        smooth1 = _smooth_features(encoded, indices, distance)
        smooth2 = _smooth_features(smooth1, indices, distance)
        smooth3 = _smooth_features(smooth2, indices, distance)
        low_feature = smooth3
        mid_feature = smooth1 - smooth3
        high_feature = encoded - smooth1
        fused = self.band_fuse(
            jt.concat(
                [
                    encoded,
                    low_feature,
                    mid_feature,
                    high_feature,
                    state.initial,
                ],
                dim=-1,
            ).reshape(-1, 4 * self.channels + 42)
        ).reshape(batch, num_points, self.channels)
        context = self.context(fused)
        output = self.output(
            jt.concat([fused, context], dim=-1).reshape(
                -1, 2 * self.channels
            )
        ).reshape(batch, num_points, 10)
        coefficients = jt.tanh(output[..., :6])
        threshold_fraction = (
            0.02 + 0.18 * jt.sigmoid(output[..., 6:9])
        )
        gate = jt.sigmoid(output[..., 9:10])

        lap8 = _weighted_laplacian(
            state.base_query, state.indices[:, :, :8]
        )
        lap16 = _weighted_laplacian(
            state.base_query, state.indices[:, :, :16]
        )
        lap32 = _weighted_laplacian(
            state.base_query,
            state.indices[:, :, : self.max_k],
        )
        vector_bands = jt.stack(
            [lap32, lap16 - lap32, lap8 - lap16],
            dim=2,
        )
        radius = state.base_distance[
            :, :, :16
        ].mean(dim=2)
        thresholds = (
            threshold_fraction * radius
        ).unsqueeze(-1)
        band_norms = vector_norm(
            vector_bands, keepdims=False
        ).unsqueeze(-1)
        shrink = jt.maximum(
            band_norms - thresholds,
            jt.zeros_like(band_norms),
        ) / (band_norms + 1e-8)
        shrunk_bands = vector_bands * shrink
        parent_update = getattr(
            state,
            "parent_update",
            jt.zeros_like(state.base_query),
        )
        bases = jt.concat(
            [
                shrunk_bands,
                parent_update.unsqueeze(2),
                (state.v29_query - state.v12_query).unsqueeze(2),
                state.base_delta.unsqueeze(2),
            ],
            dim=2,
        )
        update = (
            coefficients.unsqueeze(-1) * bases
        ).sum(dim=2) / 6.0
        cap = (
            0.055 * radius
            + 0.040 * vector_norm(parent_update)
            + 1e-5
        )
        update = bounded_vector(gate * update, cap)
        return state.base_query + update, {
            "coefficients": coefficients,
            "thresholds": threshold_fraction,
            "band_shrink": shrink[..., 0],
            "band_vectors": vector_bands,
            "update_gate": gate,
            "update": update,
        }


class RedescendingConservativeFluxHead(nn.Module):
    def __init__(
        self,
        max_k=32,
        channels=104,
        num_steps=2,
    ):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        self.input_proj = LinearBNReLU(42, 72)
        self.encoder = SparseGeoAttention(
            72, self.channels, geometry_dim=9
        )
        self.context = InvariantTokenContext(
            self.channels, num_tokens=6
        )
        self.fuse = LinearBNReLU(
            2 * self.channels + 42, self.channels
        )
        self.edge_hidden = LinearBNReLU(
            2 * self.channels + 9, self.channels
        )
        self.edge_output = nn.Linear(self.channels, 8)
        self.route = nn.Linear(self.channels, 3)
        self.edge_output.weight.assign(
            jt.zeros_like(self.edge_output.weight)
        )
        self.edge_output.bias.assign(
            jt.zeros_like(self.edge_output.bias)
        )
        self.route.weight.assign(jt.zeros_like(self.route.weight))
        self.route.bias.assign(jt.zeros_like(self.route.bias))

    def _flux(self, features, fields):
        indices = fields["indices"]
        geometry = fields["geometry"]
        batch, num_points, channels = features.shape
        k = int(indices.shape[2])
        neighbors = batched_index(features, indices)
        centers = features.unsqueeze(2).broadcast(
            (batch, num_points, k, channels)
        )
        edge = jt.concat(
            [centers, neighbors - centers, geometry],
            dim=-1,
        )
        output = self.edge_output(
            self.edge_hidden(
                edge.reshape(-1, edge.shape[-1])
            )
        ).reshape(batch, num_points, k, 8)
        coefficients = jt.tanh(output[..., :6])
        pressure = jt.tanh(output[..., 6:7])
        learned_scale = (
            0.20 + 0.80 * jt.sigmoid(output[..., 7:8])
        )
        strain = jt.abs(geometry[..., 3:5]).mean(
            dim=-1, keepdims=True
        )
        robust_weight = 1.0 / (
            1.0 + (strain / learned_scale) ** 4
        )
        importance = robust_weight / (
            robust_weight.sum(dim=2, keepdims=True) + 1e-8
        )
        center_delta = fields["base_delta"].unsqueeze(2).broadcast(
            (batch, num_points, k, 3)
        )
        bases = jt.stack(
            [
                fields["current_relative"],
                fields["noisy_relative"],
                fields["v29_relative"],
                fields["base_delta_relative"],
                fields["correction_relative"],
                center_delta,
            ],
            dim=3,
        )
        surface_edge = (
            coefficients.unsqueeze(-1) * bases
        ).sum(dim=3)
        surface = (
            importance * surface_edge
        ).sum(dim=2)
        distance = fields["distance"]
        radius = distance.mean(dim=2, keepdims=True)
        radial_error = (
            radius - distance
        ) / (radius + 1e-6)
        direction = fields["current_relative"] / (
            distance + 1e-6
        )
        coverage = (
            importance
            * pressure
            * radial_error
            * direction
        ).sum(dim=2) * radius.reshape(
            batch, num_points, 1
        )
        surface = surface - surface.mean(dim=1, keepdims=True)
        coverage = coverage - coverage.mean(dim=1, keepdims=True)
        return surface, coverage, robust_weight

    def execute(self, state):
        batch, num_points = state.base_query.shape[:2]
        indices, geometry = state.prefix(self.max_k)
        projected = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, num_points, 72)
        encoded = self.encoder(projected, indices, geometry)
        context = self.context(encoded)
        features = self.fuse(
            jt.concat(
                [encoded, context, state.initial], dim=-1
            ).reshape(-1, 2 * self.channels + 42)
        ).reshape(batch, num_points, self.channels)
        route_output = self.route(
            features.reshape(-1, self.channels)
        ).reshape(batch, num_points, 3)
        route = nn.softmax(route_output[..., :2], dim=-1)
        gate = jt.sigmoid(route_output[..., 2:3])
        current = state.base_query
        step_predictions = []
        robust_weights = []
        surface_fields = []
        coverage_fields = []
        for _ in range(self.num_steps):
            fields = _graph_fields(state, current, state.indices)
            surface, coverage, robust = self._flux(
                features, fields
            )
            mixed = (
                route[..., 0:1] * surface
                + route[..., 1:2] * coverage
            )
            radius = fields["distance"][
                :, :, :16
            ].mean(dim=2)
            cap = (
                0.050 * radius
                + 0.035 * vector_norm(state.parent_update)
                + 1e-5
            ) / float(self.num_steps)
            update = bounded_vector(
                gate * mixed / float(self.num_steps),
                cap,
            )
            current = current + update
            step_predictions.append(current)
            robust_weights.append(robust)
            surface_fields.append(surface)
            coverage_fields.append(coverage)
        return current, {
            "step_predictions": step_predictions,
            "route": route,
            "update_gate": gate,
            "robust_weight": jt.stack(
                robust_weights, dim=2
            ),
            "surface_field": jt.stack(
                surface_fields, dim=2
            ),
            "coverage_field": jt.stack(
                coverage_fields, dim=2
            ),
            "update": current - state.base_query,
        }


class DualTopologyConsensusHead(nn.Module):
    def __init__(self, max_k=32, channels=104):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.input_proj = LinearBNReLU(42, 72)
        self.shared_encoder = SparseGeoAttention(
            72, self.channels, geometry_dim=9
        )
        self.cross = LocalCrossAttention(
            self.channels, key_dim=32
        )
        self.shared_field = DenseEquivariantField(
            self.channels
        )
        self.route_fuse = LinearBNReLU(
            3 * self.channels + 42, self.channels
        )
        self.route = nn.Linear(self.channels, 3)
        self.route.weight.assign(jt.zeros_like(self.route.weight))
        self.route.bias.assign(jt.zeros_like(self.route.bias))

    def execute(self, state):
        batch, num_points = state.base_query.shape[:2]
        base_indices = state.indices
        noisy_indices = knn_indices(state.noisy, self.max_k)
        base_fields = _graph_fields(
            state, state.base_query, base_indices
        )
        noisy_fields = _graph_fields(
            state, state.base_query, noisy_indices
        )
        projected = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, num_points, 72)
        base_features = self.shared_encoder(
            projected,
            base_indices,
            base_fields["geometry"],
        )
        noisy_features = self.shared_encoder(
            projected,
            noisy_indices,
            noisy_fields["geometry"],
        )
        base_cross, _ = self.cross(
            base_features,
            noisy_features,
            base_indices,
            base_fields["geometry"],
        )
        noisy_cross, _ = self.cross(
            noisy_features,
            base_features,
            noisy_indices,
            noisy_fields["geometry"],
        )
        base_field, _, _ = self.shared_field(
            base_cross, base_fields
        )
        noisy_field, _, _ = self.shared_field(
            noisy_cross, noisy_fields
        )
        difference = jt.abs(base_cross - noisy_cross)
        route_features = self.route_fuse(
            jt.concat(
                [
                    base_cross,
                    noisy_cross,
                    difference,
                    state.initial,
                ],
                dim=-1,
            ).reshape(-1, 3 * self.channels + 42)
        ).reshape(batch, num_points, self.channels)
        route_output = self.route(
            route_features.reshape(-1, self.channels)
        ).reshape(batch, num_points, 3)
        route = nn.softmax(route_output[..., :2], dim=-1)
        trust = jt.sigmoid(route_output[..., 2:3])
        agreement = 0.5 + 0.5 * _cosine(
            base_field, noisy_field
        )
        mixed = (
            route[..., 0:1] * base_field
            + route[..., 1:2] * noisy_field
        )
        radius = state.base_distance[
            :, :, :16
        ].mean(dim=2)
        cap = (
            0.055 * radius
            + 0.040 * vector_norm(state.parent_update)
            + 1e-5
        )
        update = bounded_vector(
            trust * (0.25 + 0.75 * agreement) * mixed,
            cap,
        )
        return state.base_query + update, {
            "route": route,
            "trust": trust,
            "agreement": agreement,
            "base_field": base_field,
            "noisy_field": noisy_field,
            "feature_disagreement": difference,
            "update": update,
        }


class ReversibleDynamicBridgeHead(nn.Module):
    def __init__(
        self,
        max_k=24,
        channels=96,
        num_steps=2,
    ):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        self.input_proj = LinearBNReLU(42, self.channels)
        self.step_condition = nn.Linear(1, self.channels)
        self.shared_encoder = SparseGeoAttention(
            self.channels,
            self.channels,
            geometry_dim=9,
        )
        self.shared_forward = DenseEquivariantField(
            self.channels
        )
        self.shared_reverse = DenseEquivariantField(
            self.channels
        )
        self.route_fuse = LinearBNReLU(
            3 * self.channels, self.channels
        )
        self.route = nn.Linear(self.channels, 3)
        self.route.weight.assign(jt.zeros_like(self.route.weight))
        self.route.bias.assign(jt.zeros_like(self.route.bias))

    def execute(self, state):
        batch, num_points = state.base_query.shape[:2]
        projected = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, num_points, self.channels)
        fixed_indices = state.indices[:, :, : self.max_k]
        current = state.base_query
        step_predictions = []
        routes = []
        agreements = []
        forward_fields = []
        reverse_fields = []
        for step in range(self.num_steps):
            topology_source = (
                state.noisy if step == 0 else current
            )
            dynamic_indices = knn_indices(
                topology_source, self.max_k
            )
            fixed_fields = _graph_fields(
                state, current, fixed_indices
            )
            dynamic_fields = _graph_fields(
                state, current, dynamic_indices
            )
            step_value = jt.ones(
                (batch, num_points, 1)
            ) * float(step + 1) / float(self.num_steps)
            conditioned = projected + self.step_condition(
                step_value.reshape(-1, 1)
            ).reshape(batch, num_points, self.channels)
            fixed_features = self.shared_encoder(
                conditioned,
                fixed_indices,
                fixed_fields["geometry"],
            )
            dynamic_features = self.shared_encoder(
                conditioned,
                dynamic_indices,
                dynamic_fields["geometry"],
            )
            fused = self.route_fuse(
                jt.concat(
                    [
                        fixed_features,
                        dynamic_features,
                        jt.abs(
                            fixed_features - dynamic_features
                        ),
                    ],
                    dim=-1,
                ).reshape(-1, 3 * self.channels)
            ).reshape(batch, num_points, self.channels)
            fixed_forward, _, _ = self.shared_forward(
                fixed_features, fixed_fields
            )
            dynamic_forward, _, _ = self.shared_forward(
                dynamic_features, dynamic_fields
            )
            fixed_reverse, _, _ = self.shared_reverse(
                fixed_features, fixed_fields
            )
            dynamic_reverse, _, _ = self.shared_reverse(
                dynamic_features, dynamic_fields
            )
            route_output = self.route(
                fused.reshape(-1, self.channels)
            ).reshape(batch, num_points, 3)
            route = nn.softmax(route_output[..., :2], dim=-1)
            trust = jt.sigmoid(route_output[..., 2:3])
            forward = (
                route[..., 0:1] * fixed_forward
                + route[..., 1:2] * dynamic_forward
            )
            reverse = (
                route[..., 0:1] * fixed_reverse
                + route[..., 1:2] * dynamic_reverse
            )
            agreement = 0.5 + 0.5 * _cosine(
                forward, -reverse
            )
            midpoint_field = 0.5 * (forward - reverse)
            radius = fixed_fields["distance"][
                :, :, :16
            ].mean(dim=2)
            cap = (
                0.045 * radius
                + 0.030 * vector_norm(state.parent_update)
                + 1e-5
            ) / float(self.num_steps)
            update = bounded_vector(
                trust
                * (0.20 + 0.80 * agreement)
                * midpoint_field
                / float(self.num_steps),
                cap,
            )
            current = current + update
            step_predictions.append(current)
            routes.append(route)
            agreements.append(agreement)
            forward_fields.append(forward)
            reverse_fields.append(reverse)
        return current, {
            "step_predictions": step_predictions,
            "route": jt.stack(routes, dim=2),
            "reversible_agreement": jt.stack(
                agreements, dim=2
            ),
            "forward_field": jt.stack(
                forward_fields, dim=2
            ),
            "reverse_field": jt.stack(
                reverse_fields, dim=2
            ),
            "update": current - state.base_query,
        }


class FrozenHierarchicalCompetitor(nn.Module):
    def __init__(
        self,
        parent,
        parent_kind,
        mode,
        parent_strength=1.25,
        max_k=32,
        channels=104,
        num_steps=2,
    ):
        super().__init__()
        if parent_kind not in ("v40", "v41"):
            raise ValueError(
                "unsupported parent kind: %s" % parent_kind
            )
        if mode not in SUPPORTED_MODES:
            raise ValueError("unsupported mode: %s" % mode)
        if not isinstance(parent, FrozenBaseCompetitor):
            raise TypeError("parent must be a frozen base competitor")
        self.parent = parent
        self.parent_kind = str(parent_kind)
        self.base_kind = str(parent_kind)
        self.mode = str(mode)
        self.parent_strength = float(parent_strength)
        for parameter in self.parent.parameters():
            parameter.stop_grad()
        if mode == "v43_robust_spectral_shrinkage":
            self.head = RobustSpectralShrinkageHead(
                max_k=max_k, channels=channels
            )
        elif mode == "v44_redescending_conservative_flux":
            self.head = RedescendingConservativeFluxHead(
                max_k=max_k,
                channels=channels,
                num_steps=num_steps,
            )
        elif mode == "v45_dual_topology_consensus":
            self.head = DualTopologyConsensusHead(
                max_k=max_k, channels=channels
            )
        else:
            self.head = ReversibleDynamicBridgeHead(
                max_k=max_k,
                channels=channels,
                num_steps=num_steps,
            )

    def trainable_parameters(self):
        return list(self.head.parameters())

    def set_frozen_eval(self):
        self.parent.eval()
        self.parent.base.eval()
        self.parent.base.v29.eval()
        self.parent.base.v29.baseline.eval()

    def frozen_outputs(self, noisy):
        self.set_frozen_eval()
        with jt.no_grad():
            lower_outputs = self.parent.frozen_outputs(noisy)
            parent_outputs = self.parent(
                noisy,
                refinement_strength=self.parent_strength,
                frozen_outputs=lower_outputs,
            )
        return parent_outputs

    def execute(
        self,
        noisy,
        refinement_strength=1.0,
        frozen_outputs=None,
    ):
        if frozen_outputs is None:
            frozen_outputs = self.frozen_outputs(noisy)
        parent_outputs = frozen_outputs
        parent_query = parent_outputs[0]
        lower_base_query = parent_outputs[1]
        v29_query = parent_outputs[2]
        v12_query = parent_outputs[3]
        parent_auxiliary = parent_outputs[5]
        v29_outputs = parent_outputs[7]
        state = GeometryState(
            noisy,
            v12_query,
            v29_query,
            parent_query,
            v29_outputs[2],
            v29_outputs[3],
            self.head.max_k,
        )
        state.parent_update = parent_query - lower_base_query
        state.parent_auxiliary = parent_auxiliary
        refined, auxiliary = self.head(state)
        strength = float(refinement_strength)
        if abs(strength) < 1e-12:
            prediction = parent_query
        else:
            prediction = parent_query + strength * (
                refined - parent_query
            )
        stitch_confidence = parent_auxiliary[
            "stitch_confidence"
        ]
        auxiliary["stitch_confidence"] = stitch_confidence
        auxiliary["parent_stitch_confidence"] = (
            stitch_confidence
        )
        return (
            prediction,
            parent_query,
            v29_query,
            v12_query,
            refined,
            auxiliary,
            parent_outputs,
            v29_outputs,
        )


def patch_based_hierarchical(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    beta=12.0,
    patch_batch=8,
    refinement_strength=1.0,
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
    predictions = []
    confidences = []
    with jt.no_grad():
        for start in range(0, num_patches, int(patch_batch)):
            batch = jt.array(
                centered[
                    start : start + int(patch_batch)
                ].astype(np.float32)
            )
            output = model(
                batch,
                refinement_strength=float(
                    refinement_strength
                ),
            )
            predictions.append(output[0].numpy())
            confidences.append(
                output[5]["stitch_confidence"].numpy()
            )
    predictions = (
        np.concatenate(predictions, axis=0)
        + seeds[:, None, :]
    )
    confidences = np.concatenate(confidences, axis=0)
    return confidence_stitch(
        predictions,
        confidences,
        point_indices,
        normalized_distances,
        original,
        beta,
    )
