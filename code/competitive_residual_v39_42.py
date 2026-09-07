import math

import jittor as jt
import numpy as np
from jittor import nn

from cross_patch_consensus_v33 import V29CrossPatchConsensus
from iterativepfn_v12 import batched_index, knn_indices
from jittor_port_models import farthest_point_sampling_np, knn_points_np
from neural_laplacian_flow_v34 import V29StableNeuralLaplacianFlow
from pseudo_query_corrector_v29 import (
    InvariantTokenContext,
    LinearBNReLU,
    SparseGeoAttention,
)


SUPPORTED_MODES = (
    "v39_recurrent_consensus",
    "v40_conservative_dual_flow",
    "v41_spectral_residual",
    "v42_scalar_vector_equivariant",
)


def vector_norm(value, keepdims=True):
    return jt.sqrt((value ** 2).sum(-1, keepdims=keepdims) + 1e-8)


def bounded_vector(value, cap):
    scale = jt.minimum(
        jt.ones_like(cap),
        cap / (vector_norm(value) + 1e-8),
    )
    return value * scale


def project_normal(value, normal):
    return (value * normal).sum(-1, keepdims=True) * normal


class GeometryState:
    FEATURE_DIM = 42
    GEOMETRY_DIM = 9

    def __init__(
        self,
        noisy,
        v12_query,
        v29_query,
        base_query,
        correction,
        stage_displacements,
        max_k,
    ):
        self.noisy = noisy
        self.v12_query = v12_query
        self.v29_query = v29_query
        self.base_query = base_query
        self.correction = correction
        self.stage_displacements = stage_displacements
        self.max_k = int(max_k)
        self.indices = knn_indices(base_query, self.max_k)
        self.base_relative = (
            batched_index(base_query, self.indices)
            - base_query.unsqueeze(2)
        )
        self.noisy_relative = (
            batched_index(noisy, self.indices)
            - noisy.unsqueeze(2)
        )
        self.v29_relative = (
            batched_index(v29_query, self.indices)
            - v29_query.unsqueeze(2)
        )
        base_delta = base_query - v29_query
        self.base_delta = base_delta
        self.base_delta_relative = (
            batched_index(base_delta, self.indices)
            - base_delta.unsqueeze(2)
        )
        self.correction_relative = (
            batched_index(correction, self.indices)
            - correction.unsqueeze(2)
        )
        self.base_distance = vector_norm(self.base_relative)
        self.noisy_distance = vector_norm(self.noisy_relative)
        self.v29_distance = vector_norm(self.v29_relative)
        self.base_scale = self.base_distance.mean(
            dim=2, keepdims=True
        )
        self.noisy_scale = self.noisy_distance.mean(
            dim=2, keepdims=True
        )
        self.v29_scale = self.v29_distance.mean(
            dim=2, keepdims=True
        )
        self.strain_noisy = (
            self.base_distance - self.noisy_distance
        ) / (self.noisy_scale + 1e-8)
        self.strain_v29 = (
            self.base_distance - self.v29_distance
        ) / (self.v29_scale + 1e-8)
        cosine_noisy = (
            self.base_relative * self.noisy_relative
        ).sum(-1, keepdims=True) / (
            self.base_distance * self.noisy_distance + 1e-8
        )
        cosine_v29 = (
            self.base_relative * self.v29_relative
        ).sum(-1, keepdims=True) / (
            self.base_distance * self.v29_distance + 1e-8
        )
        self.geometry = jt.concat(
            [
                self.base_distance / (self.base_scale + 1e-8),
                self.noisy_distance / (self.noisy_scale + 1e-8),
                self.v29_distance / (self.v29_scale + 1e-8),
                self.strain_noisy,
                self.strain_v29,
                cosine_noisy,
                cosine_v29,
                vector_norm(self.base_delta_relative)
                / (self.base_scale + 1e-8),
                vector_norm(self.correction_relative)
                / (self.base_scale + 1e-8),
            ],
            dim=-1,
        )
        self.initial = self._initial_features()

    @staticmethod
    def _stats(distance, count):
        count = min(int(count), int(distance.shape[2]))
        selected = distance[:, :, :count]
        mean = selected.mean(dim=2)
        variance = (
            (selected - mean.unsqueeze(2)) ** 2
        ).mean(dim=2)
        return jt.concat(
            [
                mean,
                jt.sqrt(variance + 1e-8),
                selected.max(dim=2),
            ],
            dim=-1,
        )

    @staticmethod
    def _cosine(first, second):
        return (first * second).sum(-1, keepdims=True) / (
            vector_norm(first) * vector_norm(second) + 1e-8
        )

    @staticmethod
    def _mean_std(value):
        mean = value.mean(dim=2)
        std = jt.sqrt(
            ((value - mean.unsqueeze(2)) ** 2).mean(dim=2)
            + 1e-8
        )
        return jt.concat([mean, std], dim=-1)

    def _initial_features(self):
        stage_magnitude = vector_norm(
            self.stage_displacements, keepdims=False
        )
        if int(stage_magnitude.shape[2]) != 4:
            raise ValueError("expected four frozen V12 stages")
        base_velocity = self.v12_query - self.noisy
        correction = self.v29_query - self.v12_query
        base_delta = self.base_query - self.v29_query
        total = self.base_query - self.noisy
        features = jt.concat(
            [
                stage_magnitude,
                vector_norm(base_velocity),
                vector_norm(correction),
                vector_norm(base_delta),
                vector_norm(total),
                self._cosine(correction, base_velocity),
                self._cosine(base_delta, correction),
                self._cosine(base_delta, total),
                self._stats(self.base_distance, 8),
                self._stats(self.base_distance, 16),
                self._stats(self.base_distance, self.max_k),
                self._stats(self.noisy_distance, 8),
                self._stats(self.noisy_distance, 16),
                self._stats(self.noisy_distance, self.max_k),
                self._stats(self.v29_distance, 8),
                self._stats(self.v29_distance, 16),
                self._stats(self.v29_distance, self.max_k),
                self._mean_std(self.strain_noisy),
                self._mean_std(self.strain_v29),
            ],
            dim=-1,
        )
        if int(features.shape[-1]) != self.FEATURE_DIM:
            raise ValueError(
                "geometry feature mismatch: %d"
                % int(features.shape[-1])
            )
        return features

    def prefix(self, count):
        count = min(int(count), self.max_k)
        return (
            self.indices[:, :, :count],
            self.geometry[:, :, :count],
        )

    def current_fields(self, current, count):
        count = min(int(count), self.max_k)
        indices = self.indices[:, :, :count]
        current_relative = (
            batched_index(current, indices) - current.unsqueeze(2)
        )
        return {
            "current_relative": current_relative,
            "noisy_relative": self.noisy_relative[:, :, :count],
            "v29_relative": self.v29_relative[:, :, :count],
            "base_delta_relative": self.base_delta_relative[
                :, :, :count
            ],
            "correction_relative": self.correction_relative[
                :, :, :count
            ],
            "base_delta": self.base_delta,
            "distance": vector_norm(current_relative),
            "base_distance": self.base_distance[:, :, :count],
            "geometry": self.geometry[:, :, :count],
            "indices": indices,
        }


class DenseEquivariantField(nn.Module):
    def __init__(self, channels, geometry_dim=9):
        super().__init__()
        self.channels = int(channels)
        self.edge_hidden = LinearBNReLU(
            2 * self.channels + int(geometry_dim),
            self.channels,
        )
        self.edge_output = nn.Linear(self.channels, 7)
        self.edge_output.weight.assign(
            jt.zeros_like(self.edge_output.weight)
        )
        self.edge_output.bias.assign(
            jt.zeros_like(self.edge_output.bias)
        )

    def execute(self, features, fields):
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
            self.edge_hidden(edge.reshape(-1, edge.shape[-1]))
        ).reshape(batch, num_points, k, 7)
        coefficients = jt.tanh(output[..., :6])
        importance = nn.softmax(output[..., 6], dim=2)
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
        edge_field = (
            coefficients.unsqueeze(-1) * bases
        ).sum(dim=3)
        field = (
            importance.unsqueeze(-1) * edge_field
        ).sum(dim=2)
        return field, coefficients, importance


class LocalCrossAttention(nn.Module):
    def __init__(self, channels, key_dim=32):
        super().__init__()
        self.channels = int(channels)
        self.key_dim = int(key_dim)
        self.query = nn.Linear(self.channels, self.key_dim)
        self.key = nn.Linear(self.channels, self.key_dim)
        self.value = nn.Linear(self.channels, self.channels)
        self.geometry_key = nn.Linear(9, self.key_dim)
        self.geometry_value = nn.Linear(9, self.channels)
        self.fuse = LinearBNReLU(
            2 * self.channels, self.channels
        )

    def execute(self, target, source, indices, geometry):
        batch, num_points, channels = target.shape
        neighbors = batched_index(source, indices)
        logits = (
            self.query(target).unsqueeze(2)
            * (
                self.key(neighbors)
                + self.geometry_key(geometry)
            )
        ).sum(-1) / math.sqrt(float(self.key_dim))
        attention = nn.softmax(logits, dim=2)
        values = (
            self.value(neighbors)
            + self.geometry_value(geometry)
        )
        context = (
            attention.unsqueeze(-1) * values
        ).sum(dim=2)
        fused = self.fuse(
            jt.concat([target, context], dim=-1).reshape(
                -1, 2 * channels
            )
        ).reshape(batch, num_points, channels)
        return fused, attention


class RecurrentConsensusHead(nn.Module):
    def __init__(self, max_k=32, channels=96, num_steps=3):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        self.input_proj = LinearBNReLU(42, 64)
        self.shared_scale_encoder = SparseGeoAttention(
            64, self.channels, geometry_dim=9
        )
        self.scale_fuse = LinearBNReLU(
            3 * self.channels + 42, self.channels
        )
        self.context = InvariantTokenContext(
            self.channels, num_tokens=6
        )
        self.step_condition = nn.Linear(1, self.channels)
        self.step_fuse = LinearBNReLU(
            2 * self.channels, self.channels
        )
        self.field = DenseEquivariantField(self.channels)
        self.route = nn.Linear(self.channels, 4)
        self.route.weight.assign(jt.zeros_like(self.route.weight))
        self.route.bias.assign(jt.zeros_like(self.route.bias))

    def execute(self, state):
        batch, num_points = state.base_query.shape[:2]
        initial = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, num_points, 64)
        encoded = []
        for count in (8, 16, self.max_k):
            indices, geometry = state.prefix(count)
            encoded.append(
                self.shared_scale_encoder(
                    initial, indices, geometry
                )
            )
        fused = self.scale_fuse(
            jt.concat(
                encoded + [state.initial], dim=-1
            ).reshape(
                -1, 3 * self.channels + 42
            )
        ).reshape(batch, num_points, self.channels)
        context = self.context(fused)
        current = state.base_query
        step_predictions = []
        routes = []
        step_gates = []
        fields_per_step = []
        for step in range(self.num_steps):
            step_value = jt.ones(
                (batch, num_points, 1)
            ) * float(step + 1) / float(self.num_steps)
            conditioned = fused + self.step_condition(
                step_value.reshape(-1, 1)
            ).reshape(batch, num_points, self.channels)
            features = self.step_fuse(
                jt.concat(
                    [conditioned, context], dim=-1
                ).reshape(-1, 2 * self.channels)
            ).reshape(batch, num_points, self.channels)
            scale_fields = []
            for count in (8, 16, self.max_k):
                field, _, _ = self.field(
                    features,
                    state.current_fields(current, count),
                )
                scale_fields.append(field)
            route_output = self.route(
                features.reshape(-1, self.channels)
            ).reshape(batch, num_points, 4)
            route = nn.softmax(route_output[..., :3], dim=-1)
            step_gate = jt.sigmoid(route_output[..., 3:4])
            mixed = (
                jt.stack(scale_fields, dim=2)
                * route.unsqueeze(-1)
            ).sum(dim=2)
            radius = state.base_distance[
                :, :, :16
            ].mean(dim=2)
            cap = (
                0.070 * radius
                + 0.060 * vector_norm(state.base_delta)
                + 1e-5
            )
            update = bounded_vector(
                step_gate * mixed / float(self.num_steps),
                cap / float(self.num_steps),
            )
            current = current + update
            step_predictions.append(current)
            routes.append(route)
            step_gates.append(step_gate)
            fields_per_step.append(
                jt.stack(scale_fields, dim=2)
            )
        return current, {
            "step_predictions": step_predictions,
            "route": jt.stack(routes, dim=2),
            "step_gate": jt.stack(step_gates, dim=2),
            "branch_fields": jt.stack(fields_per_step, dim=2),
            "update": current - state.base_query,
        }


class SpectralResidualHead(nn.Module):
    def __init__(self, max_k=32, channels=112):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.input_proj = LinearBNReLU(42, 72)
        self.encoder1 = SparseGeoAttention(
            72, self.channels, geometry_dim=9
        )
        self.encoder2 = SparseGeoAttention(
            self.channels, self.channels, geometry_dim=9
        )
        self.band_fuse = nn.ModuleList(
            [
                LinearBNReLU(
                    self.channels + 42, self.channels
                )
                for _ in range(3)
            ]
        )
        self.band_fields = nn.ModuleList(
            [
                DenseEquivariantField(self.channels)
                for _ in range(3)
            ]
        )
        self.route_fuse = LinearBNReLU(
            3 * self.channels + 42, self.channels
        )
        self.route = nn.Linear(self.channels, 3)
        self.route.weight.assign(jt.zeros_like(self.route.weight))
        self.route.bias.assign(jt.zeros_like(self.route.bias))

    @staticmethod
    def _smooth(features, state, count):
        count = min(int(count), state.max_k)
        indices = state.indices[:, :, :count]
        distance = state.base_distance[:, :, :count]
        neighbors = batched_index(features, indices)
        radius = distance.mean(dim=2, keepdims=True)
        weights = jt.exp(
            -distance / (radius + 1e-6)
        )
        weights = weights / (
            weights.sum(dim=2, keepdims=True) + 1e-8
        )
        return (weights * neighbors).sum(dim=2)

    def execute(self, state):
        batch, num_points = state.base_query.shape[:2]
        indices, geometry = state.prefix(self.max_k)
        projected = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, num_points, 72)
        encoded = self.encoder1(
            projected, indices, geometry
        )
        encoded = self.encoder2(
            encoded, indices, geometry
        )
        smooth8 = self._smooth(encoded, state, 8)
        smooth32 = self._smooth(
            encoded, state, self.max_k
        )
        raw_bands = (
            smooth32,
            smooth8 - smooth32,
            encoded - smooth8,
        )
        band_features = []
        band_fields = []
        fields = state.current_fields(
            state.base_query, self.max_k
        )
        for index, raw in enumerate(raw_bands):
            band = self.band_fuse[index](
                jt.concat(
                    [raw, state.initial], dim=-1
                ).reshape(
                    -1, self.channels + 42
                )
            ).reshape(batch, num_points, self.channels)
            field, _, _ = self.band_fields[index](
                band, fields
            )
            band_features.append(band)
            band_fields.append(field)
        route_features = self.route_fuse(
            jt.concat(
                band_features + [state.initial], dim=-1
            ).reshape(
                -1, 3 * self.channels + 42
            )
        ).reshape(batch, num_points, self.channels)
        route = nn.softmax(
            self.route(
                route_features.reshape(
                    -1, self.channels
                )
            ).reshape(batch, num_points, 3),
            dim=-1,
        )
        update = (
            jt.stack(band_fields, dim=2)
            * route.unsqueeze(-1)
        ).sum(dim=2)
        radius = state.base_distance[
            :, :, :16
        ].mean(dim=2)
        cap = (
            0.075 * radius
            + 0.060 * vector_norm(state.base_delta)
            + 1e-5
        )
        update = bounded_vector(update, cap)
        refined = state.base_query + update
        return refined, {
            "band_route": route,
            "branch_fields": jt.stack(
                band_fields, dim=2
            ),
            "low_feature": raw_bands[0],
            "mid_feature": raw_bands[1],
            "high_feature": raw_bands[2],
            "update": update,
        }


class ConservativeDualFlowHead(nn.Module):
    def __init__(self, max_k=32, channels=104, num_steps=2):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        self.input_proj = LinearBNReLU(42, 72)
        self.encoder1 = SparseGeoAttention(
            72, self.channels, geometry_dim=9
        )
        self.encoder2 = SparseGeoAttention(
            self.channels, self.channels, geometry_dim=9
        )
        self.context = InvariantTokenContext(
            self.channels, num_tokens=6
        )
        self.fuse = LinearBNReLU(
            2 * self.channels + 42, self.channels
        )
        self.surface_field = DenseEquivariantField(
            self.channels
        )
        self.coverage_hidden = LinearBNReLU(
            2 * self.channels + 9, self.channels
        )
        self.coverage_output = nn.Linear(self.channels, 1)
        self.route = nn.Linear(self.channels, 3)
        self.coverage_output.weight.assign(
            jt.zeros_like(self.coverage_output.weight)
        )
        self.coverage_output.bias.assign(
            jt.zeros_like(self.coverage_output.bias)
        )
        self.route.weight.assign(jt.zeros_like(self.route.weight))
        self.route.bias.assign(jt.zeros_like(self.route.bias))

    def _coverage_field(self, features, fields):
        indices = fields["indices"]
        geometry = fields["geometry"]
        relative = fields["current_relative"]
        distance = fields["distance"]
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
        pressure = jt.tanh(
            self.coverage_output(
                self.coverage_hidden(
                    edge.reshape(-1, edge.shape[-1])
                )
            ).reshape(batch, num_points, k, 1)
        )
        radius = distance.mean(dim=2, keepdims=True)
        radial_error = (
            radius - distance
        ) / (radius + 1e-6)
        direction = relative / (distance + 1e-6)
        field = (
            pressure * radial_error * direction
        ).mean(dim=2) * radius.reshape(
            batch, num_points, 1
        )
        field = field - field.mean(dim=1, keepdims=True)
        return field, pressure

    def execute(self, state):
        batch, num_points = state.base_query.shape[:2]
        indices, geometry = state.prefix(self.max_k)
        projected = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, num_points, 72)
        encoded = self.encoder1(
            projected, indices, geometry
        )
        encoded = self.encoder2(
            encoded, indices, geometry
        )
        context = self.context(encoded)
        features = self.fuse(
            jt.concat(
                [encoded, context, state.initial], dim=-1
            ).reshape(
                -1, 2 * self.channels + 42
            )
        ).reshape(batch, num_points, self.channels)
        current = state.base_query
        step_predictions = []
        routes = []
        surface_fields = []
        coverage_fields = []
        coverage_pressures = []
        for _ in range(self.num_steps):
            fields = state.current_fields(
                current, self.max_k
            )
            surface, _, _ = self.surface_field(
                features, fields
            )
            coverage, pressure = self._coverage_field(
                features, fields
            )
            route_output = self.route(
                features.reshape(-1, self.channels)
            ).reshape(batch, num_points, 3)
            route = nn.softmax(route_output[..., :2], dim=-1)
            step_gate = jt.sigmoid(route_output[..., 2:3])
            mixed = (
                route[..., 0:1] * surface
                + route[..., 1:2] * coverage
            )
            radius = fields["distance"][
                :, :, :16
            ].mean(dim=2)
            cap = (
                0.065 * radius
                + 0.050 * vector_norm(state.base_delta)
                + 1e-5
            ) / float(self.num_steps)
            update = bounded_vector(
                step_gate * mixed / float(self.num_steps),
                cap,
            )
            current = current + update
            step_predictions.append(current)
            routes.append(route)
            surface_fields.append(surface)
            coverage_fields.append(coverage)
            coverage_pressures.append(pressure)
        return current, {
            "step_predictions": step_predictions,
            "route": jt.stack(routes, dim=2),
            "surface_field": jt.stack(
                surface_fields, dim=2
            ),
            "coverage_field": jt.stack(
                coverage_fields, dim=2
            ),
            "coverage_pressure": jt.stack(
                coverage_pressures, dim=2
            ),
            "update": current - state.base_query,
        }


class ScalarVectorBlock(nn.Module):
    def __init__(self, channels, vector_channels=8):
        super().__init__()
        self.channels = int(channels)
        self.vector_channels = int(vector_channels)
        self.scalar_encoder = SparseGeoAttention(
            self.channels,
            self.channels,
            geometry_dim=9,
        )
        self.edge_hidden = LinearBNReLU(
            2 * self.channels + 9,
            self.channels,
        )
        self.edge_vector_gate = nn.Linear(
            self.channels, self.vector_channels + 1
        )
        self.vector_mix = nn.Linear(
            self.vector_channels,
            self.vector_channels,
            bias=False,
        )
        self.vector_gate = nn.Linear(
            self.channels, self.vector_channels
        )
        self.scalar_fuse = LinearBNReLU(
            self.channels + self.vector_channels,
            self.channels,
        )

    def execute(
        self,
        scalar,
        vectors,
        indices,
        geometry,
        relative,
    ):
        batch, num_points, channels = scalar.shape
        k = int(indices.shape[2])
        scalar = self.scalar_encoder(
            scalar, indices, geometry
        )
        scalar_neighbors = batched_index(scalar, indices)
        scalar_centers = scalar.unsqueeze(2).broadcast(
            (batch, num_points, k, channels)
        )
        edge = jt.concat(
            [
                scalar_centers,
                scalar_neighbors - scalar_centers,
                geometry,
            ],
            dim=-1,
        )
        edge_output = self.edge_vector_gate(
            self.edge_hidden(
                edge.reshape(-1, edge.shape[-1])
            )
        ).reshape(
            batch,
            num_points,
            k,
            self.vector_channels + 1,
        )
        vector_gate = jt.tanh(
            edge_output[..., : self.vector_channels]
        )
        importance = nn.softmax(
            edge_output[..., self.vector_channels],
            dim=2,
        )
        flat_vectors = vectors.reshape(
            batch, num_points, self.vector_channels * 3
        )
        neighbor_vectors = batched_index(
            flat_vectors, indices
        ).reshape(
            batch,
            num_points,
            k,
            self.vector_channels,
            3,
        )
        messages = (
            neighbor_vectors
            + vector_gate.unsqueeze(-1)
            * relative.unsqueeze(3)
        )
        aggregated = (
            importance.unsqueeze(-1).unsqueeze(-1)
            * messages
        ).sum(dim=2)
        mixed = self.vector_mix(
            aggregated.permute(0, 1, 3, 2).reshape(
                -1, self.vector_channels
            )
        ).reshape(
            batch,
            num_points,
            3,
            self.vector_channels,
        ).permute(0, 1, 3, 2)
        point_gate = jt.sigmoid(
            self.vector_gate(
                scalar.reshape(-1, self.channels)
            )
        ).reshape(
            batch,
            num_points,
            self.vector_channels,
            1,
        )
        vectors = vectors + point_gate * mixed
        invariants = vector_norm(
            vectors, keepdims=False
        )
        scalar = self.scalar_fuse(
            jt.concat([scalar, invariants], dim=-1).reshape(
                -1, self.channels + self.vector_channels
            )
        ).reshape(batch, num_points, self.channels)
        return scalar, vectors


class ScalarVectorEquivariantHead(nn.Module):
    def __init__(
        self,
        max_k=32,
        channels=104,
        vector_channels=8,
    ):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.vector_channels = int(vector_channels)
        self.input_proj = LinearBNReLU(42, self.channels)
        self.block1 = ScalarVectorBlock(
            self.channels, self.vector_channels
        )
        self.block2 = ScalarVectorBlock(
            self.channels, self.vector_channels
        )
        self.output_coefficients = nn.Linear(
            self.channels, self.vector_channels
        )
        self.output_coefficients.weight.assign(
            jt.zeros_like(self.output_coefficients.weight)
        )
        self.output_coefficients.bias.assign(
            jt.zeros_like(self.output_coefficients.bias)
        )

    @staticmethod
    def _weighted_mean(relative, distance, count):
        count = min(int(count), int(relative.shape[2]))
        edges = relative[:, :, :count]
        distances = distance[:, :, :count]
        radius = distances.mean(dim=2, keepdims=True)
        weights = jt.exp(
            -distances / (radius + 1e-6)
        )
        weights = weights / (
            weights.sum(dim=2, keepdims=True) + 1e-8
        )
        return (weights * edges).sum(dim=2)

    def _initial_vectors(self, state):
        return jt.stack(
            [
                self._weighted_mean(
                    state.base_relative,
                    state.base_distance,
                    8,
                ),
                self._weighted_mean(
                    state.base_relative,
                    state.base_distance,
                    16,
                ),
                self._weighted_mean(
                    state.base_relative,
                    state.base_distance,
                    self.max_k,
                ),
                self._weighted_mean(
                    state.noisy_relative,
                    state.noisy_distance,
                    16,
                ),
                self._weighted_mean(
                    state.v29_relative,
                    state.v29_distance,
                    16,
                ),
                state.base_query - state.v29_query,
                state.v29_query - state.v12_query,
                state.base_query - state.noisy,
            ],
            dim=2,
        )

    def execute(self, state):
        batch, num_points = state.base_query.shape[:2]
        indices, geometry = state.prefix(self.max_k)
        scalar = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(
            batch, num_points, self.channels
        )
        vectors = self._initial_vectors(state)
        scalar, vectors = self.block1(
            scalar,
            vectors,
            indices,
            geometry,
            state.base_relative[:, :, : self.max_k],
        )
        scalar, vectors = self.block2(
            scalar,
            vectors,
            indices,
            geometry,
            state.base_relative[:, :, : self.max_k],
        )
        coefficients = jt.tanh(
            self.output_coefficients(
                scalar.reshape(-1, self.channels)
            )
        ).reshape(
            batch,
            num_points,
            self.vector_channels,
        )
        update = (
            coefficients.unsqueeze(-1) * vectors
        ).sum(dim=2) / float(self.vector_channels)
        radius = state.base_distance[
            :, :, :16
        ].mean(dim=2)
        cap = (
            0.075 * radius
            + 0.055 * vector_norm(state.base_delta)
            + 1e-5
        )
        update = bounded_vector(update, cap)
        return state.base_query + update, {
            "vector_channels": vectors,
            "vector_norms": vector_norm(
                vectors, keepdims=False
            ),
            "scalar_features": scalar,
            "coefficients": coefficients,
            "update": update,
        }


class FrozenBaseCompetitor(nn.Module):
    def __init__(
        self,
        base_model,
        base_kind,
        mode,
        max_k=32,
        channels=96,
        num_steps=3,
    ):
        super().__init__()
        if mode not in SUPPORTED_MODES:
            raise ValueError("unsupported mode: %s" % mode)
        if base_kind not in ("v33", "v34"):
            raise ValueError("unsupported base kind: %s" % base_kind)
        if base_kind == "v33" and not isinstance(
            base_model, V29CrossPatchConsensus
        ):
            raise TypeError("V33 competitor requires V33 base")
        if base_kind == "v34" and not isinstance(
            base_model, V29StableNeuralLaplacianFlow
        ):
            raise TypeError("V34 competitor requires V34 base")
        self.base = base_model
        self.base_kind = str(base_kind)
        self.mode = str(mode)
        for parameter in self.base.parameters():
            parameter.stop_grad()
        if mode == "v39_recurrent_consensus":
            self.head = RecurrentConsensusHead(
                max_k=max_k,
                channels=channels,
                num_steps=num_steps,
            )
        elif mode == "v40_conservative_dual_flow":
            self.head = ConservativeDualFlowHead(
                max_k=max_k,
                channels=channels,
                num_steps=num_steps,
            )
        elif mode == "v41_spectral_residual":
            self.head = SpectralResidualHead(
                max_k=max_k,
                channels=channels,
            )
        else:
            self.head = ScalarVectorEquivariantHead(
                max_k=max_k,
                channels=channels,
            )

    def trainable_parameters(self):
        return list(self.head.parameters())

    def frozen_outputs(self, noisy):
        self.base.eval()
        self.base.v29.eval()
        self.base.v29.baseline.eval()
        with jt.no_grad():
            v29_outputs = self.base.v29_outputs(noisy)
            if self.base_kind == "v33":
                base_outputs = self.base(
                    noisy,
                    consensus_strength=1.0,
                    v29_outputs=v29_outputs,
                )
            else:
                base_outputs = self.base(
                    noisy,
                    flow_strength=1.0,
                    v29_outputs=v29_outputs,
                )
        return v29_outputs, base_outputs

    def execute(
        self,
        noisy,
        refinement_strength=1.0,
        frozen_outputs=None,
    ):
        if frozen_outputs is None:
            frozen_outputs = self.frozen_outputs(noisy)
        v29_outputs, base_outputs = frozen_outputs
        base_query = base_outputs[0]
        v29_query = v29_outputs[0]
        v12_query = v29_outputs[1]
        correction = v29_outputs[2]
        stage_displacements = v29_outputs[3]
        state = GeometryState(
            noisy,
            v12_query,
            v29_query,
            base_query,
            correction,
            stage_displacements,
            self.head.max_k,
        )
        refined, auxiliary = self.head(state)
        strength = float(refinement_strength)
        prediction = base_query + strength * (
            refined - base_query
        )
        if self.base_kind == "v33":
            base_confidence = base_outputs[4][
                "patch_confidence"
            ]
        else:
            base_confidence = jt.ones(
                (noisy.shape[0], noisy.shape[1], 1)
            )
        learned_confidence = auxiliary.get(
            "query_confidence",
            jt.ones_like(base_confidence),
        )
        auxiliary["stitch_confidence"] = (
            base_confidence
            * (
                1.0
                + strength * (learned_confidence - 1.0)
            )
        )
        auxiliary["base_stitch_confidence"] = base_confidence
        return (
            prediction,
            base_query,
            v29_query,
            v12_query,
            refined,
            auxiliary,
            base_outputs,
            v29_outputs,
        )


def confidence_stitch(
    predictions,
    confidences,
    point_indices,
    normalized_distances,
    original,
    beta,
):
    flat_indices = point_indices.reshape(-1)
    spatial = np.exp(
        -float(beta) * normalized_distances
    ).astype(np.float32)
    calibrated = np.clip(
        confidences[..., 0], 0.70, 1.30
    )
    weights = (spatial * calibrated).reshape(-1)
    flat_predictions = predictions.reshape(-1, 3)
    weight_sum = np.zeros(
        original.shape[0], dtype=np.float32
    )
    output_sum = np.zeros_like(
        original, dtype=np.float32
    )
    np.add.at(weight_sum, flat_indices, weights)
    for axis in range(3):
        np.add.at(
            output_sum[:, axis],
            flat_indices,
            weights * flat_predictions[:, axis],
        )
    valid = weight_sum > 1e-8
    output = original.copy()
    output[valid] = (
        output_sum[valid] / weight_sum[valid, None]
    )
    return output


def patch_based_competitor(
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
        for start in range(
            0, num_patches, int(patch_batch)
        ):
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
