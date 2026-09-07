import math

import jittor as jt
import numpy as np
from jittor import nn

from competitive_residual_v39_42 import (
    DenseEquivariantField,
    GeometryState,
    ScalarVectorBlock,
    bounded_vector,
    confidence_stitch,
    vector_norm,
)
from hierarchical_residual_v43_46 import _graph_fields
from iterativepfn_v12 import batched_index, knn_indices
from jittor_port_models import farthest_point_sampling_np, knn_points_np
from pseudo_query_corrector_v29 import LinearBNReLU, SparseGeoAttention


MODES = {
    "v62": "v62_surface_first_remesher",
    "v63": "v63_dual_topology_vector_backbone",
    "v64": "v64_trajectory_conditioned_projection",
    "v65": "v65_surface_jet_tensor_backbone",
}


def _zero_linear(layer):
    layer.weight.assign(jt.zeros_like(layer.weight))
    if layer.bias is not None:
        layer.bias.assign(jt.zeros_like(layer.bias))


def _weighted_mean(relative, distance, count):
    count = min(int(count), int(relative.shape[2]))
    edges = relative[:, :, :count]
    distances = distance[:, :, :count]
    radius = distances.mean(dim=2, keepdims=True)
    weights = jt.exp(-distances / (radius + 1e-6))
    weights = weights / (
        weights.sum(dim=2, keepdims=True) + 1e-8
    )
    return (weights * edges).sum(dim=2)


def _matrix_multiply(first, second):
    rows = []
    for row in range(3):
        columns = []
        for column in range(3):
            value = first[..., row, 0] * second[..., 0, column]
            value = value + first[..., row, 1] * second[..., 1, column]
            value = value + first[..., row, 2] * second[..., 2, column]
            columns.append(value)
        rows.append(jt.stack(columns, dim=-1))
    return jt.stack(rows, dim=-2)


def _matrix_vector(matrix, vector):
    values = []
    for row in range(3):
        value = matrix[..., row, 0] * vector[..., 0]
        value = value + matrix[..., row, 1] * vector[..., 1]
        value = value + matrix[..., row, 2] * vector[..., 2]
        values.append(value)
    return jt.stack(values, dim=-1)


def _identity_like(reference):
    identity = jt.array(
        np.eye(3, dtype=np.float32)
    ).reshape((1,) * (len(reference.shape) - 2) + (3, 3))
    return identity.broadcast(reference.shape)


def _surface_projector(relative, distance, count):
    count = min(int(count), int(relative.shape[2]))
    selected = relative[:, :, :count]
    selected_distance = distance[:, :, :count]
    radius = selected_distance.mean(dim=2, keepdims=True)
    weights = jt.exp(
        -selected_distance / (radius + 1e-6)
    )
    weights = weights / (
        weights.sum(dim=2, keepdims=True) + 1e-8
    )
    outer = (
        selected.unsqueeze(-1) * selected.unsqueeze(-2)
    )
    covariance = (
        weights.unsqueeze(-1) * outer
    ).sum(dim=2)
    trace = (
        covariance[..., 0, 0]
        + covariance[..., 1, 1]
        + covariance[..., 2, 2]
    ).unsqueeze(-1).unsqueeze(-1)
    covariance = covariance / (trace + 1e-8)
    identity = _identity_like(covariance)
    complement = identity - covariance
    square = _matrix_multiply(complement, complement)
    fourth = _matrix_multiply(square, square)
    fourth_trace = (
        fourth[..., 0, 0]
        + fourth[..., 1, 1]
        + fourth[..., 2, 2]
    ).unsqueeze(-1).unsqueeze(-1)
    normal = fourth / (fourth_trace + 1e-8)
    tangent = identity - normal
    return normal, tangent, covariance


def _project_edges(projector, relative):
    batch, points, neighbors = relative.shape[:3]
    expanded = projector.unsqueeze(2).broadcast(
        (batch, points, neighbors, 3, 3)
    )
    return _matrix_vector(expanded, relative)


class SurfaceFirstRemesherHead(nn.Module):
    """Two unrolled surface-measure steps with learned energy parameters."""

    def __init__(self, max_k=24, channels=112, num_steps=2):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        self.input_proj = LinearBNReLU(42, self.channels)
        self.encoder1 = SparseGeoAttention(
            self.channels, self.channels, geometry_dim=9
        )
        self.encoder2 = SparseGeoAttention(
            self.channels, self.channels, geometry_dim=9
        )
        self.energy_hidden = LinearBNReLU(
            self.channels + 7, self.channels
        )
        self.energy_output = nn.Linear(self.channels, 7)
        _zero_linear(self.energy_output)

    def _step(self, state, current, scalar, step):
        indices = (
            state.indices[:, :, : self.max_k]
            if step == 0
            else knn_indices(current, self.max_k)
        )
        fields = _graph_fields(state, current, indices)
        relative = fields["current_relative"]
        distance = fields["distance"]
        normal, tangent, covariance = _surface_projector(
            relative, distance, min(16, self.max_k)
        )
        tangent_edges = _project_edges(tangent, relative)
        normal_edges = _project_edges(normal, relative)
        radius = distance[:, :, :16].mean(dim=2)
        covariance_diag = jt.stack(
            [
                covariance[..., 0, 0],
                covariance[..., 1, 1],
                covariance[..., 2, 2],
            ],
            dim=-1,
        )
        geometry = jt.concat(
            [
                radius,
                vector_norm(
                    _weighted_mean(relative, distance, 8)
                ),
                vector_norm(state.base_delta),
                covariance_diag,
                jt.ones_like(radius) * float(step),
            ],
            dim=-1,
        )
        energy = self.energy_output(
            self.energy_hidden(
                jt.concat([scalar, geometry], dim=-1).reshape(
                    -1, self.channels + 7
                )
            )
        ).reshape(
            current.shape[0], current.shape[1], 7
        )
        spacing = radius * (
            0.72 + 0.36 * jt.sigmoid(energy[..., 0:1])
        )
        normalized_distance = distance / (spacing.unsqueeze(2) + 1e-8)
        robust = jt.exp(
            -0.75 * normalized_distance ** 2
        )
        measure_force = jt.tanh(
            1.0 - normalized_distance
        )
        measure_weights = robust / (
            robust.sum(dim=2, keepdims=True) + 1e-8
        )
        tangent_force = (
            measure_weights
            * measure_force
            * tangent_edges
            / (distance + 1e-8)
        ).sum(dim=2)
        surface_force = (
            measure_weights * normal_edges
        ).sum(dim=2)
        laplacian = (
            measure_weights * relative
        ).sum(dim=2)
        gains = jt.tanh(energy[..., 1:4])
        gate = jt.sigmoid(energy[..., 4:5])
        damping = 0.35 + 0.65 * jt.sigmoid(
            energy[..., 5:6]
        )
        stop = jt.sigmoid(energy[..., 6:7])
        update = gate * damping * (
            gains[..., 0:1] * tangent_force
            + gains[..., 1:2] * surface_force
            + 0.35 * gains[..., 2:3] * laplacian
        )
        cap = (
            0.060 * radius
            + 0.025 * vector_norm(state.base_delta)
            + 1e-5
        )
        update = bounded_vector(update, cap) * (1.0 - 0.5 * stop)
        return current + update, {
            "indices": indices,
            "normal_projector": normal,
            "tangent_projector": tangent,
            "spacing": spacing,
            "robust_weight": measure_weights,
            "update": update,
        }

    def execute(self, state):
        batch, points = state.base_query.shape[:2]
        scalar = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, points, self.channels)
        indices, geometry = state.prefix(self.max_k)
        scalar = self.encoder1(scalar, indices, geometry)
        current = state.base_query
        steps = []
        diagnostics = []
        for step in range(self.num_steps):
            current, diagnostic = self._step(
                state, current, scalar, step
            )
            steps.append(current)
            diagnostics.append(diagnostic)
            fields = _graph_fields(
                state, current, diagnostic["indices"]
            )
            scalar = self.encoder2(
                scalar,
                diagnostic["indices"],
                fields["geometry"],
            )
        return current, {
            "step_predictions": steps,
            "step_diagnostics": diagnostics,
            "update": current - state.base_query,
        }


class DualTopologyVectorBackboneHead(nn.Module):
    """Deep scalar-vector state propagation over noisy and V45 graphs."""

    def __init__(
        self,
        max_k=32,
        channels=112,
        vector_channels=12,
    ):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.vector_channels = int(vector_channels)
        self.input_proj = LinearBNReLU(42, self.channels)
        self.blocks = nn.ModuleList(
            [
                ScalarVectorBlock(
                    self.channels, self.vector_channels
                )
                for _ in range(4)
            ]
        )
        self.decode1 = nn.Linear(
            self.channels, self.vector_channels
        )
        self.decode2 = nn.Linear(
            self.channels, self.vector_channels
        )
        self.trust = nn.Linear(self.channels, 1)
        _zero_linear(self.decode1)
        _zero_linear(self.decode2)
        _zero_linear(self.trust)

    def _initial_vectors(self, state):
        means = jt.stack(
            [
                _weighted_mean(
                    state.base_relative, state.base_distance, 8
                ),
                _weighted_mean(
                    state.base_relative, state.base_distance, 16
                ),
                _weighted_mean(
                    state.base_relative,
                    state.base_distance,
                    self.max_k,
                ),
                _weighted_mean(
                    state.noisy_relative, state.noisy_distance, 8
                ),
                _weighted_mean(
                    state.noisy_relative, state.noisy_distance, 16
                ),
                state.base_query - state.noisy,
                state.base_query - state.v29_query,
                state.v29_query - state.v12_query,
            ],
            dim=2,
        )
        stages = state.stage_displacements
        return jt.concat([means, stages], dim=2)

    def _decode(self, scalar, vectors, layer, cap):
        coefficients = jt.tanh(
            layer(scalar.reshape(-1, self.channels))
        ).reshape(
            scalar.shape[0],
            scalar.shape[1],
            self.vector_channels,
        )
        update = (
            coefficients.unsqueeze(-1) * vectors
        ).sum(dim=2) / math.sqrt(float(self.vector_channels))
        return bounded_vector(update, cap), coefficients

    def execute(self, state):
        batch, points = state.base_query.shape[:2]
        scalar = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, points, self.channels)
        vectors = self._initial_vectors(state)
        base_indices = state.indices[:, :, : self.max_k]
        noisy_indices = knn_indices(state.noisy, self.max_k)
        graph_sequence = [base_indices, noisy_indices, base_indices]
        for block, indices in zip(
            self.blocks[:3], graph_sequence
        ):
            fields = _graph_fields(
                state, state.base_query, indices
            )
            scalar, vectors = block(
                scalar,
                vectors,
                indices,
                fields["geometry"],
                fields["current_relative"],
            )
        radius = state.base_distance[
            :, :, :16
        ].mean(dim=2)
        cap1 = (
            0.042 * radius
            + 0.022 * vector_norm(state.base_delta)
            + 1e-5
        )
        update1, coefficients1 = self._decode(
            scalar, vectors, self.decode1, cap1
        )
        first = state.base_query + update1
        dynamic_indices = knn_indices(first, self.max_k)
        dynamic_fields = _graph_fields(
            state, first, dynamic_indices
        )
        scalar, vectors = self.blocks[3](
            scalar,
            vectors,
            dynamic_indices,
            dynamic_fields["geometry"],
            dynamic_fields["current_relative"],
        )
        cap2 = 0.65 * cap1
        update2, coefficients2 = self._decode(
            scalar, vectors, self.decode2, cap2
        )
        trust = jt.sigmoid(
            self.trust(scalar.reshape(-1, self.channels))
        ).reshape(batch, points, 1)
        refined = state.base_query + trust * (
            update1 + update2
        )
        return refined, {
            "step_predictions": [first, refined],
            "scalar_features": scalar,
            "vector_channels": vectors,
            "coefficients1": coefficients1,
            "coefficients2": coefficients2,
            "trust": trust,
            "update": refined - state.base_query,
        }


class TrajectoryConditionedProjectionHead(nn.Module):
    """Shared finite-step vector field conditioned by frozen trajectory."""

    def __init__(
        self,
        max_k=24,
        channels=120,
        num_steps=3,
    ):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        self.input_proj = LinearBNReLU(42, self.channels)
        self.step_condition = nn.Linear(5, self.channels)
        self.shared_encoder1 = SparseGeoAttention(
            self.channels, self.channels, geometry_dim=9
        )
        self.shared_encoder2 = SparseGeoAttention(
            self.channels, self.channels, geometry_dim=9
        )
        self.field = DenseEquivariantField(self.channels)
        self.mix = nn.Linear(self.channels, 5)
        self.stop = nn.Linear(self.channels, 1)
        _zero_linear(self.mix)
        _zero_linear(self.stop)

    def execute(self, state):
        batch, points = state.base_query.shape[:2]
        initial = self.input_proj(
            state.initial.reshape(-1, 42)
        ).reshape(batch, points, self.channels)
        current = state.base_query
        steps = []
        mixes = []
        stops = []
        fields_log = []
        for step in range(self.num_steps):
            t = float(step) / float(max(1, self.num_steps - 1))
            trajectory = state.stage_displacements[
                :, :, min(step, 3)
            ]
            condition_values = jt.concat(
                [
                    vector_norm(trajectory),
                    vector_norm(state.base_delta),
                    vector_norm(current - state.base_query),
                    jt.ones(
                        (batch, points, 1)
                    )
                    * t,
                    jt.ones(
                        (batch, points, 1)
                    )
                    / float(self.num_steps),
                ],
                dim=-1,
            )
            condition = self.step_condition(
                condition_values.reshape(-1, 5)
            ).reshape(batch, points, self.channels)
            indices = knn_indices(current, self.max_k)
            fields = _graph_fields(state, current, indices)
            features = self.shared_encoder1(
                initial + condition,
                indices,
                fields["geometry"],
            )
            features = self.shared_encoder2(
                features,
                indices,
                fields["geometry"],
            )
            learned_field, _, _ = self.field(
                features, fields
            )
            mix = jt.tanh(
                self.mix(
                    features.reshape(-1, self.channels)
                )
            ).reshape(batch, points, 5)
            stop = jt.sigmoid(
                self.stop(
                    features.reshape(-1, self.channels)
                )
            ).reshape(batch, points, 1)
            bases = jt.stack(
                [
                    learned_field,
                    trajectory,
                    state.base_delta,
                    state.correction,
                    _weighted_mean(
                        fields["current_relative"],
                        fields["distance"],
                        min(16, self.max_k),
                    ),
                ],
                dim=2,
            )
            update = (
                mix.unsqueeze(-1) * bases
            ).sum(dim=2) / math.sqrt(5.0)
            radius = fields["distance"][
                :, :, :16
            ].mean(dim=2)
            cap = (
                (0.032 - 0.006 * t) * radius
                + 0.012 * vector_norm(state.base_delta)
                + 1e-5
            )
            update = bounded_vector(update, cap)
            update = update * (1.0 - 0.45 * stop)
            current = current + update
            steps.append(current)
            mixes.append(mix)
            stops.append(stop)
            fields_log.append(learned_field)
        return current, {
            "step_predictions": steps,
            "mix": jt.stack(mixes, dim=2),
            "stop": jt.stack(stops, dim=2),
            "learned_field": jt.stack(fields_log, dim=2),
            "update": current - state.base_query,
        }


class SurfaceJetTensorHead(nn.Module):
    """Cross-scale polynomial surface projectors inside the backbone."""

    def __init__(
        self,
        max_k=32,
        channels=120,
        num_steps=2,
    ):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        self.input_proj = LinearBNReLU(50, self.channels)
        self.encoder1 = SparseGeoAttention(
            self.channels, self.channels, geometry_dim=9
        )
        self.encoder2 = SparseGeoAttention(
            self.channels, self.channels, geometry_dim=9
        )
        self.edge_hidden = LinearBNReLU(
            2 * self.channels + 15, self.channels
        )
        self.edge_output = nn.Linear(self.channels, 7)
        self.point_gate = nn.Linear(self.channels, 1)
        _zero_linear(self.edge_output)
        _zero_linear(self.point_gate)

    def _jet(self, relative, distance):
        fine_normal, fine_tangent, fine_covariance = (
            _surface_projector(relative, distance, 12)
        )
        coarse_normal, _, coarse_covariance = (
            _surface_projector(
                relative, distance, self.max_k
            )
        )
        difference = fine_normal - coarse_normal
        curvature = (
            difference ** 2
        ).sum([-1, -2], keepdims=False).unsqueeze(-1)
        covariance_gap = (
            (fine_covariance - coarse_covariance) ** 2
        ).sum([-1, -2], keepdims=False).unsqueeze(-1)
        diagonal = jt.stack(
            [
                fine_normal[..., 0, 0],
                fine_normal[..., 1, 1],
                fine_normal[..., 2, 2],
                coarse_normal[..., 0, 0],
                coarse_normal[..., 1, 1],
                coarse_normal[..., 2, 2],
            ],
            dim=-1,
        )
        invariants = jt.concat(
            [curvature, covariance_gap, diagonal], dim=-1
        )
        return (
            fine_normal,
            fine_tangent,
            coarse_normal,
            invariants,
        )

    def _step(self, state, current, scalar, step):
        indices = (
            state.indices[:, :, : self.max_k]
            if step == 0
            else knn_indices(current, self.max_k)
        )
        fields = _graph_fields(state, current, indices)
        relative = fields["current_relative"]
        distance = fields["distance"]
        fine_normal, fine_tangent, coarse_normal, invariants = (
            self._jet(relative, distance)
        )
        fine_normal_edge = _project_edges(
            fine_normal, relative
        )
        fine_tangent_edge = _project_edges(
            fine_tangent, relative
        )
        coarse_normal_edge = _project_edges(
            coarse_normal, relative
        )
        neighbors = batched_index(scalar, indices)
        centers = scalar.unsqueeze(2).broadcast(
            (
                scalar.shape[0],
                scalar.shape[1],
                self.max_k,
                self.channels,
            )
        )
        point_invariants = invariants.unsqueeze(2).broadcast(
            (
                scalar.shape[0],
                scalar.shape[1],
                self.max_k,
                8,
            )
        )
        edge_invariants = jt.concat(
            [fields["geometry"], point_invariants[:, :, :, :6]],
            dim=-1,
        )
        edge = jt.concat(
            [
                centers,
                neighbors - centers,
                edge_invariants,
            ],
            dim=-1,
        )
        output = self.edge_output(
            self.edge_hidden(
                edge.reshape(-1, edge.shape[-1])
            )
        ).reshape(
            scalar.shape[0],
            scalar.shape[1],
            self.max_k,
            7,
        )
        coefficients = jt.tanh(output[..., :6])
        importance = nn.softmax(output[..., 6], dim=2)
        bases = jt.stack(
            [
                fine_normal_edge,
                fine_tangent_edge,
                coarse_normal_edge,
                relative,
                fields["noisy_relative"],
                fields["base_delta_relative"],
            ],
            dim=3,
        )
        edge_field = (
            coefficients.unsqueeze(-1) * bases
        ).sum(dim=3)
        field = (
            importance.unsqueeze(-1) * edge_field
        ).sum(dim=2)
        gate = jt.sigmoid(
            self.point_gate(
                scalar.reshape(-1, self.channels)
            )
        ).reshape(
            scalar.shape[0], scalar.shape[1], 1
        )
        radius = distance[:, :, :16].mean(dim=2)
        cap = (
            0.040 * radius
            + 0.020 * vector_norm(state.base_delta)
            + 1e-5
        )
        update = bounded_vector(gate * field, cap)
        return current + update, {
            "indices": indices,
            "invariants": invariants,
            "coefficients": coefficients,
            "importance": importance,
            "update": update,
        }

    def execute(self, state):
        relative = state.base_relative[:, :, : self.max_k]
        distance = state.base_distance[:, :, : self.max_k]
        _, _, _, invariants = self._jet(relative, distance)
        batch, points = state.base_query.shape[:2]
        scalar = self.input_proj(
            jt.concat([state.initial, invariants], dim=-1).reshape(
                -1, 50
            )
        ).reshape(batch, points, self.channels)
        indices, geometry = state.prefix(self.max_k)
        scalar = self.encoder1(scalar, indices, geometry)
        current = state.base_query
        steps = []
        diagnostics = []
        for step in range(self.num_steps):
            current, diagnostic = self._step(
                state, current, scalar, step
            )
            steps.append(current)
            diagnostics.append(diagnostic)
            fields = _graph_fields(
                state, current, diagnostic["indices"]
            )
            scalar = self.encoder2(
                scalar,
                diagnostic["indices"],
                fields["geometry"],
            )
        return current, {
            "step_predictions": steps,
            "step_diagnostics": diagnostics,
            "update": current - state.base_query,
        }


class FrozenOrthogonalDenoiser(nn.Module):
    def __init__(
        self,
        v45_model,
        mode,
        v45_strength=1.25,
        max_k=32,
        channels=112,
        num_steps=2,
    ):
        super().__init__()
        if mode not in MODES.values():
            raise ValueError("unsupported mode: %s" % mode)
        self.v45 = v45_model
        self.mode = str(mode)
        self.v45_strength = float(v45_strength)
        for parameter in self.v45.parameters():
            parameter.stop_grad()
        if mode == MODES["v62"]:
            self.head = SurfaceFirstRemesherHead(
                max_k=max_k,
                channels=channels,
                num_steps=num_steps,
            )
        elif mode == MODES["v63"]:
            self.head = DualTopologyVectorBackboneHead(
                max_k=max_k,
                channels=channels,
                vector_channels=12,
            )
        elif mode == MODES["v64"]:
            self.head = TrajectoryConditionedProjectionHead(
                max_k=max_k,
                channels=channels,
                num_steps=max(3, num_steps),
            )
        else:
            self.head = SurfaceJetTensorHead(
                max_k=max_k,
                channels=channels,
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
        strength = float(refinement_strength)
        if abs(strength) < 1e-12:
            stitch_confidence = v45_auxiliary[
                "stitch_confidence"
            ]
            auxiliary = {
                "step_predictions": [base_query],
                "update": jt.zeros_like(base_query),
                "stitch_confidence": stitch_confidence,
                "parent_stitch_confidence": stitch_confidence,
                "exact_v45_bypass": True,
            }
            return (
                base_query,
                base_query,
                v29_query,
                v12_query,
                base_query,
                auxiliary,
                v45_outputs,
                v29_outputs,
            )
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
        prediction = base_query + strength * (
            refined - base_query
        )
        stitch_confidence = v45_auxiliary[
            "stitch_confidence"
        ]
        auxiliary["stitch_confidence"] = stitch_confidence
        auxiliary["parent_stitch_confidence"] = stitch_confidence
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


def patch_based_orthogonal_pair(
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


def patch_based_orthogonal(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    beta=12.0,
    patch_batch=8,
    refinement_strength=1.0,
):
    strength = float(refinement_strength)
    if abs(strength) < 1e-12:
        original = np.asarray(points, dtype=np.float32)
        num_points = original.shape[0]
        num_patches = max(
            1, int(seed_k * num_points / patch_size)
        )
        seed_indices = farthest_point_sampling_np(
            original, num_patches
        )
        seeds = original[seed_indices]
        patch_distances, point_indices, patches = (
            knn_points_np(
                seeds,
                original,
                min(patch_size, num_points),
            )
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
                    batch, refinement_strength=0.0
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
    base, active = patch_based_orthogonal_pair(
        model,
        points,
        patch_size=patch_size,
        seed_k=seed_k,
        beta=beta,
        patch_batch=patch_batch,
    )
    return base + strength * (active - base)
