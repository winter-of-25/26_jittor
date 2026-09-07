import math

import jittor as jt
from jittor import nn

from competitive_residual_v39_42 import (
    GeometryState,
    bounded_vector,
    vector_norm,
)
from hierarchical_residual_v43_46 import _graph_fields
from iterativepfn_v12 import batched_index, knn_indices
from orthogonal_backbones_v62_v65 import (
    _project_edges,
    _surface_projector,
    _zero_linear,
)
from pseudo_query_corrector_v29 import LinearBNReLU, SparseGeoAttention


MODE = "v67_noise_conditioned_dual_path"


class NoiseConditionedDualPathHead(nn.Module):
    """Patch-noise-conditioned recurrent surface and trajectory experts."""

    def __init__(self, max_k=32, channels=128, num_steps=3):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        self.input_proj = LinearBNReLU(52, self.channels)
        self.encoders = nn.ModuleList(
            [
                SparseGeoAttention(
                    self.channels,
                    self.channels,
                    geometry_dim=9,
                )
                for _ in range(self.num_steps + 1)
            ]
        )
        self.context_hidden = LinearBNReLU(10, self.channels)
        self.film = nn.Linear(self.channels, 2 * self.channels)
        self.noise_output = nn.Linear(self.channels, 1)
        self.edge_hidden = LinearBNReLU(
            2 * self.channels + 19,
            self.channels,
        )
        self.edge_output = nn.Linear(self.channels, 3 * 7)
        self.router_hidden = LinearBNReLU(
            2 * self.channels + 10,
            self.channels,
        )
        self.router_output = nn.Linear(self.channels, 3)
        self.step_control = nn.Linear(self.channels + 10, 2)
        for layer in (
            self.film,
            self.noise_output,
            self.edge_output,
            self.router_output,
            self.step_control,
        ):
            _zero_linear(layer)

    def _local_context(self, state, current, fields):
        relative = fields["current_relative"]
        distance = fields["distance"]
        fine_normal, _, fine_covariance = _surface_projector(
            relative, distance, 12
        )
        coarse_normal, _, coarse_covariance = _surface_projector(
            relative, distance, self.max_k
        )
        projector_gap = (
            (fine_normal - coarse_normal) ** 2
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
            [projector_gap, covariance_gap, diagonal], dim=-1
        )
        radius = distance[:, :, :16].mean(dim=2)
        noisy_distance = vector_norm(fields["noisy_relative"])
        roughness = jt.abs(noisy_distance - distance).mean(dim=2)
        roughness = roughness / (radius + 1e-8)
        parent_magnitude = vector_norm(state.parent_update)
        parent_magnitude = parent_magnitude / (radius + 1e-8)
        context = jt.concat(
            [invariants, roughness, parent_magnitude], dim=-1
        )
        return (
            context,
            fine_normal,
            coarse_normal,
            radius,
        )

    def _condition(self, scalar, context):
        pooled = context.mean(dim=1)
        global_scalar = self.context_hidden(pooled)
        film = self.film(global_scalar)
        scale = jt.tanh(film[:, : self.channels]).unsqueeze(1)
        shift = film[:, self.channels :].unsqueeze(1)
        scalar = scalar * (1.0 + 0.25 * scale) + 0.10 * shift
        noise_estimate = 0.004 + 0.018 * jt.sigmoid(
            self.noise_output(global_scalar)
        )
        return scalar, global_scalar, noise_estimate

    def _expert_fields(
        self,
        state,
        scalar,
        fields,
        context,
        fine_normal,
        coarse_normal,
    ):
        indices = fields["indices"]
        batch, points, neighbors_count = indices.shape
        neighbors = batched_index(scalar, indices)
        centers = scalar.unsqueeze(2).broadcast(
            (batch, points, neighbors_count, self.channels)
        )
        edge_context = context.unsqueeze(2).broadcast(
            (batch, points, neighbors_count, 10)
        )
        edge = jt.concat(
            [
                centers,
                neighbors - centers,
                fields["geometry"],
                edge_context,
            ],
            dim=-1,
        )
        output = self.edge_output(
            self.edge_hidden(edge.reshape(-1, edge.shape[-1]))
        ).reshape(batch, points, neighbors_count, 3, 7)
        coefficients = jt.tanh(output[..., :6])
        importance = nn.softmax(output[..., 6], dim=2)

        relative = fields["current_relative"]
        fine_normal_edge = _project_edges(fine_normal, relative)
        fine_tangent_edge = relative - fine_normal_edge
        coarse_normal_edge = _project_edges(coarse_normal, relative)
        parent_center = state.parent_update.unsqueeze(2).broadcast(
            (batch, points, neighbors_count, 3)
        )
        base_center = fields["base_delta"].unsqueeze(2).broadcast(
            (batch, points, neighbors_count, 3)
        )
        bases = [
            jt.stack(
                [
                    fine_normal_edge,
                    fine_tangent_edge,
                    relative,
                    fields["noisy_relative"],
                    fields["base_delta_relative"],
                    parent_center,
                ],
                dim=3,
            ),
            jt.stack(
                [
                    coarse_normal_edge,
                    fine_tangent_edge,
                    relative,
                    fields["noisy_relative"],
                    fields["v29_relative"],
                    base_center,
                ],
                dim=3,
            ),
            jt.stack(
                [
                    fine_normal_edge,
                    coarse_normal_edge,
                    fields["correction_relative"],
                    fields["base_delta_relative"],
                    fields["v29_relative"],
                    parent_center,
                ],
                dim=3,
            ),
        ]
        expert_fields = []
        for expert_index, expert_bases in enumerate(bases):
            edge_field = (
                coefficients[:, :, :, expert_index].unsqueeze(-1)
                * expert_bases
            ).sum(dim=3)
            field = (
                importance[:, :, :, expert_index].unsqueeze(-1)
                * edge_field
            ).sum(dim=2)
            expert_fields.append(field)
        return (
            jt.stack(expert_fields, dim=2),
            coefficients,
            importance,
        )

    def execute(self, state):
        initial_fields = _graph_fields(
            state,
            state.base_query,
            state.indices[:, :, : self.max_k],
        )
        (
            initial_context,
            _,
            _,
            _,
        ) = self._local_context(
            state, state.base_query, initial_fields
        )
        batch, points = state.base_query.shape[:2]
        v67_input_feature = jt.concat(
            [state.initial, initial_context], dim=-1
        )
        scalar = self.input_proj(
            v67_input_feature.reshape(-1, 52)
        ).reshape(batch, points, self.channels)
        scalar, global_scalar, noise_estimate = self._condition(
            scalar, initial_context
        )
        scalar = self.encoders[0](
            scalar,
            initial_fields["indices"],
            initial_fields["geometry"],
        )

        current = state.base_query
        steps = []
        diagnostics = []
        routers = []
        for step in range(self.num_steps):
            previous = current
            indices = (
                state.indices[:, :, : self.max_k]
                if step == 0
                else knn_indices(current, self.max_k)
            )
            fields = _graph_fields(state, current, indices)
            context, fine_normal, coarse_normal, radius = (
                self._local_context(state, current, fields)
            )
            scalar = self.encoders[step + 1](
                scalar, indices, fields["geometry"]
            )
            expert_fields, coefficients, importance = (
                self._expert_fields(
                    state,
                    scalar,
                    fields,
                    context,
                    fine_normal,
                    coarse_normal,
                )
            )
            global_points = global_scalar.unsqueeze(1).broadcast(
                (batch, points, self.channels)
            )
            router = nn.softmax(
                self.router_output(
                    self.router_hidden(
                        jt.concat(
                            [scalar, global_points, context], dim=-1
                        ).reshape(-1, 2 * self.channels + 10)
                    )
                ).reshape(batch, points, 3),
                dim=2,
            )
            field = (
                router.unsqueeze(-1) * expert_fields
            ).sum(dim=2)
            control = self.step_control(
                jt.concat([scalar, context], dim=-1).reshape(
                    -1, self.channels + 10
                )
            ).reshape(batch, points, 2)
            gain = jt.sigmoid(control[..., 0:1])
            stop = jt.sigmoid(control[..., 1:2])
            effective_gate = gain * (1.0 - 0.45 * stop)
            pre_cap_update = effective_gate * field
            cap = (
                (0.050 - 0.006 * float(step)) * radius
                + 0.020 * vector_norm(state.parent_update)
                + 1e-5
            )
            update = bounded_vector(pre_cap_update, cap)
            current = current + update
            steps.append(current)
            routers.append(router)
            diagnostics.append(
                {
                    "indices": indices,
                    "invariants": context[..., :8],
                    "context": context,
                    "coefficients": coefficients,
                    "importance": importance,
                    "expert_fields": expert_fields,
                    "router": router,
                    "gain": gain,
                    "stop": stop,
                    "effective_gate": effective_gate,
                    "pre_cap_update": pre_cap_update,
                    "cap": cap,
                    "radius": radius,
                    "previous": previous,
                    "field": field,
                    "update": update,
                }
            )
        return current, {
            "lineage_feature": state.initial,
            "v67_input_feature": v67_input_feature,
            "step_predictions": steps,
            "step_diagnostics": diagnostics,
            "router": jt.stack(routers, dim=2),
            "noise_estimate": noise_estimate,
            "update": current - state.base_query,
        }


class FrozenNoiseConditionedDenoiser(nn.Module):
    """Frozen B03/V65 parent with an exact-zero noise-conditioned branch."""

    def __init__(
        self,
        parent,
        parent_strength=1.5,
        max_k=32,
        channels=128,
        num_steps=3,
    ):
        super().__init__()
        self.parent = parent
        self.mode = MODE
        self.parent_strength = float(parent_strength)
        for parameter in self.parent.parameters():
            parameter.stop_grad()
        self.head = NoiseConditionedDualPathHead(
            max_k=max_k,
            channels=channels,
            num_steps=num_steps,
        )

    def trainable_parameters(self):
        return list(self.head.parameters())

    def set_frozen_eval(self):
        self.parent.eval()
        self.parent.set_frozen_eval()

    def frozen_outputs(self, noisy):
        self.set_frozen_eval()
        with jt.no_grad():
            lower = self.parent.frozen_outputs(noisy)
            output = self.parent(
                noisy,
                refinement_strength=self.parent_strength,
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
        parent_outputs = frozen_outputs
        parent_query = parent_outputs[0]
        v45_query = parent_outputs[1]
        v29_query = parent_outputs[2]
        v12_query = parent_outputs[3]
        parent_auxiliary = parent_outputs[5]
        v29_outputs = parent_outputs[7]
        strength = float(refinement_strength)
        if abs(strength) < 1e-12:
            confidence = parent_auxiliary["stitch_confidence"]
            auxiliary = {
                "step_predictions": [parent_query],
                "update": jt.zeros_like(parent_query),
                "router": jt.ones(
                    (parent_query.shape[0], parent_query.shape[1], 1, 3)
                )
                / 3.0,
                "noise_estimate": jt.zeros(
                    (parent_query.shape[0], 1)
                ),
                "stitch_confidence": confidence,
                "parent_stitch_confidence": confidence,
                "exact_parent_bypass": True,
            }
            return (
                parent_query,
                parent_query,
                v29_query,
                v12_query,
                parent_query,
                auxiliary,
                parent_outputs,
                v29_outputs,
            )
        state = GeometryState(
            noisy,
            v12_query,
            v29_query,
            parent_query,
            v29_outputs[2],
            v29_outputs[3],
            self.head.max_k,
        )
        state.parent_update = parent_query - v45_query
        refined, auxiliary = self.head(state)
        prediction = parent_query + strength * (
            refined - parent_query
        )
        confidence = parent_auxiliary["stitch_confidence"]
        auxiliary["stitch_confidence"] = confidence
        auxiliary["parent_stitch_confidence"] = confidence
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
