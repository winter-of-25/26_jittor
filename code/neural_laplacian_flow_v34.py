import numpy as np

import jittor as jt
from jittor import nn

from iterativepfn_v12 import batched_index, knn_indices
from jittor_port_models import farthest_point_sampling_np, knn_points_np
from pseudo_query_corrector_v29 import (
    InvariantTokenContext,
    LinearBNReLU,
    SparseGeoAttention,
)


def vector_norm(value, keepdims=True):
    return jt.sqrt((value ** 2).sum(-1, keepdims=keepdims) + 1e-8)


class StableNeuralLaplacianFlowHead(nn.Module):
    """Shared-weight micro-step graph flow with a bounded equivariant update."""

    FEATURE_DIM = 44
    GEOMETRY_DIM = 11
    NUM_BASES = 9

    def __init__(self, max_k=32, channels=128, num_steps=2):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.num_steps = int(num_steps)
        if self.max_k < 32:
            raise ValueError("max_k must be at least 32")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")

        self.input_proj = LinearBNReLU(self.FEATURE_DIM, 64)
        self.spatial1 = SparseGeoAttention(
            64, 96, geometry_dim=self.GEOMETRY_DIM
        )
        self.spatial2 = SparseGeoAttention(
            96, self.channels, geometry_dim=self.GEOMETRY_DIM
        )
        self.context = InvariantTokenContext(self.channels, num_tokens=6)
        self.recurrent_fuse = LinearBNReLU(
            3 * self.channels + self.FEATURE_DIM, self.channels
        )

        self.edge_hidden = LinearBNReLU(
            2 * self.channels + self.GEOMETRY_DIM, self.channels
        )
        self.edge_logits = nn.Linear(self.channels, 3)
        self.update_head = nn.Linear(
            self.channels, self.NUM_BASES + 2
        )

        # The new branch starts as an exact V29 identity mapping.
        for layer in (self.edge_logits, self.update_head):
            layer.weight.assign(jt.zeros_like(layer.weight))
            layer.bias.assign(jt.zeros_like(layer.bias))

    @staticmethod
    def _stats(distance, count):
        selected = distance[:, :, : int(count)]
        mean = selected.mean(dim=2)
        variance = ((selected - mean.unsqueeze(2)) ** 2).mean(dim=2)
        std = jt.sqrt(variance + 1e-8)
        maximum = selected.max(dim=2)
        return jt.concat([mean, std, maximum], dim=-1)

    def _geometry(
        self,
        current,
        v29_query,
        noisy,
        v12_query,
        correction,
        previous_update,
        indices,
        time_value,
    ):
        current_relative = (
            batched_index(current, indices) - current.unsqueeze(2)
        )
        v29_relative = (
            batched_index(v29_query, indices) - v29_query.unsqueeze(2)
        )
        noisy_relative = (
            batched_index(noisy, indices) - noisy.unsqueeze(2)
        )
        v12_relative = (
            batched_index(v12_query, indices) - v12_query.unsqueeze(2)
        )
        correction_relative = (
            batched_index(correction, indices) - correction.unsqueeze(2)
        )
        previous_relative = (
            batched_index(previous_update, indices)
            - previous_update.unsqueeze(2)
        )

        current_distance = vector_norm(current_relative)
        v29_distance = vector_norm(v29_relative)
        noisy_distance = vector_norm(noisy_relative)
        v12_distance = vector_norm(v12_relative)
        correction_distance = vector_norm(correction_relative)
        previous_distance = vector_norm(previous_relative)
        current_scale = current_distance.mean(dim=2, keepdims=True)
        v29_scale = v29_distance.mean(dim=2, keepdims=True)
        noisy_scale = noisy_distance.mean(dim=2, keepdims=True)
        v12_scale = v12_distance.mean(dim=2, keepdims=True)
        cosine_current_v29 = (
            current_relative * v29_relative
        ).sum(-1, keepdims=True) / (
            current_distance * v29_distance + 1e-8
        )
        cosine_v29_noisy = (
            v29_relative * noisy_relative
        ).sum(-1, keepdims=True) / (
            v29_distance * noisy_distance + 1e-8
        )
        time_channel = jt.ones_like(current_distance) * float(time_value)
        geometry = jt.concat(
            [
                current_distance / (current_scale + 1e-8),
                v29_distance / (v29_scale + 1e-8),
                noisy_distance / (noisy_scale + 1e-8),
                v12_distance / (v12_scale + 1e-8),
                (current_distance - v29_distance) / (v29_scale + 1e-8),
                (v29_distance - noisy_distance) / (noisy_scale + 1e-8),
                cosine_current_v29,
                cosine_v29_noisy,
                correction_distance / (v29_scale + 1e-8),
                previous_distance / (v29_scale + 1e-8),
                time_channel,
            ],
            dim=-1,
        )
        return {
            "geometry": geometry,
            "current_relative": current_relative,
            "v29_relative": v29_relative,
            "noisy_relative": noisy_relative,
            "v12_relative": v12_relative,
            "current_distance": current_distance,
            "v29_distance": v29_distance,
            "noisy_distance": noisy_distance,
            "v12_distance": v12_distance,
            "correction_distance": correction_distance,
        }

    def _initial_features(
        self,
        noisy,
        current,
        v12_query,
        v29_query,
        correction,
        stage_displacements,
        previous_update,
        fields,
        time_value,
    ):
        stage_magnitude = vector_norm(stage_displacements, keepdims=False)
        if stage_magnitude.shape[2] != 4:
            raise ValueError("V34 expects four V12 trajectory stages")
        base_velocity = v12_query - noisy
        base_magnitude = vector_norm(base_velocity)
        correction_magnitude = vector_norm(correction)
        drift_magnitude = vector_norm(current - v29_query)
        previous_magnitude = vector_norm(previous_update)
        correction_cosine = (
            correction * base_velocity
        ).sum(-1, keepdims=True) / (
            correction_magnitude * base_magnitude + 1e-8
        )

        strain_current = (
            fields["current_distance"] - fields["v29_distance"]
        ) / (fields["v29_distance"].mean(dim=2, keepdims=True) + 1e-8)
        strain_noisy = (
            fields["v29_distance"] - fields["noisy_distance"]
        ) / (fields["noisy_distance"].mean(dim=2, keepdims=True) + 1e-8)
        strain_current_mean = strain_current.mean(dim=2)
        strain_current_std = jt.sqrt(
            ((strain_current - strain_current_mean.unsqueeze(2)) ** 2).mean(dim=2)
            + 1e-8
        )
        strain_noisy_mean = strain_noisy.mean(dim=2)
        strain_noisy_std = jt.sqrt(
            ((strain_noisy - strain_noisy_mean.unsqueeze(2)) ** 2).mean(dim=2)
            + 1e-8
        )
        correction_mean = fields["correction_distance"].mean(dim=2)
        correction_std = jt.sqrt(
            (
                (
                    fields["correction_distance"]
                    - correction_mean.unsqueeze(2)
                ) ** 2
            ).mean(dim=2)
            + 1e-8
        )
        time_feature = jt.ones_like(base_magnitude) * float(time_value)
        features = jt.concat(
            [
                stage_magnitude,
                base_magnitude,
                correction_magnitude,
                drift_magnitude,
                previous_magnitude,
                correction_cosine,
                self._stats(fields["current_distance"], 8),
                self._stats(fields["current_distance"], 16),
                self._stats(fields["current_distance"], 32),
                self._stats(fields["v29_distance"], 8),
                self._stats(fields["v29_distance"], 32),
                self._stats(fields["noisy_distance"], 8),
                self._stats(fields["noisy_distance"], 32),
                self._stats(fields["v12_distance"], 8),
                self._stats(fields["v12_distance"], 32),
                strain_current_mean,
                strain_current_std,
                strain_noisy_mean,
                strain_noisy_std,
                correction_mean,
                correction_std,
                time_feature,
                time_feature ** 2,
            ],
            dim=-1,
        )
        if features.shape[2] != self.FEATURE_DIM:
            raise ValueError("unexpected V34 feature dimension: %s" % (features.shape,))
        return features

    def _neural_laplacians(self, features, indices, fields):
        batch, num_points, channels = features.shape
        k = indices.shape[2]
        neighbors = batched_index(features, indices)
        centers = features.unsqueeze(2).broadcast(
            (batch, num_points, k, channels)
        )
        edge = jt.concat(
            [centers, neighbors - centers, fields["geometry"]], dim=-1
        )
        logits = self.edge_logits(
            self.edge_hidden(edge.reshape(-1, edge.shape[-1]))
        ).reshape(batch, num_points, k, 3)
        laplacians = []
        weights = []
        for scale_index, count in enumerate((8, 16, 32)):
            selected_logits = logits[:, :, :count, scale_index]
            selected_weights = nn.softmax(selected_logits, dim=2).unsqueeze(-1)
            laplacian = (
                selected_weights
                * fields["current_relative"][:, :, :count]
            ).sum(dim=2)
            laplacians.append(laplacian)
            weights.append(selected_weights)
        return laplacians, weights

    def _step(
        self,
        noisy,
        current,
        v12_query,
        v29_query,
        correction,
        stage_displacements,
        previous_update,
        hidden,
        step_index,
    ):
        batch, num_points, _ = current.shape
        time_value = float(step_index + 1) / float(self.num_steps)
        indices = knn_indices(current, self.max_k)
        fields = self._geometry(
            current,
            v29_query,
            noisy,
            v12_query,
            correction,
            previous_update,
            indices,
            time_value,
        )
        initial = self._initial_features(
            noisy,
            current,
            v12_query,
            v29_query,
            correction,
            stage_displacements,
            previous_update,
            fields,
            time_value,
        )
        encoded = self.input_proj(initial.reshape(-1, self.FEATURE_DIM)).reshape(
            batch, num_points, 64
        )
        encoded = self.spatial1(
            encoded, indices[:, :, :16], fields["geometry"][:, :, :16]
        )
        encoded = self.spatial2(encoded, indices, fields["geometry"])
        context = self.context(encoded)
        fused = self.recurrent_fuse(
            jt.concat([encoded, context, hidden, initial], dim=-1).reshape(
                -1, 3 * self.channels + self.FEATURE_DIM
            )
        ).reshape(batch, num_points, self.channels)
        laplacians, edge_weights = self._neural_laplacians(
            fused, indices, fields
        )
        lap8, lap16, lap32 = laplacians
        base_velocity = v12_query - noisy
        bases = jt.stack(
            [
                lap8,
                lap16,
                lap32,
                lap8 - lap16,
                lap16 - lap32,
                v29_query - current,
                correction,
                base_velocity,
                previous_update,
            ],
            dim=2,
        )

        head = self.update_head(fused)
        raw_coefficients = jt.tanh(head[:, :, : self.NUM_BASES])
        coefficient_norm = jt.abs(raw_coefficients).sum(dim=-1, keepdims=True)
        coefficients = 0.45 * raw_coefficients / (1.0 + coefficient_norm)
        mixed = (coefficients.unsqueeze(-1) * bases).sum(dim=2)

        radius = fields["current_distance"].mean(dim=2)
        disagreement = vector_norm(lap8 - lap32) / (radius + 1e-5)
        stability = jt.sigmoid(
            2.0 - 4.0 * disagreement + head[:, :, self.NUM_BASES : self.NUM_BASES + 1]
        )
        learned_scale = 0.85 + 0.30 * jt.tanh(
            head[:, :, self.NUM_BASES + 1 : self.NUM_BASES + 2]
        )
        raw_update = stability * learned_scale * mixed
        cap_fraction = 0.18 - 0.04 * time_value
        cap = (
            cap_fraction * radius
            + 0.08 * vector_norm(correction)
            + 1e-5
        )
        update_norm = vector_norm(raw_update)
        update = raw_update / (1.0 + update_norm / cap)
        next_points = current + update
        auxiliary = {
            "coefficients": coefficients,
            "raw_coefficients": raw_coefficients,
            "edge_weights": edge_weights,
            "laplacians": laplacians,
            "disagreement": disagreement,
            "stability": stability,
            "learned_scale": learned_scale,
            "cap": cap,
            "update": update,
        }
        return next_points, update, fused, auxiliary

    def execute(
        self,
        noisy,
        v12_query,
        v29_query,
        correction,
        stage_displacements,
    ):
        batch, num_points, _ = noisy.shape
        current = v29_query
        previous_update = jt.zeros_like(v29_query)
        hidden = jt.zeros((batch, num_points, self.channels))
        predictions = []
        step_auxiliary = []
        for step_index in range(self.num_steps):
            current, previous_update, hidden, auxiliary = self._step(
                noisy,
                current,
                v12_query,
                v29_query,
                correction,
                stage_displacements,
                previous_update,
                hidden,
                step_index,
            )
            predictions.append(current)
            step_auxiliary.append(auxiliary)
        return current, predictions, step_auxiliary


class V29StableNeuralLaplacianFlow(nn.Module):
    def __init__(self, v29_model, max_k=32, channels=128, num_steps=2):
        super().__init__()
        self.v29 = v29_model
        for parameter in self.v29.parameters():
            parameter.stop_grad()
        self.flow = StableNeuralLaplacianFlowHead(
            max_k=max_k, channels=channels, num_steps=num_steps
        )

    def trainable_parameters(self):
        return list(self.flow.parameters())

    def v29_outputs(self, noisy):
        self.v29.eval()
        self.v29.baseline.eval()
        with jt.no_grad():
            baseline_outputs = self.v29.baseline_outputs(noisy)
            v29_query, v12_query, correction, _, intermediates = self.v29(
                noisy, correction_scale=1.0, baseline_outputs=baseline_outputs
            )
            stage_displacements = jt.stack(baseline_outputs[1], dim=2)
        return (
            v29_query,
            v12_query,
            correction,
            stage_displacements,
            intermediates,
        )

    def execute(self, noisy, flow_strength=1.0, v29_outputs=None):
        if v29_outputs is None:
            v29_outputs = self.v29_outputs(noisy)
        (
            v29_query,
            v12_query,
            correction,
            stage_displacements,
            intermediates,
        ) = v29_outputs
        refined, step_predictions, step_auxiliary = self.flow(
            noisy,
            v12_query,
            v29_query,
            correction,
            stage_displacements,
        )
        prediction = v29_query + float(flow_strength) * (
            refined - v29_query
        )
        return (
            prediction,
            v29_query,
            v12_query,
            refined,
            step_predictions,
            step_auxiliary,
            intermediates,
        )


def soft_stitch(predictions, point_indices, normalized_distances, original, beta):
    flat_indices = point_indices.reshape(-1)
    weights = np.exp(-float(beta) * normalized_distances).astype(np.float32).reshape(-1)
    flat_predictions = predictions.reshape(-1, 3)
    weight_sum = np.zeros(original.shape[0], dtype=np.float32)
    output_sum = np.zeros_like(original, dtype=np.float32)
    np.add.at(weight_sum, flat_indices, weights)
    for axis in range(3):
        np.add.at(
            output_sum[:, axis],
            flat_indices,
            weights * flat_predictions[:, axis],
        )
    valid = weight_sum > 1e-8
    output = original.copy()
    output[valid] = output_sum[valid] / weight_sum[valid, None]
    return output


def patch_based_neural_laplacian_flow(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    beta=12.0,
    patch_batch=10,
    flow_strength=1.0,
):
    original = np.asarray(points, dtype=np.float32)
    num_points = original.shape[0]
    num_patches = max(1, int(seed_k * num_points / patch_size))
    seed_indices = farthest_point_sampling_np(original, num_patches)
    seeds = original[seed_indices]
    patch_distances, point_indices, patches = knn_points_np(
        seeds, original, min(patch_size, num_points)
    )
    centered = patches - seeds[:, None, :]
    normalized_distances = patch_distances / (patch_distances[:, -1:] + 1e-8)
    outputs = []
    with jt.no_grad():
        for start in range(0, num_patches, int(patch_batch)):
            batch = jt.array(
                centered[start : start + int(patch_batch)].astype(np.float32)
            )
            prediction = model(
                batch, flow_strength=float(flow_strength)
            )[0]
            outputs.append(prediction.numpy())
    predictions = np.concatenate(outputs, axis=0) + seeds[:, None, :]
    return soft_stitch(
        predictions,
        point_indices,
        normalized_distances,
        original,
        float(beta),
    )
