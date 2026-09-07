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


class CrossPatchConsensusHead(nn.Module):
    """Dual-context equivariant residual with agreement-aware confidence."""

    def __init__(self, max_k=48, view_k=24, channels=128):
        super().__init__()
        self.max_k = int(max_k)
        self.view_k = int(view_k)
        self.channels = int(channels)

        self.input_proj = LinearBNReLU(35, 64)
        self.base_encoder = SparseGeoAttention(64, 96, geometry_dim=8)
        self.view_encoder = SparseGeoAttention(96, self.channels, geometry_dim=8)
        self.view_context = InvariantTokenContext(self.channels, num_tokens=6)
        self.view_fuse = LinearBNReLU(
            2 * self.channels + 35, self.channels
        )

        self.edge_hidden = LinearBNReLU(
            2 * self.channels + 8, self.channels
        )
        self.edge_output = nn.Linear(self.channels, 6)
        self.precision = nn.Linear(self.channels, 1)

        self.consensus_fuse = LinearBNReLU(
            3 * self.channels + 35, self.channels
        )
        self.agreement_bias = nn.Linear(self.channels, 1)
        self.residual_scale = nn.Linear(self.channels, 1)
        self.patch_confidence = nn.Linear(self.channels, 1)

        for layer in (
            self.edge_output,
            self.precision,
            self.agreement_bias,
            self.residual_scale,
            self.patch_confidence,
        ):
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

    def _geometry(self, noisy, v12_query, v29_query, correction, indices):
        v29_relative = batched_index(v29_query, indices) - v29_query.unsqueeze(2)
        noisy_relative = batched_index(noisy, indices) - noisy.unsqueeze(2)
        v12_relative = batched_index(v12_query, indices) - v12_query.unsqueeze(2)
        correction_relative = (
            batched_index(correction, indices) - correction.unsqueeze(2)
        )
        v29_distance = vector_norm(v29_relative)
        noisy_distance = vector_norm(noisy_relative)
        v12_distance = vector_norm(v12_relative)
        correction_distance = vector_norm(correction_relative)
        v29_scale = v29_distance.mean(dim=2, keepdims=True)
        noisy_scale = noisy_distance.mean(dim=2, keepdims=True)
        v12_scale = v12_distance.mean(dim=2, keepdims=True)
        strain_noisy = (v29_distance - noisy_distance) / (noisy_scale + 1e-8)
        strain_v12 = (v29_distance - v12_distance) / (v12_scale + 1e-8)
        cosine_noisy = (v29_relative * noisy_relative).sum(-1, keepdims=True) / (
            v29_distance * noisy_distance + 1e-8
        )
        cosine_v12 = (v29_relative * v12_relative).sum(-1, keepdims=True) / (
            v29_distance * v12_distance + 1e-8
        )
        geometry = jt.concat(
            [
                v29_distance / (v29_scale + 1e-8),
                noisy_distance / (noisy_scale + 1e-8),
                v12_distance / (v12_scale + 1e-8),
                strain_noisy,
                strain_v12,
                cosine_noisy,
                cosine_v12,
                correction_distance / (v29_scale + 1e-8),
            ],
            dim=-1,
        )
        return {
            "geometry": geometry,
            "v29_relative": v29_relative,
            "noisy_relative": noisy_relative,
            "v12_relative": v12_relative,
            "correction_relative": correction_relative,
            "v29_distance": v29_distance,
            "noisy_distance": noisy_distance,
            "v12_distance": v12_distance,
            "strain_noisy": strain_noisy,
        }

    def _initial_features(
        self, stage_displacements, base_velocity, correction, fields
    ):
        stage_magnitude = vector_norm(stage_displacements, keepdims=False)
        base_magnitude = vector_norm(base_velocity)
        correction_magnitude = vector_norm(correction)
        base_axis = base_velocity / base_magnitude
        correction_cosine = (correction * base_axis).sum(-1, keepdims=True) / (
            correction_magnitude + 1e-8
        )
        strain_mean = fields["strain_noisy"].mean(dim=2)
        strain_std = jt.sqrt(
            (
                (
                    fields["strain_noisy"] - strain_mean.unsqueeze(2)
                ) ** 2
            ).mean(dim=2)
            + 1e-8
        )
        correction_distance = vector_norm(fields["correction_relative"])
        correction_mean = correction_distance.mean(dim=2)
        correction_std = jt.sqrt(
            (
                (correction_distance - correction_mean.unsqueeze(2)) ** 2
            ).mean(dim=2)
            + 1e-8
        )
        return jt.concat(
            [
                stage_magnitude,
                base_magnitude,
                correction_magnitude,
                correction_cosine,
                self._stats(fields["v29_distance"], 8),
                self._stats(fields["v29_distance"], self.view_k),
                self._stats(fields["v29_distance"], self.max_k),
                self._stats(fields["noisy_distance"], 8),
                self._stats(fields["noisy_distance"], self.view_k),
                self._stats(fields["noisy_distance"], self.max_k),
                self._stats(fields["v12_distance"], 8),
                self._stats(fields["v12_distance"], self.max_k),
                strain_mean,
                strain_std,
                correction_mean,
                correction_std,
            ],
            dim=-1,
        )

    def _encode_view(self, base, initial, indices, geometry):
        encoded = self.view_encoder(base, indices, geometry)
        context = self.view_context(encoded)
        fused = self.view_fuse(
            jt.concat([encoded, context, initial], dim=-1).reshape(
                -1, 2 * self.channels + 35
            )
        ).reshape(encoded.shape[0], encoded.shape[1], self.channels)
        return fused

    def _vector_field(
        self,
        features,
        indices,
        geometry,
        v29_relative,
        noisy_relative,
        v12_relative,
        correction_relative,
        correction,
    ):
        batch, num_points, channels = features.shape
        k = indices.shape[2]
        neighbors = batched_index(features, indices)
        centers = features.unsqueeze(2).broadcast((batch, num_points, k, channels))
        edge = jt.concat([centers, neighbors - centers, geometry], dim=-1)
        output = self.edge_output(
            self.edge_hidden(edge.reshape(-1, edge.shape[-1]))
        ).reshape(batch, num_points, k, 6)
        coefficients = jt.tanh(output[:, :, :, :5])
        importance = nn.softmax(output[:, :, :, 5], dim=2).unsqueeze(-1)
        center_correction = correction.unsqueeze(2).broadcast(
            (batch, num_points, k, 3)
        )
        bases = jt.stack(
            [
                v29_relative,
                noisy_relative,
                v12_relative,
                correction_relative,
                center_correction,
            ],
            dim=3,
        )
        candidates = (coefficients.unsqueeze(-1) * bases).sum(dim=3)
        field = (importance * candidates).sum(dim=2)
        return field, coefficients, importance

    def execute(
        self,
        noisy,
        v12_query,
        v29_query,
        correction,
        stage_displacements,
    ):
        batch, num_points, _ = noisy.shape
        base_velocity = v12_query - noisy
        indices = knn_indices(v29_query, self.max_k)
        fields = self._geometry(
            noisy, v12_query, v29_query, correction, indices
        )
        initial = self._initial_features(
            stage_displacements, base_velocity, correction, fields
        )
        base = self.input_proj(initial.reshape(-1, 35)).reshape(
            batch, num_points, 64
        )
        base = self.base_encoder(
            base, indices[:, :, :16], fields["geometry"][:, :, :16]
        )

        near_indices = indices[:, :, : self.view_k]
        near_geometry = fields["geometry"][:, :, : self.view_k]
        wide_indices = indices[:, :, : 2 * self.view_k : 2]
        wide_geometry = fields["geometry"][:, :, : 2 * self.view_k : 2]
        near_features = self._encode_view(
            base, initial, near_indices, near_geometry
        )
        wide_features = self._encode_view(
            base, initial, wide_indices, wide_geometry
        )

        near_field, near_coefficients, near_importance = self._vector_field(
            near_features,
            near_indices,
            near_geometry,
            fields["v29_relative"][:, :, : self.view_k],
            fields["noisy_relative"][:, :, : self.view_k],
            fields["v12_relative"][:, :, : self.view_k],
            fields["correction_relative"][:, :, : self.view_k],
            correction,
        )
        wide_field, wide_coefficients, wide_importance = self._vector_field(
            wide_features,
            wide_indices,
            wide_geometry,
            fields["v29_relative"][:, :, : 2 * self.view_k : 2],
            fields["noisy_relative"][:, :, : 2 * self.view_k : 2],
            fields["v12_relative"][:, :, : 2 * self.view_k : 2],
            fields["correction_relative"][:, :, : 2 * self.view_k : 2],
            correction,
        )

        precision_logits = jt.concat(
            [self.precision(near_features), self.precision(wide_features)],
            dim=-1,
        )
        precision_weights = nn.softmax(precision_logits, dim=-1)
        stacked_fields = jt.stack([near_field, wide_field], dim=2)
        consensus_field = (
            precision_weights.unsqueeze(-1) * stacked_fields
        ).sum(dim=2)

        consensus_features = self.consensus_fuse(
            jt.concat(
                [
                    near_features,
                    wide_features,
                    jt.abs(near_features - wide_features),
                    initial,
                ],
                dim=-1,
            ).reshape(-1, 3 * self.channels + 35)
        ).reshape(batch, num_points, self.channels)
        radius = fields["v29_distance"].mean(dim=2)
        disagreement = vector_norm(near_field - wide_field) / (radius + 1e-5)
        agreement = jt.sigmoid(
            2.0
            - 6.0 * disagreement
            + self.agreement_bias(consensus_features)
        )
        scale = 0.85 + 0.30 * jt.tanh(
            self.residual_scale(consensus_features)
        )
        raw_residual = agreement * scale * consensus_field
        cap = 0.30 * radius + 0.15 * vector_norm(correction) + 1e-5
        residual_norm = vector_norm(raw_residual)
        residual = raw_residual / (1.0 + residual_norm / cap)
        confidence = 1.0 + 0.25 * jt.tanh(
            self.patch_confidence(consensus_features)
        )
        return v29_query + residual, {
            "residual": residual,
            "near_field": near_field,
            "wide_field": wide_field,
            "agreement": agreement,
            "disagreement": disagreement,
            "precision_weights": precision_weights,
            "patch_confidence": confidence,
            "scale": scale,
            "cap": cap,
            "near_coefficients": near_coefficients,
            "wide_coefficients": wide_coefficients,
            "near_importance": near_importance,
            "wide_importance": wide_importance,
        }


class V29CrossPatchConsensus(nn.Module):
    def __init__(self, v29_model, max_k=48, view_k=24, channels=128):
        super().__init__()
        self.v29 = v29_model
        for parameter in self.v29.parameters():
            parameter.stop_grad()
        self.consensus = CrossPatchConsensusHead(
            max_k=max_k, view_k=view_k, channels=channels
        )

    def trainable_parameters(self):
        return list(self.consensus.parameters())

    def v29_outputs(self, noisy):
        self.v29.eval()
        self.v29.baseline.eval()
        with jt.no_grad():
            baseline_outputs = self.v29.baseline_outputs(noisy)
            v29_query, v12_query, correction, _, intermediates = self.v29(
                noisy, correction_scale=1.0, baseline_outputs=baseline_outputs
            )
            stage_displacements = jt.stack(baseline_outputs[1], dim=2)
        return v29_query, v12_query, correction, stage_displacements, intermediates

    def execute(self, noisy, consensus_strength=1.0, v29_outputs=None):
        if v29_outputs is None:
            v29_outputs = self.v29_outputs(noisy)
        v29_query, v12_query, correction, stage_displacements, intermediates = (
            v29_outputs
        )
        refined, auxiliary = self.consensus(
            noisy,
            v12_query,
            v29_query,
            correction,
            stage_displacements,
        )
        prediction = v29_query + float(consensus_strength) * (
            refined - v29_query
        )
        return prediction, v29_query, v12_query, refined, auxiliary, intermediates


def confidence_stitch(
    predictions,
    confidences,
    point_indices,
    normalized_distances,
    original,
    beta,
    confidence_power,
):
    flat_indices = point_indices.reshape(-1)
    spatial = np.exp(-float(beta) * normalized_distances).astype(np.float32)
    calibrated = np.clip(confidences[..., 0], 0.75, 1.25) ** float(
        confidence_power
    )
    weights = (spatial * calibrated).reshape(-1)
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


def patch_based_cross_patch_consensus(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    beta=12.0,
    patch_batch=12,
    consensus_strength=1.0,
    confidence_power=1.0,
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
    predictions = []
    confidences = []
    with jt.no_grad():
        for start in range(0, num_patches, int(patch_batch)):
            batch = jt.array(
                centered[start : start + int(patch_batch)].astype(np.float32)
            )
            prediction, _, _, _, auxiliary, _ = model(
                batch, consensus_strength=float(consensus_strength)
            )
            predictions.append(prediction.numpy())
            if float(consensus_strength) == 0.0:
                confidences.append(
                    np.ones(
                        (prediction.shape[0], prediction.shape[1], 1),
                        dtype=np.float32,
                    )
                )
            else:
                confidences.append(auxiliary["patch_confidence"].numpy())
    predictions = np.concatenate(predictions, axis=0) + seeds[:, None, :]
    confidences = np.concatenate(confidences, axis=0)
    return confidence_stitch(
        predictions,
        confidences,
        point_indices,
        normalized_distances,
        original,
        float(beta),
        float(confidence_power),
    )
