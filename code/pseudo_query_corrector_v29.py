import math

import jittor as jt
import numpy as np
from jittor import nn

from iterativepfn_v12 import IterativePFN, batched_index, knn_indices
from jittor_port_models import farthest_point_sampling_np, knn_points_np


class LinearBNReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(int(input_dim), int(output_dim))
        self.bn = nn.BatchNorm1d(int(output_dim))
        self.relu = nn.ReLU()

    def execute(self, features):
        return self.relu(self.bn(self.linear(features)))


class InvariantTokenContext(nn.Module):
    def __init__(self, channels=128, num_tokens=8, key_dim=32):
        super().__init__()
        self.channels = int(channels)
        self.key_dim = int(key_dim)
        self.token_logits = nn.Linear(self.channels, int(num_tokens))
        self.query = nn.Linear(self.channels, self.key_dim)
        self.key = nn.Linear(self.channels, self.key_dim)
        self.value = nn.Linear(self.channels, self.channels)
        self.output = LinearBNReLU(self.channels, self.channels)

    def execute(self, features):
        batch, num_points, channels = features.shape
        assignment = nn.softmax(self.token_logits(features), dim=1)
        tokens = (features.unsqueeze(2) * assignment.unsqueeze(-1)).sum(dim=1)
        logits = (
            self.query(features).unsqueeze(2) * self.key(tokens).unsqueeze(1)
        ).sum(-1) / math.sqrt(float(self.key_dim))
        attention = nn.softmax(logits, dim=2)
        context = (
            attention.unsqueeze(-1) * self.value(tokens).unsqueeze(1)
        ).sum(dim=2)
        return self.output(context.reshape(-1, channels)).reshape(
            batch, num_points, channels
        )


class SparseGeoAttention(nn.Module):
    def __init__(self, input_dim, output_dim, geometry_dim=6, key_dim=32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.geometry_dim = int(geometry_dim)
        self.key_dim = int(key_dim)
        self.query = nn.Linear(self.input_dim, self.key_dim)
        self.key = nn.Linear(self.input_dim, self.key_dim)
        self.value = nn.Linear(self.input_dim, self.output_dim)
        self.geo_key = nn.Linear(self.geometry_dim, self.key_dim)
        self.geo_value = nn.Linear(self.geometry_dim, self.output_dim)
        self.message1 = LinearBNReLU(
            2 * self.input_dim + self.geometry_dim, self.output_dim
        )
        self.message2 = LinearBNReLU(self.output_dim, self.output_dim)
        self.fuse = LinearBNReLU(2 * self.output_dim, self.output_dim)
        self.skip = nn.Linear(self.input_dim, self.output_dim)
        self.relu = nn.ReLU()

    def execute(self, features, indices, geometry):
        batch, num_points, channels = features.shape
        k = indices.shape[2]
        neighbors = batched_index(features, indices)
        centers = features.unsqueeze(2).broadcast((batch, num_points, k, channels))

        queries = self.query(features).unsqueeze(2)
        keys = self.key(neighbors) + self.geo_key(geometry)
        logits = (queries * keys).sum(-1) / math.sqrt(float(self.key_dim))
        attention = nn.softmax(logits, dim=2)
        values = self.value(neighbors) + self.geo_value(geometry)
        attended = (attention.unsqueeze(-1) * values).sum(dim=2)

        edge = jt.concat([centers, neighbors - centers, geometry], dim=-1)
        messages = self.message2(
            self.message1(edge.reshape(-1, edge.shape[-1]))
        ).reshape(batch, num_points, k, self.output_dim)
        pooled = messages.max(dim=2)
        fused = self.fuse(jt.concat([attended, pooled], dim=-1).reshape(
            -1, 2 * self.output_dim
        )).reshape(batch, num_points, self.output_dim)
        skip = self.skip(features.reshape(-1, channels)).reshape(
            batch, num_points, self.output_dim
        )
        return self.relu(fused + skip)


class PseudoQueryCorrector(nn.Module):
    """Two-stage equivariant correction field around frozen V12 pseudo-queries."""

    def __init__(self, max_k=48, channels=128):
        super().__init__()
        self.max_k = int(max_k)
        self.channels = int(channels)
        self.input_proj = LinearBNReLU(19, 64)
        self.observe1 = SparseGeoAttention(64, 96)
        self.observe2 = SparseGeoAttention(96, self.channels)
        self.observe3 = SparseGeoAttention(self.channels, self.channels)
        self.token_context = InvariantTokenContext(self.channels, num_tokens=8)
        self.stage1_fuse = LinearBNReLU(2 * self.channels, self.channels)
        self.stage1_edge = LinearBNReLU(2 * self.channels + 6, self.channels)
        self.stage1_coeff = nn.Linear(self.channels, 5)

        self.refine = SparseGeoAttention(self.channels, self.channels)
        self.stage2_fuse = LinearBNReLU(2 * self.channels, self.channels)
        self.stage2_edge = LinearBNReLU(2 * self.channels + 6, self.channels)
        self.stage2_coeff = nn.Linear(self.channels, 5)
        self.trajectory = nn.Linear(self.channels, 4)
        self.gate1 = LinearBNReLU(2 * self.channels, 64)
        self.gate2 = nn.Linear(64, 1)

        for layer in (self.stage1_coeff, self.stage2_coeff, self.trajectory):
            layer.weight.assign(jt.zeros_like(layer.weight))
            layer.bias.assign(jt.zeros_like(layer.bias))
        self.gate2.bias.assign(
            jt.array(np.asarray([-1.5], dtype=np.float32))
        )

    @staticmethod
    def _distance(relative):
        return jt.sqrt((relative ** 2).sum(-1, keepdims=True) + 1e-10)

    @staticmethod
    def _stats(distance, count):
        selected = distance[:, :, : int(count)]
        mean = selected.mean(dim=2)
        variance = ((selected - mean.unsqueeze(2)) ** 2).mean(dim=2)
        std = jt.sqrt(variance + 1e-10)
        maximum = selected.max(dim=2)
        return jt.concat([mean, std, maximum], dim=-1)

    def _geometry(self, query, noisy, base_velocity, indices):
        query_relative = batched_index(query, indices) - query.unsqueeze(2)
        noisy_relative = batched_index(noisy, indices) - noisy.unsqueeze(2)
        velocity_relative = (
            batched_index(base_velocity, indices) - base_velocity.unsqueeze(2)
        )
        query_distance = self._distance(query_relative)
        noisy_distance = self._distance(noisy_relative)
        velocity_distance = self._distance(velocity_relative)
        query_scale = query_distance.mean(dim=2, keepdims=True)
        noisy_scale = noisy_distance.mean(dim=2, keepdims=True)
        cosine = (query_relative * noisy_relative).sum(-1, keepdims=True) / (
            query_distance * noisy_distance + 1e-8
        )
        signed_strain = (query_distance - noisy_distance) / (noisy_scale + 1e-8)
        geometry = jt.concat(
            [
                query_distance / (query_scale + 1e-8),
                noisy_distance / (noisy_scale + 1e-8),
                signed_strain,
                jt.abs(signed_strain),
                cosine,
                velocity_distance / (noisy_scale + 1e-8),
            ],
            dim=-1,
        )
        return geometry, query_relative, noisy_relative, velocity_relative, query_distance

    def _initial_features(
        self, noisy, anchor, stage_displacements, indices, geometry, query_distance
    ):
        noisy_relative = batched_index(noisy, indices) - noisy.unsqueeze(2)
        noisy_distance = self._distance(noisy_relative)
        stage_magnitude = jt.sqrt(
            (stage_displacements ** 2).sum(-1) + 1e-10
        )
        base_velocity = anchor - noisy
        base_magnitude = self._distance(base_velocity.unsqueeze(2)).reshape(
            noisy.shape[0], noisy.shape[1], 1
        )
        signed_strain = geometry[:, :, :, 2:3]
        strain_mean = signed_strain.mean(dim=2)
        strain_std = jt.sqrt(
            ((signed_strain - strain_mean.unsqueeze(2)) ** 2).mean(dim=2) + 1e-10
        )
        return jt.concat(
            [
                stage_magnitude,
                base_magnitude,
                self._stats(query_distance, 8),
                self._stats(query_distance, self.max_k),
                self._stats(noisy_distance, 8),
                self._stats(noisy_distance, self.max_k),
                strain_mean,
                strain_std,
            ],
            dim=-1,
        )

    def _vector_field(
        self,
        features,
        indices,
        geometry,
        query_relative,
        noisy_relative,
        velocity_relative,
        base_velocity,
        hidden_layer,
        output_layer,
    ):
        batch, num_points, channels = features.shape
        k = indices.shape[2]
        neighbors = batched_index(features, indices)
        centers = features.unsqueeze(2).broadcast((batch, num_points, k, channels))
        edge = jt.concat([centers, neighbors - centers, geometry], dim=-1)
        hidden = hidden_layer(edge.reshape(-1, edge.shape[-1]))
        output = output_layer(hidden).reshape(batch, num_points, k, 5)
        coefficients = jt.tanh(output[:, :, :, :4])
        importance = nn.softmax(output[:, :, :, 4], dim=2).unsqueeze(-1)
        center_velocity = base_velocity.unsqueeze(2).broadcast(
            (batch, num_points, k, 3)
        )
        bases = jt.stack(
            [query_relative, noisy_relative, velocity_relative, center_velocity],
            dim=3,
        )
        candidates = (coefficients.unsqueeze(-1) * bases).sum(dim=3)
        field = (importance * candidates).sum(dim=2)
        return field, coefficients, importance

    def execute(self, noisy, anchor, displacements):
        batch, num_points, _ = noisy.shape
        base_velocity = anchor - noisy
        stage_displacements = jt.stack(displacements, dim=2)
        indices = knn_indices(anchor, self.max_k)
        geometry, anchor_rel, noisy_rel, velocity_rel, anchor_dist = self._geometry(
            anchor, noisy, base_velocity, indices
        )
        initial = self._initial_features(
            noisy, anchor, stage_displacements, indices, geometry, anchor_dist
        )
        features = self.input_proj(initial.reshape(-1, 19)).reshape(
            batch, num_points, 64
        )
        features = self.observe1(
            features, indices[:, :, :16], geometry[:, :, :16]
        )
        features = self.observe2(
            features, indices[:, :, :32], geometry[:, :, :32]
        )
        features = self.observe3(features, indices, geometry)
        context = self.token_context(features)
        stage1_features = self.stage1_fuse(
            jt.concat([features, context], dim=-1).reshape(
                -1, 2 * self.channels
            )
        ).reshape(batch, num_points, self.channels)
        stage1, coeff1, importance1 = self._vector_field(
            stage1_features,
            indices,
            geometry,
            anchor_rel,
            noisy_rel,
            velocity_rel,
            base_velocity,
            self.stage1_edge,
            self.stage1_coeff,
        )

        pseudo_query = anchor + stage1
        pseudo_geometry, pseudo_rel, noisy_rel2, velocity_rel2, _ = self._geometry(
            pseudo_query, noisy, base_velocity, indices
        )
        refined = self.refine(stage1_features, indices, pseudo_geometry)
        stage2_features = self.stage2_fuse(
            jt.concat([refined, context], dim=-1).reshape(
                -1, 2 * self.channels
            )
        ).reshape(batch, num_points, self.channels)
        stage2, coeff2, importance2 = self._vector_field(
            stage2_features,
            indices,
            pseudo_geometry,
            pseudo_rel,
            noisy_rel2,
            velocity_rel2,
            base_velocity,
            self.stage2_edge,
            self.stage2_coeff,
        )
        trajectory_weights = 0.25 * jt.tanh(
            self.trajectory(stage2_features)
        )
        trajectory = (
            trajectory_weights.unsqueeze(-1) * stage_displacements
        ).sum(dim=2)
        reliability = jt.sigmoid(
            self.gate2(
                self.gate1(
                    jt.concat([stage2_features, context], dim=-1).reshape(
                        -1, 2 * self.channels
                    )
                )
            )
        ).reshape(batch, num_points, 1)
        raw_correction = reliability * (stage1 + stage2 + trajectory)

        anchor_radius = anchor_dist.mean(dim=2)
        base_magnitude = jt.sqrt((base_velocity ** 2).sum(-1, keepdims=True) + 1e-10)
        cap = 0.65 * anchor_radius + 0.35 * base_magnitude + 1e-5
        correction_norm = jt.sqrt(
            (raw_correction ** 2).sum(-1, keepdims=True) + 1e-10
        )
        correction = raw_correction / (1.0 + correction_norm / cap)
        return correction, {
            "reliability": reliability,
            "stage1": stage1,
            "stage2": stage2,
            "trajectory": trajectory,
            "trajectory_weights": trajectory_weights,
            "coefficients_stage1": coeff1,
            "coefficients_stage2": coeff2,
            "importance_stage1": importance1,
            "importance_stage2": importance2,
            "cap": cap,
        }


class V12PseudoQueryCorrector(nn.Module):
    def __init__(self, baseline, max_k=48, channels=128):
        super().__init__()
        self.baseline = baseline
        for parameter in self.baseline.parameters():
            parameter.stop_grad()
        self.corrector = PseudoQueryCorrector(max_k=max_k, channels=channels)

    def trainable_parameters(self):
        return list(self.corrector.parameters())

    def baseline_outputs(self, noisy):
        self.baseline.eval()
        with jt.no_grad():
            anchor, displacements, intermediates = self.baseline(noisy)
        return anchor, displacements, intermediates

    def execute(self, noisy, correction_scale=1.0, baseline_outputs=None):
        if baseline_outputs is None:
            baseline_outputs = self.baseline_outputs(noisy)
        anchor, displacements, intermediates = baseline_outputs
        correction, auxiliary = self.corrector(noisy, anchor, displacements)
        prediction = anchor + float(correction_scale) * correction
        return prediction, anchor, correction, auxiliary, intermediates


def soft_stitch(predictions, point_indices, normalized_distances, original, beta):
    flat_indices = point_indices.reshape(-1)
    weights = np.exp(-float(beta) * normalized_distances).astype(np.float32).reshape(-1)
    flat_predictions = predictions.reshape(-1, 3)
    weight_sum = np.zeros(original.shape[0], dtype=np.float32)
    output_sum = np.zeros_like(original, dtype=np.float32)
    np.add.at(weight_sum, flat_indices, weights)
    for axis in range(3):
        np.add.at(output_sum[:, axis], flat_indices, weights * flat_predictions[:, axis])
    valid = weight_sum > 1e-8
    output = original.copy()
    output[valid] = output_sum[valid] / weight_sum[valid, None]
    return output


def patch_based_pseudo_query_corrector(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    beta=12.0,
    patch_batch=16,
    correction_scale=1.0,
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
            batch = jt.array(centered[start : start + int(patch_batch)].astype(np.float32))
            prediction, _, _, _, _ = model(
                batch, correction_scale=float(correction_scale)
            )
            outputs.append(prediction.numpy())
    predictions = np.concatenate(outputs, axis=0) + seeds[:, None, :]
    return soft_stitch(
        predictions, point_indices, normalized_distances, original, float(beta)
    )


def build_v12_from_args(model_args):
    return IterativePFN(
        num_modules=int(model_args["num_modules"]),
        frame_knn=int(model_args["frame_knn"]),
        embedding_dim=int(model_args["embedding_dim"]),
        output_scale=float(model_args["output_scale"]),
        fine_knn=int(model_args.get("fine_knn", 16)),
        coarse_knn=int(model_args.get("coarse_knn", 64)),
    )
