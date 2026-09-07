import math

import jittor as jt
import numpy as np
from jittor import nn

from jittor_port_models import farthest_point_sampling_np, knn_points_np


def knn_indices(features, k):
    distances = ((features.unsqueeze(2) - features.unsqueeze(1)) ** 2).sum(-1)
    _, indices = jt.topk(distances, k=k + 1, dim=-1, largest=False)
    return indices[:, :, 1:]


def batched_index(points, indices):
    batch, _, channels = points.shape
    _, centers, neighbors = indices.shape
    batch_indices = jt.arange(batch).reshape(batch, 1, 1).broadcast((batch, centers, neighbors))
    return points[batch_indices, indices].reshape(batch, centers, neighbors, channels)


class LinearBNReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def execute(self, features):
        return self.relu(self.bn(self.linear(features)))


class DynamicEdgeConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.message_1 = LinearBNReLU(2 * input_dim, output_dim)
        self.message_2 = LinearBNReLU(output_dim, output_dim)
        self.skip = LinearBNReLU(input_dim, output_dim)
        context_dim = min(64, max(16, output_dim // 2))
        self.context_1 = LinearBNReLU(5 * input_dim, context_dim)
        self.context_2 = nn.Linear(context_dim, output_dim)
        self.context_2.weight.assign(jt.zeros_like(self.context_2.weight))
        self.context_2.bias.assign(jt.zeros_like(self.context_2.bias))

    def execute(self, features, indices, fine_indices, coarse_indices):
        batch, num_points, channels = features.shape
        neighbors = batched_index(features, indices)
        k = neighbors.shape[2]
        centers = features.unsqueeze(2).broadcast((batch, num_points, k, channels))
        edges = jt.concat([centers, neighbors - centers], dim=-1)
        messages = self.message_2(self.message_1(edges.reshape(-1, 2 * channels)))
        messages = messages.reshape(batch, num_points, k, -1).max(dim=2)
        skip = self.skip(features.reshape(-1, channels)).reshape(batch, num_points, -1)

        fine_neighbors = batched_index(features, fine_indices)
        coarse_neighbors = batched_index(features, coarse_indices)
        fine_delta = fine_neighbors - features.unsqueeze(2)
        coarse_delta = coarse_neighbors - features.unsqueeze(2)
        context = jt.concat(
            [
                features,
                fine_delta.mean(dim=2),
                jt.abs(fine_delta).max(dim=2),
                coarse_delta.mean(dim=2),
                jt.abs(coarse_delta).max(dim=2),
            ],
            dim=-1,
        )
        context = self.context_1(context.reshape(-1, 5 * channels))
        context = self.context_2(context).reshape(batch, num_points, self.output_dim)
        return messages + skip + context


class IterationModule(nn.Module):
    def __init__(
        self,
        k=32,
        embedding_dim=512,
        output_scale=0.10,
        fine_knn=16,
        coarse_knn=64,
    ):
        super().__init__()
        self.k = int(k)
        self.fine_knn = int(fine_knn)
        self.coarse_knn = int(coarse_knn)
        if not 1 <= self.fine_knn <= self.k <= self.coarse_knn:
            raise ValueError("expected fine_knn <= k <= coarse_knn")
        self.embedding_dim = int(embedding_dim)
        self.output_scale = float(output_scale)
        self.conv1 = DynamicEdgeConv(3, 16)
        self.conv2 = DynamicEdgeConv(16, 48)
        self.conv3 = DynamicEdgeConv(48, 144)
        self.conv4 = DynamicEdgeConv(16 + 48 + 144, self.embedding_dim)
        self.linear1 = nn.Linear(self.embedding_dim, 256, bias=False)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def execute(self, points):
        geometry_indices = knn_indices(points, self.coarse_knn)
        fine_indices = geometry_indices[:, :, : self.fine_knn]
        indices = geometry_indices[:, :, : self.k]
        features_1 = self.conv1(points, indices, fine_indices, geometry_indices)
        indices = knn_indices(features_1, self.k)
        features_2 = self.conv2(features_1, indices, fine_indices, geometry_indices)
        indices = knn_indices(features_2, self.k)
        features_3 = self.conv3(features_2, indices, fine_indices, geometry_indices)
        combined = jt.concat([features_1, features_2, features_3], dim=-1)
        indices = knn_indices(features_3, self.k)
        features = self.conv4(combined, indices, fine_indices, geometry_indices)
        displacement = self.relu(self.linear1(features))
        displacement = self.relu(self.linear2(displacement))
        return self.output_scale * jt.tanh(self.linear3(displacement))


class IterativePFN(nn.Module):
    def __init__(
        self,
        num_modules=4,
        frame_knn=32,
        embedding_dim=512,
        output_scale=0.10,
        fine_knn=16,
        coarse_knn=64,
    ):
        super().__init__()
        self.num_modules = int(num_modules)
        self.frame_knn = int(frame_knn)
        self.embedding_dim = int(embedding_dim)
        self.output_scale = float(output_scale)
        self.fine_knn = int(fine_knn)
        self.coarse_knn = int(coarse_knn)
        self.modules_list = nn.ModuleList(
            [
                IterationModule(
                    k=self.frame_knn,
                    embedding_dim=self.embedding_dim,
                    output_scale=self.output_scale,
                    fine_knn=self.fine_knn,
                    coarse_knn=self.coarse_knn,
                )
                for _ in range(self.num_modules)
            ]
        )

    def execute(self, points):
        current = points
        displacements = []
        intermediates = []
        for module in self.modules_list:
            displacement = module(current)
            current = current + displacement
            displacements.append(displacement)
            intermediates.append(current)
        return current, displacements, intermediates

    def adaptive_nearest_loss(self, noisy, adaptive_targets):
        batch, num_points, _ = noisy.shape
        seed_distance = (noisy ** 2).sum(-1)
        max_distance = seed_distance[:, -1:] / 9.0 + 1e-10
        weights = jt.exp(-seed_distance / max_distance)
        weights = weights / (weights.sum(dim=1, keepdims=True) + 1e-10)
        current = noisy
        losses = []
        for module, target in zip(self.modules_list, adaptive_targets):
            displacement = module(current)
            distances = ((current.unsqueeze(2) - target.unsqueeze(1)) ** 2).sum(-1)
            _, indices = jt.topk(distances, k=1, dim=-1, largest=False)
            nearest = batched_index(target, indices).reshape(batch, num_points, 3)
            target_displacement = nearest - current
            point_loss = ((displacement - target_displacement) ** 2).sum(-1)
            losses.append((weights * point_loss).sum(dim=1).mean())
            current = current + displacement
        return sum(losses), losses


def soft_stitch(predictions, point_indices, normalized_distances, original, beta):
    flat_indices = point_indices.reshape(-1)
    weights = np.exp(-beta * normalized_distances).astype(np.float32).reshape(-1)
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


def patch_based_iterativepfn(
    model,
    points,
    patch_size=1000,
    seed_k=6,
    seed_k_alpha=1,
    beta=8.0,
    output_scales=(1.0,),
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
    patch_step = max(1, int(math.ceil(num_points / (seed_k_alpha * patch_size))))
    outputs_by_scale = {float(scale): [] for scale in output_scales}
    with jt.no_grad():
        for start in range(0, num_patches, patch_step):
            batch = jt.array(centered[start : start + patch_step].astype(np.float32))
            denoised, _, _ = model(batch)
            denoised_np = denoised.numpy()
            input_np = centered[start : start + patch_step]
            for scale in outputs_by_scale:
                outputs_by_scale[scale].append(input_np + scale * (denoised_np - input_np))
    results = {}
    for scale, patch_outputs in outputs_by_scale.items():
        patch_outputs = np.concatenate(patch_outputs, axis=0) + seeds[:, None, :]
        results[scale] = soft_stitch(
            patch_outputs, point_indices, normalized_distances, original, beta
        )
    return results


def patch_based_iterativepfn_selected(
    model,
    points,
    module_count,
    patch_size=1000,
    seed_k=6,
    seed_k_alpha=1,
    beta=12.0,
    output_scale=1.0,
):
    original = np.asarray(points, dtype=np.float32)
    num_points = original.shape[0]
    module_count = int(module_count)
    if module_count < 1 or module_count > model.num_modules:
        raise ValueError(
            f"module_count must be in [1, {model.num_modules}], got {module_count}"
        )
    num_patches = max(1, int(seed_k * num_points / patch_size))
    seed_indices = farthest_point_sampling_np(original, num_patches)
    seeds = original[seed_indices]
    patch_distances, point_indices, patches = knn_points_np(
        seeds, original, min(patch_size, num_points)
    )
    centered = patches - seeds[:, None, :]
    normalized_distances = patch_distances / (patch_distances[:, -1:] + 1e-8)
    patch_step = max(1, int(math.ceil(num_points / (seed_k_alpha * patch_size))))
    selected_outputs = []
    with jt.no_grad():
        for start in range(0, num_patches, patch_step):
            input_np = centered[start : start + patch_step]
            batch = jt.array(input_np.astype(np.float32))
            _, _, intermediates = model(batch)
            selected = intermediates[module_count - 1].numpy()
            selected_outputs.append(
                input_np + float(output_scale) * (selected - input_np)
            )
    patch_outputs = np.concatenate(selected_outputs, axis=0) + seeds[:, None, :]
    return soft_stitch(
        patch_outputs,
        point_indices,
        normalized_distances,
        original,
        float(beta),
    )
