import json
import math
from typing import List, Optional

import jittor as jt
import numpy as np
from jittor import nn


def get_knn_idx(x, y, k, offset=0):
    k_total = k + offset
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    _, idx = jt.topk(dist, k=k_total, dim=-1, largest=False)
    return idx[:, :, offset:]


def batched_index_points(x, idx):
    b, n, c = x.shape
    _, m, k = idx.shape
    batch_idx = jt.arange(b).reshape(b, 1, 1).broadcast((b, m, k))
    return x[batch_idx, idx]


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation: Optional[str] = "ReLU"):
        super().__init__()
        if activation == "ReLU":
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
            )
            self.lin = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
            )
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise ValueError("Unsupported activation")

    def execute(self, x, knn_idx):
        b, n, c = x.shape
        neighbors = batched_index_points(x, knn_idx)
        k = neighbors.shape[2]
        centers = x.unsqueeze(2).broadcast((b, n, k, c))
        edge_feat = jt.concat([centers, neighbors - centers], dim=-1)
        msg = self.mlp(edge_feat.reshape(-1, edge_feat.shape[-1])).reshape(b, n, k, -1)
        out = msg.max(dim=2)
        base = self.lin(x.reshape(-1, c)).reshape(b, n, -1)
        return out + base


class DynamicEdgeConv(EdgeConv):
    pass


class FeatureExtraction(nn.Module):
    def __init__(self, k=32, input_dim=3, embedding_dim=256, distance_estimation=False):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.distance_estimation = distance_estimation
        self.conv1 = DynamicEdgeConv(input_dim, embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(embedding_dim // 8 + embedding_dim // 4, embedding_dim, activation=None)

    def normalize_patch(self, pcl):
        scale = jt.sqrt((pcl ** 2).sum(-1, keepdims=True))
        scale = scale.max(dim=-2, keepdims=True)
        return pcl / (scale + 1e-8)  # type: ignore

    def get_knn_graph(self, x):
        return get_knn_idx(x, x, self.k + 1)[:, :, 1:]

    def execute(self, x):
        b, n, _ = x.shape
        if self.distance_estimation:
            x = self.normalize_patch(x)

        knn_idx = self.get_knn_graph(x)
        x1 = self.conv1(x, knn_idx)
        knn_idx = self.get_knn_graph(x1)
        x2 = self.conv2(x1, knn_idx)
        knn_idx = self.get_knn_graph(x2)
        x_cat = jt.concat([x1, x2], dim=-1)
        x3 = self.conv3(x_cat, knn_idx)
        return x3


class Decoder(nn.Module):
    def __init__(self, z_dim, dim, out_dim, hidden_size):
        super().__init__()
        self.out_dim = out_dim
        self.lin_1 = nn.Linear(z_dim, z_dim)
        self.bn_1 = nn.BatchNorm1d(z_dim)
        self.lin_2 = nn.Linear(z_dim, hidden_size)
        self.bn_2 = nn.BatchNorm1d(hidden_size)
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        self.lin_3.weight.assign(jt.zeros_like(self.lin_3.weight))
        self.lin_3.bias.assign(jt.zeros_like(self.lin_3.bias))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def execute(self, c, b=None, n=None):
        net = self.dropout(self.relu(self.bn_1(self.lin_1(c))))
        net = self.dropout(self.relu(self.bn_2(self.lin_2(net))))
        if self.out_dim == 1:
            assert b is not None and n is not None
            net = net.reshape(b, n, -1)
            net = jt.max(net, dim=1, keepdims=True)
            net = jt.sigmoid(self.lin_3(net))
        else:
            net = self.lin_3(net)
        return net


class PointwiseScalarDecoder(nn.Module):
    def __init__(self, z_dim, hidden_size):
        super().__init__()
        self.lin_1 = nn.Linear(z_dim, z_dim)
        self.bn_1 = nn.BatchNorm1d(z_dim)
        self.lin_2 = nn.Linear(z_dim, hidden_size)
        self.bn_2 = nn.BatchNorm1d(hidden_size)
        self.lin_3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def execute(self, feat):
        b, n, c = feat.shape
        net = feat.reshape(-1, c)
        net = self.dropout(self.relu(self.bn_1(self.lin_1(net))))
        net = self.dropout(self.relu(self.bn_2(self.lin_2(net))))
        return jt.sigmoid(self.lin_3(net)).reshape(b, n, 1)


class PointwiseVectorDecoder(nn.Module):
    def __init__(self, z_dim, hidden_size, out_dim=3):
        super().__init__()
        self.out_dim = out_dim
        self.lin_1 = nn.Linear(z_dim, z_dim)
        self.bn_1 = nn.BatchNorm1d(z_dim)
        self.lin_2 = nn.Linear(z_dim, hidden_size)
        self.bn_2 = nn.BatchNorm1d(hidden_size)
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def execute(self, feat):
        b, n, c = feat.shape
        net = feat.reshape(-1, c)
        net = self.dropout(self.relu(self.bn_1(self.lin_1(net))))
        net = self.dropout(self.relu(self.bn_2(self.lin_2(net))))
        return self.lin_3(net).reshape(b, n, self.out_dim)


class VelocityModule(nn.Module):
    def __init__(self, frame_knn=32, num_train_points=128, dsm_sigma=0.01,
                 feat_embedding_dim=256, decoder_hidden_dim=64):
        super().__init__()
        self.frame_knn = frame_knn
        self.num_train_points = num_train_points
        self.dsm_sigma = dsm_sigma
        self.encoder = FeatureExtraction(k=frame_knn, input_dim=3, embedding_dim=feat_embedding_dim)
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=decoder_hidden_dim,
        )

    def get_supervised_loss(self, pcl_noisy_l2, pcl_noisy_mix, pcl_clean):
        b, n, d = pcl_noisy_mix.shape
        pnt_idx = np.random.permutation(n)[:self.num_train_points]
        pnt_idx = jt.array(pnt_idx).int32()
        feat = self.encoder(pcl_noisy_mix)
        feat = feat[:, pnt_idx, :]
        pcl_noisy_l2 = pcl_noisy_l2[:, pnt_idx, :]
        pcl_clean = pcl_clean[:, pnt_idx, :]
        target = pcl_clean - pcl_noisy_l2
        pred = self.decoder(feat.reshape(-1, feat.shape[-1])).reshape(b, len(pnt_idx), d)
        return (((pred - target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps=4):
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(num_steps):
                feat = self.encoder(pcl_next)
                pred = self.decoder(feat.reshape(-1, feat.shape[-1])).reshape(pcl_next.shape)
                pcl_next = pcl_next + (1.0 / num_steps) * pred
        return pcl_next, None


class CoupledVMArch(nn.Module):
    def __init__(self, velocity_nets: Optional[List[VelocityModule]] = None, num_modules=2,
                 frame_knn=32, num_train_points=128, dsm_sigma=0.01,
                 feat_embedding_dim=256, decoder_hidden_dim=64):
        super().__init__()
        self.frame_knn = frame_knn
        self.num_train_points = num_train_points
        self.dsm_sigma = dsm_sigma
        self.tot_its = 3
        self.num_modules = num_modules
        if velocity_nets is None:
            velocity_nets = [
                VelocityModule(
                    frame_knn=frame_knn,
                    num_train_points=num_train_points,
                    dsm_sigma=dsm_sigma,
                    feat_embedding_dim=feat_embedding_dim,
                    decoder_hidden_dim=decoder_hidden_dim,
                )
                for _ in range(num_modules)
            ]
        self.velocity_nets = nn.ModuleList(velocity_nets)
        self.num_modules = len(velocity_nets)

    def get_supervised_loss(self, pcl_clean, pcl_noisy_l2, pcl_seeds_t, original_time_step):
        b, n, d = pcl_noisy_l2.shape
        grad_target = pcl_clean - pcl_noisy_l2
        total_dir_loss = 0.0
        total_consistency_loss = 0.0

        curr_step = (original_time_step * self.num_modules) / self.num_modules
        curr_step = curr_step.unsqueeze(1).unsqueeze(2)
        pcl_noisy = curr_step * pcl_clean + (1 - curr_step) * pcl_noisy_l2
        pcl_noisy = pcl_noisy - pcl_seeds_t

        for mod in range(self.num_modules):
            feat = self.velocity_nets[mod].encoder(pcl_noisy)
            pred_dir = self.velocity_nets[mod].decoder(feat.reshape(-1, feat.shape[-1])).reshape(b, n, d)
            total_dir_loss += (((pred_dir - grad_target) ** 2)).sum(dim=-1).mean()
            pcl_noisy = pcl_noisy + ((1.0 - original_time_step.unsqueeze(1).unsqueeze(2)) / self.num_modules) * pred_dir

            if mod < self.num_modules - 1:
                curr_step_plus = (original_time_step * (self.num_modules - (mod + 1)) + (mod + 1)) / self.num_modules
                curr_step_plus = curr_step_plus.unsqueeze(1).unsqueeze(2)
                pcl_interp = curr_step_plus * pcl_clean + (1 - curr_step_plus) * pcl_noisy_l2
                pcl_interp = pcl_interp - pcl_seeds_t
                total_consistency_loss += (((pcl_interp - pcl_noisy) ** 2)).sum(dim=-1).mean()

        return (total_dir_loss + 10.0 * total_consistency_loss) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy):
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(self.tot_its):
                for mod in range(self.num_modules):
                    feat = self.velocity_nets[mod].encoder(pcl_next)
                    pred_dir = self.velocity_nets[mod].decoder(feat.reshape(-1, feat.shape[-1])).reshape(pcl_next.shape)
                    pcl_next = pcl_next + (1.0 / self.tot_its) * (1.0 / self.num_modules) * pred_dir
        return pcl_next, None


class StraightPCF(nn.Module):
    def __init__(self, velocity_nets: Optional[List[VelocityModule]] = None, num_modules=2,
                 frame_knn=32, tot_its=2, num_train_points=128, dsm_sigma=0.01,
                 feat_embedding_dim=128, decoder_hidden_dim=64, distance_estimation=True,
                 adaptive_distance=False, local_distance_weight=0.5,
                 local_distance_loss_weight=0.2, residual_refine=False,
                 residual_refine_weight=0.1, residual_reg_weight=0.01):
        super().__init__()
        self.frame_knn = frame_knn
        self.tot_its = tot_its
        self.num_train_points = num_train_points
        self.dsm_sigma = dsm_sigma
        self.distance_estimation = distance_estimation
        self.adaptive_distance = adaptive_distance
        self.local_distance_weight = local_distance_weight
        self.local_distance_loss_weight = local_distance_loss_weight
        self.residual_refine = residual_refine
        self.residual_refine_weight = residual_refine_weight
        self.residual_reg_weight = residual_reg_weight
        if velocity_nets is None:
            velocity_nets = [
                VelocityModule(
                    frame_knn=frame_knn,
                    num_train_points=num_train_points,
                    dsm_sigma=dsm_sigma,
                    feat_embedding_dim=feat_embedding_dim,
                    decoder_hidden_dim=decoder_hidden_dim,
                )
                for _ in range(num_modules)
            ]
        self.velocity_nets = nn.ModuleList(velocity_nets)
        self.num_modules = len(velocity_nets)
        self.encoder = FeatureExtraction(
            k=frame_knn,
            input_dim=3,
            embedding_dim=feat_embedding_dim,
            distance_estimation=distance_estimation,
        )
        self.decoder = Decoder(
            z_dim=feat_embedding_dim,
            dim=3,
            out_dim=1,
            hidden_size=decoder_hidden_dim,
        )
        if adaptive_distance:
            self.local_decoder = PointwiseScalarDecoder(
                z_dim=feat_embedding_dim,
                hidden_size=decoder_hidden_dim,
            )
        else:
            self.local_decoder = None
        if residual_refine:
            self.residual_decoder = PointwiseVectorDecoder(
                z_dim=feat_embedding_dim,
                hidden_size=decoder_hidden_dim,
                out_dim=3,
            )
        else:
            self.residual_decoder = None

    def distance_scale(self, feat_d, b, n):
        pred_global = self.decoder(feat_d.reshape(-1, feat_d.shape[-1]), b=b, n=n).reshape(b, 1, 1)
        if not self.adaptive_distance or self.local_decoder is None:
            return pred_global, pred_global
        pred_local = self.local_decoder(feat_d)
        mixed = (1.0 - self.local_distance_weight) * pred_global + self.local_distance_weight * pred_local
        return mixed, pred_global

    def get_supervised_loss(self, pcl_clean, pcl_noisy_l2, pcl_seeds_t, original_time_step):
        b, n, d = pcl_noisy_l2.shape
        curr_step = original_time_step.unsqueeze(1).unsqueeze(2)
        pcl_noisy = curr_step * pcl_clean + (1 - curr_step) * pcl_noisy_l2
        point_ratio = jt.sqrt(((pcl_clean - pcl_noisy) ** 2).sum(dim=-1, keepdims=True)) / (
            jt.sqrt(((pcl_clean - pcl_noisy_l2) ** 2).sum(dim=-1, keepdims=True)) + 1e-8
        )
        ratio = point_ratio[:, 0, 0]

        pcl_clean = pcl_clean - pcl_seeds_t
        pcl_noisy = pcl_noisy - pcl_seeds_t
        feat_d = self.encoder(pcl_noisy)
        step_scale, pred_global = self.distance_scale(feat_d, b, n)
        loss = ((pred_global.reshape(b) - ratio) ** 2).mean()
        if self.adaptive_distance and self.local_decoder is not None:
            local_loss = ((step_scale - point_ratio) ** 2).mean()
            loss = loss + self.local_distance_loss_weight * local_loss

        for mod in range(self.num_modules):
            feat = self.velocity_nets[mod].encoder(pcl_noisy)
            pred_dir = self.velocity_nets[mod].decoder(feat.reshape(-1, feat.shape[-1])).reshape(b, n, d)
            pcl_noisy = pcl_noisy + (1.0 / self.num_modules) * step_scale * pred_dir

        residual_reg = 0.0
        if self.residual_refine and self.residual_decoder is not None:
            residual = self.residual_decoder(feat_d)
            pcl_noisy = pcl_noisy + self.residual_refine_weight * residual
            residual_reg = self.residual_reg_weight * (residual ** 2).sum(dim=-1).mean()

        finetune_loss = 2e2 * ((pcl_clean - pcl_noisy) ** 2).sum(dim=-1).mean()
        return (loss + finetune_loss + residual_reg) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy):
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            feat_d = self.encoder(pcl_next)
            step_scale, _ = self.distance_scale(feat_d, pcl_next.shape[0], pcl_next.shape[1])
            for _ in range(self.tot_its):
                for mod in range(self.num_modules):
                    feat = self.velocity_nets[mod].encoder(pcl_next)
                    pred_dir = self.velocity_nets[mod].decoder(feat.reshape(-1, feat.shape[-1])).reshape(pcl_next.shape)
                    pcl_next = pcl_next + (1.0 / self.tot_its) * (1.0 / self.num_modules) * step_scale * pred_dir
            if self.residual_refine and self.residual_decoder is not None:
                residual = self.residual_decoder(feat_d)
                pcl_next = pcl_next + self.residual_refine_weight * residual
        return pcl_next, None


def farthest_point_sampling_np(points, num_samples):
    n = points.shape[0]
    if num_samples >= n:
        return np.arange(n, dtype=np.int64)
    selected = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(n, np.inf, dtype=np.float64)
    farthest = 0
    for i in range(num_samples):
        selected[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return selected


def knn_points_np(seeds, points, k):
    diff = seeds[:, None, :] - points[None, :, :]
    dist = np.sum(diff * diff, axis=-1)
    idx = np.argpartition(dist, kth=min(k - 1, points.shape[0] - 1), axis=1)[:, :k]
    row = np.arange(seeds.shape[0])[:, None]
    local_dist = dist[row, idx]
    order = np.argsort(local_dist, axis=1)
    idx = idx[row, order]
    local_dist = local_dist[row, order]
    nn = points[idx]
    return local_dist, idx, nn


def patch_based_denoise(model, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1):
    if isinstance(pcl_noisy, jt.Var):
        pcl_input = pcl_noisy.numpy().astype(np.float32)
    else:
        pcl_input = np.asarray(pcl_noisy, dtype=np.float32)
    n, _ = pcl_input.shape
    num_patches = max(1, int(seed_k * n / patch_size))
    seed_idx = farthest_point_sampling_np(pcl_input, num_patches)
    seed_pnts = pcl_input[seed_idx]
    patch_dists, point_idxs, patches = knn_points_np(seed_pnts, pcl_input, min(patch_size, n))

    patches_centered = patches - seed_pnts[:, None, :]
    patch_dists = patch_dists / (patch_dists[:, -1:] + 1e-8)
    all_dists = np.full((num_patches, n), np.inf, dtype=np.float32)
    for i in range(num_patches):
        all_dists[i, point_idxs[i]] = patch_dists[i]
    best_patch_idx = np.exp(-all_dists).argmax(axis=0)

    patch_step = max(1, int(math.ceil(n / (seed_k_alpha * patch_size))))
    patches_denoised = []
    i = 0
    while i < num_patches:
        curr = jt.array(patches_centered[i:i + patch_step].astype(np.float32))
        out, _ = model.denoise_langevin_dynamics(curr)
        patches_denoised.append(out.numpy())
        i += patch_step
    patches_denoised = np.concatenate(patches_denoised, axis=0) + seed_pnts[:, None, :]

    pcl_out = []
    for pidx in range(n):
        patch_id = int(best_patch_idx[pidx])
        selected = patches_denoised[patch_id][point_idxs[patch_id] == pidx]
        if selected.shape[0] == 0:
            pcl_out.append(pcl_input[pidx:pidx + 1])
        else:
            pcl_out.append(selected[:1])
    return jt.array(np.concatenate(pcl_out, axis=0).astype(np.float32))


def save_args_json(path, args_dict):
    with open(path, "w") as f:
        json.dump(args_dict, f, indent=2)
