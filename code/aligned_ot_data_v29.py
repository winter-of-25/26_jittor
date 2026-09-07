import os

import numpy as np
from jittor.dataset import Dataset
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance

from robust_surface_data import (
    load_relpaths,
    normalize_surface,
    random_rotation,
    sample_surface,
)


VAL_FAMILIES = (
    "laplace_low",
    "laplace_mid",
    "laplace_high",
    "gaussian",
    "anisotropic",
    "spatial",
    "student",
    "compound",
    "uniform_ball",
)

TRAIN_FAMILIES = (
    "laplace",
    "gaussian",
    "anisotropic",
    "spatial",
    "student",
    "compound",
    "uniform_ball",
)

TRAIN_FAMILY_PROBABILITIES = (
    0.55,
    0.12,
    0.08,
    0.08,
    0.06,
    0.07,
    0.04,
)


def _tangent_noise(rng, normals, scale):
    raw = rng.normal(size=normals.shape).astype(np.float32)
    tangent = raw - (raw * normals).sum(axis=1, keepdims=True) * normals
    tangent /= np.sqrt((tangent ** 2).sum(axis=1, keepdims=True) + 1e-12)
    magnitude = rng.normal(0.0, scale, size=(normals.shape[0], 1)).astype(np.float32)
    return tangent * magnitude


def calibrated_corrupt(clean, normals, rng, family=None):
    if family is None:
        family = str(
            rng.choice(
                TRAIN_FAMILIES,
                p=TRAIN_FAMILY_PROBABILITIES,
            )
        )

    if family == "laplace":
        sigma = float(rng.uniform(0.0075, 0.0140))
        noise = rng.laplace(0.0, sigma, size=clean.shape).astype(np.float32)
    elif family == "laplace_low":
        sigma = float(rng.uniform(0.0080, 0.0092))
        noise = rng.laplace(0.0, sigma, size=clean.shape).astype(np.float32)
    elif family == "laplace_mid":
        sigma = float(rng.uniform(0.0092, 0.0112))
        noise = rng.laplace(0.0, sigma, size=clean.shape).astype(np.float32)
    elif family == "laplace_high":
        sigma = float(rng.uniform(0.0112, 0.0130))
        noise = rng.laplace(0.0, sigma, size=clean.shape).astype(np.float32)
    elif family == "gaussian":
        # Match the RMS displacement of the calibrated Laplace range.
        sigma = float(rng.uniform(0.0095, 0.0180))
        noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    elif family == "anisotropic":
        sigma = float(rng.uniform(0.0075, 0.0140))
        normal_noise = rng.normal(0.0, sigma, size=(clean.shape[0], 1)).astype(np.float32)
        noise = normal_noise * normals + _tangent_noise(rng, normals, 0.55 * sigma)
    elif family == "spatial":
        sigma = float(rng.uniform(0.0075, 0.0140))
        direction = rng.normal(size=(1, 3)).astype(np.float32)
        direction /= np.sqrt((direction ** 2).sum() + 1e-12)
        phase = rng.uniform(-np.pi, np.pi)
        coordinate = clean @ direction.T
        local_scale = 0.55 + 0.90 * (
            0.5 + 0.5 * np.sin(7.0 * coordinate[:, 0] + phase)
        )
        noise = (
            rng.laplace(0.0, sigma, size=clean.shape).astype(np.float32)
            * local_scale[:, None].astype(np.float32)
        )
    elif family == "student":
        sigma = float(rng.uniform(0.0070, 0.0125))
        noise = (
            rng.standard_t(4.0, size=clean.shape).astype(np.float32)
            * (sigma / np.sqrt(2.0))
        )
    elif family == "compound":
        sigma = float(rng.uniform(0.0075, 0.0135))
        dense = rng.laplace(
            0.0, 0.75 * sigma, size=clean.shape
        ).astype(np.float32)
        axial = (
            rng.normal(
                0.0, 0.65 * sigma, size=(clean.shape[0], 1)
            ).astype(np.float32)
            * normals
        )
        noise = dense + axial
        count = max(1, int(round(0.003 * clean.shape[0])))
        outliers = rng.choice(clean.shape[0], size=count, replace=False)
        directions = rng.normal(size=(count, 3)).astype(np.float32)
        directions /= np.sqrt(
            (directions ** 2).sum(axis=1, keepdims=True) + 1e-12
        )
        noise[outliers] += directions * rng.uniform(
            0.018, 0.035, size=(count, 1)
        ).astype(np.float32)
    elif family == "uniform_ball":
        sigma = float(rng.uniform(0.010, 0.019))
        direction = rng.normal(size=clean.shape).astype(np.float32)
        direction /= np.sqrt(
            (direction ** 2).sum(axis=1, keepdims=True) + 1e-12
        )
        radius = sigma * (
            rng.uniform(size=(clean.shape[0], 1)).astype(np.float32)
            ** (1.0 / 3.0)
        )
        noise = direction * radius
    else:
        raise ValueError("unknown family: %s" % family)
    return (clean + noise).astype(np.float32), np.float32(sigma), family


def greedy_balanced_assignment(source, target, k=32):
    """Fast one-to-one approximation of EMD, followed by an exact small residual solve."""

    count = source.shape[0]
    k = min(int(k), count)
    distances, indices = cKDTree(target).query(source, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    source_ids = np.repeat(np.arange(count, dtype=np.int64), k)
    target_ids = np.asarray(indices, dtype=np.int64).reshape(-1)
    edge_distances = np.asarray(distances).reshape(-1)
    order = np.argsort(edge_distances)
    assignment = np.full(count, -1, dtype=np.int64)
    used = np.zeros(count, dtype=bool)
    for edge in order:
        source_id = source_ids[edge]
        target_id = target_ids[edge]
        if assignment[source_id] < 0 and not used[target_id]:
            assignment[source_id] = target_id
            used[target_id] = True

    missing = np.flatnonzero(assignment < 0)
    available = np.flatnonzero(~used)
    if missing.size:
        residual_cost = distance.cdist(
            source[missing], target[available], metric="sqeuclidean"
        )
        rows, columns = linear_sum_assignment(residual_cost)
        assignment[missing[rows]] = available[columns]
    if (assignment < 0).any() or np.unique(assignment).size != count:
        raise RuntimeError("balanced assignment failed")
    return assignment


class AlignedOTPatchDataset(Dataset):
    def __init__(
        self,
        data_root,
        list_path,
        num_points=32768,
        patch_size=1000,
        patch_ratio=1.2,
        augment=True,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        alignment_k=32,
    ):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_root = data_root
        self.relpaths = load_relpaths(list_path)
        self.num_points = int(num_points)
        self.patch_size = int(patch_size)
        self.patch_ratio = float(patch_ratio)
        self.augment = bool(augment)
        self.alignment_k = int(alignment_k)
        self.set_attrs(total_len=len(self.relpaths))

    def mesh_path(self, relpath):
        return os.path.join(
            self.data_root,
            "dataset_train",
            relpath,
            "models",
            "model_normalized.obj",
        )

    def make_item(self, index, rng, family=None):
        dense_count = int(round(self.num_points * self.patch_ratio))
        clean_dense, clean_normals = sample_surface(
            self.mesh_path(self.relpaths[index]), dense_count, rng
        )
        clean_dense = normalize_surface(clean_dense)
        if self.augment:
            rotation = random_rotation(rng)
            clean_dense = clean_dense @ rotation.T
            clean_normals = clean_normals @ rotation.T
            if rng.rand() < 0.5:
                clean_dense *= float(rng.uniform(0.90, 1.10))
        clean_normals /= np.sqrt(
            (clean_normals ** 2).sum(axis=1, keepdims=True) + 1e-12
        )

        clean_source = clean_dense[: self.num_points]
        source_normals = clean_normals[: self.num_points]
        noisy, sigma, family = calibrated_corrupt(
            clean_source, source_normals, rng, family
        )
        seed_index = int(rng.randint(noisy.shape[0]))
        seed = noisy[seed_index : seed_index + 1]
        _, noisy_indices = cKDTree(noisy).query(
            seed[0], k=min(self.patch_size, noisy.shape[0])
        )
        clean_patch_size = int(round(self.patch_size * self.patch_ratio))
        _, clean_indices = cKDTree(clean_dense).query(
            seed[0], k=min(clean_patch_size, clean_dense.shape[0])
        )
        noisy_indices = np.asarray(noisy_indices)
        clean_indices = np.asarray(clean_indices)
        noisy_indices = noisy_indices[rng.permutation(noisy_indices.shape[0])]

        noisy_patch = noisy[noisy_indices]
        clean_pair = clean_source[noisy_indices]
        normal_pair = source_normals[noisy_indices]
        assignment = greedy_balanced_assignment(
            noisy_patch, clean_pair, k=self.alignment_k
        )
        clean_ot = clean_pair[assignment]
        normal_ot = normal_pair[assignment]
        family_id = VAL_FAMILIES.index(family) if family in VAL_FAMILIES else -1
        return {
            "pcl_noisy": (noisy_patch - seed).astype(np.float32),
            "pcl_clean": (clean_dense[clean_indices] - seed).astype(np.float32),
            "pcl_clean_pair": (clean_pair - seed).astype(np.float32),
            "pcl_clean_ot": (clean_ot - seed).astype(np.float32),
            "pcl_clean_normal": normal_pair.astype(np.float32),
            "pcl_ot_normal": normal_ot.astype(np.float32),
            "noise_std": np.asarray([sigma], dtype=np.float32),
            "family_id": np.asarray([family_id], dtype=np.int32),
        }

    def __getitem__(self, index):
        seed = np.random.randint(0, 2 ** 31 - 1)
        return self.make_item(index, np.random.RandomState(seed))


class AlignedOTValidationDataset(AlignedOTPatchDataset):
    def __init__(self, *args, max_shapes=24, val_seed=27027, **kwargs):
        kwargs.update({"augment": False, "shuffle": False, "num_workers": 0})
        super().__init__(*args, **kwargs)
        self.relpaths = self.relpaths[: int(max_shapes)]
        self.val_seed = int(val_seed)
        self.set_attrs(total_len=len(self.relpaths) * len(VAL_FAMILIES))

    def __getitem__(self, index):
        shape_index = int(index) // len(VAL_FAMILIES)
        family_index = int(index) % len(VAL_FAMILIES)
        rng = np.random.RandomState(self.val_seed + int(index) * 7919)
        return self.make_item(shape_index, rng, VAL_FAMILIES[family_index])
