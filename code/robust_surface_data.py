import os

import numpy as np
import trimesh
from jittor.dataset import Dataset
from scipy.spatial import cKDTree


TRAIN_FAMILIES = ("near", "gaussian", "laplace", "anisotropic", "spatial", "outlier", "compound")
VAL_FAMILIES = TRAIN_FAMILIES + ("student", "uniform_ball")


def load_relpaths(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def random_rotation(rng):
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1.0
    return matrix.astype(np.float32)


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def normalize_surface(points):
    center = (points.max(axis=0) + points.min(axis=0)) / 2.0
    points = points - center
    scale = float(np.sqrt((points ** 2).sum(axis=1)).max())
    return (points / max(scale, 1e-12)).astype(np.float32)


def sample_surface(mesh_path, count, rng):
    # trimesh uses NumPy's global RNG, so bracket the call for deterministic validation.
    state = np.random.get_state()
    np.random.seed(int(rng.randint(0, 2 ** 31 - 1)))
    try:
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        points, face_indices = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    points = np.asarray(points, dtype=np.float32).copy()
    normals = np.asarray(mesh.face_normals[np.asarray(face_indices)], dtype=np.float32).copy()
    return points, normals


def _tangent_noise(rng, normals, scale):
    raw = rng.normal(size=normals.shape).astype(np.float32)
    tangent = raw - (raw * normals).sum(axis=1, keepdims=True) * normals
    tangent /= np.sqrt((tangent ** 2).sum(axis=1, keepdims=True) + 1e-12)
    magnitude = rng.normal(0.0, scale, size=(normals.shape[0], 1)).astype(np.float32)
    return tangent * magnitude


def corrupt_points(clean, normals, rng, family=None):
    if family is None:
        family = str(rng.choice(TRAIN_FAMILIES, p=(0.15, 0.20, 0.20, 0.15, 0.15, 0.10, 0.05)))
    outlier_mask = np.zeros((clean.shape[0], 1), dtype=np.float32)

    if family == "near":
        sigma = float(rng.uniform(0.00005, 0.0035))
        noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
        expert = 0
    elif family == "gaussian":
        sigma = log_uniform(rng, 0.003, 0.020)
        noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
        expert = 1
    elif family == "laplace":
        sigma = log_uniform(rng, 0.003, 0.016)
        noise = rng.laplace(0.0, sigma, size=clean.shape).astype(np.float32)
        expert = 1
    elif family == "anisotropic":
        sigma = log_uniform(rng, 0.004, 0.025)
        tangent_sigma = log_uniform(rng, 0.001, min(0.008, sigma))
        axial = rng.normal(0.0, sigma, size=(clean.shape[0], 1)).astype(np.float32) * normals
        noise = axial + _tangent_noise(rng, normals, tangent_sigma)
        expert = 2
    elif family == "spatial":
        sigma = log_uniform(rng, 0.002, 0.022)
        direction = rng.normal(size=(3,)).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-12
        coordinate = clean @ direction
        coordinate = (coordinate - coordinate.mean()) / (coordinate.std() + 1e-8)
        local_scale = sigma * (0.35 + 1.30 / (1.0 + np.exp(-coordinate)))
        noise = rng.normal(size=clean.shape).astype(np.float32) * local_scale[:, None]
        expert = 2
    elif family == "outlier":
        sigma = log_uniform(rng, 0.002, 0.008)
        noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
        count = max(1, int(round(clean.shape[0] * rng.uniform(0.01, 0.05))))
        indices = rng.choice(clean.shape[0], count, replace=False)
        direction = rng.normal(size=(count, 3)).astype(np.float32)
        direction /= np.sqrt((direction ** 2).sum(axis=1, keepdims=True) + 1e-12)
        noise[indices] += direction * rng.uniform(0.02, 0.08, size=(count, 1)).astype(np.float32)
        outlier_mask[indices] = 1.0
        expert = 2
    elif family == "compound":
        sigma = log_uniform(rng, 0.003, 0.016)
        dense = rng.laplace(0.0, sigma, size=clean.shape).astype(np.float32)
        axial = rng.normal(0.0, sigma * 0.7, size=(clean.shape[0], 1)).astype(np.float32) * normals
        noise = 0.65 * dense + 0.35 * axial
        expert = 2
    elif family == "student":
        sigma = log_uniform(rng, 0.003, 0.020)
        noise = (rng.standard_t(3.0, size=clean.shape) * (sigma / np.sqrt(3.0))).astype(np.float32)
        expert = 2
    elif family == "uniform_ball":
        sigma = log_uniform(rng, 0.003, 0.025)
        direction = rng.normal(size=clean.shape).astype(np.float32)
        direction /= np.sqrt((direction ** 2).sum(axis=1, keepdims=True) + 1e-12)
        radius = sigma * rng.uniform(size=(clean.shape[0], 1)).astype(np.float32) ** (1.0 / 3.0)
        noise = direction * radius
        expert = 2
    else:
        raise ValueError("unknown noise family: %s" % family)

    return (clean + noise).astype(np.float32), np.float32(sigma), np.int32(expert), outlier_mask


class RobustSurfacePatchDataset(Dataset):
    def __init__(
        self,
        data_root,
        list_path,
        num_points=32768,
        patch_size=1000,
        patch_ratio=1.2,
        augment=True,
        batch_size=4,
        shuffle=True,
        num_workers=8,
    ):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_root = data_root
        self.relpaths = load_relpaths(list_path)
        self.num_points = int(num_points)
        self.patch_size = int(patch_size)
        self.patch_ratio = float(patch_ratio)
        self.augment = bool(augment)
        self.set_attrs(total_len=len(self.relpaths))

    def mesh_path(self, relpath):
        return os.path.join(self.data_root, "dataset_train", relpath, "models", "model_normalized.obj")

    def make_item(self, index, rng, family=None):
        dense_count = int(round(self.num_points * self.patch_ratio))
        clean_dense, clean_normals = sample_surface(self.mesh_path(self.relpaths[index]), dense_count, rng)
        clean_dense = normalize_surface(clean_dense)
        if self.augment:
            rotation = random_rotation(rng)
            clean_dense = clean_dense @ rotation.T
            clean_normals = clean_normals @ rotation.T
            if rng.rand() < 0.5:
                scale = float(rng.uniform(0.90, 1.10))
                clean_dense *= scale
        clean_normals /= np.sqrt((clean_normals ** 2).sum(axis=1, keepdims=True) + 1e-12)

        clean_source = clean_dense[: self.num_points]
        source_normals = clean_normals[: self.num_points]
        noisy, sigma, expert, outlier_mask = corrupt_points(clean_source, source_normals, rng, family)
        seed_index = int(rng.randint(noisy.shape[0]))
        seed = noisy[seed_index : seed_index + 1]
        _, noisy_indices = cKDTree(noisy).query(seed[0], k=min(self.patch_size, noisy.shape[0]))
        clean_patch_size = int(round(self.patch_size * self.patch_ratio))
        _, clean_indices = cKDTree(clean_dense).query(seed[0], k=min(clean_patch_size, clean_dense.shape[0]))
        noisy_indices = np.asarray(noisy_indices)
        clean_indices = np.asarray(clean_indices)
        permutation = rng.permutation(noisy_indices.shape[0])
        noisy_indices = noisy_indices[permutation]
        return {
            "pcl_noisy": (noisy[noisy_indices] - seed).astype(np.float32),
            "pcl_clean": (clean_dense[clean_indices] - seed).astype(np.float32),
            "pcl_clean_pair": (clean_source[noisy_indices] - seed).astype(np.float32),
            "pcl_clean_normal": source_normals[noisy_indices].astype(np.float32),
            "noise_std": np.asarray([sigma], dtype=np.float32),
            "expert_label": np.asarray([expert], dtype=np.int32),
            "outlier_mask": outlier_mask[noisy_indices].astype(np.float32),
            "family_id": np.asarray([VAL_FAMILIES.index(family) if family else -1], dtype=np.int32),
        }

    def __getitem__(self, index):
        return self.make_item(index, np.random.RandomState(np.random.randint(0, 2 ** 31 - 1)))


class RobustSurfaceValidationDataset(RobustSurfacePatchDataset):
    def __init__(self, *args, max_shapes=12, val_seed=8051, **kwargs):
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
