import os
from typing import List, Tuple

import numpy as np
import trimesh
from jittor.dataset import Dataset


def load_relpaths(list_path: str) -> List[str]:
    with open(list_path, "r") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def normalize_unit_sphere(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    center = (points.max(axis=0, keepdims=True) + points.min(axis=0, keepdims=True)) / 2.0
    points = points - center
    scale = float(np.sqrt((points ** 2).sum(axis=1)).max())
    if scale > 1e-8:
        points = points / scale
    return points.astype(np.float32), center.astype(np.float32), scale


def denormalize_unit_sphere(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return (points * scale + center).astype(np.float32)


def random_rotate(points: np.ndarray) -> np.ndarray:
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_y = np.array(
        [[cos_t, 0.0, sin_t],
         [0.0, 1.0, 0.0],
         [-sin_t, 0.0, cos_t]],
        dtype=np.float32,
    )
    return points @ rot_y.T


def random_small_rotate(points: np.ndarray, max_degree: float = 10.0) -> np.ndarray:
    angles = np.deg2rad(np.random.uniform(-max_degree, max_degree, size=3)).astype(np.float32)
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rot_x = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, cx, -sx],
         [0.0, sx, cx]],
        dtype=np.float32,
    )
    rot_y = np.array(
        [[cy, 0.0, sy],
         [0.0, 1.0, 0.0],
         [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rot_z = np.array(
        [[cz, -sz, 0.0],
         [sz, cz, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rot = rot_z @ rot_y @ rot_x
    return points @ rot.T


def random_scale(points: np.ndarray, low: float = 0.9, high: float = 1.1) -> np.ndarray:
    scale = float(np.random.uniform(low, high))
    return (points * scale).astype(np.float32)


def random_anisotropic_scale(points: np.ndarray, low: float = 0.85, high: float = 1.15) -> np.ndarray:
    scales = np.random.uniform(low, high, size=(1, 3)).astype(np.float32)
    return (points * scales).astype(np.float32)


def random_translate(points: np.ndarray, shift: float = 0.1) -> np.ndarray:
    delta = np.random.uniform(-shift, shift, size=(1, 3)).astype(np.float32)
    return (points + delta).astype(np.float32)


def random_density_resample(points: np.ndarray, keep_ratio_low: float = 0.6, keep_ratio_high: float = 0.95) -> np.ndarray:
    num_points = points.shape[0]
    keep_ratio = float(np.random.uniform(keep_ratio_low, keep_ratio_high))
    keep_num = max(32, min(num_points, int(num_points * keep_ratio)))
    keep_idx = np.random.choice(num_points, size=keep_num, replace=False)
    kept = points[keep_idx]
    if keep_num < num_points:
        refill_idx = np.random.choice(keep_num, size=num_points - keep_num, replace=True)
        kept = np.concatenate([kept, kept[refill_idx]], axis=0)
    shuffle_idx = np.random.permutation(num_points)
    return kept[shuffle_idx].astype(np.float32)


def random_flip(points: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        points[:, 0] = -points[:, 0]
    if np.random.rand() < 0.5:
        points[:, 2] = -points[:, 2]
    return points


def add_laplace_noise(points: np.ndarray, noise_std_min: float, noise_std_max: float) -> Tuple[np.ndarray, float]:
    noise_std = float(np.random.uniform(noise_std_min, noise_std_max))
    noisy = points + np.random.laplace(0.0, noise_std, size=points.shape).astype(np.float32)
    return noisy.astype(np.float32), noise_std


def add_gaussian_noise(points: np.ndarray, noise_std_min: float, noise_std_max: float) -> Tuple[np.ndarray, float]:
    noise_std = float(np.random.uniform(noise_std_min, noise_std_max))
    noisy = points + np.random.normal(0.0, noise_std, size=points.shape).astype(np.float32)
    return noisy.astype(np.float32), noise_std


def add_uniform_ball_noise(points: np.ndarray, noise_std_min: float, noise_std_max: float) -> Tuple[np.ndarray, float]:
    noise_std = float(np.random.uniform(noise_std_min, noise_std_max))
    num_points = points.shape[0]
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=num_points).astype(np.float32)
    costheta = np.random.uniform(-1.0, 1.0, size=num_points).astype(np.float32)
    theta = np.arccos(costheta).astype(np.float32)
    radius = (noise_std * np.random.uniform(0.0, 1.0, size=num_points) ** (1.0 / 3.0)).astype(np.float32)
    noise = np.zeros_like(points, dtype=np.float32)
    noise[:, 0] = radius * np.sin(theta) * np.cos(phi)
    noise[:, 1] = radius * np.sin(theta) * np.sin(phi)
    noise[:, 2] = radius * np.cos(theta)
    return (points + noise).astype(np.float32), noise_std


def add_directional_noise(points: np.ndarray, noise_std_min: float, noise_std_max: float,
                          lateral_ratio: float = 0.2, hetero_strength: float = 0.35) -> Tuple[np.ndarray, float]:
    """Simulate sensor noise dominated by one acquisition/ray direction."""
    noise_std = float(np.random.uniform(noise_std_min, noise_std_max))
    direction = np.random.normal(size=(3,)).astype(np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    projection = points @ direction
    projection = (projection - projection.mean()) / (projection.std() + 1e-8)
    local_scale = 1.0 + hetero_strength * np.tanh(projection)
    axial = np.random.normal(0.0, noise_std, size=(points.shape[0], 1)).astype(np.float32)
    axial = axial * local_scale[:, None] * direction[None, :]
    lateral = np.random.normal(0.0, noise_std * lateral_ratio, size=points.shape).astype(np.float32)
    return (points + axial + lateral).astype(np.float32), noise_std


def add_mixed_noise(points: np.ndarray, noise_std_min: float, noise_std_max: float,
                    gaussian_prob: float = 0.3, uniform_prob: float = 0.1,
                    directional_prob: float = 0.0, directional_lateral_ratio: float = 0.2,
                    directional_hetero_strength: float = 0.35) -> Tuple[np.ndarray, float]:
    weights = np.array(
        [
            max(0.0, 1.0 - gaussian_prob - uniform_prob - directional_prob),
            max(0.0, gaussian_prob),
            max(0.0, uniform_prob),
            max(0.0, directional_prob),
        ],
        dtype=np.float32,
    )
    if float(weights.sum()) <= 0.0:
        weights = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    weights = weights / weights.sum()
    noise_type = int(np.random.choice(4, p=weights))
    if noise_type == 1:
        return add_gaussian_noise(points, noise_std_min, noise_std_max)
    if noise_type == 2:
        return add_uniform_ball_noise(points, noise_std_min, noise_std_max)
    if noise_type == 3:
        return add_directional_noise(
            points,
            noise_std_min,
            noise_std_max,
            lateral_ratio=directional_lateral_ratio,
            hetero_strength=directional_hetero_strength,
        )
    return add_laplace_noise(points, noise_std_min, noise_std_max)


def add_training_noise(points: np.ndarray, noise_std_min: float, noise_std_max: float,
                       noise_mode: str = "laplace", mixed_gaussian_prob: float = 0.3,
                       mixed_uniform_prob: float = 0.1, mixed_directional_prob: float = 0.0,
                       directional_lateral_ratio: float = 0.2,
                       directional_hetero_strength: float = 0.35) -> Tuple[np.ndarray, float]:
    if noise_mode == "gaussian":
        return add_gaussian_noise(points, noise_std_min, noise_std_max)
    if noise_mode == "uniform":
        return add_uniform_ball_noise(points, noise_std_min, noise_std_max)
    if noise_mode == "directional":
        return add_directional_noise(
            points,
            noise_std_min,
            noise_std_max,
            lateral_ratio=directional_lateral_ratio,
            hetero_strength=directional_hetero_strength,
        )
    if noise_mode == "mixed":
        return add_mixed_noise(
            points,
            noise_std_min,
            noise_std_max,
            gaussian_prob=mixed_gaussian_prob,
            uniform_prob=mixed_uniform_prob,
            directional_prob=mixed_directional_prob,
            directional_lateral_ratio=directional_lateral_ratio,
            directional_hetero_strength=directional_hetero_strength,
        )
    return add_laplace_noise(points, noise_std_min, noise_std_max)


def sample_mesh_points(obj_path: str, num_points: int) -> np.ndarray:
    mesh = trimesh.load(obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    # trimesh may return a TrackedArray, which Jittor's collate_batch does not support.
    points = np.asarray(mesh.sample(num_points), dtype=np.float32).copy()
    return points


def knn_indices_np(seed_points: np.ndarray, points: np.ndarray, k: int) -> np.ndarray:
    diff = seed_points[:, None, :] - points[None, :, :]
    dist = np.sum(diff * diff, axis=-1)
    idx = np.argpartition(dist, kth=min(k - 1, points.shape[0] - 1), axis=1)[:, :k]
    row = np.arange(seed_points.shape[0])[:, None]
    local_dist = dist[row, idx]
    order = np.argsort(local_dist, axis=1)
    return idx[row, order]


def make_vm_patch(noisy_l2: np.ndarray, clean: np.ndarray, patch_size: int):
    seed_idx = np.random.randint(0, noisy_l2.shape[0], size=(1,))
    seed_points = noisy_l2[seed_idx]
    nn_idx = knn_indices_np(seed_points, noisy_l2, patch_size)[0]
    pat_a = noisy_l2[nn_idx]
    pat_b = clean[nn_idx]
    t = np.random.uniform(1e-8, 1.0, size=(patch_size, 1)).astype(np.float32)
    pat_t = t * pat_b + (1 - t) * pat_a
    seed_points_t = t[0:1] * clean[seed_idx] + (1 - t[0:1]) * noisy_l2[seed_idx]
    pat_a = pat_a - seed_points_t
    pat_b = pat_b - seed_points_t
    pat_t = pat_t - seed_points_t
    return pat_a.astype(np.float32), pat_t.astype(np.float32), pat_b.astype(np.float32)


def make_straightpcf_patch(noisy_l2: np.ndarray, clean: np.ndarray, patch_size: int):
    seed_idx = np.random.randint(0, noisy_l2.shape[0], size=(1,))
    seed_points = noisy_l2[seed_idx]
    nn_idx = knn_indices_np(seed_points, noisy_l2, patch_size)[0]
    pat_a = noisy_l2[nn_idx]
    pat_b = clean[nn_idx]
    t = np.random.uniform(1e-8, 1.0, size=(1,)).astype(np.float32)
    seed_points_t = t.reshape(1, 1) * clean[seed_idx] + (1 - t.reshape(1, 1)) * noisy_l2[seed_idx]
    return pat_a.astype(np.float32), pat_b.astype(np.float32), seed_points_t.astype(np.float32), t.astype(np.float32)


class CompetitionTrainPatchDataset(Dataset):
    def __init__(self, data_root: str, list_path: str, mode: str = "vm",
                 num_points: int = 50000, patch_size: int = 1000,
                 noise_std_min: float = 0.005, noise_std_max: float = 0.02,
                 augment_rotate: bool = True, augment_small_rotate: bool = True,
                 augment_scale: bool = True, scale_low: float = 0.9, scale_high: float = 1.1,
                 augment_flip: bool = True, small_rotate_degree: float = 10.0,
                 augment_translate: bool = False, translate_shift: float = 0.1,
                 augment_anisotropic_scale: bool = False,
                 aniso_scale_low: float = 0.85, aniso_scale_high: float = 1.15,
                 augment_density: bool = False, density_keep_low: float = 0.6, density_keep_high: float = 0.95,
                 noise_mode: str = "laplace", mixed_gaussian_prob: float = 0.3, mixed_uniform_prob: float = 0.1,
                 mixed_directional_prob: float = 0.0, directional_lateral_ratio: float = 0.2,
                 directional_hetero_strength: float = 0.35,
                 batch_size: int = 8,
                 shuffle: bool = True, num_workers: int = 4):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_root = data_root
        self.relpaths = load_relpaths(list_path)
        self.mode = mode
        self.num_points = num_points
        self.patch_size = patch_size
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.augment_rotate = augment_rotate
        self.augment_small_rotate = augment_small_rotate
        self.augment_scale = augment_scale
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.augment_flip = augment_flip
        self.small_rotate_degree = small_rotate_degree
        self.augment_translate = augment_translate
        self.translate_shift = translate_shift
        self.augment_anisotropic_scale = augment_anisotropic_scale
        self.aniso_scale_low = aniso_scale_low
        self.aniso_scale_high = aniso_scale_high
        self.augment_density = augment_density
        self.density_keep_low = density_keep_low
        self.density_keep_high = density_keep_high
        self.noise_mode = noise_mode
        self.mixed_gaussian_prob = mixed_gaussian_prob
        self.mixed_uniform_prob = mixed_uniform_prob
        self.mixed_directional_prob = mixed_directional_prob
        self.directional_lateral_ratio = directional_lateral_ratio
        self.directional_hetero_strength = directional_hetero_strength
        self.set_attrs(total_len=len(self.relpaths))

    def _obj_path(self, relpath: str) -> str:
        return os.path.join(self.data_root, "dataset_train", relpath, "models", "model_normalized.obj")

    def __getitem__(self, idx):
        relpath = self.relpaths[idx]
        clean = sample_mesh_points(self._obj_path(relpath), self.num_points)
        clean, _, _ = normalize_unit_sphere(clean)
        if self.augment_density:
            clean = random_density_resample(clean, self.density_keep_low, self.density_keep_high)
        if self.augment_rotate:
            clean = random_rotate(clean)
        if self.augment_small_rotate:
            clean = random_small_rotate(clean, self.small_rotate_degree)
        if self.augment_flip:
            clean = random_flip(clean.copy())
        if self.augment_scale:
            clean = random_scale(clean, self.scale_low, self.scale_high)
        if self.augment_anisotropic_scale:
            clean = random_anisotropic_scale(clean, self.aniso_scale_low, self.aniso_scale_high)
        if self.augment_translate:
            clean = random_translate(clean, self.translate_shift)
        noisy_l2, noise_std = add_training_noise(
            clean,
            self.noise_std_min,
            self.noise_std_max,
            noise_mode=self.noise_mode,
            mixed_gaussian_prob=self.mixed_gaussian_prob,
            mixed_uniform_prob=self.mixed_uniform_prob,
            mixed_directional_prob=self.mixed_directional_prob,
            directional_lateral_ratio=self.directional_lateral_ratio,
            directional_hetero_strength=self.directional_hetero_strength,
        )

        if self.mode == "vm":
            pcl_noisy_l2, pcl_noisy_mix, pcl_clean = make_vm_patch(noisy_l2, clean, self.patch_size)
            return {
                "pcl_noisy_l2": pcl_noisy_l2,
                "pcl_noisy_mix": pcl_noisy_mix,
                "pcl_clean": pcl_clean,
            }

        pcl_noisy_l2, pcl_clean, seed_points_t, original_time_step = make_straightpcf_patch(noisy_l2, clean, self.patch_size)
        return {
            "pcl_noisy_l2": pcl_noisy_l2,
            "pcl_clean": pcl_clean,
            "seed_points_t": seed_points_t,
            "original_time_step": original_time_step,
        }


class CompetitionValDataset:
    def __init__(self, data_root: str, list_path: str, num_points: int = 10000,
                 noise_std: float = 0.015, max_items: int = 32):
        self.data_root = data_root
        self.relpaths = load_relpaths(list_path)[:max_items]
        self.num_points = num_points
        self.noise_std = noise_std

    def _obj_path(self, relpath: str) -> str:
        return os.path.join(self.data_root, "dataset_train", relpath, "models", "model_normalized.obj")

    def __len__(self):
        return len(self.relpaths)

    def __getitem__(self, idx):
        relpath = self.relpaths[idx]
        clean = sample_mesh_points(self._obj_path(relpath), self.num_points)
        clean, _, _ = normalize_unit_sphere(clean)
        noisy = clean + np.random.laplace(0.0, self.noise_std, size=clean.shape).astype(np.float32)
        return {
            "relpath": relpath,
            "pcl_clean": clean.astype(np.float32),
            "pcl_noisy": noisy.astype(np.float32),
        }


class CompetitionPredictDataset:
    def __init__(self, data_root: str, list_path: str):
        self.data_root = data_root
        self.relpaths = load_relpaths(list_path)

    def _npy_path(self, relpath: str) -> str:
        return os.path.join(self.data_root, "dataset_test_noisy", relpath, "noisy.npy")

    def __len__(self):
        return len(self.relpaths)

    def __getitem__(self, idx):
        relpath = self.relpaths[idx]
        noisy = np.load(self._npy_path(relpath)).astype(np.float32)
        return {
            "relpath": relpath,
            "pcl_noisy": noisy,
        }
