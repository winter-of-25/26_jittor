import hashlib
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from aligned_ot_data_v29 import calibrated_corrupt
from b20_utils import array_sha256, load_json
from full_cloud_validation_v62_v65 import _cloud_distances, official_sample_score
from jittor_port_data_base import denormalize_unit_sphere, normalize_unit_sphere
from orthogonal_backbones_v62_v65 import patch_based_orthogonal_pair
from robust_surface_data import normalize_surface, sample_surface


def _bootstrap_interval(values, seed=20260824, samples=5000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    return {
        "samples": int(samples),
        "mean": float(values.mean()),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
    }


class B24FixedSuite:
    def __init__(
        self,
        data_root,
        manifest_path,
        split="all",
        screen_only=False,
        point_count=50000,
        surface_count=250000,
    ):
        self.data_root = data_root
        self.manifest_path = manifest_path
        self.manifest = load_json(manifest_path)
        self.point_count = int(point_count)
        self.surface_count = int(surface_count)
        self.entries = [
            item
            for item in self.manifest["entries"]
            if (split == "all" or item["split"] == split)
            and (not screen_only or item["screen"])
        ]

    def _mesh_path(self, relpath):
        return os.path.join(
            self.data_root,
            "dataset_train",
            relpath,
            "models",
            "model_normalized.obj",
        )

    def _scenario(self, entry):
        count = self.point_count + self.surface_count
        surface, normals = sample_surface(
            self._mesh_path(entry["path"]),
            count,
            np.random.RandomState(int(entry["surface_seed"])),
        )
        surface = normalize_surface(surface)
        normals /= np.sqrt((normals ** 2).sum(axis=1, keepdims=True) + 1e-12)
        clean = surface[: self.point_count].astype(np.float32)
        clean_normals = normals[: self.point_count].astype(np.float32)
        noisy, sigma, family = calibrated_corrupt(
            clean,
            clean_normals,
            np.random.RandomState(int(entry["primary_seed"])),
            family=entry["primary_family"],
        )
        clean_tree = cKDTree(clean)
        surface_tree = cKDTree(surface)
        noisy_cd, noisy_p2s = _cloud_distances(noisy, clean, clean_tree, surface_tree)
        return {
            "path": entry["path"],
            "family": family,
            "seed": int(entry["primary_seed"]),
            "sigma": float(sigma),
            "clean": clean,
            "noisy": noisy,
            "clean_tree": clean_tree,
            "surface_tree": surface_tree,
            "noisy_cd": noisy_cd,
            "noisy_p2s": noisy_p2s,
            "noisy_sha256": array_sha256(noisy),
            "surface_sha256": array_sha256(surface),
        }

    def describe(self):
        return {
            "shape_count": len(self.entries),
            "point_count": self.point_count,
            "surface_count": self.surface_count,
            "families": list(self.manifest["primary_families"]),
            "source": "exact archived B20 SCREEN20/FULL100 manifest",
        }

    def evaluate(
        self,
        model,
        strengths,
        patch_size,
        seed_k,
        beta,
        patch_batch,
        output_cache_dir="",
    ):
        strengths = tuple(float(value) for value in strengths)
        records = {value: [] for value in strengths}
        started = time.time()
        model.eval()
        if output_cache_dir:
            os.makedirs(output_cache_dir, exist_ok=True)
        for index, entry in enumerate(self.entries, 1):
            scenario = self._scenario(entry)
            normalized, center, scale = normalize_unit_sphere(scenario["noisy"])
            cache_key = hashlib.sha256(
                (
                    scenario["path"]
                    + "|"
                    + scenario["family"]
                    + "|"
                    + str(scenario["seed"])
                    + "|"
                    + scenario["noisy_sha256"]
                ).encode("utf-8")
            ).hexdigest()
            cache_path = (
                os.path.join(output_cache_dir, cache_key + ".npz")
                if output_cache_dir
                else ""
            )
            if cache_path and os.path.isfile(cache_path):
                cached = np.load(cache_path)
                teacher = cached["teacher"]
                active = cached["active"]
            else:
                teacher_normalized, active_normalized = patch_based_orthogonal_pair(
                    model,
                    normalized,
                    patch_size=patch_size,
                    seed_k=seed_k,
                    beta=beta,
                    patch_batch=patch_batch,
                )
                teacher = denormalize_unit_sphere(teacher_normalized, center, scale)
                active = denormalize_unit_sphere(active_normalized, center, scale)
                if cache_path:
                    np.savez(
                        cache_path,
                        teacher=teacher.astype(np.float32),
                        active=active.astype(np.float32),
                    )
            teacher_cd, teacher_p2s = _cloud_distances(
                teacher,
                scenario["clean"],
                scenario["clean_tree"],
                scenario["surface_tree"],
            )
            teacher_cd_score = official_sample_score(teacher_cd, scenario["noisy_cd"])
            teacher_p2s_score = official_sample_score(
                teacher_p2s, scenario["noisy_p2s"]
            )
            teacher_score = 0.5 * (teacher_cd_score + teacher_p2s_score)
            for strength in strengths:
                candidate = teacher + strength * (active - teacher)
                candidate_cd, candidate_p2s = _cloud_distances(
                    candidate,
                    scenario["clean"],
                    scenario["clean_tree"],
                    scenario["surface_tree"],
                )
                candidate_cd_score = official_sample_score(
                    candidate_cd, scenario["noisy_cd"]
                )
                candidate_p2s_score = official_sample_score(
                    candidate_p2s, scenario["noisy_p2s"]
                )
                candidate_score = 0.5 * (candidate_cd_score + candidate_p2s_score)
                records[strength].append(
                    {
                        "path": scenario["path"],
                        "family": scenario["family"],
                        "seed": scenario["seed"],
                        "noisy_sha256": scenario["noisy_sha256"],
                        "surface_sha256": scenario["surface_sha256"],
                        "teacher": {
                            "cd_score": teacher_cd_score,
                            "p2s_score": teacher_p2s_score,
                            "score": teacher_score,
                        },
                        "candidate": {
                            "cd_score": candidate_cd_score,
                            "p2s_score": candidate_p2s_score,
                            "score": candidate_score,
                        },
                        "delta": {
                            "cd_score": candidate_cd_score - teacher_cd_score,
                            "p2s_score": candidate_p2s_score - teacher_p2s_score,
                            "score": candidate_score - teacher_score,
                        },
                        "movement_rms": float(
                            np.sqrt(np.mean((candidate - teacher) ** 2))
                        ),
                    }
                )
            print(
                "B24Validation [%d/%d] %s %s"
                % (index, len(self.entries), scenario["path"], scenario["family"]),
                flush=True,
            )
        elapsed = time.time() - started
        return {
            "%.2f" % strength: summarize(records[strength], self.describe(), elapsed)
            for strength in strengths
        }


def summarize(samples, suite, elapsed):
    total = np.asarray([item["delta"]["score"] for item in samples], dtype=np.float64)
    cd = np.asarray([item["delta"]["cd_score"] for item in samples], dtype=np.float64)
    p2s = np.asarray([item["delta"]["p2s_score"] for item in samples], dtype=np.float64)
    movement = np.asarray([item["movement_rms"] for item in samples], dtype=np.float64)
    family_means = {}
    for family in sorted(set(item["family"] for item in samples)):
        values = [item["delta"]["score"] for item in samples if item["family"] == family]
        family_means[family] = float(np.mean(values))
    return {
        "metric": "paired official-style full-cloud delta vs exact B20 C4",
        "suite": suite,
        "validation_seconds": float(elapsed),
        "mean_delta": float(total.mean()),
        "median_delta": float(np.median(total)),
        "mean_cd_delta": float(cd.mean()),
        "mean_p2s_delta": float(p2s.mean()),
        "win_rate": float(np.mean(total > 0.0)),
        "worst_family": float(min(family_means.values())),
        "mean_movement_rms": float(movement.mean()),
        "family_mean_delta": family_means,
        "bootstrap": _bootstrap_interval(total),
        "worst_five": sorted(samples, key=lambda item: item["delta"]["score"])[:5],
        "samples": samples,
    }

