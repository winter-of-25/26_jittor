import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from aligned_ot_data_v29 import calibrated_corrupt
from jittor_port_data_base import (
    denormalize_unit_sphere,
    normalize_unit_sphere,
)
from orthogonal_backbones_v62_v65 import (
    patch_based_orthogonal_pair,
)
from robust_surface_data import (
    load_relpaths,
    normalize_surface,
    sample_surface,
)


def official_sample_score(prediction_distance, noisy_distance):
    value = 100.0 * (
        1.0
        - float(prediction_distance)
        / max(float(noisy_distance), 1e-12)
    )
    return float(np.clip(value, 0.0, 100.0))


def _cloud_distances(points, clean, clean_tree, surface_tree):
    pred_to_clean = clean_tree.query(
        points, k=1, workers=-1
    )[0]
    prediction_tree = cKDTree(points)
    clean_to_pred = prediction_tree.query(
        clean, k=1, workers=-1
    )[0]
    point_to_surface = surface_tree.query(
        points, k=1, workers=-1
    )[0]
    chamfer = (
        np.mean(pred_to_clean ** 2)
        + np.mean(clean_to_pred ** 2)
    )
    p2s = np.mean(point_to_surface ** 2)
    return float(chamfer), float(p2s)


class FullCloudValidationSuite:
    """Fixed full-cloud validation through the submission inference path."""

    def __init__(
        self,
        data_root,
        list_path,
        max_shapes=6,
        families=(
            "laplace_low",
            "laplace_mid",
            "laplace_high",
            "compound",
        ),
        point_count=50000,
        surface_count=250000,
        seed=620065,
    ):
        self.data_root = data_root
        self.relpaths = load_relpaths(list_path)[
            : int(max_shapes)
        ]
        self.families = tuple(families)
        self.point_count = int(point_count)
        self.surface_count = int(surface_count)
        self.seed = int(seed)
        self.scenarios = self._build()

    def _mesh_path(self, relpath):
        return os.path.join(
            self.data_root,
            "dataset_train",
            relpath,
            "models",
            "model_normalized.obj",
        )

    def _build(self):
        scenarios = []
        for shape_index, relpath in enumerate(self.relpaths):
            shape_rng = np.random.RandomState(
                self.seed + 1000003 * shape_index
            )
            count = self.point_count + self.surface_count
            surface, normals = sample_surface(
                self._mesh_path(relpath), count, shape_rng
            )
            surface = normalize_surface(surface)
            normals = normals / (
                np.sqrt(
                    (normals ** 2).sum(
                        axis=1, keepdims=True
                    )
                    + 1e-12
                )
            )
            clean = surface[: self.point_count].astype(
                np.float32
            )
            clean_normal = normals[
                : self.point_count
            ].astype(np.float32)
            clean_tree = cKDTree(clean)
            surface_tree = cKDTree(surface)
            for family_index, family in enumerate(
                self.families
            ):
                noise_rng = np.random.RandomState(
                    self.seed
                    + 1000003 * shape_index
                    + 7919 * (family_index + 1)
                )
                noisy, sigma, _ = calibrated_corrupt(
                    clean,
                    clean_normal,
                    noise_rng,
                    family=family,
                )
                noisy_cd, noisy_p2s = _cloud_distances(
                    noisy,
                    clean,
                    clean_tree,
                    surface_tree,
                )
                scenarios.append(
                    {
                        "shape": relpath,
                        "family": family,
                        "sigma": float(sigma),
                        "clean": clean,
                        "noisy": noisy,
                        "clean_tree": clean_tree,
                        "surface_tree": surface_tree,
                        "noisy_cd": noisy_cd,
                        "noisy_p2s": noisy_p2s,
                    }
                )
        return scenarios

    def describe(self):
        return {
            "shape_count": len(self.relpaths),
            "scenario_count": len(self.scenarios),
            "families": list(self.families),
            "point_count": self.point_count,
            "surface_count": self.surface_count,
            "mean_noisy_cd": float(
                np.mean(
                    [
                        item["noisy_cd"]
                        for item in self.scenarios
                    ]
                )
            ),
            "mean_noisy_p2s": float(
                np.mean(
                    [
                        item["noisy_p2s"]
                        for item in self.scenarios
                    ]
                )
            ),
        }

    def evaluate(
        self,
        model,
        strengths,
        patch_size=1000,
        seed_k=6,
        beta=12.0,
        patch_batch=8,
    ):
        model.eval()
        model.set_frozen_eval()
        strengths = tuple(float(value) for value in strengths)
        records = {
            strength: [] for strength in strengths
        }
        started = time.time()
        for index, scenario in enumerate(self.scenarios, 1):
            normalized, center, scale = normalize_unit_sphere(
                scenario["noisy"]
            )
            base_normalized, active_normalized = (
                patch_based_orthogonal_pair(
                    model,
                    normalized,
                    patch_size=patch_size,
                    seed_k=seed_k,
                    beta=beta,
                    patch_batch=patch_batch,
                )
            )
            base = denormalize_unit_sphere(
                base_normalized, center, scale
            )
            active = denormalize_unit_sphere(
                active_normalized, center, scale
            )
            for strength in strengths:
                prediction = base + strength * (
                    active - base
                )
                cd, p2s = _cloud_distances(
                    prediction,
                    scenario["clean"],
                    scenario["clean_tree"],
                    scenario["surface_tree"],
                )
                records[strength].append(
                    {
                        "shape": scenario["shape"],
                        "family": scenario["family"],
                        "sigma": scenario["sigma"],
                        "cd": cd,
                        "p2s": p2s,
                        "noisy_cd": scenario["noisy_cd"],
                        "noisy_p2s": scenario["noisy_p2s"],
                        "cd_score": official_sample_score(
                            cd, scenario["noisy_cd"]
                        ),
                        "p2s_score": official_sample_score(
                            p2s, scenario["noisy_p2s"]
                        ),
                    }
                )
            print(
                "FullCloudValidation [%d/%d] %s %s"
                % (
                    index,
                    len(self.scenarios),
                    scenario["shape"],
                    scenario["family"],
                ),
                flush=True,
            )
        summaries = {}
        for strength, samples in records.items():
            cd_score = float(
                np.mean(
                    [item["cd_score"] for item in samples]
                )
            )
            p2s_score = float(
                np.mean(
                    [item["p2s_score"] for item in samples]
                )
            )
            family_scores = {}
            for family in self.families:
                selected = [
                    item
                    for item in samples
                    if item["family"] == family
                ]
                family_scores[family] = float(
                    np.mean(
                        [
                            0.5
                            * (
                                item["cd_score"]
                                + item["p2s_score"]
                            )
                            for item in selected
                        ]
                    )
                )
            summaries[strength] = {
                "score": 0.5 * (cd_score + p2s_score),
                "cd_score": cd_score,
                "p2s_score": p2s_score,
                "mean_cd": float(
                    np.mean([item["cd"] for item in samples])
                ),
                "mean_p2s": float(
                    np.mean([item["p2s"] for item in samples])
                ),
                "worst_family_score": min(
                    family_scores.values()
                ),
                "family_scores": family_scores,
                "samples": samples,
            }
        base_score = summaries[0.0]["score"]
        for summary in summaries.values():
            summary["gain_vs_v45"] = (
                summary["score"] - base_score
            )
        best_strength = max(
            summaries,
            key=lambda value: (
                summaries[value]["score"],
                summaries[value]["worst_family_score"],
            ),
        )
        active_strengths = [
            value for value in summaries if value > 0.0
        ]
        best_active = max(
            active_strengths,
            key=lambda value: (
                summaries[value]["score"],
                summaries[value]["worst_family_score"],
            ),
        )
        return {
            "metric": (
                "per-sample clipped official-style "
                "0.5*CD+0.5*P2S"
            ),
            "full_inference_path": True,
            "refinement_strength": float(best_strength),
            "selection_score": float(
                summaries[best_strength]["score"]
            ),
            "active_refinement_strength": float(best_active),
            "active_selection_score": float(
                summaries[best_active]["score"]
            ),
            "v45_score": float(base_score),
            "active_gain_vs_v45": float(
                summaries[best_active]["score"] - base_score
            ),
            "validation_seconds": time.time() - started,
            "suite": self.describe(),
            "strengths": {
                "%.3f" % key: value
                for key, value in summaries.items()
            },
        }


def dump_record(path, record):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
