import argparse
import gc
import inspect
import json
import math
import os
import time

import jittor as jt
import numpy as np
from jittor import nn

from aligned_ot_data_v29 import calibrated_corrupt
from b20_model import build_b20, set_frozen_eval as set_b20_inference
from b20_utils import (
    all_finite,
    make_validation_manifest,
    max_parameter_change,
    parameter_count,
    parameter_snapshot,
    read_lines,
    sha256_file,
    write_json,
)
from b24_cagrad import (
    CAGRAD_C,
    brute_force_weight,
    cagrad_direction,
    cagrad_optimizer_step,
    gram_cosine,
    solve_two_task_weight,
    surrogate_loss,
    task_gradient_pair,
)
from b24_model import (
    B20_MODEL_BUILD_SEED,
    build_b24,
    manifest_fingerprint,
    prepare_stage,
    recovered_head_manifest,
    recovered_head_parameters,
    student_parameters,
)
from jittor_port_data_base import normalize_unit_sphere
from orthogonal_backbones_v62_v65 import (
    patch_based_orthogonal,
    patch_based_orthogonal_pair,
)
from robust_surface_data import normalize_surface, sample_surface
from train_b24 import (
    EXPECTED_TEACHER_SHA256,
    add_model_arguments,
    b24_task_losses,
    make_dataset,
)


EXPECTED_PARENT_SHA = {
    "v12": "62fb5e8fa4aaa93d66dc5e7011ec9eb512cb1d7d4b90767faae3f959a798cf48",
    "v12_args": "a2cea9a1f2b3e46d7985ebd7c51513199d8fd20969cdf968b89dc6c56bc8358e",
    "v29": "ad6039b2200d4f6a254f5b8349bde7932b8cf9a1e9fd745d728926daa79bcfc9",
    "v33": "41bda3e8e1dd3f4b2584f55c20fbe04088108b41472daf95e7c8cf80b50a4b57",
    "v41": "8e7f15742a058cc484b2b498e917d8369bd7bf16172193e5d4c1b5a17a2a63ab",
    "v45": "c71f2a72997e009fe7f21886dc9c8ad4511f7b42c5538fea9c16a277b9b72cd1",
    "v65": "80bd13b37def623a356a6cda413b258de42c0d0891f39d33f07fa26809a06527",
    "teacher20": EXPECTED_TEACHER_SHA256,
}


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--data_root", default="/root")
    parser.add_argument("--train_list", default="/root/datalist/train_b.txt")
    parser.add_argument("--val_list", default="/root/datalist/validate_b.txt")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=32768)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--patch_ratio", type=float, default=1.2)
    parser.add_argument("--alignment_k", type=int, default=32)
    parser.add_argument("--patch_batch", type=int, default=6)
    parser.add_argument("--full_cloud_audits", type=int, default=4)
    parser.add_argument("--reproduction_only", type=int, default=0)
    parser.add_argument("--conflict_steps", type=int, default=50)
    parser.add_argument("--smoke_steps", type=int, default=200)
    parser.add_argument("--use_cuda", type=int, default=1)
    add_model_arguments(parser)
    return parser


def _full_cloud(data_root, relpath, seed):
    mesh_path = os.path.join(
        data_root,
        "dataset_train",
        relpath,
        "models",
        "model_normalized.obj",
    )
    rng = np.random.RandomState(int(seed))
    clean, normals = sample_surface(mesh_path, 50000, rng)
    clean = normalize_surface(clean)
    normals /= np.sqrt((normals ** 2).sum(axis=1, keepdims=True) + 1e-12)
    noisy, _, _ = calibrated_corrupt(clean, normals, rng, family="laplace_mid")
    return noisy.astype(np.float32)


def _numpy_direction(first, second, weight, c=CAGRAD_C):
    g0 = 0.5 * (first + second)
    gw = float(weight) * first + (1.0 - float(weight)) * second
    g0_norm = float(np.linalg.norm(g0))
    gw_norm = float(np.linalg.norm(gw))
    if g0_norm < 1e-12 or gw_norm < 1e-12:
        return g0 / (1.0 + c * c)
    coefficient = c * g0_norm / (gw_norm + 1e-12)
    return (g0 + coefficient * gw) / (1.0 + c * c)


def _solver_audit(seed):
    rng = np.random.RandomState(int(seed))
    max_weight_error = 0.0
    min_cosine = 1.0
    max_norm_error = 0.0
    for _ in range(100):
        first = rng.normal(size=256).astype(np.float64)
        second = rng.normal(size=256).astype(np.float64)
        aa = float(np.dot(first, first))
        ab = float(np.dot(first, second))
        bb = float(np.dot(second, second))
        weight = solve_two_task_weight(aa, ab, bb)
        reference_weight = brute_force_weight(aa, ab, bb, points=50001)
        direction = _numpy_direction(first, second, weight)
        reference = _numpy_direction(first, second, reference_weight)
        cosine = float(
            np.dot(direction, reference)
            / (np.linalg.norm(direction) * np.linalg.norm(reference) + 1e-12)
        )
        norm_error = float(
            abs(np.linalg.norm(direction) - np.linalg.norm(reference))
            / (np.linalg.norm(reference) + 1e-12)
        )
        max_weight_error = max(max_weight_error, abs(weight - reference_weight))
        min_cosine = min(min_cosine, cosine)
        max_norm_error = max(max_norm_error, norm_error)
    if max_weight_error > 0.002 or min_cosine < 0.99999 or max_norm_error > 1e-4:
        raise RuntimeError("B24 CAGrad solver audit failed")

    canonical = {}
    cases = {
        "identical": (np.ones(8), np.ones(8)),
        "opposite": (np.ones(8), -np.ones(8)),
        "orthogonal": (
            np.asarray([1.0, 0.0] * 4),
            np.asarray([0.0, 1.0] * 4),
        ),
    }
    for name, (first, second) in cases.items():
        aa = float(np.dot(first, first))
        ab = float(np.dot(first, second))
        bb = float(np.dot(second, second))
        weight = solve_two_task_weight(aa, ab, bb)
        direction = _numpy_direction(first, second, weight)
        if not np.isfinite(direction).all():
            raise RuntimeError("non-finite canonical CAGrad case: %s" % name)
        canonical[name] = {
            "weight": weight,
            "cosine": gram_cosine(aa, ab, bb),
            "direction_norm": float(np.linalg.norm(direction)),
        }
    return {
        "cases": 100,
        "max_weight_error_vs_brute": max_weight_error,
        "min_direction_cosine": min_cosine,
        "max_relative_norm_error": max_norm_error,
        "canonical": canonical,
    }


def _autograd_audit(seed):
    rng = np.random.RandomState(int(seed))
    values = rng.normal(size=(16,)).astype(np.float32)
    variable = jt.array(values)
    variable.start_grad()
    gradient = jt.grad((variable * variable).sum(), variable, retain_graph=False)
    derivative_error = float(np.max(np.abs(gradient.numpy() - 2.0 * values)))
    if derivative_error > 1e-4:
        raise RuntimeError("jt.grad analytical derivative audit failed")

    parameter = jt.array(rng.normal(size=(32,)).astype(np.float32))
    parameter.start_grad()
    direction_np = rng.normal(size=(32,)).astype(np.float32)
    direction = jt.array(direction_np)
    direction.stop_grad()
    surrogate = surrogate_loss([parameter], [direction])
    surrogate_gradient = jt.grad(surrogate, parameter, retain_graph=False)
    surrogate_error = float(np.max(np.abs(surrogate_gradient.numpy() - direction_np)))
    if surrogate_error > 1e-7:
        raise RuntimeError("B24 surrogate gradient audit failed")
    source = inspect.getsource(cagrad_optimizer_step)
    if source.count("optimizer.zero_grad()") != 1 or source.count("optimizer.step(") != 1:
        raise RuntimeError("B24 optimizer source contains an unexpected extra gradient path")
    return {
        "jt_grad_max_abs_error": derivative_error,
        "surrogate_gradient_max_abs_error": surrogate_error,
        "optimizer_zero_grad_calls": 1,
        "optimizer_step_calls": 1,
        "normal_average_loss_backward_calls": 0,
    }


def main():
    args = _parser().parse_args()
    jt.flags.use_cuda = args.use_cuda
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    os.makedirs(args.run_root, exist_ok=True)
    if (args.num_points, args.patch_size, args.alignment_k) != (32768, 1000, 32):
        raise RuntimeError("B24 preflight geometry invariant mismatch")
    paths = {
        "v12": args.baseline_ckpt,
        "v12_args": args.baseline_args,
        "v29": args.v29_ckpt,
        "v33": args.v33_ckpt,
        "v41": args.v41_ckpt,
        "v45": args.v45_ckpt,
        "v65": args.v65_ckpt,
        "teacher20": args.teacher_ckpt,
    }
    actual_sha = {name: sha256_file(path) for name, path in paths.items()}
    if actual_sha != EXPECTED_PARENT_SHA:
        raise RuntimeError("B24 parent provenance mismatch")
    write_json(
        os.path.join(args.run_root, "parent_manifest.json"),
        {
            "version": "B24",
            "formal_parent": "B20 C4",
            "formal_parent_online": {"score": 81.41, "cd_score": 70.10, "p2s_score": 92.71},
            "whole_anchor_online": {"score": 81.24, "cd_score": 70.18, "p2s_score": 92.31},
            "paths": {name: {"path": paths[name], "sha256": actual_sha[name]} for name in paths},
            "internal_b20_strength": 1.25,
            "v65_strength": 1.50,
        },
    )
    manifest_path = os.path.join(args.run_root, "validation_manifest.json")
    manifest = make_validation_manifest(args.train_list, args.val_list, manifest_path)
    if manifest["train_count"] != 19699 or manifest["validation_count"] != 100:
        raise RuntimeError("B24 fixed validation split mismatch")
    solver_report = _solver_audit(args.seed + 1)
    autograd_report = _autograd_audit(args.seed + 2)

    dataset = make_dataset(args)
    if len(dataset.relpaths) != 19699:
        raise RuntimeError("B24 preflight did not see full B-train")
    batch = next(iter(dataset))
    noisy_np = batch["pcl_noisy"].numpy().copy()
    noisy = jt.array(noisy_np)

    args.head_ckpt = args.teacher_ckpt
    np.random.seed(B20_MODEL_BUILD_SEED)
    jt.set_global_seed(B20_MODEL_BUILD_SEED)
    archived = build_b20(args, load_head=True)
    set_b20_inference(archived)
    with jt.no_grad():
        archived_frozen = archived.frozen_outputs(noisy)
        archived_patch = archived(
            noisy,
            refinement_strength=1.25,
            frozen_outputs=archived_frozen,
        )
    archived_patch_prediction = archived_patch[0].numpy().copy()
    archived_patch_confidence = archived_patch[5]["stitch_confidence"].numpy().copy()

    relpaths = read_lines(args.train_list)[:2] + read_lines(args.val_list)[:2]
    relpaths = relpaths[: int(args.full_cloud_audits)]
    archived_clouds = []
    cloud_inputs = []
    for index, relpath in enumerate(relpaths):
        cloud = _full_cloud(args.data_root, relpath, args.seed + 10 + index)
        normalized, _, _ = normalize_unit_sphere(cloud)
        prediction = patch_based_orthogonal(
            archived,
            normalized,
            patch_size=1000,
            seed_k=6,
            beta=12.0,
            patch_batch=args.patch_batch,
            refinement_strength=1.25,
        )
        cloud_inputs.append(normalized)
        archived_clouds.append(prediction)
        print("B24 archived reproduction [%d/%d]" % (index + 1, len(relpaths)), flush=True)
    del archived, archived_patch, archived_frozen
    gc.collect()
    jt.sync_all(True)

    model = build_b24(args)
    full_parameters = student_parameters(model)
    head_parameters = recovered_head_parameters(model)
    if parameter_count(full_parameters) != 623663:
        raise RuntimeError("B24 full V67 parameter count mismatch")
    head_manifest_a = recovered_head_manifest(model)
    head_manifest_b = recovered_head_manifest(model)
    if manifest_fingerprint(head_manifest_a) != manifest_fingerprint(head_manifest_b):
        raise RuntimeError("B24 HEAD_SET recovery mismatch across repeated executions")
    write_json(os.path.join(args.run_root, "b24_recovered_head_set.json"), head_manifest_a)

    parameter_error = max(
        float(np.max(np.abs(left.numpy() - right.numpy())))
        for left, right in zip(model.teacher.head.parameters(), model.student.head.parameters())
    )
    if parameter_error != 0.0:
        raise RuntimeError("B24 student is not an exact teacher parameter copy")
    components = model.forward_components(noisy)
    teacher_patch_error = float(
        np.max(np.abs(components["teacher_prediction"].numpy() - archived_patch_prediction))
    )
    student_patch_error = float(
        np.max(
            np.abs(
                components["student_raw"].numpy()
                - components["teacher_prediction"].numpy()
            )
        )
    )
    archived_confidence_drift = float(
        np.max(
            np.abs(
                components["teacher_output"][5]["stitch_confidence"].numpy()
                - archived_patch_confidence
            )
        )
    )
    wrapped = model(noisy, refinement_strength=1.0)
    confidence_error = float(
        jt.abs(
            wrapped[5]["stitch_confidence"]
            - wrapped[7][5]["stitch_confidence"]
        ).max().item()
    )
    cap_ratio = float(
        (
            jt.sqrt((components["delta"] ** 2).sum(-1) + 1e-12)
            / (components["r32"].reshape(components["delta"].shape[:2]) + 1e-12)
        ).max().item()
    )
    print(
        "B24 exact patch audit teacher_vs_archived=%.9g student_vs_teacher=%.9g confidence_same_forward=%.9g archived_confidence_drift=%.9g cap_ratio=%.9g"
        % (
            teacher_patch_error,
            student_patch_error,
            confidence_error,
            archived_confidence_drift,
            cap_ratio,
        ),
        flush=True,
    )
    # Parameter/buffer identity is audited exactly above. Independent GPU
    # executions of this stateful KNN lineage are diagnostic only; the strict
    # contract is same-forward student/teacher and confidence identity.
    if student_patch_error > 1e-7 or confidence_error > 1e-7:
        raise RuntimeError("B24 exact patch initialization audit failed")
    if cap_ratio > 0.100001:
        raise RuntimeError("B24 bounded residual cap failed")

    full_cloud_records = []
    for index, (relpath, normalized, archived_prediction) in enumerate(
        zip(relpaths, cloud_inputs, archived_clouds)
    ):
        teacher_prediction, active_prediction = patch_based_orthogonal_pair(
            model,
            normalized,
            patch_size=1000,
            seed_k=6,
            beta=12.0,
            patch_batch=args.patch_batch,
        )
        teacher_error = float(np.max(np.abs(teacher_prediction - archived_prediction)))
        student_error = float(np.max(np.abs(active_prediction - teacher_prediction)))
        print(
            "B24 full-cloud audit [%d/%d] teacher_vs_archived=%.9g student_vs_teacher=%.9g"
            % (index + 1, len(relpaths), teacher_error, student_error),
            flush=True,
        )
        if student_error > 1e-5:
            raise RuntimeError("B24 full-cloud exact initialization audit failed")
        full_cloud_records.append(
            {
                "path": relpath,
                "teacher_vs_archived_max_abs": teacher_error,
                "student_vs_teacher_max_abs": student_error,
            }
        )
        print("B24 student reproduction [%d/%d]" % (index + 1, len(relpaths)), flush=True)

    reproduction_path = os.path.join(args.run_root, "reproduction_passed.json")
    reproduction_report = {
        "status": "PASS",
        "teacher_student_parameter_max_abs": parameter_error,
        "teacher_patch_vs_archived_max_abs": teacher_patch_error,
        "student_patch_vs_teacher_max_abs": student_patch_error,
        "confidence_same_forward_max_abs": confidence_error,
        "independent_archived_confidence_drift": archived_confidence_drift,
        "full_cloud_reproduction": full_cloud_records,
    }
    if args.reproduction_only:
        if len(full_cloud_records) != int(args.full_cloud_audits):
            raise RuntimeError("B24 reproduction-only audit count mismatch")
        write_json(reproduction_path, reproduction_report)
        print(
            "B24_REPRODUCTION_PASS %s"
            % json.dumps(reproduction_report, sort_keys=True),
            flush=True,
        )
        return
    if not full_cloud_records:
        if not os.path.isfile(reproduction_path):
            raise RuntimeError("B24 core preflight requires reproduction_passed.json")
        with open(reproduction_path, "r", encoding="utf-8") as handle:
            reproduction_report = json.load(handle)
        if reproduction_report.get("status") != "PASS":
            raise RuntimeError("B24 saved reproduction audit is not PASS")
        full_cloud_records = reproduction_report["full_cloud_reproduction"]

    # Structural equality checks retain several complete teacher/student
    # coordinate graphs. Drop them before any four-gradient CAGrad audit so
    # the optimizer smoke measures production memory rather than audit overlap.
    del components, wrapped, noisy
    del archived_patch_prediction, archived_patch_confidence
    cloud_inputs.clear()
    archived_clouds.clear()
    gc.collect()
    jt.sync_all(True)

    teacher_before = parameter_snapshot(list(model.teacher.head.parameters()))
    whole_before = parameter_snapshot(list(model.whole.parameters()))
    student_initial = parameter_snapshot(full_parameters)
    prepare_stage(model, head_parameters)
    head_ids = {id(parameter) for parameter in head_parameters}
    optimizer_a = nn.Adam(head_parameters, lr=4e-5, weight_decay=2e-6)
    cd_task, sf_task, raw_losses, metrics = b24_task_losses(model, batch)
    gradients = task_gradient_pair(cd_task, sf_task, head_parameters, raw_losses=raw_losses)
    if gradients["gram"][0] <= 0.0 or gradients["gram"][2] <= 0.0:
        raise RuntimeError("B24 Stage-A task-specific gradient is zero")
    _, one_step = cagrad_optimizer_step(
        optimizer_a, head_parameters, gradients, clip_norm=2.5
    )
    jt.sync_all(True)
    head_change = max_parameter_change(head_parameters, [
        value for parameter, value in zip(full_parameters, student_initial) if id(parameter) in head_ids
    ])
    frozen_changes = [
        float(np.max(np.abs(parameter.numpy() - initial)))
        for parameter, initial in zip(full_parameters, student_initial)
        if id(parameter) not in head_ids
    ]
    if head_change <= 1e-8 or max(frozen_changes or [0.0]) != 0.0:
        raise RuntimeError("B24 Stage-A scope or active movement audit failed")
    model.student.head.load(args.teacher_ckpt)

    prepare_stage(model, full_parameters)
    optimizer_b = nn.Adam(full_parameters, lr=2.5e-5, weight_decay=3e-6)
    iterator = iter(dataset)
    smoke_started = time.time()
    conflict_cosines = []
    cap_max = 0.0
    first_terms = None
    for step in range(args.smoke_steps):
        try:
            current_batch = next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            current_batch = next(iterator)
        cd_task, sf_task, raw_losses, metrics = b24_task_losses(model, current_batch)
        include_raw = step < int(args.conflict_steps)
        gradients = task_gradient_pair(
            cd_task,
            sf_task,
            full_parameters,
            raw_losses=raw_losses if include_raw else None,
        )
        if gradients["gram"][0] <= 0.0 or gradients["gram"][2] <= 0.0:
            raise RuntimeError("B24 Stage-B task-specific gradient is zero")
        _, report = cagrad_optimizer_step(
            optimizer_b, full_parameters, gradients, clip_norm=2.0
        )
        values = {key: float(value.item()) for key, value in metrics.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError("non-finite B24 200-batch smoke")
        cap_max = max(cap_max, values["cap_ratio_max"])
        if include_raw:
            conflict_cosines.append(
                {
                    "raw": float(gradients["raw"]["cosine"]),
                    "effective": float(gradients["cosine"]),
                }
            )
        if first_terms is None:
            first_terms = {**values, **report}
        del gradients
        jt.sync_all(True)
        if (step + 1) % 20 == 0:
            if not all_finite(full_parameters):
                raise RuntimeError("non-finite B24 parameters during smoke")
            print("B24 Preflight smoke [%d/%d]" % (step + 1, args.smoke_steps), flush=True)
    if not all_finite(full_parameters):
        raise RuntimeError("non-finite B24 parameters after smoke")
    if cap_max > 0.100001:
        raise RuntimeError("B24 cap exceeded during smoke")
    teacher_change = max_parameter_change(list(model.teacher.head.parameters()), teacher_before)
    whole_change = max_parameter_change(list(model.whole.parameters()), whole_before)
    if teacher_change != 0.0 or whole_change != 0.0:
        raise RuntimeError("B24 immutable teacher/whole changed during preflight")

    report = {
        "status": "PASS",
        "version": "B24",
        "cuda_enabled": bool(args.use_cuda),
        "batch_size": int(args.batch_size),
        "train_count": manifest["train_count"],
        "validation_count": manifest["validation_count"],
        "teacher_sha256": actual_sha["teacher20"],
        "full_v67_parameters": parameter_count(full_parameters),
        "head_set_parameters": head_manifest_a["numel"],
        "head_set_repeat_exact": True,
        "teacher_student_parameter_max_abs": parameter_error,
        "teacher_patch_vs_archived_max_abs": teacher_patch_error,
        "student_patch_vs_teacher_max_abs": student_patch_error,
        "confidence_max_abs": confidence_error,
        "independent_archived_confidence_drift": archived_confidence_drift,
        "full_cloud_reproduction": full_cloud_records,
        "reproduction_process_isolated": True,
        "initial_cap_ratio_max": cap_ratio,
        "smoke_cap_ratio_max": cap_max,
        "one_cagrad_step": one_step,
        "stage_a_head_change": head_change,
        "stage_a_frozen_change": max(frozen_changes or [0.0]),
        "teacher_change": teacher_change,
        "whole_change": whole_change,
        "solver_audit": solver_report,
        "autograd_audit": autograd_report,
        "conflict_logging_steps": len(conflict_cosines),
        "conflict_logging_finite": bool(
            all(np.isfinite(item["raw"] + item["effective"]) for item in conflict_cosines)
        ),
        "smoke_steps": int(args.smoke_steps),
        "smoke_seconds": float(time.time() - smoke_started),
        "first_smoke_terms": first_terms,
        "no_double_gradient": True,
        "cagrad_c": CAGRAD_C,
        "cagrad_rescale": 1,
        "cagrad_divisor": 1.16,
        "stage_scopes": {
            "A": "exact archived B20 Stage-C HEAD_SET",
            "B": "all 623663 V67 parameters",
            "C": "exact archived B20 Stage-C HEAD_SET",
        },
    }
    write_json(os.path.join(args.run_root, "preflight_passed.json"), report)
    print("B24_PREFLIGHT_PASS %s" % json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
