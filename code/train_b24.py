import argparse
import json
import math
import os
import time

import jittor as jt
import numpy as np
from jittor import nn

from aligned_ot_data_v29 import AlignedOTPatchDataset
from b20_utils import (
    all_finite,
    cosine_schedule,
    max_parameter_change,
    parameter_count,
    parameter_snapshot,
    sha256_file,
    smooth_positive,
    write_json,
)
from b24_cagrad import (
    CAGRAD_C,
    cagrad_direction,
    cagrad_optimizer_step,
    grouped_gram,
    task_gradient_pair,
)
from b24_model import (
    B24_MODE,
    build_b24,
    manifest_fingerprint,
    prepare_stage,
    recovered_head_manifest,
    recovered_head_parameters,
    save_student,
    student_parameters,
)
from train_joint_point_normal import surface_terms_per_sample


EXPECTED_TEACHER_SHA256 = os.environ.get(
    "B24_EXPECTED_TEACHER_SHA256",
    "511ccd594c81cccb1ab769fd52c3049e6c316375ddce91b89204f86fce0ab8ef",
)


def add_model_arguments(parser):
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--baseline_args", required=True)
    parser.add_argument("--v29_ckpt", required=True)
    parser.add_argument("--v33_ckpt", required=True)
    parser.add_argument("--v41_ckpt", required=True)
    parser.add_argument("--v45_ckpt", required=True)
    parser.add_argument("--v65_ckpt", required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--parent_strength", type=float, default=1.25)
    parser.add_argument("--v45_strength", type=float, default=1.25)
    parser.add_argument("--v65_strength", type=float, default=1.50)
    parser.add_argument("--seed", type=int, default=8242401)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--eligible_json", required=True)
    parser.add_argument("--data_root", default="/root")
    parser.add_argument("--train_list", default="/root/datalist/train_b.txt")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=32768)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--patch_ratio", type=float, default=1.2)
    parser.add_argument("--alignment_k", type=int, default=32)
    parser.add_argument("--stage0_batches", type=int, default=256)
    parser.add_argument("--stage_a_hours", type=float, default=2.5)
    parser.add_argument("--stage_a_hard", type=float, default=2.8)
    parser.add_argument("--stage_b_hours", type=float, default=8.5)
    parser.add_argument("--stage_b_hard", type=float, default=9.4)
    parser.add_argument("--stage_b_midpoint", type=float, default=4.3)
    parser.add_argument("--stage_c_hours", type=float, default=2.5)
    parser.add_argument("--stage_c_hard", type=float, default=2.8)
    parser.add_argument("--stage_a_lr", type=float, default=4e-5)
    parser.add_argument("--stage_a_min_lr", type=float, default=5e-6)
    parser.add_argument("--stage_b_lr", type=float, default=2.5e-5)
    parser.add_argument("--stage_b_min_lr", type=float, default=5e-7)
    parser.add_argument("--stage_c_lr", type=float, default=7e-6)
    parser.add_argument("--stage_c_min_lr", type=float, default=2e-7)
    parser.add_argument("--weight_decay_a", type=float, default=2e-6)
    parser.add_argument("--weight_decay_bc", type=float, default=3e-6)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--use_cuda", type=int, default=1)
    add_model_arguments(parser)
    return parser


def make_dataset(args, batch_size=None, shuffle=True):
    return AlignedOTPatchDataset(
        args.data_root,
        args.train_list,
        num_points=args.num_points,
        patch_size=args.patch_size,
        patch_ratio=args.patch_ratio,
        augment=True,
        batch_size=int(batch_size or args.batch_size),
        shuffle=bool(shuffle),
        num_workers=args.num_workers,
        alignment_k=args.alignment_k,
    )


def b24_task_losses(model, batch):
    noisy = batch["pcl_noisy"]
    clean = batch["pcl_clean"]
    components = model.forward_components(noisy)
    candidate = components["candidate"]
    teacher = components["teacher_prediction"]
    whole = components["whole_prediction"]

    candidate_surface, _, candidate_cd = surface_terms_per_sample(candidate, clean)
    with jt.no_grad():
        noisy_surface, _, noisy_cd = surface_terms_per_sample(noisy, clean)
        teacher_surface, _, teacher_cd = surface_terms_per_sample(teacher, clean)
        whole_surface, _, whole_cd = surface_terms_per_sample(whole, clean)

    r_cd = candidate_cd / (noisy_cd + 1e-12)
    r_surface = candidate_surface / (noisy_surface + 1e-12)
    r_cd_teacher = teacher_cd / (noisy_cd + 1e-12)
    r_surface_teacher = teacher_surface / (noisy_surface + 1e-12)
    r_cd_whole = whole_cd / (noisy_cd + 1e-12)
    r_surface_whole = whole_surface / (noisy_surface + 1e-12)

    cd_anchor = jt.minimum(r_cd_teacher, r_cd_whole)
    surface_anchor = jt.minimum(r_surface_teacher, r_surface_whole)
    cd_anchor.stop_grad()
    surface_anchor.stop_grad()
    score = 0.5 * (r_cd + r_surface)
    teacher_score = 0.5 * (r_cd_teacher + r_surface_teacher)
    total_regret = smooth_positive(score - teacher_score, tau=0.020).mean()
    cd_guard = smooth_positive(r_cd - cd_anchor, tau=0.015).mean()
    surface_guard = smooth_positive(
        r_surface - surface_anchor, tau=0.015
    ).mean()
    cd_task = r_cd.mean() + 0.12 * cd_guard + 0.08 * total_regret
    surface_task = (
        r_surface.mean() + 0.16 * surface_guard + 0.08 * total_regret
    )

    delta_norm = jt.sqrt((components["delta"] ** 2).sum(-1) + 1e-12)
    cap = components["cap"].reshape(delta_norm.shape)
    metrics = {
        "T_cd": cd_task,
        "T_sf": surface_task,
        "cd_guard": cd_guard,
        "sf_guard": surface_guard,
        "total_regret": total_regret,
        "r_cd_student": r_cd.mean(),
        "r_cd_b20": r_cd_teacher.mean(),
        "r_cd_whole": r_cd_whole.mean(),
        "r_sf_student": r_surface.mean(),
        "r_sf_b20": r_surface_teacher.mean(),
        "r_sf_whole": r_surface_whole.mean(),
        "student_b20_rms": jt.sqrt((components["delta"] ** 2).mean() + 1e-12),
        "cap_hit_rate": ((delta_norm >= 0.999 * cap) * jt.ones_like(cap)).mean(),
        "cap_ratio_max": (delta_norm / (components["r32"].reshape(delta_norm.shape) + 1e-12)).max(),
    }
    return cd_task, surface_task, (r_cd.mean(), r_surface.mean()), metrics


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "fraction_negative": float(np.mean(array < 0.0)),
    }


def _named_subset(model, parameters):
    ids = {id(parameter) for parameter in parameters}
    return [
        (name, parameter)
        for name, parameter in model.student.head.named_parameters()
        if id(parameter) in ids
    ]


def gradient_conflict_audit(model, dataset, batches, output_path):
    parameters = student_parameters(model)
    # Stage 0 is a measurement at the exact B20 C4 state. Keep every BatchNorm
    # in eval mode so running buffers do not move even though gradients are on.
    model.set_frozen_eval()
    for parameter in parameters:
        parameter.start_grad()
    iterator = iter(dataset)
    raw_cosines = []
    effective_cosines = []
    raw_cd_norms = []
    raw_sf_norms = []
    effective_cd_norms = []
    effective_sf_norms = []
    weights = []
    lambdas = []
    dots_cd = []
    dots_sf = []
    module_samples = {"raw": [], "effective": []}
    degenerate = 0
    named = _named_subset(model, parameters)
    started = time.time()
    for index in range(int(batches)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            batch = next(iterator)
        cd_task, sf_task, raw_losses, _ = b24_task_losses(model, batch)
        gradients = task_gradient_pair(
            cd_task, sf_task, parameters, raw_losses=raw_losses
        )
        raw_aa, _, raw_bb = gradients["raw"]["gram"]
        eff_aa, _, eff_bb = gradients["gram"]
        _, cagrad = cagrad_direction(gradients, clip_norm=2.0, c=CAGRAD_C)
        raw_cosines.append(gradients["raw"]["cosine"])
        effective_cosines.append(gradients["cosine"])
        raw_cd_norms.append(math.sqrt(max(raw_aa, 0.0)))
        raw_sf_norms.append(math.sqrt(max(raw_bb, 0.0)))
        effective_cd_norms.append(math.sqrt(max(eff_aa, 0.0)))
        effective_sf_norms.append(math.sqrt(max(eff_bb, 0.0)))
        weights.append(cagrad["weight"])
        lambdas.append(cagrad["lambda"])
        dots_cd.append(cagrad["dot_cd_direction"])
        dots_sf.append(cagrad["dot_surface_direction"])
        degenerate += int(cagrad["degenerate"])
        if index % 32 == 0:
            module_samples["raw"].append(
                grouped_gram(
                    named,
                    gradients["raw"]["cd"],
                    gradients["raw"]["surface"],
                )
            )
            module_samples["effective"].append(
                grouped_gram(named, gradients["cd"], gradients["surface"])
            )
        del gradients
        jt.sync_all(True)
        if (index + 1) % 16 == 0:
            print(
                "B24 Stage0 [%d/%d] raw_cos=%.5f effective_cos=%.5f"
                % (index + 1, batches, raw_cosines[-1], effective_cosines[-1]),
                flush=True,
            )

    per_module = {}
    for kind, samples in module_samples.items():
        names = sorted({name for sample in samples for name in sample})
        per_module[kind] = {
            name: {
                "mean_cosine": float(
                    np.mean([sample[name]["cosine"] for sample in samples if name in sample])
                ),
                "samples": int(sum(name in sample for sample in samples)),
            }
            for name in names
        }
    report = {
        "version": "B24",
        "status": "PASS",
        "batches": int(batches),
        "updates": 0,
        "elapsed_seconds": float(time.time() - started),
        "raw": {
            "cosine": _summary(raw_cosines),
            "mean_cd_norm": float(np.mean(raw_cd_norms)),
            "mean_surface_norm": float(np.mean(raw_sf_norms)),
        },
        "effective": {
            "cosine": _summary(effective_cosines),
            "mean_cd_norm": float(np.mean(effective_cd_norms)),
            "mean_surface_norm": float(np.mean(effective_sf_norms)),
            "mean_norm_ratio_cd_over_surface": float(
                np.mean(np.asarray(effective_cd_norms) / (np.asarray(effective_sf_norms) + 1e-12))
            ),
        },
        "cagrad": {
            "c": CAGRAD_C,
            "mean_weight": float(np.mean(weights)),
            "mean_lambda": float(np.mean(lambdas)),
            "rescale_divisor": 1.16,
            "mean_dot_cd_direction": float(np.mean(dots_cd)),
            "mean_dot_surface_direction": float(np.mean(dots_sf)),
            "degenerate_count": int(degenerate),
        },
        "per_module": per_module,
    }
    write_json(output_path, report)
    return report


def _metric_values(metrics):
    return {key: float(value.item()) for key, value in metrics.items()}


def _save_candidate(model, args, label, stage, total_hours, stage_hours, steps):
    path = os.path.join(args.save_dir, "b24_%s_cavr.pkl" % label.lower())
    save_student(model, path)
    return {
        "label": label,
        "stage": stage,
        "path": path,
        "sha256": sha256_file(path),
        "train_hours": float(total_hours),
        "stage_hours": float(stage_hours),
        "steps": int(steps),
    }


def _run_stage(
    model,
    dataset,
    stage,
    target_hours,
    hard_hours,
    peak_lr,
    min_lr,
    weight_decay,
    clip_norm,
    log_interval,
    midpoint=None,
):
    parameters = (
        recovered_head_parameters(model) if stage in ("A", "C") else student_parameters(model)
    )
    prepare_stage(model, parameters)
    optimizer = nn.Adam(parameters, lr=peak_lr, weight_decay=weight_decay)
    started = time.time()
    steps = 0
    epochs = 0
    midpoint_done = False
    recent = {}
    while True:
        epochs += 1
        prepare_stage(model, parameters)
        for batch in dataset:
            elapsed = time.time() - started
            if elapsed >= float(target_hours) * 3600.0:
                break
            if elapsed > float(hard_hours) * 3600.0:
                raise RuntimeError("B24 Stage %s exceeded hard budget" % stage)
            optimizer.lr = cosine_schedule(
                elapsed,
                float(target_hours) * 3600.0,
                peak_lr,
                min_lr,
                warmup_fraction=0.05,
            )
            cd_task, sf_task, raw_losses, metrics = b24_task_losses(model, batch)
            include_raw = steps % int(log_interval) == 0
            gradients = task_gradient_pair(
                cd_task,
                sf_task,
                parameters,
                raw_losses=raw_losses if include_raw else None,
            )
            _, cagrad = cagrad_optimizer_step(
                optimizer,
                parameters,
                gradients,
                clip_norm=clip_norm,
                c=CAGRAD_C,
            )
            metric_values = _metric_values(metrics)
            metric_values.update(
                {
                    "effective_cosine": gradients["cosine"],
                    "g_cd_norm": math.sqrt(max(gradients["gram"][0], 0.0)),
                    "g_sf_norm": math.sqrt(max(gradients["gram"][2], 0.0)),
                    "cagrad_w": cagrad["weight"],
                    "cagrad_lambda": cagrad["lambda"],
                    "dot_cd_d": cagrad["dot_cd_direction"],
                    "dot_sf_d": cagrad["dot_surface_direction"],
                    "clip_scale": cagrad["clip_scale"],
                    "raw_cosine": (
                        gradients["raw"]["cosine"] if gradients["raw"] is not None else np.nan
                    ),
                }
            )
            if not np.isfinite(metric_values["T_cd"] + metric_values["T_sf"]):
                raise RuntimeError("non-finite B24 Stage %s task" % stage)
            steps += 1
            for key, value in metric_values.items():
                if np.isfinite(value):
                    recent.setdefault(key, []).append(float(value))
                    if len(recent[key]) > int(log_interval):
                        recent[key].pop(0)
            del gradients
            jt.sync_all(True)
            elapsed = time.time() - started
            if midpoint and not midpoint_done and elapsed >= float(midpoint[0]) * 3600.0:
                midpoint[1](elapsed / 3600.0, steps)
                midpoint_done = True
            if steps % int(log_interval) == 0:
                summary = " ".join(
                    "%s=%.6f" % (key, float(np.mean(values)))
                    for key, values in recent.items()
                )
                print(
                    "B24 stage=%s epoch=%d step=%d time=%.3fh lr=%.3e %s"
                    % (stage, epochs, steps, elapsed / 3600.0, optimizer.lr, summary),
                    flush=True,
                )
        if time.time() - started >= float(target_hours) * 3600.0:
            break
    elapsed = time.time() - started
    if elapsed > float(hard_hours) * 3600.0:
        raise RuntimeError("B24 Stage %s ended beyond hard budget" % stage)
    if midpoint and not midpoint_done:
        midpoint[1](elapsed / 3600.0, steps)
    return {
        "stage": stage,
        "hours": elapsed / 3600.0,
        "steps": int(steps),
        "epochs": int(epochs),
        "trainable_parameters": parameter_count(parameters),
    }


def main():
    args = make_parser().parse_args()
    if (args.num_points, args.patch_size, args.alignment_k) != (32768, 1000, 32):
        raise RuntimeError("B24 geometry invariant mismatch")
    if (
        abs(args.parent_strength - 1.25) > 1e-12
        or abs(args.v45_strength - 1.25) > 1e-12
        or abs(args.v65_strength - 1.50) > 1e-12
    ):
        raise RuntimeError("B24 frozen whole strength mismatch")
    if sha256_file(args.teacher_ckpt) != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("B24 teacher C4 SHA256 mismatch")
    hard_total = args.stage_a_hard + args.stage_b_hard + args.stage_c_hard
    if hard_total > 15.0 + 1e-12:
        raise RuntimeError("B24 hard stage budget exceeds 15 hours")

    jt.flags.use_cuda = args.use_cuda
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    os.makedirs(args.run_root, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = make_dataset(args)
    if len(dataset.relpaths) != 19699:
        raise RuntimeError("B24 did not see all 19699 B-train meshes")
    model = build_b24(args)
    full_parameters = student_parameters(model)
    if parameter_count(full_parameters) != 623663:
        raise RuntimeError("B24 full V67 parameter count mismatch")

    manifest_a = recovered_head_manifest(model)
    manifest_b = recovered_head_manifest(model)
    if manifest_fingerprint(manifest_a) != manifest_fingerprint(manifest_b):
        raise RuntimeError("B24 HEAD_SET recovery was not deterministic")
    write_json(os.path.join(args.run_root, "b24_recovered_head_set.json"), manifest_a)
    write_json(
        os.path.join(args.run_root, "train_args.json"),
        {**vars(args), "mode": B24_MODE, "full_training_items": len(dataset.relpaths)},
    )
    teacher_before = parameter_snapshot(list(model.teacher.head.parameters()))
    whole_before = parameter_snapshot(list(model.whole.parameters()))
    student_before_audit = parameter_snapshot(list(model.student.head.parameters()))

    audit_path = os.path.join(args.run_root, "gradient_conflict_audit.json")
    gradient_conflict_audit(model, dataset, args.stage0_batches, audit_path)
    if max_parameter_change(list(model.student.head.parameters()), student_before_audit) != 0.0:
        raise RuntimeError("B24 Stage0 changed student parameters or running buffers")

    candidates = []
    total_hours = 0.0
    stage_a = _run_stage(
        model,
        dataset,
        "A",
        args.stage_a_hours,
        args.stage_a_hard,
        args.stage_a_lr,
        args.stage_a_min_lr,
        args.weight_decay_a,
        2.5,
        args.log_interval,
    )
    total_hours += stage_a["hours"]
    candidates.append(
        _save_candidate(model, args, "C1", "A_last", total_hours, stage_a["hours"], stage_a["steps"])
    )

    def save_midpoint(stage_hours, steps):
        candidates.append(
            _save_candidate(
                model,
                args,
                "C2",
                "B_mid",
                total_hours + stage_hours,
                stage_hours,
                steps,
            )
        )

    stage_b = _run_stage(
        model,
        dataset,
        "B",
        args.stage_b_hours,
        args.stage_b_hard,
        args.stage_b_lr,
        args.stage_b_min_lr,
        args.weight_decay_bc,
        2.0,
        args.log_interval,
        midpoint=(args.stage_b_midpoint, save_midpoint),
    )
    total_hours += stage_b["hours"]
    candidates.append(
        _save_candidate(model, args, "C3", "B_last", total_hours, stage_b["hours"], stage_b["steps"])
    )

    stage_c = _run_stage(
        model,
        dataset,
        "C",
        args.stage_c_hours,
        args.stage_c_hard,
        args.stage_c_lr,
        args.stage_c_min_lr,
        args.weight_decay_bc,
        1.8,
        args.log_interval,
    )
    total_hours += stage_c["hours"]
    candidates.append(
        _save_candidate(model, args, "C4", "C_last", total_hours, stage_c["hours"], stage_c["steps"])
    )
    candidates.sort(key=lambda item: int(item["label"].replace("C", "")))
    if [item["label"] for item in candidates] != ["C1", "C2", "C3", "C4"]:
        raise RuntimeError("B24 candidate registry is incomplete")
    if max_parameter_change(list(model.teacher.head.parameters()), teacher_before) != 0.0:
        raise RuntimeError("B24 immutable teacher changed")
    if max_parameter_change(list(model.whole.parameters()), whole_before) != 0.0:
        raise RuntimeError("B24 immutable whole parent changed")
    if not all_finite(full_parameters):
        raise RuntimeError("B24 student contains non-finite parameters")
    registry = {"version": "B24", "candidates": candidates}
    write_json(args.eligible_json, registry)
    write_json(
        os.path.join(args.run_root, "training_complete.json"),
        {
            "version": "B24",
            "mode": B24_MODE,
            "batch_size": int(args.batch_size),
            "full_training_items": len(dataset.relpaths),
            "pure_train_hours": float(total_hours),
            "stage0_batches": int(args.stage0_batches),
            "stages": {"A": stage_a, "B": stage_b, "C": stage_c},
            "trainable_parameters": parameter_count(full_parameters),
            "head_set_parameters": manifest_a["numel"],
            "teacher_sha256": sha256_file(args.teacher_ckpt),
            "eligible_checkpoints": candidates,
            "cagrad": {"c": CAGRAD_C, "rescale": 1, "divisor": 1.16},
        },
    )
    print("B24_TRAINING_COMPLETE hours=%.3f" % total_hours, flush=True)


if __name__ == "__main__":
    main()
