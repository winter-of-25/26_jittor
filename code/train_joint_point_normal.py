import argparse
import json
import math
import os
import random
import time

import jittor as jt
import numpy as np
from jittor import nn

from aligned_ot_data_v29 import AlignedOTPatchDataset
from competitive_residual_v39_42 import (
    FrozenBaseCompetitor,
    project_normal,
    vector_norm,
)
from cross_patch_consensus_v33 import V29CrossPatchConsensus
from full_cloud_validation_v60 import (
    FullCloudValidationSuite,
)
from hierarchical_residual_v43_46 import (
    FrozenHierarchicalCompetitor,
)
from iterativepfn_v12 import batched_index, knn_indices
from joint_point_normal_v60 import (
    MODE,
    FrozenJointPointNormalDenoiser,
)
from pseudo_query_corrector_v29 import (
    V12PseudoQueryCorrector,
    build_v12_from_args,
)


def vector_charbonnier(value):
    return jt.sqrt(
        (value ** 2).sum(-1) + 1e-8
    ).mean()


def vector_charbonnier_per_sample(value):
    return jt.sqrt(
        (value ** 2).sum(-1) + 1e-8
    ).mean(dim=1)


def surface_terms_per_sample(prediction, clean_dense):
    distances = (
        (
            prediction.unsqueeze(2)
            - clean_dense.unsqueeze(1)
        )
        ** 2
    ).sum(-1)
    pred_nearest, _ = jt.topk(
        distances, k=1, dim=2, largest=False
    )
    clean_nearest, _ = jt.topk(
        distances, k=1, dim=1, largest=False
    )
    batch = prediction.shape[0]
    oneway = pred_nearest.reshape(
        batch, -1
    ).mean(dim=1)
    coverage = clean_nearest.reshape(
        batch, -1
    ).mean(dim=1)
    return oneway, coverage, 0.5 * (
        oneway + coverage
    )


def plane_squared_per_sample(
    prediction, clean_pair, clean_normal
):
    signed = (
        (prediction - clean_pair) * clean_normal
    ).sum(-1)
    return (signed ** 2).mean(dim=1)


def density_spectrum_loss(prediction, clean_pair, k=8):
    indices = knn_indices(clean_pair, int(k))
    prediction_edges = (
        batched_index(prediction, indices)
        - prediction.unsqueeze(2)
    )
    target_edges = (
        batched_index(clean_pair, indices)
        - clean_pair.unsqueeze(2)
    )
    prediction_lengths = vector_norm(
        prediction_edges, keepdims=False
    )
    target_lengths = vector_norm(
        target_edges, keepdims=False
    )
    return jt.abs(
        prediction_lengths - target_lengths
    ).mean()


def unoriented_normal_loss(prediction, target):
    cosine = (prediction * target).sum(-1)
    return (1.0 - cosine ** 2).mean()


def training_loss(model, batch, guard_scale=1.0):
    noisy = batch["pcl_noisy"]
    clean_dense = batch["pcl_clean"]
    clean_ot = batch["pcl_clean_ot"]
    clean_normal = batch["pcl_ot_normal"]
    frozen_outputs = model.frozen_outputs(noisy)
    (
        prediction,
        base_query,
        _,
        v12_query,
        refined,
        auxiliary,
        _,
        _,
    ) = model(
        noisy,
        refinement_strength=1.0,
        frozen_outputs=frozen_outputs,
    )
    target_update = clean_ot - base_query
    predicted_update = refined - base_query
    target_loss = vector_charbonnier(
        predicted_update - target_update
    )
    target_normal_update = project_normal(
        target_update, clean_normal
    )
    predicted_normal_update = project_normal(
        predicted_update, clean_normal
    )
    normal_update_loss = vector_charbonnier(
        predicted_normal_update - target_normal_update
    )
    tangent_update_loss = vector_charbonnier(
        (
            predicted_update - predicted_normal_update
        )
        - (target_update - target_normal_update)
    )
    step_loss = jt.array(0.0)
    for weight, step_prediction in enumerate(
        auxiliary["step_predictions"], 1
    ):
        step_loss = step_loss + float(
            weight
        ) * vector_charbonnier(
            step_prediction - clean_ot
        )
    step_loss = step_loss / float(
        sum(
            range(
                1,
                len(auxiliary["step_predictions"]) + 1,
            )
        )
    )
    raw_normal_loss = unoriented_normal_loss(
        auxiliary["raw_normal"], clean_normal
    )
    filtered_normal_loss = unoriented_normal_loss(
        auxiliary["filtered_normal"], clean_normal
    )
    oneway_values, coverage_values, cd_values = (
        surface_terms_per_sample(prediction, clean_dense)
    )
    plane_values = plane_squared_per_sample(
        prediction, clean_ot, clean_normal
    )
    pair_values = vector_charbonnier_per_sample(
        prediction - clean_ot
    )
    oneway = oneway_values.mean()
    coverage = coverage_values.mean()
    chamfer = cd_values.mean()
    plane = plane_values.mean()
    pair = pair_values.mean()
    density8 = density_spectrum_loss(
        prediction, clean_ot, k=8
    )
    density16 = density_spectrum_loss(
        prediction, clean_ot, k=16
    )
    update_indices = knn_indices(base_query, 8)
    update_neighbors = batched_index(
        predicted_update, update_indices
    )
    update_smoothness = vector_charbonnier(
        update_neighbors
        - predicted_update.unsqueeze(2)
    )
    centroid_drift = (
        predicted_update.mean(dim=1) ** 2
    ).sum(-1).mean()
    route_mean = auxiliary["route"].mean([0, 1, 2])
    route_balance = (
        (route_mean - 1.0 / 3.0) ** 2
    ).mean()
    with jt.no_grad():
        (
            base_oneway,
            base_coverage,
            base_cd,
        ) = surface_terms_per_sample(
            base_query, clean_dense
        )
        base_plane = plane_squared_per_sample(
            base_query, clean_ot, clean_normal
        )
        base_pair = vector_charbonnier_per_sample(
            base_query - clean_ot
        )
        v12_oneway, _, _ = surface_terms_per_sample(
            v12_query, clean_dense
        )
    zeros = jt.zeros_like(cd_values)
    cd_regret = jt.maximum(
        cd_values - 0.999 * base_cd, zeros
    ).mean()
    plane_regret = jt.maximum(
        plane_values - 1.000 * base_plane, zeros
    ).mean()
    coverage_regret = jt.maximum(
        coverage_values - 1.001 * base_coverage, zeros
    ).mean()
    surface_regret = jt.maximum(
        oneway_values - 1.001 * base_oneway, zeros
    ).mean()
    pair_regret = jt.maximum(
        pair_values - 0.999 * base_pair, zeros
    ).mean()
    v12_surface_regret = jt.maximum(
        oneway_values - 0.998 * v12_oneway, zeros
    ).mean()
    guards = float(guard_scale) * (
        520.0 * cd_regret
        + 180.0 * plane_regret
        + 260.0 * coverage_regret
        + 260.0 * surface_regret
        + 0.45 * pair_regret
        + 160.0 * v12_surface_regret
    )
    loss = (
        0.42 * target_loss
        + 0.18 * normal_update_loss
        + 0.10 * tangent_update_loss
        + 0.10 * step_loss
        + 0.008 * raw_normal_loss
        + 0.030 * filtered_normal_loss
        + 80.0 * oneway
        + 145.0 * coverage
        + 180.0 * chamfer
        + 95.0 * plane
        + 0.05 * density8
        + 0.04 * density16
        + 0.020 * update_smoothness
        + 24.0 * centroid_drift
        + 0.006 * route_balance
        + guards
    )
    return loss, {
        "loss": loss,
        "target": target_loss,
        "normal_update": normal_update_loss,
        "tangent_update": tangent_update_loss,
        "raw_normal": raw_normal_loss,
        "filtered_normal": filtered_normal_loss,
        "pair": pair,
        "cd2": chamfer,
        "oneway2": oneway,
        "coverage2": coverage,
        "p2s_proxy2": plane,
        "density8": density8,
        "density16": density16,
        "update_smoothness": update_smoothness,
        "centroid_drift": centroid_drift,
        "route_balance": route_balance,
        "cd_regret": cd_regret,
        "plane_regret": plane_regret,
        "coverage_regret": coverage_regret,
        "surface_regret": surface_regret,
        "pair_regret": pair_regret,
    }


def _build_v45_chain(args):
    with open(
        args.baseline_args, "r", encoding="utf-8"
    ) as handle:
        baseline_args = json.load(handle)
    baseline = build_v12_from_args(baseline_args)
    baseline.load(args.baseline_ckpt)
    v29_model = V12PseudoQueryCorrector(
        baseline, max_k=48, channels=128
    )
    v29_model.load(args.v29_ckpt)
    lower_model = V29CrossPatchConsensus(
        v29_model,
        max_k=48,
        view_k=24,
        channels=128,
    )
    lower_model.load(args.v33_ckpt)
    lower_model.eval()
    parent = FrozenBaseCompetitor(
        lower_model,
        "v33",
        "v41_spectral_residual",
        max_k=32,
        channels=112,
        num_steps=1,
    )
    parent.load(args.v41_ckpt)
    parent.eval()
    v45_model = FrozenHierarchicalCompetitor(
        parent,
        "v41",
        "v45_dual_topology_consensus",
        parent_strength=args.parent_strength,
        max_k=24,
        channels=104,
        num_steps=1,
    )
    v45_model.load(args.v45_ckpt)
    v45_model.eval()
    return v45_model


def build_model(args):
    model = FrozenJointPointNormalDenoiser(
        _build_v45_chain(args),
        v45_strength=args.v45_strength,
        max_k=args.max_k,
        channels=args.channels,
        normal_channels=args.normal_channels,
        num_steps=args.num_steps,
    )
    if args.resume_ckpt:
        model.load(args.resume_ckpt)
    return model


def _save_record(
    model, save_dir, prefix, record, active=False
):
    suffix = "_best_active" if active else "_best"
    json_name = (
        "best_active.json" if active else "best.json"
    )
    model.save(
        os.path.join(
            save_dir, "%s%s.pkl" % (prefix, suffix)
        )
    )
    with open(
        os.path.join(save_dir, json_name),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(record, handle, indent=2, sort_keys=True)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v60")
    parser.add_argument("--mode", default=MODE)
    parser.add_argument(
        "--data_root", default="/root/starter_code"
    )
    parser.add_argument(
        "--train_list",
        default="/root/starter_code/datalist/train.txt",
    )
    parser.add_argument(
        "--val_list",
        default="/root/starter_code/datalist/validate.txt",
    )
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--baseline_args", required=True)
    parser.add_argument("--v29_ckpt", required=True)
    parser.add_argument("--v33_ckpt", required=True)
    parser.add_argument("--v41_ckpt", required=True)
    parser.add_argument("--v45_ckpt", required=True)
    parser.add_argument(
        "--parent_strength", type=float, default=1.25
    )
    parser.add_argument(
        "--v45_strength", type=float, default=1.25
    )
    parser.add_argument("--resume_ckpt", default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=32768)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--patch_ratio", type=float, default=1.2)
    parser.add_argument("--alignment_k", type=int, default=32)
    parser.add_argument("--max_k", type=int, default=40)
    parser.add_argument("--channels", type=int, default=144)
    parser.add_argument(
        "--normal_channels", type=int, default=112
    )
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--min_lr", type=float, default=8e-7)
    parser.add_argument(
        "--weight_decay", type=float, default=3e-6
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=3.0
    )
    parser.add_argument(
        "--target_train_hours", type=float, default=10.8
    )
    parser.add_argument(
        "--hard_train_hours", type=float, default=11.8
    )
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument(
        "--val_interval_hours", type=float, default=2.7
    )
    parser.add_argument(
        "--val_strengths",
        default="0.00,0.35,0.55,0.75,1.00,1.25",
    )
    parser.add_argument("--full_val_shapes", type=int, default=6)
    parser.add_argument(
        "--full_val_families",
        default=(
            "laplace_low,laplace_mid,"
            "laplace_high,compound"
        ),
    )
    parser.add_argument(
        "--full_val_surface_count",
        type=int,
        default=250000,
    )
    parser.add_argument("--seed_k", type=int, default=6)
    parser.add_argument("--beta", type=float, default=12.0)
    parser.add_argument("--patch_batch", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=600728)
    parser.add_argument("--use_cuda", type=int, default=1)
    return parser


def main():
    args = make_parser().parse_args()
    if args.mode != MODE:
        raise ValueError("V60 mode must be %s" % MODE)
    if not (
        1.0
        < args.target_train_hours
        < args.hard_train_hours
        <= 12.0
    ):
        raise ValueError("invalid V60 training time budget")
    jt.flags.use_cuda = args.use_cuda
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    model = build_model(args)
    train_dataset = AlignedOTPatchDataset(
        args.data_root,
        args.train_list,
        num_points=args.num_points,
        patch_size=args.patch_size,
        patch_ratio=args.patch_ratio,
        augment=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        alignment_k=args.alignment_k,
    )
    validation_suite = FullCloudValidationSuite(
        args.data_root,
        args.val_list,
        max_shapes=args.full_val_shapes,
        families=tuple(
            value
            for value in args.full_val_families.split(",")
            if value
        ),
        point_count=50000,
        surface_count=args.full_val_surface_count,
        seed=600060,
    )
    print(
        "V60 validation suite %s"
        % json.dumps(
            validation_suite.describe(), sort_keys=True
        ),
        flush=True,
    )
    optimizer = nn.Adam(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    prefix = args.version.lower()
    with open(
        os.path.join(
            args.save_dir, "%s_args.json" % prefix
        ),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    parameter_count = int(
        sum(
            np.prod(parameter.shape)
            for parameter in model.trainable_parameters()
        )
    )
    total_batches = int(
        math.ceil(
            len(train_dataset.relpaths)
            / float(args.batch_size)
        )
    )
    strengths = tuple(
        float(value)
        for value in args.val_strengths.split(",")
    )
    print(
        "V60 mode=%s trainable_parameters=%d "
        "train_items=%d batch=%d target=%.2fh hard=%.2fh"
        % (
            args.mode,
            parameter_count,
            len(train_dataset.relpaths),
            args.batch_size,
            args.target_train_hours,
            args.hard_train_hours,
        ),
        flush=True,
    )
    train_seconds = 0.0
    epoch_durations = []
    next_validation_hours = 0.0
    best_score = -float("inf")
    best_active_score = -float("inf")
    final_record = None
    final_epoch = 0
    for epoch in range(1, args.max_epochs + 1):
        train_hours = train_seconds / 3600.0
        if train_hours >= args.target_train_hours:
            break
        if epoch_durations:
            estimate = (
                np.mean(epoch_durations[-3:]) / 3600.0
            )
            if (
                train_hours >= 0.85 * args.target_train_hours
                and train_hours + 1.05 * estimate
                > args.hard_train_hours
            ):
                print(
                    "Deadline guard before epoch %d: "
                    "train=%.3fh estimate=%.3fh"
                    % (epoch, train_hours, estimate),
                    flush=True,
                )
                break
        progress = min(
            1.0,
            train_hours
            / max(args.target_train_hours, 1e-8),
        )
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        optimizer.lr = args.min_lr + (
            args.lr - args.min_lr
        ) * cosine
        if epoch == 1:
            optimizer.lr = min(
                optimizer.lr, 0.5 * args.lr
            )
        model.train()
        model.set_frozen_eval()
        epoch_started = time.time()
        metrics_log = {}
        batches = 0
        partial_epoch = False
        for batch in train_dataset:
            current_epoch_seconds = (
                time.time() - epoch_started
            )
            if (
                train_seconds + current_epoch_seconds
                >= args.hard_train_hours * 3600.0
            ):
                partial_epoch = True
                print(
                    "Hard training deadline at batch boundary",
                    flush=True,
                )
                break
            guard_scale = min(
                1.0,
                max(
                    0.0,
                    (
                        train_seconds
                        + current_epoch_seconds
                    )
                    / 3600.0
                    / 1.0,
                ),
            )
            loss, metrics = training_loss(
                model, batch, guard_scale=guard_scale
            )
            optimizer.backward(loss)
            optimizer.clip_grad_norm(args.max_grad_norm)
            optimizer.step()
            batches += 1
            for key, value in metrics.items():
                metrics_log.setdefault(key, []).append(
                    float(value.item())
                )
            if batches % args.log_interval == 0:
                elapsed = time.time() - epoch_started
                eta = (
                    elapsed / batches
                ) * (total_batches - batches)
                print(
                    "Progress epoch=%d batch=%d/%d "
                    "step=%.3fs eta=%.1fmin train=%.3fh "
                    "loss=%.6f"
                    % (
                        epoch,
                        batches,
                        total_batches,
                        elapsed / batches,
                        eta / 60.0,
                        (
                            train_seconds + elapsed
                        )
                        / 3600.0,
                        float(loss.item()),
                    ),
                    flush=True,
                )
        epoch_duration = time.time() - epoch_started
        train_seconds += epoch_duration
        if partial_epoch:
            break
        epoch_durations.append(epoch_duration)
        final_epoch = epoch
        summary = " ".join(
            "%s=%.6f" % (key, np.mean(values))
            for key, values in metrics_log.items()
        )
        print(
            "Epoch [%d] lr=%.3e time=%.1fs train=%.3fh %s"
            % (
                epoch,
                float(optimizer.lr),
                epoch_duration,
                train_seconds / 3600.0,
                summary,
            ),
            flush=True,
        )
        model.save(
            os.path.join(
                args.save_dir, "%s_last.pkl" % prefix
            )
        )
        average_epoch_hours = (
            np.mean(epoch_durations[-3:]) / 3600.0
        )
        near_end = (
            train_seconds / 3600.0
            + 1.05 * average_epoch_hours
            >= args.target_train_hours
        )
        should_validate = (
            train_seconds / 3600.0
            >= next_validation_hours
            or near_end
        )
        if should_validate:
            epoch_path = os.path.join(
                args.save_dir,
                "%s_epoch%03d.pkl" % (prefix, epoch),
            )
            model.save(epoch_path)
            record = validation_suite.evaluate(
                model,
                strengths,
                patch_size=args.patch_size,
                seed_k=args.seed_k,
                beta=args.beta,
                patch_batch=args.patch_batch,
            )
            record.update(
                {
                    "epoch": epoch,
                    "train_hours": train_seconds / 3600.0,
                    "mode": args.mode,
                    "train_items": len(
                        train_dataset.relpaths
                    ),
                }
            )
            final_record = record
            next_validation_hours = (
                train_seconds / 3600.0
                + args.val_interval_hours
            )
            print(
                "Validation %s"
                % json.dumps(record, sort_keys=True),
                flush=True,
            )
            with open(
                os.path.join(
                    args.save_dir, "validation.jsonl"
                ),
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(record, sort_keys=True)
                    + "\n"
                )
            score = float(record["selection_score"])
            if score > best_score:
                best_score = score
                _save_record(
                    model,
                    args.save_dir,
                    prefix,
                    record,
                    active=False,
                )
            active_score = float(
                record["active_selection_score"]
            )
            if active_score > best_active_score:
                best_active_score = active_score
                _save_record(
                    model,
                    args.save_dir,
                    prefix,
                    record,
                    active=True,
                )
        if near_end:
            break
    train_hours = train_seconds / 3600.0
    if final_record is None:
        raise RuntimeError("V60 produced no validation record")
    if train_hours > 12.0:
        raise RuntimeError(
            "V60 exceeded 12 training hours: %.3f"
            % train_hours
        )
    active_path = os.path.join(
        args.save_dir, "%s_best_active.pkl" % prefix
    )
    if not os.path.exists(active_path):
        raise RuntimeError(
            "V60 produced no active checkpoint"
        )
    completion = {
        "version": args.version,
        "mode": args.mode,
        "final_epoch": final_epoch,
        "train_hours": train_hours,
        "train_items": len(train_dataset.relpaths),
        "batch_size": args.batch_size,
        "trainable_parameters": parameter_count,
        "best_selection_score": best_score,
        "best_active_selection_score": best_active_score,
    }
    with open(
        os.path.join(
            args.save_dir, "training_complete.json"
        ),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            completion, handle, indent=2, sort_keys=True
        )
    print(
        "TRAINING_COMPLETE %s"
        % json.dumps(completion, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
