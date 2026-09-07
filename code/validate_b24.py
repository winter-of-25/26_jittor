import argparse
import functools
import os

import jittor as jt

from b20_utils import load_json, sha256_file, write_json
from b24_model import build_b24, load_student, set_inference
from fixed_validation_b24 import B24FixedSuite
from full_cloud_validation_v62_v65 import FullCloudValidationSuite
from train_b24 import EXPECTED_TEACHER_SHA256, add_model_arguments


ACTIVE_STRENGTHS = (0.50, 0.75, 1.00)


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--eligible_json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data_root", default="/root")
    parser.add_argument("--val_list", default="/root/datalist/validate_b.txt")
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--seed_k", type=int, default=6)
    parser.add_argument("--beta", type=float, default=12.0)
    parser.add_argument("--patch_batch", type=int, default=6)
    parser.add_argument("--surface_count", type=int, default=250000)
    parser.add_argument("--use_cuda", type=int, default=1)
    add_model_arguments(parser)
    return parser


def _checkpoint_order(label):
    return int(str(label).replace("C", ""))


def _compare(left, right):
    a = left["record"]
    b = right["record"]
    mean_gap = float(a["mean_delta"]) - float(b["mean_delta"])
    if abs(mean_gap) >= 0.015:
        return -1 if mean_gap > 0 else 1
    tie_a = (
        float(a["worst_family"]),
        float(a["mean_cd_delta"]),
        float(a["mean_p2s_delta"]),
        float(a["median_delta"]),
        float(a["win_rate"]),
        -float(a["mean_movement_rms"]),
        -_checkpoint_order(left["checkpoint_label"]),
        -float(left["strength"]),
    )
    tie_b = (
        float(b["worst_family"]),
        float(b["mean_cd_delta"]),
        float(b["mean_p2s_delta"]),
        float(b["median_delta"]),
        float(b["win_rate"]),
        -float(b["mean_movement_rms"]),
        -_checkpoint_order(right["checkpoint_label"]),
        -float(right["strength"]),
    )
    if tie_a == tie_b:
        return 0
    return -1 if tie_a > tie_b else 1


def _evaluate_checkpoint(model, candidate, suite, strengths, output_path, args):
    keys = ["%.2f" % value for value in strengths]
    if os.path.isfile(output_path):
        cached = load_json(output_path)
        if (
            cached.get("checkpoint_sha256") == candidate["sha256"]
            and all(key in cached.get("strengths", {}) for key in keys)
        ):
            return cached
    load_student(model, candidate["path"])
    set_inference(model)
    records = suite.evaluate(
        model,
        strengths,
        patch_size=args.patch_size,
        seed_k=args.seed_k,
        beta=args.beta,
        patch_batch=args.patch_batch,
        output_cache_dir=os.path.join(
            args.run_root,
            "validation",
            "output_cache",
            candidate["sha256"],
        ),
    )
    value = {
        "version": "B24",
        "checkpoint_label": candidate["label"],
        "checkpoint": candidate["path"],
        "checkpoint_sha256": candidate["sha256"],
        "strengths": records,
    }
    write_json(output_path, value)
    return value


def main():
    args = _parser().parse_args()
    jt.flags.use_cuda = args.use_cuda
    if sha256_file(args.teacher_ckpt) != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("B24 validation teacher SHA mismatch")
    if not os.path.isfile(os.path.join(args.run_root, "training_complete.json")):
        raise RuntimeError("B24 validation requires training_complete.json")
    eligible = load_json(args.eligible_json)["candidates"]
    if [item["label"] for item in eligible] != ["C1", "C2", "C3", "C4"]:
        raise RuntimeError("B24 eligible checkpoint registry mismatch")
    for candidate in eligible:
        if sha256_file(candidate["path"]) != candidate["sha256"]:
            raise RuntimeError("B24 candidate checkpoint hash mismatch")
    model = build_b24(args)
    validation_root = os.path.join(args.run_root, "validation")
    screen_root = os.path.join(validation_root, "screen20")
    full_root = os.path.join(validation_root, "full100")
    os.makedirs(screen_root, exist_ok=True)
    os.makedirs(full_root, exist_ok=True)

    screen_suite = B24FixedSuite(
        args.data_root,
        args.manifest,
        split="all",
        screen_only=True,
        surface_count=args.surface_count,
    )
    screen = []
    for candidate in eligible:
        result = _evaluate_checkpoint(
            model,
            candidate,
            screen_suite,
            ACTIVE_STRENGTHS,
            os.path.join(screen_root, candidate["label"].lower() + ".json"),
            args,
        )
        for strength in ACTIVE_STRENGTHS:
            screen.append(
                {
                    "checkpoint_label": candidate["label"],
                    "checkpoint": candidate["path"],
                    "checkpoint_sha256": candidate["sha256"],
                    "strength": strength,
                    "record": result["strengths"]["%.2f" % strength],
                }
            )
    screen.sort(key=functools.cmp_to_key(_compare))
    top4 = screen[:4]
    write_json(
        os.path.join(screen_root, "ranking.json"),
        {
            "version": "B24",
            "candidate_count": 12,
            "ranking": screen,
            "top4": top4,
        },
    )

    full_suite = B24FixedSuite(
        args.data_root,
        args.manifest,
        split="all",
        screen_only=False,
        surface_count=args.surface_count,
    )
    full = []
    for candidate in eligible:
        selected_strengths = sorted(
            item["strength"]
            for item in top4
            if item["checkpoint_label"] == candidate["label"]
        )
        if not selected_strengths:
            continue
        result = _evaluate_checkpoint(
            model,
            candidate,
            full_suite,
            selected_strengths,
            os.path.join(full_root, candidate["label"].lower() + ".json"),
            args,
        )
        for strength in selected_strengths:
            full.append(
                {
                    "checkpoint_label": candidate["label"],
                    "checkpoint": candidate["path"],
                    "checkpoint_sha256": candidate["sha256"],
                    "strength": strength,
                    "record": result["strengths"]["%.2f" % strength],
                }
            )
    full.sort(key=functools.cmp_to_key(_compare))
    selected = full[0]
    write_json(
        os.path.join(full_root, "ranking.json"),
        {"version": "B24", "candidate_count": len(full), "ranking": full},
    )

    load_student(model, selected["checkpoint"])
    set_inference(model)
    native_suite = FullCloudValidationSuite(
        args.data_root,
        args.val_list,
        max_shapes=6,
        families=("laplace_low", "laplace_mid", "laplace_high", "compound"),
        point_count=50000,
        surface_count=200000,
        seed=624024,
    )
    native = native_suite.evaluate(
        model,
        (0.0, float(selected["strength"])),
        patch_size=args.patch_size,
        seed_k=args.seed_k,
        beta=args.beta,
        patch_batch=args.patch_batch,
    )
    native["base_semantics"] = "exact archived B20 C4 at internal strength 1.25"
    native["active_semantics"] = "bounded CAVR student residual over B20"
    write_json(os.path.join(validation_root, "native24.json"), native)

    record = selected["record"]
    selection = {
        "version": "B24",
        "policy": "best active FULL100 candidate; local sign never gates packaging",
        "checkpoint_label": selected["checkpoint_label"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "outer_strength": float(selected["strength"]),
        "active_strength_required": True,
        "teacher_checkpoint": args.teacher_ckpt,
        "teacher_sha256": sha256_file(args.teacher_ckpt),
        "teacher_internal_strength": 1.25,
        "manifest_sha256": sha256_file(args.manifest),
        "full100": {
            key: record[key]
            for key in (
                "mean_delta",
                "median_delta",
                "mean_cd_delta",
                "mean_p2s_delta",
                "win_rate",
                "worst_family",
                "mean_movement_rms",
            )
        },
        "native24_path": os.path.join(validation_root, "native24.json"),
        "parent_online": {"score": 81.41, "cd_score": 70.10, "p2s_score": 92.71},
        "local_quality_gated_packaging": False,
    }
    selection_path = os.path.join(args.run_root, "selection_locked.json")
    write_json(selection_path, selection)
    write_json(
        os.path.join(args.run_root, "test_access_allowed.json"),
        {
            "status": "PASS_ACTIVE",
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "teacher_sha256": selection["teacher_sha256"],
            "outer_strength": selection["outer_strength"],
            "selection_sha256": sha256_file(selection_path),
        },
    )
    print(
        "B24_SELECTION_LOCKED %s outer=%.2f delta=%+.6f cd=%+.6f p2s=%+.6f"
        % (
            selection["checkpoint_label"],
            selection["outer_strength"],
            record["mean_delta"],
            record["mean_cd_delta"],
            record["mean_p2s_delta"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

