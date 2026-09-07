import argparse
import os

import jittor as jt
import numpy as np

from b20_utils import load_json, read_lines, sha256_file
from b24_model import build_b24, set_inference
from jittor_port_data_base import (
    CompetitionPredictDataset,
    denormalize_unit_sphere,
    normalize_unit_sphere,
)
from orthogonal_backbones_v62_v65 import patch_based_orthogonal
from train_b24 import EXPECTED_TEACHER_SHA256, add_model_arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--data_root", default="/root")
    parser.add_argument("--test_list", default="/root/datalist/test_b.txt")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--seed_k", type=int, default=6)
    parser.add_argument("--beta", type=float, default=12.0)
    parser.add_argument("--patch_batch", type=int, default=6)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--use_cuda", type=int, default=1)
    add_model_arguments(parser)
    args = parser.parse_args()

    required = [
        os.path.join(args.run_root, "training_complete.json"),
        os.path.join(args.run_root, "selection_locked.json"),
        os.path.join(args.run_root, "test_access_allowed.json"),
    ]
    if not all(os.path.isfile(path) for path in required):
        raise RuntimeError("B24 test isolation gate is not open")
    selection = load_json(required[1])
    if selection.get("version") != "B24":
        raise RuntimeError("invalid B24 selection lock")
    outer_strength = float(selection["outer_strength"])
    if outer_strength not in (0.50, 0.75, 1.00):
        raise RuntimeError("B24 final inference requires a locked active outer strength")
    if sha256_file(selection["checkpoint"]) != selection["checkpoint_sha256"]:
        raise RuntimeError("B24 selected student checkpoint hash mismatch")
    if sha256_file(args.teacher_ckpt) != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("B24 teacher checkpoint hash mismatch")
    if selection["teacher_sha256"] != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("B24 selection references the wrong teacher")
    if len(read_lines(args.test_list)) != 200:
        raise RuntimeError("B24 expects exactly 200 B-test shapes")

    jt.flags.use_cuda = args.use_cuda
    model = build_b24(args, student_ckpt=selection["checkpoint"])
    set_inference(model)
    dataset = CompetitionPredictDataset(args.data_root, args.test_list)
    for index, sample in enumerate(dataset, 1):
        relpath = sample["relpath"]
        noisy = sample["pcl_noisy"].astype(np.float32)
        save_path = os.path.join(
            args.output_root,
            "dataset_test_noisy",
            relpath,
            "denoised.npy",
        )
        if args.resume and os.path.isfile(save_path):
            value = np.load(save_path)
            if value.shape == (50000, 3) and np.isfinite(value).all():
                print("B24Infer [%d/%d] skip %s" % (index, len(dataset), relpath), flush=True)
                continue
        normalized, center, scale = normalize_unit_sphere(noisy)
        denoised = patch_based_orthogonal(
            model,
            normalized,
            patch_size=args.patch_size,
            seed_k=args.seed_k,
            beta=args.beta,
            patch_batch=args.patch_batch,
            refinement_strength=outer_strength,
        )
        denoised = denormalize_unit_sphere(denoised, center, scale)
        if denoised.shape != (50000, 3) or not np.isfinite(denoised).all():
            raise RuntimeError("invalid B24 output for %s" % relpath)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, denoised.astype(np.float32))
        print("B24Infer [%d/%d] save %s" % (index, len(dataset), relpath), flush=True)


if __name__ == "__main__":
    main()

