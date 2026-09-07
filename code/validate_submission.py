import argparse
import hashlib
import json
import os
import zipfile

import numpy as np


def load_relpaths(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/root/starter_code")
    parser.add_argument("--test_list", default="/root/starter_code/datalist/test.txt")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--zip_path", default="")
    args = parser.parse_args()

    relpaths = load_relpaths(args.test_list)
    expected_members = set()
    total_points = 0
    for relpath in relpaths:
        noisy_path = os.path.join(
            args.data_root, "dataset_test_noisy", relpath, "noisy.npy"
        )
        output_path = os.path.join(
            args.output_root, "dataset_test_noisy", relpath, "denoised.npy"
        )
        if not os.path.isfile(output_path):
            raise FileNotFoundError(output_path)
        noisy = np.load(noisy_path, mmap_mode="r")
        output = np.load(output_path)
        if output.shape != noisy.shape:
            raise ValueError("shape mismatch %s: %s != %s" % (relpath, output.shape, noisy.shape))
        if output.ndim != 2 or output.shape[1] != 3:
            raise ValueError("invalid point shape %s: %s" % (relpath, output.shape))
        if not np.isfinite(output).all():
            raise ValueError("non-finite output: %s" % relpath)
        total_points += int(output.shape[0])
        expected_members.add(
            ("dataset_test_noisy/%s/denoised.npy" % relpath).replace("\\", "/")
        )

    discovered = set()
    output_base = os.path.join(args.output_root, "dataset_test_noisy")
    for root, _, files in os.walk(output_base):
        for name in files:
            if name == "denoised.npy":
                discovered.add(os.path.relpath(os.path.join(root, name), args.output_root).replace("\\", "/"))
    if discovered != expected_members:
        raise ValueError(
            "output member mismatch missing=%s extra=%s"
            % (sorted(expected_members - discovered), sorted(discovered - expected_members))
        )

    result = {
        "files": len(expected_members),
        "total_points": total_points,
        "output_root": os.path.abspath(args.output_root),
    }
    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "r") as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError("corrupt zip member: %s" % corrupt)
            zip_members = {
                name for name in archive.namelist() if name.endswith("denoised.npy")
            }
        if zip_members != expected_members:
            raise ValueError(
                "zip member mismatch missing=%s extra=%s"
                % (sorted(expected_members - zip_members), sorted(zip_members - expected_members))
            )
        result.update(
            {
                "zip_path": os.path.abspath(args.zip_path),
                "zip_bytes": os.path.getsize(args.zip_path),
                "sha256": sha256(args.zip_path),
            }
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
