import hashlib
import json
import os

import jittor as jt
import numpy as np


PRIMARY_FAMILIES = (
    "laplace_low",
    "laplace_mid",
    "laplace_high",
    "compound",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def stable_hex(salt, value):
    return hashlib.sha256((salt + "|" + value).encode("utf-8")).hexdigest()


def stable_seed(salt, value):
    return int(stable_hex(salt, value)[:8], 16) & 0x7FFFFFFF


def make_validation_manifest(train_list, val_list, output_path):
    train = read_lines(train_list)
    validation = read_lines(val_list)
    if len(train) != 19699 or len(validation) != 100:
        raise RuntimeError(
            "B20 requires 19699 train and 100 validation shapes, got %d/%d"
            % (len(train), len(validation))
        )
    overlap = sorted(set(train).intersection(validation))
    if overlap:
        raise RuntimeError("train/validation overlap: %s" % overlap[:5])
    split_salt = "B14_RSFT_SPLIT_V1_20260814"
    screen_salt = "B14_RSFT_SCREEN_V1"
    if os.path.isfile(output_path):
        manifest = load_json(output_path)
        if (
            manifest.get("split_salt") != split_salt
            or manifest.get("screen_salt") != screen_salt
            or manifest.get("train_count") != 19699
            or manifest.get("validation_count") != 100
        ):
            raise RuntimeError("existing B20 validation manifest is incompatible")
        return manifest
    ordered = sorted(validation, key=lambda item: stable_hex(split_salt, item))
    dev = set(ordered[:60])
    screen_order = sorted(dev, key=lambda item: stable_hex(screen_salt, item))
    screen = set(screen_order[:20])
    entries = []
    for index, relpath in enumerate(ordered):
        family_index = index % len(PRIMARY_FAMILIES)
        entries.append(
            {
                "path": relpath,
                "path_sha256": hashlib.sha256(relpath.encode("utf-8")).hexdigest(),
                "split": "dev" if relpath in dev else "lockbox",
                "screen": relpath in screen,
                "surface_seed": stable_seed(split_salt + "_surface", relpath),
                "primary_family": PRIMARY_FAMILIES[family_index],
                "primary_seed": stable_seed(split_salt + "_primary", relpath),
            }
        )
    manifest = {
        "version": 1,
        "split_salt": split_salt,
        "screen_salt": screen_salt,
        "dev_count": 60,
        "lockbox_count": 40,
        "screen_count": 20,
        "train_count": len(train),
        "validation_count": len(validation),
        "train_validation_overlap": 0,
        "primary_families": list(PRIMARY_FAMILIES),
        "entries": entries,
    }
    write_json(output_path, manifest)
    return manifest


def parameter_count(parameters):
    return int(sum(np.prod(parameter.shape) for parameter in parameters))


def parameter_snapshot(parameters):
    return [parameter.numpy().copy() for parameter in parameters]


def max_parameter_change(parameters, before):
    values = [
        float(np.max(np.abs(parameter.numpy() - initial)))
        for parameter, initial in zip(parameters, before)
    ]
    return max(values) if values else 0.0


def all_finite(parameters):
    return all(np.isfinite(parameter.numpy()).all() for parameter in parameters)


def smooth_positive(value, tau=0.02):
    return float(tau) * jt.nn.softplus(value / float(tau))


def cosine_schedule(elapsed, target, peak, minimum, warmup_fraction=0.05):
    progress = min(1.0, max(0.0, float(elapsed) / max(float(target), 1e-8)))
    warmup = float(warmup_fraction)
    if progress < warmup:
        return 0.1 * float(peak) + 0.9 * float(peak) * progress / warmup
    phase = (progress - warmup) / max(1.0 - warmup, 1e-8)
    cosine = 0.5 * (1.0 + np.cos(np.pi * phase))
    return float(minimum) + (float(peak) - float(minimum)) * float(cosine)


def array_sha256(value):
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
