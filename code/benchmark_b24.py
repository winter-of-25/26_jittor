import argparse
import json
import os
import subprocess
import time

import jittor as jt
import numpy as np
from jittor import nn

from b20_utils import write_json
from b24_cagrad import cagrad_optimizer_step, task_gradient_pair
from b24_model import build_b24, prepare_stage, student_parameters
from train_b24 import add_model_arguments, b24_task_losses, make_dataset


def _gpu_memory_mib():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    return max([int(value.strip()) for value in output if value.strip()] or [0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default="/root")
    parser.add_argument("--train_list", default="/root/datalist/train_b.txt")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=32768)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--patch_ratio", type=float, default=1.2)
    parser.add_argument("--alignment_k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--use_cuda", type=int, default=1)
    add_model_arguments(parser)
    args = parser.parse_args()
    jt.flags.use_cuda = args.use_cuda
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    dataset = make_dataset(args)
    model = build_b24(args)
    parameters = student_parameters(model)
    prepare_stage(model, parameters)
    optimizer = nn.Adam(parameters, lr=2.5e-5, weight_decay=3e-6)
    iterator = iter(dataset)
    started = time.time()
    first = None
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            batch = next(iterator)
        cd_task, surface_task, _, metrics = b24_task_losses(model, batch)
        gradients = task_gradient_pair(cd_task, surface_task, parameters)
        _, cagrad = cagrad_optimizer_step(
            optimizer, parameters, gradients, clip_norm=2.0
        )
        if first is None:
            first = {
                "T_cd": float(cd_task.item()),
                "T_surface": float(surface_task.item()),
                "cd_gradient_norm": float(gradients["gram"][0] ** 0.5),
                "surface_gradient_norm": float(gradients["gram"][2] ** 0.5),
                "cosine": float(gradients["cosine"]),
                "cagrad": cagrad,
                "cap_ratio_max": float(metrics["cap_ratio_max"].item()),
            }
        del gradients
        jt.sync_all(True)
        print(
            "B24Benchmark batch=%d [%d/%d]" % (args.batch_size, step + 1, args.steps),
            flush=True,
        )
    elapsed = time.time() - started
    report = {
        "status": "PASS",
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "elapsed_seconds": float(elapsed),
        "seconds_per_step": float(elapsed / args.steps),
        "samples_per_second": float(args.batch_size * args.steps / elapsed),
        "gpu_memory_mib": int(_gpu_memory_mib()),
        "first_step": first,
    }
    write_json(args.output, report)
    print("B24_BENCHMARK_PASS %s" % json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

