import json
import os

import jittor as jt
import numpy as np
from jittor import nn

from b20_model import (
    B20_MODE,
    build_whole_parent,
    head_parameters as archived_head_parameters,
    stage_c_parameters as archived_stage_c_parameters,
)
from competitive_residual_v39_42 import bounded_vector, vector_norm
from iterativepfn_v12 import batched_index
from noise_conditioned_dual_path_v67 import FrozenNoiseConditionedDenoiser


B24_MODE = "b24_cavr_v2"
INTERNAL_B20_STRENGTH = 1.25
B20_MODEL_BUILD_SEED = 8206701


class CAVRDenoiser(nn.Module):
    """B20 teacher plus an exact-copy V67 student with a bounded residual."""

    def __init__(self, whole, teacher_ckpt, student_ckpt=None):
        super().__init__()
        self.whole = whole
        self.teacher = FrozenNoiseConditionedDenoiser(
            whole,
            parent_strength=1.50,
            max_k=32,
            channels=128,
            num_steps=3,
        )
        self.student = FrozenNoiseConditionedDenoiser(
            whole,
            parent_strength=1.50,
            max_k=32,
            channels=128,
            num_steps=3,
        )
        self.teacher.mode = B20_MODE
        self.student.mode = B20_MODE
        self.mode = B24_MODE
        self.teacher.head.load(teacher_ckpt)
        self.student.head.load(student_ckpt or teacher_ckpt)
        self.set_frozen_eval()

    def set_frozen_eval(self):
        self.eval()
        self.whole.eval()
        self.whole.set_frozen_eval()
        self.teacher.eval()
        self.teacher.head.eval()
        self.student.eval()
        self.student.head.eval()
        for parameter in self.whole.parameters():
            parameter.stop_grad()
        for parameter in self.teacher.head.parameters():
            parameter.stop_grad()

    def _whole_outputs(self, noisy):
        self.whole.eval()
        self.whole.set_frozen_eval()
        with jt.no_grad():
            lower = self.whole.frozen_outputs(noisy)
            return self.whole(
                noisy,
                refinement_strength=1.50,
                frozen_outputs=lower,
            )

    @staticmethod
    def _initial_r32(noisy, student_output):
        indices = student_output[5]["step_diagnostics"][0]["indices"]
        relative = batched_index(noisy, indices) - noisy.unsqueeze(2)
        radius = vector_norm(relative).max(dim=2)
        radius.stop_grad()
        return radius

    def forward_components(self, noisy):
        whole_outputs = self._whole_outputs(noisy)
        with jt.no_grad():
            teacher_output = self.teacher(
                noisy,
                refinement_strength=INTERNAL_B20_STRENGTH,
                frozen_outputs=whole_outputs,
            )
        student_output = self.student(
            noisy,
            refinement_strength=INTERNAL_B20_STRENGTH,
            frozen_outputs=whole_outputs,
        )
        teacher_prediction = teacher_output[0]
        student_raw = student_output[0]
        r32 = self._initial_r32(noisy, student_output)
        cap = 0.10 * r32
        capped_delta = bounded_vector(student_raw - teacher_prediction, cap)
        candidate = teacher_prediction + capped_delta
        return {
            "whole_outputs": whole_outputs,
            "teacher_output": teacher_output,
            "student_output": student_output,
            "whole_prediction": whole_outputs[0],
            "teacher_prediction": teacher_prediction,
            "student_raw": student_raw,
            "candidate": candidate,
            "delta": capped_delta,
            "cap": cap,
            "r32": r32,
        }

    def execute(self, noisy, refinement_strength=1.0, frozen_outputs=None):
        if frozen_outputs is not None:
            raise RuntimeError("B24 owns the shared whole forward; external frozen_outputs are forbidden")
        components = self.forward_components(noisy)
        teacher_output = components["teacher_output"]
        teacher_prediction = components["teacher_prediction"]
        candidate = components["candidate"]
        strength = float(refinement_strength)
        prediction = teacher_prediction + strength * (
            candidate - teacher_prediction
        )
        auxiliary = dict(components["student_output"][5])
        auxiliary.update(
            {
                "stitch_confidence": teacher_output[5]["stitch_confidence"],
                "parent_stitch_confidence": teacher_output[5]["stitch_confidence"],
                "b24_delta": components["delta"],
                "b24_cap": components["cap"],
                "b24_r32": components["r32"],
                "teacher_prediction": teacher_prediction,
                "whole_prediction": components["whole_prediction"],
                "student_raw": components["student_raw"],
                "internal_b20_strength": INTERNAL_B20_STRENGTH,
            }
        )
        return (
            prediction,
            teacher_prediction,
            teacher_output[2],
            teacher_output[3],
            candidate,
            auxiliary,
            components["whole_outputs"],
            teacher_output,
        )


def build_b24(args, student_ckpt=""):
    # Several historical lineage modules retain deterministic, non-checkpointed
    # initialization state. Recreate the exact B20 construction seed, then
    # restore the B24 experiment seed before any data iteration or update.
    np.random.seed(B20_MODEL_BUILD_SEED)
    jt.set_global_seed(B20_MODEL_BUILD_SEED)
    teacher_ckpt = os.path.abspath(args.teacher_ckpt)
    whole = build_whole_parent(args)
    model = CAVRDenoiser(whole, teacher_ckpt, student_ckpt or teacher_ckpt)
    experiment_seed = int(getattr(args, "seed", B20_MODEL_BUILD_SEED))
    np.random.seed(experiment_seed)
    jt.set_global_seed(experiment_seed)
    return model


def student_parameters(model):
    return archived_head_parameters(model.student)


def recovered_head_parameters(model):
    # Execute the archived Stage-C selector rather than recreating its name rule.
    return archived_stage_c_parameters(model.student)


def recovered_head_manifest(model):
    selected = recovered_head_parameters(model)
    selected_ids = {id(parameter) for parameter in selected}
    records = []
    for name, parameter in model.student.head.named_parameters():
        if id(parameter) in selected_ids:
            records.append(
                {
                    "name": name,
                    "shape": [int(value) for value in parameter.shape],
                    "numel": int(np.prod(parameter.shape)),
                }
            )
    if len(records) != len(selected):
        raise RuntimeError("could not map every archived Stage-C parameter to a name")
    return {
        "source": "b20_model.stage_c_parameters executed on student24",
        "count": len(records),
        "numel": int(sum(item["numel"] for item in records)),
        "parameters": records,
    }


def manifest_fingerprint(manifest):
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def prepare_stage(model, parameters):
    model.set_frozen_eval()
    all_student = student_parameters(model)
    for parameter in all_student:
        parameter.stop_grad()
    if len(parameters) == len(all_student):
        model.student.head.train()
    else:
        model.student.head.eval()
    for parameter in parameters:
        parameter.start_grad()


def set_inference(model):
    model.set_frozen_eval()
    for parameter in student_parameters(model):
        parameter.stop_grad()


def save_student(model, path):
    model.student.head.save(path)


def load_student(model, path):
    model.student.head.load(path)
