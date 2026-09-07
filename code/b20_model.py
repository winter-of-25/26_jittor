import numpy as np

from noise_conditioned_dual_path_v67 import FrozenNoiseConditionedDenoiser
from orthogonal_backbones_v62_v65 import MODES, FrozenOrthogonalDenoiser
from train_joint_point_normal import _build_v45_chain


B20_MODE = "b20_recoverability_native_v67"


def learnable_parameters(module):
    return [
        parameter
        for name, parameter in module.named_parameters()
        if not name.endswith("running_mean") and not name.endswith("running_var")
    ]


def _fresh_normal_initialization(head, seed):
    """Reinitialize nonzero hidden matrices while preserving zero-output heads."""
    rng = np.random.RandomState(int(seed))
    for _, parameter in head.named_parameters():
        value = parameter.numpy()
        if value.ndim < 2 or not np.any(value):
            continue
        fan_in = max(1, int(np.prod(value.shape[1:])))
        initialized = rng.normal(0.0, np.sqrt(2.0 / fan_in), value.shape)
        parameter.assign(initialized.astype(np.float32))
    for layer in (
        head.film,
        head.noise_output,
        head.edge_output,
        head.router_output,
        head.step_control,
    ):
        layer.weight.assign(np.zeros(layer.weight.shape, dtype=np.float32))
        layer.bias.assign(np.zeros(layer.bias.shape, dtype=np.float32))


def build_whole_parent(args):
    parent = FrozenOrthogonalDenoiser(
        _build_v45_chain(args),
        MODES["v65"],
        v45_strength=float(args.v45_strength),
        max_k=32,
        channels=120,
        num_steps=2,
    )
    parent.load(args.v65_ckpt)
    parent.eval()
    parent.set_frozen_eval()
    for parameter in parent.parameters():
        parameter.stop_grad()
    return parent


def build_b20(args, load_head=True):
    model = FrozenNoiseConditionedDenoiser(
        build_whole_parent(args),
        parent_strength=float(args.v65_strength),
        max_k=32,
        channels=128,
        num_steps=3,
    )
    model.mode = B20_MODE
    _fresh_normal_initialization(model.head, int(args.seed))
    if load_head and getattr(args, "head_ckpt", ""):
        model.head.load(args.head_ckpt)
    return model


def head_parameters(model):
    return learnable_parameters(model.head)


def stage_c_parameters(model):
    selected = []
    tokens = ("edge_output", "router_output", "step_control", "noise_output")
    for name, parameter in model.head.named_parameters():
        if any(token in name for token in tokens):
            selected.append(parameter)
    if not selected:
        raise RuntimeError("B20 Stage-C head selection is empty")
    return selected


def prepare_train(model, parameters):
    model.eval()
    model.parent.eval()
    model.parent.set_frozen_eval()
    all_head = head_parameters(model)
    if len(parameters) == len(all_head):
        model.head.train()
    else:
        # Stage C freezes feature towers and their BatchNorm running state.
        model.head.eval()
    for parameter in model.parent.parameters():
        parameter.stop_grad()
    for parameter in all_head:
        parameter.stop_grad()
    for parameter in parameters:
        parameter.start_grad()


def set_frozen_eval(model):
    model.eval()
    model.parent.eval()
    model.parent.set_frozen_eval()
    model.head.eval()


def save_head(model, path):
    model.head.save(path)


def load_head(model, path):
    model.head.load(path)
