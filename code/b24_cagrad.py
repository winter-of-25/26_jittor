import math

import jittor as jt
import numpy as np


CAGRAD_C = 0.40
CAGRAD_RESCALE_DIVISOR = 1.0 + CAGRAD_C * CAGRAD_C
EPS = 1e-12


def _detached_gradients(loss, parameters, retain_graph):
    graph_gradients = jt.grad(loss, parameters, retain_graph=retain_graph)
    detached = []
    for gradient in graph_gradients:
        gradient.stop_grad()
        detached.append(gradient)
    del graph_gradients
    return detached


def gradient_gram(first, second):
    aa = None
    ab = None
    bb = None
    for left, right in zip(first, second):
        current_aa = (left * left).sum()
        current_ab = (left * right).sum()
        current_bb = (right * right).sum()
        aa = current_aa if aa is None else aa + current_aa
        ab = current_ab if ab is None else ab + current_ab
        bb = current_bb if bb is None else bb + current_bb
    if aa is None:
        raise RuntimeError("CAGrad received an empty parameter list")
    return (
        float(np.float64(aa.item())),
        float(np.float64(ab.item())),
        float(np.float64(bb.item())),
    )


def gram_cosine(aa, ab, bb):
    return float(ab / math.sqrt(max(aa * bb, EPS)))


def solve_two_task_weight(aa, ab, bb, c=CAGRAD_C, iterations=24):
    aa = float(aa)
    ab = float(ab)
    bb = float(bb)
    c = float(c)
    g0_sq = max(0.0, 0.25 * (aa + 2.0 * ab + bb))
    d_dot_g0 = 0.5 * (aa - bb)

    def derivative(weight):
        weight = float(weight)
        d_dot_gw = weight * (aa - ab) + (1.0 - weight) * (ab - bb)
        gw_sq = (
            weight * weight * aa
            + 2.0 * weight * (1.0 - weight) * ab
            + (1.0 - weight) * (1.0 - weight) * bb
        )
        return d_dot_g0 + c * math.sqrt(g0_sq) * d_dot_gw / math.sqrt(
            max(gw_sq, 0.0) + EPS
        )

    if derivative(0.0) >= 0.0:
        return 0.0
    if derivative(1.0) <= 0.0:
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(int(iterations)):
        middle = 0.5 * (lower + upper)
        if derivative(middle) < 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def objective_two_task(weight, aa, ab, bb, c=CAGRAD_C):
    weight = float(weight)
    g0_sq = max(0.0, 0.25 * (aa + 2.0 * ab + bb))
    gw_dot_g0 = 0.5 * (
        weight * (aa + ab) + (1.0 - weight) * (ab + bb)
    )
    gw_sq = (
        weight * weight * aa
        + 2.0 * weight * (1.0 - weight) * ab
        + (1.0 - weight) * (1.0 - weight) * bb
    )
    return gw_dot_g0 + float(c) * math.sqrt(g0_sq * max(gw_sq, 0.0))


def brute_force_weight(aa, ab, bb, c=CAGRAD_C, points=200001):
    weights = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    g0_sq = max(0.0, 0.25 * (aa + 2.0 * ab + bb))
    dot = 0.5 * (weights * (aa + ab) + (1.0 - weights) * (ab + bb))
    norm_sq = (
        weights * weights * aa
        + 2.0 * weights * (1.0 - weights) * ab
        + (1.0 - weights) * (1.0 - weights) * bb
    )
    values = dot + float(c) * math.sqrt(g0_sq) * np.sqrt(np.maximum(norm_sq, 0.0))
    return float(weights[int(np.argmin(values))])


def task_gradient_pair(cd_loss, surface_loss, parameters, raw_losses=None):
    raw = None
    if raw_losses is not None:
        raw_cd = _detached_gradients(raw_losses[0], parameters, retain_graph=True)
        raw_sf = _detached_gradients(raw_losses[1], parameters, retain_graph=True)
        raw_gram = gradient_gram(raw_cd, raw_sf)
        raw = {
            "cd": raw_cd,
            "surface": raw_sf,
            "gram": raw_gram,
            "cosine": gram_cosine(*raw_gram),
        }
    cd = _detached_gradients(cd_loss, parameters, retain_graph=True)
    surface = _detached_gradients(surface_loss, parameters, retain_graph=False)
    gram = gradient_gram(cd, surface)
    return {
        "cd": cd,
        "surface": surface,
        "gram": gram,
        "cosine": gram_cosine(*gram),
        "raw": raw,
    }


def cagrad_direction(gradients, clip_norm, c=CAGRAD_C):
    cd = gradients["cd"]
    surface = gradients["surface"]
    aa, ab, bb = gradients["gram"]
    g0_sq = max(0.0, 0.25 * (aa + 2.0 * ab + bb))
    weight = solve_two_task_weight(aa, ab, bb, c=c)
    gw_sq = max(
        0.0,
        weight * weight * aa
        + 2.0 * weight * (1.0 - weight) * ab
        + (1.0 - weight) * (1.0 - weight) * bb,
    )
    degenerate = math.sqrt(g0_sq) < EPS or math.sqrt(gw_sq) < EPS
    coefficient = (
        0.0
        if degenerate
        else float(c) * math.sqrt(g0_sq) / (math.sqrt(gw_sq) + EPS)
    )
    alpha = (0.5 + coefficient * weight) / (1.0 + float(c) * float(c))
    beta = (0.5 + coefficient * (1.0 - weight)) / (
        1.0 + float(c) * float(c)
    )
    direction_norm_sq = max(
        0.0,
        alpha * alpha * aa + 2.0 * alpha * beta * ab + beta * beta * bb,
    )
    direction_norm = math.sqrt(direction_norm_sq)
    clip_scale = min(1.0, float(clip_norm) / (direction_norm + EPS))
    alpha *= clip_scale
    beta *= clip_scale
    directions = []
    for cd_gradient, surface_gradient in zip(cd, surface):
        value = alpha * cd_gradient + beta * surface_gradient
        value.stop_grad()
        directions.append(value)
    return directions, {
        "aa": aa,
        "ab": ab,
        "bb": bb,
        "cosine": gram_cosine(aa, ab, bb),
        "weight": float(weight),
        "lambda": float(coefficient),
        "rescale_divisor": float(1.0 + float(c) * float(c)),
        "direction_norm_before_clip": float(direction_norm),
        "clip_scale": float(clip_scale),
        "dot_cd_direction": float(alpha * aa + beta * ab),
        "dot_surface_direction": float(alpha * ab + beta * bb),
        "degenerate": bool(degenerate),
    }


def surrogate_loss(parameters, directions):
    value = None
    for parameter, direction in zip(parameters, directions):
        term = (parameter * direction).sum()
        value = term if value is None else value + term
    if value is None:
        raise RuntimeError("cannot build a surrogate for zero parameters")
    return value


def cagrad_optimizer_step(optimizer, parameters, gradients, clip_norm, c=CAGRAD_C):
    directions, report = cagrad_direction(gradients, clip_norm=clip_norm, c=c)
    surrogate = surrogate_loss(parameters, directions)
    optimizer.zero_grad()
    optimizer.step(surrogate)
    return directions, report


def grouped_gram(named_parameters, first, second):
    groups = {}
    for (name, _), left, right in zip(named_parameters, first, second):
        group = name.split(".", 1)[0]
        record = groups.setdefault(group, [0.0, 0.0, 0.0])
        record[0] += float((left * left).sum().item())
        record[1] += float((left * right).sum().item())
        record[2] += float((right * right).sum().item())
    return {
        name: {
            "aa": values[0],
            "ab": values[1],
            "bb": values[2],
            "cosine": gram_cosine(*values),
        }
        for name, values in sorted(groups.items())
    }

