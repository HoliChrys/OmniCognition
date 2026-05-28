"""
MetaCog-Mem — manifold geometry.

Replaces the previous edge-based graph paradigm with parameter-free
operations on a point cloud:

  - `effective_embedding`  : orig + decayed-active + latent
  - `apply_pull`           : geometric distillation (pull or push)
  - `k_nearest`            : kNN on effective embeddings (the "graph" view)

All operations are deterministic COMPUTATIONs on vectors and counters.
No hyperparameter; step sizes derive from observation counts and elapsed
time.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vector = Tuple[float, ...]

# Smallest vector norm we accept as non-degenerate before refusing to
# normalize. Pure numerical guard, not a tuning knob.
_EPS = 1e-12


def vec_add(a: Vector, b: Vector) -> Vector:
    return tuple(x + y for x, y in zip(a, b))


def vec_sub(a: Vector, b: Vector) -> Vector:
    return tuple(x - y for x, y in zip(a, b))


def vec_scale(a: Vector, s: float) -> Vector:
    return tuple(x * s for x in a)


def vec_norm(a: Vector) -> float:
    return math.sqrt(sum(x * x for x in a))


def vec_normalize(a: Vector) -> Vector:
    n = vec_norm(a)
    if n < _EPS:
        return tuple(0.0 for _ in a)
    return tuple(x / n for x in a)


def cosine(a: Vector, b: Vector) -> float:
    na, nb = vec_norm(a), vec_norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def distance(a: Vector, b: Vector) -> float:
    return vec_norm(vec_sub(a, b))


def decay_factor(t_now: float, t_last_obs: float) -> float:
    """Lazy time decay: 1 / (1 + Δt). COMPUTATION on counters/time."""
    dt = max(0.0, t_now - t_last_obs)
    return 1.0 / (1.0 + dt)


def effective_embedding(point: "Point", t_now: float) -> Vector:  # noqa: F821
    """orig + active(decayed by elapsed time) + latent.

    The stored `delta_active` is the value at `t_last_obs`. Reading at
    a later `t_now` applies the lazy decay multiplicatively.
    """
    decay = decay_factor(t_now, point.t_last_obs)
    active_now = vec_scale(point.delta_active, decay)
    return vec_add(vec_add(point.embedding_orig, active_now), point.delta_latent)


def apply_pull(
    point_i: "Point",  # noqa: F821
    point_j: "Point",  # noqa: F821
    polarity: float,
    t_now: float,
) -> None:
    """Geometric distillation between two points.

    Positive polarity → pull together. Negative → push apart.
    Mutates `delta_active` and `delta_latent` of both points.

    Step size is `1 / (1 + n_obs)` per point — parameter-free, decreases
    with epistemic experience (Bayesian-flavored, no learning rate).

    Order with respect to `apply_observation`:
      apply_pull MUST run before apply_observation, because apply_pull
      reads `t_last_obs` to compute lazy decay, and apply_observation
      overwrites it.
    """
    # 1. Lazy-refresh active offsets (apply accumulated time decay)
    decay_i = decay_factor(t_now, point_i.t_last_obs)
    decay_j = decay_factor(t_now, point_j.t_last_obs)
    point_i.delta_active = vec_scale(point_i.delta_active, decay_i)
    point_j.delta_active = vec_scale(point_j.delta_active, decay_j)

    # 2. Compute direction from CURRENT effective embeddings
    eff_i = effective_embedding(point_i, t_now)
    eff_j = effective_embedding(point_j, t_now)
    raw_dir = vec_sub(eff_j, eff_i)
    if vec_norm(raw_dir) < _EPS:
        # Collision: no defined direction; skip the geometric update,
        # but counters can still be updated by apply_observation.
        return
    direction = vec_normalize(raw_dir)

    # 3. Step size — pure COMPUTATION on counters
    n_obs_i = point_i.n_corrob + point_i.n_contra
    n_obs_j = point_j.n_corrob + point_j.n_contra
    pas_i = 1.0 / (1.0 + n_obs_i)
    pas_j = 1.0 / (1.0 + n_obs_j)

    # 4. Apply pull (or push if polarity < 0)
    sign = 1.0 if polarity > 0 else -1.0
    point_i.delta_active = vec_add(
        point_i.delta_active, vec_scale(direction, sign * pas_i)
    )
    point_j.delta_active = vec_add(
        point_j.delta_active, vec_scale(direction, -sign * pas_j)
    )

    # 5. Update latent as the incremental mean of active over time
    new_n_i = n_obs_i + 1
    point_i.delta_latent = vec_scale(
        vec_add(vec_scale(point_i.delta_latent, n_obs_i), point_i.delta_active),
        1.0 / new_n_i,
    )
    new_n_j = n_obs_j + 1
    point_j.delta_latent = vec_scale(
        vec_add(vec_scale(point_j.delta_latent, n_obs_j), point_j.delta_active),
        1.0 / new_n_j,
    )


def k_nearest(
    query_embedding: Vector,
    points: Sequence["Point"],  # noqa: F821
    k: int,
    t_now: float,
) -> List[Tuple[float, "Point"]]:  # noqa: F821
    """Top-k points by cosine similarity on effective embeddings.

    This is the on-demand graph view: no edges are stored, neighbors
    are computed each time from the current geometry.
    """
    scored: List[Tuple[float, "Point"]] = []  # noqa: F821
    for p in points:
        eff = effective_embedding(p, t_now)
        scored.append((cosine(query_embedding, eff), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
