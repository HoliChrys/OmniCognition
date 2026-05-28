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
from typing import List, Optional, Sequence, Tuple

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
    *,
    bidirectional: bool = True,
) -> None:
    """Geometric distillation between two points.

    Positive polarity → pull together. Negative → push apart.

    By default both points move (bidirectional). For Chasles compression
    we pass `bidirectional=False` so that only `point_i` is repositioned
    and `point_j` (the anchor) remains fixed.

    Step size is `1 / (1 + n_obs)` per point — parameter-free, decreases
    with epistemic experience.

    Order with respect to `apply_observation`:
      apply_pull MUST run before apply_observation, because apply_pull
      reads `t_last_obs` to compute lazy decay, and apply_observation
      overwrites it.
    """
    # Lazy-refresh i's stored active (we're going to mutate it)
    decay_i = decay_factor(t_now, point_i.t_last_obs)
    point_i.delta_active = vec_scale(point_i.delta_active, decay_i)

    # Compute direction from CURRENT effective embeddings
    eff_i = effective_embedding(point_i, t_now)
    eff_j = effective_embedding(point_j, t_now)
    raw_dir = vec_sub(eff_j, eff_i)
    if vec_norm(raw_dir) < _EPS:
        return
    direction = vec_normalize(raw_dir)

    sign = 1.0 if polarity > 0 else -1.0
    n_obs_i = point_i.n_corrob + point_i.n_contra
    pas_i = 1.0 / (1.0 + n_obs_i)
    point_i.delta_active = vec_add(
        point_i.delta_active, vec_scale(direction, sign * pas_i)
    )
    new_n_i = n_obs_i + 1
    point_i.delta_latent = vec_scale(
        vec_add(vec_scale(point_i.delta_latent, n_obs_i), point_i.delta_active),
        1.0 / new_n_i,
    )

    if not bidirectional:
        return

    # Symmetric pull on point_j
    decay_j = decay_factor(t_now, point_j.t_last_obs)
    point_j.delta_active = vec_scale(point_j.delta_active, decay_j)
    n_obs_j = point_j.n_corrob + point_j.n_contra
    pas_j = 1.0 / (1.0 + n_obs_j)
    point_j.delta_active = vec_add(
        point_j.delta_active, vec_scale(direction, -sign * pas_j)
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

    Pure geometric primitive : considers EVERY point in `points`,
    regardless of epistemic state. State-based filtering is the job
    of `retrieve` — this function stays unbiased so introspection
    over latent points remains possible.
    """
    scored: List[Tuple[float, "Point"]] = []  # noqa: F821
    for p in points:
        eff = effective_embedding(p, t_now)
        scored.append((cosine(query_embedding, eff), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def retrieve_for_observator(
    query_embedding: Vector,
    points: Sequence["Point"],  # noqa: F821
    k: int,
    t_now: float,
    observator_id: str,
    exclude_states: Optional[set] = None,
) -> List[Tuple[float, "Point"]]:  # noqa: F821
    """Retrieve top-k using a specific observator's view of each
    point's state.

    For points without a view for this observator, falls back to the
    point's default state (the named observator inherits the
    aggregated consensus until it forms its own opinion).
    """
    if exclude_states is None:
        exclude_states = set()
    eligible: List["Point"] = []  # noqa: F821
    for p in points:
        view = p.observator_views.get(observator_id)
        state = view.state if view else p.state
        if state in exclude_states:
            continue
        eligible.append(p)
    return k_nearest(query_embedding, eligible, k, t_now)


def retrieve(
    query_embedding: Vector,
    points: Sequence["Point"],  # noqa: F821
    k: int,
    t_now: float,
    exclude_states: Optional[set] = None,
) -> List[Tuple[float, "Point"]]:  # noqa: F821
    """Top-k retrieval.

    By default applies NO state filter. The system's ZERO-DELETION
    policy is implemented geometrically by `apply_exile` rather than
    by suppression at the retrieval layer : latent points get pushed
    away from the active centroid, so they naturally score below
    active points for typical queries.

    `exclude_states` remains available as an OPT-IN hard filter for
    callers that want explicit suppression for a specific use case.
    """
    if exclude_states is None:
        return k_nearest(query_embedding, points, k, t_now)
    eligible = [p for p in points if p.state not in exclude_states]
    return k_nearest(query_embedding, eligible, k, t_now)


def compute_centroid(
    points: Sequence["Point"],  # noqa: F821
    t_now: float,
    active_only: bool = True,
) -> Optional[Vector]:
    """Centroid of effective embeddings.

    By default restricted to active epistemic states
    (CONJECTURE, CORROBORATED, WARRANTED) so the centroid reflects the
    consensus the system is currently building. Pass active_only=False
    to include latent points in the centroid.
    """
    from metacog.epistemic import EpistemicState
    if active_only:
        active = {
            EpistemicState.CONJECTURE,
            EpistemicState.CORROBORATED,
            EpistemicState.WARRANTED,
        }
        eligible = [p for p in points if p.state in active]
    else:
        eligible = list(points)
    if not eligible:
        return None
    embeddings = [effective_embedding(p, t_now) for p in eligible]
    d = len(embeddings[0])
    return tuple(sum(e[i] for e in embeddings) / len(embeddings) for i in range(d))


def apply_exile(
    point: "Point",  # noqa: F821
    population: Sequence["Point"],  # noqa: F821
    t_now: float,
) -> bool:
    """Push a latent point AWAY from the active centroid.

    This is the GEOMETRIC substitute for suppression. Rather than
    removing or hiding a point that fell into a latent state, we move
    it to the periphery of the manifold. Typical queries (which live
    in the active region) naturally score it lower; queries in the
    latent direction can still surface it — by design.

    Step size : 1 / (1 + n_obs)         — same parameter-free formula
    Direction : (eff_point - centroid)  — orthogonal-ish to consensus

    Returns True if a push was applied, False if no centroid existed
    or the point is already at the centroid (degenerate direction).
    """
    centroid = compute_centroid(population, t_now, active_only=True)
    if centroid is None:
        return False

    decay = decay_factor(t_now, point.t_last_obs)
    point.delta_active = vec_scale(point.delta_active, decay)

    eff_p = effective_embedding(point, t_now)
    raw_dir = vec_sub(eff_p, centroid)
    if vec_norm(raw_dir) < _EPS:
        return False
    direction = vec_normalize(raw_dir)

    n_obs = point.n_corrob + point.n_contra
    pas = 1.0 / (1.0 + n_obs)

    point.delta_active = vec_add(point.delta_active, vec_scale(direction, pas))

    new_n = n_obs + 1
    point.delta_latent = vec_scale(
        vec_add(vec_scale(point.delta_latent, n_obs), point.delta_active),
        1.0 / new_n,
    )
    return True
