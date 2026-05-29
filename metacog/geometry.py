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


def retrieve_with_lineage(
    query_embedding: Vector,
    points: Sequence["Point"],  # noqa: F821
    k: int,
    t_now: float,
    *,
    lineage_depth: int = 7,
    cosine_pool_size: int = 30,
    rrf_k: int = 60,
) -> List[Tuple[float, "Point"]]:  # noqa: F821
    """Hybrid retrieval : cosine kNN + lineage expansion + RRF fusion.

    Phase 1 — cosine kNN over effective embeddings (pool of
              `cosine_pool_size` candidates).
    Phase 2 — for each cosine hit, expand via lineage :
                parents, children, sequence_prev, sequence_next.
              The expansion is breadth-first up to `lineage_depth`.
              A lineage-discovered point inherits the rank of the
              seed that brought it.
    Phase 3 — Reciprocal Rank Fusion :
                score(p) = (1 / (rrf_k + rank_cosine(p)))
                         + (1 / (rrf_k + rank_lineage(p)))

    `rrf_k = 60` is the standard Reciprocal Rank Fusion constant
    (Cormack, Clarke & Buettcher 2009) — not a tuning knob.
    """
    points_by_id = {p.id: p for p in points}

    pool = k_nearest(query_embedding, points, cosine_pool_size, t_now)
    cosine_rank_by_id: dict[str, int] = {
        p.id: i for i, (_, p) in enumerate(pool)
    }

    # BFS expansion with UNCERTAINTY PROPAGATION pruning.
    # Branch stops when the accumulated σ_path exceeds the emergent
    # threshold derived from the population's own σ distribution.
    # Lineage rank is also inflated by σ_path so high-uncertainty
    # nodes drop in the final RRF ranking.
    from metacog.uncertainty import beta_sigma, hop_sigma, prune_threshold

    lineage_rank_by_id: dict[str, int] = {}
    sigma_by_id: dict[str, float] = {}
    pool_len = max(1, len(pool))
    sigma_cutoff = prune_threshold(points)

    for seed_rank, (_, seed) in enumerate(pool):
        sigma_by_id.setdefault(seed.id, beta_sigma(seed))
        frontier = {seed.id}
        for _depth in range(1, lineage_depth + 1):
            next_frontier: set[str] = set()
            for pid in frontier:
                p = points_by_id.get(pid)
                if p is None:
                    continue
                current_sigma = sigma_by_id.get(pid, beta_sigma(p))
                neighbors: list[str] = []
                neighbors.extend(p.parents)
                neighbors.extend(p.children)
                if p.sequence_prev:
                    neighbors.append(p.sequence_prev)
                if p.sequence_next:
                    neighbors.append(p.sequence_next)
                for nid in neighbors:
                    if nid not in points_by_id or nid in lineage_rank_by_id:
                        continue
                    child = points_by_id[nid]
                    hop = hop_sigma(p, child, t_now)
                    propagated = math.sqrt(current_sigma * current_sigma
                                           + hop * hop)
                    # Prune over-uncertain branches
                    if sigma_cutoff is not None and propagated > sigma_cutoff:
                        continue
                    sigma_by_id[nid] = propagated
                    # Rank inflation : seed_rank + σ_propagated × pool_len
                    effective_rank = seed_rank + int(propagated * pool_len)
                    lineage_rank_by_id[nid] = effective_rank
                    next_frontier.add(nid)
            frontier = next_frontier

    candidates = set(cosine_rank_by_id) | set(lineage_rank_by_id)
    rrf_scores: dict[str, float] = {}
    for pid in candidates:
        score = 0.0
        if pid in cosine_rank_by_id:
            score += 1.0 / (rrf_k + cosine_rank_by_id[pid])
        if pid in lineage_rank_by_id:
            score += 1.0 / (rrf_k + lineage_rank_by_id[pid])
        rrf_scores[pid] = score

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
    return [(score, points_by_id[pid]) for pid, score in ranked]


def retrieve_hybrid(
    query_text: str,
    points: Sequence["Point"],  # noqa: F821
    k: int,
    t_now: float,
    *,
    encoder,
    extractor,
    use_lineage: bool = False,
    lineage_depth: int = 7,
    pool_per_signal: int = 14,
    rrf_k: int = 60,
) -> List[Tuple[float, "Point"]]:  # noqa: F821
    """Hybrid retrieval :
      - cosine on KEYWORD embeddings       (entity-level match)
      - BM25 on full content text          (verbatim / rare-token match)
      - Reciprocal Rank Fusion → top-k
      - optional lineage expansion on the fused top-k

    Each signal contributes a pool of `pool_per_signal` candidates,
    then RRF fuses with the canonical constant `rrf_k = 60`. The two
    pools are complementary : keyword cosine catches paraphrased
    references to the same entity ; BM25 catches verbatim dates,
    names, and rare tokens.

    Points without a `keywords_embedding` are skipped from the cosine
    signal but remain candidates via BM25.
    """
    from metacog.bm25 import bm25_score

    points_by_id = {p.id: p for p in points}

    # Phase 1 — cosine on keyword embeddings
    query_kw = extractor.extract(query_text, n=8) if extractor else []
    if query_kw:
        query_kw_text = " ".join(query_kw)
        query_kw_emb = tuple(encoder.encode(query_kw_text))
    else:
        query_kw_emb = tuple(encoder.encode(query_text))

    cosine_pool: List[Tuple[float, "Point"]] = []  # noqa: F821
    for p in points:
        if not p.keywords_embedding:
            continue
        s = cosine(query_kw_emb, tuple(p.keywords_embedding))
        cosine_pool.append((s, p))
    cosine_pool.sort(key=lambda x: x[0], reverse=True)
    cosine_pool = cosine_pool[:pool_per_signal]

    # Phase 2 — BM25 on full content
    bm25_pool = bm25_score(query_text, points, k_pool=pool_per_signal)

    # Phase 3 — Reciprocal Rank Fusion
    rrf_scores: dict[str, float] = {}
    for rank, (_, p) in enumerate(cosine_pool):
        rrf_scores[p.id] = rrf_scores.get(p.id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (_, p) in enumerate(bm25_pool):
        rrf_scores[p.id] = rrf_scores.get(p.id, 0.0) + 1.0 / (rrf_k + rank)

    if not rrf_scores:
        return []

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [pid for pid, _ in fused[:k]]

    # Phase 4 — optional lineage expansion with uncertainty propagation
    if use_lineage:
        from metacog.uncertainty import beta_sigma, hop_sigma, prune_threshold

        lineage_rank_by_id: dict[str, int] = {}
        sigma_by_id: dict[str, float] = {}
        seed_pool_len = max(1, len(top_ids))
        sigma_cutoff = prune_threshold(points)

        for seed_rank, pid in enumerate(top_ids):
            seed_p = points_by_id.get(pid)
            if seed_p is None:
                continue
            sigma_by_id.setdefault(pid, beta_sigma(seed_p))
            frontier = {pid}
            for _depth in range(1, lineage_depth + 1):
                next_frontier: set[str] = set()
                for fid in frontier:
                    p = points_by_id.get(fid)
                    if p is None:
                        continue
                    current_sigma = sigma_by_id.get(fid, beta_sigma(p))
                    neighbors: list[str] = []
                    neighbors.extend(p.parents)
                    neighbors.extend(p.children)
                    if p.sequence_prev:
                        neighbors.append(p.sequence_prev)
                    if p.sequence_next:
                        neighbors.append(p.sequence_next)
                    for nid in neighbors:
                        if nid not in points_by_id or nid in lineage_rank_by_id:
                            continue
                        child = points_by_id[nid]
                        hop = hop_sigma(p, child, t_now)
                        propagated = math.sqrt(current_sigma * current_sigma
                                               + hop * hop)
                        if sigma_cutoff is not None and propagated > sigma_cutoff:
                            continue
                        sigma_by_id[nid] = propagated
                        effective_rank = seed_rank + int(propagated * seed_pool_len)
                        lineage_rank_by_id[nid] = effective_rank
                        next_frontier.add(nid)
                frontier = next_frontier
        for pid, lrank in lineage_rank_by_id.items():
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + lrank)
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = [
        (score, points_by_id[pid])
        for pid, score in fused[:k]
        if pid in points_by_id
    ]
    return results


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
