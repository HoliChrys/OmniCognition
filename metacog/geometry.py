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


def effective_keyword_embedding(point: "Point", t_now: float) -> Vector:  # noqa: F821
    """Effective KEYWORD embedding for geometric operations.

    Returns keywords_embedding + decayed(delta_active) + delta_latent
    when the point has a keyword embedding ; falls back to the
    content-level effective_embedding otherwise.

    Used by apply_pull, apply_exile, and collision detection — the
    geometric "pull / push / merge" mechanism operates at the entity
    level (keywords) rather than at the surface-form level (content).
    Two points with the same entities but different phrasing are
    treated as attracted ; two points with different entities are
    treated as orthogonal, regardless of how similar their content
    reads on the surface.

    The same delta_active / delta_latent vectors apply to BOTH the
    content embedding and the keyword embedding (they share the
    encoder dimension), so a single Hebbian drift carries through
    both representations consistently.
    """
    if not point.keywords_embedding:
        return effective_embedding(point, t_now)
    decay = decay_factor(t_now, point.t_last_obs)
    active_now = vec_scale(point.delta_active, decay)
    return vec_add(
        vec_add(tuple(point.keywords_embedding), active_now),
        point.delta_latent,
    )


def apply_pull(
    point_i: "Point",  # noqa: F821
    point_j: "Point",  # noqa: F821
    polarity: float,
    t_now: float,
    *,
    bidirectional: bool = True,
) -> None:
    """Geometric distillation between two points OPERATING ON THE
    KEYWORD EMBEDDING.

    Positive polarity → pull together. Negative → push apart.

    By default both points move (bidirectional). For Chasles
    compression we pass `bidirectional=False` so that only `point_i`
    is repositioned and `point_j` (the anchor) remains fixed.

    Step size is `1 / (1 + n_obs)` per point — parameter-free.

    Direction is computed from effective_keyword_embedding so the
    pull acts at the entity level : two points with the same
    keywords are pulled together even if their surface-form content
    differs, and pushed apart pulls them along the keyword axis
    rather than the noisy content axis.
    """
    # Lazy-refresh i's stored active (we're going to mutate it)
    decay_i = decay_factor(t_now, point_i.t_last_obs)
    point_i.delta_active = vec_scale(point_i.delta_active, decay_i)

    eff_i = effective_keyword_embedding(point_i, t_now)
    eff_j = effective_keyword_embedding(point_j, t_now)
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


def geometric_spread(
    seed_points: Sequence["Point"],  # noqa: F821
    all_points: Sequence["Point"],  # noqa: F821
    t_now: float,
) -> List[Tuple[float, "Point"]]:
    """Edge-free analog of HeLa-Mem's spreading activation.

    HeLa propagates base scores through LEARNED Hebbian edges
    (S(v_j) += β·Σ S(v_i)·w_ij, with hyperparameters β and a threshold
    θ). We have no edges — but `apply_pull` already drags co-corroborated
    points together IN THE MANIFOLD, so co-activation is encoded as
    geometric proximity rather than as an edge weight. Spreading is then
    just : for each base seed, gather its manifold neighbours.

    Neighbour membership uses the SAME emergent threshold the collision
    machinery uses — (median − σ) of all pairwise keyword-embedding
    distances — so there is NO new hyperparameter (no β, no θ). A point
    is a spread-neighbour of a seed iff its distance is below that
    population-derived cutoff.

    Returns (distance, point) for every neighbour found, closest first,
    de-duplicated. The seeds themselves are excluded.
    """
    if len(all_points) < 4 or not seed_points:
        return []

    # Emergent threshold over keyword-embedding distances (same statistic
    # as collision_threshold : median − σ).
    embs = {p.id: effective_keyword_embedding(p, t_now) for p in all_points}
    dists: List[float] = []
    pts = list(all_points)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(distance(embs[pts[i].id], embs[pts[j].id]))
    if not dists:
        return []
    dists.sort()
    median = dists[len(dists) // 2]
    mean = sum(dists) / len(dists)
    sigma = math.sqrt(sum((d - mean) ** 2 for d in dists) / len(dists))
    threshold = max(0.0, median - sigma)

    seed_ids = {p.id for p in seed_points}
    found: dict[str, float] = {}
    for seed in seed_points:
        es = embs.get(seed.id)
        if es is None:
            continue
        for p in all_points:
            if p.id in seed_ids:
                continue
            d = distance(es, embs[p.id])
            if d < threshold:
                if p.id not in found or d < found[p.id]:
                    found[p.id] = d
    by_id = {p.id: p for p in all_points}
    ranked = sorted(found.items(), key=lambda x: x[1])
    return [(d, by_id[pid]) for pid, d in ranked]


def retrieve_hybrid(
    query_text: str,
    points: Sequence["Point"],  # noqa: F821
    k: int,
    t_now: float,
    *,
    encoder,
    extractor,
    use_lineage: bool = False,
    use_spreading: bool = False,
    use_fuzzy: bool = True,
    lineage_depth: int = 7,
    pool_per_signal: int = 20,
    rrf_k: int = 60,
    prefer_kind: Optional["PointKind"] = None,  # noqa: F821
    restrict_kind: Optional["PointKind"] = None,  # noqa: F821
    text_index=None,
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

    `prefer_kind` (étape 5 — preference routing) : when set to a
    PointKind, points of that kind that already appear in the
    cosine ∪ BM25 candidate set receive an extra RRF contribution
    ranked by their best existing rank. This is purely additive — no
    hyperparameter, no boost factor, no down-weighting of other kinds.
    Use case : "how do I X" queries pass prefer_kind=PointKind.ACTION
    so ACTIONs already known to be relevant outrank co-relevant FACTs.
    """
    from metacog.bm25 import bm25_score

    # Restrict the candidate set to a single kind when asked (the
    # meta-walk retrieves FACT-only / ACTION-only). Applied before any
    # pool is built so every signal respects it ; lineage stays within
    # the restricted set too.
    if restrict_kind is not None:
        points = [p for p in points if p.kind == restrict_kind]

    points_by_id = {p.id: p for p in points}

    # Phase 1 — cosine on keyword embeddings
    query_kw = extractor.extract(query_text, n=8) if extractor else []
    if query_kw:
        # Symmetric with the points' position-weighted keyword
        # embedding so the cosine respects keyword salience order.
        from metacog.keywords import position_weighted_keyword_embedding
        pwe = position_weighted_keyword_embedding(query_kw, encoder)
        query_kw_emb = pwe if pwe is not None else tuple(encoder.encode(query_text))
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

    # Phase 1b — dense cosine on the FULL-CONTENT effective embedding.
    # Keyword cosine matches at the entity level but discards most of the
    # turn's meaning ; full-content dense cosine is the strong semantic
    # signal (especially with a real sentence encoder). The two are
    # complementary and fused below. COMPUTATION on vectors — A(·) ⊥ P.
    query_content_emb = tuple(encoder.encode(query_text))
    content_pool: List[Tuple[float, "Point"]] = []  # noqa: F821
    for p in points:
        eff = effective_embedding(p, t_now)
        content_pool.append((cosine(query_content_emb, eff), p))
    content_pool.sort(key=lambda x: x[0], reverse=True)
    content_pool = content_pool[:pool_per_signal]

    # Phase 2 — BM25 on raw content text (pure lexical channel).
    # bm25_score now always indexes content tokens — query_keywords are
    # passed only to add stemmed-variant coverage on top of raw tokens.
    bm25_pool = bm25_score(
        query_text, points, k_pool=pool_per_signal,
        query_keywords=query_kw or None,
        text_index=text_index,
    )

    # Phase 2b — fuzzy lexical (Levenshtein) : recovers morphological /
    # spelling variants of rare entities that exact BM25 misses. Edit
    # budget is length-relative (parameter-free).
    fuzzy_pool: List[Tuple[float, "Point"]] = []  # noqa: F821
    if use_fuzzy:
        from metacog.fuzzy import fuzzy_score
        fuzzy_pool = fuzzy_score(query_text, points, k_pool=pool_per_signal)

    # Phase 3 — Reciprocal Rank Fusion across all base signals
    rrf_scores: dict[str, float] = {}
    best_rank_by_id: dict[str, int] = {}
    for pool in (cosine_pool, content_pool, bm25_pool, fuzzy_pool):
        for rank, (_, p) in enumerate(pool):
            rrf_scores[p.id] = rrf_scores.get(p.id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank_by_id[p.id] = min(best_rank_by_id.get(p.id, rank), rank)

    # Étape 5 — preference routing : 3rd RRF signal for matching kind.
    # Points of the preferred kind that are already candidates get an
    # additional 1/(rrf_k + r) contribution, where r is their best rank
    # among the preferred-kind subset. Parameter-free (canonical rrf_k).
    if prefer_kind is not None:
        kind_candidates = [
            (best_rank_by_id[pid], pid)
            for pid in best_rank_by_id
            if points_by_id[pid].kind == prefer_kind
        ]
        kind_candidates.sort(key=lambda x: x[0])
        for kind_rank, (_, pid) in enumerate(kind_candidates):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + kind_rank)

    if not rrf_scores:
        return []

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [pid for pid, _ in fused[:k]]

    # Phase 3.5 — geometric spreading activation (edge-free analog of
    # HeLa's Hebbian spreading). Spread from the fused base top-k to
    # their manifold neighbours and add a RRF signal. Parameter-free :
    # neighbour membership uses the emergent (median − σ) threshold.
    if use_spreading:
        seeds = [points_by_id[pid] for pid in top_ids if pid in points_by_id]
        spread = geometric_spread(seeds, points, t_now)
        for srank, (_dist, p) in enumerate(spread):
            rrf_scores[p.id] = rrf_scores.get(p.id, 0.0) + 1.0 / (rrf_k + srank)
        if spread:
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

    eff_p = effective_keyword_embedding(point, t_now)
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
