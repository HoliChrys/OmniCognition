"""
Tests for the ZERO-DELETION policy.

Properties :
  P1   No code path ever reduces the size of the points cloud.
  P2   An INVALID point persists in the cloud (just exiled from recall).
  P3   `retrieve()` excludes INVALID and DEPRECATED by default.
  P4   `k_nearest()` (raw primitive) is unbiased — returns all points.
  P5   `retrieve(..., exclude_states=set())` can surface latent points
       for introspection.
  P6   RESURRECTION : an INVALID point that gets new corroborations
       transitions back to an active state. Memory is reversible.
"""

from __future__ import annotations

from metacog import (
    EpistemicState,
    Observation,
    Point,
    SourceClass,
    apply_observation,
    compute_centroid,
    detect_collisions,
    distance,
    effective_embedding,
    k_nearest,
    process_observation,
    resolve_collision,
    retrieve,
    sleep_cycle_collisions,
)


# ---------------------------------------------------------------------------
# Stubs (reused from test_collision style)
# ---------------------------------------------------------------------------


class _BagOfWordsEncoder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self._cache: dict = {}

    def _word_vec(self, w: str):
        if w not in self._cache:
            h = abs(hash(("bow", w)))
            self._cache[w] = tuple(((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self.dim))
        return self._cache[w]

    def encode(self, text: str):
        words = text.lower().split()
        if not words:
            return tuple(0.0 for _ in range(self.dim))
        acc = [0.0] * self.dim
        for w in words:
            wv = self._word_vec(w)
            for i in range(self.dim):
                acc[i] += wv[i]
        return tuple(v / len(words) for v in acc)


class _SetOpsLLM:
    def extract_common(self, a: str, b: str) -> str:
        wa = a.lower().split()
        wb = set(b.lower().split())
        seen = set()
        out = []
        for w in wa:
            if w in wb and w not in seen:
                out.append(w)
                seen.add(w)
        return " ".join(out)

    def remove_overlap(self, full: str, common: str) -> str:
        cset = set(common.lower().split())
        return " ".join(w for w in full.lower().split() if w not in cset)


def _make_invalid_point(encoder, pid: str, content: str) -> Point:
    p = Point(id=pid, content=content, embedding_orig=encoder.encode(content))
    for i in range(3):
        obs = Observation(
            source=SourceClass.OBSERVER,
            signal_type="contradiction",
            polarity=-1.0,
            target_node_ids=[pid],
            timestamp=float(i),
        )
        apply_observation(p, obs, population=[p])
    return p


# ---------------------------------------------------------------------------
# P1 — no deletion across many operations
# ---------------------------------------------------------------------------


def test_no_deletion_after_full_lifecycle():
    enc = _BagOfWordsEncoder()
    llm = _SetOpsLLM()
    points = [
        Point(id=f"P{i}", content=f"unique content {i} alpha", embedding_orig=enc.encode(f"unique content {i} alpha"))
        for i in range(8)
    ]
    initial_ids = {p.id for p in points}

    # 50 random-ish observations
    for t in range(50):
        a = points[t % len(points)]
        b = points[(t * 3 + 1) % len(points)]
        polarity = 1.0 if t % 3 != 0 else -1.0
        obs = Observation(
            source=SourceClass.COMPUTATION,
            signal_type="corroboration" if polarity > 0 else "contradiction",
            polarity=polarity,
            target_node_ids=[a.id, b.id],
            timestamp=float(t),
        )
        process_observation(points, obs, population=points)

    # Run a couple of sleep cycles
    for cycle_t in (100.0, 200.0):
        sleep_cycle_collisions(points, llm, enc, t_now=cycle_t)

    # All original ids are still present (none deleted)
    current_ids = {p.id for p in points}
    assert initial_ids.issubset(current_ids)
    # The cloud may have GROWN (children from collisions) — but it
    # NEVER shrinks.
    assert len(points) >= 8


# ---------------------------------------------------------------------------
# P2 — INVALID persists
# ---------------------------------------------------------------------------


def test_invalid_point_remains_in_cloud():
    enc = _BagOfWordsEncoder()
    pop = [Point(id=f"P{i}", content=f"c{i}", embedding_orig=enc.encode(f"c{i}")) for i in range(5)]
    bad = _make_invalid_point(enc, "BAD", "this fact is contested")
    pop.append(bad)

    # The bad point must have reached INVALID
    assert bad.state == EpistemicState.INVALID
    # AND still be in the cloud
    assert bad in pop
    assert any(p.id == "BAD" for p in pop)


# ---------------------------------------------------------------------------
# P3, P4, P5 — retrieve vs k_nearest behavior
# ---------------------------------------------------------------------------


def test_retrieve_does_not_filter_by_default():
    """The default retrieve() applies NO state filter. ZERO suppression.

    Invalid and Deprecated points may surface — but the geometric exile
    mechanism pushes them away from the active centroid so they tend to
    score lower for typical queries."""
    enc = _BagOfWordsEncoder()
    pop = [
        Point(id="OK1", content="active memory one",
              embedding_orig=enc.encode("active memory one")),
    ]
    bad = _make_invalid_point(enc, "BAD", "active memory contested")
    pop.append(bad)

    results = retrieve(enc.encode("active memory"), pop, k=10, t_now=10.0)
    ids = {p.id for _, p in results}
    # By default, BAD is NOT filtered out — it just lives in a latent
    # region of the manifold.
    assert "BAD" in ids
    assert "OK1" in ids


def test_retrieve_filter_is_opt_in():
    """When the caller explicitly wants hard suppression, it must be
    requested via `exclude_states`."""
    enc = _BagOfWordsEncoder()
    bad = _make_invalid_point(enc, "BAD", "active")
    pop = [
        Point(id="OK", content="active",
              embedding_orig=enc.encode("active")),
        bad,
    ]
    results = retrieve(
        enc.encode("active"), pop, k=10, t_now=10.0,
        exclude_states={EpistemicState.INVALID, EpistemicState.DEPRECATED},
    )
    ids = {p.id for _, p in results}
    assert "BAD" not in ids
    assert "OK" in ids


def test_k_nearest_remains_unbiased():
    enc = _BagOfWordsEncoder()
    pop = [
        Point(id="OK", content="active",
              embedding_orig=enc.encode("active")),
    ]
    bad = _make_invalid_point(enc, "BAD", "active again")
    pop.append(bad)

    results = k_nearest(enc.encode("active"), pop, k=10, t_now=10.0)
    ids = {p.id for _, p in results}
    # raw primitive returns BOTH — state-blind
    assert "OK" in ids
    assert "BAD" in ids


def test_exile_pushes_invalid_away_from_active_centroid():
    """Geometric proof : after invalidation, the point's effective
    embedding is FARTHER from the active centroid than equivalent
    active points."""
    enc = _BagOfWordsEncoder()
    # A cluster of active points
    pop = [
        Point(id=f"A{i}", content=f"alpha topic {i}",
              embedding_orig=enc.encode(f"alpha topic {i}"))
        for i in range(5)
    ]
    # An additional point that will be invalidated (sits in the same
    # region initially)
    bad = Point(id="BAD", content="alpha topic disputed",
                embedding_orig=enc.encode("alpha topic disputed"))
    pop.append(bad)

    # Before any contradiction, BAD sits near the cluster
    centroid_pre = compute_centroid(pop, t_now=0.0)
    eff_bad_pre = effective_embedding(bad, 0.0)
    dist_pre = distance(eff_bad_pre, centroid_pre)

    # Three contradictions push it to INVALID + trigger exile pushes
    for t in range(3):
        obs = Observation(
            source=SourceClass.OBSERVER,
            signal_type="contradiction",
            polarity=-1.0,
            target_node_ids=["BAD"],
            timestamp=float(t),
        )
        apply_observation(bad, obs, population=pop)

    assert bad.state == EpistemicState.INVALID
    # Re-measure : centroid is now of ACTIVE points only (BAD is excluded)
    centroid_post = compute_centroid(pop, t_now=3.0)
    eff_bad_post = effective_embedding(bad, 3.0)
    dist_post = distance(eff_bad_post, centroid_post)

    assert dist_post > dist_pre


def test_exiled_points_tend_to_rank_lower_for_active_queries():
    """For a query in the active region, exiled points score below the
    active ones (geometrically, not by hard filter)."""
    enc = _BagOfWordsEncoder()
    pop = [
        Point(id=f"A{i}", content=f"alpha topic {i}",
              embedding_orig=enc.encode(f"alpha topic {i}"))
        for i in range(5)
    ]
    bad = Point(id="BAD", content="alpha topic disputed",
                embedding_orig=enc.encode("alpha topic disputed"))
    pop.append(bad)
    for t in range(3):
        obs = Observation(
            source=SourceClass.OBSERVER,
            signal_type="contradiction",
            polarity=-1.0,
            target_node_ids=["BAD"],
            timestamp=float(t),
        )
        apply_observation(bad, obs, population=pop)
    assert bad.state == EpistemicState.INVALID

    # Active-region query
    query = enc.encode("alpha topic")
    results = retrieve(query, pop, k=len(pop), t_now=3.0)
    ranked_ids = [p.id for _, p in results]
    # BAD should not be the top result for an active-region query.
    assert ranked_ids[0] != "BAD"


def test_zero_state_filter_in_default_path():
    """Sanity check : nowhere in the default retrieve path is a state
    filter applied. The architecture is purely geometric."""
    enc = _BagOfWordsEncoder()
    bad = _make_invalid_point(enc, "BAD", "x")
    pop = [
        Point(id="A", content="y", embedding_orig=enc.encode("y")),
        bad,
    ]
    # No exclude_states passed → no filtering, BAD must be reachable.
    results = retrieve(enc.encode("x"), pop, k=10, t_now=10.0)
    assert any(p.id == "BAD" for _, p in results)


# ---------------------------------------------------------------------------
# P6 — resurrection
# ---------------------------------------------------------------------------


def test_invalid_point_resurrects_on_new_corroborations():
    enc = _BagOfWordsEncoder()
    pop = [Point(id=f"P{i}", content=f"c{i}",
                 embedding_orig=enc.encode(f"c{i}")) for i in range(5)]
    bad = _make_invalid_point(enc, "BAD", "claim under dispute")
    pop.append(bad)
    assert bad.state == EpistemicState.INVALID

    # New evidence rolls in : a series of corroborations
    for t in range(10, 20):
        obs = Observation(
            source=SourceClass.OBSERVER,
            signal_type="corroboration",
            polarity=+1.0,
            target_node_ids=["BAD"],
            timestamp=float(t),
        )
        apply_observation(bad, obs, population=pop)

    # Counters have moved : n_corrob now greatly exceeds n_contra
    assert bad.n_corrob > bad.n_contra
    # State recomputed by A(·) reflects the reversal — point is back
    # in an active epistemic class.
    assert bad.state in {
        EpistemicState.CORROBORATED,
        EpistemicState.WARRANTED,
    }
    # The point identity (id) is preserved across the round trip
    assert bad.id == "BAD"
    # Retrieve picks it up again
    results = retrieve(enc.encode("claim under dispute"), pop, k=5, t_now=20.0)
    assert any(p.id == "BAD" for _, p in results)
