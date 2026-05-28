"""
Tests for the manifold geometry (replacement for the graph paradigm).

Each test validates one property of the parameter-free dynamics
designed in the hand simulation:

  G1  Corroboration reduces the distance between two points.
  G2  Contradiction increases the distance between two points.
  G3  Silence decays delta_active while preserving delta_latent.
  G4  Re-spike after silence recovers from latent (no restart at 0).
  G5  k_nearest produces the on-demand graph view.
  G6  No-laundering invariant still holds after geometric pulls.
  G7  A(·) still ⊥ P after geometric updates.
  G8  Symmetric pull: i and j move toward each other equally (modulo
      their own step sizes).
  G9  Step size 1 / (1 + n_obs) — saturation behaviour is visible.
"""

from __future__ import annotations

from metacog import (
    LaunderingError,  # noqa: F401  (used implicitly by Observation)
    Observation,
    Point,
    SourceClass,
    apply_observation,
    apply_pull,
    assert_no_laundering,
    assign_status,
    cosine,
    decay_factor,
    distance,
    effective_embedding,
    k_nearest,
    process_observation,
    vec_norm,
)


# ---------- G1, G2 — basic dynamics ---------------------------------------

def test_corroboration_reduces_distance():
    a = Point(id="A", content="...", embedding_orig=(0.0, 0.0))
    b = Point(id="B", content="...", embedding_orig=(10.0, 0.0))
    d_before = distance(effective_embedding(a, 0.0), effective_embedding(b, 0.0))

    apply_pull(a, b, +1.0, 1.0)
    a.t_last_obs = 1.0
    b.t_last_obs = 1.0
    a.n_corrob += 1
    b.n_corrob += 1

    d_after = distance(effective_embedding(a, 1.0), effective_embedding(b, 1.0))
    assert d_after < d_before


def test_contradiction_increases_distance():
    a = Point(id="A", content="...", embedding_orig=(0.0, 0.0))
    c = Point(id="C", content="...", embedding_orig=(0.0, 10.0))
    d_before = distance(effective_embedding(a, 0.0), effective_embedding(c, 0.0))

    apply_pull(a, c, -1.0, 1.0)
    a.t_last_obs = 1.0
    c.t_last_obs = 1.0
    a.n_contra += 1
    c.n_contra += 1

    d_after = distance(effective_embedding(a, 1.0), effective_embedding(c, 1.0))
    assert d_after > d_before


# ---------- G3 — silence decays active, preserves latent -----------------

def test_silence_decays_active_preserves_latent():
    a = Point(id="A", content="...", embedding_orig=(0.0, 0.0))
    b = Point(id="B", content="...", embedding_orig=(10.0, 0.0))

    apply_pull(a, b, +1.0, 1.0)
    a.t_last_obs = 1.0
    b.t_last_obs = 1.0
    a.n_corrob += 1
    b.n_corrob += 1

    active_norm_at_obs = vec_norm(a.delta_active)
    latent_at_obs = a.delta_latent
    assert active_norm_at_obs > 0
    assert vec_norm(latent_at_obs) > 0

    # Read effective embedding far in the future — active should be ~0,
    # but latent should still appear in the embedding.
    eff_far = effective_embedding(a, 100.0)
    decay = decay_factor(100.0, a.t_last_obs)
    active_far_norm = active_norm_at_obs * decay
    assert active_far_norm < 0.05 * active_norm_at_obs  # heavy decay
    # Latent never changes outside of an observation
    assert a.delta_latent == latent_at_obs
    # Effective embedding is dominated by orig + latent
    expected_far = tuple(o + l for o, l in zip(a.embedding_orig, a.delta_latent))
    # The decayed active is a small perturbation
    for got, exp in zip(eff_far, expected_far):
        assert abs(got - exp) < 0.05


# ---------- G4 — re-spike recovers from latent ----------------------------

def test_respike_recovers_from_latent():
    a = Point(id="A", content="...", embedding_orig=(0.0, 0.0))
    b = Point(id="B", content="...", embedding_orig=(10.0, 0.0))

    # Two consecutive corroborations
    for t in (1.0, 2.0):
        apply_pull(a, b, +1.0, t)
        a.t_last_obs = t
        b.t_last_obs = t
        a.n_corrob += 1
        b.n_corrob += 1

    latent_after_obs = a.delta_latent
    assert vec_norm(latent_after_obs) > 0

    # Distance at t=30 (long silence): should be close to orig+latent
    d_at_30 = distance(effective_embedding(a, 30.0), effective_embedding(b, 30.0))

    # Re-spike at t=31
    apply_pull(a, b, +1.0, 31.0)
    a.t_last_obs = 31.0
    b.t_last_obs = 31.0
    a.n_corrob += 1
    b.n_corrob += 1

    d_at_31 = distance(effective_embedding(a, 31.0), effective_embedding(b, 31.0))

    # Re-spike pulled them closer than the silence equilibrium
    assert d_at_31 < d_at_30


# ---------- G5 — kNN view -------------------------------------------------

def test_k_nearest_returns_top_k_by_cosine():
    points = [
        Point(id="A", content="...", embedding_orig=(1.0, 0.0)),
        Point(id="B", content="...", embedding_orig=(0.0, 1.0)),
        Point(id="C", content="...", embedding_orig=(-1.0, 0.0)),
        Point(id="D", content="...", embedding_orig=(0.5, 0.5)),
    ]
    query = (1.0, 0.0)
    result = k_nearest(query, points, k=2, t_now=0.0)
    assert len(result) == 2
    assert result[0][1].id == "A"  # perfect cosine match
    # Second best is D (cos≈0.707) above B (cos=0) and C (cos=-1)
    assert result[1][1].id == "D"


# ---------- G6 — no-laundering after geometry -----------------------------

def test_no_laundering_after_many_pulls():
    points = [
        Point(id=f"P{i}", content=f"...{i}", embedding_orig=(float(i), 0.0))
        for i in range(5)
    ]
    for t in range(10):
        a = points[t % 5]
        b = points[(t + 1) % 5]
        obs = Observation(
            source=SourceClass.COMPUTATION,
            signal_type="corroboration",
            polarity=+1.0,
            target_node_ids=[a.id, b.id],
            timestamp=float(t),
        )
        process_observation(points, obs, population=points)

    report = assert_no_laundering(points)
    assert report.ok
    assert report.by_source[SourceClass.GENERATOR] == 0


# ---------- G7 — A(·) ⊥ P  still holds after geometric updates ----------

def test_a_independent_of_P_after_pulls():
    points = [
        Point(id=f"P{i}", content=f"different-content-{i}",
              embedding_orig=(float(i), 0.0))
        for i in range(6)
    ]
    for t in range(6):
        obs = Observation(
            source=SourceClass.COMPUTATION,
            signal_type="corroboration",
            polarity=+1.0,
            target_node_ids=[points[t % 6].id, points[(t + 1) % 6].id],
            timestamp=float(t),
        )
        process_observation(points, obs, population=points)

    # Build two points with identical counters but radically different
    # content, embedding, and deltas — A(·) should still return the same.
    a = Point(id="ghost-A", content="X", embedding_orig=(0.0, 0.0),
              n_corrob=points[0].n_corrob, n_contra=points[0].n_contra,
              delta_active=(50.0, 50.0), delta_latent=(50.0, 50.0))
    b = Point(id="ghost-B", content="Y", embedding_orig=(999.0, 999.0),
              n_corrob=points[0].n_corrob, n_contra=points[0].n_contra,
              delta_active=(0.0, 0.0), delta_latent=(0.0, 0.0))
    pop = points + [a, b]
    assert assign_status(a, pop) == assign_status(b, pop)


# ---------- G8 — symmetric pull ------------------------------------------

def test_symmetric_pull_moves_both_points():
    a = Point(id="A", content="...", embedding_orig=(0.0, 0.0))
    b = Point(id="B", content="...", embedding_orig=(10.0, 0.0))

    apply_pull(a, b, +1.0, 1.0)

    # Both should have non-zero active offsets, in opposite directions
    assert vec_norm(a.delta_active) > 0
    assert vec_norm(b.delta_active) > 0
    # A moves in +x direction (toward B), B moves in -x direction (toward A)
    assert a.delta_active[0] > 0
    assert b.delta_active[0] < 0


# ---------- G9 — saturation by pas = 1 / (1 + n_obs) --------------------

def test_step_size_decreases_with_observations():
    a = Point(id="A", content="...", embedding_orig=(0.0, 0.0))
    b = Point(id="B", content="...", embedding_orig=(10.0, 0.0))

    distances = []
    for t in range(1, 6):
        d_before = distance(effective_embedding(a, float(t)),
                            effective_embedding(b, float(t)))
        apply_pull(a, b, +1.0, float(t))
        a.t_last_obs = float(t)
        b.t_last_obs = float(t)
        a.n_corrob += 1
        b.n_corrob += 1
        d_after = distance(effective_embedding(a, float(t)),
                           effective_embedding(b, float(t)))
        distances.append(d_before - d_after)

    # Each successive corroboration should produce a smaller distance
    # change than the previous one (saturation).
    for i in range(1, len(distances)):
        # Allow ties or small numerical noise; require monotonic non-increase.
        assert distances[i] <= distances[i - 1] + 1e-9


# ---------- G10 — collision protection -----------------------------------

def test_pull_skips_at_collision():
    """When two points share the same effective embedding, direction is
    undefined; apply_pull must skip the geometric update without error."""
    a = Point(id="A", content="...", embedding_orig=(0.0, 0.0))
    b = Point(id="B", content="...", embedding_orig=(0.0, 0.0))

    apply_pull(a, b, +1.0, 1.0)

    # Neither active should have changed
    assert a.delta_active == (0.0, 0.0)
    assert b.delta_active == (0.0, 0.0)


# ---------- G11 — process_observation orchestration ----------------------

def test_process_observation_integrates_geometry_and_counters():
    a = Point(id="A", content="...", embedding_orig=(0.0, 0.0))
    b = Point(id="B", content="...", embedding_orig=(10.0, 0.0))
    population = [a, b]

    obs = Observation(
        source=SourceClass.OBSERVER,
        signal_type="corroboration",
        polarity=+1.0,
        target_node_ids=["A", "B"],
        timestamp=1.0,
    )
    entries = process_observation(population, obs, population=population)
    assert len(entries) == 2

    # Both counters incremented
    assert a.n_corrob == 1 and b.n_corrob == 1
    # Both geometry offsets non-zero (pull occurred before observation)
    assert vec_norm(a.delta_active) > 0
    assert vec_norm(b.delta_active) > 0
    # t_last_obs updated
    assert a.t_last_obs == 1.0 and b.t_last_obs == 1.0
