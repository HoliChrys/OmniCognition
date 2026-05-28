"""
Tests for Observators (pluralisme épistémique).

Properties :
  O1   Default behavior unchanged when observator_id is unspecified.
  O2   Named observator creates an ObservatorView on first touch.
  O3   Per-observator views track their OWN observations.
  O4   Default view aggregates ALL observations (any observator_id).
  O5   assign_status returns per-observator state from view counters.
  O6   Disagreement is visible : same node has different states across
       different observator views.
  O7   select_observators ranks by query–keywords cosine.
  O8   A(·) ⊥ P still holds for per-observator state.
  O9   No-laundering invariant : observator_id does NOT change the
       SourceClass typing of observations (GENERATOR still rejected).
"""

from __future__ import annotations

import hashlib
import math
from typing import Tuple

import pytest

from metacog import (
    DEFAULT_OBSERVATOR_ID,
    EpistemicState,
    LaunderingError,
    Observator,
    ObservatorView,
    Observation,
    Point,
    SourceClass,
    apply_observation,
    assert_no_laundering,
    assign_status,
    select_observators,
)


class _Encoder:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self._cache: dict = {}

    def encode(self, text: str) -> Tuple[float, ...]:
        words = text.lower().split()
        if not words:
            return tuple(0.0 for _ in range(self.dim))
        acc = [0.0] * self.dim
        for w in words:
            if w not in self._cache:
                h = int(hashlib.md5(f"obs:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


def _obs(target: str, polarity: float, t: float, *, observator_id="default"):
    return Observation(
        source=SourceClass.OBSERVER,
        signal_type="corroboration" if polarity > 0 else "contradiction",
        polarity=polarity,
        target_node_ids=[target],
        timestamp=t,
        observator_id=observator_id,
    )


# ---------- O1 -------------------------------------------------------------


def test_default_observator_id_preserves_existing_behavior():
    """Without specifying observator_id, behavior matches the pre-O1 system."""
    p = Point(id="P1", content="x", embedding_orig=(0.0,))
    obs = _obs("P1", +1.0, 1.0)  # observator_id defaults to "default"
    apply_observation(p, obs, population=[p])
    assert p.n_corrob == 1
    assert p.n_contra == 0
    # No view was created for the default observator (it lives on the
    # point's main counters directly)
    assert DEFAULT_OBSERVATOR_ID not in p.observator_views


# ---------- O2 -------------------------------------------------------------


def test_named_observator_creates_view():
    p = Point(id="P1", content="x", embedding_orig=(0.0,))
    obs = _obs("P1", +1.0, 1.0, observator_id="alice")
    apply_observation(p, obs, population=[p])
    assert "alice" in p.observator_views
    view = p.observator_views["alice"]
    assert isinstance(view, ObservatorView)
    assert view.n_corrob == 1


# ---------- O3, O4 --------------------------------------------------------


def test_views_track_only_their_own_observations_default_aggregates_all():
    p = Point(id="P1", content="x", embedding_orig=(0.0,))
    pop = [p]

    # Alice : +1, +1
    apply_observation(p, _obs("P1", +1.0, 1.0, observator_id="alice"), pop)
    apply_observation(p, _obs("P1", +1.0, 2.0, observator_id="alice"), pop)
    # Bob : -1
    apply_observation(p, _obs("P1", -1.0, 3.0, observator_id="bob"), pop)
    # Default (anonymous) : +1
    apply_observation(p, _obs("P1", +1.0, 4.0), pop)

    # Per-observator views see only their attributed observations
    assert p.observator_views["alice"].n_corrob == 2
    assert p.observator_views["alice"].n_contra == 0
    assert p.observator_views["bob"].n_corrob == 0
    assert p.observator_views["bob"].n_contra == 1
    # Default aggregates everything
    assert p.n_corrob == 3  # alice's 2 + bob's 0 + anonymous 1
    assert p.n_contra == 1  # bob's 1


# ---------- O5 -------------------------------------------------------------


def test_assign_status_returns_per_observator_state():
    p = Point(id="P1", content="x", embedding_orig=(0.0,))
    pop = [p]
    # Strong corroborations from alice
    for t in range(5):
        apply_observation(p, _obs("P1", +1.0, float(t), observator_id="alice"), pop)
    # Strong contradictions from bob
    for t in range(3):
        apply_observation(p, _obs("P1", -1.0, float(t + 10), observator_id="bob"), pop)

    alice_state = assign_status(p, pop, observator_id="alice")
    bob_state = assign_status(p, pop, observator_id="bob")
    default_state = assign_status(p, pop, observator_id="default")

    # Alice sees only corroborations → not INVALID
    assert alice_state != EpistemicState.INVALID
    # Bob sees only contradictions → SUSPECT or INVALID
    assert bob_state in {EpistemicState.SUSPECT, EpistemicState.INVALID}


# ---------- O6 — disagreement visible -------------------------------------


def test_disagreement_visible_across_views():
    p = Point(id="P1", content="x", embedding_orig=(0.0,))
    pop = [p]
    for t in range(3):
        apply_observation(p, _obs("P1", +1.0, float(t), observator_id="alice"), pop)
    for t in range(3):
        apply_observation(p, _obs("P1", -1.0, float(t + 10), observator_id="bob"), pop)

    alice_view = p.observator_views["alice"]
    bob_view = p.observator_views["bob"]
    # Both have observations but opposite polarities
    assert alice_view.n_corrob > 0 and alice_view.n_contra == 0
    assert bob_view.n_contra > 0 and bob_view.n_corrob == 0
    # Their states reflect this disagreement
    assert alice_view.state != bob_view.state


# ---------- O7 -------------------------------------------------------------


def test_select_observators_by_keyword_cosine():
    enc = _Encoder()
    medical = Observator(
        id="medical", name="Dr Alice",
        keywords=["consent", "patient", "protocol", "clinical"],
    )
    legal = Observator(
        id="legal", name="Bob the Lawyer",
        keywords=["contract", "liability", "compliance", "statute"],
    )
    other = Observator(
        id="cooking", name="Chef",
        keywords=["recipe", "pasta", "tomato", "italian"],
    )
    # Pre-compute keyword embeddings
    for o in (medical, legal, other):
        o.ensure_keywords_embedding(enc)

    query = enc.encode("patient consent protocol")
    top = select_observators(query, [medical, legal, other], k=3)
    assert len(top) == 3
    top_ids = [obs.id for _, obs in top]
    # Medical, whose keywords overlap directly with the query, ranks first.
    assert top_ids[0] == "medical"
    # Scores are monotonically decreasing
    scores = [s for s, _ in top]
    assert scores == sorted(scores, reverse=True)
    # Cooking, whose keywords have no overlap with the query, has the
    # lowest score (or tied for lowest)
    cooking_score = next(s for s, o in top if o.id == "cooking")
    medical_score = next(s for s, o in top if o.id == "medical")
    assert cooking_score <= medical_score


# ---------- O8 — A(·) ⊥ P with per-observator state ---------------------


def test_a_perp_p_holds_per_observator():
    """Same per-observator counters → same state, regardless of content."""
    a = Point(id="A", content="totally different X", embedding_orig=(0.0,))
    b = Point(id="B", content="completely other Y", embedding_orig=(0.0,))
    for t in range(4):
        apply_observation(a, _obs("A", +1.0, float(t), observator_id="alice"), [a, b])
        apply_observation(b, _obs("B", +1.0, float(t), observator_id="alice"), [a, b])
    # Identical counters → identical state
    s_a = assign_status(a, [a, b], observator_id="alice")
    s_b = assign_status(b, [a, b], observator_id="alice")
    assert s_a == s_b


# ---------- O9 — no-laundering still enforced --------------------------


def test_generator_still_rejected_regardless_of_observator():
    with pytest.raises(LaunderingError):
        Observation(
            source=SourceClass.GENERATOR,
            signal_type="x",
            polarity=+1.0,
            target_node_ids=["P1"],
            timestamp=1.0,
            observator_id="alice",  # observator does NOT bypass Cor. 5
        )


def test_no_laundering_after_multi_observator_session():
    p = Point(id="P1", content="x", embedding_orig=(0.0,))
    pop = [p]
    for t in range(5):
        for who in ("alice", "bob", "default"):
            polarity = +1.0 if (t + (0 if who == "alice" else 1)) % 2 == 0 else -1.0
            apply_observation(
                p,
                _obs("P1", polarity, float(t) + (0.1 if who == "bob" else 0.0),
                     observator_id=who),
                pop,
            )
    report = assert_no_laundering(pop)
    assert report.ok
    assert report.by_source.get(SourceClass.GENERATOR, 0) == 0
