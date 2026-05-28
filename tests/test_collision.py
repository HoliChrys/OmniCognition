"""
Tests for collision detection and resolution.

Validated design (per user decisions) :
  Q1 — Threshold : Option A, emergent (median - σ of pairwise distances)
  Q2 — Parent identity : Option Z, counters kept + n_revision += 1
  Q3 — LLM is content-only, traced as GENERATOR in CollisionEvent
  Q4 — Cascade : max 2 iterations of detect→resolve per cycle

Properties tested :
  C1   collision_threshold is emergent — no tuning constants
  C2   detect_collisions finds only the anomalously close pairs
  C3   resolve_collision creates a CHILD with two parents
  C4   parents keep counters but n_revision += 1 (Option Z)
  C5   parents' uncertainty strictly rises after revision (Beta dilution)
  C6   child starts in CONJECTURE
  C7   child.lineage_depth = max(parents.lineage_depth) + 1
  C8   no-laundering invariant holds across collisions
  C9   A(·) ⊥ P still holds after revisions
  C10  cascade limited to 2 iterations per cycle
  C11  empty common → resolve_collision returns None (noise guard)
"""

from __future__ import annotations

import math
from typing import Tuple

from metacog import (
    EpistemicState,
    MAX_CASCADE_ITERATIONS,
    Observation,
    Point,
    SourceClass,
    assert_no_laundering,
    assign_status,
    collision_threshold,
    detect_collisions,
    distance,
    effective_embedding,
    process_observation,
    resolve_collision,
    sleep_cycle_collisions,
)


# ---------------------------------------------------------------------------
# Deterministic stubs (only used in tests, never in production code)
# ---------------------------------------------------------------------------


class _BagOfWordsEncoder:
    """Deterministic test encoder : sum of per-word hashed vectors."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self._cache: dict[str, Tuple[float, ...]] = {}

    def _word_vec(self, word: str) -> Tuple[float, ...]:
        if word not in self._cache:
            h = abs(hash(("bow", word)))
            self._cache[word] = tuple(
                ((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self.dim)
            )
        return self._cache[word]

    def encode(self, text: str) -> Tuple[float, ...]:
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
    """Deterministic test LLM : word-set common / difference, N-way."""

    def extract_common(self, texts) -> str:
        texts = list(texts)
        if len(texts) < 2:
            return ""
        word_sets = [set(t.lower().split()) for t in texts]
        common = word_sets[0].copy()
        for ws in word_sets[1:]:
            common &= ws
        if not common:
            return ""
        seen: set[str] = set()
        out: list[str] = []
        for w in texts[0].lower().split():
            if w in common and w not in seen:
                out.append(w)
                seen.add(w)
        return " ".join(out)

    def remove_overlap(self, full: str, common: str) -> str:
        cset = set(common.lower().split())
        return " ".join(w for w in full.lower().split() if w not in cset)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pop(encoder: _BagOfWordsEncoder, contents: list[Tuple[str, str]]) -> list[Point]:
    return [
        Point(id=pid, content=text, embedding_orig=encoder.encode(text))
        for pid, text in contents
    ]


# ---------------------------------------------------------------------------
# C1 — emergent threshold
# ---------------------------------------------------------------------------


def test_threshold_is_emergent_from_distribution():
    enc = _BagOfWordsEncoder()
    points = _build_pop(
        enc,
        [
            ("P1", "alpha beta"),
            ("P2", "gamma delta"),
            ("P3", "epsilon zeta"),
            ("P4", "alpha alphax"),  # close to P1
            ("P5", "alpha alphay"),  # close to P1 and P4
        ],
    )
    t1 = collision_threshold(points, 0.0)
    assert t1 is not None
    assert t1 > 0
    # Shrinking the population to fewer than 4 points → no threshold
    assert collision_threshold(points[:3], 0.0) is None


# ---------------------------------------------------------------------------
# C2 — detection finds anomalously close pairs
# ---------------------------------------------------------------------------


def test_detect_collisions_finds_close_pairs():
    enc = _BagOfWordsEncoder()
    points = _build_pop(
        enc,
        [
            ("FAR1", "alpha"),
            ("FAR2", "beta"),
            ("FAR3", "gamma"),
            ("FAR4", "delta"),
            ("CLOSE_A", "epsilon zeta eta theta"),
            ("CLOSE_B", "epsilon zeta eta iota"),  # shares 3 of 4 words
        ],
    )
    collisions = detect_collisions(points, 0.0)
    ids_in_collisions = {p.id for pair in collisions for p in (pair[0], pair[1])}
    assert "CLOSE_A" in ids_in_collisions
    assert "CLOSE_B" in ids_in_collisions


# ---------------------------------------------------------------------------
# C3, C6, C7 — resolution structure
# ---------------------------------------------------------------------------


def test_resolve_creates_child_with_parents_and_correct_lineage():
    enc = _BagOfWordsEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="dr sarah berkeley counseling",
              embedding_orig=enc.encode("dr sarah berkeley counseling"))
    b = Point(id="B", content="dr sarah berkeley psychology",
              embedding_orig=enc.encode("dr sarah berkeley psychology"))

    result = resolve_collision([a, b], llm, enc, t_now=1.0,
                               trigger_distance=0.1, threshold=0.2)
    assert result is not None
    child = result.child
    # Child has both parents
    assert set(child.parents) == {"A", "B"}
    # Child starts in CONJECTURE
    assert child.state == EpistemicState.CONJECTURE
    # Lineage depth = max(parents) + 1
    assert child.lineage_depth == 1
    # Child content captures the common words
    common_words = set(child.content.split())
    assert "dr" in common_words
    assert "sarah" in common_words
    assert "berkeley" in common_words
    # Parents have the child registered
    assert child.id in a.children
    assert child.id in b.children
    # Result.parents reflects the list of all parents
    assert {p.id for p in result.parents} == {"A", "B"}


# ---------------------------------------------------------------------------
# C4, C5 — Option Z : counters kept, revision dilutes confidence
# ---------------------------------------------------------------------------


def test_parents_retain_counters_n_revision_incremented():
    enc = _BagOfWordsEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="dr sarah berkeley counseling",
              embedding_orig=enc.encode("dr sarah berkeley counseling"),
              n_corrob=4, n_contra=1, n_uses=3)
    b = Point(id="B", content="dr sarah berkeley psychology",
              embedding_orig=enc.encode("dr sarah berkeley psychology"),
              n_corrob=2, n_contra=2, n_uses=5)

    n_corrob_a_before, n_contra_a_before = a.n_corrob, a.n_contra
    n_corrob_b_before, n_contra_b_before = b.n_corrob, b.n_contra

    resolve_collision([a, b], llm, enc, t_now=1.0,
                      trigger_distance=0.05, threshold=0.2)

    # Counters unchanged
    assert a.n_corrob == n_corrob_a_before
    assert a.n_contra == n_contra_a_before
    assert b.n_corrob == n_corrob_b_before
    assert b.n_contra == n_contra_b_before
    # n_revision incremented
    assert a.n_revision == 1
    assert b.n_revision == 1


def test_uncertainty_rises_after_revision():
    enc = _BagOfWordsEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="dr sarah berkeley counseling",
              embedding_orig=enc.encode("dr sarah berkeley counseling"),
              n_corrob=6, n_contra=2)
    b = Point(id="B", content="dr sarah berkeley psychology",
              embedding_orig=enc.encode("dr sarah berkeley psychology"),
              n_corrob=6, n_contra=2)

    u_before_a = a.uncertainty
    resolve_collision([a, b], llm, enc, t_now=1.0,
                      trigger_distance=0.05, threshold=0.2)
    u_after_a = a.uncertainty
    assert u_after_a > u_before_a


# ---------------------------------------------------------------------------
# C8 — no laundering
# ---------------------------------------------------------------------------


def test_no_laundering_after_collision_cycle():
    enc = _BagOfWordsEncoder()
    llm = _SetOpsLLM()
    points = _build_pop(
        enc,
        [
            ("P1", "alpha beta"),
            ("P2", "gamma delta"),
            ("P3", "epsilon zeta"),
            ("P4", "eta theta"),
            ("A1", "common token unique_a"),
            ("A2", "common token unique_b"),
        ],
    )
    # Add some observations so audit logs are non-empty
    for i, p in enumerate(points):
        obs = Observation(
            source=SourceClass.OBSERVER,
            signal_type="corroboration",
            polarity=+1.0,
            target_node_ids=[p.id],
            timestamp=float(i),
        )
        process_observation([p], obs, population=points)

    sleep_cycle_collisions(points, llm, enc, t_now=10.0)
    # Even after spawning children and revising parents, no GENERATOR
    # has been logged as an observation source.
    report = assert_no_laundering(points)
    assert report.ok
    assert report.by_source.get(SourceClass.GENERATOR, 0) == 0


# ---------------------------------------------------------------------------
# C9 — A(·) ⊥ P after revisions
# ---------------------------------------------------------------------------


def test_a_perp_p_holds_after_collision_revision():
    enc = _BagOfWordsEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="alpha beta common",
              embedding_orig=enc.encode("alpha beta common"),
              n_corrob=4, n_contra=1)
    b = Point(id="B", content="gamma delta common",
              embedding_orig=enc.encode("gamma delta common"),
              n_corrob=4, n_contra=1)
    pop = [a, b] + _build_pop(enc, [(f"F{i}", f"filler{i}") for i in range(5)])
    resolve_collision([a, b], llm, enc, t_now=1.0,
                      trigger_distance=0.05, threshold=0.2)
    # Build a ghost point with identical counters but a totally different
    # content / embedding — A(·) must return the same state.
    ghost = Point(
        id="ghost",
        content="ZZZ totally different content",
        embedding_orig=(99.0, 99.0, 99.0, 99.0),
        n_corrob=a.n_corrob, n_contra=a.n_contra, n_revision=a.n_revision,
    )
    pop.append(ghost)
    assert assign_status(a, pop) == assign_status(ghost, pop)


# ---------------------------------------------------------------------------
# C10 — cascade limit
# ---------------------------------------------------------------------------


def test_cascade_limited_to_two_iterations():
    """Build a degenerate cluster where children of children would also
    collide. Verify the cycle stops at MAX_CASCADE_ITERATIONS."""

    enc = _BagOfWordsEncoder()
    llm = _SetOpsLLM()
    # Five very similar points — each pair shares many words.
    contents = [
        ("S1", "common token alpha beta1"),
        ("S2", "common token alpha beta2"),
        ("S3", "common token alpha beta3"),
        ("S4", "common token alpha beta4"),
        ("S5", "common token alpha beta5"),
        # Pad with different points so threshold isn't trivially 0
        ("F1", "completely unrelated"),
        ("F2", "another distinct thing"),
        ("F3", "third orthogonal item"),
    ]
    points = _build_pop(enc, contents)
    report = sleep_cycle_collisions(points, llm, enc, t_now=1.0)
    assert report.iterations <= MAX_CASCADE_ITERATIONS == 2
    # The cycle ran at least once (something WAS too close)
    assert report.iterations >= 1


# ---------------------------------------------------------------------------
# C11 — empty common is a noise guard
# ---------------------------------------------------------------------------


def test_resolve_returns_none_when_no_real_overlap():
    enc = _BagOfWordsEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="alpha beta",
              embedding_orig=enc.encode("alpha beta"))
    b = Point(id="B", content="gamma delta",
              embedding_orig=enc.encode("gamma delta"))
    result = resolve_collision([a, b], llm, enc, t_now=1.0,
                               trigger_distance=0.05, threshold=0.2)
    assert result is None
    # And the parents are untouched
    assert a.n_revision == 0
    assert b.n_revision == 0
    assert a.children == []
    assert b.children == []


# ---------------------------------------------------------------------------
# C12 — CollisionEvent fully audits the operation
# ---------------------------------------------------------------------------


def test_collision_event_records_both_source_classes():
    enc = _BagOfWordsEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="dr sarah counseling",
              embedding_orig=enc.encode("dr sarah counseling"))
    b = Point(id="B", content="dr sarah psychology",
              embedding_orig=enc.encode("dr sarah psychology"))
    result = resolve_collision([a, b], llm, enc, t_now=1.0,
                               trigger_distance=0.1, threshold=0.2)
    assert result is not None
    event = result.event
    # Detection of proximity is COMPUTATION (deterministic distance)
    assert event.detection_class == SourceClass.COMPUTATION
    # Content generation (LLM extract_common + remove_overlap) is GENERATOR
    assert event.content_generation_class == SourceClass.GENERATOR
    # Parent ids are recorded (≥ 2)
    assert set(event.parent_ids) == {"A", "B"}
    # No anchors used for this proximity collision
    assert event.anchor_ids == ()
