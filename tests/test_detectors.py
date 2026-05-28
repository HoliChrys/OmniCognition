"""
Tests for the deterministic signal detectors.

Properties :
  D1   REFORMULATION fires on similar consecutive queries.
  D2   REFORMULATION suppressed by a topic-change marker.
  D3   CONTRADICTION fires on negation + content overlap.
  D4   CONTRADICTION not fired without negation.
  D5   CORROBORATION fires on overlap + extension, no negation.
  D6   CORROBORATION suppressed by negation.
  D7   CONTINUATION fires on marker + topic shift.
  D8   CONTINUATION suppressed if topic unchanged.
  D9   REPETITION_OF_NEED fires when memory had unused matching point.
  D10  All detected observations are COMPUTATION class (no laundering).
  D11  Multiple detectors can fire on the same turn.
  D12  Empty history → no detection.
  D13  End-to-end : detected observation drives counter update.
"""

from __future__ import annotations

import hashlib
import math
from typing import Tuple

from metacog import (
    EpistemicState,
    Observation,
    Point,
    SourceClass,
    analyze_user_turn,
    apply_observation,
    ConversationLog,
    detect_contradiction,
    detect_continuation,
    detect_corroboration,
    detect_reformulation,
    detect_repetition_of_need,
    TurnRecord,
)


class _BagOfWordsEncoder:
    """Simhash-style deterministic encoder for tests.

    Each word hashes to a ±1 vector in dim-dimensional space; the
    encoding of a text is the L2-normalized sum. Unrelated texts give
    cosines near 0 (random-walk), identical texts give cosine = 1.
    """

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
                h = int(hashlib.md5(f"bow:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0
                    for i in range(self.dim)
                )
            wv = self._cache[w]
            for i in range(self.dim):
                acc[i] += wv[i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


def _turn(t: float, speaker: str, text: str, encoder, ids=()) -> TurnRecord:
    return TurnRecord(
        timestamp=t,
        speaker=speaker,
        text=text,
        embedding=encoder.encode(text),
        retrieved_point_ids=list(ids),
    )


# ---------- D1, D2 — reformulation ----------------------------------------


def test_reformulation_detected_on_similar_consecutive_queries():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "user", "where did i meet sarah berkeley", enc))
    log.record(_turn(2.0, "assistant", "at the conference", enc, ids=["P1", "P2"]))
    new_turn = _turn(3.0, "user", "where did i meet sarah berkeley please", enc)

    obs = detect_reformulation(new_turn, log)
    assert obs is not None
    assert obs.signal_type == "reformulation"
    assert obs.polarity == -1.0
    assert obs.source == SourceClass.COMPUTATION
    assert set(obs.target_node_ids) == {"P1", "P2"}


def test_reformulation_suppressed_by_transition_marker():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "user", "where did i meet sarah", enc))
    log.record(_turn(2.0, "assistant", "at the conference", enc, ids=["P1"]))
    new_turn = _turn(3.0, "user", "by the way where did i meet sarah", enc)
    assert detect_reformulation(new_turn, log) is None


# ---------- D3, D4 — contradiction ---------------------------------------


def test_contradiction_detected_with_negation_and_overlap():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "assistant", "you met sarah at berkeley",
                     enc, ids=["P1"]))
    new_turn = _turn(2.0, "user", "no not berkeley", enc)
    obs = detect_contradiction(new_turn, log)
    assert obs is not None
    assert obs.signal_type == "contradiction"
    assert obs.polarity == -1.0
    assert "P1" in obs.target_node_ids


def test_contradiction_not_detected_without_negation():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "assistant", "you met sarah at berkeley",
                     enc, ids=["P1"]))
    new_turn = _turn(2.0, "user", "yes berkeley correct", enc)
    assert detect_contradiction(new_turn, log) is None


# ---------- D5, D6 — corroboration ---------------------------------------


def test_corroboration_detected_with_overlap_and_extension():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "assistant", "you met sarah at berkeley",
                     enc, ids=["P1"]))
    new_turn = _turn(2.0, "user", "berkeley yes and also stanford", enc)
    obs = detect_corroboration(new_turn, log)
    assert obs is not None
    assert obs.signal_type == "corroboration"
    assert obs.polarity == +1.0


def test_corroboration_suppressed_by_negation():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "assistant", "you met sarah at berkeley",
                     enc, ids=["P1"]))
    new_turn = _turn(2.0, "user", "no berkeley wrong also stanford", enc)
    assert detect_corroboration(new_turn, log) is None


# ---------- D7, D8 — continuation ----------------------------------------


def test_continuation_detected_with_marker_and_topic_shift():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "user", "where did i meet sarah", enc))
    log.record(_turn(2.0, "assistant", "at berkeley conference", enc, ids=["P1"]))
    new_turn = _turn(3.0, "user", "ok now tell me pizza recipe italian napoli",
                     enc)
    obs = detect_continuation(new_turn, log)
    assert obs is not None
    assert obs.signal_type == "continuation"
    assert obs.polarity == +1.0


def test_continuation_not_detected_without_marker():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "user", "where did i meet sarah", enc))
    log.record(_turn(2.0, "assistant", "at berkeley", enc, ids=["P1"]))
    new_turn = _turn(3.0, "user", "now what about pizza recipe", enc)
    assert detect_continuation(new_turn, log) is None


# ---------- D9 — repetition of need --------------------------------------


def test_repetition_of_need_when_memory_had_unused_match():
    enc = _BagOfWordsEncoder()
    points = [
        Point(id="P1", content="sarah berkeley conference 2023",
              embedding_orig=enc.encode("sarah berkeley conference 2023")),
        Point(id="P2", content="pizza recipe italian",
              embedding_orig=enc.encode("pizza recipe italian")),
    ]
    log = ConversationLog()
    log.record(_turn(1.0, "assistant", "i don't know", enc, ids=["P2"]))
    new_turn = _turn(2.0, "user", "sarah berkeley conference 2023", enc)

    obs = detect_repetition_of_need(new_turn, log, points, t_now=2.0)
    assert obs is not None
    assert obs.signal_type == "repetition_of_need"
    assert obs.polarity == -1.0
    assert "P1" in obs.target_node_ids
    assert "P2" not in obs.target_node_ids


# ---------- D10 — invariants ---------------------------------------------


def test_all_detected_observations_are_computation_class():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "user", "alpha", enc))
    log.record(_turn(2.0, "assistant", "alpha response", enc, ids=["P1"]))
    points = [
        Point(id="P1", content="alpha",
              embedding_orig=enc.encode("alpha")),
    ]
    new_turn = _turn(3.0, "user", "alpha", enc)
    observations = analyze_user_turn(new_turn, log, points, encoder=enc)
    for obs in observations:
        assert obs.source == SourceClass.COMPUTATION
        assert obs.polarity in {-1.0, +1.0}


# ---------- D11 — multiple detectors can co-fire -------------------------


def test_multiple_detectors_can_fire_on_same_turn():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    log.record(_turn(1.0, "user", "where did i meet sarah berkeley", enc))
    log.record(_turn(2.0, "assistant", "berkeley conference", enc, ids=["P1"]))
    points = [
        Point(id="P1", content="berkeley conference",
              embedding_orig=enc.encode("berkeley conference")),
    ]
    # Turn with high cosine to prev query AND negation + overlap
    new_turn = _turn(3.0, "user",
                     "no where did i meet sarah berkeley conference please",
                     enc)
    observations = analyze_user_turn(new_turn, log, points, encoder=enc)
    signal_types = {o.signal_type for o in observations}
    # Reformulation should fire (high cosine to prev query)
    # Contradiction should fire (negation + overlap with assistant)
    assert "reformulation" in signal_types
    assert "contradiction" in signal_types


# ---------- D12 — cold start ---------------------------------------------


def test_no_detection_on_first_turn():
    enc = _BagOfWordsEncoder()
    log = ConversationLog()
    points = []
    new_turn = _turn(1.0, "user", "first message", enc)
    assert analyze_user_turn(new_turn, log, points, encoder=enc) == []


# ---------- D13 — end-to-end pipeline ------------------------------------


def test_pipeline_user_turn_to_counter_update():
    enc = _BagOfWordsEncoder()
    points = [
        Point(id="P1", content="met sarah at berkeley",
              embedding_orig=enc.encode("met sarah at berkeley")),
    ]
    log = ConversationLog()
    log.record(_turn(1.0, "user", "where did i meet sarah", enc))
    log.record(_turn(2.0, "assistant", "at berkeley", enc, ids=["P1"]))
    new_turn = _turn(3.0, "user", "no not berkeley", enc)

    observations = analyze_user_turn(new_turn, log, points, encoder=enc)
    contradiction = next(
        (o for o in observations if o.signal_type == "contradiction"), None
    )
    assert contradiction is not None

    n_contra_before = points[0].n_contra
    apply_observation(points[0], contradiction, population=points)
    assert points[0].n_contra == n_contra_before + 1
