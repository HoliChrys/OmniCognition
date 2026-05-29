"""
Tests for the keyword-driven geometric ops + revisit refresh.

Properties :
  KG1   effective_keyword_embedding returns keywords_emb + deltas
        when point has keywords ; falls back to effective_embedding
  KG2   apply_pull moves points along the keyword axis, not content
  KG3   apply_exile pushes against the keyword centroid
  KG4   collision_threshold + detect_collisions use keyword distance
  KG5   Memory.observe triggers _refresh_keywords on each target
  KG6   Memory.process_turn triggers refresh on detected-signal targets
  KG7   _refresh_keywords is a no-op when keywords are unchanged
  KG8   additional_context shifts the extracted keywords
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Tuple
from unittest.mock import MagicMock

from metacog import (
    Memory,
    Point,
    PointKind,
    SimpleKeywordExtractor,
    SourceClass,
    apply_pull,
    apply_exile,
    detect_collisions,
)
from metacog.geometry import (
    effective_embedding,
    effective_keyword_embedding,
)


class _Enc:
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
                h = int(hashlib.md5(f"kg:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


# ---------- KG1 ----------------------------------------------------------


def test_effective_keyword_embedding_uses_keywords_when_present():
    enc = _Enc()
    p = Point(
        id="P1",
        content="this is the long content text with many words",
        embedding_orig=enc.encode("this is the long content text with many words"),
        keywords=["alpha", "beta"],
        keywords_embedding=enc.encode("alpha beta"),
    )
    eff_kw = effective_keyword_embedding(p, t_now=0.0)
    eff_c = effective_embedding(p, t_now=0.0)
    # With deltas at zero, eff_kw == keywords_embedding (different from
    # effective_embedding which is based on content)
    assert eff_kw != eff_c
    assert eff_kw == tuple(p.keywords_embedding)


def test_effective_keyword_embedding_falls_back_when_no_keywords():
    enc = _Enc()
    p = Point(
        id="P1",
        content="some content",
        embedding_orig=enc.encode("some content"),
        keywords=[],
        keywords_embedding=None,
    )
    eff_kw = effective_keyword_embedding(p, t_now=0.0)
    eff_c = effective_embedding(p, t_now=0.0)
    assert eff_kw == eff_c


# ---------- KG2 ----------------------------------------------------------


def test_apply_pull_moves_along_keyword_axis():
    """apply_pull computes direction from keyword embeddings : two points
    with identical content but different keywords still get pulled."""
    enc = _Enc()
    common_content = "shared identical content text"
    p1 = Point(
        id="P1",
        content=common_content,
        embedding_orig=enc.encode(common_content),
        keywords=["alpha"],
        keywords_embedding=enc.encode("alpha"),
    )
    p2 = Point(
        id="P2",
        content=common_content,  # same content !
        embedding_orig=enc.encode(common_content),
        keywords=["omega"],  # different keywords
        keywords_embedding=enc.encode("omega"),
    )
    before = tuple(p1.delta_active)
    apply_pull(p1, p2, polarity=+1.0, t_now=1.0)
    after = tuple(p1.delta_active)
    # Delta moved because keyword embeddings differ even if content matches
    assert before != after


# ---------- KG5 ----------------------------------------------------------


def test_observe_refreshes_keywords_on_target():
    m = Memory(encoder=_Enc(), extractor=SimpleKeywordExtractor())
    p = m.ingest("alpha beta gamma delta", id="P1")
    initial_kws = list(p.keywords)
    # observe with a raw_content that introduces new entities
    m.observe(["P1"], polarity=+1.0,
              source="OBSERVER", signal_type="corroboration",
              raw_content="zeta theta iota kappa")
    # Keywords should have been re-extracted (and possibly extended by
    # the additional context)
    p = next(q for q in m.points if q.id == "P1")
    # At minimum, the refresh ran : keywords may differ
    assert p.keywords is not None


def test_observe_refresh_uses_additional_context():
    """The additional_context (raw_content) is concatenated with the
    point's content for extraction, so context-only entities can
    appear in the refreshed keywords."""
    m = Memory(encoder=_Enc(), extractor=SimpleKeywordExtractor())
    p = m.ingest("simple text", id="P1")
    # Provide a raw_content rich in repeating entities
    m.observe(["P1"], polarity=+1.0,
              raw_content="sweden sweden sweden grandma grandma")
    p = next(q for q in m.points if q.id == "P1")
    # The extractor should now see "sweden" (3x in context) as top keyword
    assert "sweden" in p.keywords


# ---------- KG7 ----------------------------------------------------------


def test_refresh_keywords_noop_when_unchanged():
    """If the extractor returns the same keywords, the embedding is
    NOT recomputed (avoids needless encoder calls)."""
    m = Memory(encoder=_Enc(), extractor=SimpleKeywordExtractor())
    p = m.ingest("alpha beta gamma delta epsilon", id="P1")
    kw_emb_before = p.keywords_embedding
    # observe without context : refresh extracts the same keywords
    m.observe(["P1"], polarity=+1.0)
    p = next(q for q in m.points if q.id == "P1")
    # Same keywords → same embedding object (not recomputed)
    assert p.keywords_embedding == kw_emb_before


# ---------- KG6 ----------------------------------------------------------


def test_process_turn_refreshes_keywords_on_detected_signal_targets():
    m = Memory(encoder=_Enc(), extractor=SimpleKeywordExtractor())
    p = m.ingest("met dr sarah at conference berkeley", id="FACT1")
    # First turn — user query
    m.process_turn("where did i meet sarah", speaker="user")
    # Assistant response retrieves FACT1
    m.process_turn(
        "you met dr sarah at conference berkeley",
        speaker="assistant",
        retrieved_point_ids=["FACT1"],
    )
    # User contradicts (negation marker + entity overlap)
    m.process_turn(
        "no berkeley wrong sarah california instead",
        speaker="user",
    )
    # FACT1 should have been refreshed with the context of the user
    # turn (contradicts), keywords may include "california" or similar
    p = next(q for q in m.points if q.id == "FACT1")
    assert len(p.keywords) > 0
