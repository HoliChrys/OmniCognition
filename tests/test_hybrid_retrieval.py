"""
Tests for retrieve_hybrid — cosine on keywords + BM25 on content + RRF.

Properties :
  H1   SimpleKeywordExtractor returns top frequency content words,
       stopwords removed
  H2   bm25_score ranks documents containing rare query tokens higher
  H3   Memory.ingest auto-populates keywords + keywords_embedding
  H4   retrieve_hybrid surfaces entity-matched chunks (via keyword cosine)
  H5   retrieve_hybrid surfaces verbatim-matched chunks (via BM25)
  H6   retrieve_hybrid returns top-k after RRF fusion of both signals
  H7   default Memory.retrieve k=7 (matches the chosen budget)
  H8   use_hybrid + use_lineage composes correctly
"""

from __future__ import annotations

import hashlib
import math
from typing import Tuple

from metacog import (
    Memory,
    Point,
    PointKind,
    SimpleKeywordExtractor,
    bm25_score,
    retrieve_hybrid,
)


class _Enc:
    """Signed simhash encoder, deterministic via hashlib."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self._cache: dict = {}

    def encode(self, text: str) -> Tuple[float, ...]:
        words = text.lower().split()
        if not words:
            return tuple(0.0 for _ in range(self.dim))
        acc = [0.0] * self.dim
        for w in words:
            if w not in self._cache:
                h = int(hashlib.md5(f"h:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


# ---------- H1 -----------------------------------------------------------


def test_simple_keyword_extractor_removes_stopwords():
    ext = SimpleKeywordExtractor()
    kws = ext.extract("the cat sat on the mat with the dog", n=3)
    # Stopwords (the, on, with) removed ; content words remain
    assert "cat" in kws or "mat" in kws or "dog" in kws
    for stop in ("the", "on", "with"):
        assert stop not in kws


def test_simple_keyword_extractor_ranks_by_frequency():
    ext = SimpleKeywordExtractor()
    kws = ext.extract("alpha alpha alpha beta gamma", n=2)
    # alpha appears 3x, ranks first
    assert kws[0] == "alpha"


def test_simple_keyword_extractor_skips_short_words():
    ext = SimpleKeywordExtractor(min_length=4)
    kws = ext.extract("cat dog bird elephant rhino lion", n=3)
    for k in kws:
        assert len(k) >= 4


# ---------- H2 -----------------------------------------------------------


def test_bm25_ranks_rare_tokens_higher():
    enc = _Enc()
    pts = [
        Point(id="P1", content="generic generic generic generic",
              embedding_orig=enc.encode("generic")),
        Point(id="P2", content="generic with rare_token zaphod beeblebrox",
              embedding_orig=enc.encode("generic")),
        Point(id="P3", content="generic generic generic generic",
              embedding_orig=enc.encode("generic")),
    ]
    results = bm25_score("zaphod beeblebrox", pts, k_pool=10)
    assert len(results) >= 1
    assert results[0][1].id == "P2"


def test_bm25_returns_empty_on_no_match():
    enc = _Enc()
    pts = [
        Point(id="P1", content="alpha beta gamma",
              embedding_orig=enc.encode("alpha")),
    ]
    assert bm25_score("xenophon orinoco", pts) == []


# ---------- H3 -----------------------------------------------------------


def test_memory_ingest_auto_populates_keywords():
    m = Memory(encoder=_Enc())
    p = m.ingest("Caroline lives in Berkeley California", id="T1")
    assert len(p.keywords) > 0
    assert p.keywords_embedding is not None
    assert len(p.keywords_embedding) == m.encoder.dim


def test_memory_ingest_skips_keywords_if_extractor_none():
    m = Memory(encoder=_Enc(), extractor=None)
    p = m.ingest("Caroline lives in Berkeley", id="T1")
    assert p.keywords == []
    assert p.keywords_embedding is None


# ---------- H4, H5 — hybrid surfaces both signal types ------------------


def test_hybrid_surfaces_entity_matched_via_keyword_cosine():
    m = Memory(encoder=_Enc())
    # Entity-related : different surface form but same topic
    m.ingest("Caroline mentions her Berkeley counseling career", id="entity1")
    m.ingest("Caroline studied counseling at Berkeley", id="entity2")
    # Distractors
    for i in range(5):
        m.ingest(f"unrelated random pottery topic {i}", id=f"D{i}")

    results = m.retrieve("Berkeley counseling", k=7, use_hybrid=True)
    ids = {r["id"] for r in results}
    assert "entity1" in ids or "entity2" in ids


def test_hybrid_surfaces_verbatim_via_bm25():
    m = Memory(encoder=_Enc())
    # Verbatim rare token : the BM25 signal will dominate
    m.ingest("the answer is 7 May 2023 specifically", id="date1")
    # Distractors
    for i in range(10):
        m.ingest(f"unrelated chunk about other topic {i}", id=f"D{i}")

    results = m.retrieve("when is 7 May 2023", k=7, use_hybrid=True)
    ids = [r["id"] for r in results]
    assert "date1" in ids


# ---------- H6 -----------------------------------------------------------


def test_retrieve_hybrid_returns_at_most_k():
    m = Memory(encoder=_Enc())
    for i in range(20):
        m.ingest(f"chunk number {i} with content alpha beta",
                 id=f"P{i}")
    results = m.retrieve("alpha beta", k=7, use_hybrid=True)
    assert len(results) <= 7


# ---------- H7 -----------------------------------------------------------


def test_memory_retrieve_default_k_is_7():
    m = Memory(encoder=_Enc())
    for i in range(15):
        m.ingest(f"point {i} alpha beta", id=f"P{i}")
    results = m.retrieve("alpha")  # no k specified
    assert len(results) == 7


# ---------- H8 -----------------------------------------------------------


def test_hybrid_composes_with_lineage():
    enc = _Enc()
    m = Memory(encoder=enc)
    m.ingest("alpha bridge mention", id="A", sequence_prev=None)
    m.ingest("orthogonal but structurally adjacent", id="B",
             sequence_prev="A")
    for i in range(8):
        m.ingest(f"unrelated noise {i}", id=f"D{i}")

    # Hybrid + lineage : A surfaced by hybrid, B brought by lineage
    results = m.retrieve("alpha", k=7, use_hybrid=True, use_lineage=True)
    ids = {r["id"] for r in results}
    assert "A" in ids
    assert "B" in ids  # via sequence_next from A


# ---------- direct retrieve_hybrid call ----------------------------------


def test_retrieve_hybrid_direct_skips_points_without_keyword_embedding():
    enc = _Enc()
    ext = SimpleKeywordExtractor()
    pts = []
    # Point with keywords + embedding
    p1 = Point(
        id="P1", content="alpha beta gamma delta",
        embedding_orig=enc.encode("alpha beta gamma delta"),
        keywords=["alpha", "beta", "gamma"],
        keywords_embedding=tuple(enc.encode("alpha beta gamma")),
    )
    # Point WITHOUT keyword embedding (keywords_embedding=None)
    p2 = Point(
        id="P2", content="alpha rare_unique_token",
        embedding_orig=enc.encode("alpha"),
    )
    pts = [p1, p2]
    results = retrieve_hybrid(
        "alpha rare_unique_token", pts, k=5, t_now=0.0,
        encoder=enc, extractor=ext,
    )
    ids = [p.id for _, p in results]
    # Both surface : p1 via cosine on keywords, p2 via BM25 on content
    assert "P1" in ids
    assert "P2" in ids
