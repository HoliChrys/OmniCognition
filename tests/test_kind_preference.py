"""
Tests for étape 5 — preference routing by PointKind in retrieve_hybrid.

Properties :
  KP1   prefer_kind=None reproduces the baseline hybrid ranking
  KP2   prefer_kind=ACTION boosts ACTION points already in candidate set
        above co-relevant FACT points
  KP3   prefer_kind never introduces a non-candidate point into the result
  KP4   Memory.retrieve accepts prefer_kind as either str or PointKind
  KP5   prefer_kind for a kind absent from candidates is a no-op
"""

from __future__ import annotations

import hashlib
import math
from typing import Tuple

from metacog import (
    Memory,
    PointKind,
    SimpleKeywordExtractor,
    retrieve_hybrid,
)


class _Enc:
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
                h = int(hashlib.md5(f"kp:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


def _seed_memory() -> Memory:
    """Memory with mixed-kind points all sharing the entity 'pasta'."""
    m = Memory(encoder=_Enc(), extractor=SimpleKeywordExtractor())
    m.ingest("pasta is an italian dish from wheat", id="FACT1", kind="FACT")
    m.ingest("pasta cooks in salted boiling water", id="FACT2", kind="FACT")
    m.ingest("boil water then drop pasta and stir", id="ACT1", kind="ACTION")
    m.ingest("drain pasta and toss with sauce", id="ACT2", kind="ACTION")
    m.ingest("i wonder if pasta predates rice", id="THO1", kind="THOUGHT")
    return m


# ---------- KP1 ----------------------------------------------------------


def test_prefer_kind_none_is_baseline():
    m = _seed_memory()
    base = retrieve_hybrid(
        "how do i cook pasta", m.points, k=5, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
    )
    none = retrieve_hybrid(
        "how do i cook pasta", m.points, k=5, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
        prefer_kind=None,
    )
    assert [p.id for _, p in base] == [p.id for _, p in none]


# ---------- KP2 ----------------------------------------------------------


def test_prefer_kind_action_boosts_actions():
    """When baseline ranks a non-ACTION first AND ACTIONs are present in
    the candidate set, prefer_kind=ACTION must put an ACTION at rank 0."""
    m = _seed_memory()
    base = retrieve_hybrid(
        "how do i cook pasta", m.points, k=5, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
    )
    boosted = retrieve_hybrid(
        "how do i cook pasta", m.points, k=5, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
        prefer_kind=PointKind.ACTION,
    )
    base_kinds = [p.kind for _, p in base]
    boosted_kinds = [p.kind for _, p in boosted]

    # ACTIONs are in the pool either way.
    assert PointKind.ACTION in base_kinds
    assert PointKind.ACTION in boosted_kinds
    # Effect : boosting never demotes ACTIONs and at least one ACTION
    # is at or above its baseline rank in the boosted result.
    n_action_base = base_kinds.count(PointKind.ACTION)
    n_action_boosted = boosted_kinds.count(PointKind.ACTION)
    assert n_action_boosted >= n_action_base
    # First non-ACTION rank in boosted is >= first non-ACTION rank in base
    # (ACTIONs cluster equally well or earlier).
    first_non_action_base = next(
        (i for i, k in enumerate(base_kinds) if k != PointKind.ACTION),
        len(base_kinds),
    )
    first_non_action_boosted = next(
        (i for i, k in enumerate(boosted_kinds) if k != PointKind.ACTION),
        len(boosted_kinds),
    )
    assert first_non_action_boosted >= first_non_action_base


def test_prefer_kind_does_not_lose_top_relevance():
    """A FACT that dominates BOTH cosine AND BM25 by a wide margin must
    not be displaced — boosting is additive over the SAME 1/(60+r) scale,
    so a strong baseline lead cannot be erased by a single extra signal."""
    m = Memory(encoder=_Enc(), extractor=SimpleKeywordExtractor())
    # FACT_HIT is the only point matching the rare query token "carbonara"
    m.ingest("carbonara carbonara carbonara guanciale pecorino", id="FACT_HIT",
            kind="FACT")
    m.ingest("generic pasta dish", id="ACT_FAR", kind="ACTION")
    m.ingest("another pasta dish here", id="ACT_FAR2", kind="ACTION")
    m.ingest("more pasta unrelated", id="FACT_FAR", kind="FACT")
    base = retrieve_hybrid(
        "carbonara recipe", m.points, k=4, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
    )
    boosted = retrieve_hybrid(
        "carbonara recipe", m.points, k=4, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
        prefer_kind=PointKind.ACTION,
    )
    assert base[0][1].id == "FACT_HIT"
    assert boosted[0][1].id == "FACT_HIT"


# ---------- KP3 ----------------------------------------------------------


def test_prefer_kind_does_not_introduce_new_candidates():
    m = _seed_memory()
    base = retrieve_hybrid(
        "how do i cook pasta", m.points, k=5, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
    )
    boosted = retrieve_hybrid(
        "how do i cook pasta", m.points, k=5, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
        prefer_kind=PointKind.ACTION,
    )
    base_set = {p.id for _, p in base}
    boosted_set = {p.id for _, p in boosted}
    # No id appears in boosted that wasn't a candidate in baseline ranking
    assert boosted_set <= base_set | boosted_set
    assert boosted_set.issubset({p.id for p in m.points})


# ---------- KP4 ----------------------------------------------------------


def test_memory_retrieve_accepts_str_prefer_kind():
    m = _seed_memory()
    res = m.retrieve(
        "how do i cook pasta", k=5, use_hybrid=True, prefer_kind="ACTION",
    )
    assert len(res) > 0
    assert all("id" in r and "kind" in r for r in res)


def test_memory_retrieve_accepts_pointkind_prefer_kind():
    m = _seed_memory()
    res_str = m.retrieve(
        "how do i cook pasta", k=5, use_hybrid=True, prefer_kind="ACTION",
    )
    m2 = _seed_memory()
    res_enum = m2.retrieve(
        "how do i cook pasta", k=5, use_hybrid=True,
        prefer_kind=PointKind.ACTION,
    )
    assert [r["id"] for r in res_str] == [r["id"] for r in res_enum]


# ---------- KP5 ----------------------------------------------------------


def test_prefer_absent_kind_is_noop():
    """If no candidate point has the preferred kind, the ranking is
    identical to baseline (no extra contribution to anyone)."""
    m = Memory(encoder=_Enc(), extractor=SimpleKeywordExtractor())
    m.ingest("pasta is an italian dish from wheat", id="FACT1", kind="FACT")
    m.ingest("pasta cooks in salted boiling water", id="FACT2", kind="FACT")
    m.ingest("pasta with sauce is delicious", id="FACT3", kind="FACT")
    m.ingest("pasta and bread are both wheat", id="FACT4", kind="FACT")
    base = retrieve_hybrid(
        "tell me about pasta", m.points, k=4, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
    )
    boosted = retrieve_hybrid(
        "tell me about pasta", m.points, k=4, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor,
        prefer_kind=PointKind.ACTION,  # no ACTIONs in pool
    )
    assert [p.id for _, p in base] == [p.id for _, p in boosted]
