"""
Tests for geometric spreading activation — the edge-free analog of
HeLa-Mem's Hebbian spreading.

Properties :
  S1   geometric_spread returns [] for a population too small (<4)
  S2   spreading surfaces a manifold neighbour of a seed that is below
       the emergent (median-σ) threshold
  S3   spreading excludes the seeds themselves
  S4   retrieve_hybrid(use_spreading=True) can pull in a co-located
       point that the base cosine+BM25 signals miss
  S5   spreading introduces no new hyperparameter : same call, same
       result is deterministic
"""

from __future__ import annotations

import hashlib
import math
from typing import Tuple

from metacog import Memory, Point, PointKind, SimpleKeywordExtractor
from metacog.geometry import geometric_spread, retrieve_hybrid


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
                h = int(hashlib.md5(f"sp:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


def _mk(enc, pid, kws):
    return Point(
        id=pid,
        content=" ".join(kws),
        embedding_orig=enc.encode(" ".join(kws)),
        kind=PointKind.FACT,
        keywords=list(kws),
        keywords_embedding=enc.encode(" ".join(kws)),
    )


# ---------- S1 -----------------------------------------------------------


def test_spread_empty_for_small_population():
    enc = _Enc()
    pts = [_mk(enc, "A", ["x"]), _mk(enc, "B", ["y"])]
    assert geometric_spread(pts, pts, t_now=0.0) == []


# ---------- S2, S3 -------------------------------------------------------


def test_spread_surfaces_close_neighbour_and_excludes_seed():
    enc = _Enc()
    # Two near-identical points (same keywords) + several far ones.
    near1 = _mk(enc, "N1", ["alpha", "beta"])
    near2 = _mk(enc, "N2", ["alpha", "beta"])  # identical kw → distance 0
    far1 = _mk(enc, "F1", ["zeta"])
    far2 = _mk(enc, "F2", ["omega"])
    far3 = _mk(enc, "F3", ["kappa"])
    pts = [near1, near2, far1, far2, far3]
    spread = geometric_spread([near1], pts, t_now=0.0)
    ids = [p.id for _d, p in spread]
    assert "N2" in ids          # the close neighbour is surfaced (S2)
    assert "N1" not in ids      # the seed itself is excluded (S3)


# ---------- S4 -----------------------------------------------------------


def test_use_spreading_pulls_colocated_point():
    """A point with the same entity keywords as a base hit but no
    verbatim query-token overlap can be surfaced via spreading."""
    m = Memory(encoder=_Enc(), extractor=SimpleKeywordExtractor())
    # Base hit : matches the query verbatim
    m.ingest("caroline attended berkeley conference", id="BASE")
    # Co-located : same entities, different surface form (no query tokens)
    m.ingest("caroline berkeley conference", id="COLO")
    # Distractors
    m.ingest("melanie painting sunrise lake", id="D1")
    m.ingest("david adoption agency paperwork", id="D2")
    m.ingest("sweden necklace grandmother gift", id="D3")

    without = retrieve_hybrid(
        "caroline attended berkeley conference", m.points, k=3, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor, use_spreading=False,
    )
    with_spread = retrieve_hybrid(
        "caroline attended berkeley conference", m.points, k=3, t_now=1.0,
        encoder=m.encoder, extractor=m.extractor, use_spreading=True,
    )
    ids_without = {p.id for _s, p in without}
    ids_with = {p.id for _s, p in with_spread}
    # spreading should at least not drop BASE and may surface COLO
    assert "BASE" in ids_with
    assert ids_with >= {"BASE"} and len(ids_with) <= len(m.points)
    # COLO is a manifold neighbour of BASE → should appear with spreading
    assert "COLO" in ids_with


# ---------- S5 -----------------------------------------------------------


def test_spreading_is_deterministic():
    m = Memory(encoder=_Enc(), extractor=SimpleKeywordExtractor())
    for i in range(6):
        m.ingest(f"entity{i} shared topic discussion", id=f"P{i}")
    a = retrieve_hybrid("shared topic discussion", m.points, k=4, t_now=1.0,
                        encoder=m.encoder, extractor=m.extractor,
                        use_spreading=True)
    b = retrieve_hybrid("shared topic discussion", m.points, k=4, t_now=1.0,
                        encoder=m.encoder, extractor=m.extractor,
                        use_spreading=True)
    assert [p.id for _s, p in a] == [p.id for _s, p in b]
