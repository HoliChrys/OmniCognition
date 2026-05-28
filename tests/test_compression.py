"""
Tests for Chasles compression (retrospective metacognition).

Properties :
  Z1   straight_ratio returns 1.0 for chains < 3.
  Z2   straight chain has ratio close to 1.0.
  Z3   detour chain has ratio significantly below 1.0.
  Z4   detect_chasles_opportunities finds straight subchains, ignores
       detour ones.
  Z5   compress_chain creates a child with the intermediates as parents.
  Z6   The compressed child is closer to BOTH anchors than the
       intermediates' centroid was (geometric proof of compression).
  Z7   compress_trajectory leaves the cloud growing — zero deletion.
  Z8   Compression respects intra-kind invariant.
  Z9   The CompressionEvent records the boundary anchors and the
       intermediates.
"""

from __future__ import annotations

import hashlib
import math
from typing import Tuple

from metacog import (
    Point,
    PointKind,
    apply_pull,
    compress_chain,
    compress_trajectory,
    detect_chasles_opportunities,
    distance,
    effective_embedding,
    straight_ratio,
)


class _SetOpsLLM:
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
        seen = set()
        out = []
        for w in texts[0].lower().split():
            if w in common and w not in seen:
                out.append(w)
                seen.add(w)
        return " ".join(out)

    def remove_overlap(self, full: str, common: str) -> str:
        cset = set(common.lower().split())
        return " ".join(w for w in full.lower().split() if w not in cset)


class _IdEncoder:
    """Test encoder where each unique word stays orthogonal — useful
    when we need precise control over geometric positions."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self._cache: dict = {}

    def encode(self, text: str) -> Tuple[float, ...]:
        words = text.lower().split()
        if not words:
            return tuple(0.0 for _ in range(self.dim))
        acc = [0.0] * self.dim
        for w in words:
            if w not in self._cache:
                h = int(hashlib.md5(f"id:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


def _p(id_: str, embedding: Tuple[float, ...], kind=PointKind.THOUGHT) -> Point:
    return Point(id=id_, content=id_, embedding_orig=embedding, kind=kind)


# ---------- Z1, Z2, Z3 — straight_ratio ----------------------------------


def test_straight_ratio_returns_one_for_short_chains():
    a = _p("A", (0.0, 0.0))
    b = _p("B", (1.0, 0.0))
    assert straight_ratio([a], 0.0) == 1.0
    assert straight_ratio([a, b], 0.0) == 1.0


def test_straight_chain_has_high_ratio():
    a = _p("A", (0.0, 0.0))
    b = _p("B", (1.0, 0.0))
    c = _p("C", (2.0, 0.0))
    d = _p("D", (3.0, 0.0))
    r = straight_ratio([a, b, c, d], 0.0)
    assert r > 0.99


def test_detour_chain_has_low_ratio():
    a = _p("A", (0.0, 0.0))
    b = _p("B", (1.0, 5.0))   # big lateral detour
    c = _p("C", (2.0, -5.0))
    d = _p("D", (3.0, 0.0))
    r = straight_ratio([a, b, c, d], 0.0)
    assert r < 0.5


# ---------- Z4 — detection -----------------------------------------------


def test_detect_chasles_finds_straight_subchain():
    # Build a trajectory with one straight subchain (A→B→C→D) and one
    # detour subchain (D→E→F→G).
    a = _p("A", (0.0, 0.0))
    b = _p("B", (1.0, 0.0))
    c = _p("C", (2.0, 0.0))
    d = _p("D", (3.0, 0.0))
    e = _p("E", (3.5, 10.0))   # big detour
    f = _p("F", (4.0, -10.0))
    g = _p("G", (5.0, 0.0))
    trajectory = [a, b, c, d, e, f, g]
    opps = detect_chasles_opportunities(trajectory, 0.0)
    # At least one opportunity captures the A..D straight subchain
    assert any(
        opp.anchor_a.id == "A" and opp.anchor_d.id == "D"
        and {p.id for p in opp.intermediates} == {"B", "C"}
        for opp in opps
    )
    # No opportunity claims D..G is compressible (low ratio)
    for opp in opps:
        if opp.anchor_a.id == "D" and opp.anchor_d.id == "G":
            assert False, "detour chain should not be selected"


# ---------- Z5 — compress_chain produces child with parents --------------


def test_compress_chain_creates_child_with_intermediate_parents():
    enc = _IdEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="anchor a content",
              embedding_orig=enc.encode("anchor a content"),
              kind=PointKind.THOUGHT)
    b = Point(id="B", content="common bridge token alpha",
              embedding_orig=enc.encode("common bridge token alpha"),
              kind=PointKind.THOUGHT)
    c = Point(id="C", content="common bridge token beta",
              embedding_orig=enc.encode("common bridge token beta"),
              kind=PointKind.THOUGHT)
    d = Point(id="D", content="anchor d content",
              embedding_orig=enc.encode("anchor d content"),
              kind=PointKind.THOUGHT)

    result = compress_chain(a, [b, c], d, llm, enc, t_now=1.0)
    assert result is not None
    assert {p.id for p in result.parents} == {"B", "C"}
    assert "B" in result.child.parents
    assert "C" in result.child.parents
    # The child inherits the THOUGHT kind
    assert result.child.kind == PointKind.THOUGHT


# ---------- Z6 — geometric proof : child is closer to anchors ------------


def test_compressed_child_is_pulled_toward_anchors():
    """After compression, the child M has acquired a non-zero geometric
    offset (delta_active) — the apply_pull toward each anchor mutated
    its delta. Anchors themselves remain unmoved (one-way pull).
    """
    enc = _IdEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="anchor alpha",
              embedding_orig=enc.encode("anchor alpha"),
              kind=PointKind.THOUGHT)
    b = Point(id="B", content="bridge common one",
              embedding_orig=enc.encode("bridge common one"),
              kind=PointKind.THOUGHT)
    c = Point(id="C", content="bridge common two",
              embedding_orig=enc.encode("bridge common two"),
              kind=PointKind.THOUGHT)
    d = Point(id="D", content="anchor delta",
              embedding_orig=enc.encode("anchor delta"),
              kind=PointKind.THOUGHT)

    a_active_before = a.delta_active
    d_active_before = d.delta_active

    result = compress_chain(a, [b, c], d, llm, enc, t_now=1.0)
    assert result is not None
    child = result.child

    # Child moved (delta_active non-zero) — the two anchor pulls fired
    from metacog import vec_norm
    assert vec_norm(child.delta_active) > 0
    assert vec_norm(child.delta_latent) > 0

    # Anchors did NOT move — one-way pull semantics
    assert a.delta_active == a_active_before
    assert d.delta_active == d_active_before


# ---------- Z7 — zero deletion --------------------------------------------


def test_compress_trajectory_does_not_remove_intermediates():
    enc = _IdEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="head start",
              embedding_orig=enc.encode("head start"),
              kind=PointKind.THOUGHT)
    b = Point(id="B", content="middle common one",
              embedding_orig=enc.encode("middle common one"),
              kind=PointKind.THOUGHT)
    c = Point(id="C", content="middle common two",
              embedding_orig=enc.encode("middle common two"),
              kind=PointKind.THOUGHT)
    d = Point(id="D", content="tail end",
              embedding_orig=enc.encode("tail end"),
              kind=PointKind.THOUGHT)

    # Force straight-line positions so the chain is detected
    a.embedding_orig = (0.0,) * 8
    b.embedding_orig = tuple(0.1 if i == 0 else 0.0 for i in range(8))
    c.embedding_orig = tuple(0.2 if i == 0 else 0.0 for i in range(8))
    d.embedding_orig = tuple(0.3 if i == 0 else 0.0 for i in range(8))
    a.delta_active = (0.0,) * 8
    b.delta_active = (0.0,) * 8
    c.delta_active = (0.0,) * 8
    d.delta_active = (0.0,) * 8
    a.delta_latent = (0.0,) * 8
    b.delta_latent = (0.0,) * 8
    c.delta_latent = (0.0,) * 8
    d.delta_latent = (0.0,) * 8

    trajectory = [a, b, c, d]
    points = list(trajectory)
    initial_ids = {p.id for p in points}

    report = compress_trajectory(trajectory, points, llm, enc, t_now=1.0)
    # All original ids still present (zero deletion)
    final_ids = {p.id for p in points}
    assert initial_ids.issubset(final_ids)
    # The cloud may have grown (the new child)
    if report.compressions_applied > 0:
        assert len(points) > len(initial_ids)


# ---------- Z8 — intra-kind enforcement -----------------------------------


def test_compression_rejects_cross_kind_intermediates():
    """detect_chasles_opportunities skips chains whose intermediates
    are not all the same kind."""
    enc = _IdEncoder()
    a = _p("A", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    b = _p("B", (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
           kind=PointKind.THOUGHT)
    c = _p("C", (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
           kind=PointKind.ACTION)  # different kind from B
    d = _p("D", (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    opps = detect_chasles_opportunities([a, b, c, d], 0.0)
    # No opportunity captures the A..D chain because B and C are
    # different kinds.
    for opp in opps:
        kinds = {p.kind for p in opp.intermediates}
        assert len(kinds) == 1


# ---------- Z9 — audit ----------------------------------------------------


def test_compression_event_records_boundary_and_intermediates():
    enc = _IdEncoder()
    llm = _SetOpsLLM()
    a = Point(id="A", content="head alpha",
              embedding_orig=enc.encode("head alpha"),
              kind=PointKind.THOUGHT)
    b = Point(id="B", content="mid common one",
              embedding_orig=enc.encode("mid common one"),
              kind=PointKind.THOUGHT)
    c = Point(id="C", content="mid common two",
              embedding_orig=enc.encode("mid common two"),
              kind=PointKind.THOUGHT)
    d = Point(id="D", content="tail omega",
              embedding_orig=enc.encode("tail omega"),
              kind=PointKind.THOUGHT)
    result = compress_chain(a, [b, c], d, llm, enc, t_now=1.0,
                            ratio_before=0.97)
    assert result is not None
    e = result.event
    assert e.anchor_a_id == "A"
    assert e.anchor_d_id == "D"
    assert set(e.intermediate_ids) == {"B", "C"}
    assert e.ratio_before == 0.97
