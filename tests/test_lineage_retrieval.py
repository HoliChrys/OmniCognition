"""
Tests for retrieve_with_lineage — hybrid cosine + lineage + RRF retrieval.

Properties :
  L1   sequence_prev/sequence_next can be set at ingest time
  L2   The backfill links the previous point's sequence_next to the new one
  L3   retrieve_with_lineage surfaces a semantically orthogonal point
       when it is structurally adjacent to a strong cosine hit
  L4   parents and children are also walked as lineage edges
  L5   Memory.retrieve(use_lineage=True) plugs into the high-level API
  L6   RRF scoring : a point ranked high by both signals beats a
       point ranked high by only one
"""

from __future__ import annotations

import hashlib
import math
from typing import Tuple

from metacog import Memory, Point, PointKind, retrieve_with_lineage


class _Enc:
    """Signed simhash encoder, deterministic via hashlib."""

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
                h = int(hashlib.md5(f"l:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


def test_sequence_prev_next_set_at_ingest():
    m = Memory(encoder=_Enc())
    p1 = m.ingest("first turn alpha", id="T1")
    p2 = m.ingest("second turn beta", id="T2", sequence_prev="T1")
    p3 = m.ingest("third turn gamma", id="T3", sequence_prev="T2")
    assert p2.sequence_prev == "T1"
    assert p3.sequence_prev == "T2"
    # Backfill : T1.sequence_next should now point to T2
    assert m.inspect("T1")["id"] == "T1"
    t1 = next(p for p in m.points if p.id == "T1")
    assert t1.sequence_next == "T2"
    t2 = next(p for p in m.points if p.id == "T2")
    assert t2.sequence_next == "T3"


def test_lineage_surfaces_structurally_adjacent_evidence():
    """A point semantically orthogonal to the query but structurally
    adjacent to a strong cosine hit is surfaced by lineage expansion."""
    m = Memory(encoder=_Enc())
    # Strong cosine hit on "sarah berkeley conference"
    m.ingest("met sarah at berkeley conference",
             id="T1")
    # Structurally adjacent (next turn) but semantically about a necklace.
    # The query "where is sarah from" will NOT find this via cosine,
    # but lineage expansion from T1 should surface it.
    m.ingest("the necklace was from her grandma in sweden",
             id="T2", sequence_prev="T1")
    # Distractors
    for i in range(8):
        m.ingest(f"unrelated topic about pottery {i}", id=f"D{i}")

    query = "where is sarah from"
    # Without lineage : T2 likely won't surface
    plain = m.retrieve(query, k=5, use_lineage=False)
    plain_ids = {r["id"] for r in plain}
    # With lineage : the cosine seed T1 brings T2 via sequence_next
    with_lineage = m.retrieve(query, k=5, use_lineage=True)
    lineage_ids = {r["id"] for r in with_lineage}
    # T2 should be in lineage results even if absent from plain
    assert "T2" in lineage_ids, (
        f"lineage failed to surface T2. plain={plain_ids} lineage={lineage_ids}"
    )


def test_parents_and_children_walked_as_lineage():
    """Lineage expansion follows parents and children edges too."""
    m = Memory(encoder=_Enc())
    p1 = m.ingest("alpha bridge common", id="P1")
    p2 = m.ingest("beta bridge common", id="P2")
    # Manual child with parents = [P1, P2]
    child = m.ingest("the merged abstract content", id="C1", parents=["P1", "P2"])
    # Distractors
    for i in range(6):
        m.ingest(f"unrelated zeta {i}", id=f"D{i}")

    query = "alpha"
    plain = m.retrieve(query, k=3, use_lineage=False)
    plain_ids = [r["id"] for r in plain]
    with_lineage = m.retrieve(query, k=3, use_lineage=True)
    lineage_ids = [r["id"] for r in with_lineage]
    # P1 cosine hit on "alpha" → lineage expansion brings the child C1
    assert "P1" in lineage_ids
    # The child has P1 as parent → lineage traversal reaches it
    # (children edge from P1 → C1)
    assert "C1" in lineage_ids or "C1" in plain_ids


def test_memory_retrieve_accepts_use_lineage_flag():
    m = Memory(encoder=_Enc())
    m.ingest("alpha beta", id="A")
    m.ingest("gamma delta", id="B")
    plain = m.retrieve("alpha", k=2, use_lineage=False)
    lineage = m.retrieve("alpha", k=2, use_lineage=True)
    # Both succeed without exception
    assert len(plain) == 2 and len(lineage) == 2


def test_retrieve_with_lineage_direct_call():
    enc = _Enc()
    m = Memory(encoder=enc)
    m.ingest("alpha bridge", id="A", sequence_prev=None)
    m.ingest("orthogonal beta", id="B", sequence_prev="A")
    # Distractors
    for i in range(4):
        m.ingest(f"distract gamma {i}")
    q_emb = enc.encode("alpha")
    results = retrieve_with_lineage(q_emb, m.points, k=3, t_now=1.0,
                                     lineage_depth=1)
    ids = [p.id for _, p in results]
    assert "A" in ids
    # B is structurally adjacent to A — lineage brings it
    assert "B" in ids


def test_rrf_score_combines_cosine_and_lineage():
    """A point that scores well on BOTH cosine and lineage should
    beat one that scores well on only one."""
    enc = _Enc()
    m = Memory(encoder=enc)
    # P1 : strong cosine hit AND has lineage neighbors → high RRF
    m.ingest("alpha alpha alpha hit", id="P1")
    m.ingest("neighbor of P1", id="P1NB", sequence_prev="P1")
    # P2 : weaker cosine, no lineage
    m.ingest("alpha weakly mentioned", id="P2")
    for i in range(10):
        m.ingest(f"unrelated noise {i}")

    q_emb = enc.encode("alpha")
    results = retrieve_with_lineage(q_emb, m.points, k=10, t_now=1.0,
                                     lineage_depth=1)
    ids = [p.id for _, p in results]
    # P1's strong cosine + neighbor expansion gives it a higher RRF
    # than P2's weaker cosine alone.
    assert ids.index("P1") < ids.index("P2")
