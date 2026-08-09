"""
Concept B : need-odds (ACT-R base-level activation) blended into ranking.

retrieve() blends the base relevance score with each candidate's need-odds
(recency×frequency of journal accesses) behind `recency_weight`. Default 0.0 =
OFF (order identical to before), matching mnema's default; >0 re-ranks toward
recently/often-accessed nodes.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory


def _seed(m: Memory):
    for i in range(6):
        m.ingest(f"shared common topic chunk {i}", kind="FACT", id=f"P{i}")


def test_recency_weight_zero_is_identity():
    q = "shared common topic"
    m_off = Memory(encoder=SimpleEncoder(), journal=Journal())
    _seed(m_off)
    base = [h["id"] for h in m_off.retrieve(q, k=6)]
    # hammer a node, but with weight 0 the ranking must NOT move
    for _ in range(8):
        m_off.record_retrieval([base[-1]], query_text="hammer")
    assert [h["id"] for h in m_off.retrieve(q, k=6)] == base


def test_recency_weight_lifts_hammered_node():
    q = "shared common topic"
    m = Memory(encoder=SimpleEncoder(), journal=Journal())
    _seed(m)
    base = [h["id"] for h in m.retrieve(q, k=6)]
    target = base[-1]                                   # base-last
    for _ in range(8):
        m.record_retrieval([target], query_text="hammer")
    m.recency_weight = 1.0
    ranked = [h["id"] for h in m.retrieve(q, k=6)]
    assert ranked.index(target) < base.index(target)   # moved up


def test_recency_weight_without_journal_is_safe():
    m = Memory(encoder=SimpleEncoder())                 # no journal
    _seed(m)
    m.recency_weight = 1.0                              # ignored (no journal)
    ids = [h["id"] for h in m.retrieve("shared common topic", k=6)]
    assert len(ids) == 6                                # no crash, still ranks
