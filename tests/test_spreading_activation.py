"""
Concept C : associative spreading activation via the co-retrieval log.

retrieve() uses its base top hits as seeds and boosts / injects nodes that were
historically co-retrieved with them (journal), behind `spreading_weight`. This
is the associative half of ACT-R activation, distinct from the geometric
`use_spreading` (embedding neighbours). Default 0.0 = OFF. It surfaces
associatively-relevant nodes the cosine missed — the oblique win.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory


def _seed(m: Memory):
    for i in range(6):
        m.ingest(f"shared common topic chunk {i}", kind="FACT", id=f"P{i}")


def test_spreading_weight_zero_is_identity():
    q = "shared common topic"
    m = Memory(encoder=SimpleEncoder(), journal=Journal())
    _seed(m)
    base = [h["id"] for h in m.retrieve(q, k=3)]
    # forge an association, but with weight 0 the candidate set must not change
    for _ in range(5):
        m.record_retrieval([base[0], "P5"], query_text="assoc")
    assert [h["id"] for h in m.retrieve(q, k=3)] == base


def test_spreading_injects_cosine_missed_associate():
    q = "shared common topic"
    m = Memory(encoder=SimpleEncoder(), journal=Journal())
    _seed(m)
    base = [h["id"] for h in m.retrieve(q, k=3)]
    missed = next(f"P{i}" for i in range(6) if f"P{i}" not in base)
    for _ in range(5):
        m.record_retrieval([base[0], missed], query_text="assoc")
    m.spreading_weight = 1.0
    ranked = [h["id"] for h in m.retrieve(q, k=3)]
    assert missed in ranked and missed not in base       # injected by association


def test_spreading_without_journal_is_safe():
    m = Memory(encoder=SimpleEncoder())
    _seed(m)
    m.spreading_weight = 1.0                              # ignored (no journal)
    ids = [h["id"] for h in m.retrieve("shared common topic", k=3)]
    assert len(ids) == 3
