"""
Tests for the lineage neighbour expansion — the walk's "offer multiple
possibilities" rule when grounding is thin (the gold turn is likely just
out of the seed's reach).

Deterministic, no LLM : we drive `MetaWalker.neighbor_possibilities()` and
`grounding_is_thin()` directly on a sequence-linked memory.
"""

from __future__ import annotations

from metacog.memory import Memory
from metacog.defaults import SimpleEncoder
from metacog.meta_walk import MetaWalker


def _chain(n=8):
    """A conversation session D5:1..D5:n linked by sequence_prev."""
    m = Memory(encoder=SimpleEncoder())
    prev = None
    for i in range(1, n + 1):
        m.ingest(f"turn number {i} content", kind="FACT", id=f"D5:{i}",
                 sequence_prev=prev)
        prev = f"D5:{i}"
    return m


def test_gold_between_two_retrieved_neighbors_is_surfaced():
    """The exact cat3 failure shape : the walk retrieved D5:3 and D5:6 but
    not the wealth-implying gold D5:5 that sits between D5:4 and D5:6."""
    m = _chain()
    w = MetaWalker("anything", m, commit=False)
    w._fact_ids_cum = ["D5:3", "D5:6"]
    w._relevant_cum = []  # thin → gate open
    nb = {n["id"] for n in w.neighbor_possibilities(window=2)}
    assert "D5:5" in nb            # the gold, one hop from D5:4 / D5:6
    assert "D5:4" in nb


def test_neighbors_exclude_already_seen():
    m = _chain()
    w = MetaWalker("x", m, commit=False)
    w._fact_ids_cum = ["D5:3", "D5:6"]
    w._relevant_cum = []
    ids = {n["id"] for n in w.neighbor_possibilities(window=2)}
    assert "D5:3" not in ids and "D5:6" not in ids


def test_neighbors_labeled_and_capped():
    m = _chain(20)
    w = MetaWalker("x", m, commit=False)
    w._fact_ids_cum = [f"D5:{i}" for i in (2, 6, 10, 14, 18)]
    w._relevant_cum = []
    nb = w.neighbor_possibilities(window=2, cap=4)
    assert len(nb) == 4
    assert all(n["relevance"] == "neighbor" for n in nb)


def test_window_bounds_distance():
    m = _chain()
    w = MetaWalker("x", m, commit=False)
    w._fact_ids_cum = ["D5:4"]
    w._relevant_cum = []
    ids = {n["id"] for n in w.neighbor_possibilities(window=1)}
    # window=1 → only immediate neighbours D5:3 and D5:5
    assert ids == {"D5:3", "D5:5"}


def test_empty_when_nothing_retrieved():
    m = _chain()
    w = MetaWalker("x", m, commit=False)
    w._fact_ids_cum = []
    assert w.neighbor_possibilities() == []


def test_gate_thin_vs_solid_grounding():
    m = _chain()
    w = MetaWalker("x", m, commit=False)
    w._relevant_cum = []
    assert w.grounding_is_thin() is True
    w._relevant_cum = [m.points[0], m.points[1], m.points[2]]
    assert w.grounding_is_thin() is False


def test_forward_link_derived_from_sequence_prev():
    """Ingestion sets only sequence_prev ; the forward neighbour must still
    be reachable via the derived reverse index."""
    m = _chain()
    # D5:5's forward neighbour is D5:6 (whose sequence_prev == D5:5).
    w = MetaWalker("x", m, commit=False)
    w._fact_ids_cum = ["D5:5"]
    w._relevant_cum = []
    ids = {n["id"] for n in w.neighbor_possibilities(window=1)}
    assert "D5:6" in ids   # forward
    assert "D5:4" in ids   # backward
