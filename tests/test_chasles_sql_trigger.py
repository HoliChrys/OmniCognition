"""
Chasles collision driven by an SQL query.

Memory.compress_chasles branches on the journal : with one, the trigger is a
QUERY (chasles_path_candidates over the path_traversals table) rather than the
in-memory n_spike scan — the DB is the driver, like lateral. The logged
collision_event arms the per-signature refractory so re-running is a no-op until
the path is travelled again. Without a journal the in-memory scan still runs
(fallback, unchanged).
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory
from metacog.meta_walk import MetaWalker


class _FakeLLM:
    """Minimal LLM stub satisfying resolve_collision's contract."""

    def extract_common(self, texts):
        return "common"

    def remove_overlap(self, full, common):
        out = full.replace(common, "").strip()
        return out or f"distinct {full[:6]}"


def _mem_with_path(journal):
    m = Memory(encoder=SimpleEncoder(), journal=journal, llm=_FakeLLM())
    for i, nid in enumerate(["P0", "P1", "P2", "P3"]):
        m.ingest(f"shared common topic chunk {i}", kind="FACT", id=nid)
    return m


def test_compress_chasles_is_sql_driven_with_journal():
    j = Journal()
    m = _mem_with_path(j)
    # travel the SAME path twice -> a frequent path in the DB (no n_spike set)
    m.log_path(["P0", "P1", "P2", "P3"], t=1.0)
    m.log_path(["P0", "P1", "P2", "P3"], t=1.0)
    assert [c["signature"] for c in m.chasles_path_candidates(min_freq=2)] \
        == ["P0>P1>P2>P3"]
    events = m.compress_chasles()                 # trigger = the SQL query
    assert len(events) == 1
    assert any(p.id == events[0]["child_id"] for p in m.points)
    # refractory now armed by the logged collision_event -> re-run is a no-op
    assert m.compress_chasles() == []
    assert m.chasles_path_candidates(min_freq=2) == []


def test_no_journal_falls_back_to_in_memory_scan():
    m = Memory(encoder=SimpleEncoder(), llm=_FakeLLM())
    by_id = {}
    for i, nid in enumerate(["P0", "P1", "P2", "P3"]):
        p = m.ingest(f"shared common topic chunk {i}", kind="FACT", id=nid)
        by_id[nid] = p
    for i in range(4, 10):
        m.ingest(f"distractor {i}", kind="FACT", id=f"P{i}")
    for nid in ("P0", "P1", "P2", "P3"):
        by_id[nid].n_spike = 50
    by_id["P0"].tags.append("ref:P1")
    for tag in ("ref:P2", "ref:P9", "ref:P8"):
        by_id["P1"].tags.append(tag)
    for tag in ("ref:P3", "ref:P7", "ref:P6"):
        by_id["P2"].tags.append(tag)
    m._spike_total_hops = 30
    events = m.compress_chasles()                 # no journal -> n_spike scan
    assert len(events) == 1


def test_sql_trigger_skips_infrequent_path():
    j = Journal()
    m = _mem_with_path(j)
    m.log_path(["P0", "P1", "P2", "P3"], t=1.0)    # travelled once < min_freq 2
    assert m.compress_chasles() == []


def test_walk_feeds_path_traversals_without_double_counting_hops():
    """The walk's finalize hook logs the traversed chain (with_hops=False), so
    path frequency accrues while effective_spike stays driven by record_hop."""
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    a = m.ingest("alpha", kind="FACT", id="A")
    b = m.ingest("beta", kind="FACT", id="B")
    w = MetaWalker.__new__(MetaWalker)
    # exercise the finalize hook directly on a minimal, hand-built chain
    w.memory = m
    w.commit = True
    w._path_logged = False
    w._fact_star_chain = [a, b]
    w._action_star_chain = []
    w._log_traversed_path()
    rows = j.frequent_paths(min_len=2, min_freq=1)
    assert [r["signature"] for r in rows] == ["A>B"]
    # with_hops=False -> no phantom hops were logged
    assert j.effective_spike("B") == 0
