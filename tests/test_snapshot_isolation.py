"""
Tests for `Memory.snapshot()` — per-QA isolation that lets a concurrent,
read-only harness safely hand each worker the FULL tool surface (including
state-mutating tools).

The contract:
  - heavy stateless resources (encoder / llm / extractor / …) are SHARED
    by identity (cheap, no rebuild);
  - all mutable state (points, ledgers, observators, counters, dicts) is
    DEEP-COPIED, so mutating the snapshot never touches the original;
  - `storage_path` is cleared so a snapshot `save()` can't clobber disk.
"""

from __future__ import annotations

from metacog.memory import Memory
from metacog.defaults import SimpleEncoder


def _mem():
    m = Memory(encoder=SimpleEncoder())
    m.ingest("alpha fact about cats", kind="FACT", id="A")
    m.ingest("beta fact about dogs", kind="FACT", id="B")
    return m


def test_snapshot_shares_heavy_resources_by_identity():
    m = _mem()
    s = m.snapshot()
    assert s.encoder is m.encoder
    assert s.llm is m.llm
    assert s.executor is m.executor
    assert s.extractor is m.extractor


def test_snapshot_deepcopies_points_list_and_each_point():
    m = _mem()
    s = m.snapshot()
    assert s.points is not m.points
    assert {p.id for p in s.points} == {p.id for p in m.points}
    for sp, op in zip(s.points, m.points):
        assert sp is not op  # each Point is its own object


def test_snapshot_clears_storage_path():
    m = _mem()
    m.storage_path = "/tmp/should_not_be_inherited.json"
    s = m.snapshot()
    assert s.storage_path is None


def test_mutating_snapshot_does_not_grow_original():
    m = _mem()
    n0 = len(m.points)
    s = m.snapshot()
    s.ingest("gamma — only in the snapshot", kind="FACT", id="C")
    assert len(m.points) == n0
    assert len(s.points) == n0 + 1
    assert not any(p.id == "C" for p in m.points)


def test_mutating_a_snapshot_point_does_not_leak_to_original():
    m = _mem()
    s = m.snapshot()
    s.points[0].add_tag("mutated:yes")
    assert "mutated:yes" in s.points[0].tags
    assert "mutated:yes" not in m.points[0].tags


def test_two_snapshots_are_mutually_isolated():
    """Models two concurrent QA workers : each mutates its own copy."""
    m = _mem()
    a = m.snapshot()
    b = m.snapshot()
    a.ingest("worker-A note", kind="FACT", id="WA")
    b.points[0].add_tag("worker:b")
    assert any(p.id == "WA" for p in a.points)
    assert not any(p.id == "WA" for p in b.points)
    assert "worker:b" in b.points[0].tags
    assert "worker:b" not in a.points[0].tags
    # and the original is untouched by either
    assert not any(p.id == "WA" for p in m.points)
    assert "worker:b" not in m.points[0].tags


def test_snapshot_isolates_counters_and_ledgers():
    """Counter / dict state must be copied, not aliased."""
    m = _mem()
    m._spike_total_hops = 5
    s = m.snapshot()
    s._spike_total_hops = 99
    s._atom_parent["x"] = "y"
    assert m._spike_total_hops == 5
    assert "x" not in m._atom_parent


def test_bench_agent_exposes_all_tools():
    """The LoCoMo meta-agent must no longer restrict the tool surface."""
    import benchmarks.locomo.mcp_meta_agent as A
    assert A._ALLOWED_TOOLS is None
