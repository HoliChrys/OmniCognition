"""
Offline decay-forgetting (#2) — the other half of need-odds.

Query time only decays (ranking) ; forgetting is an OFFLINE sleep pass that
prunes nodes gone cold, using the same need_odds under the fitted decay exponent.
Conservative + emergent : only non-tool nodes already accessed then gone cold
below (mean - std) of the accessed population ; tools and the untried are kept ;
dry-run by default.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.epistemic import Point, PointKind
from metacog.journal import Journal
from metacog.memory import Memory
from metacog.skills import TOOL_TAG, is_tool


def _mem_with_hot_and_cold():
    m = Memory(encoder=SimpleEncoder(), journal=Journal())
    for i in range(6):
        m.ingest(f"hot fact {i}", kind="FACT", id=f"H{i}")
    for i in range(2):
        m.ingest(f"cold fact {i}", kind="FACT", id=f"C{i}")
    enc = SimpleEncoder()
    tool = Point(id="tool::x", content="a reusable tool",
                 embedding_orig=tuple(enc.encode("a reusable tool")),
                 kind=PointKind.ACTION, tags=[TOOL_TAG])
    m.points.append(tool)
    now = 10_000
    for i in range(6):                              # hot : accessed just now
        m.journal.log_retrieval("q", [f"H{i}"], ts=now - 1)
    for i in range(2):                              # cold : accessed long ago
        m.journal.log_retrieval("q", [f"C{i}"], ts=now - 5_000)
    m.journal.log_retrieval("q", ["tool::x"], ts=now - 5_000)   # cold tool
    return m, now


def test_forget_dry_run_flags_cold_outliers_only():
    m, now = _mem_with_hot_and_cold()
    out = m.forget(t=now, apply=False)
    assert set(out["candidates"]) == {"C0", "C1"}   # tool protected, hot kept
    assert out["forgotten"] == []                    # dry-run removes nothing
    assert len(m.points) == 9


def test_forget_apply_prunes_cold_keeps_hot_and_tools():
    m, now = _mem_with_hot_and_cold()
    out = m.forget(t=now, apply=True)
    assert set(out["forgotten"]) == {"C0", "C1"}
    ids = {p.id for p in m.points}
    assert "C0" not in ids and "C1" not in ids
    assert all(f"H{i}" in ids for i in range(6))     # hot survive
    assert any(is_tool(p) for p in m.points)         # tool survives


def test_forget_never_touches_the_untried():
    """A never-accessed node is not a decay case -> kept."""
    m, now = _mem_with_hot_and_cold()
    m.ingest("brand new never retrieved", kind="FACT", id="NEW")
    out = m.forget(t=now, apply=True)
    assert "NEW" in {p.id for p in m.points}
    assert "NEW" not in out["forgotten"]


def test_forget_noop_without_journal():
    m = Memory(encoder=SimpleEncoder())
    for i in range(6):
        m.ingest(f"f{i}", kind="FACT", id=f"F{i}")
    assert m.forget(apply=True)["forgotten"] == []
    assert len(m.points) == 6


def test_forget_noop_with_too_few_accessed():
    m = Memory(encoder=SimpleEncoder(), journal=Journal())
    m.ingest("a", kind="FACT", id="A")
    m.journal.log_retrieval("q", ["A"], ts=1.0)
    assert m.forget(t=100, apply=True)["forgotten"] == []


def test_sleep_runs_forget_when_enabled():
    m, now = _mem_with_hot_and_cold()
    m.forget_enabled = True
    out = m.sleep(t=now)
    assert set(out.get("forgotten", [])) == {"C0", "C1"}
    # off by default -> sleep does not prune
    m2, now2 = _mem_with_hot_and_cold()
    assert "forgotten" not in m2.sleep(t=now2)
