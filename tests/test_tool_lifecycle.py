"""
Tool lifecycle (#3) — the retract / feedback / correct half of the agent-tool
tier, which was previously append-only.

- retire_tool : soft-deprecate (match_tool stops reusing it) or hard-remove.
- report_tool : reinforce on success ; auto-retire after repeated failures.
- update_tool : correct a tool's body (re-embedded ; revives a deprecated one).
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.skills import is_tool, tools_in


class _FakeLLM:
    """No generate -> deterministic fallback synthesis."""


NEED = "reverse a string end to end"


def _mem_with_tool():
    m = Memory(encoder=SimpleEncoder(), llm=_FakeLLM())
    tid = m.ensure_tool(NEED, how="iterate characters last to first")["tool"]["id"]
    return m, tid


def test_retire_soft_stops_reuse():
    m, tid = _mem_with_tool()
    assert m.ensure_tool(NEED)["reused"] is True          # reusable before
    m.retire_tool(tid)                                     # soft-deprecate
    r = m.ensure_tool(NEED)
    assert r["reused"] is False                            # no longer reused
    assert r["tool"]["id"] != tid                          # a fresh tool instead


def test_retire_hard_removes():
    m, tid = _mem_with_tool()
    m.retire_tool(tid, hard=True)
    assert all(p.id != tid for p in m.points)


def test_report_ok_reinforces():
    m, tid = _mem_with_tool()
    before = next(p.n_uses for p in m.points if p.id == tid)
    out = m.report_tool(tid, ok=True)
    assert out["ok"] and out["n_uses"] == before + 1


def test_report_failures_auto_retire():
    m, tid = _mem_with_tool()
    assert m.report_tool(tid, ok=False)["retired"] is False   # 1st failure
    out = m.report_tool(tid, ok=False)                        # 2nd -> retire
    assert out["retired"] is True
    assert m.ensure_tool(NEED)["reused"] is False             # deprecated, skipped


def test_report_ok_clears_failure_streak():
    m, tid = _mem_with_tool()
    m.report_tool(tid, ok=False)
    m.report_tool(tid, ok=True)                               # resets streak
    assert m.report_tool(tid, ok=False)["retired"] is False   # only 1 again


def test_update_tool_rewrites_and_revives():
    m, tid = _mem_with_tool()
    m.retire_tool(tid)                                        # deprecated
    out = m.update_tool(tid, content="reverse via two-pointer swap")
    tool = next(p for p in m.points if p.id == tid)
    assert tool.content == "reverse via two-pointer swap"
    assert "deprecated" not in (tool.tags or [])             # update revived it
    assert out["n_revision"] == 1
    assert m.ensure_tool(NEED)["reused"] is True             # reusable again


def test_lifecycle_ops_noop_on_non_tool():
    m = Memory(encoder=SimpleEncoder(), llm=_FakeLLM())
    m.ingest("just a fact", kind="FACT", id="F")
    assert m.retire_tool("F")["retired"] is None
    assert m.report_tool("F", ok=True)["tool_id"] is None
    assert m.update_tool("F", content="x")["tool_id"] is None
