"""
Skill/tool lifecycle : create-on-need -> store as memory node -> organic
retrieval -> reuse. Regression guard for the capability-cache match, which
missed on an IDENTICAL query because the tool's placeholder name led its
keyword list (position-weighted embedding weights early keywords most). The
query-derived scope must lead so a reuse lookup aligns.
"""

from __future__ import annotations

import os
import tempfile

from metacog.defaults import SimpleEncoder
from metacog.epistemic import Point, PointKind
from metacog.memory import Memory
from metacog.skills import is_tool, tools_in


class _FakeLLM:
    """No `generate` -> deterministic fallback synthesis."""


NEED = "reverse a string end to end"


def _mem():
    m = Memory(encoder=SimpleEncoder(), llm=_FakeLLM())
    m.skills_enabled = True
    return m


def test_ensure_tool_creates_a_contextualized_node():
    m = _mem()
    r = m.ensure_tool(NEED, how="iterate characters last to first")
    assert r["tool"] is not None and r["reused"] is False
    tool = next(p for p in m.points if p.id == r["tool"]["id"])
    assert is_tool(tool)
    # scope (query-derived) leads the keywords, name follows
    assert tool.keywords[0] != "tool"
    assert {"reverse", "string"} <= set(tool.keywords)


def test_identical_need_reuses_not_regenerates():
    m = _mem()
    r1 = m.ensure_tool(NEED, how="iterate characters last to first")
    r2 = m.ensure_tool(NEED)                       # same need
    assert r2["reused"] is True
    assert r2["tool"]["id"] == r1["tool"]["id"]
    assert len(tools_in(m.points)) == 1            # no duplicate generated


def test_unrelated_need_does_not_falsely_reuse():
    m = _mem()
    m.ensure_tool(NEED, how="iterate characters last to first")
    assert m.match_tool("bake a chocolate cake from scratch") is None


def test_canonical_retrieve_finds_the_tool():
    m = _mem()
    r = m.ensure_tool(NEED, how="iterate characters last to first")
    hits = [h["id"] for h in m.retrieve(NEED, k=3)]
    assert r["tool"]["id"] in hits                 # organic discovery, no special API


def test_reuse_survives_save_load():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pkl")
        m = _mem()
        m.storage_path = path
        rid = m.ensure_tool(NEED, how="iterate characters last to first")["tool"]["id"]
        m.save()
        m2 = Memory(encoder=SimpleEncoder(), llm=_FakeLLM(), storage_path=path)
        assert any(p.id == rid for p in tools_in(m2.points))
        assert m2.ensure_tool(NEED)["reused"] is True     # cache hit after reload


def test_recurrence_crystallizes_a_tool():
    m = _mem()
    enc = SimpleEncoder()
    pop = {"sort a list ascending": 12, "count the words": 2,
           "parse a date": 2, "trim whitespace": 2}
    for content, reps in pop.items():
        for i in range(reps):
            a = Point(id=f"{content[:5]}{i}", content=content,
                      embedding_orig=tuple(enc.encode(content)),
                      kind=PointKind.ACTION, keywords=content.split())
            m.record_action_generation(a, None, f"{content} — approach {i}", [])
    out = m.crystallize_skills()
    assert out["crystallized"] >= 1
    assert len(tools_in(m.points)) >= 1
