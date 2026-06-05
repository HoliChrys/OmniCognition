"""
Tests for the `presearch` batch-reconnaissance MCP tool and `list_tags`
namespace glossary.

`presearch` is the GATE the agent uses BEFORE spending a full walk on a
query: it returns the top-k nearest hits per query, without triggering
the uncertainty-governed walk or the knowledge-graph cascade. A weak
query (no hits / wrong hits) is rejected here, so the agent reformulates
and retries without paying the walk cost.

These tests use a real Memory (no LLM needed) — `retrieve` is the only
backend involved.
"""

from __future__ import annotations

import asyncio
import json

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory


def _run(coro):
    return asyncio.run(coro)


async def _session(mem):
    from mcp.shared.memory import create_connected_server_and_client_session
    from metacog.mcp_server import build_app

    app = build_app(memory=mem)
    return create_connected_server_and_client_session(app)


async def _call(s, name, **kw):
    r = await s.call_tool(name, kw)
    txt = r.content[0].text if r.content else "null"
    try:
        return json.loads(txt)
    except (ValueError, TypeError):
        return txt


def _mem():
    m = Memory(encoder=SimpleEncoder())
    # 4 documents, distinct topics — kNN must clearly separate them.
    m.ingest("Melanie painted a sunrise in 2022", kind="FACT", id="F1") \
        .add_tag("ref:date:2022", "session:s1", "person:melanie")
    m.ingest("Jon visited Paris in 2023", kind="FACT", id="F2") \
        .add_tag("ref:date:2023", "session:s2", "person:jon")
    m.ingest("Gina cooked pasta last weekend", kind="FACT", id="F3") \
        .add_tag("ref:date:2022", "session:s2", "person:gina")
    m.ingest("the quantum chromodynamics conjecture remains unproven",
             kind="FACT", id="F4").add_tag("topic:physics")
    return m


# ---------------------------------------------------------------------------
# presearch shape + batching
# ---------------------------------------------------------------------------


def test_presearch_returns_top_k_per_query():
    async def go():
        m = _mem()
        async with await _session(m) as s:
            await s.initialize()
            out = await _call(s, "presearch",
                              queries=["sunrise painting", "Paris visit"],
                              k_per_query=3)
            assert out["n_queries"] == 2
            assert len(out["results"]) == 2
            for r in out["results"]:
                assert "query" in r and "hits" in r
                assert len(r["hits"]) <= 3
                # Each hit carries the agent-facing trim
                for h in r["hits"]:
                    assert set(h.keys()) >= {"id", "content", "score", "kind", "tags"}

    _run(go())


def test_presearch_orders_by_relevance():
    async def go():
        m = _mem()
        async with await _session(m) as s:
            await s.initialize()
            out = await _call(s, "presearch",
                              queries=["Melanie sunrise"], k_per_query=4)
            top = out["results"][0]["hits"][0]
            # F1 is the on-topic doc; under any sane retrieval it ranks 1st.
            assert top["id"] == "F1"

    _run(go())


# ---------------------------------------------------------------------------
# presearch tag pre-filter (hard restrict, like scoped_answer)
# ---------------------------------------------------------------------------


def test_presearch_tag_prefilter_hard_restricts_pool():
    async def go():
        m = _mem()
        async with await _session(m) as s:
            await s.initialize()
            # Only session:s2 — F2 and F3 are eligible, F1 and F4 are not.
            out = await _call(s, "presearch",
                              queries=["any topic"], k_per_query=4,
                              tags=["session:s2"])
            hits = out["results"][0]["hits"]
            ids = {h["id"] for h in hits}
            assert ids <= {"F2", "F3"}, f"leak outside session:s2 — got {ids}"

    _run(go())


def test_presearch_tag_prefilter_hierarchical_ancestor():
    async def go():
        m = _mem()
        async with await _session(m) as s:
            await s.initialize()
            # `ref:date` (ancestor) → matches every F1/F2/F3 (2022 or 2023),
            # F4 has no ref:date and must be excluded.
            out = await _call(s, "presearch",
                              queries=["anything"], k_per_query=10,
                              tags=["ref:date"], match="exact")
            ids = {h["id"] for h in out["results"][0]["hits"]}
            assert "F4" not in ids
            assert ids <= {"F1", "F2", "F3"}

    _run(go())


def test_presearch_tag_prefilter_regex_range():
    async def go():
        m = _mem()
        async with await _session(m) as s:
            await s.initialize()
            out = await _call(s, "presearch",
                              queries=["anything"], k_per_query=10,
                              tags=[r"^ref:date:2022$"], match="regex")
            ids = {h["id"] for h in out["results"][0]["hits"]}
            assert ids <= {"F1", "F3"}, f"regex leak — got {ids}"

    _run(go())


def test_presearch_empty_queries_returns_empty_results():
    async def go():
        m = _mem()
        async with await _session(m) as s:
            await s.initialize()
            out = await _call(s, "presearch", queries=[], k_per_query=3)
            assert out["n_queries"] == 0 and out["results"] == []

    _run(go())


def test_presearch_restores_full_memory_after_call():
    """The pre-filter swaps `memory.points` for the scoped scope — it MUST
    restore the full point list after, or every subsequent retrieve would
    silently search the stale scope."""
    async def go():
        m = _mem()
        n_before = len(m.points)
        async with await _session(m) as s:
            await s.initialize()
            await _call(s, "presearch", queries=["x"], k_per_query=1,
                        tags=["session:s2"])
            assert len(m.points) == n_before, "memory.points not restored"

    _run(go())


# ---------------------------------------------------------------------------
# list_tags MCP exposure
# ---------------------------------------------------------------------------


def test_list_tags_returns_glossary_ordered_by_depth():
    async def go():
        m = _mem()
        async with await _session(m) as s:
            await s.initialize()
            out = await _call(s, "list_tags")
            ns = out["namespaces"]
            # Flat tags (depth 0) come first; depth-1 ('ref:date', etc.) after.
            depths = [t.count(":") for t in ns]
            assert depths == sorted(depths), f"not depth-ordered: {ns}"
            # No raw values (leaves) — `ref:date:2022` MUST NOT be present.
            assert all(not t.startswith("ref:date:") or t == "ref:date"
                       for t in ns)
            # Parent prefixes we expect from the fixture
            assert {"ref", "ref:date", "session", "person"} <= set(ns)

    _run(go())


def test_presearch_and_list_tags_are_registered_mcp_tools():
    async def go():
        m = Memory(encoder=SimpleEncoder())
        async with await _session(m) as s:
            await s.initialize()
            listed = await s.list_tools()
            names = {t.name for t in listed.tools}
            assert "presearch" in names
            assert "list_tags" in names

    _run(go())
