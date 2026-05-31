"""
Smoke tests for the MCP tool layer (metacog.mcp_server.build_app) coupled
to a live Memory.

Verifies the tool<->memory contract end to end over a real connected
client/server session : every tool is exposed, and the non-LLM tools
(ingest / retrieve / inspect / stats / sleep / audit / observe /
process_turn / save) round-trip correctly against the memory they wrap.

The LLM-driven tools (walk_start / walk_next / reason) are exercised
separately in the LoCoMo agent path ; here we keep the suite fast and
hermetic by not requiring a live model.
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


def test_all_tools_are_exposed():
    async def go():
        mem = Memory(encoder=SimpleEncoder())
        async with await _session(mem) as s:
            await s.initialize()
            listed = await s.list_tools()
            names = {t.name for t in listed.tools}
            expected = {
                "ingest", "retrieve", "walk_start", "walk_next", "reason",
                "sleep", "inspect", "audit", "stats", "save",
                "observe", "process_turn", "declare_observator",
                "detect_polarized", "spawn_observators", "route",
                "list_communities",
            }
            assert expected <= names, f"missing tools: {expected - names}"

    _run(go())


def test_ingest_then_retrieve_and_inspect():
    async def go():
        mem = Memory(encoder=SimpleEncoder())
        async with await _session(mem) as s:
            await s.initialize()
            r = await _call(s, "ingest",
                            content="Caroline studies counseling and mental health",
                            kind="FACT", id="D2")
            assert r["id"] == "D2"
            # memory actually mutated
            assert any(p.id == "D2" for p in mem.points)

            ins = await _call(s, "inspect", point_id="D2")
            assert ins["id"] == "D2" and ins["kind"] == "FACT"
            # the inspect tool now surfaces keywords + tags + spike count
            assert "keywords" in ins and "tags" in ins and "n_spike" in ins
            assert ins["keywords"]            # SimpleKeywordExtractor populated them

            rr = await _call(s, "retrieve", query="counseling mental health", k=3)
            ids = [x["id"] for x in rr] if isinstance(rr, list) else [rr.get("id")]
            assert "D2" in ids

    _run(go())


def test_stats_and_sleep_reflect_memory_state():
    async def go():
        mem = Memory(encoder=SimpleEncoder())
        async with await _session(mem) as s:
            await s.initialize()
            for i in range(3):
                await _call(s, "ingest", content=f"fact number {i}",
                            kind="FACT", id=f"P{i}")
            st = await _call(s, "stats")
            assert st["n_points"] == 3
            assert st["by_kind"].get("FACT") == 3

            # sleep is wired to BOTH geometric collision and lateral collapse
            sl = await _call(s, "sleep")
            assert "resolved_count" in sl
            assert "lateral_collided_groups" in sl    # lateral exposed via tool
            assert sl["lateral_collided_groups"] == 0  # disabled by default

    _run(go())


def test_audit_clean_on_fresh_memory():
    async def go():
        mem = Memory(encoder=SimpleEncoder())
        async with await _session(mem) as s:
            await s.initialize()
            await _call(s, "ingest", content="a clean fact", kind="FACT", id="D1")
            au = await _call(s, "audit")
            assert au["ok"] is True
            assert au["violations"] == []

    _run(go())
