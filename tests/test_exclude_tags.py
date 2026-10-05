"""TAC-930 (B2) : `retrieve(exclude_tags=…)` leaves tagged points out of the
SEARCH POOL, so they never take a top-k slot.

Measured on the live gate (Paperclip TAC-336) : tachikoma's pre-turn recall
filled its 5 slots with the session's own captured turns (scores 0.940–0.942),
all repeating a wrong answer, and the corpus note that contradicts them was no
longer recalled. Excluding `session:<id>` must bring the note back — a filter
applied AFTER the top-k would only empty the block.
"""

from __future__ import annotations

import asyncio
import json

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory


def _corpus() -> Memory:
    m = Memory(encoder=SimpleEncoder())
    m.ingest("proj-03 port to iris quality trust layer is documented", kind="FACT",
             id="NOTE")
    for i in range(5):
        p = m.ingest(f"proj-03 port to iris quality trust layer echo {i}", kind="FACT",
                     id=f"ECHO{i}")
        p.add_tag("session:Lobby:tachikoma.paralelle.GenAI:main", f"turn:t{i}")
    other = m.ingest("proj-03 port to iris quality trust layer other session",
                     kind="FACT", id="OTHER")
    other.add_tag("session:lobby:x:main")
    return m


def test_excluded_points_free_their_slots_for_the_rest():
    m = _corpus()
    q = "proj-03 port to iris quality trust layer"
    base = [h["id"] for h in m.retrieve(q, k=5)]
    assert sum(i.startswith("ECHO") for i in base) == 4 and "OTHER" not in base, base

    kept = [h["id"] for h in m.retrieve(
        q, k=5, exclude_tags=["session:lobby:tachikoma.paralelle.GenAI:main"])]
    assert not any(i.startswith("ECHO") for i in kept), kept
    assert {"NOTE", "OTHER"} <= set(kept)                      # another session stays


def test_exclusion_is_case_insensitive_and_spans_every_mode():
    m = _corpus()
    q = "proj-03 port to iris quality trust layer"
    tag = ["SESSION:LOBBY:TACHIKOMA.PARALELLE.GENAI:MAIN"]
    for kw in ({}, {"use_hybrid": True, "use_spreading": True, "use_lineage": True},
               {"use_lineage": True}):
        ids = [h["id"] for h in m.retrieve(q, k=7, exclude_tags=tag, **kw)]
        assert ids and not any(i.startswith("ECHO") for i in ids), (kw, ids)


def test_no_exclusion_changes_nothing():
    m = _corpus()
    q = "proj-03 port to iris quality trust layer"
    assert m.retrieve(q, k=7) == m.retrieve(q, k=7, exclude_tags=[]) \
        == m.retrieve(q, k=7, exclude_tags=None)


def test_the_mcp_tool_forwards_exclude_tags():
    from mcp.shared.memory import create_connected_server_and_client_session
    from metacog.mcp_server import build_app

    async def go():
        async with create_connected_server_and_client_session(
                build_app(memory=_corpus())) as s:
            await s.initialize()
            r = await s.call_tool("retrieve", {
                "query": "proj-03 port to iris quality trust layer", "k": 7,
                "exclude_tags": ["session:lobby:tachikoma.paralelle.GenAI:main"]})
            ids = [json.loads(b.text).get("id") for b in r.content]
            assert "NOTE" in ids and not any(str(i).startswith("ECHO") for i in ids), ids

    asyncio.run(go())
