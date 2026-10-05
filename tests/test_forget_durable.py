"""
A forget survives a restart (TAC-323).

Before : the MCP `forget` tool only mutated the RAM state (no `save()`), and
`load` never replayed `forget_events` — a forgotten node came back in retrieval
after the next omni restart. Now (1) the tool saves through the atomic,
merge-on-stale write, and (2) `load` replays the journal's forget events the
pickle lost, setting the node INVALID (never deleted) unless it was reverted.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

from metacog.defaults import SimpleEncoder
from metacog.epistemic import EpistemicState
from metacog.memory import Memory

FACTS = {"apple": "apples grow on the orchard trees",
         "banana": "bananas ripen in the warm sun",
         "cherry": "cherries are picked in june"}


def _mem(store, journal=True):
    return Memory(encoder=SimpleEncoder(), storage_path=store,
                  journal_path="auto" if journal else None)


def _seed(store, journal=True):
    m = _mem(store, journal)
    for nid, txt in FACTS.items():
        m.ingest(txt, kind="FACT", id=nid)
    m.save()
    return m


def _served(m, query="apples grow on the orchard trees"):
    return [r["id"] for r in m.retrieve(query, k=10)]


def _state(m, nid):
    return next(p for p in m.points if p.id == nid).state


def _forget_via_mcp(mem, node_id, reason):
    async def go():
        from mcp.shared.memory import create_connected_server_and_client_session
        from metacog.mcp_server import build_app
        async with create_connected_server_and_client_session(
                build_app(memory=mem)) as s:
            await s.initialize()
            r = await s.call_tool("forget", {"node_id": node_id,
                                             "reason": reason})
            return json.loads(r.content[0].text)
    return asyncio.run(go())


def test_mcp_forget_holds_across_a_restart():
    """(a) forget through the MCP tool -> a NEW instance loading the same
    pickle does not serve the id. No journal : only the save can carry it."""
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "memory.pkl")
        m1 = _seed(store, journal=False)
        assert "apple" in _served(m1)
        assert _forget_via_mcp(m1, "apple", "user corrected")["forgotten"] == "apple"

        m2 = _mem(store, journal=False)                 # the restart
        assert _state(m2, "apple") is EpistemicState.INVALID
        assert "apple" not in _served(m2)
        assert any(p.id == "apple" for p in m2.points)  # never deleted


def test_load_replays_an_unmerged_forget_the_pickle_lost():
    """(b) a pickle written BEFORE the forget (the pre-fix path : forget_node
    without save) + a pending `forget_events` row -> load -> node INVALID."""
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "memory.pkl")
        m1 = _seed(store)
        m1.forget_node("apple", reason="user corrected")   # RAM + journal only
        assert [e["node_id"] for e in m1.journal.pending_forgets()] == ["apple"]
        m1.journal.close()

        m2 = _mem(store)                                 # the restart
        assert _state(m2, "apple") is EpistemicState.INVALID
        assert "invalidated" in next(p for p in m2.points if p.id == "apple").tags
        assert "apple" not in _served(m2)
        assert [e["id"] for e in m2._forget_log] == ["apple"]
        # the latent merge still has its event to consume
        assert [e["node_id"] for e in m2.journal.pending_forgets()] == ["apple"]


def test_load_replays_a_forget_lost_then_merged_by_sleep():
    """A forget lost at restart whose event a later sleep marked merged is
    still a forget : it is replayed too."""
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "memory.pkl")
        m1 = _seed(store)
        m1.forget_node("apple", reason="user corrected")
        # the old life : a restart served the pickle as written, then a sleep
        # merged the pending event
        m1.journal.mark_forget_merged(m1.journal.pending_forgets()[0]["id"])
        assert m1.journal.pending_forgets() == []
        m1.journal.close()

        m2 = _mem(store)
        assert _state(m2, "apple") is EpistemicState.INVALID


def test_load_does_not_replay_a_reverted_forget():
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "memory.pkl")
        m1 = _seed(store)
        m1.forget_node("apple", reason="user corrected")
        m1.save()
        assert m1.revert_merge("apple")["restored"] is True
        m1.save()
        m1.journal.close()

        m2 = _mem(store)
        assert _state(m2, "apple") is not EpistemicState.INVALID
        assert "apple" in _served(m2)


def test_load_does_not_replay_a_forget_the_pickle_already_holds():
    """The pickle saved after the forget is authoritative : its log entry
    (same t, or t = ts - 1 for entries written before the shared instant)
    marks the event as applied — no replay, no duplicate log entry."""
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "memory.pkl")
        m1 = _seed(store)
        m1.forget_node("apple", reason="user corrected")
        ev = m1.journal.pending_forgets()[0]
        assert m1._forget_log[-1]["t"] == ev["ts"]          # one instant
        m1._forget_log[-1]["t"] = ev["ts"] - 1.0            # legacy shape
        m1.save()
        m1.journal.close()

        m2 = _mem(store)
        assert len(m2._forget_log) == 1


def test_stale_writer_does_not_resurrect_a_saved_forget():
    """Two instances on one store : A forgets (and saves) ; B, loaded before,
    writes later. The merge-on-stale carries A's forget onto B's copy."""
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "memory.pkl")
        _seed(store, journal=False)
        a = _mem(store, journal=False)
        b = _mem(store, journal=False)
        _forget_via_mcp(a, "apple", "user corrected")
        b.ingest("dates are sweet", kind="FACT", id="date")
        b.save()
        assert _state(b, "apple") is EpistemicState.INVALID

        c = _mem(store, journal=False)
        assert _state(c, "apple") is EpistemicState.INVALID
        assert {"apple", "date"} <= {p.id for p in c.points}
