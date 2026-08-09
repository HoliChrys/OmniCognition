"""
Canonical tool manifest — the mnema two-tier separation.

Verifies (1) the manifest partitions the live @app.tool() surface EXACTLY (so a
new tool must be classified, keeping organic growth off the canonical surface),
(2) the canonical-core primitives actually run the memory lifecycle, and (3) the
"organic tool = contextualized memory node" principle — agent tools live as TOOL
nodes retrievable by the agent, not as hardcoded tools.
"""

from __future__ import annotations

import ast
import os
import tempfile

from metacog import canonical_tools as C
from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory


def _live_tool_names():
    """Parse mcp_server.py and collect every @app.tool()-decorated function name
    (no mcp import needed)."""
    src = open("metacog/mcp_server.py").read()
    tree = ast.parse(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool":
                    names.add(node.name)
    return names


# -- 1. manifest <-> reality --------------------------------------------------

def test_manifest_partitions_live_tools_exactly():
    live = _live_tool_names()
    assert C.ALL_KNOWN == live, (
        f"manifest out of sync — unclassified: {live - C.ALL_KNOWN} ; "
        f"stale in manifest: {C.ALL_KNOWN - live}"
    )


def test_tiers_are_disjoint():
    tiers = [C.CANONICAL, C.TOOL_TIER, C.INTERNAL, C.DEPRECATED]
    for i, a in enumerate(tiers):
        for b in tiers[i + 1:]:
            assert not (a & b), f"tiers overlap: {a & b}"


def test_surfaces_are_subsets_of_live_tools():
    live = _live_tool_names()
    assert C.EXTERNAL <= live, f"EXTERNAL has unknown tools: {C.EXTERNAL - live}"
    assert C.EXTERNAL_LIGHT <= C.EXTERNAL <= C.ALL_KNOWN
    assert C.CANONICAL <= live


def test_surface_tools_resolution():
    assert C.surface_tools("all") is None                 # no restriction
    assert C.surface_tools("external") == C.EXTERNAL
    assert C.surface_tools("external_light") == C.EXTERNAL_LIGHT
    assert C.surface_tools("canonical") == C.CANONICAL
    # mutating the returned set must not corrupt the manifest
    C.surface_tools("external").add("zzz")
    assert "zzz" not in C.EXTERNAL


def test_unknown_surface_raises():
    import pytest
    with pytest.raises(ValueError):
        C.surface_tools("bogus")


def test_external_powerful_covers_feed_ask_tools_observe():
    # the powerful-agent contract: feed + ask + manage own tools + observe
    assert {"ingest_message", "push_code", "ingest"} <= C.EXTERNAL       # feed
    assert {"retrieve", "walk_start", "assemble_set"} <= C.EXTERNAL      # ask
    assert {"ensure_tool", "match_tool", "list_tools_learned"} <= C.EXTERNAL
    assert {"stats", "inspect", "list_tags"} <= C.EXTERNAL               # observe
    # internal machinery stays OFF the external surface
    assert not (C.INTERNAL_OBSERVATORS & C.EXTERNAL)
    assert not (C.INTERNAL_BAGS & C.EXTERNAL)
    assert not (C.INTERNAL_RETRIEVAL & C.EXTERNAL)
    assert not (C.AUTONOMIC & C.EXTERNAL)          # sleep/save/audit not exposed


def test_transition_moved_things_off_the_surface():
    # sleep/process_turn are no longer exposed; bag mechanism stays internal
    assert "sleep" not in C.EXTERNAL and "process_turn" not in C.EXTERNAL
    assert "walk_keepup" in C.INTERNAL and "reason" in C.INTERNAL
    # bag DRIVER tools are internal, but the mechanism is used by assemble_set (T1)
    assert C.INTERNAL_BAGS <= C.INTERNAL and "assemble_set" in C.CANONICAL


def test_classify_and_is_canonical():
    assert C.classify("retrieve") == "canonical"
    assert C.classify("assemble_set") == "canonical"      # promoted in transition
    assert C.classify("push_code") == "canonical"         # promoted in transition
    assert C.classify("ensure_tool") == "tool_tier"
    assert C.classify("clue_search") == "internal"        # was specialized
    assert C.classify("sleep") == "internal"              # autonomic now
    assert C.classify("bag") == "internal"
    assert C.classify("walk_next") == "deprecated"
    assert C.classify("does_not_exist") == "unknown"
    assert C.classify("mark_useful") == "canonical"       # feedback exposed (#1)
    assert C.is_canonical("ingest") and not C.is_canonical("clue_search")


def test_feedback_tool_is_exposed_externally():
    assert "mark_useful" in C.EXTERNAL                     # agent can give feedback


# -- 2. canonical core runs the memory ----------------------------------------

def test_canonical_core_runs_the_memory_lifecycle():
    """ingest -> retrieve -> sleep -> save -> load -> stats, using only canonical
    operations : the memory runs without any derived/agent tool."""
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "mem.pkl")
        m = Memory(encoder=SimpleEncoder(), storage_path=store, journal_path="auto")
        m.ingest("a canonical fact", kind="FACT", id="A")
        m.ingest("another fact", kind="FACT", id="B")
        assert m.retrieve("canonical fact", k=2)          # read primitive
        m.sleep()                                          # maintain primitive
        m.save()                                           # persist primitive
        m2 = Memory(encoder=SimpleEncoder(), storage_path=store, journal_path="auto")
        assert len(m2.points) == 2                         # survived
        assert m2.stats()["total"] >= 2 if "total" in m2.stats() else True


# -- 3. organic tool = contextualized memory node -----------------------------

def test_agent_tool_is_a_contextualized_retrievable_node():
    from metacog.skills import TOOL_TAG, is_tool, tools_in
    from metacog.epistemic import Point, PointKind
    m = Memory(encoder=SimpleEncoder())
    # the agent decides "to reverse a string I need this tool" -> a TOOL node
    # carrying its context (the need becomes scope keywords/content).
    enc = SimpleEncoder()
    tool = Point(id="tool::reverse", content="reverse a string end to end",
                 embedding_orig=tuple(enc.encode("reverse a string end to end")),
                 kind=PointKind.ACTION, tags=[TOOL_TAG, "tool_string"],
                 keywords=["reverse", "string"])
    m.points.append(tool)
    assert is_tool(tool)                                   # it's a tool node
    assert any(is_tool(p) for p in tools_in(m.points))     # discoverable as a tool
    # the agent finds it by querying the memory like any other node (contextual)
    hits = [h["id"] for h in m.retrieve("reverse a string", k=3)]
    assert "tool::reverse" in hits
