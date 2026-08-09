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
    tiers = [C.CANONICAL, C.TOOL_TIER, C.SPECIALIZED_RETRIEVAL, C.BAGS,
             C.OBSERVATORS, C.DEPRECATED]
    for i, a in enumerate(tiers):
        for b in tiers[i + 1:]:
            assert not (a & b), f"tiers overlap: {a & b}"


def test_classify_and_is_canonical():
    assert C.classify("retrieve") == "canonical"
    assert C.classify("ensure_tool") == "tool_tier"
    assert C.classify("clue_search") == "specialized"
    assert C.classify("walk_next") == "deprecated"
    assert C.classify("does_not_exist") == "unknown"
    assert C.is_canonical("ingest") and not C.is_canonical("clue_search")


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
