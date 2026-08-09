"""
Canonical tool manifest — the mnema-style two-tier separation of the tool surface.

The MCP server grew 40 flat tools organically (each benchmark / subsystem added
its own). That is the wrong growth axis. The mnema model is:

  Tier 1 — CANONICAL primitives
      The fixed, minimal set the memory NEEDS to run. Same definitions used by
      internal processes and exposed over MCP to an external agent. This set does
      NOT grow organically ; new capability never lands here.

  Tier 2 — TOOL_TIER (the bridge to the organic tier)
      The primitives by which the AGENT creates and retrieves its OWN tools. Those
      agent tools are NOT hardcoded @app.tool()s — they live as TOOL nodes IN the
      memory (tag `tool`), contextualized by the task / keywords that spawned them
      ("ok, to do X I need this tool" -> a node carrying that context), and the
      agent finds them by querying the memory like any other node. Organic growth
      happens HERE, as data, not as code.

  Tier 3 — DERIVED (specialized retrieval regimes, bag-building, observators)
      Hardcoded today, but per the model these are candidates to migrate into
      memory-resident action-tools the agent selects/creates on demand — not
      permanent fixtures of the canonical surface.

`test_canonical_tools.py` asserts this manifest partitions the live @app.tool()
set EXACTLY, so a newly-added tool fails the test until it is classified — the
guard that keeps organic growth off the canonical surface.
"""

from __future__ import annotations

from typing import Set

# -- Tier 1 : canonical primitives (must exist to run the memory) --------------
CANONICAL: Set[str] = {
    # write
    "ingest", "ingest_message", "observe", "process_turn",
    # read
    "retrieve", "walk_start", "inspect", "stats", "list_tags",
    # maintain / persist
    "sleep", "save", "audit",
}

# -- Tier 2 : the agent-tool machinery (tools live as memory nodes) ------------
TOOL_TIER: Set[str] = {
    "ensure_tool", "match_tool", "list_tools_learned",
    "build_skill", "ingest_skill", "crystallize_skills",
    "get_session_skill", "capture_code_tool", "push_code",
}

# -- Tier 3 : derived / specialized (migration candidates -> memory nodes) -----
SPECIALIZED_RETRIEVAL: Set[str] = {
    "presearch", "clue_search", "event_search", "scoped_answer",
    "scoped_list", "search_nodes", "assemble_set", "reason", "walk_keepup",
}
BAGS: Set[str] = {"collect", "bag", "bags", "bag_render"}
OBSERVATORS: Set[str] = {
    "declare_observator", "detect_polarized", "spawn_observators",
    "route", "list_communities",
}

# -- Dead surface (remove) -----------------------------------------------------
DEPRECATED: Set[str] = {"walk_next"}

DERIVED: Set[str] = SPECIALIZED_RETRIEVAL | BAGS | OBSERVATORS
#: Every tool name this manifest knows about (must equal the live @app.tool() set).
ALL_KNOWN: Set[str] = CANONICAL | TOOL_TIER | DERIVED | DEPRECATED


def classify(name: str) -> str:
    """Return the tier of a tool name : 'canonical' | 'tool_tier' |
    'specialized' | 'bags' | 'observators' | 'deprecated' | 'unknown'."""
    if name in CANONICAL:
        return "canonical"
    if name in TOOL_TIER:
        return "tool_tier"
    if name in SPECIALIZED_RETRIEVAL:
        return "specialized"
    if name in BAGS:
        return "bags"
    if name in OBSERVATORS:
        return "observators"
    if name in DEPRECATED:
        return "deprecated"
    return "unknown"


def is_canonical(name: str) -> bool:
    """True for the fixed primitives that must stay on the MCP surface."""
    return name in CANONICAL
