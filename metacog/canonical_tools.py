"""
Canonical tool manifest — the mnema-style tier separation, AFTER the agreed
transition.

Three role tiers + a removal bucket. The MCP EXPOSES only the external surface
(T1 primitives + the exposed T2 agent-tool machinery) ; everything in T3 stays a
callable function used INTERNALLY (a walk mode, an autonomic maintenance pass, a
mechanism another tool drives) but leaves the agent-facing surface.

  T1 CANONICAL  — exposed primitives that serve the memory : feed, ask, observe.
  T2 TOOL_TIER  — the agent-tool machinery (tools live as memory nodes) ; the
                  create/find/reuse subset is exposed.
  T3 INTERNAL   — mechanisms, walk modes, and autonomic passes. NOT exposed ;
                  orchestrated by walk_start / assemble_set / sleep. The bag
                  MECHANISM lives here too (assemble_set + event scans use it) —
                  only its manual driver tools left the surface.
  DEPRECATED    — dead surface, kept callable for back-compat but off the surface
                  and slated for removal.

`test_canonical_tools.py` asserts the tiers partition the live @app.tool() set
EXACTLY, so a new tool must be classified — organic growth cannot silently land
on the canonical surface.

Transition applied vs the previous flat classification:
  - push_code, assemble_set  -> promoted to T1 (a core feed / a core "list all").
  - process_turn, observe     -> T3 (fold into ingest_message / detector-driven).
  - sleep, save, audit, crystallize_skills, spawn_observators -> T3 AUTONOMIC.
  - all specialized retrieval / bag driver tools / observators -> T3 INTERNAL.
  - walk_next -> DEPRECATED (removal follow-up ; a test + a bench still name it).
"""

from __future__ import annotations

from typing import Optional, Set

# -- T1 : canonical exposed primitives ----------------------------------------
CANONICAL: Set[str] = {
    # feed
    "ingest", "ingest_message", "push_code",
    # ask
    "retrieve", "walk_start", "assemble_set",
    # observe state
    "stats", "inspect", "list_tags",
    # feedback (the supervised signal that calibrates decay)
    "mark_useful",
}

# -- T2 : agent-tool machinery (tools live as memory nodes) --------------------
#: The create / find / reuse subset is exposed ; get_session_skill stays internal.
TOOL_TIER: Set[str] = {
    "ensure_tool", "match_tool", "list_tools_learned", "build_skill",
    "get_session_skill",
}
_TOOL_TIER_EXPOSED: Set[str] = {
    "ensure_tool", "match_tool", "list_tools_learned", "build_skill",
}

# -- T3 : internal mechanisms / walk modes / autonomic passes -----------------
INTERNAL_RETRIEVAL: Set[str] = {          # the walk orchestrates these
    "presearch", "clue_search", "event_search", "scoped_answer",
    "scoped_list", "search_nodes", "reason", "walk_keepup",
}
INTERNAL_BAGS: Set[str] = {               # mechanism is canonical ; drivers hidden
    "collect", "bag", "bags", "bag_render",
}
INTERNAL_OBSERVATORS: Set[str] = {
    "declare_observator", "detect_polarized", "spawn_observators",
    "route", "list_communities",
}
AUTONOMIC: Set[str] = {                    # run by the system, not called outside
    "sleep", "save", "audit", "crystallize_skills",
}
INTERNAL_MISC: Set[str] = {
    "observe", "process_turn", "capture_code_tool", "ingest_skill",
}
INTERNAL: Set[str] = (INTERNAL_RETRIEVAL | INTERNAL_BAGS | INTERNAL_OBSERVATORS
                      | AUTONOMIC | INTERNAL_MISC)

# -- Dead surface (removal follow-up) -----------------------------------------
DEPRECATED: Set[str] = {"walk_next"}

#: Every tool name this manifest knows about (must equal the live @app.tool() set).
ALL_KNOWN: Set[str] = CANONICAL | TOOL_TIER | INTERNAL | DEPRECATED

# -- MCP surfaces (what is EXPOSED) -------------------------------------------
#: Light client — the server's own contract : feed every message + code, and ask.
EXTERNAL_LIGHT: Set[str] = {"ingest_message", "push_code", "walk_start"}

#: Powerful external agent — feed, ask, manage its OWN tools, observe state.
#: = T1 canonical + the exposed T2 subset. Everything else stays internal.
EXTERNAL: Set[str] = CANONICAL | _TOOL_TIER_EXPOSED

_SURFACES = {
    "all": None,                       # no restriction (default, backward-compat)
    "canonical": CANONICAL,
    "external": EXTERNAL,
    "external_light": EXTERNAL_LIGHT,
}


def surface_tools(surface: str) -> Optional[Set[str]]:
    """The set of tool names to EXPOSE for a named surface, or None for 'all'
    (no restriction). Unknown names raise. Used by build_app to gate which
    @app.tool()s are registered ; unexposed tools stay callable internally."""
    if surface not in _SURFACES:
        raise ValueError(
            f"unknown surface {surface!r} ; choose from {sorted(_SURFACES)}")
    s = _SURFACES[surface]
    return set(s) if s is not None else None


def classify(name: str) -> str:
    """Tier of a tool name : 'canonical' | 'tool_tier' | 'internal' |
    'deprecated' | 'unknown'."""
    if name in CANONICAL:
        return "canonical"
    if name in TOOL_TIER:
        return "tool_tier"
    if name in INTERNAL:
        return "internal"
    if name in DEPRECATED:
        return "deprecated"
    return "unknown"


def is_canonical(name: str) -> bool:
    """True for the T1 primitives on the exposed canonical surface."""
    return name in CANONICAL


def is_exposed(name: str, surface: str = "external") -> bool:
    """True if `name` is exposed on the given surface ('all' -> always True)."""
    exposed = surface_tools(surface)
    return exposed is None or name in exposed
