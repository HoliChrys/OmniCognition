"""SessionStart hook : inject the memory discipline into the session.

mnema keeps this rule in a user-global CLAUDE.md the user must edit by hand ;
a plugin can carry it itself — a SessionStart hook's stdout lands in the
model's context at the start of every session (and after /clear or a resume),
with the brain this session will use. Deterministic, no LLM, no memory load.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import GAP_SENTINEL, read_payload, resolve_storage  # noqa: E402

_RULE = """[metacog memory] A persistent memory is attached (MCP server `metacog`, brain: {brain}). Standing discipline :
1. RECALL BEFORE ANSWERING anything that could depend on the user's context, past work, decisions or preferences : `retrieve(query)` first (cheap, no LLM) ; `walk_start(query)` only for oblique / multi-hop questions. A result carrying "{sentinel}" means memory has nothing — ground first (ask, read the sources, search), THEN `ingest` the durable findings.
2. FEED EVERY TURN : `ingest_message(content, role, user_id, session_id, timestamp)` for the user's messages and your own ; `push_code(...)` for every code block you produce.
3. REMEMBER what is durable : `ingest(fact)` for stable facts, decisions, preferences, constraints — never ephemeral chatter. Reasoning worth keeping goes in as kind=THOUGHT.
4. CORRECT, DON'T PILE UP : when the user corrects a remembered fact, `forget(node_id, reason, superseded_by=<new node>)` — append-only, reversible (`revert_merge`). Never re-ingest a contradiction next to the old fact.
5. RATE WHAT YOU USED : `mark_useful(retrieval_id, 0|1|2)` on the retrievals you actually relied on — this calibrates the memory's decay.
6. TOOLS ARE MEMORY : before deriving a procedure from scratch, `match_tool(query)` ; if absent, `ensure_tool(query, how)` and `report_tool(id, ok)` after use."""


def rule(brain: str) -> str:
    return _RULE.format(brain=brain, sentinel=GAP_SENTINEL)


def main() -> None:
    payload = read_payload()
    # SessionStart : plain stdout is injected as context (exit 0).
    print(rule(resolve_storage(payload.get("cwd"))))


if __name__ == "__main__":
    try:
        main()
    except Exception:                       # a hook must never break the session
        pass
