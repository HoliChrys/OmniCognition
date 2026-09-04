"""PostToolUse hook : force grounding when a metacog recall hits a GAP.

Fires after `mcp__metacog__retrieve` / `mcp__metacog__walk_start`. When the
tool response carries the gap sentinel (emitted in-band by the server when no
chunk is sufficiently activated — `Memory.abstains`, the ACT-R retrieval
threshold), it injects an imperative directive back into the model's context,
just-in-time. This is the forcing layer : a static rule loses to a strong
parametric prior exactly where you least want it to ; `additionalContext` at
the moment of need does not. Not a machine guarantee (no hook can force a tool
call) — the strongest nudge Claude Code exposes.

Silent + exit 0 on anything unexpected.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import GAP_SENTINEL, emit_context, read_payload, response_text  # noqa: E402

_DIRECTIVE = (
    "metacog recall returned a GAP (no relevant memory — the hits, if any, are "
    "background noise). You are REQUIRED, before answering from your own "
    "knowledge, to GROUND the facts the question turns on : ask the user, read "
    "the relevant files, or search — 'I already know this' does not exempt a "
    "famous topic, your parametric knowledge is unverified and stale here. Then "
    "`ingest` the durable findings (and `ingest_message` the exchange) so this "
    "gap is filled next time. Ground first, answer second. If grounding is "
    "impossible, say so explicitly instead of asserting."
)


def main() -> None:
    payload = read_payload()
    if GAP_SENTINEL not in response_text(payload):
        return
    emit_context("PostToolUse", _DIRECTIVE)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
