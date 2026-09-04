"""UserPromptSubmit hook (OPT-IN) : recall memories for every prompt.

OFF by default — set `METACOG_AUTO_RECALL=1` to enable. When on, each user
prompt runs the CHEAP recall path (`retrieve`, k=5, no walk, no LLM) against
the brain and injects the hits as `additionalContext`, so the model reads its
memory before it reads the prompt. On a gap (nothing sufficiently activated)
it injects nothing — silence is better than noise.

This is the one hook that adds latency to every turn (it opens the brain in a
fresh process), which is why it is opt-in ; the default discipline is the
model calling `retrieve` itself (SessionStart rule) with the gap hook as the
safety net.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import emit_context, load_memory, read_payload, resolve_storage  # noqa: E402

_K = 5


def main() -> None:
    if os.environ.get("METACOG_AUTO_RECALL", "").strip() not in ("1", "true", "yes"):
        return
    payload = read_payload()
    prompt = (payload.get("prompt") or "").strip()
    if len(prompt) < 8:
        return
    storage = resolve_storage(payload.get("cwd"))
    if not os.path.exists(os.path.expanduser(storage)):
        return                                  # no brain yet : nothing to recall
    mem = load_memory(storage)
    if mem.abstains(prompt):
        return
    hits = mem.retrieve(prompt, k=_K)
    if not hits:
        return
    lines = [f"- [{h['id']}] {str(h.get('content', ''))[:240]}" for h in hits]
    emit_context("UserPromptSubmit",
                 "[metacog memory] Relevant memories for this prompt (cheap recall ; "
                 "call `walk_start` for depth, `mark_useful` on what you rely on) :\n"
                 + "\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
