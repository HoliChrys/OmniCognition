"""SessionEnd hook : auto-capture the session into the brain, then sleep.

On stdin : the hook JSON ({session_id, transcript_path, cwd, reason, ...}).
It feeds the user's own typed messages into the memory as EPISODIC turns
(`ingest_message`, role=user, the session's id, each message's timestamp) —
deduplicated against what the session already indexed live — then runs one
`sleep` cycle (collision / decay-fit / forget-merge / wiki reconcile / tool
promotion) and saves. No LLM call : the cheap 90 % path.

Deliberately conservative : only the user's real messages (not tool results,
slash commands, system reminders, or assistant chatter). Silent + error-safe —
a hook must never break the session.

  python hooks/capture_session.py --dump <transcript.jsonl>   # parser-only smoke
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_memory, read_payload, resolve_storage, user_messages  # noqa: E402


def _already_indexed(mem, session_id: str) -> set:
    """Contents of the user turns this session already fed live (the server's
    own `ingest_message` prefixes `[ts] user: `)."""
    out = set()
    for p in mem.points:
        tags = p.tags or []
        if f"session:{session_id}" in tags and "role:user" in tags:
            body = p.content.split(": ", 1)
            out.add((body[1] if len(body) == 2 else p.content).strip())
    return out


def capture(transcript_path: str, storage: str, session_id: str,
            user_id: str) -> dict:
    msgs = user_messages(transcript_path)
    if not msgs:
        return {"captured": 0, "slept": False}
    mem = load_memory(storage)
    seen = _already_indexed(mem, session_id)
    n = 0
    for m in msgs:
        c = m["content"]
        if c in seen:
            continue
        mem.ingest_message(c, role="user", user_id=user_id, session_id=session_id,
                           timestamp=m.get("timestamp"), block=True)
        seen.add(c)
        n += 1
    slept = False
    try:
        mem.sleep()
        slept = True
    except Exception:
        pass
    mem.save()
    return {"captured": n, "slept": slept}


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--dump":
        for m in user_messages(sys.argv[2]):
            print("•", m["content"][:120].replace("\n", " "))
        return
    payload = read_payload()
    tp = payload.get("transcript_path")
    if not tp or not os.path.exists(tp):
        return
    storage = resolve_storage(payload.get("cwd"))
    session_id = str(payload.get("session_id") or "unknown")
    user_id = os.environ.get("METACOG_USER") or os.environ.get("USER") or "user"
    r = capture(tp, storage, session_id, user_id)
    if r["captured"]:
        print(f"[metacog] captured {r['captured']} user message(s) into {storage}",
              file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                  # never break the session
        print(f"[metacog] session capture skipped: {e}", file=sys.stderr)
