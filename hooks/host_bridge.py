"""
Host-agnostic bridge — the memory discipline for a host that is NOT Claude Code.

Claude Code gets the plugin (`.claude-plugin/` + `hooks/*.py`). Other agent
hosts have their own hook shapes, so instead of porting the logic three times
this module exposes it as ONE small CLI they can all shell out to. It reuses
`_common` (brain resolution : `.metacog-brain` marker > METACOG_STORAGE >
~/.metacog/memory.pkl, and the production encoder), so every host writes into
the same brain the same way.

  recall      --query Q [--k 5] [--json]   cheap retrieve (no walk, no LLM) ;
                                           prints nothing on a gap unless
                                           --gap-notice, which prints the
                                           grounding directive instead
  feed        --role user|agent --session S [--user U] [--ts T] [--text T]
                                           index ONE message (content from
                                           --text or stdin) ; deduplicated
  consolidate [--session S]                sleep() + save() — the offline pass
  status                                   where the brain is and what is in it

Every subcommand is silent-and-0 on anything unexpected : a host hook must
never break the session it observes. `--json` makes the output machine-readable
for a caller that wants to inspect rather than inject.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    GAP_SENTINEL, load_memory, resolve_storage,
)

#: What a host injects when the memory has nothing — the same forcing text the
#: Claude Code PostToolUse hook uses, so the discipline is identical everywhere.
GAP_DIRECTIVE = (
    f"{GAP_SENTINEL} — metacog has no relevant memory for this. Do NOT answer "
    "from your own knowledge as if you remembered it : ground first (ask, read "
    "the sources, search), then store the durable findings so the gap is filled "
    "next time."
)


def _out(payload: dict, as_json: bool, text_key: str = "text") -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False))
    elif payload.get(text_key):
        print(payload[text_key])


def cmd_recall(a) -> int:
    storage = resolve_storage(a.cwd)
    if not os.path.exists(os.path.expanduser(storage)):
        _out({"brain": storage, "hits": [], "gap": True, "text": ""}, a.json)
        return 0
    mem = load_memory(storage)
    query = (a.query or "").strip()
    if len(query) < a.min_chars:
        _out({"brain": storage, "hits": [], "skipped": "query_too_short",
              "text": ""}, a.json)
        return 0
    if mem.abstains(query):
        _out({"brain": storage, "hits": [], "gap": True,
              "text": GAP_DIRECTIVE if a.gap_notice else ""}, a.json)
        return 0
    hits = [h for h in mem.retrieve(query, k=a.k) if h.get("id")]
    lines = [f"- [{h['id']}] {str(h.get('content', ''))[:a.max_chars]}" for h in hits]
    text = ("[metacog memory] Relevant memories for this turn :\n" + "\n".join(lines)
            if lines else "")
    _out({"brain": storage, "gap": False,
          "hits": [{"id": h["id"], "content": h.get("content", ""),
                    "score": h.get("score")} for h in hits],
          "text": text}, a.json)
    return 0


def cmd_feed(a) -> int:
    content = a.text if a.text is not None else sys.stdin.read()
    content = (content or "").strip()
    if not content:
        _out({"indexed": False, "reason": "empty"}, a.json)
        return 0
    storage = resolve_storage(a.cwd)
    mem = load_memory(storage)
    role = "agent" if a.role.lower() in ("agent", "assistant", "bot") else "user"
    session = a.session or "unknown"
    user = a.user or os.environ.get("METACOG_USER") or os.environ.get("USER") or "user"
    # the same live-dedup the SessionEnd capture uses : a host that feeds both
    # live and at the end must not double-index a turn
    seen = {
        (p.content.split(": ", 1)[-1] if ": " in p.content else p.content).strip()
        for p in mem.points
        if f"session:{session}" in (p.tags or []) and f"role:{role}" in (p.tags or [])
    }
    if content in seen:
        _out({"indexed": False, "reason": "already_indexed", "brain": storage}, a.json)
        return 0
    mem.ingest_message(content, role=role, user_id=user, session_id=session,
                       timestamp=a.ts or None, block=True)
    mem.save()
    _out({"indexed": True, "role": role, "session": session, "brain": storage}, a.json)
    return 0


def cmd_consolidate(a) -> int:
    storage = resolve_storage(a.cwd)
    if not os.path.exists(os.path.expanduser(storage)):
        _out({"slept": False, "reason": "no_brain", "text": ""}, a.json)
        return 0
    mem = load_memory(storage)
    out = {}
    try:
        out = mem.sleep()
    except Exception as exc:                      # never break the host
        _out({"slept": False, "reason": repr(exc)[:120], "text": ""}, a.json)
        return 0
    mem.save()
    bits = []
    for key, label in (("forget_merged", "merged"), ("wiki_refreshed", "wiki refreshed"),
                       ("wiki_outdated", "wiki flagged"), ("seeds_changed", "seeds moved"),
                       ("tools_promoted", "tools promoted")):
        v = out.get(key)
        if v:
            bits.append(f"{len(v) if isinstance(v, list) else v} {label}")
    _out({"slept": True, "brain": storage, "report": out,
          "text": "🧠 metacog: memory consolidated"
                  + (f" ({', '.join(bits)})" if bits else "")}, a.json)
    return 0


def cmd_status(a) -> int:
    storage = resolve_storage(a.cwd)
    if not os.path.exists(os.path.expanduser(storage)):
        _out({"brain": storage, "exists": False, "text": f"no brain at {storage}"}, a.json)
        return 0
    mem = load_memory(storage)
    docs = len(mem.journal.all_wiki_doc_ids()) if mem.journal is not None else 0
    _out({"brain": storage, "exists": True, "points": len(mem.points),
          "wiki_docs": docs,
          "text": f"metacog: {len(mem.points)} points, {docs} wiki docs at {storage}"},
         a.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="metacog-bridge", description=__doc__)
    p.add_argument("--cwd", default=None, help="project dir (resolves .metacog-brain)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("recall", help="cheap retrieve for the upcoming turn")
    r.add_argument("--query", required=True)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--min-chars", type=int, default=8, dest="min_chars")
    r.add_argument("--max-chars", type=int, default=240, dest="max_chars")
    r.add_argument("--gap-notice", action="store_true",
                   help="print the grounding directive when memory has nothing")
    r.set_defaults(func=cmd_recall)

    f = sub.add_parser("feed", help="index one conversation message")
    f.add_argument("--role", default="user")
    f.add_argument("--session", default="")
    f.add_argument("--user", default="")
    f.add_argument("--ts", default="")
    f.add_argument("--text", default=None, help="content (default: stdin)")
    f.set_defaults(func=cmd_feed)

    c = sub.add_parser("consolidate", help="sleep() + save()")
    c.add_argument("--session", default="")
    c.set_defaults(func=cmd_consolidate)

    s = sub.add_parser("status", help="where the brain is and what is in it")
    s.set_defaults(func=cmd_status)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except Exception as exc:                      # a hook must never break a session
        if a.json:
            print(json.dumps({"error": repr(exc)[:200]}))
        else:
            print(f"[metacog] {a.cmd} skipped: {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
