"""One-shot migration: mnema memories -> omni, per context (TAC-506).

Reads each mnema per-context store (``contexts/<ctx>/memory.db``) and ingests
every active node into the omni memory of THE SAME context, through the gated
MCP endpoint (``x-tachikoma-context``), so the transfer itself exercises the
production path — never the pickle, never a back door.

Two rules this script enforces, both asked for:

1. PER-CONTEXT ISOLATION: a mnema node of ``global`` lands in omni's ``global``
   memory, nothing else. The context is the STORE, not a tag we invent.
2. THE CONTEXT TAG RIDES ALONG: every ingested node carries ``ctx:<name>`` in
   its tags — so inside omni the origin context is queryable/filterable
   (``list_tags``/``scoped_answer``) even though the gate already isolates the
   stores. The tag is the durable witness of WHERE the memory came from.

Non-destructive: mnema stores are read-only here, until T5 decides the
bascule. Idempotent-ish: re-running re-ingests (omni dedups by collision in
``sleep()``, not at ingest) — the ``--limit`` flag exists for a dry sample.

Usage:
    python scripts/migrate_mnema_to_omni.py [--ctx global] [--limit 20] [--omit-content]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

MNEMA_CONTEXTS = Path.home() / "data_tachikoma" / "contexts"
OMNI_URL = "http://127.0.0.1:8788/mcp"
CTX_HEADER = "x-tachikoma-context"


def _post(payload: dict, headers: dict, timeout: int = 120):
    req = urllib.request.Request(
        OMNI_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream", **headers})
    return urllib.request.urlopen(req, timeout=timeout)


def _session(ctx: str) -> dict:
    headers = {CTX_HEADER: ctx}
    r = _post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "mnema-migration", "version": "0"}}}, headers)
    sid = r.headers.get("mcp-session-id", "")
    headers["mcp-session-id"] = sid
    _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, headers)
    return headers


def _call(headers: dict, name: str, arguments: dict) -> dict:
    r = _post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}, headers)
    body = r.read().decode()
    # SSE framing: the JSON-RPC line rides in a ``data:`` event.
    import re
    m = re.search(r"^data: (.*)$", body, re.M)
    if m:
        return json.loads(m.group(1))
    return json.loads(body)


def mnema_nodes(db_path: Path, limit: int | None) -> list[dict]:
    db = sqlite3.connect(str(db_path))
    rows = db.execute(
        "SELECT id, content, tags, source FROM nodes "
        "WHERE active=1 AND content != '' "
        "ORDER BY last_access DESC").fetchall()
    if limit:
        rows = rows[:limit]
    out = []
    for nid, content, tags_json, source in rows:
        tags: list[str] = []
        try:
            tags = [str(t) for t in json.loads(tags_json or "[]")][:4]
        except Exception:
            pass
        out.append({"id": nid, "content": content, "tags": tags,
                    "source": source or ""})
    return out


def migrate(ctx: str, limit: int | None, omit_content: bool) -> int:
    db_path = MNEMA_CONTEXTS / ctx / "memory.db"
    if not db_path.exists():
        print(f"[migrate] no mnema store for {ctx!r} at {db_path} — skipped")
        return 0
    nodes = mnema_nodes(db_path, limit)
    print(f"[migrate] {ctx}: {len(nodes)} active mnema nodes to transfer")

    headers = _session(ctx)
    ok = fail = 0
    for i, node in enumerate(nodes, 1):
        # THE CONTEXT TAG RIDES ALONG: ctx:<name> first, thematic tags after,
        # omni's ingest caps at its own schema (tags list is open vocabulary).
        tags = [f"ctx:{ctx}", *node["tags"]][:8]
        content = node["content"]
        if omit_content:
            content = content[:40] + "…"
        try:
            res = _call(headers, "ingest",
                        {"content": content, "kind": "FACT", "tags": tags})
            err = (res.get("result") or {}).get("isError")
            if err:
                fail += 1
                print(f"  [{i}] isError: {str(res)[:120]}")
            else:
                ok += 1
        except Exception as exc:  # noqa: BLE001 — one node never stops the run
            fail += 1
            print(f"  [{i}] mnema#{node['id']} failed: {exc}")
        if i % 50 == 0:
            print(f"  … {i}/{len(nodes)} (ok={ok} fail={fail})")
    print(f"[migrate] {ctx}: done — ingested={ok} failed={fail}")
    return fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default=None,
                    help="single context to migrate (default: every context "
                         "that has a mnema store)")
    ap.add_argument("--limit", type=int, default=None,
                    help="dry-run sample: first N nodes per context")
    ap.add_argument("--omit-content", action="store_true",
                    help="truncate content to 40 chars (smoke test)")
    args = ap.parse_args()

    if args.ctx:
        contexts = [args.ctx]
    else:
        contexts = sorted(
            p.parent.name for p in MNEMA_CONTEXTS.glob("*/memory.db"))
    print(f"[migrate] contexts: {contexts}")

    total_fail = 0
    for ctx in contexts:
        total_fail += migrate(ctx, args.limit, args.omit_content)
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
