"""D10 (Paperclip TAC-2465) — move the mutable STATE of the served `.journal.db`
files into the ontology (`memory_wiki_doc` records, the store's `okf_vocab` /
`ref_remaps`), through the API, idempotently ; archive each file first.

For each store (``--stores ctx[,ctx/accounts/acc…]`` under ``--root``) :

1. SNAPSHOT. The file is read through SQLite's backup API into memory (a
   consistent snapshot even while the gate writes). With ``--apply`` the same
   snapshot is written to ``<archive>/<store>/memory.pkl.journal.db`` first.
   Nothing is ever deleted : the original stays where it is ; an existing
   archive is never overwritten.
2. READ what the store already holds (``GET /api/memory/points:wiki``, the load
   of `Neo4jStateJournal`).
3. SEND, ADD-ONLY, every doc the store holds NO record for (a deleted doc's
   history included), and the store-wide lists if the store's are still empty.
   A doc held on both sides with another content is REPORTED (``differ``),
   never overwritten : after the switch the store is the truth, and a stale
   file must not revert what the gate wrote there.
4. VERIFY on a fresh load : every persisted table, row for row, against the
   snapshot (``equal`` / the tables that are not). A second run sends nothing.

``tags`` is not migrated : it is the index of the points' ``memory_tags``, which
the points' records already carry ; the gate rebuilds it at open. ``--tags``
reports the index of the file against the one the points give.

Dry run by default ; ``--apply`` archives and sends. The bearer is the gate's
(``OMNI_STORE_TOKEN`` / ``OMNI_STORE_TOKEN_FILE``), the API ``TACHIKOMA_API_URL``.

    python scripts/migrate_state_to_neo4j.py --root ~/data_tachikoma/omni/contexts \\
        --archive ~/data_tachikoma/omni/state-archive/2026-11-08 \\
        --stores global,tachikoma.paralelle.GenAI [--apply] [--tags]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any, Callable, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metacog.journal import _LATE_COLUMNS, _SCHEMA, Journal  # noqa: E402
from metacog.journal_state_neo4j import (PERSISTED_TABLES, STORE_TABLES,  # noqa: E402
                                         Neo4jStateJournal, _rows, all_doc_ids,
                                         decode_doc, decode_store, encode_doc,
                                         encode_store)

JOURNAL_FILE = "memory.pkl.journal.db"


def scope(store: str) -> tuple:
    ctx, _, account = store.partition("/accounts/")
    return ctx, account


def snapshot(src: str) -> Journal:
    """A consistent in-memory copy of the file, its schema brought up to date (the
    backup replaces the schema : the idempotent DDL and late columns run again)."""
    j = Journal(":memory:")
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        s.backup(j.conn)
    finally:
        s.close()
    j.conn.executescript(_SCHEMA)
    for table, col, decl in _LATE_COLUMNS:
        have = {r[1] for r in j.conn.execute(f"PRAGMA table_info({table})")}
        if col not in have:
            j.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    j.conn.commit()
    return j


def archive(snap: Journal, dst: str) -> bool:
    """Write the snapshot to `dst`. False if `dst` already exists (never overwritten)."""
    if os.path.exists(dst):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    d = sqlite3.connect(dst)
    try:
        snap.conn.backup(d)
    finally:
        d.close()
    return True


def counts(conn: Any) -> Dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("tags", *PERSISTED_TABLES)}


def compare(src: Any, dst: Any) -> Dict[str, Any]:
    """Every persisted table as a multiset of rows, and every doc's own rows IN ORDER.

    The order ACROSS docs is not kept (a load inserts by doc id) and no read depends on
    it : every cross-doc query of `Journal` has its ORDER BY. The order within a doc is
    (`okf_fields_of` sorts by key only, list values keep their row order)."""
    def bag(conn: Any, t: str) -> list:
        return sorted(json.dumps(r, sort_keys=True) for r in _rows(conn, t))
    differ = [t for t in PERSISTED_TABLES if bag(src, t) != bag(dst, t)]
    docs = [d for d in sorted(set(all_doc_ids(src)) | set(all_doc_ids(dst)))
            if encode_doc(src, d) != encode_doc(dst, d)]
    return {"equal": not differ and not docs, "tables_differ": differ,
            "docs_differ": docs[:20]}


def tag_report(snap: Journal, client: Any) -> Dict[str, int]:
    status, body = client._call("GET", "/api/memory/points")
    if status != 200:
        return {"error": status}
    pts = body.get("points") or []
    ids = {p["point_id"] for p in pts}
    neo = {(p["point_id"], t) for p in pts for t in (p["fields"].get("memory_tags") or [])
           if isinstance(t, str) and t}
    sq = {tuple(r) for r in snap.conn.execute("SELECT node_id, tag FROM tags")}  # Row != tuple
    only_sq = sq - neo
    return {"file": len(sq), "points": len(neo), "only_file": len(only_sq),
            "only_file_not_a_point": sum(1 for n, _ in only_sq if n not in ids),
            "only_points": len(neo - sq)}


def migrate_store(store: str, *, root: str, archive_dir: Optional[str], apply: bool,
                  client_for: Callable[[str, str], Any], tags: bool = False
                  ) -> Dict[str, Any]:
    src_path = os.path.join(root, store, JOURNAL_FILE)
    if not os.path.exists(src_path):
        return {"store": store, "absent": src_path}
    snap = snapshot(src_path)
    ctx, account = scope(store)
    client = client_for(ctx, account)
    held = Neo4jStateJournal(client)
    src_ids = all_doc_ids(snap.conn)
    to_send = [d for d in src_ids if d not in held._known]
    differ = [d for d in src_ids if d in held._known
              and encode_doc(snap.conn, d) != held._doc(d)]
    only_store = sorted(set(held._known) - set(src_ids))
    src_lists = encode_store(snap.conn)
    held_lists = encode_store(held.conn)
    store_send = {f: v for f, v in src_lists.items()
                  if v != "[]" and held_lists[f] == "[]"}
    store_differ = [f for f, v in src_lists.items()
                    if held_lists[f] != "[]" and v != held_lists[f]]
    out: Dict[str, Any] = {
        "store": store, "before": {"file": counts(snap.conn), "neo4j": counts(held.conn)},
        "to_send": {"docs": len(to_send), "store_lists": sorted(store_send)},
        "differ": differ, "store_differ": store_differ, "only_in_neo4j": only_store,
    }
    if tags:
        out["tags"] = tag_report(snap, client)
    if not apply:
        return out
    if archive_dir:
        dst = os.path.join(archive_dir, store, JOURNAL_FILE)
        out["archive"] = dst + ("" if archive(snap, dst) else " (already there)")
    if to_send or store_send:
        with held._lock:
            for d in to_send:
                fields, discarded = encode_doc(snap.conn, d)
                decode_doc(held.conn, d, fields, discarded)
            if store_send:
                decode_store(held.conn, {f: store_send.get(f, "[]")
                                         for f in STORE_TABLES.values()})
            held.conn.commit()
        if not held.flush():
            out["sent"] = "FAILED (see the log)"
            return out
    out["sent"] = {"docs": held.stats["docs_sent"], "store": held.stats["store_sent"]}
    fresh = Neo4jStateJournal(client_for(ctx, account))
    out["after"] = {"neo4j": counts(fresh.conn)}
    out["verify"] = compare(snap.conn, fresh.conn)
    return out


def _served(ctx: str, account: str) -> Any:
    from metacog.store.neo4j_backend import Neo4jBackend
    return Neo4jBackend(context_path=ctx, account=account)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="OMNI_STORAGE_ROOT")
    ap.add_argument("--stores", required=True,
                    help="comma-separated store keys (ctx or ctx/accounts/acc)")
    ap.add_argument("--archive", help="archive directory (required with --apply)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--tags", action="store_true",
                    help="report the file's tag index against the points' tags")
    a = ap.parse_args(argv)
    if a.apply and not a.archive:
        ap.error("--apply needs --archive (zero deletion : the file is archived first)")
    ok = True
    for store in [s.strip() for s in a.stores.split(",") if s.strip()]:
        try:
            r = migrate_store(store, root=os.path.expanduser(a.root),
                              archive_dir=os.path.expanduser(a.archive) if a.archive else None,
                              apply=a.apply, client_for=_served, tags=a.tags)
        except Exception as e:                  # one store's failure names it, others go on
            r = {"store": store, "error": f"{type(e).__name__}: {e}"}
        ok = ok and "error" not in r and r.get("verify", {"equal": True})["equal"] \
            and r.get("sent") != "FAILED (see the log)"
        print(json.dumps(r, ensure_ascii=False), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
