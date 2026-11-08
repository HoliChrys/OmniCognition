"""JC2-6 (TAC2-1212) — move the usage tables of the served `.journal.db` files
to ClickHouse, by the infra-proxy, idempotently ; archive each file first.

For each store (``--stores ctx[,ctx/accounts/acc…]`` under ``--root``) :

1. ARCHIVE. ``<root>/<store>/memory.pkl.journal.db`` is copied, through SQLite's
   backup API (a consistent snapshot even while the gate writes), to
   ``<archive>/<store>/memory.pkl.journal.db``. Nothing is ever deleted : the
   original stays where it is and keeps the mutable state (tags, wiki, OKF),
   which moves in its own OO ticket ; an existing archive is never overwritten.
2. READ what ClickHouse already holds for the store (the API's
   ``GET /api/memory/journal``, as the gate does at open).
3. SEND the rows it does not hold, by id, through the proxy
   (``infra-proxy:<ctx>.insert_omni_journal``). The SQLite ids are kept : they
   are small integers, every id the gate mints afterwards is larger (µs).
   A flag SQLite holds in place becomes a ``journal_flags`` row (J4) when
   ClickHouse has none for that row yet — or, for the monotonic flags
   (``merged``, ``reverted``), when ClickHouse still says 0. A ``useful`` score
   re-marked between the archive and the gate's switch keeps its first value :
   the migration never overrides a flag the gate may have written itself.
4. VERIFY by re-reading : every SQLite id is in ClickHouse, table by table.
   A second run sends 0 rows.

THE CLOCKS (decided here, JC2-6). Each row keeps its ``ts`` as omni wrote it —
wall clock for ``retrievals``, ``access_events``, the flag rows and the hops of
`spike.py` ; omni's logical clock (``Memory._now``) for the paths, collisions,
forgets, merges, tool events and wiki ops — because the consumers compare ``ts``
within a table (``frequent_paths(since_ts=)``, ``access_timestamps``) and a
rewritten clock would change their answers. The ONE clock common to every row
is ``logged_at``, the wall clock of the ClickHouse write (the proxy's insert ;
for a migrated row, the time of the migration).

Dry run by default ; ``--apply`` archives and sends. ``--clickhouse URL`` sends
and reads directly on a THROWAWAY server instead of the proxy and the API (the
tests and the bench).

    python scripts/migrate_journal_to_proxy.py --root ~/data_tachikoma/omni/contexts \\
        --archive ~/data_tachikoma/omni/journal-archive/2026-11-08 \\
        --stores tachikoma.paradigm,tachikoma.paralelle.GenAI [--apply]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from journal_compare import ClickHouseAsProxy, sqlite_rows  # noqa: E402

from metacog.journal_clickhouse import _IDS  # noqa: E402
from metacog.journal_proxy import JOURNAL_TABLES  # noqa: E402

JOURNAL_FILE = "memory.pkl.journal.db"
MONOTONIC_FLAGS = frozenset({"merged", "reverted"})
BATCH = 1000


def scope(store: str) -> tuple:
    ctx, _, account = store.partition("/accounts/")
    return ctx, account


def archive(src: str, dst: str) -> bool:
    """Copy `src` to `dst` (consistent snapshot). False if `dst` already exists."""
    if os.path.exists(dst):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    return True


def _current_flags(rows: List[Dict[str, Any]]) -> Dict[tuple, int]:
    cur: Dict[tuple, tuple] = {}
    for r in rows:
        k = (r["target_table"], int(r["target_id"]), r["flag"])
        if k not in cur or int(r["id"]) > cur[k][0]:
            cur[k] = (int(r["id"]), int(r["value"]))
    return {k: v for k, (_, v) in cur.items()}


def missing_rows(source: Dict[str, List[dict]],
                 held: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """The rows of `source` ClickHouse does not hold yet (see the module doc)."""
    out: Dict[str, List[dict]] = {}
    for t in JOURNAL_TABLES:
        if t == "journal_flags":
            continue
        have = {int(r["id"]) for r in held.get(t) or ()}
        out[t] = [r for r in source.get(t) or () if int(r["id"]) not in have]
    have_flags = _current_flags(list(held.get("journal_flags") or ()))
    flags = []
    for f in source.get("journal_flags") or ():
        k = (f["target_table"], int(f["target_id"]), f["flag"])
        if k not in have_flags:
            flags.append(f)                       # id = target id : deterministic
        elif (f["flag"] in MONOTONIC_FLAGS and have_flags[k] == 0
              and int(f["value"]) != 0):
            flags.append({**f, "id": _IDS.next()})
    out["journal_flags"] = flags
    return out


def verify(source: Dict[str, List[dict]], held: Dict[str, List[dict]]) -> Dict[str, Any]:
    """Per table : SQLite rows, and how many of their ids ClickHouse lacks."""
    rep = {}
    for t in JOURNAL_TABLES:
        if t == "journal_flags":
            continue
        have = {int(r["id"]) for r in held.get(t) or ()}
        src = source.get(t) or []
        rep[t] = {"sqlite": len(src), "clickhouse": len(have),
                  "absent": sum(1 for r in src if int(r["id"]) not in have)}
    cur = _current_flags(list(held.get("journal_flags") or ()))
    rep["journal_flags"] = {
        "sqlite": len(source.get("journal_flags") or ()),
        "absent": sum(1 for f in source.get("journal_flags") or ()
                      if (f["target_table"], int(f["target_id"]), f["flag"]) not in cur)}
    return rep


def migrate_store(store: str, *, root: str, archive_dir: Optional[str], writer, loader,
                  apply: bool) -> Dict[str, Any]:
    ctx, account = scope(store)
    src = os.path.join(root, store, JOURNAL_FILE)
    rep: Dict[str, Any] = {"store": store, "ctx": ctx, "account": account}
    if not os.path.isfile(src):
        rep["status"] = "no-journal"
        return rep
    read_from = src
    if apply:
        if not archive_dir:
            raise SystemExit("--apply needs --archive (the journal is archived first)")
        dst = os.path.join(archive_dir, store, JOURNAL_FILE)
        rep["archived"] = dst if archive(src, dst) else f"{dst} (already there)"
        read_from = dst
    source = sqlite_rows(read_from)
    held = loader.load(ctx, account)
    todo = missing_rows(source, held)
    rep["to_send"] = {t: len(r) for t, r in todo.items() if r}
    if not apply:
        rep["status"] = "dry-run"
        return rep
    for t in JOURNAL_TABLES:                       # flags last : their targets first
        # `context_path` is posed by the proxy ; sent too, as `ProxyJournal` does.
        rows = [{**r, "context_path": ctx, "account": account} for r in todo.get(t) or ()]
        for i in range(0, len(rows), BATCH):
            writer.append(t, rows[i:i + BATCH])
    rep["verify"] = v = verify(source, loader.load(ctx, account))
    rep["status"] = ("ok" if all(x["absent"] == 0 for x in v.values())
                     else "incomplete")
    return rep


def _served(ctx: str):
    from metacog.journal_proxy import ApiJournalLoader, ProxyRayWriter
    return ProxyRayWriter(ctx), ApiJournalLoader()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="OMNI_STORAGE_ROOT")
    ap.add_argument("--stores", required=True,
                    help="comma separated store keys : ctx or ctx/accounts/acc")
    ap.add_argument("--archive", help="archive directory (required with --apply)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--clickhouse", help="a THROWAWAY server instead of proxy + API")
    ap.add_argument("--database", default="omni_journal")
    a = ap.parse_args(argv)
    transport = None
    if a.clickhouse:
        from urllib.parse import urlsplit

        from metacog.journal_clickhouse import ClickHouseJournal, HttpClickHouseTransport
        u = urlsplit(a.clickhouse)
        transport = HttpClickHouseTransport(f"{u.scheme}://{u.hostname}:{u.port}",
                                            user=u.username or "default",
                                            password=u.password or "")
        ClickHouseJournal(transport, "schema", database=a.database)      # the DDL
    rc = 0
    for store in [s.strip() for s in a.stores.split(",") if s.strip()]:
        ctx, _ = scope(store)
        if transport is not None:
            writer = loader = ClickHouseAsProxy(transport, a.database, ctx)
        else:
            writer, loader = _served(ctx)
        rep = migrate_store(store, root=a.root, archive_dir=a.archive, writer=writer,
                            loader=loader, apply=a.apply)
        print(json.dumps(rep, ensure_ascii=False, sort_keys=True))
        rc |= rep["status"] not in ("ok", "dry-run", "no-journal")
    return rc


if __name__ == "__main__":
    sys.exit(main())
