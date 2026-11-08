"""JC2-5 (TAC2-1204) — the consumers answer alike on SQLite, on ClickHouse, and
on the proxy journal reloaded from ClickHouse.

For each SQLite usage journal given (COPIES : opened read-only, never written),
its nine usage tables and their flags are imported into a THROWAWAY ClickHouse
(``--clickhouse http://user:password@host:port``) under a fresh context, then
every consumer is called on three journals and compared as JSON :

* ``sqlite``  — the copy itself (the reference) ;
* ``click``   — `ClickHouseJournal`, its own ClickHouse SQL (JC2-3) ;
* ``proxy``   — `ProxyJournal` loaded from those rows (the served path, card
  J5 : ``load_then_memory``), its reads answered from memory.

    python scripts/journal_compare.py --clickhouse http://test:test@127.0.0.1:18124 \\
        ~/tac2442/copies/global.journal.db ...

Exit status 1 if any consumer differs.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from typing import Any, Dict, List
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metacog.journal import APPEND_ONLY_TABLES, SQLiteJournal  # noqa: E402
from metacog.journal_clickhouse import (ClickHouseJournal,  # noqa: E402
                                        HttpClickHouseTransport)
from metacog.journal_proxy import JOURNAL_TABLES, ProxyJournal  # noqa: E402

#: SQLite flag column -> (flag name, value of the flag row ; 0/None = no row).
FLAGS = {"retrievals": ("useful", lambda v: None if v is None else int(v) + 1),
         "forget_events": ("merged", lambda v: int(v or 0)),
         "merge_ledger": ("reverted", lambda v: int(v or 0)),
         "wiki_ops": ("reverted", lambda v: int(v or 0))}


def sqlite_rows(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """The usage rows of a SQLite journal in the proxy's shape, ids kept, the
    flags SQLite holds in place turned into `journal_flags` rows (J4)."""
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    out: Dict[str, List[Dict[str, Any]]] = {t: [] for t in JOURNAL_TABLES}
    try:
        for t in APPEND_ONLY_TABLES:
            flag = FLAGS.get(t)
            for r in src.execute(f"SELECT * FROM {t} ORDER BY id"):
                d = dict(r)
                if flag is not None:
                    v = flag[1](d.pop(flag[0], None))
                    if v:
                        out["journal_flags"].append({
                            "id": int(d["id"]), "target_table": t,
                            "target_id": int(d["id"]), "flag": flag[0],
                            "value": v, "ts": d.get("ts") or 0.0})
                out[t].append(d)
    finally:
        src.close()
    return out


class ClickHouseAsProxy:
    """The proxy's INSERT (context posed) and the API's bounded SELECT, direct on a
    throwaway server : what tachikoma does, minus Ray and the API."""

    def __init__(self, transport: HttpClickHouseTransport, database: str, ctx: str):
        self.t, self.db, self.ctx = transport, database, ctx

    def append(self, table, rows):
        self.t.insert(f"{self.db}.{table}", [{**r, "context_path": self.ctx} for r in rows])

    def load(self, ctx, account=""):
        return {t: self.t.select(
            f"SELECT * EXCEPT logged_at FROM {self.db}.{t} "
            "WHERE context_path = {ctx:String} AND account = {account:String} "
            "ORDER BY id", {"ctx": ctx, "account": account}) for t in JOURNAL_TABLES}


def consumers(j, sample) -> Dict[str, Any]:
    out = {"co_retrieved_pairs": j.co_retrieved_pairs(),
           "co_retrieved_pairs_w2": j.co_retrieved_pairs(2),
           "chasles_candidates": j.chasles_candidates(),
           "pending_forgets": j.pending_forgets(),
           "active_redirects": j.active_redirects(),
           "forget_history": j.forget_history(), "merge_history": j.merge_history(),
           "useful_retrievals": j.useful_retrievals(0),
           "hop_target_counts": j.hop_target_counts(),
           "frequent_paths": j.frequent_paths(2, 1),
           "collision_events": j.collision_events(), "tool_events": j.tool_events(),
           "counts": j.counts()}
    for nid in sample:
        out[f"find_co_retrieved[{nid}]"] = j.find_co_retrieved([nid])
        out[f"access_timestamps[{nid}]"] = j.access_timestamps(nid)
        out[f"node_usefulness[{nid}]"] = j.node_usefulness(nid)
        out[f"redirect_of[{nid}]"] = j.redirect_of(nid)
    return out


def _norm(v):
    return json.loads(json.dumps(v, sort_keys=True, default=str))


def compare(path: str, transport: HttpClickHouseTransport, database: str) -> bool:
    name = os.path.basename(path).removesuffix(".journal.db")
    ctx = f"eq.{uuid.uuid4().hex[:8]}.{name}"
    rows = sqlite_rows(path)
    proxy = ClickHouseAsProxy(transport, database, ctx)
    ClickHouseJournal(transport, "schema", database=database)          # the DDL
    for t, rs in rows.items():
        if rs:
            proxy.append(t, rs)
    sq = _readonly(path)
    ch = ClickHouseJournal(transport, ctx, database=database)
    px = ProxyJournal(proxy, ctx, loader=proxy, database=database, start_flusher=False)
    sample = [r[0] for r in sq.conn.execute(
        "SELECT node_id FROM access_events GROUP BY node_id "
        "ORDER BY count(*) DESC, node_id LIMIT 15")]
    sample += [r[0] for r in sq.conn.execute(
        "SELECT absorbed_id FROM merge_ledger ORDER BY id LIMIT 5")]
    ref = _norm(consumers(sq, sample))
    ok = True
    for label, j in (("click", ch), ("proxy", px)):
        got = _norm(consumers(j, sample))
        diff = [k for k in ref if ref[k] != got[k]]
        ok &= not diff
        print(f"{name:26s} {label}: consumers={len(ref)} differ={diff or 'none'}")
        for k in diff[:5]:
            print("     ", k, "sqlite:", str(ref[k])[:300])
            print("     ", k, f"{label} :", str(got[k])[:300])
    print(f"{'':26s} rows={ {t: len(r) for t, r in rows.items() if r} } "
          f"co_retrieved_pairs={len(ref['co_retrieved_pairs'])} "
          f"(w2: {len(ref['co_retrieved_pairs_w2'])}) "
          f"pending_forgets={len(ref['pending_forgets'])} "
          f"merge_history={len(ref['merge_history'])} counts={tuple(ref['counts'])}")
    px.close()
    return ok


def _readonly(path: str) -> SQLiteJournal:
    """A SQLite journal on a COPY's bytes, without writing the copy : the file is
    read into memory (`Journal.__init__` runs DDL, which a read-only open refuses)."""
    j = SQLiteJournal(":memory:")
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        src.backup(j.conn)
    finally:
        src.close()
    return j


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clickhouse", required=True,
                    help="a THROWAWAY server, http://user:password@host:port")
    ap.add_argument("--database", default="omni_journal_eq")
    ap.add_argument("journals", nargs="+")
    a = ap.parse_args(argv)
    u = urlsplit(a.clickhouse)
    tr = HttpClickHouseTransport(f"{u.scheme}://{u.hostname}:{u.port}",
                                 user=u.username or "default", password=u.password or "")
    ok = all([compare(p, tr, a.database) for p in a.journals])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
