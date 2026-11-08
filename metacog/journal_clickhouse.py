"""
The usage journal in ClickHouse (JC·V2·M2 — JC2-3, TAC2-1202).

`ClickHouseJournal` implements `metacog.journal.JournalProtocol` : it returns
what the SQLite `Journal` returns for the same calls. Three differences in HOW,
none in WHAT :

* **Append-only, flags are events (J4).** The nine usage tables
  (`journal.APPEND_ONLY_TABLES`) are only ever INSERTed. The four flags SQLite
  rewrites in place (`retrievals.useful`, `forget_events.merged`,
  `merge_ledger.reverted`, `wiki_ops.reverted`) become rows of `journal_flags` ;
  the current value of a flag is its latest row (`argMax(value, id)`).
* **Ids are minted here**, not by an AUTOINCREMENT : a strictly increasing
  integer (microseconds since the epoch, bumped on collision). `ORDER BY id`
  therefore keeps the insertion order SQLite gives, and a migrated SQLite id
  (1, 2, 3 ...) sorts before every new one.
* **Writes are grouped per call.** One journal call is one INSERT per table it
  touches (a retrieval and its accesses travel together), sent before the call
  returns : a write is as durable as SQLite's commit, which the restart tests
  rely on. `batch_size > 1` groups several calls (rows wait until `batch_size`
  rows, a read — a read always sees the caller's own writes — or `close()`) ;
  a process that dies then loses what was still buffered.

The mutable STATE (tags, wiki docs/refs/seeds/annotations/pending, OKF fields,
vocab, ref remaps) is not a usage log : it is delegated to `state`, a SQLite
`Journal` until D10 moves it to Neo4j. `ClickHouseJournal(transport, ctx)` alone
keeps that state in memory.

FAILURE-SAFE (the omni invariant : behaviour must not depend on the journal
being up). A transport error never raises out of a journal call : a failed
flush keeps its rows for the next one (up to `max_buffer`, beyond which the
oldest rows are dropped and counted in `dropped`), a failed read returns what an
empty journal returns. Both are logged.

The transport is injected, and it is the ONLY thing that talks to ClickHouse.
`HttpClickHouseTransport` talks to a ClickHouse server directly over HTTP : it
exists for the tests and the migration bench on a throwaway server. The served
gate must reach ClickHouse through the platform (rule C5, « jamais en direct ») ;
that transport is decided on Paperclip TAC-2455 (card J5) and plugs in behind the same
`ClickHouseTransport` interface.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Protocol,
                    Sequence, Tuple)

from metacog.journal import Journal

logger = logging.getLogger(__name__)

DEFAULT_DATABASE = "omni_journal"

# ClickHouse DDL of the usage journal. `{db}` is the database. Every table
# carries `context_path` (the memory's context) first in its sort key, then
# `account` ('' = the context's own store, else a narrow account's store,
# `<ctx>/accounts/<acc>` in the gate's layout) : every read is bounded to one
# store. No TTL : the memory forgets nothing (J3).
# `logged_at` is the wall clock of the write ; `ts` is the caller's clock, kept
# as given (logical or wall, see JC2-6).
_DDL = (
    "CREATE DATABASE IF NOT EXISTS {db}",
    """CREATE TABLE IF NOT EXISTS {db}.retrievals (
        context_path String, account String DEFAULT '', id UInt64,
        query_text Nullable(String), returned_node_ids String,
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, id)""",
    """CREATE TABLE IF NOT EXISTS {db}.access_events (
        context_path String, account String DEFAULT '', id UInt64,
        node_id String, retrieval_id UInt64, rank UInt32,
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, retrieval_id, rank, id)""",
    """CREATE TABLE IF NOT EXISTS {db}.hops (
        context_path String, account String DEFAULT '', id UInt64, src_id String, dst_id String,
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, id)""",
    """CREATE TABLE IF NOT EXISTS {db}.path_traversals (
        context_path String, account String DEFAULT '', id UInt64, signature String,
        start_id String, end_id String, intermediates String, length UInt32,
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, id)""",
    """CREATE TABLE IF NOT EXISTS {db}.collision_events (
        context_path String, account String DEFAULT '', id UInt64, kind String,
        child_id Nullable(String), parent_ids String, anchor_ids String,
        trigger_distance Nullable(Float64), threshold Nullable(Float64),
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, id)""",
    """CREATE TABLE IF NOT EXISTS {db}.forget_events (
        context_path String, account String DEFAULT '', id UInt64, node_id String, reason String,
        superseded_by Nullable(String),
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, id)""",
    """CREATE TABLE IF NOT EXISTS {db}.merge_ledger (
        context_path String, account String DEFAULT '', id UInt64, absorbed_id String,
        keeper_id Nullable(String), kind String, reason String, snapshot String,
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, id)""",
    """CREATE TABLE IF NOT EXISTS {db}.tool_events (
        context_path String, account String DEFAULT '', id UInt64, tool_id String, event String,
        reason Nullable(String),
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, id)""",
    """CREATE TABLE IF NOT EXISTS {db}.wiki_ops (
        context_path String, account String DEFAULT '', id UInt64, doc_id String, target String, op String,
        before Nullable(String), after Nullable(String),
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, id)""",
    # J4 : a flag SQLite rewrites in place is a row here. `target_table`.`target_id` is the
    # flagged row ; the flag's value is the one of its highest `id`.
    """CREATE TABLE IF NOT EXISTS {db}.journal_flags (
        context_path String, account String DEFAULT '', id UInt64,
        target_table LowCardinality(String), target_id UInt64,
        flag LowCardinality(String), value Int64,
        ts Float64, logged_at DateTime64(6) DEFAULT now64(6)
    ) ENGINE = MergeTree ORDER BY (context_path, account, target_table, flag, target_id, id)""",
)

#: The journal methods that read or write mutable STATE, served by `state`.
STATE_METHODS = (
    "log_tags", "upsert_wiki_doc", "delete_wiki_doc", "get_wiki_doc",
    "set_wiki_refs", "set_ref_fingerprint", "mark_wiki_ref_outdated",
    "wiki_refs_for_doc", "docs_referencing", "remap_wiki_ref",
    "mark_wiki_ref_stale", "log_ref_remap", "ref_remaps_from",
    "mark_ref_remap_reverted", "set_okf_derived", "okf_derived_of",
    "okf_derived_docs_where", "clear_okf_derived", "upsert_seed",
    "set_seed_cache", "seeds_for_doc", "remove_seed", "add_annotation",
    "annotations_for_doc", "remove_annotation", "add_pending", "pending_for_doc",
    "resolve_pending", "propose_vocab", "vet_vocab", "vocab_status", "vocab",
    "all_wiki_doc_ids", "index_okf_fields", "okf_docs_where", "okf_fields_of",
    "okf_schema", "okf_types", "nodes_with_tag", "tag_glossary",
)


class ClickHouseTransport(Protocol):
    """What `ClickHouseJournal` needs from ClickHouse, and nothing more.

    `select` takes ClickHouse query parameters (`{name:Type}` in the SQL) : a
    value never enters the SQL text. It returns one dict per row, 64-bit
    integers as ints and NULL as None.
    """

    def command(self, sql: str) -> None: ...
    def insert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None: ...
    def select(self, sql: str,
               params: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]: ...


def _escape_param(value: str) -> str:
    # HTTP query parameters are parsed in ClickHouse's Escaped text format.
    return (value.replace("\\", "\\\\").replace("\t", "\\t")
            .replace("\n", "\\n").replace("\r", "\\r").replace("\0", "\\0"))


def _format_param(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        # An array parameter is read as an array LITERAL : its strings are
        # quoted, and escaped inside the quotes (no outer Escaped pass).
        def item(v: Any) -> str:
            if isinstance(v, str):
                return "'" + _escape_param(v).replace("'", "\\'") + "'"
            return repr(v)
        return "[" + ",".join(item(v) for v in value) + "]"
    if isinstance(value, bool):
        return "1" if value else "0"
    return _escape_param(str(value))


class HttpClickHouseTransport:
    """ClickHouse over its HTTP interface (`httpx`). DIRECT : for the tests and
    the migration bench on a throwaway server, never for the served gate (see
    the module docstring)."""

    def __init__(self, url: str, *, user: str = "default", password: str = "",
                 timeout_s: float = 10.0) -> None:
        import httpx

        self._client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout_s,
                                    auth=(user, password))

    def _post(self, sql: str, body: Optional[bytes] = None,
              params: Optional[Mapping[str, Any]] = None) -> str:
        qp: Dict[str, str] = {"async_insert": "0",
                              "output_format_json_quote_64bit_integers": "0"}
        if body is None:
            body, sql_in_url = sql.encode(), False
        else:
            qp["query"], sql_in_url = sql, True
        for k, v in (params or {}).items():
            qp[f"param_{k}"] = _format_param(v)
        r = self._client.post("/", params=qp, content=body)
        if r.status_code != 200:
            what = sql if sql_in_url else sql[:200]
            raise RuntimeError(f"ClickHouse HTTP {r.status_code} on {what!r}: "
                               f"{r.text[:500]}")
        return r.text

    def command(self, sql: str) -> None:
        self._post(sql)

    def insert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        body = "\n".join(json.dumps(dict(r), ensure_ascii=False) for r in rows)
        self._post(f"INSERT INTO {table} FORMAT JSONEachRow", body.encode())

    def select(self, sql: str,
               params: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        text = self._post(f"{sql} FORMAT JSONEachRow", params=params)
        return [json.loads(line) for line in text.splitlines() if line]

    def close(self) -> None:
        self._client.close()


class _IdClock:
    """Strictly increasing integer ids : microseconds since the epoch, bumped
    by one when two ids are asked for within the same microsecond."""

    def __init__(self) -> None:
        self._last = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._last = max(self._last + 1, time.time_ns() // 1000)
            return self._last


_IDS = _IdClock()


class ClickHouseJournal:
    """`JournalProtocol` on ClickHouse, bounded to one context (`ctx`)."""

    def __init__(self, transport: ClickHouseTransport, ctx: str, *,
                 account: str = "",
                 database: str = DEFAULT_DATABASE,
                 state: Optional[Journal] = None,
                 batch_size: int = 1, max_buffer: int = 50_000,
                 ensure_schema: bool = True) -> None:
        if not ctx:
            raise ValueError("a ClickHouse journal is bounded to a context ; "
                             "ctx is required")
        self.transport = transport
        self.ctx = ctx
        self.account = account
        self.db = database
        self.path = f"clickhouse:{database}/{ctx}" + (f"/accounts/{account}" if account else "")
        self.state = state if state is not None else Journal(":memory:")
        self.batch_size = max(1, int(batch_size))
        self.max_buffer = max(self.batch_size, int(max_buffer))
        self.dropped = 0
        self._buffer: List[Tuple[str, Dict[str, Any]]] = []
        self._lock = threading.RLock()
        for name in STATE_METHODS:
            setattr(self, name, getattr(self.state, name))
        if ensure_schema:
            self.ensure_schema()

    @property
    def conn(self):
        """The mutable-state SQLite connection (the gate lists wiki docs on it)."""
        return self.state.conn

    def ensure_schema(self) -> bool:
        """Create the database and tables if missing. False (logged) if
        ClickHouse is unreachable : the journal still answers, empty."""
        try:
            for ddl in _DDL:
                self.transport.command(ddl.format(db=self.db))
            return True
        except Exception:
            logger.warning("omni journal: ClickHouse schema not ensured for %s",
                           self.ctx, exc_info=True)
            return False

    # -- plumbing -------------------------------------------------------------

    def _append(self, table: str, row: Dict[str, Any], *, last: bool = True) -> None:
        """Buffer one row. `last=False` while a call still has rows to add : a
        flush never splits one call's rows (a retrieval and its accesses)."""
        row["context_path"] = self.ctx
        row["account"] = self.account
        with self._lock:
            self._buffer.append((table, row))
            if last and len(self._buffer) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> bool:
        """Send the buffered rows. True when the buffer is empty afterwards."""
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> bool:
        if not self._buffer:
            return True
        by_table: Dict[str, List[Dict[str, Any]]] = {}
        for table, row in self._buffer:
            by_table.setdefault(table, []).append(row)
        sent: set = set()
        try:
            for table, rows in by_table.items():
                self.transport.insert(f"{self.db}.{table}", rows)
                sent.add(table)
        except Exception:
            self._buffer = [(t, r) for t, r in self._buffer if t not in sent]
            over = len(self._buffer) - self.max_buffer
            if over > 0:
                self._buffer = self._buffer[over:]
                self.dropped += over
            logger.warning("omni journal: flush failed for %s (%d rows kept, "
                           "%d dropped so far)", self.ctx, len(self._buffer),
                           self.dropped, exc_info=True)
            return False
        self._buffer = []
        return True

    def _select(self, sql: str, **params: Any) -> Optional[List[Dict[str, Any]]]:
        """Run a read bounded to this context ({ctx:String} is always bound).
        None when ClickHouse fails : the caller answers like an empty journal."""
        with self._lock:
            self._flush_locked()
        try:
            return self.transport.select(sql.replace("{db}", self.db),
                                         {"ctx": self.ctx, "account": self.account,
                                          **params})
        except Exception:
            logger.warning("omni journal: read failed for %s", self.ctx,
                           exc_info=True)
            return None

    def _rows(self, sql: str, **params: Any) -> List[Dict[str, Any]]:
        return self._select(sql, **params) or []

    def _flag(self, table: str, target_id: int, flag: str, value: int) -> None:
        self._append("journal_flags", {
            "id": _IDS.next(), "target_table": table, "target_id": int(target_id),
            "flag": flag, "value": int(value), "ts": time.time()})

    # The set of row ids of `table` whose `flag` currently holds a non-zero
    # value (the latest flag row wins).
    _FLAGGED = ("(SELECT target_id FROM {{db}}.journal_flags "
                "WHERE context_path = {{ctx:String}} AND account = {{account:String}} AND target_table = '{table}' "
                "AND flag = '{flag}' GROUP BY target_id "
                "HAVING argMax(value, id) != 0)")

    @classmethod
    def _flagged(cls, table: str, flag: str) -> str:
        return cls._FLAGGED.format(table=table, flag=flag)

    # -- usage writes ---------------------------------------------------------

    def log_retrieval(self, query_text: str, returned_node_ids: Sequence[str],
                      ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        ids = [str(i) for i in returned_node_ids]
        rid = _IDS.next()
        self._append("retrievals", {"id": rid, "query_text": query_text,
                                    "returned_node_ids": json.dumps(ids),
                                    "ts": ts}, last=not ids)
        for rank, nid in enumerate(ids):
            self._append("access_events", {"id": _IDS.next(), "node_id": nid,
                                           "retrieval_id": rid, "rank": rank,
                                           "ts": ts}, last=rank == len(ids) - 1)
        return rid

    def mark_useful(self, retrieval_id: int, score: int) -> None:
        if score not in (0, 1, 2):
            raise ValueError(f"score must be 0, 1 or 2 ; got {score!r}")
        # `useful` is NULL until marked, and 0 is a real score : the flag row
        # stores score + 1 so that "never marked" and "marked 0" stay distinct.
        self._flag("retrievals", retrieval_id, "useful", score + 1)

    def log_hop(self, src_id: str, dst_id: str,
                ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        self._append("hops", {"id": _IDS.next(), "src_id": str(src_id),
                              "dst_id": str(dst_id), "ts": ts})

    def log_path(self, node_ids: Sequence[str],
                 ts: Optional[float] = None) -> Optional[int]:
        ids = [str(i) for i in node_ids]
        if len(ids) < 2:
            return None
        ts = time.time() if ts is None else ts
        pid = _IDS.next()
        self._append("path_traversals", {
            "id": pid, "signature": ">".join(ids), "start_id": ids[0],
            "end_id": ids[-1], "intermediates": json.dumps(ids[1:-1]),
            "length": len(ids), "ts": ts})
        return pid

    def log_collision_event(self, kind: str, *, child_id: Optional[str],
                            parent_ids: Sequence[str],
                            anchor_ids: Sequence[str] = (),
                            trigger_distance: Optional[float] = None,
                            threshold: Optional[float] = None,
                            ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        eid = _IDS.next()
        self._append("collision_events", {
            "id": eid, "kind": kind, "child_id": child_id,
            "parent_ids": json.dumps([str(i) for i in parent_ids]),
            "anchor_ids": json.dumps([str(i) for i in anchor_ids]),
            "trigger_distance": trigger_distance, "threshold": threshold,
            "ts": ts})
        return eid

    def log_forget(self, node_id: str, reason: str,
                   superseded_by: Optional[str] = None,
                   ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        eid = _IDS.next()
        self._append("forget_events", {
            "id": eid, "node_id": str(node_id), "reason": str(reason),
            "superseded_by": None if superseded_by is None else str(superseded_by),
            "ts": ts})
        return eid

    def mark_forget_merged(self, event_id: int) -> None:
        self._flag("forget_events", event_id, "merged", 1)

    def log_merge(self, absorbed_id: str, keeper_id: Optional[str], kind: str,
                  reason: str, snapshot: Optional[dict] = None,
                  ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        lid = _IDS.next()
        self._append("merge_ledger", {
            "id": lid, "absorbed_id": str(absorbed_id),
            "keeper_id": None if keeper_id is None else str(keeper_id),
            "kind": str(kind), "reason": str(reason or "unspecified"),
            "snapshot": json.dumps(snapshot or {}), "ts": ts})
        return lid

    def mark_merge_reverted(self, ledger_id: int) -> None:
        self._flag("merge_ledger", ledger_id, "reverted", 1)

    def log_tool_event(self, tool_id: str, event: str,
                       reason: Optional[str] = None,
                       ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        eid = _IDS.next()
        self._append("tool_events", {"id": eid, "tool_id": str(tool_id),
                                     "event": str(event), "reason": reason,
                                     "ts": ts})
        return eid

    def log_wiki_op(self, doc_id: str, target: str, op: str,
                    before: Optional[dict], after: Optional[dict],
                    ts: Optional[float] = None) -> int:
        ts = time.time() if ts is None else ts
        oid = _IDS.next()
        self._append("wiki_ops", {
            "id": oid, "doc_id": str(doc_id), "target": str(target), "op": op,
            "before": None if before is None else json.dumps(before),
            "after": None if after is None else json.dumps(after), "ts": ts})
        return oid

    def mark_wiki_op_reverted(self, op_id: int) -> None:
        self._flag("wiki_ops", op_id, "reverted", 1)

    # -- usage reads ----------------------------------------------------------

    def pending_forgets(self) -> List[dict]:
        rows = self._rows(
            "SELECT id, node_id, reason, superseded_by, ts FROM {db}.forget_events "
            "WHERE context_path = {ctx:String} AND account = {account:String} AND id NOT IN "
            + self._flagged("forget_events", "merged") + " ORDER BY id ASC")
        return [{"id": int(r["id"]), "node_id": r["node_id"], "reason": r["reason"],
                 "superseded_by": r["superseded_by"], "ts": r["ts"]} for r in rows]

    def forget_history(self) -> List[dict]:
        rows = self._rows(
            "SELECT id, node_id, reason, superseded_by, ts, id IN "
            + self._flagged("forget_events", "merged") + " AS merged "
            "FROM {db}.forget_events WHERE context_path = {ctx:String} AND account = {account:String} "
            "ORDER BY id ASC")
        return [{"id": int(r["id"]), "node_id": r["node_id"], "reason": r["reason"],
                 "superseded_by": r["superseded_by"], "ts": r["ts"],
                 "merged": bool(r["merged"])} for r in rows]

    _LIVE_LEDGER = ("context_path = {ctx:String} AND account = {account:String} AND id NOT IN "
                    + _FLAGGED.format(table="merge_ledger", flag="reverted"))

    def redirect_of(self, node_id: str) -> Optional[str]:
        rows = self._rows(
            "SELECT keeper_id FROM {db}.merge_ledger WHERE absorbed_id = {nid:String} "
            "AND keeper_id IS NOT NULL AND " + self._LIVE_LEDGER
            + " ORDER BY id DESC LIMIT 1", nid=str(node_id))
        return rows[0]["keeper_id"] if rows else None

    def active_redirects(self) -> dict:
        rows = self._rows(
            "SELECT absorbed_id, keeper_id FROM {db}.merge_ledger "
            "WHERE keeper_id IS NOT NULL AND " + self._LIVE_LEDGER
            + " ORDER BY id ASC")
        return {r["absorbed_id"]: r["keeper_id"] for r in rows}

    def absorbed_by(self, keeper_id: str) -> List[str]:
        rows = self._rows(
            "SELECT DISTINCT absorbed_id FROM {db}.merge_ledger "
            "WHERE keeper_id = {kid:String} AND " + self._LIVE_LEDGER
            + " ORDER BY absorbed_id", kid=str(keeper_id))
        return [r["absorbed_id"] for r in rows]

    def merge_history(self, node_id: Optional[str] = None,
                      include_reverted: bool = True) -> List[dict]:
        conds = ["context_path = {ctx:String} AND account = {account:String}"]
        params: Dict[str, Any] = {}
        if node_id is not None:
            conds.append("absorbed_id = {nid:String}")
            params["nid"] = str(node_id)
        reverted = self._flagged("merge_ledger", "reverted")
        if not include_reverted:
            conds.append("id NOT IN " + reverted)
        rows = self._rows(
            "SELECT id, absorbed_id, keeper_id, kind, reason, snapshot, ts, id IN "
            + reverted + " AS reverted FROM {db}.merge_ledger WHERE "
            + " AND ".join(conds) + " ORDER BY id DESC", **params)
        return [{"id": int(r["id"]), "absorbed_id": r["absorbed_id"],
                 "keeper_id": r["keeper_id"], "kind": r["kind"],
                 "reason": r["reason"], "snapshot": json.loads(r["snapshot"]),
                 "ts": r["ts"], "reverted": bool(r["reverted"])} for r in rows]

    def tool_events(self, tool_id: Optional[str] = None) -> List[dict]:
        if tool_id is None:
            rows = self._rows(
                "SELECT tool_id, event, reason, ts FROM {db}.tool_events "
                "WHERE context_path = {ctx:String} AND account = {account:String} ORDER BY id ASC")
        else:
            rows = self._rows(
                "SELECT tool_id, event, reason, ts FROM {db}.tool_events "
                "WHERE context_path = {ctx:String} AND account = {account:String} AND tool_id = {tid:String} "
                "ORDER BY id ASC", tid=str(tool_id))
        return [{"tool_id": r["tool_id"], "event": r["event"],
                 "reason": r["reason"], "ts": r["ts"]} for r in rows]

    _WIKI_OP_COLS = ("SELECT id, doc_id, target, op, before, after, ts, id IN "
                     + _FLAGGED.format(table="wiki_ops", flag="reverted")
                     + " AS reverted FROM {db}.wiki_ops WHERE context_path = {ctx:String} AND account = {account:String} ")

    @staticmethod
    def _wiki_op(r: Mapping[str, Any]) -> dict:
        return {"id": int(r["id"]), "doc_id": r["doc_id"], "target": r["target"],
                "op": r["op"],
                "before": None if r["before"] is None else json.loads(r["before"]),
                "after": None if r["after"] is None else json.loads(r["after"]),
                "ts": r["ts"], "reverted": bool(r["reverted"])}

    def wiki_ops_for_doc(self, doc_id: str) -> List[dict]:
        rows = self._rows(self._WIKI_OP_COLS + "AND doc_id = {did:String} "
                          "ORDER BY id DESC", did=str(doc_id))
        return [self._wiki_op(r) for r in rows]

    def get_wiki_op(self, op_id: int) -> Optional[dict]:
        rows = self._rows(self._WIKI_OP_COLS + "AND id = {oid:UInt64}",
                          oid=int(op_id))
        return self._wiki_op(rows[0]) if rows else None

    def hop_target_counts(self) -> List[Tuple[str, int]]:
        rows = self._rows(
            "SELECT dst_id, count() AS c FROM {db}.hops "
            "WHERE context_path = {ctx:String} AND account = {account:String} GROUP BY dst_id "
            "ORDER BY c DESC, dst_id ASC")
        return [(r["dst_id"], int(r["c"])) for r in rows]

    def effective_spike(self, node_id: str) -> int:
        # `instr(parent_ids, '"id"')` in SQLite = `position(...)` here : the
        # same substring test on the stored JSON text, not an array lookup.
        nid = str(node_id)
        rows = self._select(
            "SELECT count() AS c FROM {db}.hops WHERE context_path = {ctx:String} AND account = {account:String} "
            "AND dst_id = {nid:String} AND ts > ifNull((SELECT maxOrNull(ts) "
            "  FROM {db}.collision_events WHERE context_path = {ctx:String} AND account = {account:String} "
            "  AND kind = 'chasles' AND (child_id = {nid:String} "
            "  OR position(parent_ids, {q:String}) > 0 "
            "  OR position(anchor_ids, {q:String}) > 0)), 0)",
            nid=nid, q=f'"{nid}"')
        return int(rows[0]["c"]) if rows else 0

    # The aggregates get their own names : an alias equal to a column name
    # would shadow that column in WHERE (ILLEGAL_AGGREGATION).
    _PATH_COLS = ("SELECT signature, any(start_id) AS agg_start, any(end_id) AS agg_end, "
                  "any(intermediates) AS agg_intermediates, any(length) AS agg_length, "
                  "count() AS freq, max(ts) AS last_ts ")

    @staticmethod
    def _path(r: Mapping[str, Any]) -> dict:
        return Journal._path_row({
            "signature": r["signature"], "start_id": r["agg_start"],
            "end_id": r["agg_end"], "intermediates": r["agg_intermediates"],
            "length": r["agg_length"], "freq": r["freq"], "last_ts": r["last_ts"]})

    def frequent_paths(self, min_len: int = 4, min_freq: int = 2,
                       since_ts: float = 0.0) -> List[dict]:
        rows = self._rows(
            self._PATH_COLS + "FROM {db}.path_traversals "
            "WHERE context_path = {ctx:String} AND account = {account:String} AND length >= {ml:Int64} "
            "AND ts > {since:Float64} GROUP BY signature HAVING freq >= {mf:Int64} "
            "ORDER BY freq DESC, signature ASC",
            ml=int(min_len), since=float(since_ts), mf=int(min_freq))
        return [self._path(r) for r in rows]

    def chasles_candidates(self, min_len: int = 4,
                           min_freq: int = 2) -> List[dict]:
        # The refractory is per path : the last Chasles fire whose anchors
        # contain BOTH ends of the path (substring test on the JSON text, as
        # SQLite's `instr`). ClickHouse has no correlated subquery : the fires
        # are folded into two aligned arrays and scanned per traversal.
        rows = self._rows(
            "WITH (SELECT (groupArray(anchor_ids), groupArray(ts)) "
            "      FROM {db}.collision_events "
            "      WHERE context_path = {ctx:String} AND account = {account:String} AND kind = 'chasles') AS fires "
            + self._PATH_COLS + "FROM {db}.path_traversals "
            "WHERE context_path = {ctx:String} AND account = {account:String} AND length >= {ml:Int64} "
            "AND ts > ifNull(arrayMax(arrayFilter((t, a) -> "
            "  position(a, concat('\"', start_id, '\"')) > 0 "
            "  AND position(a, concat('\"', end_id, '\"')) > 0, "
            "  arrayMap(x -> toNullable(x), fires.2), fires.1)), 0) "
            "GROUP BY signature HAVING freq >= {mf:Int64} "
            "ORDER BY freq DESC, signature ASC",
            ml=int(min_len), mf=int(min_freq))
        return [self._path(r) for r in rows]

    def co_retrieved_pairs(self, window: Optional[int] = None
                           ) -> List[Tuple[Tuple[str, str], int]]:
        sql = ("SELECT a.node_id AS x, b.node_id AS y, count() AS cooc "
               "FROM {db}.access_events AS a INNER JOIN {db}.access_events AS b "
               "ON a.retrieval_id = b.retrieval_id "
               "WHERE a.context_path = {ctx:String} AND account = {account:String} AND b.context_path = {ctx:String} AND account = {account:String} "
               "AND a.node_id < b.node_id ")
        params: Dict[str, Any] = {}
        if window is not None:
            sql += "AND abs(toInt64(a.rank) - toInt64(b.rank)) <= {w:Int64} "
            params["w"] = int(window)
        sql += "GROUP BY x, y ORDER BY x, y"
        rows = self._rows(sql, **params)
        return [((r["x"], r["y"]), int(r["cooc"])) for r in rows]

    def modal_next(self, src_id: str) -> Optional[str]:
        rows = self._rows(
            "SELECT dst_id FROM {db}.hops WHERE context_path = {ctx:String} AND account = {account:String} "
            "AND src_id = {sid:String} GROUP BY dst_id "
            "ORDER BY count() DESC, dst_id ASC LIMIT 1", sid=str(src_id))
        return rows[0]["dst_id"] if rows else None

    def collision_events(self, kind: Optional[str] = None) -> List[dict]:
        sql = ("SELECT id, kind, child_id, parent_ids, anchor_ids, trigger_distance, "
               "threshold, ts FROM {db}.collision_events "
               "WHERE context_path = {ctx:String} AND account = {account:String} ")
        params: Dict[str, Any] = {}
        if kind is not None:
            sql += "AND kind = {kind:String} "
            params["kind"] = kind
        rows = self._rows(sql + "ORDER BY id ASC", **params)
        return [{
            "id": int(r["id"]), "kind": r["kind"], "child_id": r["child_id"],
            "parent_ids": json.loads(r["parent_ids"] or "[]"),
            "anchor_ids": json.loads(r["anchor_ids"] or "[]"),
            "trigger_distance": r["trigger_distance"],
            "threshold": r["threshold"], "ts": float(r["ts"]),
        } for r in rows]

    def find_co_retrieved(self, seed_node_ids: Sequence[str],
                          k: int = 7) -> List[Tuple[str, int]]:
        seeds = [str(i) for i in seed_node_ids]
        if not seeds:
            return []
        rows = self._rows(
            "SELECT b.node_id AS node_id, count() AS cooc "
            "FROM {db}.access_events AS a INNER JOIN {db}.access_events AS b "
            "ON a.retrieval_id = b.retrieval_id "
            "WHERE a.context_path = {ctx:String} AND account = {account:String} AND b.context_path = {ctx:String} AND account = {account:String} "
            "AND a.node_id IN {seeds:Array(String)} "
            "AND b.node_id NOT IN {seeds:Array(String)} "
            "GROUP BY b.node_id ORDER BY cooc DESC, node_id ASC LIMIT {k:Int64}",
            seeds=seeds, k=int(k))
        return [(r["node_id"], int(r["cooc"])) for r in rows]

    # `useful` per retrieval : the latest flag row, stored as score + 1.
    _USEFUL = ("(SELECT target_id, argMax(value, id) - 1 AS useful "
               "FROM {db}.journal_flags WHERE context_path = {ctx:String} AND account = {account:String} "
               "AND target_table = 'retrievals' AND flag = 'useful' "
               "GROUP BY target_id)")

    def node_usefulness(self, node_id: str) -> dict:
        rows = self._select(
            "SELECT countDistinctIf(a.retrieval_id, u.useful >= 2) AS pos, "
            "countDistinctIf(a.retrieval_id, u.useful = 0) AS neg "
            "FROM {db}.access_events AS a INNER JOIN " + self._USEFUL + " AS u "
            "ON a.retrieval_id = u.target_id "
            "WHERE a.context_path = {ctx:String} AND account = {account:String} AND a.node_id = {nid:String}",
            nid=str(node_id))
        if not rows:
            return {"useful": 0, "useless": 0}
        return {"useful": int(rows[0]["pos"]), "useless": int(rows[0]["neg"])}

    def retrieval_returned_ids(self, retrieval_id: int) -> List[str]:
        rows = self._rows(
            "SELECT node_id FROM {db}.access_events WHERE context_path = {ctx:String} AND account = {account:String} "
            "AND retrieval_id = {rid:UInt64} ORDER BY rank ASC", rid=int(retrieval_id))
        return [r["node_id"] for r in rows]

    def access_timestamps(self, node_id: str) -> List[float]:
        rows = self._rows(
            "SELECT ts FROM {db}.access_events WHERE context_path = {ctx:String} AND account = {account:String} "
            "AND node_id = {nid:String} ORDER BY ts ASC, id ASC", nid=str(node_id))
        return [float(r["ts"]) for r in rows]

    def useful_retrievals(self, min_score: int = 2,
                          max_score: Optional[int] = None
                          ) -> List[Tuple[int, str, List[str]]]:
        params: Dict[str, Any] = {"lo": int(min_score)}
        clause = "u.useful >= {lo:Int64}"
        if max_score is not None:
            clause += " AND u.useful <= {hi:Int64}"
            params["hi"] = int(max_score)
        rows = self._rows(
            "SELECT r.id AS id, r.query_text AS query_text, "
            "r.returned_node_ids AS returned_node_ids "
            "FROM {db}.retrievals AS r INNER JOIN " + self._USEFUL + " AS u "
            "ON r.id = u.target_id WHERE r.context_path = {ctx:String} AND account = {account:String} AND "
            + clause + " ORDER BY id ASC", **params)
        return [(int(r["id"]), r["query_text"] or "",
                 json.loads(r["returned_node_ids"] or "[]")) for r in rows]

    def counts(self) -> Tuple[int, int]:
        rows = self._select(
            "SELECT (SELECT count() FROM {db}.retrievals "
            "        WHERE context_path = {ctx:String} AND account = {account:String}) AS r, "
            "       (SELECT count() FROM {db}.access_events "
            "        WHERE context_path = {ctx:String} AND account = {account:String}) AS a")
        if not rows:
            return 0, 0
        return int(rows[0]["r"]), int(rows[0]["a"])

    def close(self) -> None:
        self.flush()
        self.state.close()


def iter_journal_methods(journal: Any) -> Iterable[str]:
    """The `JournalProtocol` methods `journal` lacks (empty = complete)."""
    from metacog.journal import JournalProtocol

    for name in dir(JournalProtocol):
        if not name.startswith("_") and name != "path" and not callable(
                getattr(journal, name, None)):
            yield name
