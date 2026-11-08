"""
The usage journal written BY THE INFRA-PROXY, read once and kept in memory
(JC·V2·M2 — JC2-4, TAC2-1203).

Decided by the product owner on card J5 (Paperclip TAC-2455, answered 08/11) :

* ``write_path = proxy_ray`` — omni never writes ClickHouse itself. Its rows go
  to the context's ``infra-proxy:<ctx>`` Ray Serve app (``insert_omni_journal``),
  which stamps ``context_path`` and INSERTs into the separate ``omni_journal``
  database under its own account (tachikoma ``scripts/clickhouse-init.sql`` :
  ``GRANT INSERT ON omni_journal.* TO agent``, nothing else) ;
* ``read_path = load_then_memory`` — one read when the context opens, through
  the API (``GET /api/memory/journal``, the only governed read path), then every
  read is answered from memory ;
* ``mutable_state = separate_ticket`` — tags, wiki docs/refs/seeds, OKF fields…
  stay in the SQLite ``state`` journal here ; moving them is an OO ticket.

`ProxyJournal` is a `ClickHouseJournal` for the SHAPE of what it writes (ids
minted client side, flags as ``journal_flags`` rows, J4 — no ``UPDATE`` ever
leaves this process) and a SQLite ``:memory:`` journal for what it reads. Every
write lands in memory at once, so a read always sees it ; the rows reach the
proxy from a background thread, grouped (``flush_interval_s``). Reads never wait
on Ray. ``flush_interval_s <= 0`` sends each call's rows before it returns
(no thread) : the tests, and a caller that prefers durability to latency.

FAILURE-SAFE, as every journal of omni : a proxy that is down or absent never
raises out of a journal call. Unsent rows wait in the buffer (bounded by
``max_buffer``, the oldest dropped and counted in ``dropped``) ; a failed load
opens an empty memory and says so (``load_error``). Memory still answers with
what this process wrote. A process that dies loses what was still buffered
(at most ``flush_interval_s`` of rows when the proxy is up).

Correspondence — every table of the SQLite journal, and where it goes :

==================  ==========================================  ==============
SQLite table        ClickHouse (``omni_journal``, by the proxy)  flag rows (J4)
==================  ==========================================  ==============
retrievals          ``retrievals``                              ``useful``
access_events       ``access_events``                           —
hops                ``hops``                                    —
path_traversals     ``path_traversals``                         —
collision_events    ``collision_events``                        —
forget_events       ``forget_events``                           ``merged``
merge_ledger        ``merge_ledger``                            ``reverted``
tool_events         ``tool_events``                             —
wiki_ops            ``wiki_ops``                                ``reverted``
(flags)             ``journal_flags``                           —
tags, wiki_docs, wiki_refs, okf_fields, wiki_ref_remaps, okf_derived,
wiki_seeds, wiki_annotations, wiki_pending, okf_vocab
                    mutable state, NOT a usage log : ``state`` (SQLite) until
                    the OO ticket moves it to Neo4j (D10, ``STATE_METHODS``)
==================  ==========================================  ==============
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.parse
import urllib.request
from typing import (Any, Callable, Dict, List, Mapping, Optional, Protocol,
                    Sequence, Tuple)

from metacog.journal import APPEND_ONLY_TABLES, SQLiteJournal
from metacog.journal_clickhouse import DEFAULT_DATABASE, ClickHouseJournal

logger = logging.getLogger(__name__)

#: What the proxy accepts, and what the API returns : the nine usage tables
#: and the flag events.
JOURNAL_TABLES = APPEND_ONLY_TABLES + ("journal_flags",)

#: The SQLite tables that are mutable state, not a usage log : they stay in
#: `state` (the correspondence table above ; a test keeps it without a hole).
STATE_TABLES = ("tags", "wiki_docs", "wiki_refs", "okf_fields", "wiki_ref_remaps",
                "okf_derived", "wiki_seeds", "wiki_annotations", "wiki_pending",
                "okf_vocab")

#: (flagged table, flag) -> how the memory copy applies the latest value.
#: `useful` is stored as score + 1 (`ClickHouseJournal.mark_useful`).
_FLAG_COLUMNS: Dict[Tuple[str, str], Tuple[str, Callable[[int], Any]]] = {
    ("retrievals", "useful"): ("useful", lambda v: v - 1 if v else None),
    ("forget_events", "merged"): ("merged", int),
    ("merge_ledger", "reverted"): ("reverted", int),
    ("wiki_ops", "reverted"): ("reverted", int),
}

#: The usage reads, all answered by the memory copy.
READ_METHODS = (
    "pending_forgets", "forget_history", "redirect_of", "active_redirects",
    "absorbed_by", "merge_history", "tool_events", "wiki_ops_for_doc",
    "get_wiki_op", "hop_target_counts", "effective_spike", "frequent_paths",
    "chasles_candidates", "co_retrieved_pairs", "modal_next", "collision_events",
    "find_co_retrieved", "node_usefulness", "retrieval_returned_ids",
    "access_timestamps", "useful_retrievals", "counts",
)


class JournalWriter(Protocol):
    """Sends rows of one table (short name, in `JOURNAL_TABLES`). Raises on failure."""

    def append(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None: ...


class JournalLoader(Protocol):
    """Returns every row of one store (a context, or a narrow account of it), by
    table (short name). Raises on failure."""

    def load(self, ctx: str, account: str = "") -> Mapping[str, Sequence[Mapping[str, Any]]]: ...


class _WriterTransport:
    """The `ClickHouseTransport` face of a writer : the parent class only inserts."""

    def __init__(self, writer: JournalWriter, database: str) -> None:
        self.writer = writer
        self._prefix = f"{database}."

    def command(self, sql: str) -> None:       # the schema is the deployment's
        pass

    def insert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        short = table[len(self._prefix):] if table.startswith(self._prefix) else table
        self.writer.append(short, rows)

    def select(self, sql: str, params: Optional[Mapping[str, Any]] = None):
        raise RuntimeError("a ProxyJournal reads from memory, never from ClickHouse")


_INT_COLUMNS = frozenset({"id", "retrieval_id", "rank", "length"})
_FLOAT_COLUMNS = frozenset({"ts", "trigger_distance", "threshold"})


def _typed(column: str, value: Any) -> Any:
    """JSON from the API may render 1.0 as 1 and a UInt64 as a string : the memory
    copy stores what SQLite would have stored."""
    if value is None:
        return None
    if column in _INT_COLUMNS:
        return int(value)
    if column in _FLOAT_COLUMNS:
        return float(value)
    return value


def apply_rows(memory: SQLiteJournal, table: str, rows: Sequence[Mapping[str, Any]]) -> int:
    """Insert journal rows (ClickHouse shape) into a SQLite journal, ids kept.

    ``journal_flags`` rows set the flag column of their target ; they must come in
    ``id`` order (the latest wins). Returns the number of rows applied. The
    memory copy is a read cache, not a journal : updating it is how it reflects
    the flag events.
    """
    if table == "journal_flags":
        n = 0
        for r in rows:
            spec = _FLAG_COLUMNS.get((r["target_table"], r["flag"]))
            if spec is None:
                logger.warning("omni journal: unknown flag %r on %r, ignored",
                               r["flag"], r["target_table"])
                continue
            column, conv = spec
            memory.conn.execute(
                f"UPDATE {r['target_table']} SET {column} = ? WHERE id = ?",
                (conv(int(r["value"])), int(r["target_id"])))
            n += 1
        memory.conn.commit()
        return n
    if table not in APPEND_ONLY_TABLES:
        raise ValueError(f"not a usage table : {table!r}")
    cols = [c[1] for c in memory.conn.execute(f"PRAGMA table_info({table})")]
    n = 0
    for r in rows:
        keep = [c for c in cols if c in r]
        memory.conn.execute(
            f"INSERT OR IGNORE INTO {table} ({', '.join(keep)}) "
            f"VALUES ({', '.join('?' for _ in keep)})",
            [_typed(c, r[c]) for c in keep])
        n += 1
    memory.conn.commit()
    return n


class ProxyJournal(ClickHouseJournal):
    """`JournalProtocol` : writes by the proxy, reads from a memory loaded once."""

    def __init__(self, writer: JournalWriter, ctx: str, *,
                 account: str = "",
                 loader: Optional[JournalLoader] = None,
                 database: str = DEFAULT_DATABASE,
                 state: Optional[SQLiteJournal] = None,
                 flush_interval_s: float = 1.0,
                 max_buffer: int = 50_000,
                 start_flusher: bool = True) -> None:
        super().__init__(_WriterTransport(writer, database), ctx,
                         account=account, database=database, state=state,
                         batch_size=1,
                         max_buffer=max_buffer, ensure_schema=False)
        self.path = "proxy:" + self.path.split(":", 1)[1]
        # `SQLiteJournal`, never `Journal` : a test harness may swap `Journal`
        # for another backend, and the memory copy must stay SQLite.
        self.memory = SQLiteJournal(":memory:")
        for name in READ_METHODS:
            setattr(self, name, getattr(self.memory, name))
        self.loaded: Dict[str, int] = {}
        self.load_error: Optional[str] = None
        if loader is not None:
            self._load(loader)
        self.flush_interval_s = float(flush_interval_s)
        self._send_lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._flusher: Optional[threading.Thread] = None
        if start_flusher and self.flush_interval_s > 0:
            self._flusher = threading.Thread(
                target=self._flush_loop, name=f"omni-journal-proxy:{ctx}", daemon=True)
            self._flusher.start()

    def _load(self, loader: JournalLoader) -> None:
        try:
            tables = loader.load(self.ctx, self.account)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("omni journal: load failed for %s — memory starts empty",
                           self.ctx, exc_info=True)
            return
        for table in JOURNAL_TABLES:            # flags last : their targets exist
            rows = sorted(tables.get(table) or (), key=lambda r: int(r["id"]))
            self.loaded[table] = apply_rows(self.memory, table, rows)

    # -- writes : memory now, proxy later ----------------------------------

    def _append(self, table: str, row: Dict[str, Any], *, last: bool = True) -> None:
        row["context_path"] = self.ctx
        row["account"] = self.account
        with self._lock:
            apply_rows(self.memory, table, [row])
            self._buffer.append((table, row))
            over = len(self._buffer) - self.max_buffer
            if over > 0:
                del self._buffer[:over]
                self.dropped += over
                logger.warning("omni journal: %d rows of %s never reached the "
                               "proxy (%d dropped so far)", over, self.ctx, self.dropped)
        if last:
            if self.flush_interval_s <= 0:
                self.flush()
            else:
                self._wake.set()

    def _flush_locked(self) -> bool:
        # The parent calls this from `flush()` and `close()` under `_lock`.
        # Sending happens outside `_lock` : a slow proxy never blocks a write.
        return True

    def flush(self) -> bool:
        """Send what is buffered. True when nothing is left to send."""
        with self._send_lock:
            with self._lock:
                batch, self._buffer = self._buffer, []
            if not batch:
                return True
            by_table: Dict[str, List[Dict[str, Any]]] = {}
            for table, row in batch:
                by_table.setdefault(table, []).append(row)
            sent: set = set()
            try:
                for table, rows in by_table.items():
                    self.transport.insert(f"{self.db}.{table}", rows)
                    sent.add(table)
            except Exception:
                keep = [(t, r) for t, r in batch if t not in sent]
                with self._lock:
                    self._buffer[:0] = keep
                    over = len(self._buffer) - self.max_buffer
                    if over > 0:
                        del self._buffer[:over]
                        self.dropped += over
                logger.warning("omni journal: send to the proxy failed for %s "
                               "(%d rows kept, %d dropped so far)", self.ctx,
                               len(keep), self.dropped, exc_info=True)
                return False
            return True

    def _flush_loop(self) -> None:
        while not self._closed.is_set():
            self._wake.wait(self.flush_interval_s)
            self._wake.clear()
            if self._buffer:
                self.flush()

    def close(self) -> None:
        self._closed.set()
        self._wake.set()
        if self._flusher is not None:
            self._flusher.join(timeout=5.0)
        self.flush()
        self.memory.close()
        self.state.close()


# -- the served writer and loader ---------------------------------------------

#: The Serve app of a context's infra-proxy (tachikoma `infra_proxy_deploy.app_name`).
PROXY_APP_PREFIX = "infra-proxy"


class ProxyRayWriter:
    """`JournalWriter` through ``infra-proxy:<ctx>`` (``insert_omni_journal``).

    Needs ``ray`` in the gate's environment and a Ray connection (``RAY_ADDRESS``).
    The handle is resolved on first use and kept on success only : resolving it
    costs ~27 s outside a replica (tachikoma ``infra_proxy_deploy.get_handle``),
    paid by the flusher thread, never by a recall. A context without a deployed
    proxy has no ClickHouse journal : its rows wait in the buffer.
    """

    def __init__(self, ctx: str, *, timeout_s: float = 30.0,
                 get_handle: Optional[Callable[[str], Any]] = None) -> None:
        self.ctx = ctx
        self.timeout_s = float(timeout_s)
        self._get_handle = get_handle or _serve_handle
        self._handle: Any = None

    def append(self, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if self._handle is None:
            self._handle = self._get_handle(self.ctx)
        try:
            self._handle.insert_omni_journal.remote(
                table, [dict(r) for r in rows]).result(timeout_s=self.timeout_s)
        except Exception:
            self._handle = None                # re-resolved next time
            raise


def _serve_handle(ctx: str) -> Any:
    import ray
    from ray import serve

    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"),
                 ignore_reinit_error=True, log_to_driver=False)
    return serve.get_app_handle(f"{PROXY_APP_PREFIX}:{ctx}")


class ApiJournalLoader:
    """`JournalLoader` through ``GET /api/memory/journal?ctx=[&account=]`` (tachikoma).

    The bearer is the Neo4j store's : the account's own token when the caller
    serves a narrow account (``store_token``), else the gate's service token
    (``OMNI_STORE_TOKEN`` or ``OMNI_STORE_TOKEN_FILE``).
    """

    def __init__(self, api_url: Optional[str] = None, *,
                 token: Optional[str] = None, timeout_s: float = 30.0) -> None:
        self.api_url = (api_url or os.environ.get("TACHIKOMA_API_URL")
                        or "http://127.0.0.1:8000").rstrip("/")
        self._token = token
        self.timeout_s = float(timeout_s)

    def token(self) -> str:
        if self._token:
            return self._token
        from metacog.store.neo4j_backend import store_token
        tok = store_token.get() or os.environ.get("OMNI_STORE_TOKEN", "")
        path = os.environ.get("OMNI_STORE_TOKEN_FILE", "")
        if not tok and path:
            with open(path, encoding="utf-8") as fh:
                tok = fh.read().strip()
        if not tok:
            raise RuntimeError("no bearer for the journal read "
                               "(OMNI_STORE_TOKEN / OMNI_STORE_TOKEN_FILE)")
        return tok

    def load(self, ctx: str, account: str = "") -> Mapping[str, Sequence[Mapping[str, Any]]]:
        q = {"ctx": ctx, **({"account": account} if account else {})}
        url = f"{self.api_url}/api/memory/journal?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.token()}"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            body = json.load(r)
        tables = body.get("tables")
        if not isinstance(tables, dict):
            raise RuntimeError(f"/api/memory/journal answered without tables : "
                               f"{str(body)[:200]}")
        return tables


def open_proxy_journal(ctx: str, account: str = "", *, state_path: str = ":memory:",
                       **kwargs: Any) -> ProxyJournal:
    """The served form : proxy writer, API loader, mutable state in `state_path`."""
    return ProxyJournal(ProxyRayWriter(ctx), ctx, account=account,
                        loader=ApiJournalLoader(), state=SQLiteJournal(state_path), **kwargs)
