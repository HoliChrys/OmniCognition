"""
The journal's mutable STATE in the ontology (D10 — Paperclip TAC-2465).

The usage journal is an event log (`journal.APPEND_ONLY_TABLES`, ClickHouse since
JC·V2·M2). The rest of `<store>.journal.db` is mutable STATE : the deepwiki
(docs, refs, OKF fields, seeds, annotations, pending changes), the OKF
vocabulary, the ref remaps, and the tag index. This module keeps that state
where the points already are : governed records in Neo4j, written and read
through tachikoma's API (D6), never into Neo4j directly.

WHERE EACH TABLE GOES

| table | record |
| -- | -- |
| `wiki_docs` | one `memory_wiki_doc` per doc (`doc_type`, `title`, `doc_tags`, `body`, `body_mode`, `observed_at`) |
| `wiki_refs`, `okf_fields`, `okf_derived`, `wiki_seeds`, `wiki_annotations`, `wiki_pending` | JSON lists on the doc's record (`refs`, `okf_fields`, `okf_derived`, `seeds`, `annotations`, `pending`), every column but `doc_id`, in row order |
| `okf_vocab`, `wiki_ref_remaps` | JSON lists on the store's `memory_store` (`okf_vocab`, `ref_remaps`) : store-wide, no doc to hang on |
| `tags` | NOT written : an index of the points' `memory_tags`, which the points' records already carry ; rebuilt at open by `Memory.reindex_tags()` |

A doc omni deleted (`Journal.delete_wiki_doc` removes the doc, its refs and its
OKF fields, and KEEPS its seeds, annotations and pending changes — history, not
the doc) is a record with `discarded = True` that still carries that history.
Nothing is ever removed from the store (zero deletion).

THE FORM : load then memory (the answer J5 gave for the journal). The state is a
`Journal(":memory:")`, loaded ONCE at open from `GET /api/memory/points:wiki`, and
it serves every read. TEMP triggers on the state tables note which doc (or the
store) a write touched ; when the outermost state call returns, each touched doc
is sent WHOLE by `POST /api/memory/points:batch` (`wiki_docs`), its row version
as optimistic lock — unless it encodes exactly as last sent or loaded (a note
re-ingested unchanged costs no call). A READ never calls the API : a write left
pending by a failure is retried by the next write or `flush()`, so an API that is
down never makes a read wait for its timeout. A batch of docs alone does not move the store's revision
(tachikoma#1758) ; a store-wide write (vocab, remaps) does, and omni's next
`save()` re-reads and merges its points, as for any other writer.

FAILURE-SAFE, like every journal of omni : a write the API refuses or cannot take
is logged and kept for the next flush (`pending_docs`) — the in-memory state
already holds it, so nothing the process serves depends on it. A process that
dies before a successful flush loses those writes ; `flush()` says whether
everything left. A conflict (another writer moved a doc) re-reads the versions
and writes once more : this process's state wins, logged — one gate writes a
store, the conflict is a migration or a test running beside it.
The LOAD is not failure-safe : a store that cannot be read is refused
(`CorruptStoreError`), never served empty — an empty wiki would be re-ingested
over the real one.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from metacog.journal import Journal
from metacog.journal_clickhouse import STATE_METHODS
from metacog.store.base import CorruptStoreError

logger = logging.getLogger(__name__)

#: The per-doc tables and the field of `memory_wiki_doc` that carries them.
DOC_TABLES = {
    "wiki_refs": "refs",
    "okf_fields": "okf_fields",
    "okf_derived": "okf_derived",
    "wiki_seeds": "seeds",
    "wiki_annotations": "annotations",
    "wiki_pending": "pending",
}
#: What `delete_wiki_doc` removes with the doc ; the other per-doc rows are history.
DOC_OWN_TABLES = ("wiki_refs", "okf_fields")
#: The store-wide tables and the field of `memory_store` that carries them.
STORE_TABLES = {"okf_vocab": "okf_vocab", "wiki_ref_remaps": "ref_remaps"}
#: Every state table this module persists (`tags` is derived, see the module header).
PERSISTED_TABLES = ("wiki_docs", *DOC_TABLES, *STORE_TABLES)

#: Marker of a store-wide write in the dirty table (a doc id is never NULL).
_STORE = None


def _rows(conn: Any, table: str, where: str = "", args: Tuple = ()) -> List[Dict[str, Any]]:
    cur = conn.execute(f"SELECT * FROM {table} {where} ORDER BY rowid", args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def encode_doc(conn: Any, doc_id: str) -> Optional[Tuple[Dict[str, Any], bool]]:
    """(fields of its `memory_wiki_doc`, discarded) of `doc_id` as `conn` holds it,
    or None when `conn` holds nothing at all for it."""
    doc = _rows(conn, "wiki_docs", "WHERE doc_id = ?", (doc_id,))
    fields: Dict[str, Any] = {}
    held = bool(doc)
    for table, field in DOC_TABLES.items():
        rows = [{k: v for k, v in r.items() if k != "doc_id"}
                for r in _rows(conn, table, "WHERE doc_id = ?", (doc_id,))]
        held = held or bool(rows)
        fields[field] = json.dumps(rows, ensure_ascii=False)
    if not held:
        return None
    if doc:
        d = doc[0]
        fields.update({
            "doc_type": d["type"], "title": d["title"],
            "doc_tags": json.loads(d["tags"]), "body": d["body"],
            "body_mode": d["body_mode"] or "", "observed_at": float(d["ts"]),
        })
    return fields, not doc


def encode_store(conn: Any) -> Dict[str, str]:
    """The `memory_store` fields carrying the store-wide state tables."""
    return {field: json.dumps(_rows(conn, table), ensure_ascii=False)
            for table, field in STORE_TABLES.items()}


def _insert(conn: Any, table: str, row: Dict[str, Any]) -> None:
    cols = list(row)
    conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) "
                 f"VALUES ({', '.join('?' for _ in cols)})", [row[c] for c in cols])


def decode_doc(conn: Any, doc_id: str, fields: Dict[str, Any], discarded: bool) -> None:
    """Write one `memory_wiki_doc` back into `conn`'s tables (the inverse of
    `encode_doc`)."""
    if not discarded:
        _insert(conn, "wiki_docs", {
            "doc_id": doc_id, "type": fields.get("doc_type") or "",
            "title": fields.get("title") or "",
            "tags": json.dumps(list(fields.get("doc_tags") or [])),
            "body": fields.get("body") or "", "ts": float(fields.get("observed_at") or 0.0),
            "body_mode": fields.get("body_mode") or None})
    for table, field in DOC_TABLES.items():
        for r in json.loads(fields.get(field) or "[]"):
            _insert(conn, table, {**r, "doc_id": doc_id})


def decode_store(conn: Any, fields: Dict[str, Any]) -> None:
    for table, field in STORE_TABLES.items():
        for r in json.loads(fields.get(field) or "[]"):
            _insert(conn, table, r)


def all_doc_ids(conn: Any) -> List[str]:
    """Every doc id `conn` holds something for (a deleted doc's history included)."""
    ids: Set[str] = set()
    for table in ("wiki_docs", *DOC_TABLES):
        ids.update(r[0] for r in conn.execute(f"SELECT DISTINCT doc_id FROM {table}"))
    return sorted(ids)


class Neo4jStateJournal(Journal):
    """The mutable state of ONE store, in memory, persisted as records (module header).

    `client` is the store's `Neo4jBackend` (its transport, bearer and scope) — or any
    object with `_call(method, path, body)` and `location`. `load=False` starts empty
    and writes nothing until a write : the migration fills it itself.
    """

    def __init__(self, client: Any, *, load: bool = True) -> None:
        super().__init__(":memory:")
        self.client = client
        self.path = f"{getattr(client, 'location', 'neo4j')}#state"
        self._versions: Dict[str, int] = {}    # doc id -> row version
        self._known: Set[str] = set()          # doc ids the store holds a record for
        self._sent: Dict[str, str] = {}         # doc id -> its encoding as the store holds it
        self.pending_docs: Set[str] = set()    # touched, not yet sent
        self.pending_store = False
        self._depth = 0
        self._lock = threading.RLock()
        self.stats = {"load_docs": 0, "flush": 0, "docs_sent": 0, "store_sent": 0,
                      "conflict": 0, "failed": 0}
        if load:
            self.load()
        self._install_triggers()
        for name in STATE_METHODS:
            setattr(self, name, self._wrap(getattr(super(), name)))

    # -- load -------------------------------------------------------------------

    def _read(self) -> Dict[str, Any]:
        status, payload = self.client._call("GET", "/api/memory/points:wiki")
        if status != 200 or not isinstance(payload, dict):
            detail = payload.get("detail") if isinstance(payload, dict) else payload
            raise CorruptStoreError(f"{self.path} : wiki load refused (HTTP {status}) : "
                                    f"{detail}")
        return payload

    def load(self) -> int:
        """Fill the in-memory tables from the store. Returns the number of docs read."""
        body = self._read()
        docs = body.get("wiki_docs") or []
        try:
            for d in docs:
                f = d.get("fields") or {}
                decode_doc(self.conn, d["doc_id"], f, bool(f.get("discarded")))
            decode_store(self.conn, {"okf_vocab": body.get("okf_vocab"),
                                     "ref_remaps": body.get("ref_remaps")})
        except Exception as e:                 # a row that does not decode : fail-closed
            raise CorruptStoreError(f"{self.path} : unreadable wiki row "
                                    f"({type(e).__name__}: {e})") from e
        self.conn.commit()
        self._versions = {d["doc_id"]: int(d.get("version") or 0) for d in docs}
        self._known = set(self._versions)
        self._sent = {d: self._encoding(d) for d in self._known}
        self.stats["load_docs"] = len(docs)
        return len(docs)

    # -- change capture -----------------------------------------------------------

    def _install_triggers(self) -> None:
        c = self.conn
        c.execute("CREATE TEMP TABLE IF NOT EXISTS _state_dirty (doc_id TEXT)")
        for table in ("wiki_docs", *DOC_TABLES):
            c.execute(f"CREATE TEMP TRIGGER _sd_{table}_i AFTER INSERT ON main.{table} "
                      "BEGIN INSERT INTO _state_dirty VALUES (NEW.doc_id); END")
            c.execute(f"CREATE TEMP TRIGGER _sd_{table}_u AFTER UPDATE ON main.{table} "
                      "BEGIN INSERT INTO _state_dirty VALUES (OLD.doc_id); "
                      "INSERT INTO _state_dirty VALUES (NEW.doc_id); END")
            c.execute(f"CREATE TEMP TRIGGER _sd_{table}_d AFTER DELETE ON main.{table} "
                      "BEGIN INSERT INTO _state_dirty VALUES (OLD.doc_id); END")
        for table in STORE_TABLES:
            for op in ("INSERT", "UPDATE", "DELETE"):
                c.execute(f"CREATE TEMP TRIGGER _sd_{table}_{op[0].lower()} AFTER {op} "
                          f"ON main.{table} BEGIN INSERT INTO _state_dirty VALUES (NULL); END")
        c.commit()

    def _drain(self) -> bool:
        """Move the trigger marks to the pending sets. True when a write left a mark."""
        rows = self.conn.execute("SELECT DISTINCT doc_id FROM _state_dirty").fetchall()
        if not rows:
            return False
        self.conn.execute("DELETE FROM _state_dirty")
        self.conn.commit()
        for (doc_id,) in rows:
            if doc_id is _STORE:
                self.pending_store = True
            else:
                self.pending_docs.add(doc_id)
        return True

    def _wrap(self, fn: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self._depth += 1
                try:
                    return fn(*args, **kwargs)
                finally:
                    self._depth -= 1
                    if self._depth == 0 and self._drain():
                        self.flush()
        call.__name__ = getattr(fn, "__name__", "state_call")
        call.__doc__ = getattr(fn, "__doc__", None)
        return call

    # -- write ----------------------------------------------------------------------

    def _doc(self, doc_id: str) -> Optional[Tuple[Dict[str, Any], bool]]:
        enc = encode_doc(self.conn, doc_id)
        if enc is None and doc_id in self._known:
            return {f: "[]" for f in DOC_TABLES.values()}, True
        return enc

    def _encoding(self, doc_id: str) -> str:
        return json.dumps(self._doc(doc_id), sort_keys=True, ensure_ascii=False)

    def _body(self, doc_ids: Iterable[str], store: bool) -> Dict[str, Any]:
        docs = []
        for doc_id in sorted(doc_ids):
            enc = self._doc(doc_id)
            if enc is None:
                continue                       # touched and gone before it was ever sent
            if self._sent.get(doc_id) == self._encoding(doc_id):
                continue                       # the store already holds exactly this
            fields, discarded = enc
            docs.append({"doc_id": doc_id, "version": self._versions.get(doc_id, 0),
                         "discarded": discarded, "fields": fields})
        body: Dict[str, Any] = {"store_version": None, "wiki_docs": docs}
        if store:
            body["store"] = encode_store(self.conn)
        return body

    def _send(self, body: Dict[str, Any]) -> Tuple[int, Any]:
        return self.client._call("POST", "/api/memory/points:batch", body)

    def flush(self) -> bool:
        """Send every touched doc (and the store-wide state if touched). True when
        nothing is left to send. Never raises (failure-safe, module header)."""
        with self._lock:
            self._drain()
            docs, store = set(self.pending_docs), self.pending_store
            if not docs and not store:
                return True
            body = self._body(docs, store)
            if not body["wiki_docs"] and not store:
                self.pending_docs -= docs
                return True
            try:
                status, payload = self._send(body)
                if status == 409:
                    self.stats["conflict"] += 1
                    logger.warning("omni state: %s moved under us (%s) — re-reading the "
                                   "versions, this process's state is written again",
                                   self.path, payload)
                    fresh = self._read()
                    self._versions.update({d["doc_id"]: int(d.get("version") or 0)
                                           for d in fresh.get("wiki_docs") or []})
                    self._known.update(self._versions)
                    body = self._body(docs, store)
                    status, payload = self._send(body)
                if status != 200 or not isinstance(payload, dict):
                    raise CorruptStoreError(f"HTTP {status} : {payload}")
            except Exception:
                self.stats["failed"] += 1
                logger.warning("omni state: flush failed for %s (%d doc(s)%s kept for the "
                               "next one)", self.path, len(docs),
                               " + store" if store else "", exc_info=True)
                return False
            for key, v in (payload.get("versions") or {}).items():
                if key.startswith("wiki:"):
                    self._versions[key[5:]] = int(v)
            for d in body["wiki_docs"]:
                self._known.add(d["doc_id"])
                self._sent[d["doc_id"]] = self._encoding(d["doc_id"])
            self.pending_docs -= docs
            if store:
                self.pending_store = False
            self.stats["flush"] += 1
            self.stats["docs_sent"] += len(body["wiki_docs"])
            self.stats["store_sent"] += int(store)
            return not self.pending_docs and not self.pending_store

    def mark_all_dirty(self) -> None:
        """Every doc and the store-wide state to send at the next flush (migration)."""
        with self._lock:
            self._drain()
            self.pending_docs.update(all_doc_ids(self.conn))
            self.pending_store = True

    def close(self) -> None:
        self.flush()
        super().close()


class SplitJournal:
    """A SQLite usage journal (the `<store>.journal.db` file) with its STATE served by
    another journal — for a store whose usage journal is not on the proxy yet.

    Every `STATE_METHODS` call and `conn` (the gate lists wiki docs on it) go to
    `state` ; everything else to `usage`. The usage file still holds its own
    (now unused) state tables : they are archived with it, never read.
    """

    def __init__(self, usage: Any, state: Any) -> None:
        self.usage = usage
        self.state = state
        self.path = getattr(usage, "path", "")
        for name in STATE_METHODS:
            setattr(self, name, getattr(state, name))

    @property
    def conn(self) -> Any:
        return self.state.conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self.usage, name)

    def close(self) -> None:
        try:
            self.state.close()
        finally:
            self.usage.close()
