"""`Neo4jBackend` — a `Memory`'s store as governed records of the ontology (OO·V2·M3,
Linear TAC2-1157 / OO2-7 TAC2-1209).

WHERE THE BYTES GO. Not into Neo4j directly : through tachikoma's API, the batch route
`POST /api/memory/points:batch` and its read `GET /api/memory/points` (D6, a working
hypothesis — the ACL has one door, `LawMachine`, and Neo4j is unreachable from outside the
API on purpose). Each point is a `memory_fact` / `memory_thought` / `memory_action` /
`memory_event` record : an `:Entity` node (id, type, `context_path`, owner) and its full row
(vectors included) ; the context state is ONE `memory_store` record, each observator a
`memory_observator`. The route writes the node AND the row (P24) and re-reads the row before
answering.

THE VERBS (`metacog/store/base.py`) :

* `load()` — ONE call that returns every point of the store (an exact scan : no `k`, no page,
  D3), its state and its observators, decoded back to omni's objects field by field. Nothing
  was ever written → `StoreNotFound` (first run). The API or the base cannot answer, or a row
  cannot be decoded → `CorruptStoreError` naming the store : never an empty memory passed off
  as a loaded one (fail-closed).
* `read_if_changed()` — the store's REVISION (`memory_store` row version, bumped by every batch
  that writes) compared with the one this backend last read or wrote ; equal → None without
  reading a point ; moved → the whole stored state, for `Memory`'s merge-on-stale.
* `upsert` / `forget` / `discard` — staged ; `commit(state)` sends them in ONE batch with the
  context state and the revision it read. ZERO DELETION : `discard` archives the record
  (`discarded = True`) — a load no longer serves it, nothing is removed ; a forget travels as the
  point's own `INVALID` state plus the `_forget_log` entry on the store.
* the OPTIMISTIC LOCK : another writer committed since (409) → `StoreConflictError` ;
  `Memory.save()` re-reads, merges and writes again — the pickle's merge-on-stale, per point.
  `lock()` is a per-process lock per store ; between processes the revision is the lock.

IDENTITY (P31). omni's ids are unique per store only (`fact_<hex8>` is 32 bits, note ids are
deterministic) : the route qualifies the NODE id by the store (`mem:<ctx>:<id>`,
`mem:<ctx>/accounts/<acc>:<id>`) and keeps omni's id in `point_id`. Nothing here needs it.

WHICH STORE. `open_backend(storage_path)` keeps the gate's layout : `<root>/<ctx>/memory.pkl`
is the context `ctx`, `<root>/<ctx>/accounts/<acc>/memory.pkl` the narrow account `acc`. No
file is written there (the SQLite usage journal next to it stays, unchanged — `journal.py`).
`METACOG_NEO4J_NAMESPACE` prefixes every context (`<ns>.<ctx>`) : TESTS and the oracle only,
so their stores never land in a real context.

WHO WRITES. The bearer is the store's OWNER (the route refuses another one, 403) : the
gate's service token for a context store (`OMNI_STORE_TOKEN`, or `OMNI_STORE_TOKEN_FILE`), the
narrow account's own token for its store (`store_token` contextvar, set by the caller per
request). No token → `CorruptStoreError` (fail-closed).
"""
from __future__ import annotations

import contextvars
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from metacog.store.base import CorruptStoreError, StoreNotFound

#: The bearer of the CURRENT request, when the caller (the gate) serves a narrow account
#: under that account's own token. Read before `OMNI_STORE_TOKEN`.
store_token: contextvars.ContextVar[str] = contextvars.ContextVar("omni_store_token",
                                                                   default="")

ENV_API = "TACHIKOMA_API_URL"
ENV_TOKEN = "OMNI_STORE_TOKEN"
ENV_TOKEN_FILE = "OMNI_STORE_TOKEN_FILE"
ENV_NAMESPACE = "METACOG_NEO4J_NAMESPACE"
ENV_TIMEOUT = "OMNI_STORE_TIMEOUT"

#: omni's kinds → the ontology's types (one class per kind, D5).
KIND_TYPES = {"FACT": "memory_fact", "THOUGHT": "memory_thought",
              "ACTION": "memory_action", "EVENT": "memory_event"}
TYPE_KINDS = {v: k for k, v in KIND_TYPES.items()}


class StoreConflictError(RuntimeError):
    """Another writer committed to the store since this backend read it (HTTP 409).
    Nothing of the batch was written ; `Memory.save()` re-reads, merges, writes again."""


def scope_of(storage_path: str) -> Tuple[str, str]:
    """(context, narrow account) of a store path in the gate's layout."""
    d = os.path.dirname(os.path.abspath(storage_path))
    parent = os.path.dirname(d)
    if os.path.basename(parent) == "accounts":
        return os.path.basename(os.path.dirname(parent)), os.path.basename(d)
    return os.path.basename(d), ""


# ── codec : omni's objects ⇄ the records' fields, field by field ───────────

def _vec(v: Any) -> List[float]:
    return [float(x) for x in v] if v is not None else []


def _enum_value(v: Any) -> Any:
    return getattr(v, "value", v)


def encode_point(p: Any, ordinal: int) -> Tuple[str, Dict[str, Any]]:
    """(record type, fields) of a `Point`. Refuses what the record cannot hold — an
    executable spec on a non-ACTION point would be lost, so it raises instead."""
    kind = p.kind.value
    t = KIND_TYPES[kind]
    f: Dict[str, Any] = {
        "ordinal": int(ordinal),
        "content": p.content,
        "epistemic_state": p.state.value,
        "keywords": list(p.keywords),
        "keywords_source": _enum_value(p.keywords_source) or "",
        "t_last_obs": float(p.t_last_obs),
        "n_corrob": int(p.n_corrob), "n_contra": int(p.n_contra),
        "n_uses": int(p.n_uses), "n_revision": int(p.n_revision),
        "n_spike": int(p.n_spike), "lineage_depth": int(p.lineage_depth),
        "embedding_orig": _vec(p.embedding_orig),
        "delta_active": _vec(p.delta_active),
        "delta_latent": _vec(p.delta_latent),
        "keywords_embedding": _vec(p.keywords_embedding),
        "memory_tags": list(p.tags),
        "parents": list(p.parents), "children": list(p.children),
        "sequence_prev": p.sequence_prev or "",
        "sequence_next": p.sequence_next or "",
        "update_log": json.dumps([_audit_out(e) for e in p.update_log]),
        "observator_views": json.dumps({k: _view_out(v)
                                        for k, v in p.observator_views.items()}),
    }
    if kind == "ACTION":
        f["exec_spec"] = json.dumps(p.exec_spec) if p.exec_spec is not None else ""
        f["execution_log"] = json.dumps(list(p.execution_log))
    elif p.exec_spec is not None or p.execution_log:
        raise ValueError(f"point {p.id!r} ({kind}) carries an exec_spec/execution_log : "
                         "only a memory_action record holds them")
    return t, f


def _audit_out(e: Any) -> Dict[str, Any]:
    o = e.observation
    return {"node_id": e.node_id,
            "observation": {"source": o.source.value, "signal_type": o.signal_type,
                            "polarity": o.polarity,
                            "target_node_ids": list(o.target_node_ids),
                            "target_is_tuple": isinstance(o.target_node_ids, tuple),
                            "timestamp": o.timestamp, "raw_content": o.raw_content,
                            "observator_id": o.observator_id},
            "confidence_before": e.confidence_before,
            "confidence_after": e.confidence_after,
            "state_before": e.state_before.value, "state_after": e.state_after.value}


def _view_out(v: Any) -> Dict[str, Any]:
    return {"n_corrob": v.n_corrob, "n_contra": v.n_contra, "n_uses": v.n_uses,
            "n_revision": v.n_revision, "state": v.state.value}


def decode_point(entity_type: str, point_id: str, f: Dict[str, Any]) -> Any:
    from metacog.epistemic import (AuditEntry, EpistemicState, Observation, Point,
                                   PointKind, SourceClass)
    from metacog.observator import ObservatorView

    def tup(v: Any) -> tuple:
        return tuple(float(x) for x in v)

    log = []
    for e in json.loads(f.get("update_log") or "[]"):
        o = e["observation"]
        tni = o["target_node_ids"]
        log.append(AuditEntry(
            node_id=e["node_id"],
            observation=Observation(
                source=SourceClass(o["source"]), signal_type=o["signal_type"],
                polarity=o["polarity"],
                target_node_ids=tuple(tni) if o.get("target_is_tuple") else list(tni),
                timestamp=o["timestamp"], raw_content=o.get("raw_content"),
                observator_id=o.get("observator_id") or "default"),
            confidence_before=e["confidence_before"],
            confidence_after=e["confidence_after"],
            state_before=EpistemicState(e["state_before"]),
            state_after=EpistemicState(e["state_after"])))
    views = {k: ObservatorView(n_corrob=v["n_corrob"], n_contra=v["n_contra"],
                               n_uses=v["n_uses"], n_revision=v["n_revision"],
                               state=EpistemicState(v["state"]))
             for k, v in json.loads(f.get("observator_views") or "{}").items()}
    kwe = f.get("keywords_embedding") or []
    spec = f.get("exec_spec") or ""
    p = Point(
        id=point_id, content=f.get("content", ""),
        embedding_orig=tup(f.get("embedding_orig") or []),
        kind=PointKind(TYPE_KINDS[entity_type]),
        delta_active=tup(f.get("delta_active") or []),
        delta_latent=tup(f.get("delta_latent") or []),
        t_last_obs=f.get("t_last_obs", 0.0),
        n_corrob=f.get("n_corrob", 0), n_contra=f.get("n_contra", 0),
        n_uses=f.get("n_uses", 0), n_revision=f.get("n_revision", 0),
        n_spike=f.get("n_spike", 0),
        exec_spec=json.loads(spec) if spec else None,
        state=EpistemicState(f.get("epistemic_state") or "CONJECTURE"),
        keywords=list(f.get("keywords") or []),
        keywords_source=(SourceClass(f["keywords_source"])
                         if f.get("keywords_source") else None),
        keywords_embedding=tup(kwe) if kwe else None,
        tags=list(f.get("memory_tags") or []),
        parents=list(f.get("parents") or []), children=list(f.get("children") or []),
        lineage_depth=f.get("lineage_depth", 0),
        sequence_prev=f.get("sequence_prev") or None,
        sequence_next=f.get("sequence_next") or None,
        update_log=log,
        execution_log=list(json.loads(f.get("execution_log") or "[]")),
        observator_views=views,
    )
    # `Point.__post_init__` normalises tags (and appends the kind tag) : the stored list
    # was written from an already normalised point, so it must come back unchanged —
    # EXCEPT the case of `account:<id>`, which the gate writes in the id's exact case
    # (`tachikoma_gate._tag_account`, TAC-274) and `__post_init__` lowers. The stored list
    # is restored when it differs by case only (the pickle never re-ran `__post_init__`,
    # so it kept that case); any other difference is still refused (TAC-2363).
    stored = list(f.get("memory_tags") or [])
    if p.tags != stored:
        if [t.lower() for t in stored] != p.tags:
            raise ValueError(f"point {point_id!r}: tags changed on decode "
                             f"({stored!r} -> {p.tags!r})")
        p.tags = stored
    return p


def encode_observator(o: Any, ordinal: int) -> Dict[str, Any]:
    return {"ordinal": int(ordinal), "title": o.name, "keywords": list(o.keywords),
            "keywords_embedding": _vec(o.keywords_embedding),
            "parent_observator_id": o.parent_observator_id or "",
            "observed_at": float(o.created_at),
            "n_observations_emitted": int(o.n_observations_emitted)}


def decode_observator(oid: str, f: Dict[str, Any]) -> Any:
    from metacog.observator import Observator
    kwe = f.get("keywords_embedding") or []
    return Observator(id=oid, name=f.get("title", ""), keywords=list(f.get("keywords") or []),
                      keywords_embedding=tuple(float(x) for x in kwe) if kwe else None,
                      parent_observator_id=f.get("parent_observator_id") or None,
                      created_at=f.get("observed_at", 0.0),
                      n_observations_emitted=f.get("n_observations_emitted", 0))


def encode_store(state: Dict[str, Any]) -> Dict[str, Any]:
    log = state.get("conversation_log")
    turns = [{"timestamp": t.timestamp, "speaker": t.speaker, "text": t.text,
              "embedding": _vec(t.embedding) if t.embedding is not None else None,
              "retrieved_point_ids": list(t.retrieved_point_ids)}
             for t in getattr(log, "turns", [])]
    return {"encoder_id": state.get("encoder_id") or "",
            "decay_exponent": float(state.get("decay_exponent", 0.5)),
            "t_clock": float(state.get("_t_clock", 0.0)),
            "spike_total_hops": int(state.get("_spike_total_hops", 0)),
            "conversation_log": json.dumps(turns),
            "forget_log": json.dumps(list(state.get("_forget_log", [])))}


def decode_store(f: Dict[str, Any]) -> Dict[str, Any]:
    from metacog.detectors import ConversationLog, TurnRecord
    turns = [TurnRecord(timestamp=t["timestamp"], speaker=t["speaker"], text=t["text"],
                        embedding=(tuple(float(x) for x in t["embedding"])
                                   if t.get("embedding") is not None else None),
                        retrieved_point_ids=list(t.get("retrieved_point_ids") or []))
             for t in json.loads(f.get("conversation_log") or "[]")]
    return {"conversation_log": ConversationLog(turns=turns),
            "_t_clock": f.get("t_clock", 0.0),
            "_spike_total_hops": f.get("spike_total_hops", 0),
            "decay_exponent": f.get("decay_exponent", 0.5),
            "_forget_log": list(json.loads(f.get("forget_log") or "[]")),
            "encoder_id": f.get("encoder_id") or None}


# ── the backend ────────────────────────────────────────────────────────────

_LOCKS: Dict[Tuple[str, str, str], threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class Neo4jBackend:
    """A store of the ontology, reached through tachikoma's API. See the module docstring."""

    kind = "neo4j"

    def __init__(self, storage_path: Optional[str] = None, *,
                 context_path: Optional[str] = None, account: str = "",
                 api_url: Optional[str] = None, token: Optional[str] = None,
                 namespace: Optional[str] = None, timeout: Optional[float] = None) -> None:
        if context_path is None:
            if not storage_path:
                raise ValueError("Neo4jBackend needs a storage_path or a context_path")
            context_path, account = scope_of(storage_path)
        ns = os.environ.get(ENV_NAMESPACE, "") if namespace is None else namespace
        self.context_path = f"{ns}.{context_path}" if ns else context_path
        self.account = account
        self.api = (api_url or os.environ.get(ENV_API, "http://127.0.0.1:8000")).rstrip("/")
        self._token = token
        self.timeout = float(timeout if timeout is not None
                             else os.environ.get(ENV_TIMEOUT, "120"))
        scope = self.context_path + (f"/accounts/{account}" if account else "")
        self.location = f"neo4j:{self.api}/{scope}"
        self._rev: Optional[int] = None        # store revision last read / written
        self._versions: Dict[str, int] = {}    # point id -> row version
        self._ordinals: Dict[str, int] = {}    # point id -> ordinal
        self._obs_versions: Dict[str, int] = {}
        self._pending: Dict[str, Any] = {}     # point id -> Point
        self._discards: List[str] = []
        self.stats = {"load": 0, "upsert": 0, "forget": 0, "discard": 0, "commit": 0,
                      "conflict": 0}
        self.last_commit: Dict[str, int] = {}

    def __repr__(self) -> str:
        return f"Neo4jBackend({self.location!r})"

    # -- transport ----------------------------------------------------------
    def token(self) -> str:
        # The request's bearer serves a NARROW account's store only : the context's store
        # is the gate's, whoever the request is from (the route refuses another owner).
        tok = (self._token or (store_token.get() if self.account else "")
               or os.environ.get(ENV_TOKEN, ""))
        if not tok and os.environ.get(ENV_TOKEN_FILE):
            try:
                with open(os.environ[ENV_TOKEN_FILE], encoding="utf-8") as fh:
                    tok = fh.read().strip()
            except OSError as e:
                raise CorruptStoreError(f"{self.location} : token file unreadable ({e})") from e
        if not tok:
            raise CorruptStoreError(
                f"{self.location} : no bearer ({ENV_TOKEN} / {ENV_TOKEN_FILE} / store_token) "
                "— refusing to serve a store it cannot read as its owner")
        return tok

    def _query(self) -> str:
        q = {"ctx": self.context_path}
        if self.account:
            q["account"] = self.account
        return urllib.parse.urlencode(q)

    def _call(self, method: str, path: str, body: Optional[dict] = None) -> Tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.api}{path}?{self._query()}", data=data, method=method,
            headers={"Authorization": f"Bearer {self.token()}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                status = r.status
        except urllib.error.HTTPError as e:
            raw, status = e.read(), e.code
        except (OSError, ValueError) as e:     # refused, reset, timeout, bad URL
            raise CorruptStoreError(f"{self.location} unreachable : "
                                    f"{type(e).__name__}: {e}") from e
        try:
            payload = json.loads(raw.decode() or "null")
        except ValueError as e:
            raise CorruptStoreError(f"{self.location} : HTTP {status}, unreadable answer "
                                    f"({len(raw)} bytes)") from e
        return status, payload

    def _ok(self, status: int, payload: Any, what: str) -> Any:
        if status == 200 and isinstance(payload, dict):
            return payload
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        raise CorruptStoreError(f"{self.location} : {what} refused (HTTP {status}) : {detail}")

    # -- lock -----------------------------------------------------------------
    def lock(self) -> threading.RLock:
        key = (self.api, self.context_path, self.account)
        with _LOCKS_GUARD:
            lk = _LOCKS.get(key)
            if lk is None:
                lk = _LOCKS[key] = threading.RLock()
            return lk

    # -- read -----------------------------------------------------------------
    def _read(self) -> Tuple[Optional[Dict[str, Any]], int]:
        status, payload = self._call("GET", "/api/memory/points")
        body = self._ok(status, payload, "load")
        rev = int(body.get("store_version") or 0)
        store = body.get("store")
        rows = body.get("points") or []
        if store is None:
            if rows or body.get("observators"):
                raise CorruptStoreError(
                    f"{self.location} : {len(rows)} point(s) but no memory_store state — "
                    "a half-written store is refused, never served")
            return None, rev
        try:
            points = [decode_point(r["kind"], r["point_id"], r["fields"]) for r in rows]
            observators = {o["observator_id"]: decode_observator(o["observator_id"],
                                                                 o["fields"])
                           for o in body.get("observators") or []}
            snapshot = {"points": points, "observators": observators,
                        **decode_store(store.get("fields") or {})}
        except Exception as e:                # a row that does not decode : fail-closed
            raise CorruptStoreError(f"{self.location} : unreadable row "
                                    f"({type(e).__name__}: {e})") from e
        self._versions = {r["point_id"]: int(r.get("version") or 0) for r in rows}
        self._ordinals = {r["point_id"]: int((r.get("fields") or {}).get("ordinal") or 0)
                          for r in rows}
        self._obs_versions = {o["observator_id"]: int(o.get("version") or 0)
                              for o in body.get("observators") or []}
        return snapshot, rev

    def load(self) -> Dict[str, Any]:
        snapshot, rev = self._read()
        self.stats["load"] += 1
        if snapshot is None:
            self._rev = rev
            raise StoreNotFound(f"{self.location} : nothing written yet")
        self._rev = rev
        return snapshot

    def revision(self) -> int:
        status, payload = self._call("GET", "/api/memory/points:revision")
        return int(self._ok(status, payload, "revision").get("store_version") or 0)

    def read_if_changed(self) -> Optional[Dict[str, Any]]:
        rev = self.revision()
        if self._rev is not None and rev == self._rev:
            return None                        # nobody wrote since : fast path
        snapshot, rev = self._read()
        self._rev = rev
        return snapshot

    # -- point writes : staged, sent by commit ---------------------------------
    def upsert(self, point: Any) -> None:
        self._pending[point.id] = point
        self.stats["upsert"] += 1

    def forget(self, point_id: str, entry: Any) -> None:
        self.stats["forget"] += 1             # the entry travels in the store's forget_log

    def discard(self, point_id: str) -> None:
        self._discards.append(point_id)
        self.stats["discard"] += 1

    def commit(self, state: Dict[str, Any]) -> None:
        pending, self._pending = self._pending, {}
        discards, self._discards = self._discards, []
        nxt = max(self._ordinals.values(), default=-1) + 1
        order = {p.id: i for i, p in enumerate(state.get("points", []))}
        points = []
        ordinals: Dict[str, int] = {}
        for pid in sorted(pending, key=lambda i: order.get(i, len(order))):
            p = pending[pid]
            if pid in self._ordinals:
                ordinal = self._ordinals[pid]
            else:
                ordinal, nxt = nxt, nxt + 1
            ordinals[pid] = ordinal
            t, fields = encode_point(p, ordinal)
            points.append({"kind": t, "point_id": pid,
                           "version": self._versions.get(pid, 0), "fields": fields})
        obs = [{"observator_id": oid, "version": self._obs_versions.get(oid, 0),
                "fields": encode_observator(o, i)}
               for i, (oid, o) in enumerate((state.get("observators") or {}).items())]
        body = {"store_version": self._rev if self._rev is not None else 0,
                "store": encode_store(state), "observators": obs, "points": points,
                "discard": [{"point_id": i, "version": self._versions.get(i, 0)}
                            for i in discards if i in self._versions]}
        status, payload = self._call("POST", "/api/memory/points:batch", body)
        if status == 409:
            self.stats["conflict"] += 1
            # Re-read in full on the retry, whatever moved (the revision or one point's
            # version) : a retry that trusted the cached revision would conflict forever.
            self._rev = None
            raise StoreConflictError(f"{self.location} : {payload}")
        out = self._ok(status, payload, "commit")
        expected = body["store_version"]
        rev = int(out.get("store_version") or 0)
        # A batch that wrote was checked against `expected` : its revision is ours. A no-op
        # batch is not checked : adopting a revision another writer produced would skip
        # the merge of what that writer wrote.
        if out.get("store_written") or rev == expected:
            self._rev = rev
        else:
            self._rev = None
        for key, v in (out.get("versions") or {}).items():
            if key.startswith("observator:"):
                self._obs_versions[key.split(":", 1)[1]] = int(v)
            else:
                self._versions[key] = int(v)
        self._ordinals.update(ordinals)
        self.stats["commit"] += 1
        self.last_commit = {"upsert": len(points), "discard": len(discards),
                            "points": len(state.get("points", [])),
                            "written": int(out.get("written") or 0)}
