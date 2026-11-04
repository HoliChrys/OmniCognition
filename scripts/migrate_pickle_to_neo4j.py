"""Migrate omni's `memory.pkl` stores into the ontology (`Neo4jBackend`) — the OPERATION
tool of OO·V2·M4 (Linear TAC2-1160 / OO2-10 TAC2-1218).

Distinct from `benchmarks/storage_baseline/neo4j.py migrate` (a one-shot copy into a FRESH
test namespace, for the oracle) : this one runs against SERVED stores, so it is

* READ-ONLY on the pickle : the file is unpickled from a read-only fd (`_read_store`), never
  through `Memory` (no `save()`, no `.lock`, no journal opened read-write) ; its sha256 is
  taken before and after, and a store whose pickle moved during the run is reported
  `pickle-moved` (rc 3) — a live writer was still at work, run it again ;
* DRY-RUN BY DEFAULT : it reads what the API already holds and prints the plan ; `--apply`
  writes ;
* IDEMPOTENT : the plan is the DIFF between the pickle and what the store holds, field by
  field, with the route's own rule (`_same` of `routers/memory_points.py` : the stored fields
  equal the JSON round trip of the wanted ones) — a second pass plans and writes 0 ;
* RESUMABLE : points go in chunks (`--chunk`, in the pickle's order), each chunk one
  `POST /api/memory/points:batch` under the store's revision (the optimistic lock) ; a run cut
  after k chunks leaves k chunks written, and the next run's diff sends only the rest ;
* GUARDED : a store whose logical clock (`t_clock`) in Neo4j is AHEAD of the pickle's was
  written after the bascule — overwriting it with the pickle would roll those writes back : it
  is refused (`ahead`, rc 3), whatever the flags. Points that Neo4j holds and the pickle does
  not are never archived silently : the store is refused (`neo4j-only-points`, rc 3) unless
  `--discard-missing` says the pickle dropped them on purpose (then `discarded = True`, zero
  deletion). Another writer committing during the run → 409 → refused (`conflict`, rc 3),
  nothing of that chunk written ;
* VERIFIABLE : `--verify` re-loads the store EXACTLY as the gate would (`Neo4jBackend.load`)
  and compares it with the pickle : points by kind and by state, the order, every point's
  full fingerprint, content and vector digests, the forgets (the `_forget_log`, the INVALID
  points, and the forgets the journal would REPLAY at load — run on both sides against a
  read-only copy of the journal), observators, conversation, clocks, encoder. IDENTIQUE → 0,
  DIFFÉRENT → 1.

WHAT IS NOT MIGRATED. The SQLite usage journal (`memory.pkl.journal.db` : retrievals, wiki,
OKF, merge ledger, forget events…) stays where it is — its move is JC·V2·M2. On `Neo4jBackend`
the gate still opens it at `<storage_path>.journal.db` (`Memory(journal_path="auto")`, the
storage path keeps the gate's layout), so `--verify` reports its tables (read-only) and the
path the gate will open.

WHICH STORES. `--root` walks the gate's layout (`<root>/<ctx>/memory.pkl`,
`<root>/<ctx>/accounts/<acc>/memory.pkl`), narrowed by `--ctx` (repeatable) ; `--store PATH`
names stores one by one. A context store is written under `--token-file` (the gate's service
bearer — the route makes its holder the OWNER of every record, so it must be the identity the
gate writes with afterwards : `OMNI_STORE_TOKEN(_FILE)`). A narrow account's store is written
under THAT account's bearer, read from `--account-tokens DIR/<acc>` ; no file → the store is
`skipped-no-token` (rc 3), never written under another identity. `--namespace` prefixes every
context (`METACOG_NEO4J_NAMESPACE`) : tests and the rehearsal only.

    # plan only (default), then write, then prove :
    python scripts/migrate_pickle_to_neo4j.py --root ~/data_tachikoma/omni/contexts \\
        --api http://127.0.0.1:8000 --token-file ~/.config/omni/store.token
    python scripts/migrate_pickle_to_neo4j.py … --apply
    python scripts/migrate_pickle_to_neo4j.py … --verify --report out.json

Tokens are always read from files, never from the command line.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sqlite3
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:          # run from a checkout, not installed
    sys.path.insert(0, os.path.dirname(HERE))

from metacog.store.base import CorruptStoreError, StoreNotFound, point_fingerprint  # noqa: E402
from metacog.store.neo4j_backend import (  # noqa: E402
    Neo4jBackend,
    encode_observator,
    encode_point,
    encode_store,
    scope_of,
)
from metacog.store.pickle_backend import _read_store  # noqa: E402

RC_OK, RC_DIFFERENT, RC_USAGE, RC_REFUSED = 0, 1, 2, 3

#: The journal tables `--verify` reports (state kept next to the store, not migrated).
JOURNAL_STATE_TABLES = ("forget_events", "wiki_docs", "wiki_refs", "wiki_seeds",
                        "wiki_annotations", "wiki_pending", "okf_fields", "okf_derived",
                        "okf_vocab", "merge_ledger", "tags", "retrievals")


# ── files ───────────────────────────────────────────────────────────────────


def file_print(path: str) -> Dict[str, Any]:
    """sha256, size and mtime of a file, read on a read-only fd ; {} when absent."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return {}
    h = hashlib.sha256()
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def discover(root: str, contexts: List[str]) -> List[str]:
    """Every store under `root` in the gate's layout (sorted), narrowed to `contexts`."""
    out = []
    for ctx in sorted(os.listdir(root)):
        if contexts and ctx not in contexts:
            continue
        d = os.path.join(root, ctx)
        if not os.path.isdir(d):
            continue
        if os.path.isfile(os.path.join(d, "memory.pkl")):
            out.append(os.path.join(d, "memory.pkl"))
        acc_root = os.path.join(d, "accounts")
        if os.path.isdir(acc_root):
            for acc in sorted(os.listdir(acc_root)):
                p = os.path.join(acc_root, acc, "memory.pkl")
                if os.path.isfile(p):
                    out.append(p)
    return out


def read_token(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        tok = fh.read().strip()
    if not tok:
        raise SystemExit(f"empty token file {path}")
    return tok


# ── the wanted rows, and the diff with what the store holds ─────────────────


def _norm(v: Any) -> Any:
    return json.loads(json.dumps(v))


def same(stored: Dict[str, Any], wanted: Dict[str, Any]) -> bool:
    """The route's rule (`routers/memory_points._same`) : every wanted field, after a JSON
    round trip, equals the stored one."""
    w = _norm(wanted)
    return all(stored.get(k) == v for k, v in w.items())


def wanted_rows(snap: Dict[str, Any]) -> Tuple[List[Tuple[str, str, Dict]], List[Tuple[str, Dict]],
                                              Dict[str, Any]]:
    """(points, observators, store) as the batch route receives them — the backend's codec,
    the ordinal = the position in the pickle (what `Neo4jBackend.commit` gives a fresh store)."""
    points = []
    for i, p in enumerate(snap.get("points", [])):
        t, fields = encode_point(p, i)
        points.append((p.id, t, fields))
    obs = [(oid, encode_observator(o, i))
           for i, (oid, o) in enumerate((snap.get("observators") or {}).items())]
    return points, obs, encode_store(snap)


def _get_held(be: Neo4jBackend) -> Tuple[int, Dict[str, Any]]:
    """What the store holds, discarded points included (the load route, one GET). Not
    `be._call` : it builds the query string itself, without `include_discarded`."""
    q = be._query() + "&include_discarded=true"
    req = urllib.request.Request(f"{be.api}/api/memory/points?{q}", method="GET",
                                 headers={"Authorization": f"Bearer {be.token()}"})
    try:
        with urllib.request.urlopen(req, timeout=be.timeout) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except (OSError, ValueError) as e:
        raise CorruptStoreError(f"{be.location} unreachable : {type(e).__name__}: {e}") from e
    payload = json.loads(raw.decode() or "null")
    if status != 200 or not isinstance(payload, dict):
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        raise CorruptStoreError(f"{be.location} : load refused (HTTP {status}) : {detail}")
    return status, payload


def plan_store(snap: Dict[str, Any], held: Dict[str, Any]) -> Dict[str, Any]:
    """The writes that bring the store to the pickle, and nothing else."""
    pts, obs, store = wanted_rows(snap)
    rows = {r["point_id"]: r for r in held.get("points") or []}
    hobs = {o["observator_id"]: o for o in held.get("observators") or []}
    hstore = held.get("store")
    write, new, changed, unchanged = [], 0, 0, 0
    for pid, t, fields in pts:
        row = rows.get(pid)
        if row is not None and row.get("kind") != t:
            raise CorruptStoreError(f"point {pid!r} is a {row.get('kind')} in Neo4j, a {t} in "
                                    "the pickle : a kind never changes — refusing")
        if row is not None and same(row.get("fields") or {}, {**fields, "discarded": False}):
            unchanged += 1
            continue
        if row is None:
            new += 1
        else:
            changed += 1
        write.append({"kind": t, "point_id": pid,
                      "version": int(row.get("version") or 0) if row else 0,
                      "fields": fields})
    in_pickle = {pid for pid, _t, _f in pts}
    discard = [{"point_id": pid, "version": int(r.get("version") or 0)}
               for pid, r in rows.items()
               if pid not in in_pickle and not (r.get("fields") or {}).get("discarded")]
    obs_write = [{"observator_id": oid,
                  "version": int((hobs.get(oid) or {}).get("version") or 0),
                  "fields": f}
                 for oid, f in obs
                 if not (oid in hobs and same(hobs[oid].get("fields") or {}, f))]
    store_change = not (hstore is not None and same(hstore.get("fields") or {}, store))
    return {"points": write, "new": new, "changed": changed, "unchanged": unchanged,
            "discard": discard, "observators": obs_write, "store": store,
            "store_change": store_change,
            "held_t_clock": ((hstore or {}).get("fields") or {}).get("t_clock"),
            "held_revision": int(held.get("store_version") or 0)}


def batches(plan: Dict[str, Any], chunk: int) -> List[Dict[str, Any]]:
    """The plan cut in chunks of points ; the store state rides on EVERY chunk (so the
    `memory_store` row is the pickle's from the first one on — the route writes that row on
    every batch that writes), the observators on the first, the discards on the last."""
    pts = plan["points"]
    parts = [pts[i:i + chunk] for i in range(0, len(pts), chunk)] or [[]]
    out = []
    for i, part in enumerate(parts):
        body = {"store": plan["store"], "points": part,
                "observators": plan["observators"] if i == 0 else [],
                "discard": plan["discard"] if i == len(parts) - 1 else []}
        if part or body["observators"] or body["discard"] or plan["store_change"]:
            out.append(body)
    return out


# ── verify : the store as the gate loads it, against the pickle ─────────────


def _vec_bytes(v: Any) -> bytes:
    if v is None:
        return b"\x00none"
    return struct.pack(f"<{len(v)}d", *[float(x) for x in v])


def digests(snap: Dict[str, Any]) -> Dict[str, Any]:
    pts = snap.get("points", [])
    hc, hv = hashlib.sha256(), hashlib.sha256()
    for p in pts:
        hc.update(json.dumps([p.id, p.kind.value, p.state.value, p.content, list(p.tags)],
                             ensure_ascii=False).encode())
        for v in (p.embedding_orig, p.delta_active, p.delta_latent, p.keywords_embedding):
            hv.update(_vec_bytes(v))
    turns = getattr(snap.get("conversation_log"), "turns", [])
    ht = hashlib.sha256(json.dumps(
        [[t.timestamp, t.speaker, t.text, list(t.retrieved_point_ids)] for t in turns],
        ensure_ascii=False).encode())
    return {
        "points": len(pts),
        "by_kind": dict(sorted(Counter(p.kind.value for p in pts).items())),
        "by_state": dict(sorted(Counter(p.state.value for p in pts).items())),
        "content_digest": hc.hexdigest(),
        "vector_digest": hv.hexdigest(),
        "forget_log": len(snap.get("_forget_log", [])),
        "forget_log_digest": hashlib.sha256(json.dumps(
            snap.get("_forget_log", []), sort_keys=True, default=str).encode()).hexdigest(),
        "invalid": sorted(p.id for p in pts if p.state.value == "INVALID"),
        "observators": sorted((snap.get("observators") or {}).keys()),
        "turns": len(turns),
        "turns_digest": ht.hexdigest(),
        "t_clock": snap.get("_t_clock"),
        "spike_total_hops": snap.get("_spike_total_hops"),
        "decay_exponent": snap.get("decay_exponent"),
        "encoder_id": snap.get("encoder_id"),
    }


def journal_copy(journal: str, into: str) -> Optional[str]:
    """A consistent copy of the journal, read through a read-only connection (`backup`)."""
    if not os.path.isfile(journal):
        return None
    dst = os.path.join(into, "journal.db")
    src = sqlite3.connect(f"file:{journal}?mode=ro", uri=True)
    try:
        out = sqlite3.connect(dst)
        src.backup(out)
        out.close()
    finally:
        src.close()
    return dst


def journal_state(journal: str) -> Dict[str, Any]:
    if not os.path.isfile(journal):
        return {"path": journal, "present": False}
    con = sqlite3.connect(f"file:{journal}?mode=ro", uri=True)
    try:
        have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                  for t in JOURNAL_STATE_TABLES if t in have}
    finally:
        con.close()
    return {"path": journal, "present": True, "rows": counts}


def replayed_forgets(snap: Dict[str, Any], journal_db: Optional[str]) -> Dict[str, Any]:
    """What `Memory.load` would replay from the journal onto this snapshot (TAC-323), run
    with omni's own code on a COPY of the snapshot and of the journal."""
    if journal_db is None:
        return {"replayed": [], "points_after": None}
    from metacog.journal import Journal
    from metacog.memory import Memory

    class _Replay:                       # the attributes `_replay_forgets` reads, nothing more
        _forget_is_live = Memory._forget_is_live
        _replay_forgets = Memory._replay_forgets

    r = _Replay()
    r.journal = Journal(journal_db)
    r.points = copy.deepcopy(snap.get("points", []))
    r._forget_log = copy.deepcopy(list(snap.get("_forget_log", [])))
    r.storage_path = "(verify)"
    try:
        ids = r._replay_forgets()
    finally:
        r.journal.conn.close()
    return {"replayed": sorted(ids),
            "points_after": [point_fingerprint(p) for p in r.points],
            "forget_log_after": len(r._forget_log)}


def verify_store(pk: Dict[str, Any], nj: Dict[str, Any], journal: str) -> Dict[str, Any]:
    dp, dn = digests(pk), digests(nj)
    diffs = [k for k in dp if dp[k] != dn[k]]
    order_same = [p.id for p in pk["points"]] == [p.id for p in nj["points"]]
    by_id = {p.id: p for p in nj["points"]}
    point_diff = [p.id for p in pk["points"]
                  if p.id not in by_id or point_fingerprint(p) != point_fingerprint(by_id[p.id])]
    nobs = nj.get("observators") or {}
    obs_diff = sorted(oid for oid, o in (pk.get("observators") or {}).items()
                      if oid not in nobs or vars(o) != vars(nobs[oid]))
    obs_diff += sorted(oid for oid in nobs if oid not in (pk.get("observators") or {}))
    with tempfile.TemporaryDirectory(prefix="omni-verify-") as tmp:
        jdb = journal_copy(journal, tmp)
        rp = replayed_forgets(pk, jdb)
        if jdb:                                   # a fresh copy for the other side
            os.remove(jdb)
            jdb = journal_copy(journal, tmp)
        rn = replayed_forgets(nj, jdb)
    replay_same = (rp["replayed"] == rn["replayed"]
                   and rp["points_after"] == rn["points_after"])
    ok = not diffs and order_same and not point_diff and not obs_diff and replay_same
    return {"verdict": "IDENTIQUE" if ok else "DIFFÉRENT",
            "pickle": dp, "neo4j": dn, "fields_different": diffs,
            "order_same": order_same, "points_different": point_diff[:50],
            "n_points_different": len(point_diff), "observators_different": obs_diff,
            "forgets_replayed": {"pickle": rp["replayed"], "neo4j": rn["replayed"],
                                 "same_points_after_replay": replay_same},
            "journal": {**journal_state(journal),
                        "opened_by_gate_at": journal,
                        "migrated": False}}


# ── one store ───────────────────────────────────────────────────────────────


def migrate_store(path: str, *, api: str, token: str, account_tokens: str, namespace: str,
                  apply: bool, verify: bool, chunk: int, timeout: float,
                  discard_missing: bool = False,
                  log=print) -> Dict[str, Any]:
    ctx, account = scope_of(path)
    key = ctx + (f"/accounts/{account}" if account else "")
    rep: Dict[str, Any] = {"store": key, "pickle_path": path, "mode": "apply" if apply else
                           "dry-run"}
    before = file_print(path)
    rep["pickle_before"] = before
    if not before or before["size"] == 0:
        rep["status"] = "empty-pickle"
        return rep
    if account:
        tf = os.path.join(account_tokens, account) if account_tokens else ""
        if not tf or not os.path.isfile(tf):
            rep["status"] = "skipped-no-token"
            rep["detail"] = (f"no bearer for account {account!r} ({tf or '--account-tokens'})"
                             " : a narrow account's store is written by that account only")
            return rep
        token = read_token(tf)
    t0 = time.perf_counter()
    snap, _st = _read_store(path)
    be = Neo4jBackend(context_path=ctx, account=account, api_url=api, token=token,
                      namespace=namespace, timeout=timeout)
    rep["neo4j_store"] = be.location
    rep["counts"] = {"points": len(snap.get("points", [])),
                     "observators": len(snap.get("observators") or {}),
                     "turns": len(getattr(snap.get("conversation_log"), "turns", [])),
                     "forget_log": len(snap.get("_forget_log", []))}
    _status, held = _get_held(be)
    plan = plan_store(snap, held)
    rev = plan["held_revision"]
    pk_clock = float(snap.get("_t_clock", 0.0))
    ahead = (held.get("store") is not None and plan["held_t_clock"] is not None
             and float(plan["held_t_clock"]) > pk_clock)
    rep["plan"] = {"points_new": plan["new"], "points_changed": plan["changed"],
                   "points_unchanged": plan["unchanged"], "discard": len(plan["discard"]),
                   "observators": len(plan["observators"]),
                   "store_change": plan["store_change"], "revision_read": rev,
                   "t_clock": {"pickle": pk_clock, "neo4j": plan["held_t_clock"]}}
    if ahead:
        rep["status"] = "ahead"
        rep["detail"] = (f"Neo4j's clock {plan['held_t_clock']} is ahead of the pickle's "
                         f"{pk_clock} : the store was written after the bascule, the pickle "
                         "would roll it back — refused")
        return rep
    if plan["discard"] and not discard_missing:
        ids = [d["point_id"] for d in plan["discard"]]
        rep["status"] = "neo4j-only-points"
        rep["neo4j_only"] = ids[:50]
        rep["detail"] = (f"{len(ids)} point(s) live in Neo4j and not in the pickle : either the "
                         "pickle dropped them since the last pass (then --discard-missing "
                         "archives them), or something wrote the store since — refused")
        return rep
    todo = batches(plan, chunk)
    rep["plan"]["batches"] = len(todo)
    written, sent = 0, 0
    if apply:
        for i, body in enumerate(todo):
            body = {**body, "store_version": rev}
            tb = time.perf_counter()
            status, out = be._call("POST", "/api/memory/points:batch", body)
            if status == 409:
                rep["status"] = "conflict"
                rep["detail"] = (f"batch {i + 1}/{len(todo)} : another writer committed "
                                 f"(409) — nothing of it written ; {out}")
                rep["written"] = written
                return rep
            out = be._ok(status, out, f"batch {i + 1}/{len(todo)}")
            rev = int(out.get("store_version") or rev)
            written += int(out.get("written") or 0)
            sent += len(body["points"])
            log(f"[migrate] {key:40s} batch {i + 1}/{len(todo)} : {len(body['points'])} "
                f"point(s) sent, {out.get('written')} written, revision {rev}, "
                f"{time.perf_counter() - tb:6.1f} s")
    rep["written"] = written
    rep["seconds_write"] = round(time.perf_counter() - t0, 2)
    if verify:
        tv = time.perf_counter()
        try:
            nj = Neo4jBackend(context_path=ctx, account=account, api_url=api, token=token,
                              namespace=namespace, timeout=timeout).load()
        except StoreNotFound:
            rep["verify"] = {"verdict": "DIFFÉRENT", "detail": "nothing in Neo4j"}
        else:
            journal = f"{path}.journal.db"
            rep["verify"] = verify_store(snap, nj, journal)
        rep["seconds_verify"] = round(time.perf_counter() - tv, 2)
    after = file_print(path)
    rep["pickle_after"] = after
    rep["pickle_unchanged"] = after == before
    if not rep["pickle_unchanged"]:
        rep["status"] = "pickle-moved"
        rep["detail"] = ("the pickle changed during the run (a live writer) : what was "
                         "written is consistent with the version read — run again")
        return rep
    rep["status"] = "ok"
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="OMNI_STORAGE_ROOT : every store in the gate's layout")
    src.add_argument("--store", action="append", default=[],
                     help="one memory.pkl (repeatable)")
    ap.add_argument("--ctx", action="append", default=[],
                    help="with --root : only this context (repeatable)")
    ap.add_argument("--api", default=os.environ.get("TACHIKOMA_API_URL",
                                                    "http://127.0.0.1:8000"))
    ap.add_argument("--token-file", required=True,
                    help="the gate's bearer (owner of the context stores)")
    ap.add_argument("--account-tokens", default="",
                    help="directory of <account> files, each that account's bearer")
    ap.add_argument("--namespace", default="", help="TESTS and rehearsal only")
    ap.add_argument("--apply", action="store_true", help="write (default : dry-run)")
    ap.add_argument("--discard-missing", action="store_true",
                    help="archive (discarded=True) the points Neo4j holds and the pickle "
                         "no longer has — default : refuse the store")
    ap.add_argument("--verify", action="store_true",
                    help="reload from Neo4j and compare with the pickle")
    ap.add_argument("--chunk", type=int, default=200, help="points per batch")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per HTTP call")
    ap.add_argument("--report", default="", help="write the JSON report there")
    a = ap.parse_args(argv)
    if a.chunk < 1:
        ap.error("--chunk must be >= 1")
    stores = discover(a.root, a.ctx) if a.root else [os.path.abspath(p) for p in a.store]
    if not stores:
        print("no store found", file=sys.stderr)
        return RC_USAGE
    token = read_token(a.token_file)
    reports, rc = [], RC_OK
    for path in stores:
        log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
        try:
            r = migrate_store(path, api=a.api.rstrip("/"), token=token,
                              account_tokens=a.account_tokens, namespace=a.namespace,
                              apply=a.apply, verify=a.verify, chunk=a.chunk,
                              discard_missing=a.discard_missing,
                              timeout=a.timeout, log=log)
        except CorruptStoreError as e:
            ctx, account = scope_of(path)
            r = {"store": ctx + (f"/accounts/{account}" if account else ""),
                 "pickle_path": path, "status": "error", "detail": str(e)}
        reports.append(r)
        v = (r.get("verify") or {}).get("verdict", "")
        p = r.get("plan") or {}
        log(f"[migrate] {r['store']:40s} {r['status']:16s} "
            f"new={p.get('points_new', '-')} changed={p.get('points_changed', '-')} "
            f"unchanged={p.get('points_unchanged', '-')} discard={p.get('discard', '-')} "
            f"store={'yes' if p.get('store_change') else 'no'} "
            f"written={r.get('written', '-')} {r.get('seconds_write', '-')} s"
            + (f" verify={v}" if v else "")
            + (f" — {r['detail']}" if r.get("detail") else ""))
        if r["status"] not in ("ok", "empty-pickle"):
            rc = RC_REFUSED
        elif v == "DIFFÉRENT" and rc == RC_OK:
            rc = RC_DIFFERENT
    out = {"api": a.api, "namespace": a.namespace, "mode": "apply" if a.apply else "dry-run",
           "stores": reports, "rc": rc}
    if a.report:
        fd = os.open(a.report, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=str)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
