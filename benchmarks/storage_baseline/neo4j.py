"""The oracle on `Neo4jBackend` (OO·V2·M3, Linear TAC2-1157 / OO2-8).

Two pieces, for TESTS only (the served migration is OO·V2·M4, `scripts/migrate_pickle_to_neo4j.py`):

* `migrate` — a SIMPLE copy of the stores a `build.py` wrote (pickle) into a test
  namespace of a tachikoma API (its stores must be empty — a store already there is
  refused ; empty them between runs, the namespace's contexts are kept) : every store of `<work>/stores` (contexts and narrow
  accounts) is read with `PickleBackend` and written with `Neo4jBackend` — every point,
  in the pickle's order, plus the state and the observators, in one batch per store. Its
  contexts are created in the API's hierarchy (`<ns>.<ctx>`) ; a narrow account's store is
  written under THAT account's token (the route refuses another writer), after it is
  granted `read`/`write` on its context.
* `factory` — the `--memory-factory` of `run.py` / `latency.py` : the gate's
  `ContextualMemory`, its stores on `Neo4jBackend` (`METACOG_STORE=neo4j`, the namespace in
  `METACOG_NEO4J_NAMESPACE`), a narrow account's store read and written under its token.

    python -m benchmarks.storage_baseline.neo4j migrate --work W --namespace oo2t.rX \\
        --api http://127.0.0.1:8104 --token-file T --account-tokens D
    METACOG_STORE=neo4j METACOG_NEO4J_NAMESPACE=oo2t.rX TACHIKOMA_API_URL=… \\
    OMNI_STORE_TOKEN_FILE=T OMNI_ORACLE_ACCOUNT_TOKENS=D \\
        python -m benchmarks.storage_baseline.run --work W --out O \\
        --memory-factory benchmarks.storage_baseline.neo4j:factory

`--account-tokens D` is a directory of `<account>` files, each holding that account's
bearer (written by `migrate` from the API's login — a test API only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

ENV_ACCOUNT_TOKENS = "OMNI_ORACLE_ACCOUNT_TOKENS"


def _http(api: str, token: str, method: str, path: str,
          body: Optional[dict] = None) -> Tuple[int, Any]:
    req = urllib.request.Request(
        f"{api}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def _stores(work: str) -> List[Tuple[str, str, str]]:
    """(context, account, pickle path) of every store a build wrote."""
    out = []
    root = os.path.join(work, "stores")
    for d, _dirs, files in os.walk(root):
        if "memory.pkl" not in files:
            continue
        rel = os.path.relpath(d, root).split(os.sep)
        ctx, account = (rel[0], rel[2]) if len(rel) == 3 and rel[1] == "accounts" \
            else (rel[0], "")
        out.append((ctx, account, os.path.join(d, "memory.pkl")))
    return sorted(out)


def _ensure_contexts(api: str, token: str, ns: str, contexts: List[str]) -> None:
    root, _, run = ns.partition(".")
    if not run or "." in run:
        raise SystemExit(f"--namespace {ns!r}: expected <galaxy>.<run>")
    def exists(path: str) -> bool:
        return _http(api, token, "GET",
                     f"/api/hierarchy/{urllib.parse.quote(path, safe='')}")[0] == 200

    # Contexts are created ONCE and kept: each adds a hierarchical tag chain, and every
    # register pays for the number of chains (`hierarchical_tags._DEDUP`, see the report
    # of TAC2-1157). A fixed namespace, emptied between runs, keeps that number constant.
    if not exists(root):
        _http(api, token, "POST", "/api/hierarchy/galaxies", {"name": root})
    if not exists(ns):
        code, out = _http(api, token, "POST", "/api/hierarchy/systems",
                          {"name": run, "galaxy_id": root})
        if code != 200:
            raise SystemExit(f"cannot create {ns}: HTTP {code} {out}")
    made = set()
    for ctx in contexts:
        parent = ns
        for part in ctx.split("."):
            path = f"{parent}.{part}"
            if path not in made and exists(path):
                made.add(path)
            if path not in made:
                code, out = _http(api, token, "POST", "/api/hierarchy/spaces",
                                  {"name": part, "system_id": parent})
                if code != 200:
                    raise SystemExit(f"cannot create {path}: HTTP {code} {out}")
                made.add(path)
            parent = path


def _account_token(api: str, admin: str, account: str, ctx: str, out_dir: str) -> str:
    """The account's bearer (dev login of a TEST API), granted read/write on its context."""
    code, body = _http(api, "", "POST", "/api/auth/login",
                       {"username": account, "password": "dev"})
    if code != 200 or not (body or {}).get("token"):
        raise SystemExit(f"no token for {account}: HTTP {code}")
    code, out = _http(api, admin, "POST", "/api/users/acl/share",
                      {"user_id": account, "resource_id": ctx, "resource_type": "context",
                       "actions": ["read", "write"]})
    if code != 200:
        raise SystemExit(f"cannot grant {account} on {ctx}: HTTP {code} {out}")
    os.makedirs(out_dir, mode=0o700, exist_ok=True)
    path = os.path.join(out_dir, account)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(body["token"])
    return body["token"]


def migrate(work: str, ns: str, api: str, token_file: str, tokens_dir: str) -> Dict:
    from metacog.store import Neo4jBackend, PickleBackend, StoreNotFound

    with open(token_file, encoding="utf-8") as fh:
        admin = fh.read().strip()
    stores = _stores(work)
    _ensure_contexts(api, admin, ns, sorted({c for c, _a, _p in stores}))
    report = {}
    for ctx, account, path in stores:
        tok = (_account_token(api, admin, account, f"{ns}.{ctx}", tokens_dir)
               if account else admin)
        snap = PickleBackend(path).load()
        be = Neo4jBackend(context_path=ctx, account=account, api_url=api, token=tok,
                          namespace=ns)
        try:
            be.load()
            raise SystemExit(f"{be.location} already holds a store: a fresh namespace only")
        except StoreNotFound:
            pass
        t0 = time.perf_counter()
        for p in snap["points"]:
            be.upsert(p)
        be.commit(snap)
        dt = time.perf_counter() - t0
        back = be.load()
        same = [p.id for p in back["points"]] == [p.id for p in snap["points"]]
        key = ctx + (f"/accounts/{account}" if account else "")
        report[key] = {"points": len(snap["points"]), "observators": len(snap["observators"]),
                       "written": be.last_commit.get("written"), "seconds": round(dt, 2),
                       "reloaded_same_order": same}
        print(f"[migrate] {key:40s} {len(snap['points']):4d} points in {dt:6.1f} s "
              f"order {'ok' if same else 'DIFFERENT'}", file=sys.stderr)
        if not same:
            raise SystemExit(f"{key}: the reloaded points differ from the pickle")
    return report


def factory(storage_root: str, notes_root: Optional[str]) -> Any:
    """`harness.default_memory_factory` on Neo4jBackend : the same proxy, the same shared
    models ; a narrow account's store under the account's own token."""
    from benchmarks.storage_baseline import harness as H
    from metacog.store import backend_name
    from metacog.store.neo4j_backend import store_token
    from metacog.tachikoma_gate import ContextualMemory

    if backend_name() != "neo4j":
        raise SystemExit("factory: METACOG_STORE=neo4j is required")
    tokens_dir = os.environ.get(ENV_ACCOUNT_TOKENS, "")

    class _Proxy(ContextualMemory):
        def _new_memory(self, key: str) -> Any:
            parts = key.split(os.sep)
            account = parts[2] if len(parts) == 3 and parts[1] == "accounts" else ""
            if not account:
                return super()._new_memory(key)
            with open(os.path.join(tokens_dir, account), encoding="utf-8") as fh:
                tok = fh.read().strip()
            t = store_token.set(tok)
            try:
                m = super()._new_memory(key)
            finally:
                store_token.reset(t)
            m._own_backend()._token = tok
            return m

    proxy = _Proxy(storage_root, notes_root)
    proxy._models = H.shared_models()
    return proxy


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate")
    m.add_argument("--work", required=True)
    m.add_argument("--namespace", required=True)
    m.add_argument("--api", required=True)
    m.add_argument("--token-file", required=True)
    m.add_argument("--account-tokens", required=True)
    m.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = migrate(a.work, a.namespace, a.api.rstrip("/"), a.token_file, a.account_tokens)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
