"""The live harness of the `neo4j` store tests (OO·V2·M3, Linear TAC2-1157 / OO2-9).

The tests marked `neo4j` run `Memory` on `Neo4jBackend` against a REAL tachikoma API and
its REAL Neo4j — not a fake : the point of the backend is what that pair does (the
governed write, the row, the node). They are not part of the offline suite : without
`OMNI_TEST_API_URL` they are SKIPPED, and `-rs` says why. Once it is set, everything they
need is required — an unreachable API or Neo4j FAILS, it never skips.

Environment :

* `OMNI_TEST_API_URL`     the API (a test instance, never a served one) ;
* `OMNI_TEST_TOKEN_FILE`  a file holding an admin bearer of that API (never on a command
                          line) — the owner the stores are written under ;
* `OMNI_TEST_NEO4J_HTTP`  the Neo4j HTTP endpoint behind that API (`http://127.0.0.1:7475`),
                          read with `NEO4J_USER` (default `neo4j`) / `NEO4J_PASSWORD` from the
                          environment : the measures read the NODES, where the data lands.

ISOLATION OF THE TEST DATA. Every store lives under ONE test namespace, `oo2t.pool`
(`OMNI_TEST_NAMESPACE`) : omni's `ctx` is `oo2t.pool.<ctx>` (`METACOG_NEO4J_NAMESPACE`), and
each test writes its own contexts (`c1`…, the eight of `test_context_isolation`). Nothing
lands in a real context. The contexts are created once through the API's hierarchy and
KEPT : creating contexts is not free on the graph — each one adds a hierarchical tag chain,
and `hierarchical_tags._DEDUP` (run by EVERY register) grew from 0.3 s at 61 chains to
9-25 s at 91 (measured 2026-11-03 on the test Neo4j, 84 of them from fresh per-test
namespaces). So the namespace is fixed and its STORES must be empty when a run starts : the
`live_ns` fixture refuses (fails) a context that already holds a store. Emptying them is
the operator's cleanup, outside the suite (omni never deletes a memory record, so no test
does) : the memory rows and nodes whose context starts with `oo2t.`.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import pytest

API = os.environ.get("OMNI_TEST_API_URL", "").rstrip("/")
NEO4J_HTTP = os.environ.get("OMNI_TEST_NEO4J_HTTP", "").rstrip("/")
ROOT = "oo2t"
NAMESPACE = os.environ.get("OMNI_TEST_NAMESPACE", f"{ROOT}.pool")

requires_live = [
    pytest.mark.neo4j,
    pytest.mark.skipif(not API, reason="neo4j store tests need OMNI_TEST_API_URL "
                                       "(a test tachikoma API) — see tests/neo4j_live.py"),
]


def token() -> str:
    path = os.environ.get("OMNI_TEST_TOKEN_FILE", "")
    assert path, "OMNI_TEST_TOKEN_FILE is required by the neo4j store tests"
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def api(method: str, path: str, body: Optional[dict] = None,
        query: Optional[Dict[str, str]] = None) -> tuple[int, Any]:
    url = f"{API}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
    req = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def cypher(query: str, **params: Any) -> List[Dict[str, Any]]:
    """Rows of a read query on the Neo4j behind the test API."""
    assert NEO4J_HTTP, "OMNI_TEST_NEO4J_HTTP is required by the neo4j store tests"
    auth = base64.b64encode(
        f"{os.environ.get('NEO4J_USER', 'neo4j')}:{os.environ['NEO4J_PASSWORD']}".encode())
    req = urllib.request.Request(
        f"{NEO4J_HTTP}/db/neo4j/tx/commit", method="POST",
        data=json.dumps({"statements": [{"statement": query, "parameters": params}]}).encode(),
        headers={"Authorization": f"Basic {auth.decode()}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    assert not out.get("errors"), out["errors"]
    res = out["results"][0]
    return [dict(zip(res["columns"], row["row"])) for row in res["data"]]


class Namespace:
    """The test contexts, under `NAMESPACE` in the API's hierarchy (created once)."""

    def __init__(self, prefix: str = NAMESPACE) -> None:
        root, _, self.name = prefix.partition(".")
        assert self.name and "." not in self.name, f"namespace {prefix!r}: <galaxy>.<name>"
        self.prefix = prefix
        self._made: set = set()
        if not _exists(root):
            _create("galaxies", {"name": root})
        if not _exists(prefix):
            _create("systems", {"name": self.name, "galaxy_id": root})
        self._made |= {root, prefix}

    def real(self, ctx: str) -> str:
        """The context the API sees for omni's `ctx`."""
        return f"{self.prefix}.{ctx}"

    def ensure(self, ctx: str) -> str:
        parts = ctx.split(".")
        parent = self.prefix
        for p in parts:
            path = f"{parent}.{p}"
            if path not in self._made:
                if not _exists(path):
                    _create("spaces", {"name": p, "system_id": parent})
                self._made.add(path)
            parent = path
        return self.real(ctx)

    def fresh(self, ctx: str, account: str = "") -> str:
        """`ensure(ctx)`, and FAIL if its store is not empty (see the module docstring)."""
        real = self.ensure(ctx)
        body = self.load(ctx, account, include_discarded=True)
        assert body["store"] is None and not body["points"], (
            f"{real} already holds a store ({body['counts']}) : empty the test namespace "
            "first (memory rows and nodes under 'oo2t.'), the suite never deletes")
        return real

    def load(self, ctx: str, account: str = "", include_discarded: bool = False
             ) -> Dict[str, Any]:
        q = {"ctx": self.real(ctx)}
        if account:
            q["account"] = account
        if include_discarded:
            q["include_discarded"] = "true"
        code, body = api("GET", "/api/memory/points", query=q)
        assert code == 200, (code, body)
        return body

    def point_ids(self, ctx: str, account: str = "") -> set:
        return {p["point_id"] for p in self.load(ctx, account)["points"]}

    def nodes_of(self, point_id: str) -> List[str]:
        """`context_path` of every memory NODE of this namespace that carries omni's id."""
        rows = cypher("MATCH (n:Entity) WHERE n.id STARTS WITH $p AND n.id ENDS WITH $s "
                      "RETURN n.context_path AS ctx ORDER BY ctx",
                      p=f"mem:{self.prefix}.", s=f":{point_id}")
        return [r["ctx"] for r in rows]


_SHARED: Dict[str, Namespace] = {}


def shared_namespace(key: str) -> Namespace:
    """ONE `Namespace` object per `key` for the test process (the contexts it knows it
    made). `test_context_isolation`'s tests share the eight contexts : each writes tokens
    no other test writes, and the journals — where most of its measures land — stay fresh
    per test. Their stores must be empty at the first use (`fresh`)."""
    ns = _SHARED.get(key)
    if ns is None:
        ns = _SHARED[key] = Namespace()
        ns.first_use = True
    return ns


def _exists(path: str) -> bool:
    code, _ = api("GET", f"/api/hierarchy/{urllib.parse.quote(path, safe='')}")
    return code == 200


def _create(level: str, body: dict, ok_if_exists: bool = False) -> None:
    code, out = api("POST", f"/api/hierarchy/{level}", body)
    if code != 200 and not (ok_if_exists and code in (400, 409)):
        raise AssertionError(f"cannot create test context {body} : HTTP {code} {out}")


@pytest.fixture
def live_ns(monkeypatch):
    """A fresh namespace, and `METACOG_STORE=neo4j` pointed at it (subprocesses
    inherit it)."""
    ns = shared_namespace("store")
    monkeypatch.setenv("METACOG_STORE", "neo4j")
    monkeypatch.setenv("TACHIKOMA_API_URL", API)
    monkeypatch.setenv("OMNI_STORE_TOKEN", token())
    monkeypatch.setenv("METACOG_NEO4J_NAMESPACE", ns.prefix)
    return ns
