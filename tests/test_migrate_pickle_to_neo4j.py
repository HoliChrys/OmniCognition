"""`scripts/migrate_pickle_to_neo4j.py` — the operation migration (OO·V2·M4, Linear
TAC2-1160 / OO2-10 TAC2-1218), offline.

The script talks to tachikoma's batch route ; here the route is a FAKE served on loopback by
this module (`_FakeRoute`), with the route's rules that the script relies on : a point whose
fields equal the stored ones is not written (`_same`), the optimistic lock (a stale
`store_version` or point `version` → 409, nothing written), the `memory_store` row written on
every batch that writes, discarded points archived and hidden from a load. The real route is
exercised by the rehearsal (OO2-11, the oracle on Neo4j), not here.

  M1  dry-run (default) : the plan names every point, NOTHING is posted ;
  M2  --apply writes every point, the observators and the state ; --verify IDENTIQUE ;
      a second --apply plans 0 and posts nothing (idempotent) ;
  M3  the pickle is read only : same sha256 and mtime, no `.lock`, no journal created ;
  M4  a run cut after one chunk (the API fails the second batch) resumes : the next run
      sends only the rest, every point is written exactly once, --verify IDENTIQUE ;
  M5  a store whose Neo4j clock is ahead of the pickle (written after the bascule) is
      refused, nothing posted ;
  M6  --verify catches one changed field (DIFFÉRENT, rc 1) — the positive witness ;
  M7  a narrow account's store without that account's token is skipped, never written
      under another identity ;
  M8  a forget in the journal that the pickle lost is replayed the same way on both sides ;
  M9  a point Neo4j holds and the pickle does not is refused, archived only on
      --discard-missing (never deleted).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.observator import Observator

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "migrate_pickle_to_neo4j",
    os.path.join(os.path.dirname(_HERE), "scripts", "migrate_pickle_to_neo4j.py"))
mig = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mig)


def _same(stored, wanted):
    w = json.loads(json.dumps(wanted))
    return all(stored.get(k) == v for k, v in w.items())


class _FakeRoute:
    """The batch route's contract, in memory. Lock tokens : 0 = never written."""

    def __init__(self):
        self.stores = {}
        self.posts = []
        self.fail_on_post = None          # 1-based index of the POST that fails (500)
        self.lock = threading.Lock()

    def store(self, ctx, account):
        return self.stores.setdefault((ctx, account), {"points": {}, "observators": {},
                                                       "store": None})

    def load(self, ctx, account, include_discarded):
        s = self.store(ctx, account)
        pts = [{"kind": r["kind"], "point_id": pid, "version": r["version"],
                "fields": r["fields"]}
               for pid, r in s["points"].items()
               if include_discarded or not r["fields"].get("discarded")]
        pts.sort(key=lambda p: (p["fields"].get("ordinal") or 0, p["point_id"]))
        obs = sorted(({"observator_id": oid, "version": r["version"], "fields": r["fields"]}
                      for oid, r in s["observators"].items()),
                     key=lambda o: (o["fields"].get("ordinal") or 0, o["observator_id"]))
        st = s["store"]
        return {"store_version": st["version"] if st else 0, "store": st,
                "observators": obs, "points": pts}

    def batch(self, ctx, account, body):
        with self.lock:
            self.posts.append({**body, "_scope": (ctx, account)})
            if self.fail_on_post == len(self.posts):
                return 500, {"detail": "injected failure"}
            s = self.store(ctx, account)
            rev = s["store"]["version"] if s["store"] else 0
            plan, conflicts = [], []
            for p in body.get("points", []):
                row = s["points"].get(p["point_id"])
                wanted = {**p["fields"], "discarded": False}
                if row and _same(row["fields"], wanted):
                    continue
                if (row["version"] if row else 0) != (p.get("version") or 0):
                    conflicts.append(p["point_id"])
                plan.append(("p", p["point_id"], p["kind"], wanted))
            for d in body.get("discard", []):
                row = s["points"][d["point_id"]]
                if row["fields"].get("discarded"):
                    continue
                plan.append(("p", d["point_id"], row["kind"], {"discarded": True}))
            for o in body.get("observators", []):
                row = s["observators"].get(o["observator_id"])
                if row and _same(row["fields"], o["fields"]):
                    continue
                plan.append(("o", o["observator_id"], None, o["fields"]))
            store_change = body.get("store") is not None and not (
                s["store"] and _same(s["store"]["fields"], body["store"]))
            if not plan and not store_change:
                return 200, {"written": 0, "store_version": rev, "store_written": False,
                             "versions": {}}
            if body.get("store_version") is not None and body["store_version"] != rev:
                conflicts.append("store")
            if conflicts:
                return 409, {"detail": {"conflicts": conflicts}}
            for what, key, kind, fields in plan:
                tab = s["points"] if what == "p" else s["observators"]
                row = tab.get(key) or {"kind": kind, "version": 0, "fields": {}}
                row["fields"] = {**row["fields"], **json.loads(json.dumps(fields))}
                row["version"] += 1
                tab[key] = row
            st = s["store"] or {"version": 0, "fields": {}}
            st["fields"] = {**st["fields"], **json.loads(json.dumps(body.get("store") or {}))}
            st["version"] += 1
            s["store"] = st
            return 200, {"written": len(plan), "store_version": st["version"],
                         "store_written": True, "versions": {}}


@pytest.fixture
def route():
    fake = _FakeRoute()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, payload):
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _q(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            return u.path, q.get("ctx", ""), q.get("account", ""), q

        def do_GET(self):
            path, ctx, acc, q = self._q()
            assert self.headers["Authorization"].startswith("Bearer ")
            if path == "/api/memory/points":
                self._send(200, fake.load(ctx, acc, q.get("include_discarded") == "true"))
            elif path == "/api/memory/points:revision":
                self._send(200, {"store_version": fake.load(ctx, acc, False)["store_version"]})
            else:
                self._send(404, {"detail": path})

        def do_POST(self):
            path, ctx, acc, _q = self._q()
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(*fake.batch(ctx, acc, body))

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    fake.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield fake
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def stores(tmp_path):
    """A gate-layout root : context `alpha` (12 points, an observator, a forget, turns of
    conversation are not needed) and the narrow account `bob` of `alpha`."""
    root = tmp_path / "contexts"
    path = root / "alpha" / "memory.pkl"
    path.parent.mkdir(parents=True)
    m = Memory(encoder=SimpleEncoder(), storage_path=str(path), journal_path="auto")
    for i in range(12):
        m.ingest(f"fact number {i} about topic {i % 3}", id=f"f{i:02d}")
    m.ingest("an action", kind="ACTION", id="a1")
    m.observators["obs1"] = Observator(id="obs1", name="watcher", keywords=["topic"])
    m.forget_node("f03", "superseded in the test")
    m.save()
    m.journal.conn.close()
    acc = root / "alpha" / "accounts" / "bob" / "memory.pkl"
    acc.parent.mkdir(parents=True)
    b = Memory(encoder=SimpleEncoder(), storage_path=str(acc))
    b.ingest("bob's own fact", id="b1")
    b.save()
    tok = tmp_path / "gate.token"
    tok.write_text("gate-token")
    accounts = tmp_path / "accounts"
    accounts.mkdir()
    (accounts / "bob").write_text("bob-token")
    return {"root": str(root), "alpha": str(path), "bob": str(acc), "token": str(tok),
            "accounts": str(accounts), "tmp": tmp_path}


def _run(route, stores, *extra, ctx="alpha", accounts=True):
    rep = stores["tmp"] / "report.json"
    rc = mig.main(["--root", stores["root"], "--ctx", ctx, "--api", route.url,
                   "--token-file", stores["token"], "--report", str(rep), *extra]
                  + (["--account-tokens", stores["accounts"]] if accounts else []))
    return rc, json.loads(rep.read_text())


def _alpha(out):
    return next(s for s in out["stores"] if s["store"] == "alpha")


def test_m1_dry_run_posts_nothing(route, stores):
    rc, out = _run(route, stores)
    a = _alpha(out)
    assert a["mode"] == "dry-run" and a["status"] == "ok"
    assert a["plan"]["points_new"] == 13 and a["plan"]["points_unchanged"] == 0
    assert a["plan"]["store_change"] and a["plan"]["observators"] == 1
    assert route.posts == []


def test_m2_apply_then_idempotent(route, stores):
    rc, out = _run(route, stores, "--apply", "--verify")
    a = _alpha(out)
    assert a["status"] == "ok" and a["written"] == 13 + 1
    assert a["verify"]["verdict"] == "IDENTIQUE", a["verify"]
    assert a["verify"]["pickle"]["by_kind"] == {"ACTION": 1, "FACT": 12}
    assert a["verify"]["pickle"]["invalid"] == ["f03"]
    n = len(route.posts)
    rc, out = _run(route, stores, "--apply", "--verify")
    a = _alpha(out)
    assert a["plan"]["points_new"] == 0 and a["plan"]["points_changed"] == 0
    assert a["plan"]["points_unchanged"] == 13 and not a["plan"]["store_change"]
    assert a["written"] == 0 and len(route.posts) == n          # nothing posted
    assert a["verify"]["verdict"] == "IDENTIQUE"


def test_m3_the_pickle_is_read_only(route, stores):
    p = stores["alpha"]
    before = (hashlib.sha256(open(p, "rb").read()).hexdigest(), os.stat(p).st_mtime_ns)
    os.remove(p + ".lock")
    jpath = p + ".journal.db"
    jbefore = (hashlib.sha256(open(jpath, "rb").read()).hexdigest(), os.stat(jpath).st_mtime_ns)
    rc, out = _run(route, stores, "--apply", "--verify")
    assert _alpha(out)["pickle_unchanged"] is True
    assert (hashlib.sha256(open(p, "rb").read()).hexdigest(), os.stat(p).st_mtime_ns) == before
    assert (hashlib.sha256(open(jpath, "rb").read()).hexdigest(),
            os.stat(jpath).st_mtime_ns) == jbefore
    assert not os.path.exists(p + ".lock")


def test_m4_resume_after_a_cut(route, stores):
    route.fail_on_post = 2                     # the second chunk never lands
    rc, out = _run(route, stores, "--apply", "--chunk", "5")
    assert rc == mig.RC_REFUSED and _alpha(out)["status"] == "error"
    held = route.load("alpha", "", True)
    assert len(held["points"]) == 5            # the first chunk only
    route.fail_on_post = None
    rc, out = _run(route, stores, "--apply", "--chunk", "5", "--verify")
    a = _alpha(out)
    assert a["plan"]["points_new"] == 8 and a["plan"]["points_unchanged"] == 5
    assert a["verify"]["verdict"] == "IDENTIQUE" and rc == 0
    mine = [b for b in route.posts if b["_scope"] == ("alpha", "")]
    sent = [p["point_id"] for b in mine for p in b["points"]]
    landed = [p["point_id"] for b in mine[:1] + mine[2:] for p in b["points"]]
    assert sorted(landed) == sorted(set(landed)) and len(landed) == 13
    assert len(sent) == 13 + 5                 # the failed chunk was sent, never written


def test_m5_neo4j_ahead_is_refused(route, stores):
    _run(route, stores, "--apply")
    st = route.store("alpha", "")["store"]
    st["fields"]["t_clock"] = st["fields"]["t_clock"] + 10.0     # the gate wrote since
    n = len(route.posts)
    rc, out = _run(route, stores, "--apply")
    assert rc == mig.RC_REFUSED and _alpha(out)["status"] == "ahead"
    assert len(route.posts) == n


def test_m6_verify_sees_one_changed_field(route, stores):
    _run(route, stores, "--apply")
    row = route.store("alpha", "")["points"]["f07"]
    row["fields"]["content"] = "tampered"
    rc, out = _run(route, stores, "--verify")
    a = _alpha(out)
    assert rc == mig.RC_DIFFERENT and a["verify"]["verdict"] == "DIFFÉRENT"
    assert a["verify"]["points_different"] == ["f07"]
    assert "content_digest" in a["verify"]["fields_different"]
    assert a["plan"]["points_changed"] == 1     # and the next --apply would repair it


def test_m7_narrow_account_needs_its_own_token(route, stores):
    rc, out = _run(route, stores, "--apply", accounts=False)
    bob = next(s for s in out["stores"] if s["store"] == "alpha/accounts/bob")
    assert bob["status"] == "skipped-no-token" and rc == mig.RC_REFUSED
    assert ("alpha", "bob") not in route.stores
    rc, out = _run(route, stores, "--apply", "--verify")
    bob = next(s for s in out["stores"] if s["store"] == "alpha/accounts/bob")
    assert bob["status"] == "ok" and bob["verify"]["verdict"] == "IDENTIQUE" and rc == 0


def test_m8_journal_forgets_replay_alike(route, stores):
    # a forget the pickle lost : in the journal only (the save never happened)
    from metacog.journal import Journal
    j = Journal(stores["alpha"] + ".journal.db")
    j.log_forget("f05", "lost before save", None, 999.0)
    j.conn.close()
    rc, out = _run(route, stores, "--apply", "--verify")
    v = _alpha(out)["verify"]
    assert v["forgets_replayed"]["pickle"] == ["f05"] == v["forgets_replayed"]["neo4j"]
    assert v["forgets_replayed"]["same_points_after_replay"] and v["verdict"] == "IDENTIQUE"
    assert v["journal"]["rows"]["forget_events"] == 2


def test_m9_a_point_only_in_neo4j_is_never_archived_silently(route, stores):
    _run(route, stores, "--apply")
    st = route.store("alpha", "")
    extra = json.loads(json.dumps(st["points"]["f01"]))
    extra["fields"]["ordinal"] = 99
    st["points"]["f99"] = extra                # written in Neo4j, absent from the pickle
    n = len(route.posts)
    rc, out = _run(route, stores, "--apply")
    a = _alpha(out)
    assert rc == mig.RC_REFUSED and a["status"] == "neo4j-only-points"
    assert a["neo4j_only"] == ["f99"] and len(route.posts) == n
    rc, out = _run(route, stores, "--apply", "--discard-missing", "--verify")
    a = _alpha(out)
    assert a["status"] == "ok" and a["verify"]["verdict"] == "IDENTIQUE" and rc == 0
    assert st["points"]["f99"]["fields"]["discarded"] is True     # archived, not deleted
