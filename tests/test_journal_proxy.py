"""
JC2-4 (TAC2-1203) — the usage journal written by the infra-proxy, read once
and kept in memory (card J5 : ``proxy_ray`` + ``load_then_memory``).

`ProxyJournal` must answer exactly like the SQLite `Journal` for the same
writes, before AND after a reopen that reloads what the proxy received. The
proxy and the API are fakes here ; the live half (a throwaway ClickHouse as the
proxy's target, `OMNI_TEST_CLICKHOUSE_URL`) is skipped without one.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import metacog.journal_proxy as jp
from metacog.journal import APPEND_ONLY_TABLES, JournalProtocol, SQLiteJournal
from metacog.journal_clickhouse import STATE_METHODS, iter_journal_methods
from metacog.journal_proxy import (JOURNAL_TABLES, ApiJournalLoader,
                                   ProxyJournal, ProxyRayWriter)
from tests.test_journal_backends import (_reads, _script, _transport,
                                         needs_clickhouse)


class _Proxy:
    """What `infra-proxy:<ctx>` keeps : rows by table, `context_path` POSED."""

    def __init__(self) -> None:
        self.rows: dict = {t: [] for t in JOURNAL_TABLES}
        self.calls = 0

    def append(self, table, rows):
        assert table in JOURNAL_TABLES, table
        self.calls += 1
        self.rows[table].extend(dict(r) for r in rows)

    def load(self, ctx, account=""):
        return {t: [r for r in rs
                    if r["context_path"] == ctx and r["account"] == account]
                for t, rs in self.rows.items()}


class _Dead:
    def __init__(self) -> None:
        self.calls = 0

    def append(self, table, rows):
        self.calls += 1
        raise ConnectionError("proxy down")

    def load(self, ctx, account=""):
        raise ConnectionError("api down")


def _open(proxy, ctx="ctx.a", **kw) -> ProxyJournal:
    kw.setdefault("start_flusher", False)
    return ProxyJournal(proxy, ctx, loader=proxy, **kw)


def test_implements_the_whole_protocol():
    j = _open(_Proxy())
    assert list(iter_journal_methods(j)) == []
    assert isinstance(j, JournalProtocol)
    assert set(jp.READ_METHODS).isdisjoint(STATE_METHODS)


def test_answers_like_sqlite_then_after_a_reload():
    ref = SQLiteJournal()
    ref_ids = _script(ref)
    proxy = _Proxy()
    j = _open(proxy)
    ids = _script(j)
    expected = _reads(ref, ref_ids)
    assert _reads(j, ids) == expected          # from memory, before any send
    assert j.flush() and proxy.calls > 0
    j2 = _open(proxy)                          # reopen : one load, then memory
    assert j2.load_error is None
    assert sum(j2.loaded.values()) == sum(len(r) for r in proxy.rows.values())
    assert _reads(j2, ids) == expected


def test_the_correspondence_has_no_hole():
    sqlite_tables = {r[0] for r in SQLiteJournal().conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%'")}
    assert sqlite_tables == set(APPEND_ONLY_TABLES) | set(jp.STATE_TABLES)
    assert not set(APPEND_ONLY_TABLES) & set(jp.STATE_TABLES)
    # every flag SQLite rewrites in place is an event (J4)
    flagged = {(t, c) for t in APPEND_ONLY_TABLES
               for c in ("useful", "merged", "reverted")
               if c in {r[1] for r in SQLiteJournal().conn.execute(
                   f"PRAGMA table_info({t})")}}
    assert flagged == set(jp._FLAG_COLUMNS)


def test_rows_carry_the_context_and_never_an_update():
    proxy = _Proxy()
    j = _open(proxy, ctx="ctx.b")
    rid = j.log_retrieval("q", ["A", "B"], ts=1.0)
    j.mark_useful(rid, 0)
    j.mark_useful(rid, 2)
    j.flush()
    assert {r["context_path"] for rs in proxy.rows.values() for r in rs} == {"ctx.b"}
    flags = proxy.rows["journal_flags"]
    assert [(f["target_table"], f["flag"], f["value"]) for f in flags] == [
        ("retrievals", "useful", 1), ("retrievals", "useful", 3)]
    assert len(proxy.rows["retrievals"]) == 1  # the flag is an event, the row is untouched
    assert _open(proxy, ctx="ctx.b").useful_retrievals() == [(rid, "q", ["A", "B"])]
    assert _open(proxy, ctx="ctx.other").counts() == (0, 0)


def test_a_narrow_account_is_its_own_store():
    proxy = _Proxy()
    ctx_j = _open(proxy, ctx="ctx.b")
    acc_j = _open(proxy, ctx="ctx.b", account="acc-1")
    ctx_j.log_retrieval("q", ["A"], ts=1.0)
    acc_j.log_retrieval("q", ["A", "B"], ts=1.0)
    ctx_j.flush()
    acc_j.flush()
    assert _open(proxy, ctx="ctx.b").counts() == (1, 1)
    assert _open(proxy, ctx="ctx.b", account="acc-1").counts() == (1, 2)
    assert acc_j.path == "proxy:omni_journal/ctx.b/accounts/acc-1"


def test_a_dead_proxy_changes_nothing_for_the_caller():
    dead = _Dead()
    j = ProxyJournal(dead, "ctx.a", loader=dead, max_buffer=3, start_flusher=False)
    assert j.load_error and "api down" in j.load_error
    rid = j.log_retrieval("q", ["A", "B", "C"], ts=1.0)   # never raises
    j.mark_useful(rid, 2)
    j.log_forget("A", "x", superseded_by="B")
    assert j.flush() is False
    # memory still answers what this process wrote
    assert j.counts() == (1, 3)
    assert j.useful_retrievals() == [(rid, "q", ["A", "B", "C"])]
    assert [f["node_id"] for f in j.pending_forgets()] == ["A"]
    assert j.dropped > 0 and len(j._buffer) <= 3          # bounded, counted
    j.log_tags("A", ["a:b"])                               # state still works
    assert j.nodes_with_tag("a:b") == ["A"]
    j.close()


def test_rows_kept_on_failure_are_sent_later_in_order():
    class Flaky(_Proxy):
        down = True

        def append(self, table, rows):
            if self.down:
                raise ConnectionError("down")
            super().append(table, rows)

    p = Flaky()
    j = _open(p)
    j.log_hop("A", "B", ts=1.0)
    assert j.flush() is False
    j.log_hop("B", "C", ts=2.0)
    p.down = False
    assert j.flush() is True
    assert [(r["src_id"], r["dst_id"]) for r in p.rows["hops"]] == [("A", "B"), ("B", "C")]
    assert j._buffer == []


def test_the_flusher_sends_without_being_asked():
    p = _Proxy()
    j = ProxyJournal(p, "ctx.a", flush_interval_s=0.05)
    j.log_hop("A", "B", ts=1.0)
    deadline = time.time() + 5
    while not p.rows["hops"] and time.time() < deadline:
        time.sleep(0.02)
    assert len(p.rows["hops"]) == 1
    j.close()


def test_close_sends_what_is_left():
    p = _Proxy()
    j = ProxyJournal(p, "ctx.a", flush_interval_s=3600)
    j.log_tool_event("t", "created", "new", ts=1.0)
    j.close()
    assert len(p.rows["tool_events"]) == 1


def test_apply_rows_refuses_a_state_table():
    with pytest.raises(ValueError):
        jp.apply_rows(SQLiteJournal(), "tags", [{"node_id": "A"}])


# -- the served writer and loader, against fakes ------------------------------

class _Result:
    def __init__(self, fn):
        self.fn = fn

    def result(self, timeout_s=None):
        self.timeout_s = timeout_s
        return self.fn()


class _Handle:
    def __init__(self, fail=False):
        self.got = []
        self.fail = fail
        outer = self

        class _M:
            def remote(self, table, rows):
                def go():
                    if outer.fail:
                        raise RuntimeError("replica gone")
                    outer.got.append((table, rows))
                    return "ok"
                return _Result(go)

        self.insert_omni_journal = _M()


def test_proxy_ray_writer_calls_insert_omni_journal_and_re_resolves_on_failure():
    handles = [_Handle(fail=True), _Handle()]
    asked = []

    def get_handle(ctx):
        asked.append(ctx)
        return handles[len(asked) - 1]

    w = ProxyRayWriter("ctx.a", get_handle=get_handle)
    with pytest.raises(RuntimeError):
        w.append("hops", [{"id": 1}])
    w.append("hops", [{"id": 1}])
    assert asked == ["ctx.a", "ctx.a"]
    assert handles[1].got == [("hops", [{"id": 1}])]
    w.append("hops", [{"id": 2}])                 # kept on success
    assert asked == ["ctx.a", "ctx.a"]


def test_api_loader_reads_the_route_with_the_gate_bearer(tmp_path, monkeypatch):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            body = json.dumps({"ctx": "a b", "tables": {"hops": [
                {"id": 7, "src_id": "A", "dst_id": "B", "ts": 1.0,
                 "context_path": "a b"}]}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tok = tmp_path / "token"
    tok.write_text("s3cret\n")
    monkeypatch.delenv("OMNI_STORE_TOKEN", raising=False)
    monkeypatch.setenv("OMNI_STORE_TOKEN_FILE", str(tok))
    try:
        loader = ApiJournalLoader(f"http://127.0.0.1:{srv.server_port}")
        j = ProxyJournal(_Proxy(), "a b", loader=loader, start_flusher=False)
    finally:
        srv.shutdown()
    assert seen["path"] == "/api/memory/journal?ctx=a+b"
    assert seen["auth"] == "Bearer s3cret"
    assert j.hop_target_counts() == [("B", 1)]


def test_api_loader_without_a_bearer_fails_closed(monkeypatch):
    monkeypatch.delenv("OMNI_STORE_TOKEN", raising=False)
    monkeypatch.delenv("OMNI_STORE_TOKEN_FILE", raising=False)
    j = ProxyJournal(_Proxy(), "ctx.a", loader=ApiJournalLoader("http://127.0.0.1:9"),
                     start_flusher=False)
    assert j.load_error and "no bearer" in j.load_error


# -- live : a throwaway ClickHouse stands for the proxy's target ---------------

class _ClickHouseAsProxy:
    """The proxy's INSERT (context posed) and the API's bounded SELECT, direct on
    a throwaway server : what tachikoma does, minus Ray and the API."""

    def __init__(self, transport, database):
        self.t = transport
        self.db = database
        from metacog.journal_clickhouse import ClickHouseJournal
        ClickHouseJournal(transport, "schema", database=database)   # DDL

    def append(self, table, rows):
        self.t.insert(f"{self.db}.{table}", rows)

    def load(self, ctx, account=""):
        return {t: self.t.select(
            f"SELECT * EXCEPT logged_at FROM {self.db}.{t} "
            "WHERE context_path = {ctx:String} AND account = {account:String} "
            "ORDER BY id", {"ctx": ctx, "account": account})
            for t in JOURNAL_TABLES}


@needs_clickhouse
def test_live_reload_from_clickhouse_answers_like_sqlite():
    ref = SQLiteJournal()
    expected = _reads(ref, _script(ref))
    p = _ClickHouseAsProxy(_transport(), "omni_journal_test")
    ctx = f"test.proxy.{uuid.uuid4().hex}"
    j = ProxyJournal(p, ctx, loader=p, start_flusher=False)
    ids = _script(j)
    assert j.flush()
    j2 = ProxyJournal(p, ctx, loader=p, start_flusher=False)
    assert _reads(j2, ids) == expected
    assert set(APPEND_ONLY_TABLES) <= set(j2.loaded)


def test_api_json_types_are_restored():
    j = SQLiteJournal()
    jp.apply_rows(j, "retrievals", [{"id": "7", "query_text": "q",
                                     "returned_node_ids": '["A"]', "ts": 1}])
    jp.apply_rows(j, "collision_events", [{"id": 8, "kind": "chasles", "child_id": None,
                                           "parent_ids": "[]", "anchor_ids": "[]",
                                           "trigger_distance": 1, "threshold": None,
                                           "ts": 2}])
    row = j.conn.execute("SELECT id, typeof(ts) FROM retrievals").fetchone()
    assert tuple(row) == (7, "real")
    ev = j.collision_events()[0]
    assert ev["trigger_distance"] == 1.0 and isinstance(ev["trigger_distance"], float)
