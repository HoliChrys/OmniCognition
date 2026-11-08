"""C1 (TAC-934) — context isolation, proven on ALL FOUR lanes, end to end.

Before this file, isolation was proven on ONE lane: recall, measured live on
two contexts. Capture, wiki and index had no test. Here every lane goes
through the REAL gated app (`build_gated_app`: the context middleware, the
`ContextualMemory` proxy, the FastMCP tools) over MCP streamable HTTP — the
exact handshake tachikoma's `McpTransport` speaks — and every assertion is
measured WHERE THE DATA LANDS: the SQLite journal of each context
(`<ctx>/memory.pkl.journal.db`) and its store (`<ctx>/memory.pkl`), for ALL
the contexts of the storage root, never only the two witnesses.

THE WITNESSES. `iso-alpha` (c) and `iso-beta` (c') are siblings under the root
and share no other ancestor. Six bystanders complete the eight contexts the
live root holds: the root `global`, the `tachikoma` chain down to `GenAI`,
and one child under each witness (`iso-alpha.child` is c's descendant — the
direction "a parent never reads a child" and "nothing climbs up" is pinned
on it).

THE FOUR LANES (rule C3, TAC-936: inheritance is served AT QUERY TIME by the
caller, never copied into a store — so an omni store holds ONLY its own
context's memories, and every hit's `origin_ctx` is the context asked):

| lane    | assertion                                                       |
| ------- | --------------------------------------------------------------- |
| recall  | `recall(c', q)` never returns c's fact; the retrieval is logged  |
|         | in c''s journal only (`retrieval_id`)                           |
| capture | `remember(c, f)` lands in c's journal and store, nowhere else    |
| wiki    | a `notes:*` doc of c is listed by `wiki_list(c)` only — not by   |
|         | a sibling, not by a descendant (C3: no copy), not by an ancestor |
| index   | a record projection indexed into c leaves nothing in c'         |

TWO TRAPS THAT ALREADY LIED, PINNED HERE:

* THE CREDIBLE EMPTINESS — a query that returns nothing reads as an absence.
  Every negative assertion is DOUBLED by its positive: the fact IS in c (same
  query, same tool), the journals scanned ARE the eight (counted), the
  sibling's recall DID answer (its own decoy).
* THE DEFAULT CONTEXT — no header is a 400 that births no memory, never a
  default store. Nobody "fixes" that 400 by giving it a default.

Plus the hole this test found (TAC-934): an MCP session served the context
of its `initialize`, whatever header a later call carried — a session opened
under c, called with c''s header, was served c's fact. The gate now binds a
session to its context and refuses the mismatch (409). And the ACL is asked
once per session (TAC-299), again on any other bearer, context or account.

The encoder is SimpleEncoder (hash), the reranker none: deterministic, no
model download, no network — runnable in CI.

ON BOTH BACKENDS (OO·V2·M3, TAC2-1157 / OO2-9). Every test of this file runs on the
pickle store AND on `Neo4jBackend` (`METACOG_STORE=neo4j`, the `gate` fixture is
parametrized; the `neo4j` run needs a live test API + Neo4j, `tests/neo4j_live.py`, and
is skipped without one). No case is removed or relaxed: what changes is WHERE the data is
measured — the journals stay SQLite files on both; the store is the pickle file (a byte
search) or the store's rows read back through the API's load route PLUS the memory nodes
in Neo4j (one node per point, in the context of the write and no other).
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid

import pytest

from metacog.defaults import SimpleEncoder
from tests.neo4j_live import Namespace, requires_live, shared_namespace

# Under the tree root `tachikoma`: the deployed gate maps ONLY the tree's
# descendants to a notes folder (`notes_folder`, D2 / TAC-938) — any other
# name is "outside" and would leave the wiki lane measuring nothing.
C = "tachikoma.iso-alpha"       # the witness that knows the fact
C2 = "tachikoma.iso-beta"       # its sibling — must never see it
CHILD = "tachikoma.iso-alpha.child"  # c's descendant — inherits nothing by copy
CONTEXTS = ["global", "tachikoma", "tachikoma.paralelle",
            "tachikoma.paralelle.GenAI", C, CHILD, C2, "tachikoma.iso-beta.child"]

#: Tokens that exist nowhere else: a byte search for them is unambiguous.
FACT = "The iso-alpha vault code is vermillon-7731."
FACT_TOKEN = "vermillon-7731"
QUERY = "what is the vault code vermillon"


# ── the gated app, the MCP handshake, the measures ──────────────────────

class Gate:
    """The real gated app over HTTP, one MCP session per call (McpTransport)."""

    def __init__(self, client, root: str, acl_calls: list, refuse: set,
                 ns: "Namespace | None" = None):
        self.client = client
        self.root = root
        #: The live namespace when the stores are `Neo4jBackend`'s, else None (pickle).
        self.ns = ns
        #: Every question asked to the (fake) ACL: (token, ctx), in order.
        self.acl_calls = acl_calls
        #: (token, ctx) pairs the fake ACL refuses (403).
        self.refuse = refuse

    def headers(self, ctx: str | None) -> dict:
        h = {"Accept": "application/json, text/event-stream",
             "Content-Type": "application/json",
             "Authorization": "Bearer test-token"}
        if ctx is not None:
            h["x-tachikoma-context"] = ctx
        return h

    def open(self, ctx: str) -> dict:
        """initialize + initialized; the headers of the opened session."""
        h = self.headers(ctx)
        r = self.client.post("/mcp", headers=h, json={
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "c1-isolation", "version": "0"}}})
        assert r.status_code == 200, r.text
        h["mcp-session-id"] = r.headers["mcp-session-id"]
        r = self.client.post("/mcp", headers=h, json={
            "jsonrpc": "2.0", "method": "notifications/initialized"})
        assert r.status_code in (200, 202), r.text
        return h

    def raw_call(self, headers: dict, tool: str, args: dict):
        return self.client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": args}})

    def call(self, ctx: str, tool: str, args: dict):
        r = self.raw_call(self.open(ctx), tool, args)
        assert r.status_code == 200, r.text
        frames = [ln[5:].strip() for ln in r.text.splitlines()
                  if ln.startswith("data:")]
        payload = json.loads(frames[0] if frames else r.text)
        assert "error" not in payload, payload
        result = payload["result"]
        assert not result.get("isError"), result
        sc = result.get("structuredContent")
        if sc is not None:
            return sc.get("result", sc)
        texts = [b["text"] for b in result.get("content", [])
                 if b.get("type") == "text"]
        # No structured content (a bare `-> list`): one text block per item.
        values = [json.loads(t) for t in texts]
        return values[0] if len(values) == 1 else values

    # what tachikoma sends: OmniEngine maps remember → `ingest`, recall →
    # `retrieve` (memory_engines.NATIVE_TOOLS); the contract names exist too.
    def remember(self, ctx: str, content: str, tags: list[str],
                 tool: str = "ingest") -> dict:
        args = {"content": content, "tags": tags}
        if tool == "ingest":
            args["kind"] = "FACT"
        return self.call(ctx, tool, args)

    def recall(self, ctx: str, query: str, tool: str = "retrieve") -> list:
        hits = self.call(ctx, tool, {"query": query, "k": 5})
        hits = hits if isinstance(hits, list) else [hits]
        return [h for h in hits if isinstance(h, dict) and h.get("id")]

    def wiki(self, ctx: str) -> list[str]:
        return [d["doc_id"] for d in self.call(ctx, "wiki_list",
                                               {"prefix": "notes:"})["docs"]]

    # ── where the data lands ─────────────────────────────────────────
    def journals(self) -> dict[str, str]:
        """{ctx: journal path} for EVERY journal under the storage root."""
        found = {}
        for d, _dirs, files in os.walk(self.root):
            for f in files:
                if f.endswith(".journal.db"):
                    found[os.path.relpath(d, self.root)] = os.path.join(d, f)
        return found

    def sql(self, ctx: str, query: str, params: tuple = ()) -> list:
        with sqlite3.connect(self.journals()[ctx]) as conn:
            return conn.execute(query, params).fetchall()

    def usage_sql(self, ctx: str, query: str, params: tuple,
                  clickhouse_query: str) -> list:
        """A read of the USAGE journal of `ctx` : the SQLite file, or ClickHouse
        when the suite runs on it (OMNI_TEST_JOURNAL=clickhouse|proxy, conftest)."""
        if os.environ.get("OMNI_TEST_JOURNAL") not in ("clickhouse", "proxy"):
            return self.sql(ctx, query, params)
        from tests.conftest import clickhouse_usage_rows
        rows = clickhouse_usage_rows(self.journals()[ctx], clickhouse_query,
                                     **{f"p{i}": v for i, v in enumerate(params)})
        return [tuple(r.values()) for r in rows]

    def tag_rows(self, tag: str) -> dict[str, int]:
        """Rows of `tag` in the journal of EVERY context (add_tag lowercases)."""
        return {ctx: self.sql(ctx, "SELECT COUNT(*) FROM tags WHERE tag = ?",
                              (tag.lower(),))[0][0]
                for ctx in self.journals()}

    def stores_holding(self, token: str) -> list[str]:
        """The contexts whose `memory.pkl` holds `token` (pickle keeps str
        as UTF-8: a byte search is the store itself, not an API's opinion).

        On Neo4j: the contexts whose store ROWS hold `token` (the load route, every
        point of the store), and each holding point's NODE checked to live in that
        context and in no other of the namespace."""
        if self.ns is not None:
            return self._neo4j_holding(token)
        hits = []
        for ctx in CONTEXTS:
            path = os.path.join(self.root, ctx, "memory.pkl")
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    if token.encode() in fh.read():
                        hits.append(ctx)
        return sorted(hits)

    def _neo4j_holding(self, token: str) -> list[str]:
        hits = []
        for ctx in CONTEXTS:
            body = self.ns.load(ctx)
            held = [p["point_id"] for p in body["points"]
                    if token in p["fields"].get("content", "")]
            if held:
                hits.append(ctx)
            for pid in held:
                assert self.ns.real(ctx) in self.ns.nodes_of(pid), (ctx, pid)
        return sorted(hits)

    def touch_all(self) -> None:
        """Open the eight memories, so the scans below cover eight journals —
        and assert it: a scan over fewer files would prove nothing."""
        for ctx in CONTEXTS:
            self.wiki(ctx)
        assert sorted(self.journals()) == sorted(CONTEXTS)


def _notes(root, rel: str, name: str, body: str) -> None:
    folder = root.joinpath(*rel.split("/"), "notes")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.md").write_text(body, encoding="utf-8")


@pytest.fixture(params=["pickle", pytest.param("neo4j", marks=requires_live)])
def gate(request, tmp_path, monkeypatch):
    """The gated app on a fresh storage root, ACL faked to 'yes', the test
    encoder in place of the ONNX pair (TAC-237's `models()` asks these).
    `neo4j`: the same app, its stores on `Neo4jBackend` in a fresh namespace."""
    ns = None
    if request.param == "neo4j":
        ns = shared_namespace("isolation")
        for ctx in CONTEXTS:
            if getattr(ns, "first_use", False):
                ns.fresh(ctx)               # empty at the start of the run
            else:
                ns.ensure(ctx)
        ns.first_use = False
        from tests.neo4j_live import API, token
        monkeypatch.setenv("METACOG_STORE", "neo4j")
        monkeypatch.setenv("TACHIKOMA_API_URL", API)
        monkeypatch.setenv("OMNI_STORE_TOKEN", token())
        monkeypatch.setenv("METACOG_NEO4J_NAMESPACE", ns.prefix)
    import metacog.defaults as defaults
    monkeypatch.setattr(defaults, "make_encoder", lambda: SimpleEncoder())
    monkeypatch.setattr(defaults, "make_reranker", lambda: None)

    notes = tmp_path / "tachikoma"
    # Each witness and c's child carry ONE note whose token exists nowhere
    # else. The mapping is the gate's: `iso-alpha` → <notes>/iso-alpha/notes,
    # `iso-alpha.child` → <notes>/iso-alpha/child/notes.
    _notes(notes, "iso-alpha", "alpha-runbook",
           "# Alpha runbook\n\nThe alpha relay frequency is saffron-5519.")
    _notes(notes, "iso-beta", "beta-runbook",
           "# Beta runbook\n\nThe beta relay frequency is cobalt-2290.")
    _notes(notes, "iso-alpha/child", "child-runbook",
           "# Child runbook\n\nThe child relay frequency is jade-8846.")

    from starlette.testclient import TestClient

    from metacog.tachikoma_gate import build_gated_app
    from metacog.tachikoma_gate import Denied
    acl_calls: list = []
    refuse: set = set()

    def fake_authorize(tok: str, ctx: str) -> str:
        acl_calls.append((tok, ctx))
        if (tok, ctx) in refuse:
            raise Denied(f"refus de test pour {ctx!r}")
        return "tester"

    root = str(tmp_path / "store")
    outer, _mcp, _inner = build_gated_app(root, str(notes),
                                          authorize_fn=fake_authorize)
    # The SDK's DNS-rebinding guard wants a loopback Host, as served live.
    with TestClient(outer, base_url="http://127.0.0.1:8788") as client:
        yield Gate(client, root, acl_calls, refuse, ns)


# ── the default context: there is none ─────────────────────────────────

def test_no_header_is_400_and_births_no_memory(gate):
    """Fail-closed, fixed by test: no header → 400, and NOTHING on disk —
    not a default store, not a folder. Nobody turns this 400 into a default."""
    before = sorted(os.walk(gate.root)) if os.path.exists(gate.root) else []
    for headers in (gate.headers(None),
                    {**gate.headers(None), "x-tachikoma-context": "  "}):
        r = gate.client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        assert r.status_code == 400
        assert "x-tachikoma-context" in r.json()["detail"]
    after = sorted(os.walk(gate.root)) if os.path.exists(gate.root) else []
    assert after == before == []
    # …and the gate does serve once the header is there (the 400 is the
    # header's, not a broken app's).
    assert gate.wiki(C) == ["notes:alpha-runbook"]


# ── lane 1: recall ─────────────────────────────────────────────────────

@pytest.mark.parametrize("tool", ["retrieve", "recall"])
def test_recall_never_serves_another_contexts_fact(gate, tool):
    fact = gate.remember(C, FACT, ["c1:recall-witness"])
    decoy = gate.remember(C2, "The iso-beta vault code is cobalt-1180.",
                          ["c1:recall-decoy"])
    child = gate.remember(CHILD, "The child vault code is jade-6604.",
                          ["c1:recall-decoy"])
    gate.touch_all()

    # POSITIVE: the same query, the same tool, in c — the fact comes back.
    in_c = gate.recall(C, QUERY, tool)
    assert fact["id"] in [h["id"] for h in in_c]

    for other, own in ((C2, decoy), (CHILD, child)):
        query = f"{QUERY} asked from {other}"      # unique per context
        hits = gate.recall(other, query, tool)
        # POSITIVE: the recall DID answer — its own memory, not an empty
        # result that would read as an absence.
        assert own["id"] in [h["id"] for h in hits]
        # NEGATIVE: never c's fact, by id or by content.
        assert fact["id"] not in [h["id"] for h in hits]
        assert all(FACT_TOKEN not in h.get("content", "") for h in hits)
        # WHERE IT LANDS: the retrieval_id names a row of THIS context's
        # journal, listing what was served — and only this journal logged it.
        rid = hits[0]["retrieval_id"]
        row = gate.usage_sql(
            other, "SELECT returned_node_ids FROM retrievals WHERE id = ?", (rid,),
            "SELECT returned_node_ids FROM {db}.retrievals "
            "WHERE context_path = {ctx:String} AND id = {p0:UInt64}")
        assert row and own["id"] in json.loads(row[0][0])
        assert fact["id"] not in json.loads(row[0][0])
        logged = {ctx: gate.usage_sql(
            ctx, "SELECT COUNT(*) FROM retrievals WHERE query_text = ?", (query,),
            "SELECT count() FROM {db}.retrievals "
            "WHERE context_path = {ctx:String} AND query_text = {p0:String}")[0][0]
                  for ctx in gate.journals()}
        assert logged == {ctx: int(ctx == other) for ctx in CONTEXTS}


# ── lane 2: capture ────────────────────────────────────────────────────

@pytest.mark.parametrize("tool", ["ingest", "remember"])
def test_a_capture_lands_in_its_context_only(gate, tool):
    gate.touch_all()
    gate.remember(C, FACT, ["c1:capture-witness"], tool)
    # POSITIVE then NEGATIVE, on the eight journals and the eight stores.
    assert gate.tag_rows("c1:capture-witness") == \
        {ctx: int(ctx == C) for ctx in CONTEXTS}
    assert gate.stores_holding(FACT_TOKEN) == [C]


def test_nothing_climbs_up_a_childs_capture_stays_in_the_child(gate):
    """Rule C3 §2: a write lands in the context of the call — a parent never
    reads its child's memory, the root no more than the parent."""
    gate.touch_all()
    gate.remember(CHILD, "The child archive key is umber-3317.",
                  ["c1:child-witness"])
    assert gate.tag_rows("c1:child-witness") == \
        {ctx: int(ctx == CHILD) for ctx in CONTEXTS}
    assert gate.stores_holding("umber-3317") == [CHILD]


# ── lane 3: the wiki ───────────────────────────────────────────────────

def test_a_contexts_notes_are_listed_by_that_context_only(gate):
    """Rule C3 §3 (TAC-936): the wiki of a context is ITS notes. An ancestor's
    notes are not copied into a descendant (the caller reads the ancestor at
    query time); a sibling's never reach it; the old inherited prefix
    `notes:<ancestor>/…` is gone from every list."""
    lists = {ctx: gate.wiki(ctx) for ctx in CONTEXTS}
    # POSITIVE: each context that has notes lists exactly its own.
    assert lists[C] == ["notes:alpha-runbook"]
    assert lists[C2] == ["notes:beta-runbook"]
    assert lists[CHILD] == ["notes:child-runbook"]
    # NEGATIVE: c's doc is listed nowhere else — sibling, descendant,
    # ancestors, bystanders.
    assert [ctx for ctx, docs in lists.items()
            if "notes:alpha-runbook" in docs] == [C]
    assert [ctx for ctx, docs in lists.items()
            if "notes:child-runbook" in docs] == [CHILD]
    # No doc id marks a foreign source any more.
    assert not [d for docs in lists.values() for d in docs
                if any(d.startswith(f"notes:{ctx}/") for ctx in CONTEXTS)]
    # WHERE IT LANDS: the journal rows and the content points of the note.
    docs = {ctx: [r[0] for r in gate.sql(ctx, "SELECT doc_id FROM wiki_docs")]
            for ctx in CONTEXTS}
    assert [ctx for ctx, ids in docs.items()
            if "notes:alpha-runbook" in ids] == [C]
    assert gate.tag_rows("note:notes:alpha-runbook") == \
        {ctx: int(ctx == C) for ctx in CONTEXTS}
    assert gate.tag_rows(f"ctx:{C}") == \
        {ctx: int(ctx == C) for ctx in CONTEXTS}
    # The note's CONTENT point is served from c only (it is a RAG point, not
    # yet pickled: the gate ingests notes in memory, the next write saves).
    query = "alpha relay frequency saffron"
    assert any("saffron-5519" in h["content"] for h in gate.recall(C, query))
    for other in CONTEXTS:
        if other != C:
            assert all("saffron-5519" not in h.get("content", "")
                       for h in gate.recall(other, query))


# ── lane 4: the index ──────────────────────────────────────────────────

def test_an_indexed_record_leaves_nothing_in_another_context(gate):
    """The index lane (TAC-928): tachikoma's `memory_index.index_record`
    writes a record's `memory_projection()` with tags `memory_tags()` =
    [type, id] into the memory OF THE RECORD'S CONTEXT. A record of c leaves
    nothing in c' — not a journal row, not a byte of its store, not a hit."""
    gate.touch_all()
    projection = ("AgentSession sess-c1-witness — name: iso-alpha probe, "
                  "state: active, description: indigo-4402 relay")
    tags = ["agentsession", "sess-c1-witness"]
    point = gate.remember(C, projection, tags)
    for tag in tags:
        assert gate.tag_rows(tag) == {ctx: int(ctx == C) for ctx in CONTEXTS}
    assert gate.stores_holding("indigo-4402") == [C]
    # …and the read side agrees with the disk.
    assert point["id"] in [h["id"] for h in gate.recall(C, "indigo-4402 relay")]
    assert point["id"] not in [h["id"]
                               for h in gate.recall(C2, "indigo-4402 relay")]


# ── the hole this test found: a session serves ONE context ─────────────

def test_a_session_never_serves_another_context_than_its_own(gate):
    """Measured before the fix: a session opened under c, called with c''s
    header (authorized for c'), was served c's fact — the server task of a
    session runs under the context of its `initialize`. Refused now (409),
    and an unknown session is refused (404), never served."""
    fact = gate.remember(C, FACT, ["c1:session-witness"])
    session = gate.open(C)
    # POSITIVE: the session serves the context it was opened under.
    r = gate.raw_call(session, "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 200 and fact["id"] in r.text
    # NEGATIVE: the same session under c''s header is refused, served nothing.
    crossed = {**session, "x-tachikoma-context": C2}
    r = gate.raw_call(crossed, "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 409
    assert FACT_TOKEN not in r.text and fact["id"] not in r.text
    # …nor under another account of the same context.
    r = gate.raw_call({**session, "x-tachikoma-account": "tester"},
                      "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 409
    # A session id the gate never bound: refused, never served.
    r = gate.raw_call({**gate.headers(C), "mcp-session-id": "forged"},
                      "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 404


# ── one authorization per session (TAC-299) ────────────────────────────
#
# `authorize` asks tachikoma three questions per call, each up to
# OMNI_ACL_TIMEOUT (30 s). A remote_mcp call is three HTTP requests
# (initialize, notification, tools/call) inside the lobby's 60 s: the gate
# now asks once per session, and again on ANY difference.

def test_a_full_handshake_asks_tachikoma_once(gate):
    """initialize + notifications/initialized + tools/call: ONE ACL round
    (it was three). The call is really served — not a cheap refusal."""
    fact = gate.remember(C, FACT, ["tac-299:witness"])
    gate.acl_calls.clear()
    session = gate.open(C)
    r = gate.raw_call(session, "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 200 and fact["id"] in r.text
    assert gate.acl_calls == [("test-token", C)]


def test_another_bearer_on_the_same_session_is_checked_again(gate):
    session = gate.open(C)
    gate.acl_calls.clear()
    other = {**session, "Authorization": "Bearer other-token"}
    r = gate.raw_call(other, "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 200
    assert gate.acl_calls == [("other-token", C)]
    # …and its refusal is served, never the session's earlier yes.
    gate.refuse.add(("other-token", C))
    r = gate.raw_call(other, "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 403
    # The refusal is not kept either: the session's own bearer still passes
    # on its yes, without a new question.
    gate.acl_calls.clear()
    r = gate.raw_call(session, "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 200 and gate.acl_calls == []


def test_another_context_on_the_same_session_is_checked_again(gate):
    """TAC-934 still holds: the other context goes through tachikoma, then
    the session refuses it (409) — the cache never shortcuts isolation."""
    fact = gate.remember(C, FACT, ["tac-299:crossed"])
    session = gate.open(C)
    gate.acl_calls.clear()
    crossed = {**session, "x-tachikoma-context": C2}
    r = gate.raw_call(crossed, "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 409
    assert FACT_TOKEN not in r.text and fact["id"] not in r.text
    assert gate.acl_calls == [("test-token", C2)]
    # Refused by the ACL for that context: its 403, before any session talk.
    gate.refuse.add(("test-token", C2))
    r = gate.raw_call(crossed, "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 403
    # Another account of the same context: checked again too.
    gate.acl_calls.clear()
    r = gate.raw_call({**session, "x-tachikoma-account": "tester"},
                      "retrieve", {"query": QUERY, "k": 5})
    assert r.status_code == 409 and gate.acl_calls == [("test-token", C)]


def test_a_refused_initialize_is_never_kept(gate):
    """A 403 at initialize binds no session; asking again asks the ACL again."""
    gate.refuse.add(("test-token", C))
    for _ in range(2):
        r = gate.client.post("/mcp", headers=gate.headers(C), json={
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "tac-299", "version": "0"}}})
        assert r.status_code == 403 and "mcp-session-id" not in r.headers
    assert gate.acl_calls == [("test-token", C)] * 2


def test_the_yes_dies_with_the_session(gate):
    session = gate.open(C)
    r = gate.client.delete("/mcp", headers=session)
    assert r.status_code < 300, r.text
    gate.acl_calls.clear()
    r = gate.raw_call(session, "retrieve", {"query": QUERY, "k": 5})
    # Closed: the gate no longer knows the session, and asked the ACL anew.
    assert r.status_code == 404 and gate.acl_calls == [("test-token", C)]


# ── the deepwiki of a LONG session follows its notes (TAC-938) ─────────

def _session_wiki(gate, headers) -> list[str]:
    r = gate.raw_call(headers, "wiki_list", {"prefix": "notes:"})
    assert r.status_code == 200, r.text
    frames = [ln[5:].strip() for ln in r.text.splitlines() if ln.startswith("data:")]
    result = json.loads(frames[0] if frames else r.text)["result"]
    sc = result.get("structuredContent") or json.loads(result["content"][0]["text"])
    return [d["doc_id"] for d in sc.get("result", sc)["docs"]]


def test_a_note_added_mid_session_is_listed_in_that_session(gate, tmp_path):
    """A tool call runs in the session's task, whose contextvars were copied
    at `initialize` (TAC-934): a per-request stamp held in a contextvar would
    freeze the wiki for the whole session. The note added between two calls
    of ONE session must be listed by the second."""
    headers = gate.open(C)
    assert _session_wiki(gate, headers) == ["notes:alpha-runbook"]
    _notes(tmp_path / "tachikoma", "iso-alpha", "alpha-late",
           "# Late note\n\nAdded while the session was open.")
    assert _session_wiki(gate, headers) == ["notes:alpha-late", "notes:alpha-runbook"]


# ── Neo4j only: the identity of a point is its store's (P31, TAC2-1157) ──

@pytest.fixture
def neo4j_gate(gate):
    if gate.ns is None:
        pytest.skip("Neo4j identity: the pickle has one file per store, no shared key")
    return gate


def test_the_same_omni_id_in_two_contexts_is_two_nodes(neo4j_gate):
    """omni's ids are unique per store only (`<doc_id>#<sha12>` is deterministic,
    `fact_<hex8>` is 32 bits). Kept as the node key, the second context's write would
    REWRITE the first's node (`MERGE (n:Entity {id})`) and succeed. Same id, two
    contexts: two nodes, each in its context, each store serving its own content."""
    g = neo4j_gate
    pid = f"fact_{uuid.uuid4().hex[:8]}"
    a = g.call(C, "ingest", {"content": f"alpha owns ochre-{pid}", "id": pid,
                             "kind": "FACT"})
    b = g.call(C2, "ingest", {"content": f"beta owns teal-{pid}", "id": pid,
                              "kind": "FACT"})
    assert a["id"] == b["id"] == pid
    assert g.ns.nodes_of(pid) == sorted([g.ns.real(C), g.ns.real(C2)])
    assert g.stores_holding(f"ochre-{pid}") == [C]
    assert g.stores_holding(f"teal-{pid}") == [C2]


def test_witness_a_broken_mapping_is_seen(neo4j_gate, monkeypatch):
    """POSITIVE WITNESS (rule 9): break the isolation on purpose — the backend files
    c''s store under c — and the measures above must SEE it, by rows and by nodes."""
    from metacog.store import neo4j_backend as nb
    real = nb.scope_of

    def leaky(path):
        ctx, account = real(path)
        return (C if ctx == C2 else ctx), account
    monkeypatch.setattr(nb, "scope_of", leaky)
    g = neo4j_gate
    token = f"leak-{uuid.uuid4().hex[:8]}"
    g.remember(C2, f"The iso-beta vault code is {token}.", ["c1:leak"])
    # The leak is SEEN: c''s write is in c's store, and its node in c's context.
    assert g.stores_holding(token) == [C]
