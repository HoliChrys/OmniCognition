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
session to its context and refuses the mismatch (409).

The encoder is SimpleEncoder (hash), the reranker none: deterministic, no
model download, no network — runnable in CI.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from metacog.defaults import SimpleEncoder

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

    def __init__(self, client, root: str):
        self.client = client
        self.root = root

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

    def tag_rows(self, tag: str) -> dict[str, int]:
        """Rows of `tag` in the journal of EVERY context (add_tag lowercases)."""
        return {ctx: self.sql(ctx, "SELECT COUNT(*) FROM tags WHERE tag = ?",
                              (tag.lower(),))[0][0]
                for ctx in self.journals()}

    def stores_holding(self, token: str) -> list[str]:
        """The contexts whose `memory.pkl` holds `token` (pickle keeps str
        as UTF-8: a byte search is the store itself, not an API's opinion)."""
        hits = []
        for ctx in CONTEXTS:
            path = os.path.join(self.root, ctx, "memory.pkl")
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    if token.encode() in fh.read():
                        hits.append(ctx)
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


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """The gated app on a fresh storage root, ACL faked to 'yes', the test
    encoder in place of the ONNX pair (TAC-237's `models()` asks these)."""
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
    root = str(tmp_path / "store")
    outer, _mcp, _inner = build_gated_app(root, str(notes),
                                          authorize_fn=lambda tok, ctx: "tester")
    # The SDK's DNS-rebinding guard wants a loopback Host, as served live.
    with TestClient(outer, base_url="http://127.0.0.1:8788") as client:
        yield Gate(client, root)


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
        row = gate.sql(other, "SELECT returned_node_ids FROM retrievals "
                              "WHERE id = ?", (rid,))
        assert row and own["id"] in json.loads(row[0][0])
        assert fact["id"] not in json.loads(row[0][0])
        logged = {ctx: gate.sql(ctx, "SELECT COUNT(*) FROM retrievals WHERE "
                                     "query_text = ?", (query,))[0][0]
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
