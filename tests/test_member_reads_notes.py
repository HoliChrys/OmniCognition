"""TAC-345 — decision A of TAC-344: a lobby member's recall reads its account
AND the notes of each context of its chain, never another account's facts.

A member (`x-tachikoma-account: <member>`, a `lobby` token) is served, on
EVERY stage of tachikoma's `recall_inherited` (ctx → ancestors → `global`, one
gate call per stage):

1. its own account memory (`<stage>/accounts/<member>/memory.pkl`), as before;
2. PLUS the notes of the stage's context — the points the notes chain ingested.

It never reads another account's facts (their `account:<x>` mirrors live in
the same context store), nor the manager's. The NOTE MARK cannot be forged by
a write: a `remember` tagged `deepwiki` / `note:notes:…`, a source `notes:…`,
an explicit id `notes:x#…` — none passes. Only what `_refresh_notes` read in
the folder does.

The encoder is SimpleEncoder (hash), the reranker none: deterministic, offline.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os

import pytest

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.tachikoma_gate import (
    ContextualMemory, _current_account, _current_ctx)

CTX = "tachikoma.paralelle.GenAI"
CHAIN = [CTX, "tachikoma.paralelle", "tachikoma", "global"]
MANAGER = "manager-GenAI-1545c4"
MEMBER = "tachikoma-GenAI-archiviste"
OTHER = "tachikoma-GenAI-autre"

#: One note per stage, each with a token that exists nowhere else.
NOTES = {
    CTX: "# GenAI runbook\n\nThe GenAI relay frequency is saffron-5519.",
    "tachikoma.paralelle": "# Paralelle runbook\n\nThe paralelle relay is umber-3307.",
    "tachikoma": "# Root runbook\n\nThe tachikoma relay is cobalt-2290.",
    "global": "# Global runbook\n\nThe global relay is jade-8846.",
}
NOTE_TOKENS = {CTX: "saffron-5519", "tachikoma.paralelle": "umber-3307",
               "tachikoma": "cobalt-2290", "global": "jade-8846"}

MANAGER_FACT = "The manager's private relay plan is ochre-4471."
OTHER_FACT = "The other agent's private relay is teal-6120."
FORGED_TAGS = "The forged relay note says crimson-9001."
FORGED_ID = "The forged relay note by id says amber-1764."
MEMBER_FACT = "The archiviste learned the relay port is 8788 indigo-5050."
QUERY = "what is the relay frequency"


def _notes_root(tmp_path):
    """The deployed layout (`notes_folder`): `global` reads the notes of the
    notes_root's PARENT, `tachikoma` its own, a descendant drops the head."""
    root = tmp_path / "fs" / "tachikoma"
    folders = {"global": tmp_path / "fs" / "notes",
               "tachikoma": root / "notes",
               "tachikoma.paralelle": root / "paralelle" / "notes",
               CTX: root / "paralelle" / "GenAI" / "notes"}
    for ctx, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "runbook.md").write_text(NOTES[ctx], encoding="utf-8")
    return root


# ── the proxy: what a member's `retrieve` reads ─────────────────────────

def _store(p, key):
    """A test-encoder memory at `<root>/<key>/memory.pkl`, with a journal."""
    m = p._memories.get(key)
    if m is None:
        path = os.path.join(p._root, key, "memory.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        m = Memory(storage_path=path, journal_path=path + ".journal.db",
                   encoder=SimpleEncoder())
        p._memories[key] = m
    return m


def _as(ctx, account=""):
    _current_ctx.set(ctx)
    _current_account.set(account)


@pytest.fixture(autouse=True)
def _no_leaked_caller():
    """The contextvars outlive a test (same thread): leave none behind, or a
    later file's proxy would resolve this file's member account."""
    ctx, account = _current_ctx.set(""), _current_account.set("")
    yield
    _current_ctx.reset(ctx)
    _current_account.reset(account)


def _write(p, content, tags=(), id=None):
    """What the `ingest` MCP tool does with `memory` (the proxy)."""
    pt = p.ingest(content, kind="FACT", id=id)
    if tags:
        pt.add_tag(*tags)
    if p.storage_path:
        p.save()
    return pt


def _recall(p, query=QUERY, k=7):
    return [h["content"] for h in p.retrieve(query, k=k)]


@pytest.fixture
def proxy(tmp_path):
    p = ContextualMemory(str(tmp_path / "store"), str(_notes_root(tmp_path)))
    for ctx in CHAIN:
        _store(p, ctx)
        for account in (MEMBER, OTHER):
            _store(p, os.path.join(ctx, "accounts", account))
    # The writes a member must never read, in the context store of its ctx.
    _as(CTX, "")
    _write(p, MANAGER_FACT)
    _as(CTX, OTHER)
    _write(p, OTHER_FACT)
    return p


def _hits_on_every_stage(p, account):
    """What `recall_inherited` collects: one call per stage of the chain."""
    out = {}
    for stage in CHAIN:
        _as(stage, account)
        out[stage] = _recall(p)
    return out


def test_a_member_reads_the_notes_of_its_context_and_of_every_ancestor(proxy):
    """(a) Each stage serves the note of THAT stage — no copy: the ancestor's
    note comes from the ancestor's store, asked at call time."""
    _as(CTX, MEMBER)
    _write(proxy, MEMBER_FACT)
    hits = _hits_on_every_stage(proxy, MEMBER)
    for stage in CHAIN:
        assert NOTES[stage] in hits[stage], stage
        others = [NOTES[s] for s in CHAIN if s != stage]
        assert not set(others) & set(hits[stage]), stage
    # Its own account is still read, as before.
    assert MEMBER_FACT in hits[CTX]
    # C3.3: nothing was copied into the member's account store.
    own = proxy._memories[os.path.join(CTX, "accounts", MEMBER)]
    assert [pt.content for pt in own.points] == [MEMBER_FACT]


def test_a_member_never_reads_another_account_nor_the_manager(proxy):
    """(b) The context store holds the manager's fact and OTHER's mirror
    (`account:<other>`): neither reaches the member — while the same store's
    note does (the negative doubled by its positive)."""
    ctx_mem = proxy._memories[CTX]
    held = [pt.content for pt in ctx_mem.points]
    assert MANAGER_FACT in held and OTHER_FACT in held   # they ARE there
    for query in (QUERY, "manager private relay plan ochre-4471",
                  "other agent private relay teal-6120"):
        _as(CTX, MEMBER)
        hits = _recall(proxy, query)
        assert NOTES[CTX] in hits
        assert MANAGER_FACT not in hits and OTHER_FACT not in hits


def test_a_write_dressed_as_a_note_never_passes(proxy):
    """(c) The note mark is the gate's, not the writer's. A member's write
    tagged like a note (deepwiki, note:notes:…, ctx:…, src:notes:…) is
    mirrored in the context store WITH those tags; the manager writes a point
    whose id has a note's shape and even BORROWS the real note's id. None is
    served to another member."""
    forged_tags = ["deepwiki", "note:notes:runbook", f"ctx:{CTX}",
                   "src:notes:runbook"]
    _as(CTX, OTHER)
    _write(proxy, FORGED_TAGS, tags=forged_tags, id="notes:runbook#forged")
    _as(CTX, "")
    sha = hashlib.sha256(FORGED_ID.encode()).hexdigest()[:12]
    _write(proxy, FORGED_ID, tags=forged_tags, id=f"notes:forged#{sha}")
    (real_id,) = [pid for pid, _ in proxy._note_marks[CTX].values()]
    _write(proxy, "A point that borrowed the note's id, khaki-3141.",
           tags=forged_tags, id=real_id)

    ctx_mem = proxy._memories[CTX]
    mirror = [pt for pt in ctx_mem.points if pt.content == FORGED_TAGS]
    assert mirror and "deepwiki" in mirror[0].tags        # the dress IS there
    for query in (QUERY, "forged relay note crimson-9001",
                  "forged relay note amber-1764", "borrowed note id khaki-3141"):
        _as(CTX, MEMBER)
        hits = _recall(proxy, query)
        assert NOTES[CTX] in hits                          # the real note is
        assert FORGED_TAGS not in hits and FORGED_ID not in hits
        assert not any("khaki-3141" in h for h in hits)


def test_the_manager_path_is_unchanged(proxy):
    """No account: the context memory, notes and every fact, as before."""
    _as(CTX, "")
    hits = _recall(proxy)
    assert NOTES[CTX] in hits and MANAGER_FACT in hits and OTHER_FACT in hits


def test_a_member_without_notes_reads_its_account_only(tmp_path):
    """No notes folder: nothing marked — the member's recall is its account,
    exactly as before TAC-345, and the context store is not searched."""
    p = ContextualMemory(str(tmp_path / "store"), str(tmp_path / "none"))
    _store(p, CTX)
    _store(p, os.path.join(CTX, "accounts", MEMBER))
    _as(CTX, "")
    _write(p, MANAGER_FACT)
    _as(CTX, MEMBER)
    _write(p, MEMBER_FACT)
    assert _recall(p) == [MEMBER_FACT]


def test_the_notes_cost_is_counted_in_the_stage(proxy):
    """E1: the notes pool is part of what the stage cost — the pools add up."""
    _as(CTX, MEMBER)
    _write(proxy, MEMBER_FACT)
    spent: dict = {}
    proxy.retrieve(QUERY, k=5, cost=spent)
    assert spent["pool_size"] == 2                        # own fact + the note


def test_a_note_that_stands_out_is_not_a_gap(proxy):
    """The gap verdict is taken over what the member reads: an account too
    small to have a background (never abstains) plus a note — and a store
    of the manager's facts that would decide otherwise is not consulted."""
    _as(CTX, MEMBER)
    for i in range(4):
        _write(proxy, f"member filler fact number {i} about lunch")
    assert proxy.abstains("GenAI relay frequency saffron-5519") is False


def test_a_corrected_note_is_read_in_its_new_version(proxy):
    """The mark follows the folder: a corrected note serves the new body,
    never the superseded one."""
    folder = os.path.join(os.path.dirname(proxy._notes_root), "tachikoma",
                          "paralelle", "GenAI", "notes", "runbook.md")
    new = "# GenAI runbook\n\nThe GenAI relay frequency is now sienna-7002."
    with open(folder, "w", encoding="utf-8") as fh:
        fh.write(new)
    st = os.stat(folder)
    os.utime(folder, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    proxy._notes_checked.clear()
    _as(CTX, MEMBER)
    hits = _recall(proxy)
    assert new in hits and NOTES[CTX] not in hits


# ── end to end: the real gated app, the chain asked like tachikoma ──────

def _token(user, scopes):
    """A token in tachikoma's wire format (`serialize_token`: base64 JSON)."""
    return base64.b64encode(json.dumps(
        {"user_id": user, "scopes": scopes, "signature": "sig"}).encode()).decode()


def _member(user):
    return _token(user, [f"ctx:{CTX}", "agent", "lobby"])


class _Http:
    def __init__(self, client):
        self.client = client

    def call(self, token, ctx, tool, args, recall_for=None):
        h = {"Accept": "application/json, text/event-stream",
             "Content-Type": "application/json",
             "Authorization": f"Bearer {token}", "x-tachikoma-context": ctx}
        if recall_for:
            h["x-tachikoma-recall-for"] = recall_for
        r = self.client.post("/mcp", headers=h, json={
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "tac-345", "version": "0"}}})
        assert r.status_code == 200, r.text
        h["mcp-session-id"] = r.headers["mcp-session-id"]
        self.client.post("/mcp", headers=h, json={
            "jsonrpc": "2.0", "method": "notifications/initialized"})
        r = self.client.post("/mcp", headers=h, json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": args}})
        assert r.status_code == 200, r.text
        frames = [ln[5:].strip() for ln in r.text.splitlines()
                  if ln.startswith("data:")]
        result = json.loads(frames[0] if frames else r.text)["result"]
        assert not result.get("isError"), result
        sc = result.get("structuredContent")
        if sc is not None:
            return sc.get("result", sc)
        values = [json.loads(b["text"]) for b in result.get("content", [])
                  if b.get("type") == "text"]
        return values[0] if len(values) == 1 else values

    def recall_inherited(self, token, query):
        """Every stage of the chain, as `memory_engines.recall_inherited`
        asks it: one session per stage, `recall-for` on the ancestors."""
        out = {}
        for stage in CHAIN:
            hits = self.call(token, stage, "retrieve", {"query": query, "k": 7},
                             recall_for=None if stage == CTX else CTX)
            hits = hits if isinstance(hits, list) else [hits]
            out[stage] = [h for h in hits if isinstance(h, dict)]
        return out


@pytest.fixture
def http(tmp_path, monkeypatch):
    import metacog.defaults as defaults
    from metacog import tachikoma_gate as gate
    monkeypatch.setattr(defaults, "make_encoder", lambda: SimpleEncoder())
    monkeypatch.setattr(defaults, "make_reranker", lambda: None)

    def fake_authorize(token, ctx):
        # WHO is the token's user; every one of them may read the chain.
        return json.loads(base64.b64decode(token))["user_id"]

    from starlette.testclient import TestClient
    outer, _mcp, _inner = gate.build_gated_app(
        str(tmp_path / "store"), str(_notes_root(tmp_path)),
        authorize_fn=fake_authorize)
    with TestClient(outer, base_url="http://127.0.0.1:8788") as client:
        yield _Http(client)


def test_over_http_a_member_recalls_its_account_and_the_chains_notes(http):
    manager = _token(MANAGER, [f"ctx:{CTX}", "agent"])
    http.call(manager, CTX, "ingest", {"content": MANAGER_FACT, "kind": "FACT"})
    http.call(_member(OTHER), CTX, "remember", {"content": OTHER_FACT})
    http.call(_member(OTHER), CTX, "remember", {
        "content": FORGED_TAGS, "source": "notes:runbook",
        "tags": ["deepwiki", "note:notes:runbook", f"ctx:{CTX}"]})
    http.call(_member(MEMBER), CTX, "remember", {"content": MEMBER_FACT})

    stages = http.recall_inherited(_member(MEMBER), QUERY)
    served = " ".join(h.get("content", "") for hits in stages.values()
                      for h in hits)
    for stage in CHAIN:                    # (a) each stage serves ITS note
        contents = [h.get("content") for h in stages[stage]]
        assert NOTES[stage] in contents, (stage, contents)
    assert MEMBER_FACT in served           # …and the member's own account
    for token in ("ochre-4471", "teal-6120", "crimson-9001"):   # (b), (c)
        assert token not in served

    # POSITIVE double: the manager's recall still reads all of its context.
    mine = http.call(manager, CTX, "retrieve", {"query": QUERY, "k": 7})
    contents = [h.get("content") for h in mine if isinstance(h, dict)]
    assert MANAGER_FACT in contents and NOTES[CTX] in contents
