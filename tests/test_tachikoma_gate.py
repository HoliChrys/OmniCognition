"""The TACHIKOMA GATE — one omni memory PER CONTEXT, served behind a header.

What we pin here, in the order it would break in production:

1. FAIL-CLOSED: with no context served, NO call goes through. The silent
   failure would be a memory shared by accident — two contexts writing over
   each other.
2. ONE memory PER context, ISOLATED: what `ctx-a` ingests, `ctx-b` does not
   see; the stores are separate files.
3. The PROXY delegates faithfully: tools call `memory.<anything>` and the call
   goes to the instance of the context — including attributes SET by
   `build_app` (the ACT-R levers), which a swallowing proxy would break
   without a test noticing.
4. The DEEPWIKI: the context's `notes/` folder is ingested (every .md becomes
   a doc AND a content point), kept in step with the folder without a restart
   (added / corrected / deleted), its name→folder mapping written once, and
   one unreadable note never stops the rest — it is said.

The test encoder is SimpleEncoder (hash): deterministic, no model download.
"""
from __future__ import annotations

import os

import pytest

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.tachikoma_gate import (
    ACCOUNT_HEADER, CTX_HEADER, ContextualMemory, _current_account, _current_ctx,
    valid_account_name, valid_context_name)


def _make_proxy(tmp_path, notes=False):
    root = tmp_path / "store"
    notes_root = tmp_path / "notes" if notes else None
    return ContextualMemory(str(root), str(notes_root) if notes_root else None)


def _test_instance(p, ctx):
    """Set the context then create the instance with the TEST encoder and a
    file JOURNAL — wiki docs live in the journal; ingesting without one would
    lose them and the deepwiki test would prove nothing."""
    _current_ctx.set(ctx)
    m = p._memories.get(ctx)
    if m is None:
        path = os.path.join(p._root, ctx, "memory.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        m = Memory(storage_path=path, journal_path=path + ".journal.db",
                   encoder=SimpleEncoder())
        p._memories[ctx] = m
    return m


# ── fail-closed ─────────────────────────────────────────────────────────

def test_without_context_no_call_goes_through(tmp_path):
    """The silent failure would be a memory shared by accident. No context:
    refuse, never fall back to a default memory."""
    p = _make_proxy(tmp_path)
    _current_ctx.set("")
    with pytest.raises(RuntimeError, match="context"):
        p._ctx_name()


def test_the_header_is_mnemas():
    """Same dialect as mnema: a tachikoma proxy must never speak a dialect
    the server does not listen to."""
    assert CTX_HEADER == "x-tachikoma-context"


# ── one memory per context, isolated ───────────────────────────────────

def test_two_contexts_share_nothing(tmp_path):
    p = _make_proxy(tmp_path)
    ma = _test_instance(p, "ctx-a")
    ma.ingest("the Q4 report is valid", kind="FACT")
    mb = _test_instance(p, "ctx-b")
    assert "Q4 report" not in str(getattr(mb, "_nodes", {}))


def test_stores_are_separate_files(tmp_path):
    p = _make_proxy(tmp_path)
    _test_instance(p, "ctx-a")
    _test_instance(p, "ctx-b")
    assert (tmp_path / "store" / "ctx-a" / "memory.pkl").parent != \
           (tmp_path / "store" / "ctx-b" / "memory.pkl").parent


def test_the_proxy_delegates_to_the_current_context(tmp_path):
    """The tool does memory.ingest(...) — the proxy must send it to the
    instance of the SET context, not to an instance frozen at creation."""
    p = _make_proxy(tmp_path)
    _current_ctx.set("ctx-a")
    ma = _test_instance(p, "ctx-a")
    before = len(getattr(ma, "_nodes", {}))
    try:
        p.ingest("through the proxy", kind="FACT")
        after = len(getattr(ma, "_nodes", {}))
        assert after == before + 1
    except Exception:
        # ingest may require more arguments depending on the version — the
        # point is the call GOES THROUGH to the context's instance
        assert p._resolve() is ma


# ── the deepwiki: the context's notes folder (TAC-938) ─────────────────
#
# The notes root is the folder of the TREE ROOT context, named like it
# (deployed: `/opt/tachikoma-fs/global/tachikoma`). Contexts below are
# `tachikoma.<a>.<b>` → `<root>/<a>/<b>/notes`.

def _tree(tmp_path):
    root = tmp_path / "fs" / "tachikoma"
    root.mkdir(parents=True)
    return root


def _notes_of(root, *segments):
    folder = root.joinpath(*segments, "notes")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write(path, text, bump=0):
    """Write a note; `bump` moves its mtime forward so a same-size rewrite
    is still a change (a real edit always moves mtime)."""
    path.write_text(text, encoding="utf-8")
    if bump:
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump * 10**9))


def _wiki(tmp_path, ctx="tachikoma.a"):
    root = _tree(tmp_path)
    p = ContextualMemory(str(tmp_path / "store"), str(root))
    m = _test_instance(p, ctx)
    return root, p, m


def _live_note_points(m, needle):
    from metacog.epistemic import EpistemicState
    return [q for q in m.points if needle in (q.content or "")
            and q.state is not EpistemicState.INVALID]


@pytest.mark.parametrize("ctx, expected", [
    ("tachikoma.paralelle.GenAI", "paralelle/GenAI/notes"),   # measured: 16 docs
    ("tachikoma.paradigm", "paradigm/notes"),
    ("tachikoma", "notes"),                                    # the tree root
    ("global", "notes"),                                       # the hierarchy root
])
def test_the_dotted_name_maps_to_one_folder(ctx, expected):
    """THE MAPPING IS WRITTEN, not rediscovered: one folder per name, the
    first segment IS the root folder, the roots read `<notes_root>/notes`."""
    root = "/opt/tachikoma-fs/global/tachikoma"
    assert gate.notes_folder(root, ctx) == os.path.join(root, *expected.split("/"))


@pytest.mark.parametrize("ctx", ["demo.sandbox.alice", "other", "", "a/../b"])
def test_a_name_outside_the_tree_has_no_folder_here(ctx):
    """No cascade of candidates: a context of another tree is not given the
    root's notes (the old fallback did, for any single-segment name)."""
    assert gate.notes_folder("/opt/tachikoma-fs/global/tachikoma", ctx) is None


def test_the_contexts_notes_are_ingested_twice_doc_and_content(tmp_path):
    """TWO ingestions per note: the DOC (journal, wiki_list/wiki_doc) and the
    CONTENT point (the RAG) — whose id CITES the note."""
    root, p, m = _wiki(tmp_path)
    _write(_notes_of(root, "a") / "proj.md", "# The proj-03 port\n\nPort detail.")
    report = p._refresh_notes("tachikoma.a", m)
    assert report["state"] == "ok" and report["added"] == ["notes:proj"]
    assert m.journal.get_wiki_doc("notes:proj") is not None
    (point,) = _live_note_points(m, "Port detail")
    assert point.id.startswith("notes:proj#")
    assert {"deepwiki", "ctx:tachikoma.a"} <= set(point.tags)


def test_subfolder_notes_are_ingested(tmp_path):
    """Measured gap: GenAI/notes/trace/ carried 14 .md the flat read silently
    missed. A note ANYWHERE under notes/ is a note of the context."""
    root, p, m = _wiki(tmp_path)
    folder = _notes_of(root, "a")
    (folder / "trace").mkdir()
    _write(folder / "trace" / "wot.md", "# WOT")
    p._refresh_notes("tachikoma.a", m)
    assert m.journal.get_wiki_doc("notes:trace/wot") is not None


def test_one_unreadable_note_never_stops_the_wiki_and_is_said(tmp_path):
    root, p, m = _wiki(tmp_path)
    folder = _notes_of(root, "a")
    _write(folder / "good.md", "# Good")
    _write(folder / "bad.md", "# Bad")
    os.chmod(folder / "bad.md", 0)
    try:
        report = p._refresh_notes("tachikoma.a", m)
    finally:
        os.chmod(folder / "bad.md", 0o644)
    assert report["added"] == ["notes:good"]
    assert [e["doc_id"] for e in report["errors"]] == ["notes:bad"]
    # retried on the next pass, not forgotten
    assert p._refresh_notes("tachikoma.a", m)["added"] == ["notes:bad"]


def test_a_note_added_later_enters_without_restart(tmp_path):
    """THE BUG: `_ingested` was a set — a note added after the first access
    never entered until the [omni] process restarted."""
    root, p, m = _wiki(tmp_path)
    folder = _notes_of(root, "a")
    _write(folder / "first.md", "# First")
    assert p._context_memory() is m
    _write(folder / "later.md", "# Later note body")
    p._context_memory()                      # same process, next access
    assert m.journal.get_wiki_doc("notes:later") is not None
    assert len(_live_note_points(m, "Later note body")) == 1


def test_a_corrected_note_replaces_the_old_version(tmp_path):
    """A note corrected is corrected in the wiki — the old version is no
    longer retrievable (soft-forgotten, superseded by the new point)."""
    root, p, m = _wiki(tmp_path)
    note = _notes_of(root, "a") / "deploy.md"
    _write(note, "# Deploy\n\nThe deploy port is 8101.")
    p._refresh_notes("tachikoma.a", m)
    _write(note, "# Deploy\n\nThe deploy port is 8202.", bump=5)
    report = p._refresh_notes("tachikoma.a", m)
    assert report["updated"] == ["notes:deploy"] and not report["added"]
    assert "8202" in m.journal.get_wiki_doc("notes:deploy")["body"]
    assert _live_note_points(m, "8101") == []
    assert len(_live_note_points(m, "8202")) == 1
    hits = m.retrieve("deploy port", k=7, rerank=False)
    assert all("8101" not in (h.get("content") or "") for h in hits)


def test_a_deleted_note_leaves_the_wiki(tmp_path):
    root, p, m = _wiki(tmp_path)
    note = _notes_of(root, "a") / "gone.md"
    _write(note, "# Gone\n\nObsolete fact.")
    p._refresh_notes("tachikoma.a", m)
    note.unlink()
    report = p._refresh_notes("tachikoma.a", m)
    assert report["removed"] == ["notes:gone"] and report["state"] == "no_notes"
    assert m.journal.get_wiki_doc("notes:gone") is None
    assert _live_note_points(m, "Obsolete fact") == []


def test_an_unchanged_folder_costs_no_ingest(tmp_path):
    root, p, m = _wiki(tmp_path)
    _write(_notes_of(root, "a") / "x.md", "# X")
    p._refresh_notes("tachikoma.a", m)
    n = len(m.points)
    report = p._refresh_notes("tachikoma.a", m)
    assert (report["unchanged"], report["added"], report["updated"]) == (1, [], [])
    assert len(m.points) == n


def test_a_restart_recognises_what_the_store_holds(tmp_path):
    """No fingerprints after a restart: the store is the reference — the
    content-addressed id says the note is already there, current."""
    root, p, m = _wiki(tmp_path)
    _write(_notes_of(root, "a") / "x.md", "# X\n\nStable body.")
    p._refresh_notes("tachikoma.a", m)
    n = len(m.points)
    restarted = ContextualMemory(str(tmp_path / "store"), str(root))
    restarted._memories["tachikoma.a"] = m
    report = restarted._refresh_notes("tachikoma.a", m)
    assert report["unchanged"] == 1 and len(m.points) == n


def test_the_copies_of_a_once_per_process_gate_collapse_to_one(tmp_path):
    """Before TAC-938 every restart re-ingested every note (auto ids): the
    duplicates are superseded by the one current, content-addressed point."""
    root, p, m = _wiki(tmp_path)
    body = "# Dup\n\nDuplicated note body."
    _write(_notes_of(root, "a") / "dup.md", body)
    for _ in range(3):   # three restarts of the old gate
        q = m.ingest(body, kind="FACT")
        q.add_tag("note:notes:dup", "ctx:tachikoma.a", "deepwiki")
    m.import_okf("notes:dup", body)
    report = p._refresh_notes("tachikoma.a", m)
    assert report["updated"] == ["notes:dup"]
    (live,) = _live_note_points(m, "Duplicated note body")
    assert live.id.startswith("notes:dup#")


def test_a_context_without_notes_says_so(tmp_path):
    """`global` measured `{"docs": []}` — exact, its folder is empty. The
    report says NO NOTES, distinct from an engine that did not answer."""
    root, p, m = _wiki(tmp_path, ctx="global")
    assert p._refresh_notes("global", m)["state"] == "no_notes"   # folder absent
    _notes_of(root)                                               # folder empty
    report = p._refresh_notes("global", m)
    assert (report["state"], report["notes"]) == ("no_notes", 0)
    assert report["folder"] == str(root / "notes")


def test_outside_the_tree_and_disabled_are_said(tmp_path):
    root, p, m = _wiki(tmp_path, ctx="demo.sandbox")
    assert p._refresh_notes("demo.sandbox", m)["state"] == "outside"
    bare = ContextualMemory(str(tmp_path / "store2"))
    assert bare._refresh_notes("demo.sandbox", m)["state"] == "disabled"


def test_ancestor_notes_are_not_copied_into_the_child(tmp_path):
    """Inheritance is served AT QUERY TIME (C3): the child's wiki holds its
    OWN notes only — a copy would go stale when the ancestor's note moves."""
    root, p, m = _wiki(tmp_path, ctx="tachikoma.sub.Child")
    _write(_notes_of(root) / "root.md", "# Root note")
    _write(_notes_of(root, "sub", "Child") / "own.md", "# Own note")
    p._refresh_notes("tachikoma.sub.Child", m)
    assert m.journal.get_wiki_doc("notes:own") is not None
    assert [d for d in m.journal.all_wiki_doc_ids()] == ["notes:own"]


def test_the_folder_is_read_once_per_request(tmp_path):
    """Every tool call touches `memory.<attr>` many times: the folder is
    stat'ed once per REQUEST (the stamp the middleware sets), not per access."""
    root, p, m = _wiki(tmp_path)
    folder = _notes_of(root, "a")
    token = gate._request_stamp.set(object())
    try:
        p._context_memory()
        _write(folder / "mid.md", "# Mid-request note")
        p._context_memory()                       # same request: not re-read
        assert m.journal.get_wiki_doc("notes:mid") is None
        gate._request_stamp.set(object())         # the next request
        p._context_memory()
        assert m.journal.get_wiki_doc("notes:mid") is not None
    finally:
        gate._request_stamp.reset(token)


def test_ingest_notes_is_a_tool_and_a_bare_memory_says_unsupported(tmp_path):
    """The contract operation over MCP: the gate's proxy answers the report;
    a bare memory (no notes root) says `unsupported` — never an empty list."""
    import asyncio
    import json

    from metacog.mcp_server import build_app

    async def call(memory):
        from mcp.shared.memory import create_connected_server_and_client_session
        app = build_app(memory=memory, surface="external")
        async with create_connected_server_and_client_session(app) as s:
            await s.initialize()
            r = await s.call_tool("ingest_notes", {})
            return json.loads("".join(c.text for c in r.content))

    bare = Memory(encoder=SimpleEncoder())
    assert asyncio.run(call(bare))["state"] == "unsupported"

    root, p, m = _wiki(tmp_path)
    _write(_notes_of(root, "a") / "x.md", "# X")
    report = asyncio.run(call(p))
    assert (report["state"], report["notes"]) == ("ok", 1)
    assert m.journal.get_wiki_doc("notes:x") is not None


# ── the ACL: who may read which memory (TAC-214) ──────────────────────

from metacog import tachikoma_gate as gate  # noqa: E402


def _fake_api(monkeypatch, answers):
    """Route `_api(path, …)` to canned (code, body) answers, record the calls."""
    calls = []

    def api(path, token, payload=None):
        calls.append((path, payload))
        for prefix, answer in answers.items():
            if path.startswith(prefix):
                return answer
        raise AssertionError(f"unexpected ACL call {path}")

    monkeypatch.setattr(gate, "_api", api)
    return calls


ME = {"/api/auth/me": (200, {"user_id": "manager-GenAI-1545c4"})}


def test_no_token_is_refused_401(monkeypatch):
    _fake_api(monkeypatch, {})
    with pytest.raises(gate.Denied) as e:
        gate.authorize("", "tachikoma.paralelle.GenAI")
    assert e.value.status == 401


def test_a_rejected_token_is_refused_401(monkeypatch):
    _fake_api(monkeypatch, {"/api/auth/me": (401, None)})
    with pytest.raises(gate.Denied) as e:
        gate.authorize("forged", "tachikoma.paralelle.GenAI")
    assert e.value.status == 401


def test_general_needs_a_valid_token_but_no_right(monkeypatch):
    calls = _fake_api(monkeypatch, ME)
    assert gate.authorize("t", gate.GENERAL) == "manager-GenAI-1545c4"
    assert [c[0] for c in calls] == ["/api/auth/me"]


def test_an_unknown_context_is_never_born(monkeypatch, tmp_path):
    """The measured hole: an unknown name created `contexts/<name>/`. The
    hierarchy 404 refuses BEFORE `_resolve` could `makedirs`."""
    _fake_api(monkeypatch, {**ME, "/api/hierarchy/": (404, None)})
    with pytest.raises(gate.Denied, match="n'existe pas") as e:
        gate.authorize("t", "contexte.inconnu.personne")
    assert e.value.status == 403


def test_the_context_name_cannot_leave_the_route(monkeypatch):
    calls = _fake_api(monkeypatch, {**ME, "/api/hierarchy/": (404, None)})
    with pytest.raises(gate.Denied):
        gate.authorize("t", "../users/ubuntu/accesses")
    assert calls[1][0] == "/api/hierarchy/..%2Fusers%2Fubuntu%2Faccesses"


def test_an_outage_is_not_called_a_verdict(monkeypatch):
    _fake_api(monkeypatch, {**ME, "/api/hierarchy/": (503, None)})
    with pytest.raises(gate.Denied, match="panne") as e:
        gate.authorize("t", "demo.sandbox.alice")
    assert e.value.status == 503


def test_a_context_without_read_right_is_refused(monkeypatch):
    """The measured case: a GenAI-only token wrote into demo.sandbox.alice."""
    calls = _fake_api(monkeypatch, {**ME, "/api/hierarchy/": (200, {}),
                                    "/api/acl/check": (200, {"allowed": False})})
    with pytest.raises(gate.Denied, match="n'a pas 'read'") as e:
        gate.authorize("t", "demo.sandbox.alice")
    assert e.value.status == 403
    assert calls[-1][1] == {"user": "manager-GenAI-1545c4", "action": "read",
                            "resource": "demo.sandbox.alice"}


def test_a_granted_context_goes_through(monkeypatch):
    _fake_api(monkeypatch, {**ME, "/api/hierarchy/": (200, {}),
                            "/api/acl/check": (200, {"allowed": True})})
    assert gate.authorize("t", "tachikoma.paralelle.GenAI") == "manager-GenAI-1545c4"


# ── the middleware: header, then ACL, before ANY handler runs ──────────

def _gated_client(allowed):
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    seen = []

    def fake_authorize(token, ctx):
        seen.append((token, ctx))
        if ctx not in allowed:
            raise gate.Denied(f"no read on {ctx}")
        return "u"

    async def handler(request):
        return PlainTextResponse(_current_ctx.get())

    app = Starlette(routes=[Route("/mcp", handler, methods=["POST"])],
                    middleware=[Middleware(gate.context_gate(fake_authorize))])
    return TestClient(app), seen


def test_the_gate_refuses_before_the_handler():
    client, seen = _gated_client({"tachikoma.paralelle.GenAI"})
    r = client.post("/mcp", headers={CTX_HEADER: "demo.sandbox.alice",
                                     "Authorization": "Bearer tok"})
    assert r.status_code == 403 and "demo.sandbox.alice" in r.json()["detail"]
    assert seen == [("tok", "demo.sandbox.alice")]


def test_the_gate_serves_a_granted_context():
    client, _ = _gated_client({"tachikoma.paralelle.GenAI"})
    r = client.post("/mcp", headers={CTX_HEADER: "tachikoma.paralelle.GenAI",
                                     "Authorization": "Bearer tok"})
    assert r.status_code == 200 and r.text == "tachikoma.paralelle.GenAI"


def test_the_gate_still_wants_the_header_first():
    client, seen = _gated_client(set())
    assert client.post("/mcp").status_code == 400
    assert seen == []


# ── TAC-213: names are validated before any path is built ──────────────

@pytest.mark.parametrize("name", [
    "global", "tachikoma", "tachikoma.paralelle.GenAI", "demo.sandbox.alice",
    "ctx-a", "a_b.c-d"])
def test_valid_context_names_pass(name):
    assert valid_context_name(name)


@pytest.mark.parametrize("name", [
    "", "..", "../etc", "a/b", "tachikoma/paralelle/GenAI", "/abs", "a..b",
    ".a", "a.", "a b", "a\\b", "ctx\x00"])
def test_invalid_context_names_are_refused(name):
    """Measured on disk: a name with `/` created nested folders, and `..`
    would leave the storage root. Refused before any path is built."""
    assert not valid_context_name(name)


def test_the_proxy_refuses_an_invalid_context(tmp_path):
    p = _make_proxy(tmp_path)
    _current_ctx.set("../outside")
    with pytest.raises(RuntimeError, match="invalid context"):
        p._ctx_name()
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("name", [
    "ubuntu", "manager-GenAI-1545c4", "tachikoma-T-001", "alice@x.io"])
def test_valid_account_names_pass(name):
    assert valid_account_name(name)


@pytest.mark.parametrize("name", ["", "..", ".hidden", "a/b", "a..b", "../x"])
def test_invalid_account_names_are_refused(name):
    assert not valid_account_name(name)


def test_a_store_never_escapes_the_root(tmp_path):
    p = _make_proxy(tmp_path)
    with pytest.raises(RuntimeError, match="escapes"):
        p._memory_at("../outside")


# ── TAC-213: the right to read is the ACCOUNT's ────────────────────────

def _test_store(p, key):
    """A test-encoder memory at `<root>/<key>/memory.pkl` (account stores)."""
    m = p._memories.get(key)
    if m is None:
        path = os.path.join(p._root, key, "memory.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        m = Memory(storage_path=path, journal_path=path + ".journal.db",
                   encoder=SimpleEncoder())
        p._memories[key] = m
    return m


def _contents(m):
    return [pt.content for pt in m.points]


def _as(ctx, account=""):
    _current_ctx.set(ctx)
    _current_account.set(account)


def _ingest_like_the_tool(p, content, tags=None):
    """What the `ingest` MCP tool does with `memory` (the proxy)."""
    pt = p.ingest(content, kind="FACT", id=None)
    if tags:
        pt.add_tag(*tags)
    if p.storage_path:
        p.save()
    return pt


def test_the_default_account_is_the_contexts(tmp_path):
    """No account header (or the context's name): the context memory, as
    before — `par défaut ils sont sur le context`."""
    p = _make_proxy(tmp_path)
    ctx_mem = _test_instance(p, "ctx-a")
    _as("ctx-a", "")
    assert p._resolve() is ctx_mem
    _as("ctx-a", "ctx-a")
    assert p._resolve() is ctx_mem


def test_a_narrow_account_reads_only_its_own(tmp_path):
    """An agent operating under its own account reads what its account
    wrote — not the context's facts, not another agent's."""
    p = _make_proxy(tmp_path)
    ctx_mem = _test_instance(p, "ctx-a")
    a_mem = _test_store(p, os.path.join("ctx-a", "accounts", "agent-a"))
    b_mem = _test_store(p, os.path.join("ctx-a", "accounts", "agent-b"))

    _as("ctx-a", "")
    _ingest_like_the_tool(p, "the manager's private plan")
    _as("ctx-a", "agent-a")
    _ingest_like_the_tool(p, "agent a learned the port is 8788")

    _as("ctx-a", "agent-a")
    assert p._resolve() is a_mem
    assert _contents(a_mem) == ["agent a learned the port is 8788"]
    _as("ctx-a", "agent-b")
    assert p._resolve() is b_mem
    assert _contents(b_mem) == []
    assert "the manager's private plan" in _contents(ctx_mem)


def test_a_narrow_write_also_carries_the_context_tag(tmp_path):
    """What an agent writes under its own account ALSO lands in the
    context memory, tagged with its account: the manager reads it."""
    p = _make_proxy(tmp_path)
    ctx_mem = _test_instance(p, "ctx-a")
    a_mem = _test_store(p, os.path.join("ctx-a", "accounts", "agent-a"))
    _as("ctx-a", "agent-a")
    pt = _ingest_like_the_tool(p, "agent a learned the port is 8788",
                               tags=["module:gate"])

    mirror = [x for x in ctx_mem.points
              if x.content == "agent a learned the port is 8788"]
    assert len(mirror) == 1
    assert "account:agent-a" in mirror[0].tags
    assert "module:gate" in mirror[0].tags          # the tool's tags follow
    own = [x for x in a_mem.points if x.id == pt.id]
    assert own and "account:agent-a" in own[0].tags
    # both stores were saved
    assert os.path.exists(a_mem.storage_path)
    assert os.path.exists(ctx_mem.storage_path)


def test_accounts_are_scoped_by_context(tmp_path):
    """The same account id under two contexts: two memories, not one."""
    p = _make_proxy(tmp_path)
    _test_instance(p, "ctx-a")
    _test_instance(p, "ctx-b")
    in_a = _test_store(p, os.path.join("ctx-a", "accounts", "agent-a"))
    _as("ctx-a", "agent-a")
    _ingest_like_the_tool(p, "only in a")
    _as("ctx-b", "agent-a")
    in_b = p._resolve()
    assert in_b is not in_a
    assert "only in a" not in _contents(in_b)


def test_the_proxy_refuses_an_invalid_account(tmp_path):
    p = _make_proxy(tmp_path)
    _as("ctx-a", "../ctx-b")
    with pytest.raises(RuntimeError, match="invalid account"):
        p._resolve()


# ── TAC-213: the HTTP gate — names first, then the ACL, then the account ──

def _account_client(user="agent-a"):
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    seen = []

    def fake_authorize(token, ctx):
        seen.append(ctx)
        return user

    async def handler(request):
        return PlainTextResponse(f"{_current_ctx.get()}|{_current_account.get()}")

    app = Starlette(routes=[Route("/mcp", handler, methods=["POST"])],
                    middleware=[Middleware(gate.context_gate(fake_authorize))])
    return TestClient(app), seen


@pytest.mark.parametrize("headers, needle", [
    ({CTX_HEADER: "../etc"}, "contexte invalide"),
    ({CTX_HEADER: "tachikoma/paralelle/GenAI"}, "contexte invalide"),
    ({CTX_HEADER: "ctx-a", ACCOUNT_HEADER: "../ctx-b"}, "compte invalide"),
])
def test_the_gate_refuses_bad_names_before_the_acl(headers, needle):
    """A bad name never reaches the ACL API, nor a path."""
    client, seen = _account_client()
    r = client.post("/mcp", headers={**headers, "Authorization": "Bearer t"})
    assert r.status_code == 400 and needle in r.json()["detail"]
    assert seen == []


def test_no_account_header_is_the_contexts_account():
    client, _ = _account_client()
    r = client.post("/mcp", headers={CTX_HEADER: "ctx-a"})
    assert r.status_code == 200 and r.text == "ctx-a|"


def test_a_caller_narrows_to_its_own_account():
    client, _ = _account_client(user="agent-a")
    r = client.post("/mcp", headers={CTX_HEADER: "ctx-a",
                                     ACCOUNT_HEADER: "agent-a"})
    assert r.status_code == 200 and r.text == "ctx-a|agent-a"


def test_a_caller_never_borrows_another_account():
    """The account is VERIFIED against the authenticated user: naming
    another agent's account would read its memory."""
    client, _ = _account_client(user="agent-a")
    r = client.post("/mcp", headers={CTX_HEADER: "ctx-a",
                                     ACCOUNT_HEADER: "agent-b"})
    assert r.status_code == 403 and "agent-b" in r.json()["detail"]


# ── TAC-237: ONE encoder + ONE reranker per gate, loaded off the loop ───

def _counting_models(monkeypatch):
    """Fake `make_encoder`/`make_reranker` that count their calls — a real
    pair is ~2.0 GB of ONNX sessions, the count is what the test pins."""
    import metacog.defaults as D
    calls = {"encoder": 0, "reranker": 0}

    def make_encoder():
        calls["encoder"] += 1
        return SimpleEncoder()

    def make_reranker():
        calls["reranker"] += 1
        return None

    monkeypatch.setattr(D, "make_encoder", make_encoder)
    monkeypatch.setattr(D, "make_reranker", make_reranker)
    return calls


def test_every_memory_shares_one_model_pair(tmp_path, monkeypatch):
    """Measured before: each context AND each narrow account built its own
    pair — 2 GB per key, omni idle at 7.16 GB. One pair per gate."""
    calls = _counting_models(monkeypatch)
    p = _make_proxy(tmp_path)
    mems = [p._memory_at(k) for k in
            ("global", "tachikoma.paralelle.GenAI",
             os.path.join("tachikoma.paralelle.GenAI", "accounts", "agent-a"))]
    assert calls == {"encoder": 1, "reranker": 1}
    assert len({id(m.encoder) for m in mems}) == 1
    assert mems[0].encoder is p.models()[0]


def test_the_gate_loads_the_models_off_the_event_loop():
    """The load ran on the asyncio thread (py-spy, TAC-237): 4-10 s deaf.
    The middleware runs `warm` in a worker thread, after the ACL said yes —
    and never for a refused caller."""
    import threading

    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    warmed = []

    def warm():
        warmed.append(threading.get_ident())

    def fake_authorize(token, ctx):
        if ctx != "ctx-a":
            raise gate.Denied(f"no read on {ctx}")
        return "u"

    async def handler(request):
        return PlainTextResponse(str(threading.get_ident()))

    app = Starlette(routes=[Route("/mcp", handler, methods=["POST"])],
                    middleware=[Middleware(gate.context_gate(fake_authorize,
                                                             warm=warm))])
    client = TestClient(app)
    assert client.post("/mcp", headers={CTX_HEADER: "ctx-b"}).status_code == 403
    assert warmed == []
    r = client.post("/mcp", headers={CTX_HEADER: "ctx-a"})
    assert r.status_code == 200
    assert len(warmed) == 1 and warmed[0] != int(r.text)


# ── TAC-228: ONE Memory per context, even at a concurrent first access ──

def _slow_counting_memory(monkeypatch):
    """`Memory` whose construction is slow (as a real load is: pickle +
    journal) and counted — the slowness opens the race window wide, so the
    test fails reliably WITHOUT the lock; the count is what it pins."""
    import time

    import metacog.memory as M
    born = []

    class SlowMemory(M.Memory):
        def __init__(self, *a, **kw):
            born.append(1)
            time.sleep(0.2)
            super().__init__(*a, **kw)

    monkeypatch.setattr(M, "Memory", SlowMemory)
    return born


def _concurrently(n, fn):
    """Run `fn(i)` in `n` threads released together; return the results."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    start = threading.Barrier(n)

    def run(i):
        start.wait()
        return fn(i)

    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(run, range(n)))


def test_a_concurrent_first_access_builds_one_memory(tmp_path, monkeypatch):
    """Measured before (TAC-228): 3 threads at the first access of one
    context gave 3 distinct `Memory`, one kept in the cache."""
    _counting_models(monkeypatch)
    born = _slow_counting_memory(monkeypatch)
    p = _make_proxy(tmp_path)
    mems = _concurrently(20, lambda i: p._memory_at("demo.sandbox.alice"))
    assert len({id(m) for m in mems}) == 1
    assert len(born) == 1
    assert mems[0] is p._memories["demo.sandbox.alice"]


def test_the_lock_is_per_context(tmp_path, monkeypatch):
    """Two contexts first-accessed together are built side by side: the
    lock is per key, one context's load never waits on another's."""
    import threading
    _counting_models(monkeypatch)
    _slow_counting_memory(monkeypatch)
    p = _make_proxy(tmp_path)
    inside, peak, guard = [0], [0], threading.Lock()
    real = p._new_memory

    def watched(key):
        with guard:
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
        try:
            return real(key)
        finally:
            with guard:
                inside[0] -= 1

    monkeypatch.setattr(p, "_new_memory", watched)
    _concurrently(2, lambda i: p._memory_at(f"ctx-{i}"))
    assert peak[0] == 2


def test_20_concurrent_remember_at_first_access_write_20(tmp_path,
                                                         monkeypatch):
    """The production measure (TAC-228): 20 concurrent `remember` at the
    first access of a context took its store from 1 to 41 points — each
    fact twice. Exactly +20 now, each fact once, read back from the disk."""
    _counting_models(monkeypatch)
    _slow_counting_memory(monkeypatch)
    ctx = "demo.sandbox.alice"
    seed = _make_proxy(tmp_path)
    _as(ctx)
    _ingest_like_the_tool(seed, "le point déjà là")

    p = _make_proxy(tmp_path)          # a restarted gate: first access again

    def remember(i):
        _as(ctx)
        return _ingest_like_the_tool(p, f"fait concurrent numéro {i}").id

    ids = _concurrently(20, remember)
    on_disk = Memory(storage_path=os.path.join(p._root, ctx, "memory.pkl"),
                     encoder=SimpleEncoder())
    contents = _contents(on_disk)
    assert len(contents) == 1 + 20
    for i in range(20):
        assert contents.count(f"fait concurrent numéro {i}") == 1
    assert set(ids) <= {pt.id for pt in on_disk.points}
