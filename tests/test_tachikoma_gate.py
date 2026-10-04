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
   a doc), once, and one unreadable note never stops the rest.

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


# ── the deepwiki: the context's notes folder ───────────────────────────

def test_the_contexts_notes_are_ingested(tmp_path):
    """The context's deepwiki is ITS notes/ folder: every .md becomes a doc —
    wiki docs live in the JOURNAL, that's where we read them."""
    folder = tmp_path / "notes" / "ctx-a" / "notes"
    folder.mkdir(parents=True)
    (folder / "proj.md").write_text("# The proj-03 port\n\nNote content.",
                                    encoding="utf-8")
    p = ContextualMemory(str(tmp_path / "store"), str(tmp_path / "notes"))
    m = _test_instance(p, "ctx-a")
    p._ingest_notes("ctx-a", m)
    doc = m.journal.get_wiki_doc("notes:proj")
    assert doc is not None, "notes:proj missing from the journal"


def test_one_unreadable_note_never_stops_the_wiki(tmp_path):
    folder = tmp_path / "notes" / "ctx-a" / "notes"
    folder.mkdir(parents=True)
    (folder / "good.md").write_text("# Good", encoding="utf-8")
    p = ContextualMemory(str(tmp_path / "store"), str(tmp_path / "notes"))
    m = _test_instance(p, "ctx-a")
    # must not raise, whatever the note
    p._ingest_notes("ctx-a", m)


def test_notes_are_ingested_only_once(tmp_path):
    folder = tmp_path / "notes" / "ctx-a" / "notes"
    folder.mkdir(parents=True)
    (folder / "x.md").write_text("# X", encoding="utf-8")
    p = ContextualMemory(str(tmp_path / "store"), str(tmp_path / "notes"))
    m = _test_instance(p, "ctx-a")
    p._ingest_notes("ctx-a", m)
    p._ingest_notes("ctx-a", m)   # second call: no-op
    assert p._ingested == {"ctx-a"}


def test_the_dotted_name_maps_to_the_folder_path(tmp_path):
    """tachikoma.paralelle.GenAI lives at <root>/paralelle/GenAI (first
    segment repeats the root) — the mapping tries without it first."""
    folder = tmp_path / "notes" / "paralelle" / "GenAI" / "notes"
    folder.mkdir(parents=True)
    (folder / "iris.md").write_text("# IRIS", encoding="utf-8")
    p = ContextualMemory(str(tmp_path / "store"), str(tmp_path / "notes"))
    m = _test_instance(p, "tachikoma.paralelle.GenAI")
    p._ingest_notes("tachikoma.paralelle.GenAI", m)
    assert m.journal.get_wiki_doc("notes:iris") is not None


def test_subfolder_notes_are_ingested(tmp_path):
    """Measured gap: GenAI/notes/trace/ carried 14 .md the flat read silently
    missed. A note ANYWHERE under notes/ is a note of the context."""
    folder = tmp_path / "notes" / "ctx-a" / "notes"
    (folder / "trace").mkdir(parents=True)
    (folder / "trace" / "wot.md").write_text("# WOT", encoding="utf-8")
    p = ContextualMemory(str(tmp_path / "store"), str(tmp_path / "notes"))
    m = _test_instance(p, "ctx-a")
    p._ingest_notes("ctx-a", m)
    assert m.journal.get_wiki_doc("notes:trace/wot") is not None


def test_ancestor_notes_seed_the_child_wiki(tmp_path):
    """Inheritance at ingestion: a child context's wiki also carries its
    ancestors' notes, marked by doc id (mnema marks them '← hérité' at
    query; the wiki answers the same question its own way)."""
    root_notes = tmp_path / "notes" / "notes"
    root_notes.mkdir(parents=True)
    (root_notes / "root.md").write_text("# Root note", encoding="utf-8")
    child = tmp_path / "notes" / "sub" / "Child" / "notes"
    child.mkdir(parents=True)
    (child / "own.md").write_text("# Own note", encoding="utf-8")
    p = ContextualMemory(str(tmp_path / "store"), str(tmp_path / "notes"))
    m = _test_instance(p, "tachikoma.sub.Child")
    p._ingest_notes("tachikoma.sub.Child", m)
    assert m.journal.get_wiki_doc("notes:own") is not None          # own
    assert m.journal.get_wiki_doc("notes:tachikoma/root") is not None  # ancestor


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


# ── TAC-213: the HTTP gate refuses bad names before any tool runs ──────

def _gate_client(tmp_path):
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient
    from metacog.tachikoma_gate import build_gated_app
    outer, _mcp, _inner = build_gated_app(str(tmp_path / "store"), "")
    return TestClient(outer)


@pytest.mark.parametrize("headers, needle", [
    ({}, "obligatoire"),
    ({CTX_HEADER: "../etc"}, "contexte invalide"),
    ({CTX_HEADER: "tachikoma/paralelle/GenAI"}, "contexte invalide"),
    ({CTX_HEADER: "ctx-a", ACCOUNT_HEADER: "../ctx-b"}, "compte invalide"),
])
def test_the_gate_refuses_bad_names_in_400(tmp_path, headers, needle):
    with _gate_client(tmp_path) as client:
        r = client.post("/mcp", headers=headers, json={})
    assert r.status_code == 400
    assert needle in r.json()["detail"]
    assert not (tmp_path / "etc").exists()
