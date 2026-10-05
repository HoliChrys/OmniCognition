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
    # Another tree lives in its own folder under the root, every segment
    # kept (TAC-934's layout): chosen by the NAME, whatever exists.
    ("demo.sandbox.alice", "demo/sandbox/alice/notes"),
    ("other", "other/notes"),
])
def test_the_dotted_name_maps_to_one_folder(ctx, expected):
    """THE MAPPING IS WRITTEN, not rediscovered: one folder per name, the
    first segment IS the root folder, the roots read `<notes_root>/notes`."""
    root = "/opt/tachikoma-fs/global/tachikoma"
    assert gate.notes_folder(root, ctx) == os.path.join(root, *expected.split("/"))


@pytest.mark.parametrize("ctx", ["", "a/../b", "../x", "a..b"])
def test_an_invalid_name_has_no_folder(ctx):
    """No path is ever built from a name that is not a context name."""
    assert gate.notes_folder("/opt/tachikoma-fs/global/tachikoma", ctx) is None


def test_the_rule_never_depends_on_which_folder_exists(tmp_path):
    """The old cascade tried `<root>/sandbox/notes` then `<root>/demo/sandbox/
    notes` and took the first that EXISTED: one stray folder moved a context's
    wiki. The rule reads the name only."""
    root = _tree(tmp_path)
    (root / "sandbox" / "notes").mkdir(parents=True)          # a decoy
    assert gate.notes_folder(str(root), "demo.sandbox") == \
        str(root / "demo" / "sandbox" / "notes")


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


def test_a_note_added_later_enters_without_restart(tmp_path, monkeypatch):
    """THE BUG: `_ingested` was a set — a note added after the first access
    never entered until the [omni] process restarted."""
    root, p, m = _wiki(tmp_path)
    folder = _notes_of(root, "a")
    _write(folder / "first.md", "# First")
    monkeypatch.setattr(gate, "_requests_seen", 1)          # a request
    assert p._context_memory() is m
    _write(folder / "later.md", "# Later note body")
    monkeypatch.setattr(gate, "_requests_seen", 2)          # the next one
    p._context_memory()                      # same process, next request
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


def _eagain_on(monkeypatch, fn_name, target):
    """`os.<fn_name>(target)` raises EAGAIN — the FUSE timing out under load
    (measured on holistix-baremetal, TAC-243); every other path is real."""
    import errno
    real = getattr(os, fn_name)

    def flaky(path, *args, **kwargs):
        if os.fspath(path) == str(target):
            raise OSError(errno.EAGAIN, "resource-fuse op timed out", str(target))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(os, fn_name, flaky)


@pytest.mark.parametrize("fn_name, where", [
    ("stat", ""),          # the notes/ folder itself does not answer
    ("scandir", ""),       # it answers stat, not its listing
    ("scandir", "trace"),  # a sub-folder does not answer: 14 notes measured there
])
def test_an_unreadable_folder_is_never_an_absence(tmp_path, monkeypatch,
                                                  fn_name, where):
    """MEASURED (TAC-243): the FUSE rendered EAGAIN, `os.path.isdir` read it
    as "absent" and the pass removed all 16 docs of GenAI, then saved. A read
    error says `error`: nothing removed, nothing forgotten, nothing saved —
    and the next pass, the folder readable again, finds everything in place."""
    root, p, m = _wiki(tmp_path)
    folder = _notes_of(root, "a")
    (folder / "trace").mkdir()
    _write(folder / "top.md", "# Top\n\nTop note body.")
    _write(folder / "trace" / "wot.md", "# WOT\n\nTrace note body.")
    assert len(p._refresh_notes("tachikoma.a", m)["added"]) == 2
    saved = os.stat(m.storage_path).st_mtime_ns
    saves = []
    monkeypatch.setattr(m, "save", lambda *a, **k: saves.append(1))
    with monkeypatch.context() as mp:
        _eagain_on(mp, fn_name, folder / where if where else folder)
        report = p._refresh_notes("tachikoma.a", m)
    assert report["state"] == "error"
    assert "timed out" in report["errors"][0]["error"]
    assert (report["removed"], report["added"], report["updated"]) == ([], [], [])
    assert saves == [] and os.stat(m.storage_path).st_mtime_ns == saved
    assert {"notes:top", "notes:trace/wot"} <= set(m.journal.all_wiki_doc_ids())
    assert len(_live_note_points(m, "Top note body")) == 1
    assert len(_live_note_points(m, "Trace note body")) == 1
    again = p._refresh_notes("tachikoma.a", m)        # the FUSE answers again
    assert (again["state"], again["removed"], again["unchanged"]) == ("ok", [], 2)


def _linked_wiki(tmp_path, ctx="tachikoma.a"):
    """The DEPLOYED layout: `<notes_root>/<a>/notes` is a LINK (the FUSE's)
    to the real folder on the local disk."""
    root, p, m = _wiki(tmp_path, ctx)
    disk = tmp_path / "disk" / "notes"
    disk.mkdir(parents=True)
    (root / "a").mkdir()
    (root / "a" / "notes").symlink_to(disk, target_is_directory=True)
    _write(disk / "one.md", "# One\n\nFirst note body.")
    _write(disk / "two.md", "# Two\n\nSecond note body.")
    assert len(p._refresh_notes(ctx, m)["added"]) == 2
    return root, p, m, disk


def _fuse_enoent_on(monkeypatch, target):
    """`os.stat(target)` raises ENOENT — the FUSE whose backend cannot
    resolve the folder (measured on holistix-baremetal, TAC-255)."""
    import errno
    real = os.stat

    def lying(path, *args, **kwargs):
        if os.fspath(path) == str(target):
            raise FileNotFoundError(errno.ENOENT, "No such file or directory",
                                    str(target))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(os, "stat", lying)


def _assert_wiki_intact(m, report):
    assert report["state"] == "error"
    assert (report["removed"], report["added"], report["updated"]) == ([], [], [])
    assert {"notes:one", "notes:two"} <= set(m.journal.all_wiki_doc_ids())
    assert len(_live_note_points(m, "First note body")) == 1
    assert len(_live_note_points(m, "Second note body")) == 1


def test_a_fuse_enoent_on_a_folder_present_on_disk_is_never_an_absence(
        tmp_path, monkeypatch):
    """MEASURED (TAC-255): `tachikoma-api` restarting, the FUSE answered
    ENOENT for GenAI's notes/ — which existed — and the pass removed its 17
    docs (`-17`), re-added a minute later (`+17`). The disk says the folder
    is there: nothing removed, nothing saved, the next pass finds all."""
    root, p, m, _disk = _linked_wiki(tmp_path)
    saves = []
    monkeypatch.setattr(m, "save", lambda *a, **k: saves.append(1))
    with monkeypatch.context() as mp:
        _fuse_enoent_on(mp, root / "a" / "notes")
        report = p._refresh_notes("tachikoma.a", m)
    _assert_wiki_intact(m, report)
    assert "present on disk" in report["errors"][0]["error"] and saves == []
    again = p._refresh_notes("tachikoma.a", m)
    assert (again["state"], again["removed"], again["unchanged"]) == ("ok", [], 2)


def test_after_a_restart_an_enoent_is_unconfirmed_and_removes_nothing(
        tmp_path, monkeypatch):
    """A gate that just started has read no folder: the FUSE's ENOENT has
    nothing to be checked against. The context HAS notes in its store —
    they stay. Recreating the folder EMPTY is the positive observation that
    removes them."""
    root, _p, m, disk = _linked_wiki(tmp_path)
    restarted = ContextualMemory(str(tmp_path / "store"), str(root))
    with monkeypatch.context() as mp:
        _fuse_enoent_on(mp, root / "a" / "notes")
        report = restarted._refresh_notes("tachikoma.a", m)
    _assert_wiki_intact(m, report)
    assert "unconfirmed" in report["errors"][0]["error"]
    for note in disk.iterdir():          # really deleted, gate restarted
        note.unlink()
    disk.rmdir()
    _assert_wiki_intact(m, restarted._refresh_notes("tachikoma.a", m))
    disk.mkdir()                         # recreated empty: read, confirmed
    report = restarted._refresh_notes("tachikoma.a", m)
    assert (report["state"], sorted(report["removed"])) == (
        "no_notes", ["notes:one", "notes:two"])
    assert _live_note_points(m, "First note body") == []


@pytest.mark.parametrize("linked", [True, False])
def test_a_folder_really_deleted_leaves_the_wiki(tmp_path, linked):
    """The absence CONFIRMED — by the disk behind the link, or by a plain
    folder that is its own real path — still removes the notes."""
    if linked:
        _root, p, m, folder = _linked_wiki(tmp_path)
    else:
        root, p, m = _wiki(tmp_path)
        folder = _notes_of(root, "a")
        _write(folder / "one.md", "# One\n\nFirst note body.")
        _write(folder / "two.md", "# Two\n\nSecond note body.")
        assert len(p._refresh_notes("tachikoma.a", m)["added"]) == 2
    for note in folder.iterdir():
        note.unlink()
    folder.rmdir()
    report = p._refresh_notes("tachikoma.a", m)
    assert (report["state"], sorted(report["removed"])) == (
        "no_notes", ["notes:one", "notes:two"])
    assert _live_note_points(m, "Second note body") == []


def test_a_note_that_comes_back_is_live_again_and_saved(tmp_path):
    """The -16 then +16 cycle measured on GenAI (TAC-243 §3): a note removed
    then back with the SAME body gets a fresh live point (suffixed id — never
    the forgotten node reused), and that point reaches the store on disk."""
    root, p, m = _wiki(tmp_path)
    note = _notes_of(root, "a") / "port.md"
    body = "# Port\n\nThe probe port is 48217."
    _write(note, body)
    p._refresh_notes("tachikoma.a", m)
    (first,) = _live_note_points(m, "48217")
    note.unlink()
    assert p._refresh_notes("tachikoma.a", m)["removed"] == ["notes:port"]
    assert _live_note_points(m, "48217") == []
    _write(note, body)
    assert p._refresh_notes("tachikoma.a", m)["added"] == ["notes:port"]
    (back,) = _live_note_points(m, "48217")
    assert back.id == f"{first.id}.2"
    reloaded = Memory(storage_path=m.storage_path, encoder=SimpleEncoder())
    assert [q.id for q in _live_note_points(reloaded, "48217")] == [back.id]


def test_another_tree_and_disabled_are_said(tmp_path):
    root, p, m = _wiki(tmp_path, ctx="demo.sandbox")
    report = p._refresh_notes("demo.sandbox", m)
    assert report["state"] == "no_notes"
    assert report["folder"] == str(root / "demo" / "sandbox" / "notes")
    _write(_notes_of(root, "demo", "sandbox") / "d.md", "# Demo note")
    assert p._refresh_notes("demo.sandbox", m)["added"] == ["notes:d"]
    bare = ContextualMemory(str(tmp_path / "store2"))
    assert bare._refresh_notes("demo.sandbox", m)["state"] == "disabled"


def test_the_folder_is_read_once_per_request(tmp_path):
    """Every tool call touches `memory.<attr>` many times: the folder is
    stat'ed once per REQUEST (the count the middleware bumps), not per access."""
    root, p, m = _wiki(tmp_path, ctx="tachikoma.a")
    folder = _notes_of(root, "a")
    before = gate._requests_seen
    gate._requests_seen = before + 1              # a request
    try:
        p._context_memory()
        _write(folder / "mid.md", "# Mid-request note")
        p._context_memory()                       # same request: not re-read
        assert m.journal.get_wiki_doc("notes:mid") is None
        gate._requests_seen += 1                  # the next request
        p._context_memory()
        assert m.journal.get_wiki_doc("notes:mid") is not None
    finally:
        gate._requests_seen = before


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


def test_ancestor_notes_are_never_copied_into_the_child(tmp_path):
    """Rule C3 (TAC-936): inheritance is served AT QUERY TIME, never copied at
    ingestion — a copy goes stale the day the ancestor's note is corrected.
    The child's memory holds its OWN notes, and nothing of its ancestors'."""
    root, p, m = _wiki(tmp_path, ctx="tachikoma.sub.Child")
    _write(_notes_of(root) / "root.md", "# Root note")
    _write(_notes_of(root, "sub", "Child") / "own.md", "# Own note")
    p._refresh_notes("tachikoma.sub.Child", m)
    assert m.journal.get_wiki_doc("notes:own") is not None
    assert m.journal.all_wiki_doc_ids() == ["notes:own"]
    assert [p_.content for p_ in m.points] == ["# Own note"]
    assert "ctx:tachikoma.sub.child" in m.points[0].tags   # add_tag lowercases
    # …and the ancestor's note lives in the ANCESTOR's memory, where a
    # query-time recall reads it.
    ancestor = _test_instance(p, "tachikoma")
    p._refresh_notes("tachikoma", ancestor)
    assert [p_.content for p_ in ancestor.points] == ["# Root note"]


def test_copies_left_by_an_older_gate_are_soft_forgotten(tmp_path):
    """The older gate copied ancestor notes into the child (`ctx:<ancestor>`
    tag). A recall would serve them as LOCAL memories, stale and mislabeled:
    the first access soft-forgets them (reversible, never deleted)."""
    root, p, m = _wiki(tmp_path, ctx="tachikoma.sub.Child")
    _write(_notes_of(root, "sub", "Child") / "own.md", "# Own note")
    copy = m.ingest("# Root note, copied", kind="FACT")
    copy.add_tag("note:notes:tachikoma/root", "ctx:tachikoma", "deepwiki")
    fact = m.ingest("a fact written by an agent", kind="FACT")
    p._refresh_notes("tachikoma.sub.Child", m)
    (own,) = _live_note_points(m, "# Own note")
    assert "invalidated" in copy.tags
    assert "invalidated" not in own.tags
    assert "invalidated" not in fact.tags


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


def test_general_is_no_longer_a_common_notebook(monkeypatch):
    """Rule C3 (TAC-936): `general` was the one name readable by any valid
    token, outside the chain. It is an ordinary name now — no hierarchy
    entry, refused like any unknown context; what all must read goes in
    `global`."""
    assert not hasattr(gate, "GENERAL")
    calls = _fake_api(monkeypatch, {**ME, "/api/hierarchy/": (404, None)})
    with pytest.raises(gate.Denied, match="n'existe pas"):
        gate.authorize("t", "general")
    assert [c[0] for c in calls] == ["/api/auth/me", "/api/hierarchy/general"]


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


# ── TAC-274: a lobby member's account is its TOKEN's, not its header's ──

_CTX = "tachikoma.paralelle.GenAI"
_MANAGER = "manager-GenAI-1545c4"
_MEMBER = "tachikoma-GenAI-archiviste"


def _token(user, scopes):
    """A token in tachikoma's wire format (`serialize_token`: base64 JSON)."""
    import base64
    import json
    return base64.b64encode(json.dumps(
        {"user_id": user, "scopes": scopes, "signature": "sig"}).encode()).decode()


def _member_token(user=_MEMBER):
    """What `mint-lobby-member-token` hands a member (TAC-241)."""
    return _token(user, [f"ctx:{_CTX}", "agent", "lobby"])


def _call_as(user, token, account=None):
    client, _ = _account_client(user=user)
    headers = {CTX_HEADER: _CTX, "Authorization": f"Bearer {token}"}
    if account is not None:
        headers[ACCOUNT_HEADER] = account
    return client.post("/mcp", headers=headers)


def test_a_member_without_header_is_served_its_own_account():
    """Measured before (TAC-262): served the context memory, it read the
    manager's `fact_79014e81`. The token decides, not the header."""
    r = _call_as(_MEMBER, _member_token())
    assert r.status_code == 200 and r.text == f"{_CTX}|{_MEMBER}"


def test_a_member_naming_its_own_account_is_served_it():
    r = _call_as(_MEMBER, _member_token(), account=_MEMBER)
    assert r.status_code == 200 and r.text == f"{_CTX}|{_MEMBER}"


@pytest.mark.parametrize("target", [_CTX, _MANAGER, "tachikoma-GenAI-autre"])
def test_a_member_never_reaches_the_managers_memory(target):
    """The context's own name used to mean « the context's account » for
    anyone; for a member it is another account than its token's: 403."""
    r = _call_as(_MEMBER, _member_token(), account=target)
    assert r.status_code == 403 and target in r.json()["detail"]


def test_the_manager_never_reaches_a_members_memory():
    manager_token = _token(_MANAGER, [f"ctx:{_CTX}", "agent"])
    r = _call_as(_MANAGER, manager_token, account=_MEMBER)
    assert r.status_code == 403 and _MEMBER in r.json()["detail"]


def test_the_managers_token_stays_on_the_context_memory():
    """No `lobby` scope: the context's account, header absent or its own."""
    manager_token = _token(_MANAGER, [f"ctx:{_CTX}", "agent"])
    assert _call_as(_MANAGER, manager_token).text == f"{_CTX}|"
    # The context's name as account: the proxy reads it as the context's own.
    assert _call_as(_MANAGER, manager_token, account=_CTX).text == f"{_CTX}|{_CTX}"


def test_a_member_whose_id_cannot_name_a_folder_is_refused():
    r = _call_as("../x", _member_token("../x"))
    assert r.status_code == 403


@pytest.mark.parametrize("token", ["", "t", "not base64 !", "bnVsbA==",
                                   "eyJ1c2VyX2lkIjogIngifQ=="])
def test_a_token_that_carries_no_scope_reads_as_none(token):
    """Undecodable, `null`, no `scopes` key: no scope, never an exception."""
    assert gate.token_scopes(token) == []


def test_the_account_tag_keeps_the_case_of_the_id(tmp_path):
    """Measured before (TAC-262): `account:tachikoma-genai-archiviste` in the
    tags, `accounts/tachikoma-GenAI-archiviste` on disk. Same name now."""
    p = _make_proxy(tmp_path)
    ctx_mem = _test_instance(p, _CTX)
    own_mem = _test_store(p, os.path.join(_CTX, "accounts", _MEMBER))
    _as(_CTX, _MEMBER)
    pt = _ingest_like_the_tool(p, "the archivist filed the minutes",
                               tags=["module:gate"])
    tag = f"account:{_MEMBER}"
    own = [x for x in own_mem.points if x.id == pt.id][0]
    mirror = [x for x in ctx_mem.points
              if x.content == "the archivist filed the minutes"][0]
    assert tag in own.tags and tag in mirror.tags
    assert tag.lower() not in mirror.tags
    # …the folder's very name
    assert own_mem.storage_path == os.path.join(
        p._root, _CTX, "accounts", _MEMBER, "memory.pkl")


def test_reads_find_the_exact_case_tag_and_the_older_lowercased_one(tmp_path):
    """No migration: a tag an older gate lowercased is still found, by the
    exact id and by its lowercased form, on both read paths (the in-memory
    `filter_list` and the journal's SQL `tag_scoped`)."""
    p = _make_proxy(tmp_path)
    ctx_mem = _test_instance(p, _CTX)
    _test_store(p, os.path.join(_CTX, "accounts", _MEMBER))
    _as(_CTX, _MEMBER)
    _ingest_like_the_tool(p, "written by the new gate")
    mirror_new = [x for x in ctx_mem.points
                  if x.content == "written by the new gate"][0]
    _as(_CTX, "")
    old = _ingest_like_the_tool(p, "written by an older gate")
    old.add_tag(f"account:{_MEMBER}")            # lowercased, as before
    assert f"account:{_MEMBER}".lower() in old.tags
    ctx_mem.reindex_tags()
    want = {mirror_new.id, old.id}
    for needle in (f"account:{_MEMBER}", f"account:{_MEMBER}".lower()):
        assert {x.id for x in ctx_mem.filter_list(tags=[needle])} == want
        assert {x.id for x in ctx_mem.tag_scoped(needle)} == want


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


# ── TAC-272: a recall climbs its chain READ-ONLY under `recall-for` ────
#
# The real gated app over MCP streamable HTTP. The ACL is faked by TOKEN:
# `admin` reads everything, `child` reads ONLY `iso-alpha.child` — like
# `manager-GenAI-1545c4`, which has `read` on GenAI and none on its parents.
# `top` reads ONLY `iso-alpha`: rights on a stage, none on its children.

from metacog.tachikoma_gate import RECALL_FOR_HEADER  # noqa: E402

ASKED = "iso-alpha.child"
ANCESTOR_FACT = "The iso-alpha ancestor beacon is ochre-4471."


@pytest.fixture
def chain_gate(tmp_path, monkeypatch):
    import metacog.defaults as defaults
    monkeypatch.setattr(defaults, "make_encoder", lambda: SimpleEncoder())
    monkeypatch.setattr(defaults, "make_reranker", lambda: None)

    seen = []

    def fake_authorize(token, ctx):
        seen.append((token, ctx))
        if (token == "admin" or (token == "child" and ctx == ASKED)
                or (token == "top" and ctx == "iso-alpha")):
            return token
        raise gate.Denied(f"{token!r} n'a pas 'read' sur le contexte {ctx!r}")

    from starlette.testclient import TestClient
    outer, _mcp, _inner = gate.build_gated_app(
        str(tmp_path / "store"), "", authorize_fn=fake_authorize)
    with TestClient(outer, base_url="http://127.0.0.1:8788") as client:
        yield client, seen


def _hdrs(token, ctx, recall_for=None):
    h = {"Accept": "application/json, text/event-stream",
         "Content-Type": "application/json",
         "Authorization": f"Bearer {token}", CTX_HEADER: ctx}
    if recall_for:
        h[RECALL_FOR_HEADER] = recall_for
    return h


def _open(client, h):
    r = client.post("/mcp", headers=h, json={
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "tac-272", "version": "0"}}})
    if r.status_code != 200:
        return r
    h = {**h, "mcp-session-id": r.headers["mcp-session-id"]}
    r2 = client.post("/mcp", headers=h, json={
        "jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r2.status_code in (200, 202), r2.text
    return h


def _call(client, h, tool, args):
    return client.post("/mcp", headers=h, json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": args}})


def _ok(r):
    import json
    assert r.status_code == 200, r.text
    frames = [ln[5:].strip() for ln in r.text.splitlines() if ln.startswith("data:")]
    payload = json.loads(frames[0] if frames else r.text)
    assert "error" not in payload and not payload["result"].get("isError"), payload
    return payload["result"]


def _plant(client, ctx, content):
    h = _open(client, _hdrs("admin", ctx))
    return _ok(_call(client, h, "ingest", {"content": content, "kind": "FACT"}))


@pytest.mark.parametrize("ancestor", ["iso-alpha", "global"])
def test_a_child_reader_recalls_its_ancestors_under_recall_for(chain_gate, ancestor):
    client, seen = chain_gate
    _plant(client, ancestor, ANCESTOR_FACT)
    # WITHOUT the header: the per-stage check of before — refused (the bug).
    r = _open(client, _hdrs("child", ancestor))
    assert r.status_code == 403 and ancestor in r.json()["detail"]
    # WITH it: authorized on the ASKED context, the ancestor's fact served.
    seen.clear()
    h = _open(client, _hdrs("child", ancestor, recall_for=ASKED))
    for tool in ("retrieve", "recall"):
        result = _ok(_call(client, h, tool, {"query": "ancestor beacon ochre", "k": 5}))
        assert "ochre-4471" in str(result["content"])
    assert all(ctx == ASKED for _tok, ctx in seen) and seen


@pytest.mark.parametrize("stage", [
    "iso-alpha.child",          # the asked context itself: no header needed
    "iso-alpha.child.grand",    # a child of the asked context
    "iso-alpha.sibling",        # a sibling
    "iso-beta",                 # off the chain
    "iso-alph",                 # a string prefix that is not a segment prefix
])
def test_recall_for_serves_only_strict_ancestors(chain_gate, stage):
    client, seen = chain_gate
    r = client.post("/mcp", headers=_hdrs("child", stage, recall_for=ASKED),
                    json={"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
    assert r.status_code == 403 and "ancêtre" in r.json()["detail"]
    assert seen == []            # refused before any ACL call


@pytest.mark.parametrize("tool, args", [
    ("ingest", {"content": "smuggled into the ancestor", "kind": "FACT"}),
    ("remember", {"content": "smuggled into the ancestor"}),
    ("forget", {"node_id": "x", "reason": "r"}),
    ("wiki_list", {}),           # a read, but not a recall/search tool
])
def test_nothing_but_recall_passes_under_recall_for(chain_gate, tool, args):
    client, _ = chain_gate
    h = _open(client, _hdrs("child", "iso-alpha", recall_for=ASKED))
    r = _call(client, h, tool, args)
    assert r.status_code == 403 and "lecture seule" in r.json()["detail"]
    assert repr(tool) in r.json()["detail"]
    # The admin's view of the ancestor: nothing was written there.
    ha = _open(client, _hdrs("admin", "iso-alpha"))
    stats = _ok(_call(client, ha, "stats", {}))
    assert "smuggled" not in str(stats)


def test_a_batch_smuggling_a_write_is_refused(chain_gate):
    client, _ = chain_gate
    h = _open(client, _hdrs("child", "iso-alpha", recall_for=ASKED))
    r = client.post("/mcp", headers=h, json=[
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "retrieve", "arguments": {"query": "q"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "ingest", "arguments": {"content": "x"}}}])
    assert r.status_code == 403 and "'ingest'" in r.json()["detail"]


def test_recall_for_is_post_only_and_named(chain_gate):
    client, _ = chain_gate
    r = client.get("/mcp", headers=_hdrs("child", "iso-alpha", recall_for=ASKED))
    assert r.status_code == 403 and "GET" in r.json()["detail"]
    r = client.post("/mcp", headers=_hdrs("child", "iso-alpha", recall_for="../x"),
                    json={"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
    assert r.status_code == 400


def test_the_account_check_still_applies_under_recall_for(chain_gate):
    """TAC-213 per stage: a recall stage narrows to the caller's own account,
    never to another's."""
    client, _ = chain_gate
    h = _hdrs("child", "iso-alpha", recall_for=ASKED)
    r = _open(client, {**h, ACCOUNT_HEADER: "someone-else"})
    assert r.status_code == 403 and "someone-else" in r.json()["detail"]
    h = _open(client, {**h, ACCOUNT_HEADER: "child"})
    _ok(_call(client, h, "retrieve", {"query": "anything", "k": 3}))


def test_a_recall_for_yes_never_serves_the_session_without_the_header(chain_gate):
    """TAC-299 × TAC-272: the session keeps the yes WITH the context it was
    given on. A session opened under `recall-for` (yes on the asked child,
    read-only) that drops the header is asked again on the stage itself —
    never served a write on the ancestor on the child's read."""
    client, seen = chain_gate
    h = _open(client, _hdrs("child", "iso-alpha", recall_for=ASKED))
    seen.clear()
    bare = {k: v for k, v in h.items() if k != RECALL_FOR_HEADER}
    r = _call(client, bare, "ingest", {"content": "smuggled past the cache", "kind": "FACT"})
    assert r.status_code == 403
    assert seen == [("child", "iso-alpha")]
    ha = _open(client, _hdrs("admin", "iso-alpha"))
    assert "smuggled" not in str(_ok(_call(client, ha, "stats", {})))


def test_a_recall_for_yes_never_serves_another_asked_context(chain_gate):
    """TAC-299 × TAC-272: same session, same stage, same bearer — but
    `recall-for` now names a context the bearer cannot read. Asked again on
    THAT context, never vouched for by the yes given on ASKED."""
    client, seen = chain_gate
    h = _open(client, _hdrs("child", "iso-alpha", recall_for=ASKED))
    seen.clear()
    other = {**h, RECALL_FOR_HEADER: "iso-alpha.other"}
    r = _call(client, other, "retrieve", {"query": "ancestor beacon", "k": 3})
    assert r.status_code == 403
    assert seen == [("child", "iso-alpha.other")]


def test_a_stage_yes_never_serves_a_recall_for_request(chain_gate):
    """The other direction: a session opened on the stage itself (yes on
    `iso-alpha`) that adds `recall-for` is asked about the ASKED context —
    the ACL is never skipped for a context it was not asked about. The
    refusal is not kept: the bare session still runs on its own yes."""
    client, seen = chain_gate
    h = _open(client, _hdrs("top", "iso-alpha"))
    seen.clear()
    r = _call(client, {**h, RECALL_FOR_HEADER: ASKED}, "retrieve",
              {"query": "ancestor beacon", "k": 3})
    assert r.status_code == 403
    assert seen == [("top", ASKED)]
    _ok(_call(client, h, "retrieve", {"query": "ancestor beacon", "k": 3}))
    assert seen == [("top", ASKED)]
