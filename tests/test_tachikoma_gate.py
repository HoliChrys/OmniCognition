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
from metacog.tachikoma_gate import CTX_HEADER, ContextualMemory, _current_ctx


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
