"""
Persistent journal wiring : Memory(journal_path=...) attaches a SQLite journal
that OUTLIVES the process, so the SQL triggers (co-retrieval / lateral / Chasles
/ tag index / decay history) survive a restart. Production
(Memory(storage_path=..., journal_path="auto")) gets one automatically.
"""

from __future__ import annotations

import os
import tempfile

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory


def test_auto_journal_path_derives_from_storage_path():
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "mem.pkl")
        m = Memory(encoder=SimpleEncoder(), storage_path=store, journal_path="auto")
        assert m.journal is not None
        assert m.journal.path == f"{store}.journal.db"
        assert os.path.exists(m.journal.path)


def test_auto_is_noop_without_storage_path():
    m = Memory(encoder=SimpleEncoder(), journal_path="auto")   # nothing to derive
    assert m.journal is None


def test_default_has_no_journal():
    m = Memory(encoder=SimpleEncoder())
    assert m.journal is None


def test_explicit_journal_path_opens_file():
    with tempfile.TemporaryDirectory() as d:
        jp = os.path.join(d, "j.db")
        m = Memory(encoder=SimpleEncoder(), journal_path=jp)
        assert m.journal is not None and m.journal.path == jp


def test_journal_survives_restart():
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "mem.pkl")
        # -- session 1 : ingest, index tags + a travelled path, then persist ----
        m1 = Memory(encoder=SimpleEncoder(), storage_path=store, journal_path="auto")
        a = m1.ingest("swollen finger", kind="FACT", id="A")
        b = m1.ingest("chronic tiredness", kind="FACT", id="B")
        a.tags.append("health:condition:macrodactyly")
        b.tags.append("health:condition:fatigue")
        m1.reindex_tags()
        m1.log_path(["A", "B"], t=1.0)
        m1.log_path(["A", "B"], t=1.0)                 # travelled twice
        m1.save()
        m1.journal.conn.close()                       # simulate process exit

        # -- session 2 : reopen the SAME store -> journal state is still there --
        m2 = Memory(encoder=SimpleEncoder(), storage_path=store, journal_path="auto")
        assert m2.journal is not None
        # tag index persisted (hierarchical ancestor query still works)
        assert {p.id for p in m2.tag_scoped("health")} == {"A", "B"}
        assert "health:condition:fatigue" in m2.tag_glossary_sql()
        # path-frequency persisted (Chasles trigger sees the earlier traversals)
        rows = m2.journal.frequent_paths(min_len=2, min_freq=2)
        assert [(r["signature"], r["freq"]) for r in rows] == [("A>B", 2)]


def test_snapshot_shares_journal_by_reference():
    with tempfile.TemporaryDirectory() as d:
        jp = os.path.join(d, "j.db")
        m = Memory(encoder=SimpleEncoder(), journal_path=jp)
        snap = m.snapshot()
        assert snap.journal is m.journal              # shared, not reopened
        assert snap.storage_path is None              # snapshot can't overwrite
