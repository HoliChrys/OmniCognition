"""JC2-6 (TAC2-1212) — the served gate chooses its usage journal per store.

Unset / `sqlite` : the `<store>.journal.db` file, unchanged. `proxy` : the
proxy journal for the stores of `OMNI_JOURNAL_CONTEXTS` (every store when
unset), and the SQLite file for any store that cannot get one.
"""

import pytest

import metacog.journal_proxy as jp
import metacog.tachikoma_gate as gate


@pytest.fixture
def opened(monkeypatch):
    got = []

    def fake_open(ctx, account="", *, state_path=":memory:", **kw):
        got.append((ctx, account, state_path))
        return ("proxy-journal", ctx, account)

    monkeypatch.setattr(jp, "open_proxy_journal", fake_open)
    monkeypatch.delenv(gate.JOURNAL_ENV, raising=False)
    monkeypatch.delenv(gate.JOURNAL_CONTEXTS_ENV, raising=False)
    return got


def test_unset_and_sqlite_keep_the_file_journal(opened, monkeypatch):
    assert gate.usage_journal_for("global", "/r/global/memory.pkl") is None
    monkeypatch.setenv(gate.JOURNAL_ENV, "sqlite")
    assert gate.usage_journal_for("global", "/r/global/memory.pkl") is None
    assert opened == []


def test_proxy_for_every_store_by_default(opened, monkeypatch):
    monkeypatch.setenv(gate.JOURNAL_ENV, "proxy")
    j = gate.usage_journal_for("tachikoma.paradigm", "/r/tachikoma.paradigm/memory.pkl")
    assert j == ("proxy-journal", "tachikoma.paradigm", "")
    gate.usage_journal_for("tachikoma.paradigm/accounts/ext-ubuntu-appr",
                           "/r/tachikoma.paradigm/accounts/ext-ubuntu-appr/memory.pkl")
    assert opened[-1] == ("tachikoma.paradigm", "ext-ubuntu-appr",
                          "/r/tachikoma.paradigm/accounts/ext-ubuntu-appr/memory.pkl.journal.db")


def test_proxy_only_for_the_listed_stores(opened, monkeypatch):
    monkeypatch.setenv(gate.JOURNAL_ENV, "proxy")
    monkeypatch.setenv(gate.JOURNAL_CONTEXTS_ENV,
                       "tachikoma.paradigm, tachikoma.paralelle.GenAI")
    assert gate.usage_journal_for("global", "/r/global/memory.pkl") is None
    assert gate.usage_journal_for("global/accounts/a", "/r/ga/memory.pkl") is None
    assert gate.usage_journal_for("tachikoma.paralelle.GenAI", "/r/g/memory.pkl")
    assert [o[0] for o in opened] == ["tachikoma.paralelle.GenAI"]


def test_a_listed_context_covers_its_accounts(opened, monkeypatch):
    """TAC-2492 : an account's rows go to its context's proxy, so listing the context
    is enough — an exact match left `ctx/accounts/*` on their `.journal.db`."""
    monkeypatch.setenv(gate.JOURNAL_ENV, "proxy")
    monkeypatch.setenv(gate.JOURNAL_CONTEXTS_ENV, "tachikoma.paradigm,global/accounts/b")
    assert gate.usage_journal_for("tachikoma.paradigm/accounts/a", "/r/pa/memory.pkl")
    assert gate.usage_journal_for("global/accounts/b", "/r/gb/memory.pkl")
    assert gate.usage_journal_for("global", "/r/global/memory.pkl") is None
    assert gate.usage_journal_for("tachikoma.paradigm.child", "/r/c/memory.pkl") is None
    assert [o[:2] for o in opened] == [("tachikoma.paradigm", "a"), ("global", "b")]


def test_a_proxy_journal_that_cannot_open_falls_back(opened, monkeypatch):
    def boom(*a, **k):
        raise ImportError("No module named 'ray'")

    monkeypatch.setattr(jp, "open_proxy_journal", boom)
    monkeypatch.setenv(gate.JOURNAL_ENV, "proxy")
    assert gate.usage_journal_for("global", "/r/global/memory.pkl") is None


def test_an_unknown_mode_refuses_to_start(monkeypatch):
    monkeypatch.setenv(gate.JOURNAL_ENV, "clickhouse")
    with pytest.raises(SystemExit, match="OMNI_JOURNAL"):
        gate.require_journal_mode()
    monkeypatch.setenv(gate.JOURNAL_ENV, "proxy")
    assert gate.require_journal_mode() == "proxy"
    monkeypatch.delenv(gate.JOURNAL_ENV)
    assert gate.require_journal_mode() == "sqlite"


# ── OMNI_STATE (D10, Paperclip TAC-2465) : where the mutable state lives ───────

import metacog.journal_state_neo4j as js
from metacog.journal import Journal


@pytest.fixture
def states(monkeypatch, opened):
    made = []

    class FakeState(Journal):
        def __init__(self, client):
            super().__init__(":memory:")
            made.append((client.context_path, client.account))

    monkeypatch.setattr(js, "Neo4jStateJournal", FakeState)
    monkeypatch.delenv(gate.STATE_ENV, raising=False)
    monkeypatch.delenv(gate.STATE_CONTEXTS_ENV, raising=False)
    return made


def test_unset_state_keeps_everything_in_the_file(states, tmp_path):
    assert gate.journal_for("global", str(tmp_path / "memory.pkl")) is None
    assert states == []


def test_neo4j_state_with_the_sqlite_usage_journal_is_a_split(states, monkeypatch,
                                                              tmp_path):
    monkeypatch.setenv(gate.STATE_ENV, "neo4j")
    j = gate.journal_for("global", str(tmp_path / "memory.pkl"))
    assert isinstance(j, js.SplitJournal) and states == [("global", "")]
    assert j.usage.path == str(tmp_path / "memory.pkl.journal.db")
    j.usage.close()


def test_neo4j_state_rides_in_the_proxy_journal(states, monkeypatch):
    seen = {}

    def fake_open(ctx, account="", *, state_path=":memory:", state=None, **kw):
        seen["state"] = state
        return ("proxy-journal", ctx, account)

    monkeypatch.setattr(jp, "open_proxy_journal", fake_open)
    monkeypatch.setenv(gate.JOURNAL_ENV, "proxy")
    monkeypatch.setenv(gate.STATE_ENV, "neo4j")
    j = gate.journal_for("tachikoma.paradigm/accounts/a", "/r/p/accounts/a/memory.pkl")
    assert j == ("proxy-journal", "tachikoma.paradigm", "a")
    assert isinstance(seen["state"], js.Neo4jStateJournal)
    assert states == [("tachikoma.paradigm", "a")]


def test_neo4j_state_only_for_the_listed_stores(states, monkeypatch, tmp_path):
    monkeypatch.setenv(gate.STATE_ENV, "neo4j")
    monkeypatch.setenv(gate.STATE_CONTEXTS_ENV, "tachikoma.paralelle.sdk")
    assert gate.journal_for("global", str(tmp_path / "memory.pkl")) is None
    assert states == []


def test_an_unknown_state_mode_refuses_to_start(monkeypatch):
    monkeypatch.setenv(gate.STATE_ENV, "clickhouse")
    with pytest.raises(SystemExit, match="OMNI_STATE"):
        gate.require_state_mode()
