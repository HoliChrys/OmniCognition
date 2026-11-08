"""
JC2-3 (TAC2-1202) — the usage journal has two backends, and they answer alike.

`Journal` (SQLite) is the reference. `ClickHouseJournal` must return exactly
what it returns : the same script of writes runs on both, then every read of
the protocol is compared. The ClickHouse half needs a THROWAWAY server
(`OMNI_TEST_CLICKHOUSE_URL=http://user:password@host:port`) and is skipped,
with that reason, without one. The structural and failure-safe tests below run
everywhere (a fake transport).
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit

import pytest

import metacog.journal as journal_mod
from metacog.journal import APPEND_ONLY_TABLES, JournalProtocol, SQLiteJournal
from metacog.journal_clickhouse import (STATE_METHODS, ClickHouseJournal,
                                        HttpClickHouseTransport,
                                        iter_journal_methods)

_URL = os.environ.get("OMNI_TEST_CLICKHOUSE_URL", "")
needs_clickhouse = pytest.mark.skipif(
    not _URL, reason="OMNI_TEST_CLICKHOUSE_URL unset (a throwaway ClickHouse)")

# Ids that a naive quoting would break : quotes, backslash, tab, non-ASCII,
# and one that is a prefix of another (substring tests must not confuse them).
ODD = ["plain", "it's", 'say "hi"', "back\\slash", "tab\there", "été", "A", "A1"]


class _DeadTransport:
    """A ClickHouse that is down : every call raises."""

    def __init__(self) -> None:
        self.calls = 0

    def _boom(self, *a, **k):
        self.calls += 1
        raise ConnectionError("clickhouse down")

    command = insert = select = _boom


class _RecordingTransport:
    """Accepts writes, answers reads with nothing, remembers the batches."""

    def __init__(self) -> None:
        self.batches: list = []

    def command(self, sql):
        pass

    def insert(self, table, rows):
        self.batches.append((table, list(rows)))

    def select(self, sql, params=None):
        return []


def _transport() -> HttpClickHouseTransport:
    u = urlsplit(_URL)
    return HttpClickHouseTransport(f"{u.scheme}://{u.hostname}:{u.port}",
                                   user=u.username or "default",
                                   password=u.password or "")


def _script(j) -> dict:
    """The same writes on any backend ; returns the ids it was handed."""
    ids: dict = {}
    r1 = j.log_retrieval("q1", ["A", "B", "C"], ts=1.0)
    r2 = j.log_retrieval("q2", ["A", "B"], ts=2.0)
    r3 = j.log_retrieval(None, ODD, ts=3.0)
    r4 = j.log_retrieval("q4", ["C", "A1", "A"], ts=4.0)
    j.log_retrieval("q5", [], ts=5.0)
    j.mark_useful(r1, 2)
    j.mark_useful(r2, 1)
    j.mark_useful(r2, 0)                      # re-marked : the last one wins
    j.mark_useful(r3, 2)
    j.mark_useful(r4, 0)
    ids["retrievals"] = [r1, r2, r3, r4]
    for s, d, t in [("A", "B", 1.0), ("B", "C", 2.0), ("A", "B", 3.0),
                    ("A", "C", 4.0), ("B", "C", 9.0), ("C", "D", 10.0),
                    ("it's", "été", 11.0)]:
        j.log_hop(s, d, ts=t)
    j.log_path(["A"], ts=1.0)                 # too short : not logged
    for t in (2.0, 3.0, 4.0, 12.0, 13.0):
        j.log_path(["A", "B", "C", "D"], ts=t)
    for t in (5.0, 6.0):
        j.log_path(["X", "Y", "Z", "W"], ts=t)
    j.log_path(["it's", "back\\slash", "été", "A1"], ts=7.0)
    j.log_path(["it's", "back\\slash", "été", "A1"], ts=8.0)
    ids["collisions"] = [
        j.log_collision_event("chasles", child_id="AD", parent_ids=["B", "C"],
                              anchor_ids=["A", "D"], trigger_distance=0.5,
                              threshold=0.7, ts=4.5),
        j.log_collision_event("fission", child_id=None, parent_ids=["B"], ts=6.0),
        j.log_collision_event("lateral", child_id="C", parent_ids=["A1"],
                              anchor_ids=[], ts=7.5),
    ]
    f1 = j.log_forget("B", "superseded by C", superseded_by="C", ts=20.0)
    f2 = j.log_forget("it's", "user corrected", ts=21.0)
    j.log_forget("été", "dup", superseded_by="A", ts=22.0)
    j.mark_forget_merged(f1)
    j.mark_forget_merged(f1)                  # idempotent
    ids["forgets"] = [f1, f2]
    m1 = j.log_merge("B", "C", "merge", "dup", {"state": "VALID"}, ts=30.0)
    m2 = j.log_merge("B", "D", "merge", "better", None, ts=31.0)
    j.log_merge("it's", None, "forget", "", {"tags": ["x"]}, ts=32.0)
    m4 = j.log_merge("E", "C", "lateral", "close", ts=33.0)
    j.mark_merge_reverted(m2)
    ids["ledger"] = [m1, m2, m4]
    j.log_tool_event("t1", "created", "new", ts=40.0)
    j.log_tool_event("t1", "promoted", None, ts=41.0)
    j.log_tool_event("t2", "failed", "boom", ts=42.0)
    o1 = j.log_wiki_op("doc1", "var:x", "set", None, {"v": 1}, ts=50.0)
    o2 = j.log_wiki_op("doc1", "var:x", "set", {"v": 1}, {"v": 2}, ts=51.0)
    j.log_wiki_op("doc2", "portion:p", "remove", {"p": "é"}, None, ts=52.0)
    j.mark_wiki_op_reverted(o2)
    ids["wiki_ops"] = [o1, o2]
    return ids


def _reads(j, ids: dict) -> dict:
    """Every read of the usage tables, ids replaced by their rank in `ids`."""
    # Ids differ by backend (SQLite counts per table from 1, ClickHouse mints
    # them) : an id is compared by its place among the ids of ITS table.
    ranks = {k: {v: f"{k}#{i}" for i, v in enumerate(vs)} for k, vs in ids.items()}

    def norm(x, kind):
        rank = ranks.get(kind, {})
        if isinstance(x, dict):
            return {k: (rank.get(v, "<id>") if k == "id" else v)
                    for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(norm(v, kind) for v in x)
        return x

    r1, r2, r3, r4 = ids["retrievals"]
    out = {
        "counts": j.counts(),
        "pending_forgets": norm(j.pending_forgets(), "forgets"),
        "forget_history": norm(j.forget_history(), "forgets"),
        "active_redirects": j.active_redirects(),
        "redirect_of": {n: j.redirect_of(n) for n in ["B", "E", "it's", "Z"]},
        "absorbed_by": {n: j.absorbed_by(n) for n in ["C", "D", "Z"]},
        "merge_history": norm(j.merge_history(), "ledger"),
        "merge_history_B_live": norm(j.merge_history("B", include_reverted=False), "ledger"),
        "tool_events": j.tool_events(),
        "tool_events_t1": j.tool_events("t1"),
        "wiki_ops_doc1": norm(j.wiki_ops_for_doc("doc1"), "wiki_ops"),
        "wiki_ops_doc2": norm(j.wiki_ops_for_doc("doc2"), "wiki_ops"),
        "get_wiki_op": [norm(j.get_wiki_op(o), "wiki_ops") for o in ids["wiki_ops"]],
        "hop_target_counts": j.hop_target_counts(),
        "effective_spike": {n: j.effective_spike(n)
                            for n in ["A", "B", "C", "D", "été", "Z"]},
        "frequent_paths": j.frequent_paths(),
        "frequent_paths_since": j.frequent_paths(min_len=4, min_freq=2,
                                                 since_ts=5.0),
        "chasles_candidates": j.chasles_candidates(),
        "chasles_candidates_1": j.chasles_candidates(min_len=2, min_freq=1),
        "co_retrieved_pairs": sorted(j.co_retrieved_pairs()),
        "co_retrieved_pairs_w1": sorted(j.co_retrieved_pairs(window=1)),
        "modal_next": {n: j.modal_next(n) for n in ["A", "B", "it's", "Z"]},
        "collision_events": norm(j.collision_events(), "collisions"),
        "collision_events_chasles": norm(j.collision_events("chasles"), "collisions"),
        "find_co_retrieved": {tuple(s): j.find_co_retrieved(s, k=k)
                              for s, k in [(["A"], 7), (["A", "B"], 2),
                                           ([ODD[1], ODD[3]], 7), ([], 7),
                                           (["Z"], 7)]},
        "node_usefulness": {n: j.node_usefulness(n) for n in ["A", "B", "C", "été"]},
        "retrieval_returned_ids": [j.retrieval_returned_ids(r)
                                   for r in (r1, r2, r3, r4)],
        "access_timestamps": {n: j.access_timestamps(n) for n in ["A", "C", "Z"]},
        "useful_retrievals": [(ranks["retrievals"][i], q, ns) for i, q, ns in j.useful_retrievals()],
        "useless_retrievals": [(ranks["retrievals"][i], q, ns) for i, q, ns
                               in j.useful_retrievals(min_score=0, max_score=0)],
    }
    return out


# -- structure (no ClickHouse needed) ----------------------------------------

def test_both_backends_implement_the_whole_protocol():
    ch = ClickHouseJournal(_RecordingTransport(), "ctx.a")
    assert list(iter_journal_methods(SQLiteJournal())) == []
    assert list(iter_journal_methods(ch)) == []
    assert isinstance(SQLiteJournal(), JournalProtocol)
    assert isinstance(ch, JournalProtocol)


def test_journal_is_still_the_sqlite_class():
    # 68 call sites construct `Journal(...)` : it stays the SQLite backend.
    assert journal_mod.SQLiteJournal is journal_mod.Journal or (
        os.environ.get("OMNI_TEST_JOURNAL") == "clickhouse")


def test_state_methods_are_exactly_the_non_usage_ones():
    proto = {n for n in dir(JournalProtocol)
             if not n.startswith("_") and n not in ("path", "close")}
    usage = {n for n in proto if n not in STATE_METHODS}
    assert set(STATE_METHODS) <= proto
    # every usage method is implemented on ClickHouseJournal itself
    for n in usage:
        assert n in ClickHouseJournal.__dict__, n
    assert len(APPEND_ONLY_TABLES) == 9


def test_no_update_on_the_usage_journal():
    src = open(os.path.join(os.path.dirname(journal_mod.__file__),
                            "journal_clickhouse.py")).read()
    assert "UPDATE" not in src and "ALTER TABLE" not in src
    assert "DELETE" not in src


def test_writes_are_batched_and_a_read_flushes_them():
    t = _RecordingTransport()
    j = ClickHouseJournal(t, "ctx.a", batch_size=100)
    j.log_retrieval("q", ["A", "B"], ts=1.0)
    j.log_hop("A", "B", ts=1.0)
    assert t.batches == []                     # still buffered
    j.access_timestamps("A")                   # a read sees its own writes
    tables = sorted(b[0] for b in t.batches)
    assert tables == ["omni_journal.access_events", "omni_journal.hops",
                      "omni_journal.retrievals"]
    assert all(r["context_path"] == "ctx.a" for _, rows in t.batches for r in rows)


def test_a_dead_clickhouse_changes_nothing_for_the_caller():
    t = _DeadTransport()
    j = ClickHouseJournal(t, "ctx.a", batch_size=1, max_buffer=3)
    rid = j.log_retrieval("q", ["A", "B", "C"], ts=1.0)   # never raises
    assert isinstance(rid, int)
    j.mark_useful(rid, 2)
    j.log_forget("A", "x")
    assert j.pending_forgets() == []
    assert j.redirect_of("A") is None
    assert j.active_redirects() == {}
    assert j.counts() == (0, 0)
    assert j.node_usefulness("A") == {"useful": 0, "useless": 0}
    assert j.effective_spike("A") == 0
    assert j.dropped > 0 and len(j._buffer) <= 3        # bounded, counted
    j.log_tags("A", ["a:b"])                             # state still works
    assert j.nodes_with_tag("a:b") == ["A"]
    j.close()


def test_ids_strictly_increase():
    j = ClickHouseJournal(_RecordingTransport(), "ctx.a")
    got = [j.log_hop("A", "B") or j.log_tool_event("t", "e") for _ in range(500)]
    assert got == sorted(got) and len(set(got)) == len(got)


def test_mark_useful_still_validates():
    j = ClickHouseJournal(_RecordingTransport(), "ctx.a")
    with pytest.raises(ValueError):
        j.mark_useful(1, 3)


# -- the two backends answer alike (live throwaway ClickHouse) ----------------

@needs_clickhouse
@pytest.mark.parametrize("batch_size", [1, 7, 1000])
def test_clickhouse_answers_like_sqlite(batch_size):
    ref = SQLiteJournal()
    ref_ids = _script(ref)
    ch = ClickHouseJournal(_transport(), f"test.parity.{uuid.uuid4().hex}",
                           database="omni_journal_test", batch_size=batch_size)
    ch_ids = _script(ch)
    assert _reads(ch, ch_ids) == _reads(ref, ref_ids)


@needs_clickhouse
def test_contexts_do_not_see_each_other():
    t = _transport()
    a = ClickHouseJournal(t, f"test.iso.a.{uuid.uuid4().hex}",
                          database="omni_journal_test")
    b = ClickHouseJournal(t, f"test.iso.b.{uuid.uuid4().hex}",
                          database="omni_journal_test")
    a.log_retrieval("q", ["A", "B"], ts=1.0)
    a.log_merge("A", "B", "merge", "dup", ts=2.0)
    assert b.counts() == (0, 0)
    assert b.active_redirects() == {}
    assert b.find_co_retrieved(["A"]) == []
    assert a.counts() == (1, 2)


@needs_clickhouse
def test_reopening_a_context_reads_what_was_written():
    t = _transport()
    ctx = f"test.reopen.{uuid.uuid4().hex}"
    j1 = ClickHouseJournal(t, ctx, database="omni_journal_test")
    rid = j1.log_retrieval("q", ["A"], ts=1.0)
    j1.mark_useful(rid, 2)
    j1.close()                                 # flushes
    j2 = ClickHouseJournal(t, ctx, database="omni_journal_test")
    assert j2.useful_retrievals() == [(rid, "q", ["A"])]
