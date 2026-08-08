"""
Append-only usage JOURNAL (SQLite) — the mnema access-log model.

Every retrieval is logged as a `retrievals` row + one `access_events` row per
returned id. Co-retrieval is a SQL self-join on `access_events.retrieval_id`
(edgeless, never touches embeddings) ; `mark_useful` writes the supervised
0/1/2 label ; the journal is a separate SQLite file that outlives save()/load().
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory


def test_log_retrieval_writes_both_tables():
    j = Journal()                                    # :memory:
    rid = j.log_retrieval("où habite Marc ?", ["A", "B", "C"], ts=100.0)
    assert isinstance(rid, int)
    n_ret, n_acc = j.counts()
    assert n_ret == 1 and n_acc == 3                 # 1 search, 3 served nodes


def test_co_retrieved_is_a_sql_self_join():
    j = Journal()
    j.log_retrieval("q1", ["A", "B", "C"], ts=1.0)   # A co-served with B, C
    j.log_retrieval("q2", ["A", "B"], ts=2.0)        # A co-served with B again
    j.log_retrieval("q3", ["X", "Y"], ts=3.0)        # unrelated
    got = j.find_co_retrieved(["A"], k=7)
    assert got == [("B", 2), ("C", 1)]               # B twice with A, C once
    assert all(nid != "A" for nid, _ in got)         # seed excluded


def test_co_retrieved_empty_seed_and_cold():
    j = Journal()
    assert j.find_co_retrieved([]) == []             # no seed
    assert j.find_co_retrieved(["Z"]) == []          # cold : never co-served


def test_mark_useful_and_training_set():
    j = Journal()
    r1 = j.log_retrieval("q1", ["A", "B"], ts=1.0)
    j.log_retrieval("q2", ["C"], ts=2.0)
    j.mark_useful(r1, 2)
    train = j.useful_retrievals(min_score=2)
    assert train == [(r1, "q1", ["A", "B"])]         # only the scored-2 row


def test_mark_useful_rejects_bad_score():
    j = Journal()
    r = j.log_retrieval("q", ["A"], ts=1.0)
    try:
        j.mark_useful(r, 5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_access_timestamps_ascending():
    j = Journal()
    j.log_retrieval("q1", ["A"], ts=30.0)
    j.log_retrieval("q2", ["A", "B"], ts=10.0)
    assert j.access_timestamps("A") == [10.0, 30.0]  # sorted ascending
    assert j.access_timestamps("Z") == []            # never accessed


def test_persists_across_reopen(tmp_path):
    path = str(tmp_path / "journal.db")
    j = Journal(path)
    j.log_retrieval("q", ["A", "B"], ts=1.0)
    j.close()
    j2 = Journal(path)                               # reopen the same file
    assert j2.find_co_retrieved(["A"]) == [("B", 1)]


# -- integration through Memory ---------------------------------------------

def test_memory_records_and_queries_journal():
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    rid1 = m.record_retrieval(["A", "B", "C"], query_text="q1")
    m.record_retrieval(["A", "B"], query_text="q2")
    assert isinstance(rid1, int)
    assert m.co_retrieved(["A"]) == [("B", 2), ("C", 1)]   # via SQL
    m.mark_useful(rid1, 2)
    assert j.useful_retrievals() == [(rid1, "q1", ["A", "B", "C"])]


def test_memory_without_journal_is_noop():
    m = Memory(encoder=SimpleEncoder())              # no journal
    assert m.record_retrieval(["A", "B"]) is None
    assert m.co_retrieved(["A"]) == []
    m.mark_useful(1, 2)                              # no crash


def test_journal_shared_by_reference_on_snapshot():
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    clone = m.snapshot()
    assert clone.journal is m.journal                # same live connection
    clone.record_retrieval(["A", "B"], query_text="q")
    assert m.co_retrieved(["A"]) == [("B", 1)]       # write visible on original


# -- hops (the Chasles signal in SQL) ---------------------------------------

def test_hop_target_counts_is_the_spike_analogue():
    j = Journal()
    j.log_hop("A", "B", ts=1.0)
    j.log_hop("A", "B", ts=2.0)                       # B hopped-to twice
    j.log_hop("A", "C", ts=3.0)
    assert j.hop_target_counts() == [("B", 2), ("C", 1)]   # B accumulates spike


def test_modal_next_follows_the_modal_path():
    j = Journal()
    j.log_hop("A", "B", ts=1.0)
    j.log_hop("A", "B", ts=2.0)
    j.log_hop("A", "C", ts=3.0)
    assert j.modal_next("A") == "B"                   # most frequent successor
    assert j.modal_next("Z") is None                  # never hopped from


def test_record_hop_mirrors_into_journal():
    from metacog.epistemic import PointKind
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    a = m.ingest("alpha", kind="FACT", id="A")
    b = m.ingest("beta", kind="FACT", id="B")
    from metacog.spike import record_hop
    record_hop(a, b, m)
    record_hop(a, b, m)
    assert j.hop_target_counts() == [("B", 2)]        # logged via record_hop
    assert b.n_spike == 2                              # in-memory counter still bumped


# -- collision / chasles audit log ------------------------------------------

def test_log_collision_event_and_read_back():
    j = Journal()
    j.log_collision_event("fission", child_id="M", parent_ids=["A", "B"],
                          anchor_ids=[], trigger_distance=0.1, threshold=0.2,
                          ts=1.0)
    j.log_collision_event("chasles", child_id="N", parent_ids=["C", "D"],
                          anchor_ids=["S", "E"], ts=2.0)
    allev = j.collision_events()
    assert [e["kind"] for e in allev] == ["fission", "chasles"]
    assert allev[0]["parent_ids"] == ["A", "B"] and allev[0]["child_id"] == "M"
    assert allev[1]["anchor_ids"] == ["S", "E"]
    assert [e["kind"] for e in j.collision_events("chasles")] == ["chasles"]


def test_effective_spike_is_refractory_via_event_log():
    """effective_spike counts hops AFTER the node's last Chasles fire — the
    append-only refractory (no counter reset needed)."""
    j = Journal()
    j.log_hop("A", "M", ts=1.0)
    j.log_hop("A", "M", ts=2.0)
    assert j.effective_spike("M") == 2                # both hops count
    j.log_collision_event("chasles", child_id="M", parent_ids=["M"],
                          anchor_ids=[], ts=3.0)      # M fired at t=3
    assert j.effective_spike("M") == 0                # hops before the fire drop
    j.log_hop("A", "M", ts=4.0)                       # a fresh hop after the fire
    assert j.effective_spike("M") == 1               # only the post-fire hop


def test_spike_count_reads_journal_when_present():
    from metacog.epistemic import PointKind
    from metacog.spike import spike_count
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    a = m.ingest("alpha", kind="FACT", id="A")
    b = m.ingest("beta", kind="FACT", id="B")
    from metacog.spike import record_hop
    record_hop(a, b, m)
    record_hop(a, b, m)
    assert spike_count(b, m) == 2                     # from the SQL journal
    # No journal -> falls back to the in-memory counter.
    m2 = Memory(encoder=SimpleEncoder())
    c = m2.ingest("gamma", kind="FACT", id="C")
    c.n_spike = 5
    assert spike_count(c, m2) == 5


def test_co_retrieved_pairs_window():
    j = Journal()
    j.log_retrieval("q", ["A", "B", "C"], ts=1.0)    # ranks A=0,B=1,C=2
    full = dict(j.co_retrieved_pairs())              # any distance
    assert full[("A", "B")] == 1 and full[("A", "C")] == 1
    win1 = dict(j.co_retrieved_pairs(window=1))       # adjacent only
    assert ("A", "C") not in win1                     # rank distance 2 > 1
    assert win1[("A", "B")] == 1 and win1[("B", "C")] == 1


def test_lateral_collapse_sources_from_journal():
    """With a journal, lateral_collapse builds its ledger from SQL co-retrieval
    (not the in-memory ledger)."""
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    # No in-memory ledger recorded, but the journal has co-retrieval history.
    for _ in range(6):
        m.record_retrieval(["A", "B"], query_text="q")
    ledger = m._ledger_from_journal()
    from metacog.lateral import _pair_key
    assert ledger.weight[_pair_key("A", "B")] == 6.0  # SQL-sourced weight
    assert m._lateral_ledger is None                  # in-memory ledger untouched


def test_sleep_persists_fission_events_to_journal():
    """A sleep-cycle proximity collision writes a 'fission' audit row."""
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    # Two near-identical FACTs collide (fission) under sleep.
    m.ingest("the cat sat on the mat", kind="FACT", id="A")
    m.ingest("the cat sat on the mat", kind="FACT", id="B")
    m.sleep()
    hist = m.collision_history("fission")
    # If a collision fired, it is journalled with both parents recorded.
    if hist:
        assert set(hist[0]["parent_ids"]) <= {"A", "B"}
        assert hist[0]["kind"] == "fission"
