"""
Concept A : the mark_useful feedback loop.

Retrievals are auto-labelled by what was actually USED downstream, so fit_decay
(the one learned hyperparameter) finally has supervision — the L3 loop closes
without a human. The walk scores its own retrievals at finalize by the
fact-star chain it committed to.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory


def test_score_retrievals_maps_overlap_to_0_1_2():
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    r_useful = m.record_retrieval(["A", "B", "C"], query_text="q1")   # 2 used -> 2
    r_one = m.record_retrieval(["A", "X"], query_text="q2")           # 1 used -> 1
    r_none = m.record_retrieval(["Y", "Z"], query_text="q3")          # 0 used -> 0
    scored = dict(m.score_retrievals([r_useful, r_one, r_none],
                                     used_ids=["A", "B"]))
    assert scored[r_useful] == 2
    assert scored[r_one] == 1
    assert scored[r_none] == 0


def test_feedback_unlocks_fit_decay():
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    # before any labels, fit_decay has no classes -> keeps the default
    assert m.fit_decay()["n_pos"] == 0
    ru = m.record_retrieval(["A", "B", "C"], query_text="useful")
    rn = m.record_retrieval(["X", "Y"], query_text="useless")
    m.score_retrievals([ru, rn], used_ids=["A", "B"])
    res = m.fit_decay()
    assert res["n_pos"] > 0 and res["n_neg"] > 0        # both classes present now


def test_sleep_refits_decay_from_feedback():
    """sleep() auto-maintains the decay exponent from accumulated labels."""
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    for nid in ("A", "B", "C", "D"):
        m.ingest(f"{nid} topic", kind="FACT", id=nid)
    ru = m.record_retrieval(["A", "B"], query_text="useful")   # 2 used -> 2
    rn = m.record_retrieval(["C", "D"], query_text="useless")  # 0 used -> 0
    m.score_retrievals([ru], used_ids=["A", "B"])
    m.score_retrievals([rn], used_ids=[])
    out = m.sleep()
    assert out["decay_fit_n_pos"] > 0 and out["decay_fit_n_neg"] > 0
    assert "decay_exponent" in out


def test_sleep_decay_fit_noop_without_labels():
    """Without both label classes, sleep keeps the exponent (no crash)."""
    m = Memory(encoder=SimpleEncoder(), journal=Journal())
    m.ingest("x", kind="FACT", id="A")
    out = m.sleep()
    assert out["decay_fit_n_pos"] == 0 and out["decay_exponent"] == 0.5


def test_score_retrievals_noop_without_journal():
    m = Memory(encoder=SimpleEncoder())
    assert m.score_retrievals([1, 2], ["A"]) == []


def test_retrieval_returned_ids_rank_order():
    j = Journal()
    rid = j.log_retrieval("q", ["first", "second", "third"])
    assert j.retrieval_returned_ids(rid) == ["first", "second", "third"]


def test_walk_scores_retrievals_by_usage():
    """The walk's finalize hook labels its retrievals by the used fact chain."""
    from metacog.meta_walk import MetaWalker
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    a = m.ingest("alpha", kind="FACT", id="A")
    b = m.ingest("beta", kind="FACT", id="B")
    # a retrieval that surfaced A (used) and B (not used)
    rid = m.record_retrieval(["A", "B"], query_text="w")
    w = MetaWalker.__new__(MetaWalker)
    w.memory = m
    w.commit = True
    w._retrieval_ids = [rid]
    w._fact_star_chain = [a]                            # only A was used
    w._score_retrievals_by_usage()
    # exactly one returned id used -> score 1 (labelled, not left NULL)
    labelled = j.useful_retrievals(min_score=1, max_score=1)
    assert any(r[0] == rid for r in labelled)
