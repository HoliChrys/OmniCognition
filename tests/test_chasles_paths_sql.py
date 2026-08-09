"""
Chasles paths as a first-class DB unit — 'a path is often travelled' becomes a
plain SQL query, so the Chasles collision TRIGGER is just `chasles_candidates()`.

A traversed path A>B>C>D is logged as one countable row (signature = group key).
Frequency crosses a threshold -> the path is a candidate to collapse into a
start->end shortcut (absorbing the intermediates). The refractory is derived
append-only from the chasles collision_events (like effective_spike), never
mutated.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory


def test_log_path_signature_and_anchors():
    j = Journal()
    rid = j.log_path(["A", "B", "C", "D"], ts=1.0)
    assert rid is not None
    rows = j.frequent_paths(min_len=4, min_freq=1)
    assert len(rows) == 1
    r = rows[0]
    assert r["signature"] == "A>B>C>D"
    assert r["start_id"] == "A" and r["end_id"] == "D"
    assert r["intermediates"] == ["B", "C"]      # what a shortcut absorbs
    assert r["length"] == 4 and r["freq"] == 1


def test_log_path_rejects_too_short():
    j = Journal()
    assert j.log_path(["A"]) is None
    assert j.frequent_paths(min_len=1, min_freq=1) == []


def test_frequent_paths_threshold_and_min_len():
    j = Journal()
    for _ in range(3):
        j.log_path(["A", "B", "C", "D"], ts=1.0)   # travelled 3x
    j.log_path(["A", "X", "D"], ts=1.0)            # len 3 -> below min_len 4
    j.log_path(["P", "Q", "R", "S"], ts=1.0)       # travelled 1x -> below min_freq
    cands = j.frequent_paths(min_len=4, min_freq=2)
    assert [c["signature"] for c in cands] == ["A>B>C>D"]
    assert cands[0]["freq"] == 3


def test_chasles_candidates_refractory_is_derived():
    j = Journal()
    for _ in range(2):
        j.log_path(["A", "B", "C", "D"], ts=1.0)
    # before any fire: it's a live candidate
    assert [c["signature"] for c in j.chasles_candidates(min_freq=2)] == ["A>B>C>D"]
    # Chasles fires on these anchors -> refractory silences the SAME path
    j.log_collision_event("chasles", child_id="AD", parent_ids=["B", "C"],
                          anchor_ids=["A", "D"], ts=2.0)
    assert j.chasles_candidates(min_freq=2) == []
    # travelled again AFTER the fire -> candidate again (append-only reset)
    for _ in range(2):
        j.log_path(["A", "B", "C", "D"], ts=3.0)
    assert [c["signature"] for c in j.chasles_candidates(min_freq=2)] == ["A>B>C>D"]


def test_refractory_is_scoped_to_own_anchors():
    j = Journal()
    for _ in range(2):
        j.log_path(["A", "B", "C", "D"], ts=1.0)
        j.log_path(["E", "F", "G", "H"], ts=1.0)
    # a fire on A/D anchors must NOT silence the unrelated E>F>G>H path
    j.log_collision_event("chasles", child_id="AD", parent_ids=["B", "C"],
                          anchor_ids=["A", "D"], ts=2.0)
    sigs = {c["signature"] for c in j.chasles_candidates(min_freq=2)}
    assert sigs == {"E>F>G>H"}


# -- Memory integration ------------------------------------------------------

def test_memory_log_path_feeds_spike_and_candidates():
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    for _ in range(2):
        m.log_path(["A", "B", "C", "D"], t=1.0)
    # path view -> Chasles trigger by query
    assert [c["signature"] for c in m.chasles_path_candidates(min_freq=2)] \
        == ["A>B>C>D"]
    # per-node spike view stays consistent (each hop was logged too)
    assert j.effective_spike("D") == 2
    assert j.effective_spike("B") == 2


def test_memory_path_noop_without_journal():
    m = Memory(encoder=SimpleEncoder())
    m.log_path(["A", "B", "C", "D"])              # no journal -> silent no-op
    assert m.chasles_path_candidates() == []
