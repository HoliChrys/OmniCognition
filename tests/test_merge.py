"""
Tests for true-merge deduplication — "quand l'info est pareille".

When two same-kind points carry the SAME information they collapse into a
SINGLE node : the duplicate is dropped (no parent retention, unlike
fission), its corroboration absorbed by the survivor, and an id alias is
recorded so the dropped id still resolves.
"""

from __future__ import annotations

from metacog.memory import Memory
from metacog.defaults import SimpleEncoder
from metacog.epistemic import EpistemicState, Point, PointKind
from metacog.collision import (
    detect_identical,
    merge_duplicates,
    resolve_merge,
)


def _pt(id, content, kind=PointKind.FACT, n_corrob=0):
    return Point(id=id, content=content,
                 embedding_orig=tuple(SimpleEncoder().encode(content)),
                 kind=kind, n_corrob=n_corrob)


def test_identical_content_merges_to_single_node():
    m = Memory(encoder=SimpleEncoder())
    m.ingest("Jon lost his job as a banker", kind="FACT", id="D1:2")
    m.ingest("we chatted about the weather", kind="FACT", id="D1:5")
    m.ingest("Jon lost his job as a banker", kind="FACT", id="D7:9")  # dup
    m.ingest("pizza is great", kind="FACT", id="D2:1")
    before = len(m.points)
    rep = m.consolidate_duplicates()
    assert rep["merged_count"] == 1
    assert len(m.points) == before - 1
    # exactly one of the duplicate ids survives
    ids = {p.id for p in m.points}
    assert ("D1:2" in ids) ^ ("D7:9" in ids)


def test_absorbed_id_resolves_to_survivor():
    m = Memory(encoder=SimpleEncoder())
    a = m.ingest("identical info here", kind="FACT", id="D1:1")
    b = m.ingest("identical info here", kind="FACT", id="D9:9")
    m.ingest("different one", kind="FACT", id="D2:2")
    m.ingest("another different", kind="FACT", id="D3:3")
    m.consolidate_duplicates()
    survivors = {p.id for p in m.points}
    absorbed = "D9:9" if "D1:1" in survivors else "D1:1"
    keeper = "D1:1" if "D1:1" in survivors else "D9:9"
    assert m.resolve_alias(absorbed) == keeper
    # survivors resolve to themselves
    assert m.resolve_alias(keeper) == keeper


def test_corroboration_absorbed():
    keeper = _pt("k", "same text", n_corrob=2)
    absorbed = _pt("a", "same text", n_corrob=3)
    resolve_merge(keeper, absorbed, t_now=1.0)
    # keeper gains absorbed's corroborations + 1 (the duplicate observation)
    assert keeper.n_corrob == 2 + 3 + 1


def test_different_kinds_do_not_merge():
    pts = [
        _pt("f1", "same text", kind=PointKind.FACT),
        _pt("a1", "same text", kind=PointKind.ACTION),
        _pt("x", "filler one"),
        _pt("y", "filler two"),
    ]
    pairs = detect_identical(pts, t_now=1.0)
    # the FACT/ACTION pair shares text but differs in kind -> not a merge pair
    assert all(not {a.id, b.id} == {"f1", "a1"} for a, b, _ in pairs)


def test_distinct_info_not_merged():
    m = Memory(encoder=SimpleEncoder())
    for i, txt in enumerate(["alpha beta gamma", "delta epsilon zeta",
                             "eta theta iota", "kappa lambda mu"]):
        m.ingest(txt, kind="FACT", id=f"D{i}:0")
    before = len(m.points)
    rep = m.consolidate_duplicates()
    assert rep["merged_count"] == 0
    assert len(m.points) == before


def test_deprecated_excluded_from_merge():
    pts = [
        _pt("a", "same text"),
        _pt("b", "same text"),
        _pt("c", "filler one"),
        _pt("d", "filler two"),
    ]
    pts[1].state = EpistemicState.DEPRECATED
    pairs = detect_identical(pts, t_now=1.0)
    assert all("b" not in {a.id, bb.id} for a, bb, _ in pairs)


def test_merge_pass_touches_point_once():
    # three identical points -> only one merge in a single pass (the third
    # is untouched this round, mergeable on a later pass)
    pts = [_pt(f"p{i}", "exactly the same") for i in range(3)]
    pts += [_pt("x", "f1"), _pt("y", "f2")]
    rep = merge_duplicates(pts, t_now=1.0)
    assert len(rep.merged) == 1
    assert len(pts) == 5 - 1
