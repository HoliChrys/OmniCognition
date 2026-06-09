"""
Non-kNN filtered listing — the "refer an exhaustive set" primitive.

filter_list scans by PARAMETERS (event, date range, kind, tags) and returns
EVERY match (no query embedding, no walk, no top-k), ordered by event date.
scoped_answer(list_only=True) exposes it as the scoped non-kNN query.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory


def _mem():
    m = Memory(encoder=SimpleEncoder())
    # three dated facts + one undated
    a = m.ingest("strike on the refinery", kind="FACT", id="F_A")
    b = m.ingest("market reaction to the strike", kind="FACT", id="F_B")
    c = m.ingest("later sanctions imposed", kind="FACT", id="F_C")
    d = m.ingest("an unrelated undated note", kind="FACT", id="F_D")
    a.tags.append("time:date:2022-03-01")
    b.tags.append("time:date:2022-03-05")
    c.tags.append("time:date:2022-09-01")
    t = m.ingest("how to mitigate", kind="THOUGHT", id="T_1")
    t.tags.append("time:date:2022-03-02")
    return m


def test_date_range_inclusive_and_ordered():
    m = _mem()
    pts = m.filter_list(date_from="2022-03-01", date_to="2022-03-31")
    ids = [p.id for p in pts]
    assert ids == ["F_A", "T_1", "F_B"]                 # in-range, date-ordered
    assert "F_C" not in ids and "F_D" not in ids        # out-of-range / undated


def test_kind_filter():
    m = _mem()
    facts = m.filter_list(date_from="2022-01-01", date_to="2022-12-31",
                          kind="FACT")
    assert [p.id for p in facts] == ["F_A", "F_B", "F_C"]   # no THOUGHT


def test_no_params_lists_everything_no_knn():
    m = _mem()
    pts = m.filter_list()
    assert len(pts) == 5                                 # a full scan, every node


def test_scoped_answer_list_only_returns_exhaustive_list():
    m = _mem()
    out = m.scoped_answer("anything", tags=[], list_only=True,
                          date_from="2022-03-01", date_to="2022-03-31")
    assert out["list_only"] is True
    assert out["scoped_ids"] == ["F_A", "T_1", "F_B"]   # no walk, just the scan
    assert {i["id"] for i in out["scoped_list"]} == {"F_A", "T_1", "F_B"}


def test_event_filter_restricts_to_cluster(monkeypatch):
    m = _mem()
    # stub the cluster so the test does not depend on event spawning
    monkeypatch.setattr(m, "event_cluster",
                        lambda eid: [p for p in m.points if p.id in ("F_A", "F_B")])
    pts = m.filter_list(event_id="event_x")
    assert {p.id for p in pts} == {"F_A", "F_B"}
