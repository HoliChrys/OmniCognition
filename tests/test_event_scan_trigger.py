"""
The walk's meta-cognition/ACTION launches the NON-kNN event scan.

When a focused fact reveals the input relates to an EVENT and the request refers
to an exhaustive set, the walk folds Memory.filter_list(event_id=...) into its
evidence — a targeted filtered query, not a parallel one. Gated to recall intent
and fired once per event.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.meta_walk import MetaWalker, _query_year_range


def _mem_with_event():
    m = Memory(encoder=SimpleEncoder())
    hub = m.ingest("strike on the refinery", kind="EVENT", id="event_z")
    members = []
    for i in range(6):
        f = m.ingest("tweet %d about the refinery strike" % i,
                     kind="FACT", id="F%d" % i)
        f.tags.append("event:in:event_z")
        members.append(f)
    # one member NOT near the query embedding -> kNN would miss it, the scan won't
    members[5].tags.append("time:date:2022-06-01")
    return m, members


def _walk(m, query, n=5):
    w = MetaWalker(query, m, n_stages=n, commit=False)
    for _ in range(n):
        if w.step().done:
            break
    return w


def test_year_range_helper():
    assert _query_year_range("all strikes in 2022") == ("2022-01-01", "2022-12-31")
    assert _query_year_range("no year here") == (None, None)


def test_scan_fires_on_enumeration_and_folds_all_members():
    m, members = _mem_with_event()
    w = _walk(m, "list all the tweets about the refinery strike")
    got = {p.id for p in w._relevant_cum}
    # every event member is folded in via the scan (exhaustive recall)
    assert {f.id for f in members} <= got
    assert "event_z" in getattr(w, "_scanned_events", set())
    # published to the event bag with description+schema
    assert m.bag_meta("event:event_z")["schema"]["event_id"] == "event_z"


def test_scan_does_not_fire_on_focused_query():
    m, _ = _mem_with_event()
    w = _walk(m, "what refinery was struck?")          # focused, not a set
    assert not getattr(w, "_scanned_events", set())


def test_scan_once_per_event():
    m, _ = _mem_with_event()
    w = _walk(m, "find all tweets about the strike", n=5)
    # the guard set holds the event exactly once (idempotent across stages)
    assert list(getattr(w, "_scanned_events", set())).count("event_z") == 1
