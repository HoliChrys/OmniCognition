"""
Tri-modal relevance search over a filtered node pool — WITHOUT REPLACEMENT.

search_nodes ranks the filtered pool by sim | regex | fuzzy over content and/or
tags ; nodes already in the bag are excluded, so the agent's collect-loop draws
without replacement and converges.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory


def _mem():
    m = Memory(encoder=SimpleEncoder())
    a = m.ingest("iran strike on the refinery", kind="FACT", id="A")
    b = m.ingest("market reaction in tehran", kind="FACT", id="B")
    c = m.ingest("unrelated cooking recipe", kind="FACT", id="C")
    a.tags.append("event:type:war")
    b.tags.append("event:type:war")
    b.tags.append("region:tehran")
    return m


def test_sim_keyword_overlap_on_content():
    m = _mem()
    got = [p.id for p in m.search_nodes("iran refinery strike", mode="sim",
                                        on="content")]
    assert got[0] == "A"                       # most query tokens overlap
    assert "C" not in got                       # no overlap -> excluded


def test_regex_on_tags():
    m = _mem()
    got = {p.id for p in m.search_nodes(r"event:type:war", mode="regex",
                                        on="tags")}
    assert got == {"A", "B"}                     # both carry the tag, C does not


def test_fuzzy_on_content_tolerates_typo():
    m = _mem()
    got = [p.id for p in m.search_nodes("refinary", mode="fuzzy", on="content")]
    assert "A" in got                            # 'refinary' ~ 'refinery'


def test_without_replacement_excludes_bag():
    m = _mem()
    first = [p.id for p in m.search_nodes(r"war", mode="regex",
                                          on="tags")]   # A,B via the war tag
    assert set(first) == {"A", "B"}
    m.bag_add(["A"], bag="default")             # collect A
    second = {p.id for p in m.search_nodes(r"war", mode="regex", on="tags",
                                           exclude_bag="default")}
    assert second == {"B"}                        # A no longer drawn (no remise)
    m.bag_add(["B"], bag="default")
    third = m.search_nodes(r"war", mode="regex", on="tags",
                           exclude_bag="default")
    assert third == []                            # pool exhausted -> loop ends


def test_event_filter_scopes_the_pool(monkeypatch):
    m = _mem()
    monkeypatch.setattr(m, "event_cluster",
                        lambda eid: [p for p in m.points if p.id in ("A",)])
    got = {p.id for p in m.search_nodes("iran", mode="sim", on="content",
                                        event_id="e1")}
    assert got == {"A"}                           # pool restricted to the event
