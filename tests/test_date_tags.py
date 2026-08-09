"""
Deterministic session-date tagging at ingest.

Every dated turn must carry its date as hierarchical tags (time:year/month/date)
parsed from the "[<date>]" content prefix — never left to the LLM — so a
date-scoped search reaches it. This was the conv-47 girlfriend gap: D6:6
("[20 April 2022] ...") was not date-tagged, so a [april, 2022] filter missed
it.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory, _session_date_tags, query_date_tags
from metacog.tags import filter_points


def test_query_date_tags_extracts_constraint():
    assert query_date_tags("Did James have a girlfriend during April 2022?") == [
        "time:month:april", "time:year:2022"]
    assert query_date_tags("what did X do in 2022?") == ["time:year:2022"]
    assert query_date_tags("anything in July?") == ["time:month:july"]
    assert query_date_tags("no period named") == []


def test_parser_both_formats():
    assert _session_date_tags("[20 April 2022] James: hi") == [
        "time:year:2022", "time:month:april", "time:date:2022-04-20"]
    assert _session_date_tags("[September 4, 2022] x") == [
        "time:year:2022", "time:month:september", "time:date:2022-09-04"]
    assert _session_date_tags("no bracket date") == []
    assert _session_date_tags("[not a date]") == []


def test_ingest_tags_every_dated_turn():
    mem = Memory(encoder=SimpleEncoder())
    p = mem.ingest("[20 April 2022] James: love at first sight with a stranger",
                   kind="FACT", id="D6:6")
    assert "time:month:april" in p.tags
    assert "time:year:2022" in p.tags
    assert "time:date:2022-04-20" in p.tags
    # and it is now reachable by a date-scoped filter (the girlfriend fix)
    assert "D6:6" in filter_points(mem.points,
                                   ["time:month:april", "time:year:2022"])


def test_entity_and_atom_nodes_not_date_tagged():
    # only genuine FACT turns get the prefix date ; derived nodes are skipped
    mem = Memory(encoder=SimpleEncoder())
    p = mem.ingest("[20 April 2022] x", kind="FACT", id="entity_abc")
    assert not any(t.startswith("time:") for t in p.tags)
