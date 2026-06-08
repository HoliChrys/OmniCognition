"""
Tests for the EVENT node kind and Memory.ingest_event.

PointKind.EVENT is a temporally-extended HUB (event:type:* schema + interval)
that aggregates the facts gravitating around it. Covers: kind/tags/interval,
resolution-dedup (one hub per type+name), the selective-gravitation gate, and
registry rebuild on load.
"""

from __future__ import annotations

import os
import tempfile

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.epistemic import PointKind


def _mem():
    return Memory(encoder=SimpleEncoder())


def _pulled(hub) -> bool:
    """A hub that received >=1 apply_pull has a non-zero active delta."""
    return any(abs(x) > 0 for x in hub.delta_active)


def test_event_kind_tags_and_interval():
    mem = _mem()
    e = mem.ingest_event("the war", "war", t_start="3 March 2022")
    assert e.kind is PointKind.EVENT
    assert "event" in e.tags and "event:type:war" in e.tags
    # interval start tags (right-open: no end tag => ongoing)
    assert "event:start:month:march" in e.tags
    assert "event:start:date:2022-03-03" in e.tags
    assert not any(t.startswith("event:end:") for t in e.tags)


def test_event_resolution_dedup():
    mem = _mem()
    e1 = mem.ingest_event("the war", "war")
    e2 = mem.ingest_event("the war", "war")          # same type+name -> reuse
    assert e1.id == e2.id
    assert mem._event_registry["war::the war"] == e1.id
    assert sum(1 for p in mem.points if p.kind is PointKind.EVENT) == 1
    # a different event type is a distinct hub
    e3 = mem.ingest_event("the election", "election")
    assert e3.id != e1.id


def test_selective_gravitation_gate():
    mem = _mem()
    f = mem.ingest("soldiers set up camps on the front", kind="FACT", id="D1")
    # impossible salience -> nothing gravitates
    e_none = mem.ingest_event("war A", "war", source_facts=[f], salience=2.0)
    assert not _pulled(e_none)
    # permissive salience -> the fact gravitates (hub pulled toward it)
    e_all = mem.ingest_event("war B", "war", source_facts=[f], salience=-1.0)
    assert _pulled(e_all)


def test_event_registry_rebuilt_on_load():
    enc = SimpleEncoder()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pkl")
        mem = Memory(encoder=enc, storage_path=path)
        mem.ingest_event("the war", "war")
        mem.save()
        mem2 = Memory(encoder=enc, storage_path=path)
        mem2.load()
        assert mem2._event_registry.get("war::the war")
        # resolving the same event after load reuses the persisted hub
        e = mem2.ingest_event("the war", "war")
        assert e.id == mem._event_registry["war::the war"]


class _FakeEventExtractor:
    """Scripted: maps a turn substring -> events to spawn."""
    def __init__(self, mapping):
        self.mapping = mapping
    def extract_events(self, text):
        from metacog.event_extractor import ExtractedEvent
        for sub, evs in self.mapping.items():
            if sub in text:
                return [ExtractedEvent(*e) for e in evs]
        return []


def test_auto_spawn_dedups_event_across_turns():
    enc = SimpleEncoder()
    ex = _FakeEventExtractor({
        "war": [("the war", "war", None)],
        "cake": [("birthday", "celebration", None)],
    })
    mem = Memory(encoder=enc, event_extractor=ex)
    mem.ingest("the war started over the border", kind="FACT", id="D1")
    mem.ingest("troops at the front of the war", kind="FACT", id="D2")
    mem.ingest("i baked a cake today", kind="FACT", id="D3")
    hubs = [p for p in mem.points if p.kind is PointKind.EVENT]
    # the two war turns dedup to ONE war hub; the cake is a separate event hub
    war = [h for h in hubs if "event:type:war" in h.tags]
    assert len(war) == 1
    assert any("event:type:celebration" in h.tags for h in hubs)
    assert len(hubs) == 2


def test_event_extractor_not_run_on_derived_nodes():
    enc = SimpleEncoder()
    ex = _FakeEventExtractor({"war": [("the war", "war", None)]})
    mem = Memory(encoder=enc, event_extractor=ex)
    # an entity_/atom_/event_-prefixed id must not recurse into event spawning
    mem.ingest("the war", kind="FACT", id="entity_x")
    assert not any(p.kind is PointKind.EVENT for p in mem.points)


def test_detect_event_type_via_hub_and_none():
    enc = SimpleEncoder()
    mem = Memory(encoder=enc)
    mem.ingest_event("the border war", "war")
    # a query close to the hub name detects the type via the hub node
    det = mem.detect_event_type("the border war", threshold=0.3)
    assert det and det["etype"] == "war" and det["via"] == "hub"
    # an unrelated query with no hub match and no extractor -> None
    assert mem.detect_event_type("zzzz qqqq", threshold=0.99) is None


def test_detect_routes_to_macro_after_consolidation():
    enc = SimpleEncoder()
    mem = Memory(encoder=enc)
    # two war micro-events sharing the entity "iran" -> consolidate merges them
    f1 = mem.ingest("iran blockade incident", kind="FACT", id="D1")
    f2 = mem.ingest("iran strikes incident", kind="FACT", id="D2")
    e1 = mem.ingest_event("iran blockade", "war", source_facts=[f1], salience=-1)
    e2 = mem.ingest_event("iran strikes", "war", source_facts=[f2], salience=-1)
    rep = mem.consolidate_events(min_shared=1)
    assert rep["merged"] >= 1
    # the macro is the canonical hub (largest cluster); a sub-event carries
    # event:in:<macro>. detecting the sub must route up to the macro.
    macro_ids = {e1.id, e2.id}
    det = mem.detect_event_type("iran strikes", threshold=0.2)
    assert det and det["event_id"] in macro_ids
    # the macro's cluster is the UNION (both facts)
    cl = {p.id for p in mem.event_cluster(det["event_id"])}
    assert {"D1", "D2"} <= cl
