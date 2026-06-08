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
