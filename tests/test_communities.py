"""
Tests for edge-free Level-1 community detection + activation.

Communities are detected over FACTs only (THOUGHT/ACTION and entity
beacons excluded), instantiated as named observators, and a walk can
activate one to focus FACT retrieval on its members while still seeing
shared entity beacons and walk scaffolding.
"""

from __future__ import annotations

from metacog.memory import Memory
from metacog.defaults import SimpleEncoder
from metacog.epistemic import PointKind
from metacog.meta_walk import MetaWalker
from metacog.observator import ObservatorView
from metacog.communities import (
    detect_level1_communities,
    _content_tokens,
    _detect_speakers,
)


def _mem_with_community():
    m = Memory(encoder=SimpleEncoder())
    for i in range(6):
        m.ingest(f"fact number {i} about topic", kind="FACT", id=f"D1:{i}")
    b = m.ingest("2023", kind="FACT", id="entity_x1")
    b.tags = ["date"]
    m.declare_observator("auto-comm-0", keywords=["topic"])
    for i in (0, 1, 2):
        p = next(p for p in m.points if p.id == f"D1:{i}")
        p.observator_views["auto-comm-0"] = ObservatorView()
    return m


def test_activation_restricts_to_community_facts():
    m = _mem_with_community()
    w = MetaWalker("q", m, commit=False, observator_id="auto-comm-0")
    facts = [p for p in w._all_points()
             if p.kind == PointKind.FACT and not p.id.startswith("entity_")]
    assert sorted(p.id for p in facts) == ["D1:0", "D1:1", "D1:2"]


def test_entity_beacons_always_visible():
    m = _mem_with_community()
    w = MetaWalker("q", m, commit=False, observator_id="auto-comm-0")
    beacons = [p for p in w._all_points() if p.id.startswith("entity_")]
    assert [p.id for p in beacons] == ["entity_x1"]


def test_bad_observator_id_falls_back_to_all():
    m = _mem_with_community()
    w = MetaWalker("q", m, commit=False, observator_id="nonexistent")
    facts = [p for p in w._all_points()
             if p.kind == PointKind.FACT and not p.id.startswith("entity_")]
    assert len(facts) == 6


def test_no_activation_sees_all():
    m = _mem_with_community()
    w = MetaWalker("q", m, commit=False)
    facts = [p for p in w._all_points()
             if p.kind == PointKind.FACT and not p.id.startswith("entity_")]
    assert len(facts) == 6


def test_detection_excludes_non_facts_and_beacons():
    # Build a memory with clearly two topical clusters of identical-ish text
    m = Memory(encoder=SimpleEncoder())
    for i in range(5):
        m.ingest(f"cooking pasta tomato recipe dinner {i}", kind="FACT",
                 id=f"cook{i}")
    for i in range(5):
        m.ingest(f"hiking mountain trail forest nature {i}", kind="FACT",
                 id=f"hike{i}")
    # add a THOUGHT, an ACTION, an entity beacon — must be excluded
    m.ingest("some derived thought", kind="THOUGHT", id="th1")
    m.ingest("do something", kind="ACTION", id="ac1")
    eb = m.ingest("2023", kind="FACT", id="entity_e1")
    eb.tags = ["date"]
    comms = detect_level1_communities(m.points, m.encoder, min_cluster_size=2)
    member_ids = {pid for c in comms for pid in c.point_ids}
    # No THOUGHT/ACTION/beacon ever ends up in a community
    assert "th1" not in member_ids
    assert "ac1" not in member_ids
    assert "entity_e1" not in member_ids


def test_auto_cluster_observators_creates_observators():
    m = Memory(encoder=SimpleEncoder())
    for i in range(6):
        m.ingest(f"cooking pasta tomato recipe dinner number {i}",
                 kind="FACT", id=f"cook{i}")
    for i in range(6):
        m.ingest(f"hiking mountain trail forest nature outdoors {i}",
                 kind="FACT", id=f"hike{i}")
    comms = m.auto_cluster_observators(min_cluster_size=2)
    # observators registered with auto-comm- prefix
    assert all(c["id"].startswith("auto-comm-") for c in comms)
    for c in comms:
        assert c["id"] in m.observators
        # every member carries the view marker
        for pid in c["point_ids"]:
            p = next(p for p in m.points if p.id == pid)
            assert c["id"] in p.observator_views


def test_speaker_and_noise_filtering():
    # speakers detected from "[date] Speaker:" prefix, dropped from tokens
    pts = [type("P", (), {"content": f"[8 May 2023] Melanie: hello there {i}"})()
           for i in range(6)]
    speakers = _detect_speakers(pts)
    assert "melanie" in speakers
    toks = _content_tokens("[8 May 2023] Melanie: wow that's really awesome painting",
                           drop=["melanie"])
    assert "melanie" not in toks and "may" not in toks and "wow" not in toks
    assert "painting" in toks
