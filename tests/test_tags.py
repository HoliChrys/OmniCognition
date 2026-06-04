"""
Tests for the evolutive, agnostic, multi-tag / multi-kind node typing.

Every Point is auto-typed by its kind ; tags are an open-vocabulary list
that can carry several kind tags (multi-kind), provenance tags, and any
semantic tag added over time (the self-improvement substrate).
"""

from __future__ import annotations

import pickle

from metacog.epistemic import Point, PointKind
from metacog.memory import Memory
from metacog.defaults import SimpleEncoder


def _p(kind=PointKind.FACT, tags=None):
    return Point(id="x", content="hello world", embedding_orig=(1.0, 0.0),
                 kind=kind, tags=list(tags) if tags else [])


def test_every_kind_auto_tagged():
    assert _p(PointKind.FACT).has_tag("fact")
    assert _p(PointKind.ACTION).has_tag("action")
    assert _p(PointKind.THOUGHT).has_tag("thought")


def test_kind_tag_not_duplicated():
    p = _p(PointKind.FACT, tags=["fact", "FACT"])
    assert p.tags.count("fact") == 1


def test_tags_normalized_lowercase_dedup():
    p = _p(PointKind.FACT, tags=["Date", "date", " Year "])
    assert "date" in p.tags and "year" in p.tags
    assert p.tags.count("date") == 1
    # all lowercase, no empties
    assert all(t == t.lower() and t for t in p.tags)


def test_multi_tag_coexist():
    p = _p(PointKind.FACT, tags=["date", "year"])
    assert set(["fact", "date", "year"]).issubset(set(p.tags))


def test_multi_kind():
    # a node may carry several kind tags -> multi-kind
    p = _p(PointKind.FACT)
    p.add_tag("action")
    assert set(p.kinds) == {"fact", "action"}
    assert p.kind is PointKind.FACT  # primary kind unchanged


def test_add_tag_dedup_and_chain():
    p = _p(PointKind.FACT)
    out = p.add_tag("Topic", "topic", "")
    assert out is p                      # chainable
    assert p.tags.count("topic") == 1
    assert "" not in p.tags


def test_tags_roundtrip_pickle():
    p = _p(PointKind.THOUGHT, tags=["generated", "summary"])
    p2 = pickle.loads(pickle.dumps(p))
    assert p2.has_tag("thought") and p2.has_tag("generated") and p2.has_tag("summary")


def test_ingest_tags_fact_kind():
    m = Memory(encoder=SimpleEncoder())
    f = m.ingest("a plain fact", kind="FACT", id="D1:1")
    assert f.has_tag("fact")
    a = m.ingest_action("do the thing", id="act1")
    assert a.has_tag("action")


def test_atomic_provenance_tag():
    from metacog.epistemic import SourceClass
    m = Memory(encoder=SimpleEncoder())

    class FakeAtom:
        source = SourceClass.GENERATOR
        def extract_atoms(self, text, speaker=""):
            return ["Alice lives in Berlin."]
    m.atomic_extractor = FakeAtom()
    m.ingest("Alice: I live in Berlin now", kind="FACT", id="D1:1")
    atoms = [p for p in m.points if p.id.startswith("atom_")]
    assert atoms and atoms[0].has_tag("atomic") and atoms[0].has_tag("fact")


def test_entity_provenance_tag():
    from metacog.epistemic import SourceClass
    m = Memory(encoder=SimpleEncoder())

    class FakeEnt:
        source = SourceClass.GENERATOR
        def extract_entities(self, text):
            from metacog.entities import ExtractedEntity
            return [ExtractedEntity("place", "berlin")]
    m.entity_extractor = FakeEnt()
    m.ingest("Alice: I moved to Berlin", kind="FACT", id="D1:1")
    beacons = [p for p in m.points if p.id.startswith("entity_")]
    assert beacons
    b = beacons[0]
    assert b.has_tag("entity") and b.has_tag("place") and b.has_tag("fact")
