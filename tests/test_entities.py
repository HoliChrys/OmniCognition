"""
Tests for the edge-free ENTITY-beacon mechanism.

Entities extracted from a FACT spawn tagged PointKind.FACT *beacon* nodes
that are pulled onto their source fact GEOMETRICALLY (apply_pull) — there
are NO stored edges. A query for an entity value then retrieves the real
source fact (the pull shifted its effective embedding), while the beacon
ids ("entity_*") are dropped from the returned set so they never displace
real evidence.

No live LLM : a FakeEntityExtractor scripts the extraction.
"""

from __future__ import annotations

import pytest

from metacog.memory import Memory
from metacog.defaults import SimpleEncoder
from metacog.entities import ExtractedEntity, LLMEntityExtractor
from metacog.epistemic import (
    LaunderingError,
    Observation,
    Point,
    PointKind,
    SourceClass,
)
from metacog.geometry import distance, effective_keyword_embedding


class FakeEntityExtractor:
    """Scripted entity extractor — maps a substring to a fixed entity list."""

    source = SourceClass.GENERATOR

    def __init__(self, mapping):
        self.mapping = mapping

    def extract_entities(self, text):
        for key, ents in self.mapping.items():
            if key in text:
                return ents
        return []


def _mem(mapping):
    return Memory(
        encoder=SimpleEncoder(),
        entity_extractor=FakeEntityExtractor(mapping),
    )


# --- 1. schema -------------------------------------------------------------

def test_point_has_tags_field_default_empty():
    p = Point(id="x", content="hi", embedding_orig=(1.0, 0.0))
    assert p.tags == []
    # deltas still validate against embedding dim
    assert len(p.delta_active) == 2 and len(p.delta_latent) == 2


def test_tags_roundtrip():
    p = Point(id="x", content="hi", embedding_orig=(1.0, 0.0),
              tags=["date", "year"])
    assert p.tags == ["date", "year"]


# --- 2. spawning & date decomposition -------------------------------------

def test_date_decomposition_spawns_parent_and_components():
    m = _mem({"bday": [ExtractedEntity(
        "date", "13 august 2023",
        {"day": "13", "month": "august", "year": "2023"})]})
    m.ingest("the bday is set", kind="FACT", id="D1:1")
    beacons = [p for p in m.points if p.id.startswith("entity_")]
    # parent date + day + month + year = 4
    assert len(beacons) == 4
    by_tags = {tuple(p.tags): p for p in beacons}
    assert ("date",) in by_tags                       # parent
    assert ("date", "day") in by_tags
    assert ("date", "month") in by_tags
    assert ("date", "year") in by_tags
    # bare values are the matchable first keyword
    assert by_tags[("date", "year")].keywords[0] == "2023"
    assert by_tags[("date", "month")].keywords[0] == "august"
    assert by_tags[("date", "day")].keywords[0] == "13"


def test_partial_date_only_emits_present_parts():
    m = _mem({"yr": [ExtractedEntity("date", "2022", {"year": "2022"})]})
    m.ingest("happened in yr", kind="FACT", id="D1:1")
    beacons = [p for p in m.points if p.id.startswith("entity_")]
    # parent + year only
    tagsets = sorted(tuple(p.tags) for p in beacons)
    assert tagsets == [("date",), ("date", "year")]


def test_non_date_entity_spawns_single_tagged_beacon():
    m = _mem({"swe": [ExtractedEntity("place", "sweden")]})
    m.ingest("from swe originally", kind="FACT", id="D1:1")
    beacons = [p for p in m.points if p.id.startswith("entity_")]
    assert len(beacons) == 1
    assert beacons[0].tags == ["place"]
    assert beacons[0].keywords[0] == "sweden"


# --- 3. per-fact (not deduplicated across facts) --------------------------

def test_same_entity_in_two_facts_spawns_two_beacons():
    m = _mem({"mel": [ExtractedEntity("person", "melanie")]})
    m.ingest("mel went home", kind="FACT", id="D1:1")
    m.ingest("later mel called", kind="FACT", id="D2:1")
    beacons = [p for p in m.points if p.id.startswith("entity_")]
    # one beacon per mention so each co-locates with its own fact
    assert len(beacons) == 2
    assert all(b.keywords[0] == "melanie" for b in beacons)


# --- 4. geometric "edges" : pull co-location ------------------------------

# NOTE: SimpleEncoder is a semantic-free hash, so absolute distance
# comparisons are dominated by hash noise — the *recall* benefit only
# materialises with the real sentence encoder (benchmarked separately).
# Here we assert the DETERMINISTIC mechanical contract instead: the pull
# fired and points the beacon toward its source (the edges-equivalent).

def _vec_sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def test_pull_establishes_geometric_relationship():
    # With entities: both the source fact and the beacon acquire a
    # non-zero delta_latent (they were pulled together). Without entities
    # a plain fact has a zero delta.
    m = _mem({"swe": [ExtractedEntity("place", "sweden")]})
    src = m.ingest("she is from swe originally", kind="FACT", id="D1:1")
    beacon = next(p for p in m.points if p.id.startswith("entity_"))
    assert any(abs(x) > 0 for x in src.delta_latent)      # fact was pulled
    assert any(abs(x) > 0 for x in beacon.delta_latent)   # beacon was pulled

    m0 = Memory(encoder=SimpleEncoder())  # no extractor
    f = m0.ingest("she is from swe originally", kind="FACT", id="D1:1")
    assert all(x == 0 for x in f.delta_latent)            # untouched


def test_beacon_delta_points_toward_source():
    # The first (and only) pull on a single-entity fact moves the beacon
    # along normalize(source - beacon) ; its delta_latent must therefore
    # have a POSITIVE projection on that raw direction. Deterministic,
    # independent of hash noise.
    m = _mem({"swe": [ExtractedEntity("place", "sweden")]})
    src = m.ingest("she is from swe originally", kind="FACT", id="D1:1")
    beacon = next(p for p in m.points if p.id.startswith("entity_"))
    raw_dir = _vec_sub(tuple(src.keywords_embedding),
                       tuple(beacon.keywords_embedding))
    proj = sum(d * r for d, r in zip(beacon.delta_latent, raw_dir))
    assert proj > 0


# --- 5. Cor. 5 compliance --------------------------------------------------

def test_beacons_tagged_generator():
    m = _mem({"swe": [ExtractedEntity("place", "sweden")]})
    m.ingest("from swe", kind="FACT", id="D1:1")
    beacons = [p for p in m.points if p.id.startswith("entity_")]
    assert beacons and all(
        b.keywords_source == SourceClass.GENERATOR for b in beacons)


def test_ingest_creates_no_observation_for_beacons():
    # No counters touched : a pure pull agent never corroborates itself.
    m = _mem({"swe": [ExtractedEntity("place", "sweden")]})
    m.ingest("from swe", kind="FACT", id="D1:1")
    beacons = [p for p in m.points if p.id.startswith("entity_")]
    for b in beacons:
        assert b.n_corrob == 0 and b.n_contra == 0
        assert b.update_log == []


def test_generator_observation_still_raises():
    with pytest.raises(LaunderingError):
        Observation(
            source=SourceClass.GENERATOR,
            signal_type="x",
            polarity=1.0,
            target_node_ids=["entity_abc"],
            timestamp=1.0,
        )


# --- 6. recall : real fact surfaces, beacon does not ----------------------

def test_query_surfaces_real_fact_not_beacon():
    m = _mem({"bday": [ExtractedEntity(
        "date", "13 august 2023",
        {"day": "13", "month": "august", "year": "2023"})]})
    m.ingest("my daughter bday is soon", kind="FACT", id="D1:1")
    m.ingest("we discussed travel plans", kind="FACT", id="D2:1")
    res = m.retrieve("2023", k=5, use_hybrid=True)
    ids = [r["id"] for r in res]
    assert "D1:1" in ids                                   # real fact recalled
    assert not any(i.startswith("entity_") for i in ids)   # no beacon leaked


# --- 7. off by default -----------------------------------------------------

def test_no_extractor_spawns_no_beacons():
    m = Memory(encoder=SimpleEncoder())  # entity_extractor=None
    m.ingest("My daughter birthday 13 august 2023", kind="FACT", id="D1:1")
    assert len(m.points) == 1
    assert not any(p.id.startswith("entity_") for p in m.points)


def test_off_by_default_retrieve_unchanged():
    # With no extractor, retrieve returns exactly the ingested facts.
    m = Memory(encoder=SimpleEncoder())
    m.ingest("alpha topic one", kind="FACT", id="D1:1")
    m.ingest("beta topic two", kind="FACT", id="D2:1")
    res = m.retrieve("alpha", k=5, use_hybrid=True)
    assert {r["id"] for r in res} == {"D1:1", "D2:1"}


# --- 8. extractor parsing (no live LLM) -----------------------------------

def test_llm_entity_extractor_parses_json():
    class StubLLM:
        def generate(self, prompt, max_tokens=None):
            return ('[{"type":"Person","value":"Melanie"},'
                    '{"type":"date","value":"20 January 2023",'
                    '"parts":{"day":"20","month":"January","year":"2023"}}]')

    ex = LLMEntityExtractor(llm=StubLLM())
    ents = ex.extract_entities("some text")
    assert len(ents) == 2
    assert ents[0].etype == "person" and ents[0].value == "melanie"
    assert ents[1].etype == "date"
    assert ents[1].date_parts == {"day": "20", "month": "january",
                                  "year": "2023"}


def test_llm_entity_extractor_total_on_garbage():
    class StubLLM:
        def generate(self, prompt, max_tokens=None):
            return "sorry I cannot help with that"

    ex = LLMEntityExtractor(llm=StubLLM())
    assert ex.extract_entities("x") == []


def test_llm_entity_extractor_strips_code_fence():
    class StubLLM:
        def generate(self, prompt, max_tokens=None):
            return '```json\n[{"type":"place","value":"Rome"}]\n```'

    ex = LLMEntityExtractor(llm=StubLLM())
    ents = ex.extract_entities("x")
    assert len(ents) == 1 and ents[0].value == "rome"
