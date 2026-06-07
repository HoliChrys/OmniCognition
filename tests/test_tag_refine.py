"""
Tests for latent-sleep tag refinement (metacog.tag_refine + Memory.refine_tags).

Deterministic : a scripted fake LLM maps phrases -> hierarchical tag strings,
so no live API is needed. Covers decomposition, idempotency, the additive /
failure-safe discipline, and the end-to-end hierarchical-ancestry scope.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.epistemic import Point, PointKind, SourceClass
from metacog.memory import Memory
from metacog.tag_refine import refine_phrase, refine_tags, _is_refinable
from metacog.tags import filter_points, tag_glossary


class _FakeLLM:
    """Maps a refine prompt to a scripted comma-separated tag line."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0

    def generate(self, prompt, max_tokens=80):
        self.calls += 1
        for phrase, tags in self.mapping.items():
            if f"Phrase: {phrase}\n" in prompt:
                return tags
        return ""


_MAP = {
    "fingers too big": "body:hand:finger, health:condition:macrodactyly",
    "morning run": "activity:exercise:running, time:period:morning",
    "puppy adoption": "animal:canine:puppy, life_stage:juvenile",
    "exhausted all the time": "body:energy, health:condition:fatigue",
}


def _fact(pid, content, kws):
    return Point(id=pid, content=content,
                 embedding_orig=SimpleEncoder().encode(content),
                 kind=PointKind.FACT, keywords=kws,
                 keywords_source=SourceClass.COMPUTATION)


def test_refine_phrase_decomposes_to_hierarchical_tags():
    llm = _FakeLLM(_MAP)
    tags = refine_phrase("fingers too big", llm)
    assert tags == ["body:hand:finger", "health:condition:macrodactyly"]
    # every tag is hierarchical (carries a namespace)
    assert all(":" in t for t in tags)


def test_refine_phrase_skips_structural_and_short_tags():
    llm = _FakeLLM(_MAP)
    assert not _is_refinable("fact")
    assert not _is_refinable("date")
    assert not _is_refinable("a")
    assert not _is_refinable("body:finger")     # already hierarchical
    assert refine_phrase("fact", llm) == []
    assert llm.calls == 0                         # never reached the LLM


def test_refine_phrase_never_caches_empty():
    import metacog.tag_refine as tr
    tr._REFINE_CACHE.pop("widget gizmo", None)
    empty = _FakeLLM({})                          # returns "" for everything
    assert refine_phrase("widget gizmo", empty) == []
    assert "widget gizmo" not in tr._REFINE_CACHE  # a 529 must not stick


def test_refine_tags_union_dedup():
    llm = _FakeLLM(_MAP)
    out = refine_tags(["fingers too big", "morning run"], llm)
    assert "body:hand:finger" in out and "activity:exercise:running" in out
    assert len(out) == len(set(out))             # deduped


def test_memory_refine_tags_appends_and_is_idempotent():
    enc = SimpleEncoder()
    mem = Memory(encoder=enc, llm=_FakeLLM(_MAP), tags_refine_enabled=True)
    mem.points.append(_fact("D1:27", "fingers too big and a morning run",
                            ["fingers too big", "morning run"]))
    mem.points.append(_fact("D5:2", "adopted a puppy", ["puppy adoption"]))

    rep = mem.refine_tags()
    assert rep["refined_points"] == 2
    p = next(p for p in mem.points if p.id == "D1:27")
    assert "health:condition:macrodactyly" in p.tags
    assert "refined" in p.tags                    # idempotency marker

    # second pass : nothing left to do, no duplicate tags
    before = list(p.tags)
    rep2 = mem.refine_tags()
    assert rep2["refined_points"] == 0
    assert p.tags == before


def test_refine_on_ingest_tags_land_in_glossary_registry():
    """With tags_refine_on_ingest, a FACT is hierarchically tagged at ingest
    and the new namespaces appear in the tag glossary (the registry the
    presearch scope reads)."""
    enc = SimpleEncoder()

    class _KX:
        source = SourceClass.COMPUTATION
        def extract(self, text, n=5):
            return ["fingers too big"]

    mem = Memory(encoder=enc, llm=_FakeLLM(_MAP), extractor=_KX(),
                 tags_refine_on_ingest=True)
    p = mem.ingest("my fingers are too big", kind="FACT", id="D1:27")
    assert "health:condition:macrodactyly" in p.tags
    assert "refined" in p.tags
    # the namespace is now in the glossary registry used by presearch scoping
    assert "health:condition" in tag_glossary(mem.points)
    assert filter_points(mem.points, ["health:condition"]) == {"D1:27"}


def test_scope_on_namespace_finds_inference_turn():
    """The end-to-end payoff : a turn whose surface words never mention health
    is reachable by SCOPING on the latent `health:condition` namespace, while
    a non-health turn is excluded."""
    enc = SimpleEncoder()
    mem = Memory(encoder=enc, llm=_FakeLLM(_MAP), tags_refine_enabled=True)
    mem.points.append(_fact("D1:27", "fingers too big", ["fingers too big"]))
    mem.points.append(_fact("D9", "so tired", ["exhausted all the time"]))
    mem.points.append(_fact("D5", "a puppy", ["puppy adoption"]))
    mem.refine_tags()

    hits = filter_points(mem.points, ["health:condition"])
    assert "D1:27" in hits and "D9" in hits
    assert "D5" not in hits                        # puppy is not a health turn
    # the namespace is discoverable in the glossary
    assert "health:condition" in tag_glossary(mem.points)
