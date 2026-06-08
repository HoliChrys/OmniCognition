"""
Tests for event-type schema induction + schema-driven sub-questions.

Deterministic: a scripted fake LLM returns `slot | core/peripheral` lines.
"""

from __future__ import annotations

from metacog.event_schema import (
    induce_event_schema, slot_subquestions, EventSchema, _SCHEMA_CACHE,
)


class _FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, prompt, max_tokens=200):
        self.calls += 1
        return self.text


_WAR = ("belligerents | core\nterritory | core\ntimeline | core\n"
        "casualties | peripheral\ntreaties | peripheral")


def test_induces_slots_and_marks_core():
    _SCHEMA_CACHE.pop("war", None)
    s = induce_event_schema("war", _FakeLLM(_WAR))
    assert s.etype == "war"
    assert s.slots == ["belligerents", "territory", "timeline",
                       "casualties", "treaties"]
    assert s.core == ["belligerents", "territory", "timeline"]
    assert "casualties" not in s.core


def test_schema_is_cached():
    _SCHEMA_CACHE.pop("war", None)
    llm = _FakeLLM(_WAR)
    induce_event_schema("war", llm)
    induce_event_schema("war", llm)            # second call hits the cache
    assert llm.calls == 1


def test_never_caches_empty():
    _SCHEMA_CACHE.pop("nonsense", None)
    induce_event_schema("nonsense", _FakeLLM(""))   # no parseable slots
    assert "nonsense" not in _SCHEMA_CACHE


def test_failsafe_without_llm():
    _SCHEMA_CACHE.pop("uncachedtype", None)
    s = induce_event_schema("uncachedtype", object())   # no .generate, no cache
    assert s.is_empty() and s.etype == "uncachedtype"


def test_slot_subquestions_one_per_slot():
    s = EventSchema(etype="war",
                    slots=["belligerents", "territory"], core=["belligerents"])
    qs = slot_subquestions("the Ukraine war", s)
    assert [slot for slot, _ in qs] == ["belligerents", "territory"]
    assert all("the Ukraine war" in q for _, q in qs)
    assert "belligerents" in qs[0][1]


class _FakeMem:
    """Minimal memory for fill_event_schema: scripted retrieve + llm."""
    def __init__(self, llm, hits_by_kw):
        self.llm = llm
        self.hits_by_kw = hits_by_kw
        class _P:  # minimal point with .id
            def __init__(s, i): s.id = i
        self.points = [_P(i) for kw in hits_by_kw.values() for i in kw]

    def retrieve(self, q, k=3):
        for kw, ids in self.hits_by_kw.items():
            if kw in q:
                return [{"id": i, "content": i, "score": 0.5} for i in ids][:k]
        return []


def test_fill_event_schema_decomposes_and_aggregates():
    from metacog.event_schema import fill_event_schema, _SCHEMA_CACHE
    _SCHEMA_CACHE.pop("war", None)
    llm = _FakeLLM("belligerents | core\nterritory | core\ntimeline | core")
    mem = _FakeMem(llm, {
        "belligerents": ["F1"], "territory": ["F2"], "timeline": ["F3"],
    })
    res = fill_event_schema(mem, "the border war", "war", k_per_slot=2)
    assert res["etype"] == "war"
    assert set(res["filled"]) == {"belligerents", "territory", "timeline"}
    assert res["filled"]["belligerents"][0]["id"] == "F1"
    # union of per-slot hits, deduped, one sub-question per slot
    assert res["fact_ids"] == ["F1", "F2", "F3"]


def test_fill_event_schema_empty_on_no_schema():
    from metacog.event_schema import fill_event_schema, _SCHEMA_CACHE
    _SCHEMA_CACHE.pop("blob", None)
    mem = _FakeMem(_FakeLLM(""), {})
    res = fill_event_schema(mem, "x", "blob")
    assert res["filled"] == {} and res["fact_ids"] == []
