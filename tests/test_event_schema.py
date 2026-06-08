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
    s = induce_event_schema("war", object())   # no .generate
    assert s.is_empty() and s.etype == "war"


def test_slot_subquestions_one_per_slot():
    s = EventSchema(etype="war",
                    slots=["belligerents", "territory"], core=["belligerents"])
    qs = slot_subquestions("the Ukraine war", s)
    assert [slot for slot, _ in qs] == ["belligerents", "territory"]
    assert all("the Ukraine war" in q for _, q in qs)
    assert "belligerents" in qs[0][1]
