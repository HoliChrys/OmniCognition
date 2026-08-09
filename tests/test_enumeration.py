"""Tests for the enumeration / bag list-output mode."""

from __future__ import annotations

from metacog.enumeration import is_enumeration_query, format_bag_answer


def test_detects_enumeration_intent():
    assert is_enumeration_query("find all tweets that mock the war")
    assert is_enumeration_query("list every city John visited")
    assert is_enumeration_query("which turns mention the adoption?")
    assert not is_enumeration_query("what is John's financial status?")
    assert not is_enumeration_query("when did the war start?")


class _LLM:
    def generate(self, prompt, max_tokens=40):
        return "Here are the matches: {{items}}"


def test_format_injects_bag_into_llm_placeholder():
    out = format_bag_answer("find all", [("A", "first"), ("B", "second")], _LLM())
    assert "Here are the matches:" in out
    assert "[A] first" in out and "[B] second" in out


def test_empty_bag_returns_empty():
    assert format_bag_answer("find all", [], _LLM()) == ""


def test_failsafe_no_llm_plain_list():
    out = format_bag_answer("find all", [("A", "x")], object())
    assert "Found 1 matching items" in out and "[A] x" in out


def test_memory_bag_add_dedup_order():
    from metacog.defaults import SimpleEncoder
    from metacog.memory import Memory
    mem = Memory(encoder=SimpleEncoder())
    mem.ingest("alpha", kind="FACT", id="A")
    mem.ingest("beta", kind="FACT", id="B")
    assert mem.bag_add(["A", "B"]) == 2
    assert mem.bag_add("A") == 2            # dedup
    assert mem.bag_add("B") == 2
    items = mem.bag_items()
    assert [i for i, _ in items] == ["A", "B"]   # order preserved
    assert items[0][1] == "alpha"          # content resolved
    mem.bag_clear()
    assert mem.bag_items() == []
