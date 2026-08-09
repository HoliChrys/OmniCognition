"""
Mode-aware final composition of the agent's answer (the two answer modes).

A query is either "generate a focused answer" (F1) or "refer an exhaustive set"
(recall), and the two can coexist (answer + list). `_compose_final` shapes the
final output accordingly. Deterministic via a scripted fake LLM.
"""

from __future__ import annotations

from benchmarks.locomo.mcp_meta_agent import _compose_final


class _FrameLLM:
    """format_bag_answer asks for a one-line framing with a {{placeholder}}."""
    def generate(self, prompt, max_tokens=40):
        return "Here are the items: {{list}}"


_BAG = [("D1", "first tweet"), ("D2", "second tweet")]


def test_empty_bag_keeps_focused_answer():
    ans, ids = _compose_final("Paris", [], "Where does John live?", _FrameLLM())
    assert ans == "Paris" and ids == []


def test_enumeration_query_returns_list_only():
    ans, ids = _compose_final("ignored", _BAG, "list all the tweets about X",
                              _FrameLLM())
    assert "D1" in ans and "D2" in ans
    assert "ignored" not in ans                 # focused answer dropped
    assert ids == ["D1", "D2"]


def test_no_real_answer_returns_list_only():
    ans, ids = _compose_final("not mentioned", _BAG, "what did X say?",
                              _FrameLLM())
    assert "D1" in ans and "not mentioned" not in ans
    assert ids == ["D1", "D2"]


def test_focused_answer_plus_bag_yields_both():
    ans, ids = _compose_final("It happened in May", _BAG,
                              "when did it happen?", _FrameLLM())
    assert ans.startswith("It happened in May")   # focused kept
    assert "D1" in ans and "D2" in ans            # AND the exhaustive list
    assert ids == ["D1", "D2"]
