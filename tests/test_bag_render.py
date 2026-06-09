"""
Bag rendering strategies — the agent's choice of how to surface referenced nodes.

raw / extract / interpret / mapreduce + placement inject/bare. Deterministic via
scripted fake LLMs.
"""

from __future__ import annotations

from metacog.bag_render import (
    render_raw, render_extract, render_interpretations, render_mapreduce,
    render_list, render_bag, choose_strategy,
)

_LONG = " ".join("w%d" % i for i in range(60))          # 60 tokens
_ITEMS = [("D1", _LONG), ("D2", "short content"), ("D3", _LONG)]


def test_render_raw_keeps_everything():
    out = render_raw(_ITEMS)
    assert "[D1]" in out and "[D2]" in out and "[D3]" in out
    assert "w59" in out and "..." not in out            # untouched, full content


def test_render_extract_ellipsis_20_20():
    out = render_extract(_ITEMS, head=20, tail=20)
    assert " ... " in out                                # long ones truncated
    assert "w0" in out and "w59" in out                 # head + tail kept
    assert "w30" not in out                              # middle dropped
    assert "short content" in out                        # short one untouched


class _InterpLLM:
    def generate(self, prompt, max_tokens=40):
        return "relevant clause"


def test_render_interpretations_one_per_node():
    out = render_interpretations(_ITEMS, _InterpLLM(), "q")
    assert out.count("relevant clause") == 3            # exhaustive, one each
    assert "[D1]" in out and "[D3]" in out


class _BatchLLM:
    def __init__(self):
        self.calls = 0
    def generate(self, prompt, max_tokens=220):
        self.calls += 1
        assert "Running summary so far" in prompt
        return "summary after batch %d" % self.calls


def test_render_mapreduce_rolls_in_batches():
    llm = _BatchLLM()
    out = render_mapreduce(_ITEMS, llm, "q", batch=2)    # 3 items -> 2 batches
    assert llm.calls == 2
    assert "summary after batch 2" in out               # final rolling summary


class _ChooseLLM:
    def __init__(self, word):
        self.word = word
    def generate(self, prompt, max_tokens=4):
        return self.word


def test_choose_strategy_picks_llm_word():
    assert choose_strategy("q", _ITEMS, _ChooseLLM("mapreduce")) == "mapreduce"
    assert choose_strategy("q", _ITEMS, _ChooseLLM("garbage")) in (
        "raw", "extract")                                # fallback by size


def test_choose_strategy_heuristic_no_llm():
    assert choose_strategy("q", _ITEMS, object()) == "raw"          # <=8 -> raw
    big = [("D%d" % i, "x") for i in range(12)]
    assert choose_strategy("q", big, object()) == "extract"        # >8 -> extract


def test_render_bag_bare_placement_returns_list_only():
    out = render_bag("q", [("D1", "a"), ("D2", "b")], object(),
                     strategy="raw", placement="bare")
    assert out.startswith("- [D1]") and "[D2]" in out   # no framing prose


def test_render_bag_empty_is_empty():
    assert render_bag("q", [], object()) == ""
    assert render_list("q", [], object()) == ""


# ---- map of lists : dynamic multi-value inject -------------------------------
from metacog.bag_render import render_bags


class _MultiLLM:
    def generate(self, prompt, max_tokens=240):
        # places one list, leaves the other to be appended
        return "Summary then {{alpha}} and done."


def test_render_bags_multi_inject_and_leftover_append():
    bags = {"alpha": [("A1", "a one"), ("A2", "a two")],
            "beta": [("B1", "b one")]}
    out, ids = render_bags("q", bags, _MultiLLM())
    assert "A1" in out and "A2" in out          # placed list injected
    assert "B1" in out                          # unplaced list appended
    assert set(ids) == {"A1", "A2", "B1"}


def test_render_bags_single_is_clean():
    out, ids = render_bags("list all", {"default": [("D1", "x"), ("D2", "y")]},
                           object())
    assert out.startswith("- [D1]")             # single -> plain bare list
    assert ids == ["D1", "D2"]


def test_render_bags_empty():
    assert render_bags("q", {}, object()) == ("", [])
    assert render_bags("q", {"a": []}, object()) == ("", [])
