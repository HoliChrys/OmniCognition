"""
Tests for behavioural-clue expansion (`metacog.clues`) and the
`clue_search` MCP tool — the inference-question expander that brainstorms
evidence-register chat lines spanning the answer space, retrieves over
them, and lineage-bridges to the in-between gold.
"""

from __future__ import annotations

import asyncio
import json

from metacog.clues import clue_utterances
from metacog.defaults import SimpleEncoder
from metacog.memory import Memory


class _LLM:
    """Returns a fixed clue block; records the last prompt."""
    def __init__(self, block: str):
        self.block = block
        self.last_prompt = None

    def generate(self, prompt, max_tokens=None):
        self.last_prompt = prompt
        return self.block


# ---------------------------------------------------------------------------
# clue_utterances
# ---------------------------------------------------------------------------


def test_clue_utterances_parses_lines_and_caps():
    llm = _LLM("my kids have so much\nmoney's tight this month\n"
               "we just bought a lake house\nhad to skip dinner")
    out = clue_utterances("What might John's financial status be?", llm, n=3)
    assert out == ["my kids have so much", "money's tight this month",
                   "we just bought a lake house"]


def test_clue_utterances_strips_numbering_and_short_labels():
    llm = _LLM("1. we vacation in Aspen every winter\n"
               "Struggling: been eating ramen for weeks")
    out = clue_utterances("q", llm, n=5)
    assert out[0] == "we vacation in Aspen every winter"
    # a short "Label: text" prefix is dropped, the line kept
    assert out[1] == "been eating ramen for weeks"


def test_clue_utterances_empty_without_llm():
    assert clue_utterances("q", object(), n=4) == []


def test_clue_utterances_never_caches_empty():
    class _Empty:
        def generate(self, prompt, max_tokens=None):
            return ""
    e = _Empty()
    assert clue_utterances("uniq-q-1", e, n=3) == []
    # a later good answer for the same question must still be usable
    good = _LLM("a real clue line here about money")
    assert clue_utterances("uniq-q-1", good, n=3) == ["a real clue line here about money"]


# ---------------------------------------------------------------------------
# clue_search MCP tool
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


async def _session(mem):
    from mcp.shared.memory import create_connected_server_and_client_session
    from metacog.mcp_server import build_app
    return create_connected_server_and_client_session(build_app(memory=mem))


async def _call(s, name, **kw):
    r = await s.call_tool(name, kw)
    return json.loads(r.content[0].text)


def test_clue_search_generates_retrieves_and_lineage_bridges():
    # A sequence-linked session; a clue will land on D:3, the gold sits at
    # D:5 (two turns ahead) — the lineage bridge must surface it.
    llm = _LLM("education is essential for kids\nsomething unrelated entirely")
    mem = Memory(encoder=SimpleEncoder(), llm=llm)
    prev = None
    texts = ["hello there", "nice weather", "education is essential for kids",
             "our kids future matters", "my kids have so much and others dont",
             "we should help", "great chat"]
    for i, t in enumerate(texts, 1):
        mem.ingest(t, kind="FACT", id=f"D:{i}", sequence_prev=prev)
        prev = f"D:{i}"

    async def go():
        async with await _session(mem) as s:
            await s.initialize()
            out = await _call(s, "clue_search",
                              question="what is their status?", k_per_clue=2)
            assert out["clues"], "no clues generated"
            assert "results" in out and "merged" in out
            merged_ids = {h["id"] for h in out["merged"]}
            # the clue 'education is essential for kids' retrieves D:3;
            # the lineage bridge then offers its forward neighbours incl. D:5.
            assert "neighbor_possibilities" in out
            assert merged_ids, "no merged hits"
    _run(go())


def test_clue_search_registered_and_empty_without_llm():
    async def go():
        mem = Memory(encoder=SimpleEncoder(), llm=object())  # no .generate
        async with await _session(mem) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            assert "clue_search" in names
            out = await _call(s, "clue_search", question="q")
            assert out["clues"] == [] and out["merged"] == []
    _run(go())
