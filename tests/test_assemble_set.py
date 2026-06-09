"""
Orchestrated exhaustive-set assembly — the agentic loop.

assemble_set scopes to the event, loops sim→fuzzy→regex search_nodes, judges
relevance (LLM), collects hits into the bag WITHOUT REPLACEMENT, and converges.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory


class _JudgeLLM:
    """Deterministic judge : an item is relevant iff it mentions 'iran'. Speaks
    the per-item VERDICT format and the batch `i: label` format."""
    def generate(self, prompt, max_tokens=160):
        if "ITEM :" in prompt:                          # per-item stance THOUGHT
            item = prompt.split("ITEM :", 1)[1]
            return ("VERDICT: RELEVANT" if "iran" in item.lower()
                    else "VERDICT: IRRELEVANT")
        out = []
        for line in prompt.splitlines():                # batch CoN format
            line = line.strip()
            if line.startswith("[") and "]" in line:
                idx = line[1:line.index("]")]
                if not idx.isdigit():
                    continue
                lab = "relevant" if "iran" in line.lower() else "irrelevant"
                out.append(f"{idx}: {lab}")
        return "\n".join(out)


def _mem(llm=None):
    m = Memory(encoder=SimpleEncoder(), llm=llm)
    a = m.ingest("iran strike on the refinery", kind="FACT", id="A")
    b = m.ingest("tehran market reaction in iran", kind="FACT", id="B")
    c = m.ingest("unrelated cooking recipe", kind="FACT", id="C")
    for p in (a, b, c):
        p.tags.append("event:type:war")          # same filtered pool
    return m


def test_assemble_collects_relevant_until_converged():
    m = _mem(_JudgeLLM())
    rep = m.assemble_set("war in iran", tags=["event:type:war"],
                         modes=("sim", "regex"), k=10, max_rounds=4)
    bag = {i for i, _ in m.bag_items(bag="default")}
    assert bag == {"A", "B"}                      # only the iran-relevant kept
    assert "C" not in bag                          # judged irrelevant
    assert rep["size"] == 2
    assert rep["added"][-1] == 0                   # converged (a quiet last round)


def test_without_replacement_no_double_count():
    m = _mem(_JudgeLLM())
    rep = m.assemble_set("war in iran", tags=["event:type:war"],
                         modes=("sim", "fuzzy", "regex"), k=10, max_rounds=5)
    # each relevant node counted exactly once across all rounds/modes
    assert sum(rep["added"]) == 2
    assert rep["rounds"] <= 5


def test_no_llm_accepts_all_recall_first():
    m = _mem(llm=None)                             # no judge -> recall-first
    m.assemble_set(r"war", tags=["event:type:war"], modes=("regex",),
                   on="tags", k=10, max_rounds=3)
    bag = {i for i, _ in m.bag_items(bag="default")}
    assert bag == {"A", "B", "C"}                  # all in-scope nodes collected


def test_bag_carries_assembled_schema():
    m = _mem(_JudgeLLM())
    m.assemble_set("war in iran", tags=["event:type:war"], modes=("sim",), k=10)
    meta = m.bag_meta("default")
    assert meta["schema"]["kind"] == "assembled_set"
