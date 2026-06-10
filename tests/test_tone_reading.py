"""
Ingest-time TONE READING — irony/intended-meaning THOUGHTs move to ingestion.

Tone is a property of the document, not of any query : read once at ingest
(tone:* tag + gloss THOUGHT for non-plain tones), then the stance judge batches
over the precomputed glosses (cost gate) and semantic search matches the
intended meaning. Deterministic via scripted fake LLMs.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.epistemic import PointKind, SourceClass
from metacog.memory import Memory


class _ToneLLM:
    """Marks texts containing 'lol' ironic (intended: opposite), else plain."""
    def __init__(self):
        self.calls = 0
    def generate(self, prompt, max_tokens=90):
        self.calls += 1
        if "TEXT :" in prompt:
            txt = prompt.split("TEXT :", 1)[1]
            if "lol" in txt.lower():
                return "TONE: ironic\nINTENDED: the author mocks the claim of victory"
            return "TONE: plain\nINTENDED: says what it means"
        if "ITEM :" in prompt:                       # per-item stance THOUGHT
            item = prompt.split("ITEM :", 1)[1]
            return ("VERDICT: RELEVANT" if "lol" in item.lower()
                    else "VERDICT: IRRELEVANT")
        # batch judge : relevant iff the INTENDED line mentions 'mocks'
        out = []
        cur = None
        for ln in prompt.splitlines():
            s = ln.strip()
            if s.startswith("[") and "]" in s and s[1:s.index("]")].isdigit():
                if cur is not None:
                    out.append(cur)
                cur = f"{s[1:s.index(']')]}: irrelevant"
            elif s.startswith("INTENDED") and "mocks" in s and cur:
                cur = cur.replace("irrelevant", "relevant")
        if cur is not None:
            out.append(cur)
        return "\n".join(out)


def _mem():
    m = Memory(encoder=SimpleEncoder(), llm=_ToneLLM(), tone_reading=True)
    a = m.ingest("we totally won lol", kind="FACT", id="A")
    b = m.ingest("the war is ongoing in the north", kind="FACT", id="B")
    return m, a, b


def test_tone_tag_and_gloss_spawned_at_ingest():
    m, a, b = _mem()
    assert "tone:ironic" in a.tags
    assert "tone:plain" in b.tags
    g = m.gloss_of("A")
    assert "mocks" in g                              # intended meaning stored
    assert m.gloss_of("B") == ""                     # plain -> no gloss node
    gp = next(p for p in m.points if p.id == "gloss_A")
    assert gp.kind is PointKind.THOUGHT
    assert gp.keywords_source is SourceClass.GENERATOR   # Cor. 5
    assert any(t.startswith("gloss:of:") for t in gp.tags)


def test_tone_cached_once_per_content():
    m, _, _ = _mem()
    before = m.llm.calls
    m.ingest("we totally won lol", kind="FACT", id="A2")   # same content
    assert m.llm.calls == before                     # served from _tone_cache


def test_gloss_excluded_from_doc_pools():
    m, _, _ = _mem()
    ids = {p.id for p in m.search_nodes("war", mode="sim", on="content")}
    assert not any(i.startswith("gloss_") for i in ids)


def test_judge_batches_when_tone_annotated():
    m, a, b = _mem()
    calls_before = m.llm.calls
    labels = m.oblique_labels("mocking victory claims", [a, b])
    # batch (1) + distill (1) + SECOND OPINION on the single reject (1) —
    # not one call per candidate.
    assert m.llm.calls - calls_before <= 3
    assert labels[0] == "relevant"                   # judged by INTENDED gloss
    assert labels[1] == "irrelevant"                 # reject CONFIRMED per-item


def test_second_opinion_recovers_batch_false_reject():
    """A candidate the batch wrongly rejects is re-judged per-item and kept."""
    m, a, b = _mem()

    class _Flip:
        """Batch rejects EVERYTHING ; per-item recognises the ironic one."""
        def generate(self, prompt, max_tokens=90):
            if "ITEM :" in prompt:
                item = prompt.split("ITEM :", 1)[1]
                return ("VERDICT: RELEVANT" if "lol" in item.lower()
                        else "VERDICT: IRRELEVANT")
            if "Attitude (broad)" in prompt or "Request :" in prompt:
                return "the stance"
            out = []
            for ln in prompt.splitlines():
                s = ln.strip()
                if s.startswith("[") and "]" in s and s[1:s.index("]")].isdigit():
                    out.append(f"{s[1:s.index(']')]}: irrelevant")
            return "\n".join(out)
    m.llm = _Flip()
    labels = m.oblique_labels("mocking victory claims", [a, b])
    assert labels[0] == "relevant"                   # recovered by 2nd opinion
    assert labels[1] == "irrelevant"                 # confirmed reject


def test_judge_falls_back_per_item_without_tones():
    m = Memory(encoder=SimpleEncoder(), llm=_ToneLLM())   # tone_reading OFF
    a = m.ingest("we totally won lol", kind="FACT", id="A")
    labels = m.oblique_labels("anything", [a])
    assert labels == ["relevant"]                    # per-item VERDICT path


def test_semantic_search_matches_gloss_not_surface():
    m, a, b = _mem()
    # query phrased like the INTENDED meaning, token-disjoint from A's surface
    got = m.search_nodes("author mocks the claim of victory", mode="semantic",
                         on="content")
    assert got and got[0].id == "A"                  # found via its gloss
