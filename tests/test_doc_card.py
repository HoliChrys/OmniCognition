"""
Document Card (Phase 1) — ONE structured reading per document at ingestion.

The card replaces the separate keyword/entity/event/tone reads and pre-computes
the stance + answerable-questions artifacts. Failure-safe: a failed card falls
back to the individual extractors; an unparseable card is never cached.
Deterministic via scripted fake LLMs.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.doc_card import DocCardExtractor, _CARD_CACHE, _parse_card
from metacog.epistemic import PointKind, SourceClass
from metacog.memory import Memory

_CARD_RAW = """KEYWORDS: refinery, strike, victory
ENTITIES: iran|place ; bbc|organization
EVENT: the refinery strike | strike | 2022-03-01
TONE: ironic
INTENDED: the author mocks the claim of victory
STANCE: mocks official triumphalism about the strike
QUESTIONS: who mocked the victory claims ; was the strike a success"""


class _CardLLM:
    def __init__(self):
        self.calls = 0
    def generate(self, prompt, max_tokens=240):
        self.calls += 1
        if "TEXT :" in prompt:
            return _CARD_RAW
        return ""


def _mem(llm=None):
    llm = llm or _CardLLM()
    return Memory(encoder=SimpleEncoder(), llm=llm,
                  doc_card_extractor=DocCardExtractor(llm)), llm


def test_one_call_spawns_every_artifact():
    _CARD_CACHE.clear()
    m, llm = _mem()
    f = m.ingest("we totally won the refinery strike lol", kind="FACT", id="D1")
    assert llm.calls == 1                              # ONE reading
    # keywords enriched from the card, GENERATOR-sourced
    assert f.keywords[:3] == ["refinery", "strike", "victory"]
    assert f.keywords_source is SourceClass.GENERATOR
    # entity beacons
    assert any(p.id.startswith("entity_") and "iran" in p.content
               for p in m.points)
    # event hub + gravitation
    hub = next(p for p in m.points if p.kind is PointKind.EVENT)
    assert "event:type:strike" in hub.tags
    assert f"event:in:{hub.id}" in f.tags
    # tone tag + intended gloss
    assert "tone:ironic" in f.tags
    assert "mocks" in m.gloss_of("D1")
    # stance + question thoughts (later-phase consumers)
    assert any(p.id == "stance_D1" for p in m.points)
    assert sum(1 for p in m.points if p.id.startswith("dq_D1")) == 2


def test_card_cached_once_per_content():
    _CARD_CACHE.clear()
    m, llm = _mem()
    m.ingest("same text", kind="FACT", id="A")
    before = llm.calls
    m.ingest("same text", kind="FACT", id="B")
    assert llm.calls == before                         # cache hit


def test_unparseable_card_not_cached_and_falls_back():
    _CARD_CACHE.clear()

    class _Bad:
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, max_tokens=240):
            self.calls += 1
            return "garbage with no labelled lines"
    llm = _Bad()
    ex = DocCardExtractor(llm)
    assert ex.extract_card("text one") is None
    assert "text one" not in _CARD_CACHE              # never cache unparseable
    # Memory falls back : card fails -> tone hook still runs (here: no-op LLM)
    m = Memory(encoder=SimpleEncoder(), llm=llm, doc_card_extractor=ex,
               tone_reading=True)
    f = m.ingest("text two", kind="FACT", id="X")
    assert f.id == "X"                                 # ingestion never breaks


def test_parse_minimum_is_tone():
    assert _parse_card("KEYWORDS: a, b") is None       # no TONE -> invalid
    c = _parse_card("TONE: plain\nSTANCE: NONE")
    assert c is not None and c.tone == "plain" and c.stance == ""


def test_card_artifacts_are_derived_not_documents():
    _CARD_CACHE.clear()
    m, _ = _mem()
    m.ingest("we totally won the refinery strike lol", kind="FACT", id="D1")
    ids = {p.id for p in m.search_nodes("strike", mode="sim", on="content")}
    assert not any(i.startswith(("stance_", "dq_", "gloss_")) for i in ids)


def test_doc2query_semantic_match_via_question():
    """Phase 4 : a query phrased like an answerable QUESTION finds the doc even
    when sharing no tokens with its surface (and the stance view works too)."""
    _CARD_CACHE.clear()
    m, _ = _mem()
    m.ingest("we totally won the refinery strike lol", kind="FACT", id="D1")
    m.ingest("unrelated gardening notes about tulips", kind="FACT", id="D2")
    # query ≈ the generated question "who mocked the victory claims"
    got = m.search_nodes("who mocked the victory claims", mode="semantic",
                         on="content")
    assert got and got[0].id == "D1"            # found via its dq_ view
    # query ≈ the stance card "mocks official triumphalism about the strike"
    got2 = m.search_nodes("mocks official triumphalism", mode="semantic",
                          on="content")
    assert got2 and got2[0].id == "D1"          # found via its stance_ view
