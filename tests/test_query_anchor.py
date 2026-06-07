"""
Tests for the typed query-alignment anchor (Slice A) — the ColBERT-style
primitive consumed by the clue_search Rocchio re-rank (B) and the per-stage
walk anchor (C).

Deterministic : a fake entity extractor + keyword extractor + SimpleEncoder.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.query_anchor import (
    build_query_anchor, alignment_score, QueryAnchor,
)


class _Ent:
    etype: str
    value: str
    def __init__(self, t, v): self.etype, self.value = t, v


class _FakeEntityExtractor:
    def __init__(self, mapping): self.mapping = mapping
    def extract_entities(self, text):
        return self.mapping.get(text, [])


class _FakeKeywordExtractor:
    def __init__(self, mapping): self.mapping = mapping
    def extract(self, text, n=5): return self.mapping.get(text, [])[:n]


def _anchor(q, ents=None, kws=None):
    ee = _FakeEntityExtractor({q: ents or []})
    ke = _FakeKeywordExtractor({q: kws or []})
    return build_query_anchor(q, entity_extractor=ee, keyword_extractor=ke,
                              encoder=SimpleEncoder())


def test_entities_are_salient_topics_are_soft():
    a = _anchor(
        "What did Caroline research?",
        ents=[_Ent("person", "caroline"), _Ent("topic", "adoption")],
        kws=["research"],
    )
    assert "caroline" in a.salient
    assert "research" in a.salient        # keyword → salient
    assert "adoption" in a.soft           # topic → soft (semantic)


def test_exact_verb_stem_match_scores_high():
    """'Researching adoption agencies' must exact-stem-match the salient
    verb 'research' from 'What did Caroline research?'."""
    a = _anchor("What did Caroline research?",
                ents=[_Ent("person", "caroline")], kws=["research"])
    gold = alignment_score(a, "Researching adoption agencies for our family",
                           SimpleEncoder().encode("Researching adoption agencies"))
    off = alignment_score(a, "I love pottery and painting landscapes",
                          SimpleEncoder().encode("I love pottery and painting"))
    assert gold > off
    assert gold > 0.3          # salient stem 'resea' ⊆ 'researching'


def test_drift_turn_scores_lower_than_on_topic():
    """The counselling turn (co-present topic) must NOT outscore the actual
    research turn under the anchor."""
    a = _anchor("What did Caroline research?",
                ents=[_Ent("person", "caroline")], kws=["research"])
    enc = SimpleEncoder()
    research_turn = "Researching adoption agencies"
    counsel_turn = "I'm keen on counseling or working in mental health"
    s_res = alignment_score(a, research_turn, enc.encode(research_turn))
    s_cou = alignment_score(a, counsel_turn, enc.encode(counsel_turn))
    assert s_res > s_cou


def test_empty_anchor_scores_zero():
    a = QueryAnchor()
    assert a.is_empty()
    assert alignment_score(a, "anything", SimpleEncoder().encode("anything")) == 0.0


def test_score_bounded_unit_interval():
    a = _anchor("What did Caroline research?",
                ents=[_Ent("person", "caroline")], kws=["research"])
    enc = SimpleEncoder()
    for txt in ["Researching adoption agencies caroline", "random unrelated text",
                "caroline caroline research research"]:
        s = alignment_score(a, txt, enc.encode(txt))
        assert 0.0 <= s <= 1.0


def test_maxsim_not_diluted_by_long_turn():
    """ColBERT MaxSim takes the BEST token match — a salient term present in
    a long, otherwise-unrelated turn must still score high (turn-level mean
    cosine would dilute it)."""
    a = _anchor("What did Caroline research?",
                ents=[_Ent("person", "caroline")], kws=["research"])
    enc = SimpleEncoder()
    short = "Researching adoption"
    long = ("Researching adoption agencies and also we talked about pottery "
            "and painting and camping and the weather and food and music "
            "and a hundred other unrelated things that day honestly")
    s_short = alignment_score(a, short, enc.encode(short))
    s_long = alignment_score(a, long, enc.encode(long))
    # exact stem hit on 'research' fires regardless of turn length
    assert s_long >= 0.3 and abs(s_long - s_short) < 0.5


def test_relative_thought_channel_added_to_input():
    """The anchor is INPUT + RELATIVE-thought, ADDED. The input alone cannot
    separate an oblique evidence turn from an ambient one ; once a thought
    abstracts to the answer register, `with_relative` adds a semantic channel
    that lifts the oblique turn above the ambient one — the inference bridge.
    Uses lexical-free terms so only the relative cosine can do the work."""
    enc = SimpleEncoder()
    # input with NO lexical overlap with either candidate (isolate relative).
    a = build_query_anchor("suspected health problems", encoder=enc)
    oblique = "my fingers are too big, maybe take up exercise and go for a run"
    ambient = "gaming helps me escape the stress of a heavy workload"

    # before any thought : input semantic OFF, no relative -> no separation.
    s_obl0 = alignment_score(a, oblique, enc.encode(oblique),
                             use_input_sem=False)
    s_amb0 = alignment_score(a, ambient, enc.encode(ambient),
                             use_input_sem=False)
    assert s_obl0 == s_amb0     # nothing to separate them yet

    # thought abstracts to the answer register -> added to the anchor.
    a.with_relative(["overweight", "exercise", "weight"], enc)
    assert a.relative and a.relative_emb is not None
    s_obl1 = alignment_score(a, oblique, enc.encode(oblique),
                             use_input_sem=False)
    s_amb1 = alignment_score(a, ambient, enc.encode(ambient),
                             use_input_sem=False)
    assert s_obl1 > s_amb1      # the bridge ranks the oblique evidence first


def test_with_relative_drops_terms_already_in_input():
    """A thought keyword already covered by the input's salient set adds
    nothing (would double-count) and must be filtered out."""
    a = _anchor("What did Caroline research?",
                ents=[_Ent("person", "caroline")], kws=["research"])
    a.with_relative(["research", "adoption"], SimpleEncoder())
    assert "research" not in a.relative      # already a salient input term
    assert "adoption" in a.relative


def test_with_relative_failsafe_no_encoder():
    a = _anchor("What did Caroline research?", kws=["research"])
    a.with_relative(["adoption"], encoder=None)
    assert a.relative == ["adoption"]
    assert a.relative_emb is None            # no encoder -> no embedding, no raise


def test_degrades_without_extractors():
    """No entity/keyword extractor → falls back to raw content tokens (soft),
    still usable, never raises."""
    a = build_query_anchor("What did Caroline research?", encoder=SimpleEncoder())
    assert not a.is_empty()
    assert "caroline" in a.soft and "research" in a.soft
