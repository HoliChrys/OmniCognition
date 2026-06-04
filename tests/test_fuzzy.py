"""
Tests for the Levenshtein fuzzy lexical recall signal.

Properties :
  F1  levenshtein is the standard edit distance
  F2  edit budget is length-relative (floor(len/4)) — parameter-free
  F3  fuzzy_match accepts spelling variants of rare entities, rejects
      unrelated short tokens
  F4  fuzzy_score ranks points by count of matched query tokens and
      excludes zero-match points
"""

from __future__ import annotations

from metacog import Memory, SimpleKeywordExtractor
from metacog.fuzzy import fuzzy_match, fuzzy_score, levenshtein


# ---------- F1 -----------------------------------------------------------


def test_levenshtein_basic():
    assert levenshtein("berkeley", "berkley") == 1
    assert levenshtein("caroline", "carolyn") == 2
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3


# ---------- F2, F3 -------------------------------------------------------


def test_fuzzy_match_accepts_entity_variants():
    assert fuzzy_match("berkeley", "berkley")     # 1 edit, budget 2
    assert fuzzy_match("caroline", "carolyn")     # 2 edits, budget 2
    assert fuzzy_match("necklace", "necklaces")   # plural, 1 edit


def test_fuzzy_match_rejects_unrelated_and_short():
    assert not fuzzy_match("uk", "us")            # len 2 → budget 0
    assert not fuzzy_match("paris", "tokyo")
    assert not fuzzy_match("cat", "dog")


# ---------- F4 -----------------------------------------------------------


def test_fuzzy_score_ranks_by_matched_tokens():
    m = Memory(extractor=SimpleKeywordExtractor())
    m.ingest("caroline visited berkley university campus", id="A")  # 2 variants
    m.ingest("caroline went home", id="B")                         # 1 variant
    m.ingest("totally unrelated weather report", id="C")           # 0
    ranked = fuzzy_score(
        "caroline berkeley university", m.points, k_pool=5,
    )
    ids = [p.id for _s, p in ranked]
    assert "A" in ids and "B" in ids
    assert "C" not in ids          # zero fuzzy matches excluded
    assert ids[0] == "A"           # most matched query tokens first
