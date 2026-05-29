"""
Tests for LLMKeywordExtractor — Claude-API-backed keyword extraction.

Properties :
  K1   LLMKeywordExtractor.source is GENERATOR
  K2   SimpleKeywordExtractor.source is COMPUTATION
  K3   extract returns a list of lowercase strings
  K4   parses comma-separated output
  K5   strips code fences and trailing punctuation
  K6   caches identical inputs (1 API call → multiple extract() reuses)
  K7   gracefully returns [] when the API call raises
  K8   Memory.ingest tags Point.keywords_source per extractor.source
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from metacog import (
    LLMKeywordExtractor,
    Memory,
    SimpleKeywordExtractor,
    SourceClass,
)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [MagicMock(text=text)]


def _make_extractor(reply: str) -> LLMKeywordExtractor:
    with patch("anthropic.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = _FakeResponse(reply)
        mock_cls.return_value = client
        ext = LLMKeywordExtractor(api_key="test")
    # Swap in the mocked client after construction
    ext.client = client
    return ext


# ---------- K1, K2 --------------------------------------------------------


def test_llm_extractor_source_is_generator():
    assert LLMKeywordExtractor.source == SourceClass.GENERATOR


def test_simple_extractor_source_is_computation():
    assert SimpleKeywordExtractor.source == SourceClass.COMPUTATION


# ---------- K3, K4 --------------------------------------------------------


def test_extract_parses_comma_separated():
    ext = _make_extractor("sweden, grandma, necklace, caroline, gift")
    kws = ext.extract("text about a necklace from Sweden", n=5)
    assert kws == ["sweden", "grandma", "necklace", "caroline", "gift"]


def test_extract_lowercases():
    ext = _make_extractor("Sweden, GRANDMA, Caroline")
    kws = ext.extract("about sweden", n=5)
    assert kws == ["sweden", "grandma", "caroline"]


def test_extract_respects_n_cap():
    ext = _make_extractor("a, b, c, d, e, f, g")
    kws = ext.extract("anything", n=3)
    assert kws == ["a", "b", "c"]


# ---------- K5 ------------------------------------------------------------


def test_extract_strips_code_fences():
    ext = _make_extractor("```\nsweden, grandma, necklace\n```")
    kws = ext.extract("anything", n=5)
    assert kws == ["sweden", "grandma", "necklace"]


def test_extract_strips_trailing_punctuation():
    ext = _make_extractor("sweden., grandma!, necklace?")
    kws = ext.extract("anything", n=5)
    assert kws == ["sweden", "grandma", "necklace"]


# ---------- K6 ------------------------------------------------------------


def test_extract_caches_identical_inputs():
    ext = _make_extractor("sweden, grandma")
    text = "the necklace from sweden"
    ext.extract(text, n=5)
    ext.extract(text, n=5)
    ext.extract(text, n=5)
    # client.messages.create should only have been called once
    assert ext.client.messages.create.call_count == 1


def test_cache_keyed_by_n():
    ext = _make_extractor("sweden, grandma")
    ext.extract("text", n=5)
    ext.extract("text", n=3)
    # Different n → different cache key → two API calls
    assert ext.client.messages.create.call_count == 2


# ---------- K7 ------------------------------------------------------------


def test_extract_returns_empty_on_api_failure():
    with patch("anthropic.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.side_effect = Exception("rate limit")
        mock_cls.return_value = client
        ext = LLMKeywordExtractor(api_key="test")
    ext.client.messages.create.side_effect = Exception("rate limit")
    kws = ext.extract("anything", n=5)
    assert kws == []


# ---------- K8 ------------------------------------------------------------


def test_memory_ingest_tags_keywords_source_computation():
    m = Memory(extractor=SimpleKeywordExtractor())
    p = m.ingest("alpha beta gamma delta", id="P1")
    assert p.keywords_source == SourceClass.COMPUTATION


def test_memory_ingest_tags_keywords_source_generator():
    ext = _make_extractor("sweden, grandma, caroline")
    m = Memory(extractor=ext)
    p = m.ingest("the necklace from sweden", id="P1")
    assert p.keywords_source == SourceClass.GENERATOR
    assert "sweden" in p.keywords


def test_memory_ingest_handles_no_extractor():
    m = Memory(extractor=None)
    p = m.ingest("alpha beta gamma", id="P1")
    assert p.keywords == []
    assert p.keywords_source is None
