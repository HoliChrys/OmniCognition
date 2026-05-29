"""
Keyword extractors for the hybrid retrieval pipeline.

Keywords are populated at ingest time and used by retrieve_hybrid to
compute cosine similarity at the entity level — distinct from BM25 on
full text (verbatim match) and from cosine on the full semantic
embedding.

Cor. 5 : if an LLM extractor is used, its output is tagged with
keywords_source=GENERATOR for audit but never enters the observation
set O.

Two implementations :

  SimpleKeywordExtractor   pure COMPUTATION, no LLM, no dep
                           frequency-based, stopword-filtered
                           good baseline, misses rare entities

  LLMKeywordExtractor      GENERATOR, uses Claude API
                           catches proper nouns, dates, named entities
                           cached in memory to amortize cost
                           requires ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Protocol

from metacog.epistemic import SourceClass


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")


_BASIC_STOPWORDS = frozenset({
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "with", "on", "at", "by", "from", "into",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "and", "or", "but", "not", "no", "so", "as", "if", "when",
    "where", "who", "what", "how", "why", "than", "then", "there",
    "their", "them", "his", "her", "its", "our", "your", "do", "does",
    "did", "done", "have", "has", "had", "having", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can",
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "à", "en",
    "pour", "par", "avec", "sans", "sur", "sous", "et", "ou", "ni",
    "mais", "donc", "or", "car", "ne", "pas", "non", "que", "qui",
    "quoi", "où", "comment", "pourquoi", "je", "tu", "il", "elle",
    "nous", "vous", "ils", "elles", "ce", "ça", "cette", "ces",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "est", "était", "étaient", "sont", "fut", "été",
})


class KeywordExtractor(Protocol):
    """Protocol for keyword extractors injected into Memory."""

    def extract(self, text: str, n: int = 5) -> List[str]: ...


class SimpleKeywordExtractor:
    """Frequency-based extractor : top-N content words after stopword
    removal, ranked by (frequency desc, length desc) for stable
    ordering. Pure COMPUTATION, no LLM, no external dependency.
    """

    source: SourceClass = SourceClass.COMPUTATION

    def __init__(
        self,
        stopwords: frozenset = _BASIC_STOPWORDS,
        min_length: int = 3,
    ) -> None:
        self.stopwords = stopwords
        self.min_length = min_length

    def extract(self, text: str, n: int = 5) -> List[str]:
        tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
        tokens = [
            t for t in tokens
            if t not in self.stopwords and len(t) >= self.min_length
        ]
        if not tokens:
            return []
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        # Sort by (-frequency, -length, alphabetical) for determinism
        ordered = sorted(
            freq.items(),
            key=lambda x: (-x[1], -len(x[0]), x[0]),
        )
        return [w for w, _ in ordered[:n]]


class LLMKeywordExtractor:
    """Claude-API-backed keyword extractor.

    Targets the same entities a human would tag : proper nouns, dates,
    rare named entities. Tagged GENERATOR for audit (Cor. 5) — the
    output is used as INDEXING data (drives keywords_embedding for
    cosine retrieval) but never enters the observation set O.

    Caches results in memory : ingesting the same text twice costs
    one API call.

    Default model : claude-haiku-4-5-20251001 (cheap + fast).
    Override via CLAUDE_KW_MODEL env or constructor kwarg.
    """

    source: SourceClass = SourceClass.GENERATOR

    _SYSTEM = (
        "Extract the most informative keywords from the text. "
        "Prefer proper nouns, dates, entities, rare terms. "
        "Skip stopwords and generic words. "
        "Return ONLY a comma-separated list, lowercase, no prose, no "
        "code fences. Output at most the number of items requested."
    )

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 64,
        api_key: Optional[str] = None,
    ) -> None:
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.model = model or os.environ.get(
            "CLAUDE_KW_MODEL", "claude-haiku-4-5-20251001",
        )
        self.max_tokens = max_tokens
        self._cache: dict[str, List[str]] = {}

    def extract(self, text: str, n: int = 5) -> List[str]:
        if not text:
            return []
        cache_key = f"{n}::{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        user_msg = (
            f"Extract up to {n} keywords from this text:\n{text}\n\n"
            f"Output: lowercase comma-separated, max {n} items."
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text if resp.content else ""
        except Exception:
            # If the LLM call fails (no API key, rate limit), fall back
            # to silence — the system stays usable, just with no
            # LLM-extracted keywords for this chunk.
            return []

        # Strip code fences if any, then split on commas
        raw = raw.strip()
        m = re.search(r"```(?:\w+)?\s*(.+?)\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        kws = [w.strip().lower() for w in raw.split(",") if w.strip()]
        # Drop trailing punctuation
        kws = [w.rstrip(".;:!?") for w in kws if w.rstrip(".;:!?")]
        kws = kws[:n]
        self._cache[cache_key] = kws
        return kws
