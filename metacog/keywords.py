"""
Keyword extractors for the hybrid retrieval pipeline.

Keywords are populated at ingest time and used by retrieve_hybrid to
compute cosine similarity at the entity level — distinct from BM25 on
full text (verbatim match) and from cosine on the full semantic
embedding.

Cor. 5 : if an LLM extractor is used, its output is tagged with
keywords_source=GENERATOR for audit but never enters the observation
set O.
"""

from __future__ import annotations

import re
from typing import List, Protocol


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
