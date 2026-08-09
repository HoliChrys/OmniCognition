"""
Lightweight BM25 ranker — pure Python, no external dependency.

Standard BM25 formula (Robertson & Walker 1994) with the canonical
constants k1 = 1.2, b = 0.75. These are MATHEMATICAL properties of
the BM25 algorithm itself — the same values are used by Lucene,
Elasticsearch, rank_bm25, etc. — not metacog tuning knobs.

BM25 operates on point KEYWORDS (not raw content) to avoid stop-word
contamination from conversation structure (question words, speaker
prefixes, dates). A minimal suffix-strip handles the most common
English morphological variants (researching → research, studied →
study) without introducing a stemming hyperparameter.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
_K1 = 1.2
_B = 0.75

# Suffix strips in priority order; min_stem = minimum chars left after
# stripping. Strips only unambiguous morphological endings to avoid
# over-generalization. No hyperparameter: the rule list is fixed.
_SUFFIXES: List[Tuple[str, int]] = [
    ("ings", 4), ("ing", 4),
    ("tions", 5), ("tion", 5),
    ("ness", 4), ("ities", 5), ("ity", 4),
    ("able", 5), ("ible", 5),
    ("edly", 4), ("edly", 4), ("ed", 4),
    ("iest", 4), ("ier", 4), ("ies", 4),
    ("ers", 4), ("er", 4),
    ("est", 4),
    ("ly", 4),
    ("s", 4),
]


def _stem(word: str) -> str:
    """Minimal suffix-strip. Returns the shortest stem ≥ min_stem chars."""
    for suffix, min_stem in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def stem_tokens(tokens: List[str]) -> List[str]:
    return [_stem(t) for t in tokens]


def bm25_score(
    query: str,
    points: Sequence["Point"],  # noqa: F821
    *,
    k_pool: int = 14,
    query_keywords: Optional[List[str]] = None,
    text_index=None,
) -> List[Tuple[float, "Point"]]:  # noqa: F821
    """Score points against the query and return the top k_pool by BM25.

    Indexes on raw content tokens — BM25 is the pure lexical channel and
    must always run on the actual text, not on the keyword summary.
    Keywords are for the semantic embedding channel; BM25 handles verbatim
    matches, rare tokens, abbreviations (VR, AI, TV), and proper nouns
    that the keyword extractor may filter or miss.

    query_keywords — when provided, also appended to the raw query tokens
    so the two channels reinforce each other on stemmed variants.
    """
    if not points:
        return []

    # Index on raw content: always content-first, keywords appended for
    # morphological coverage (stem variants the content tokeniser misses).
    # With a `text_index` (Phase 2) the per-doc token lists are memoized at
    # first touch instead of re-tokenized+re-stemmed on every query — the
    # produced lists are byte-for-byte identical (see text_index.bm25_doc).
    docs: List[List[str]] = []
    if text_index is not None:
        for p in points:
            docs.append(text_index.bm25_doc(p))
    else:
        for p in points:
            content_tokens = stem_tokens(tokenize(p.content or "")[:40])
            if p.keywords:
                kw_tokens = stem_tokens([kw.lower() for kw in p.keywords])
                content_set = set(content_tokens)
                extra_kw = [t for t in kw_tokens if t not in content_set]
                docs.append(content_tokens + extra_kw[:8])
            else:
                docs.append(content_tokens)

    non_empty = sum(1 for d in docs if d)
    if non_empty == 0:
        return []
    avgdl = sum(len(d) for d in docs) / non_empty
    n_total = len(docs)

    # Always start from raw query tokens; append extracted keywords for
    # stemmed-variant coverage without losing verbatim terms.
    q_tokens = stem_tokens(tokenize(query))
    if query_keywords:
        kw_set = set(q_tokens)
        q_tokens = q_tokens + [t for t in stem_tokens([k.lower() for k in query_keywords])
                                if t not in kw_set]
    if not q_tokens:
        return []

    idfs: dict[str, float] = {}
    for q in set(q_tokens):
        n_qi = sum(1 for d in docs if q in d)
        idfs[q] = math.log((n_total - n_qi + 0.5) / (n_qi + 0.5) + 1.0)

    scores: List[Tuple[float, "Point"]] = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        dl = len(doc)
        s = 0.0
        for q in q_tokens:
            tf = doc.count(q)
            if tf == 0:
                continue
            num = idfs[q] * tf * (_K1 + 1)
            denom = tf + _K1 * (1 - _B + _B * dl / avgdl)
            s += num / denom
        if s > 0:
            scores.append((s, points[i]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:k_pool]
