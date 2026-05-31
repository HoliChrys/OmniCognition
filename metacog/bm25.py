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
) -> List[Tuple[float, "Point"]]:  # noqa: F821
    """Score points against the query and return the top k_pool by BM25.

    Indexes on point KEYWORDS (already extracted, stop-word-free) with
    minimal suffix-stripping on both sides so that morphological
    variants (researching ↔ research, studied ↔ study) are matched.

    query_keywords — if provided, use these pre-extracted keywords as the
    query token set (symmetric with the points' keywords).  When absent,
    tokenize query directly (used as a fallback in the meta-walk).
    """
    if not points:
        return []

    # Build stemmed keyword-based documents for each point.
    docs: List[List[str]] = []
    for p in points:
        if p.keywords:
            docs.append(stem_tokens([kw.lower() for kw in p.keywords]))
        else:
            # Fallback for points with no keywords: tokenize short content.
            docs.append(stem_tokens(tokenize(p.content or "")[:20]))

    non_empty = sum(1 for d in docs if d)
    if non_empty == 0:
        return []
    avgdl = sum(len(d) for d in docs) / non_empty
    n_total = len(docs)

    if query_keywords:
        q_tokens = stem_tokens([kw.lower() for kw in query_keywords])
    else:
        q_tokens = stem_tokens(tokenize(query))
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
