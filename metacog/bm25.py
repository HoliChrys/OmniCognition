"""
Lightweight BM25 ranker — pure Python, no external dependency.

Standard BM25 formula (Robertson & Walker 1994) with the canonical
constants k1 = 1.2, b = 0.75. These are MATHEMATICAL properties of
the BM25 algorithm itself — the same values are used by Lucene,
Elasticsearch, rank_bm25, etc. — not metacog tuning knobs.

BM25 captures verbatim / rare-token matching (dates, proper names,
specific phrases) that complements the entity-level cosine on
keywords.
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence, Tuple


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
_K1 = 1.2
_B = 0.75


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def bm25_score(
    query: str,
    points: Sequence["Point"],  # noqa: F821
    *,
    k_pool: int = 14,
) -> List[Tuple[float, "Point"]]:  # noqa: F821
    """Score every point's content against the query and return the
    top `k_pool` by BM25 score. Points without content are skipped.
    """
    if not points:
        return []
    docs = [tokenize(p.content) for p in points]
    non_empty = sum(1 for d in docs if d)
    if non_empty == 0:
        return []
    avgdl = sum(len(d) for d in docs) / non_empty
    n_total = len(docs)
    q_tokens = tokenize(query)
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
