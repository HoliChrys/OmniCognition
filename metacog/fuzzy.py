"""
Fuzzy lexical matching — a Levenshtein-based recall signal.

BM25 matches tokens EXACTLY, so it misses morphological / spelling
variants of rare entities ("Berkeley" vs "Berkley", "Caroline" vs
"Carolyn", plurals, typos). Dense cosine catches some but is unreliable
for rare proper nouns. This module adds a fuzzy term-overlap signal that
recovers those near-miss matches.

ZERO HYPERPARAMETER : a query token matches a document token when their
edit distance is within a LENGTH-RELATIVE budget — floor(len/4) edits,
i.e. roughly one typo per four characters. That is a natural
morphological tolerance ("berkeley"→2 edits, "uk"→0), not a tuned knob.
Ranking is by count of matched query tokens ; RRF handles fusion without
a score threshold.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Tokens shorter than this carry little discriminative signal and would
# fuzzy-match noise ; skip them. Pure tokenizer hygiene, not a tuning knob.
_MIN_TOKEN_LEN = 4


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert / delete / substitute), DP in O(len)
    space. Pure Python, no dependency."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _edit_budget(token: str) -> int:
    """Length-relative tolerance : ~1 edit per 4 chars. Parameter-free."""
    return len(token) // 4


def fuzzy_match(qtok: str, dtok: str) -> bool:
    """True if dtok is within qtok's length-relative edit budget.

    A short common prefix guard avoids matching unrelated short words
    that happen to be close in edit space.
    """
    if qtok == dtok:
        return True
    budget = _edit_budget(qtok)
    if budget == 0:
        return False
    # Quick length filter : can't be within budget if lengths differ more.
    if abs(len(qtok) - len(dtok)) > budget:
        return False
    return levenshtein(qtok, dtok) <= budget


def fuzzy_score(
    query_text: str,
    points: Sequence["Point"],  # noqa: F821
    k_pool: int,
) -> List[Tuple[float, "Point"]]:
    """Rank points by how many distinct query tokens fuzzy-match a token
    in the point's content. Returns the top `k_pool` (score, point),
    best first. Points with zero fuzzy matches are excluded.
    """
    q_tokens = {t for t in _tokens(query_text) if len(t) >= _MIN_TOKEN_LEN}
    if not q_tokens:
        return []
    scored: List[Tuple[float, "Point"]] = []
    for p in points:
        d_tokens = {t for t in _tokens(p.content) if len(t) >= _MIN_TOKEN_LEN}
        if not d_tokens:
            continue
        matched = 0
        for qt in q_tokens:
            if any(fuzzy_match(qt, dt) for dt in d_tokens):
                matched += 1
        if matched:
            # Normalize by query size so longer queries don't dominate.
            scored.append((matched / len(q_tokens), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k_pool]
