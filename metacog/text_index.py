"""
Text index — per-document token artifacts computed ONCE (Phase 2 of
docs/ingest_index_plan.md).

`search_nodes` re-tokenized the whole pool on EVERY call (× modes × rounds) and
`bm25_score` re-tokenized + re-stemmed every document on EVERY query. Both are
pure functions of the document — so they are computed once and memoized here :

  * `raw_tokens(point)`   the lowercase alnum token SET of the content
                          (exactly `re.findall(r"[a-z0-9]+", content.lower())` —
                          byte-for-byte what search_nodes' `_toks` produced).
  * `bm25_doc(point)`     the stemmed BM25 doc list (content[:40] + up to 8
                          keyword stems) — byte-for-byte what bm25_score built.
  * `postings`            raw token -> point ids, for candidate generation.

COMPUTE-ON-MISS : points are created in several places (ingest, hubs, walker
generations) — any point not yet indexed is indexed on first touch, so the
index is self-healing and equivalence with the naive scans is guaranteed by
construction. Content is immutable ; keywords MAY mutate (the document card
overwrites them), so the bm25 memo is keyed by the keyword tuple too.
Not pickled — `Memory.load()` starts empty and the index refills lazily.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Set, Tuple

from metacog.bm25 import stem_tokens, tokenize

_RAW_RE = re.compile(r"[a-z0-9]+")


def raw_token_set(text: str) -> FrozenSet[str]:
    """The exact token set search_nodes' `_toks` produced."""
    return frozenset(_RAW_RE.findall((text or "").lower()))


def bm25_doc_tokens(content: str, keywords) -> List[str]:
    """The exact stemmed doc list bm25_score built per call."""
    content_tokens = stem_tokens(tokenize(content or "")[:40])
    if keywords:
        kw_tokens = stem_tokens([kw.lower() for kw in keywords])
        content_set = set(content_tokens)
        extra = [t for t in kw_tokens if t not in content_set]
        return content_tokens + extra[:8]
    return content_tokens


class TextIndex:
    """Lazy, self-healing per-point token index (content immutable ; keyword-
    dependent artifacts re-memoized when keywords change)."""

    def __init__(self) -> None:
        self._raw: Dict[str, FrozenSet[str]] = {}
        self._bm25: Dict[str, Tuple[int, List[str]]] = {}   # id -> (kw_key, doc)
        self.postings: Dict[str, Set[str]] = {}

    def raw_tokens(self, point) -> FrozenSet[str]:
        toks = self._raw.get(point.id)
        if toks is None:
            toks = raw_token_set(point.content)
            self._raw[point.id] = toks
            for t in toks:
                self.postings.setdefault(t, set()).add(point.id)
        return toks

    def bm25_doc(self, point) -> List[str]:
        kw_key = hash(tuple(point.keywords or ()))
        hit = self._bm25.get(point.id)
        if hit is not None and hit[0] == kw_key:
            return hit[1]
        doc = bm25_doc_tokens(point.content, point.keywords)
        self._bm25[point.id] = (kw_key, doc)
        return doc

    def candidates(self, query_tokens, points) -> List:
        """Points sharing >= 1 raw token with the query — the exact set the
        naive sim scan scored > 0. Touches every point once (lazy index) then
        answers from postings."""
        for p in points:                          # ensure all indexed (lazy)
            self.raw_tokens(p)
        ids: Set[str] = set()
        for t in query_tokens:
            ids |= self.postings.get(t, set())
        return [p for p in points if p.id in ids]

    def discard(self, pid: str) -> None:
        toks = self._raw.pop(pid, None)
        if toks:
            for t in toks:
                s = self.postings.get(t)
                if s is not None:
                    s.discard(pid)
        self._bm25.pop(pid, None)
