"""
Semantic tag index — embed each hierarchical-tag SEGMENT, search by meaning.

`metacog.tags` matches tags lexically : exact, fuzzy (Levenshtein), regex. That
covers typos and prefixes but not MEANING : a query "weight issue" never finds
the namespace `health:condition:macrodactyly`, and "doctor" never finds
`health:condition:fatigue`, because they share no characters.

This module adds the missing semantic channel WITHOUT bloating every tag into a
vector. The unit indexed is the SEGMENT — each `:`-separated word of a
hierarchical tag, taken solo (`health:condition:macrodactyly` ->
`health`, `condition`, `macrodactyly`). Segments are few, reused across many
tags, and individually meaningful, so:

  1. embed every distinct segment once (cached) ;
  2. `nearest_segments(query)` returns the segments closest in meaning ;
  3. `locate(segment)` greps the STRUCTURED glossary for the full namespaces
     that segment appears in — so a semantic hit on `condition` resurfaces
     `health:condition`, `building:condition`, … to scope a presearch on.

Stage 1 is semantic (embedding), stage 3 is structural (grep/fuzzy over the
glossary) — exactly the "find the nearest tag, then look where it sits in the
structured glossary" flow. Encoder-agnostic (any `.encode(str)->vec`), fully
failure-safe, embeddings cached across calls.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from metacog.tags import tag_glossary


def tag_segments(points: Any, *, include_flat: bool = True) -> List[str]:
    """Every distinct `:`-segment across all tags on `points`, order-stable.

    `health:condition:macrodactyly` contributes health, condition,
    macrodactyly ; a flat tag (`fact`) contributes itself when
    `include_flat`. Pure structural / kind tags are kept — the caller decides
    what to query ; this is just the segment vocabulary."""
    seen: set = set()
    out: List[str] = []
    for p in points:
        for t in getattr(p, "tags", None) or []:
            if not isinstance(t, str) or not t:
                continue
            if ":" in t:
                segs = t.split(":")
            elif include_flat:
                segs = [t]
            else:
                continue
            for s in segs:
                s = s.strip().lower()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    if a is None or b is None:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


class TagIndex:
    """Semantic index over hierarchical-tag segments.

    Cheap to keep around : it embeds only the (small) segment vocabulary and
    caches every vector, so rebuilding after new tags appear re-encodes just
    the new segments."""

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self._emb: Dict[str, tuple] = {}        # segment -> embedding (cache)
        self._segments: List[str] = []

    def build(self, points: Any) -> "TagIndex":
        """(Re)build the segment vocabulary from `points`, encoding only the
        segments not already cached. Returns self."""
        self._segments = tag_segments(points)
        for s in self._segments:
            if s not in self._emb:
                try:
                    self._emb[s] = tuple(self.encoder.encode(s))
                except Exception:
                    self._emb[s] = None
        return self

    def nearest_segments(
        self, query: str, k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Top-`k` tag segments closest in meaning to `query` (cosine)."""
        q = (query or "").strip()
        if not q or not self._segments:
            return []
        try:
            qv = tuple(self.encoder.encode(q))
        except Exception:
            return []
        scored = [
            (s, _cos(qv, self._emb.get(s)))
            for s in self._segments if self._emb.get(s) is not None
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def locate(self, segment: str, points: Any) -> List[str]:
        """Where a `segment` sits in the structured glossary — the "grep"
        stage. Returns the glossary NAMESPACES (parent prefixes) whose own
        `:`-segments include `segment` first (the concrete scope filters —
        `health:condition`, `building:condition`), then any FULL tags that
        contain it but whose namespace prefix did not already cover it (so a
        leaf hit on `fatigue` still resolves to `health:condition:fatigue`).
        Order-stable, deduped."""
        seg = (segment or "").strip().lower()
        if not seg:
            return []
        out: List[str] = []
        for ns in tag_glossary(points):
            if seg in ns.split(":") and ns not in out:
                out.append(ns)
        # full tags carrying the segment, not already implied by a namespace
        covered = set(out)
        for p in points:
            for t in getattr(p, "tags", None) or []:
                if not isinstance(t, str) or seg not in t.split(":"):
                    continue
                if t in covered:
                    continue
                if any(t == ns or t.startswith(ns + ":") for ns in out):
                    continue
                out.append(t)
                covered.add(t)
        return out

    def search(
        self, query: str, points: Any, *, k: int = 5,
    ) -> List[Dict[str, Any]]:
        """End-to-end : nearest segments to `query`, each expanded to the
        glossary namespaces it appears in (semantic find -> structural locate).

        Returns [{"segment", "score", "namespaces"}], best first — a ranked
        set of concrete scope filters for a presearch / scoped_answer."""
        out: List[Dict[str, Any]] = []
        for seg, score in self.nearest_segments(query, k=k):
            out.append({
                "segment": seg,
                "score": score,
                "namespaces": self.locate(seg, points),
            })
        return out
