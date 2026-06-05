"""
Tag namespace tooling — parent-prefix glossary and tri-modal matching.

Tags are an open-vocabulary list normalised lowercase on every Point
(see `epistemic.Point.__post_init__`). Several conventions in the
codebase are *hierarchical* with `:` as the separator:

    ref:date:2022      ref:skill:plot:path:src/plot.py
    session:s1         user:u42        person:melanie

The LEAF (last `:`-separated segment) is a specific value — `2022`,
`s1`, `melanie`. The PARENT PREFIXES are the namespace keys an agent
actually wants to discover, type a query against, and combine:
`ref`, `ref:date`, `ref:skill`, `ref:skill:plot`, `session`, `user`,
`person`. A glossary of those prefixes is far smaller and more useful
than the raw tag set (which mixes namespace keys with their values).

This module exposes three things:

  • `parent_prefixes(tag)` — drop the leaf, keep every parent prefix.
  • `tag_glossary(points)` — aggregate parent prefixes across a Point
    collection, ordered by hierarchy depth then alphabetically. This
    is the agent-facing namespace catalogue.
  • `match_tags(point, patterns, mode)` — exact / fuzzy / regex tag
    matching, including the hierarchical-prefix conventions, so a
    `scoped_answer` / pre-search / walk filter can be expressed in any
    of those modes without changes at every call site.

Fuzzy matching reuses the zero-dep Levenshtein in `metacog.fuzzy`
(length-relative edit budget, no hyper-parameter).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set, Tuple

from metacog.fuzzy import fuzzy_match


# ---------------------------------------------------------------------------
# Hierarchical parsing
# ---------------------------------------------------------------------------


def parent_prefixes(tag: str) -> List[str]:
    """Return every parent prefix of a hierarchical tag, leaf removed.

    Hierarchical tags use `:` as the separator (`ref:date:2022`,
    `ref:skill:plot:path:src/plot.py`). The LEAF — the rightmost
    segment — is the concrete value; everything to its left is a
    namespace prefix. We keep ALL parent prefixes (every ancestor),
    not just the immediate parent, so a query against either `ref`
    or `ref:date` resolves the same `ref:date:2022` tag.

        parent_prefixes("ref:date:2022")    # -> ["ref", "ref:date"]
        parent_prefixes("session:s1")       # -> ["session"]
        parent_prefixes("fact")              # -> []   (no hierarchy)
        parent_prefixes("")                  # -> []

    The original tag is NEVER returned (callers compose it explicitly
    if they want it included).
    """
    if not tag or ":" not in tag:
        return []
    segs = tag.split(":")
    out: List[str] = []
    for i in range(1, len(segs)):
        out.append(":".join(segs[:i]))
    return out


def is_hierarchical(tag: str) -> bool:
    """True if the tag carries at least one namespace segment (`:`)."""
    return bool(tag) and ":" in tag


def leaf(tag: str) -> str:
    """The rightmost segment of a hierarchical tag (the concrete value)."""
    if not tag or ":" not in tag:
        return tag or ""
    return tag.rsplit(":", 1)[1]


# ---------------------------------------------------------------------------
# Glossary aggregation
# ---------------------------------------------------------------------------


def tag_glossary(
    points: Iterable, *, include_flat: bool = True,
) -> List[str]:
    """Return the namespace catalogue across `points`.

    For every hierarchical tag carried by any Point, every parent
    prefix is added to the glossary. Optionally (`include_flat=True`,
    the default) FLAT tags (no `:`) are included verbatim — they ARE
    namespaces in their own right (`fact`, `thought`, `tool`).

    The glossary is ordered DETERMINISTICALLY by hierarchy depth (the
    number of `:` separators, shallowest first) and, within each depth,
    alphabetically. This is the structural order an agent wants when
    discovering available filters: top-level namespaces first, then
    their children.

        Memory with tags {ref:date:2022, ref:skill:plot, session:s1, fact}
        glossary -> ["fact", "ref", "session", "ref:date", "ref:skill"]
    """
    bag: Set[str] = set()
    for p in points:
        for t in getattr(p, "tags", None) or []:
            if not isinstance(t, str) or not t:
                continue
            if is_hierarchical(t):
                for pref in parent_prefixes(t):
                    bag.add(pref)
            elif include_flat:
                bag.add(t)
    return sorted(bag, key=lambda s: (s.count(":"), s))


# ---------------------------------------------------------------------------
# Tri-modal tag matching
# ---------------------------------------------------------------------------


# A glossary prefix is the namespace key (`ref:date`). A point carrying
# `ref:date:2022` should be considered to "have" the namespace `ref:date`.
# `_match_one` encodes that hierarchy-aware semantic on top of the three
# match modes (exact / fuzzy / regex). Exact = string equality OR
# hierarchical ancestry; fuzzy = lev distance on the segments; regex =
# `re.search` on every tag (raw, unmodified).
_VALID_MODES = ("exact", "fuzzy", "regex")


def _ancestor_of(needle: str, tag: str) -> bool:
    """True if `needle` IS `tag` or is one of its parent prefixes.

    Lets a filter on `ref:date` match a Point carrying `ref:date:2022`.
    """
    if needle == tag:
        return True
    return is_hierarchical(tag) and needle in parent_prefixes(tag)


def _fuzzy_ancestor(needle: str, tag: str) -> bool:
    """Fuzzy version of `_ancestor_of`.

    Each `:`-segment of the needle must fuzzy-match (Levenshtein
    length-budget) the corresponding segment of the tag from the left.
    `ref:dat` would still match `ref:date:2022` (one typo per 4 chars).
    The needle is allowed to be SHORTER than the tag (hierarchical
    ancestry) but not longer (you cannot fuzzy-match a deeper namespace
    than what the tag carries).
    """
    n_segs = needle.split(":")
    t_segs = tag.split(":")
    if len(n_segs) > len(t_segs):
        return False
    return all(fuzzy_match(n, t) for n, t in zip(n_segs, t_segs))


def match_tag(
    point_tags: Sequence[str], pattern: str, *, mode: str = "exact",
) -> bool:
    """Return True if `pattern` matches any tag on `point_tags` under `mode`.

    Modes:
      exact  — equality OR hierarchical ancestry (default; the old
               behaviour `pattern in point.tags` is a subset of this).
      fuzzy  — segment-wise Levenshtein, allows typos / morphological
               variants on hierarchical needles.
      regex  — `re.search(pattern, tag)` on every raw tag. Lets a caller
               express e.g. `r"^ref:date:202[0-9]$"` or `r"session:s\\d+"`.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown match mode {mode!r}; expected one of {_VALID_MODES}")
    if not pattern:
        return False
    if mode == "regex":
        try:
            rx = re.compile(pattern, flags=re.IGNORECASE)
        except re.error:
            return False
        return any(rx.search(t) for t in point_tags)
    needle = pattern.strip().lower()
    if not needle:
        return False
    if mode == "exact":
        return any(_ancestor_of(needle, t) for t in point_tags)
    # fuzzy
    return any(_fuzzy_ancestor(needle, t) for t in point_tags)


def match_tags_all(
    point_tags: Sequence[str], patterns: Sequence[str], *, mode: str = "exact",
) -> bool:
    """True if EVERY pattern matches the Point under `mode` (AND semantics).

    Mirrors the existing `tagset.issubset(point.tags)` used by
    `scoped_answer` — a Point must satisfy every requested tag.
    """
    if not patterns:
        return True
    return all(match_tag(point_tags, p, mode=mode) for p in patterns)


def filter_points(
    points: Iterable, patterns: Sequence[str], *, mode: str = "exact",
) -> Set[str]:
    """Return the set of Point ids satisfying ALL `patterns` under `mode`.

    Drop-in replacement for `scoped_answer`'s
    `{p.id for p in points if tagset.issubset({...})}` that also covers
    fuzzy / regex / hierarchical-ancestor matching. Empty `patterns`
    returns every id (no filter).
    """
    pts = list(points)
    if not patterns:
        return {p.id for p in pts}
    return {
        p.id for p in pts
        if match_tags_all(getattr(p, "tags", None) or [], patterns, mode=mode)
    }
