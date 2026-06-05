"""
Tests for the parent-prefix glossary + tri-modal tag matching
(`metacog/tags.py`). Pure, deterministic — no LLM, no encoder fixtures.
"""

from __future__ import annotations

from metacog.epistemic import Point, PointKind
from metacog.tags import (
    parent_prefixes,
    is_hierarchical,
    leaf,
    tag_glossary,
    match_tag,
    match_tags_all,
    filter_points,
)


def _p(pid: str, *tags: str) -> Point:
    return Point(
        id=pid, content="x", embedding_orig=(1.0, 0.0),
        kind=PointKind.FACT, tags=list(tags),
    )


# ---------------------------------------------------------------------------
# parent_prefixes
# ---------------------------------------------------------------------------


def test_parent_prefixes_keeps_every_ancestor_drops_leaf():
    assert parent_prefixes("ref:date:2022") == ["ref", "ref:date"]
    assert parent_prefixes("ref:skill:plot:path:src/plot.py") == [
        "ref", "ref:skill", "ref:skill:plot", "ref:skill:plot:path",
    ]


def test_parent_prefixes_single_segment_is_empty():
    assert parent_prefixes("fact") == []
    assert parent_prefixes("") == []


def test_parent_prefixes_two_segments_yields_one_parent():
    assert parent_prefixes("session:s1") == ["session"]


def test_is_hierarchical_and_leaf():
    assert is_hierarchical("ref:date:2022")
    assert not is_hierarchical("fact")
    assert leaf("ref:date:2022") == "2022"
    assert leaf("fact") == "fact"
    assert leaf("") == ""


# ---------------------------------------------------------------------------
# tag_glossary
# ---------------------------------------------------------------------------


def test_glossary_aggregates_parents_and_orders_by_depth_then_alpha():
    pts = [
        _p("a", "ref:date:2022", "session:s1", "fact"),
        _p("b", "ref:skill:plot", "user:u42"),
        _p("c", "tool:yes"),
    ]
    g = tag_glossary(pts)
    # depth 0 first (flat), then depth 1 (one ':')
    assert g == [
        "fact", "ref", "session", "tool", "user",
        "ref:date", "ref:skill",
    ]


def test_glossary_can_exclude_flat_tags():
    pts = [_p("a", "ref:date:2022", "fact")]
    assert tag_glossary(pts, include_flat=False) == ["ref", "ref:date"]


def test_glossary_dedups_across_points():
    pts = [
        _p("a", "ref:date:2022"),
        _p("b", "ref:date:2023"),
        _p("c", "ref:date:2024"),
    ]
    # All three share the same parents — exactly one entry each.
    assert tag_glossary(pts, include_flat=False) == ["ref", "ref:date"]


def test_glossary_ignores_empty_and_non_string():
    pts = [_p("a", "ref:x:1"), Point(
        id="z", content="x", embedding_orig=(1.0, 0.0),
        kind=PointKind.FACT, tags=["ref:x:2", "", " "],
    )]
    g = tag_glossary(pts, include_flat=False)
    assert g == ["ref", "ref:x"]


# ---------------------------------------------------------------------------
# match_tag (exact / fuzzy / regex)
# ---------------------------------------------------------------------------


def test_exact_matches_equality_and_hierarchical_ancestry():
    # Tag on the Point is `ref:date:2022` (post-normalize lowercase).
    tags = ["ref:date:2022", "fact"]
    assert match_tag(tags, "ref:date:2022")
    assert match_tag(tags, "ref:date")     # ancestor
    assert match_tag(tags, "ref")          # ancestor
    assert match_tag(tags, "fact")
    assert not match_tag(tags, "session")
    assert not match_tag(tags, "ref:date:2023")


def test_exact_is_case_insensitive():
    tags = ["ref:date:2022"]
    assert match_tag(tags, "REF:Date")


def test_fuzzy_tolerates_one_typo_per_four_chars():
    tags = ["session:s1", "ref:date:2022"]
    # "sessio" is 1 edit away from "session" (7 chars → budget 1)
    assert match_tag(tags, "sessio", mode="fuzzy")
    # "session:s2" — segment 2 differs by 1 edit, but len("s1")=2 → budget 0
    assert not match_tag(tags, "session:s2", mode="fuzzy")


def test_fuzzy_needle_cannot_be_deeper_than_tag():
    tags = ["session"]
    # Pattern has 2 segs, tag has 1 — false even if first segment fuzzy-OK.
    assert not match_tag(tags, "session:s1", mode="fuzzy")


def test_regex_runs_re_search_on_each_tag():
    tags = ["ref:date:2022", "ref:date:2023", "session:s7"]
    assert match_tag(tags, r"^ref:date:202[0-9]$", mode="regex")
    assert match_tag(tags, r"session:s\d+", mode="regex")
    assert not match_tag(tags, r"^user:", mode="regex")


def test_regex_bad_pattern_returns_false_not_raise():
    assert not match_tag(["x"], "(unclosed", mode="regex")


def test_match_tags_all_requires_every_pattern():
    tags = ["ref:date:2022", "session:s1", "fact"]
    assert match_tags_all(tags, ["session", "ref:date"])
    assert not match_tags_all(tags, ["session", "user"])
    assert match_tags_all(tags, [])      # empty = vacuously true


# ---------------------------------------------------------------------------
# filter_points
# ---------------------------------------------------------------------------


def test_filter_points_exact_with_ancestor_match():
    pts = [
        _p("a", "ref:date:2022", "session:s1"),
        _p("b", "ref:date:2023", "session:s2"),
        _p("c", "session:s1"),
    ]
    ids = filter_points(pts, ["ref:date", "session:s1"], mode="exact")
    assert ids == {"a"}


def test_filter_points_regex_picks_range():
    pts = [
        _p("a", "ref:date:2022"),
        _p("b", "ref:date:2030"),
        _p("c", "ref:date:1999"),
    ]
    ids = filter_points(pts, [r"^ref:date:20[0-9]{2}$"], mode="regex")
    assert ids == {"a", "b"}


def test_filter_points_empty_patterns_returns_all():
    pts = [_p("a"), _p("b"), _p("c")]
    assert filter_points(pts, []) == {"a", "b", "c"}
