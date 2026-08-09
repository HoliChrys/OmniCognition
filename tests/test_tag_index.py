"""
Tests for the semantic tag-segment index (metacog.tag_index).

Deterministic : SimpleEncoder gives stable vectors. Semantic-ordering asserts
use an EXACT-segment query (its own embedding -> cosine 1.0, always the top
hit, encoder-agnostic) ; structural asserts cover segment extraction and the
glossary "grep" locate stage.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.epistemic import Point, PointKind
from metacog.tag_index import TagIndex, tag_segments


def _p(pid, tags):
    return Point(id=pid, content=pid,
                 embedding_orig=SimpleEncoder().encode(pid),
                 kind=PointKind.FACT, keywords=[], tags=tags)


def _points():
    return [
        _p("D1", ["body:hand:finger", "health:condition:macrodactyly"]),
        _p("D9", ["health:condition:fatigue"]),
        _p("D5", ["animal:canine:puppy"]),
    ]


def test_tag_segments_splits_hierarchy_dedup():
    segs = tag_segments(_points())
    for s in ("body", "hand", "finger", "health", "condition",
              "macrodactyly", "fatigue", "animal", "canine", "puppy"):
        assert s in segs
    assert len(segs) == len(set(segs))            # deduped
    # the FACT kind tag is a flat segment, included
    assert "fact" in segs


def test_nearest_segment_is_self_for_exact_query():
    idx = TagIndex(SimpleEncoder()).build(_points())
    near = idx.nearest_segments("health", k=3)
    assert near and near[0][0] == "health"
    assert near[0][1] > 0.99                       # cosine with itself


def test_locate_returns_namespaces_then_full_tags():
    idx = TagIndex(SimpleEncoder()).build(_points())
    pts = _points()
    # a namespace-key segment -> the parent prefixes
    assert "health:condition" in idx.locate("condition", pts)
    # a leaf segment with no namespace cover -> the full tag
    assert "health:condition:fatigue" in idx.locate("fatigue", pts)
    # a namespace that covers the segment suppresses the redundant full tag
    loc = idx.locate("health", pts)
    assert "health" in loc and "health:condition" in loc


def test_search_end_to_end_shape():
    idx = TagIndex(SimpleEncoder()).build(_points())
    res = idx.search("fatigue", _points(), k=2)
    assert res and res[0]["segment"] == "fatigue"
    assert "health:condition:fatigue" in res[0]["namespaces"]
    assert set(res[0]) == {"segment", "score", "namespaces"}


def test_empty_and_failsafe():
    idx = TagIndex(SimpleEncoder()).build([])
    assert idx.nearest_segments("anything") == []
    assert idx.search("anything", []) == []
    # rebuild after new tags only re-encodes the newcomers (cache kept)
    idx.build(_points())
    assert idx.nearest_segments("health", k=1)[0][0] == "health"


def test_incremental_build_keeps_cache():
    idx = TagIndex(SimpleEncoder())
    idx.build([_p("D1", ["health:condition:fatigue"])])
    cached = dict(idx._emb)
    idx.build(_points())                            # adds more segments
    # previously-cached vectors are untouched (same object identity)
    for seg, vec in cached.items():
        assert idx._emb[seg] is vec
