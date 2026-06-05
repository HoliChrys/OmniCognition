"""
Tests for association segmentation of evidence (metacog.answer_cluster).

Default clusterer is the HDBSCAN-style mutual-reachability method (chosen
over link_communities: empirically cleaner on the small/sparse evidence
sets we work with — native outlier handling, no chaining). link_communities
is kept selectable via `method=` for comparison.

Uses the REAL semantic encoder (MiniLM via SemanticEncoder) on realistically
SIZED topic groups — mutual reachability with min_cluster_size=2 needs a
few neighbours per topic (the real synthesis runs on ~12-15 evidence facts),
so toy 2-item topics are degenerate and not representative.
"""

from __future__ import annotations

import pytest

from benchmarks.locomo.encoders import SemanticEncoder
from metacog.answer_cluster import associate_clusters, group_evidence

_COUNSEL = [
    "I'm keen on counseling or working in mental health",
    "Lately I've been looking into counseling and therapy",
    "I want to pursue a psychology and counseling certification",
    "Mental health support and therapy really matter to me",
]
_ART = [
    "Oil painting and watercolor landscapes are my passion",
    "I love sketching and drawing in my free time",
]
_OUTLIER = "We went camping by the lake last weekend"


@pytest.fixture(scope="module")
def enc():
    return SemanticEncoder()


def _embed(enc, texts):
    return [enc.encode(t) for t in texts]


def test_outlier_not_merged_into_coherent_group(enc):
    items = _COUNSEL + [_OUTLIER]                      # 0-3 counsel, 4 outlier
    groups = associate_clusters(_embed(enc, items))
    counsel = next(g for g in groups if 0 in g)
    assert {0, 1, 2, 3} <= set(counsel)               # coherent group holds
    assert 4 not in counsel                            # camping outlier excluded


def test_two_topics_kept_separate(enc):
    items = _COUNSEL + _ART                             # 0-3 counsel, 4-5 art
    groups = associate_clusters(_embed(enc, items))
    counsel = next(g for g in groups if 0 in g)
    # the art items must not all be absorbed into the counselling group
    assert not ({4, 5} <= set(counsel))


def test_unrelated_items_do_not_all_collapse(enc):
    items = ["quantum field theory and particle physics",
             "a recipe for sourdough banana bread",
             "training plan for a spring marathon",
             "the history of the Roman aqueducts"]
    groups = associate_clusters(_embed(enc, items))
    assert len(groups) >= 2                             # not one big blob


def test_strongest_group_first(enc):
    items = _COUNSEL + [_OUTLIER]
    groups = associate_clusters(_embed(enc, items))
    assert len(groups[0]) >= 2                          # coherent cluster leads


def test_group_evidence_maps_items(enc):
    items = _COUNSEL + [_OUTLIER]
    groups = group_evidence(items, _embed(enc, items))
    big = next(g for g in groups if items[0] in g)
    assert items[1] in big and _OUTLIER not in big


def test_both_methods_selectable(enc):
    items = _COUNSEL + [_OUTLIER]
    embs = _embed(enc, items)
    for method in ("mutual_reachability", "link_communities"):
        groups = associate_clusters(embs, method=method)
        counsel = next(g for g in groups if 0 in g)
        assert 4 not in counsel                         # outlier excluded by both


def test_handles_tiny_and_empty(enc):
    assert associate_clusters([]) == []
    assert associate_clusters(_embed(enc, ["only one item"])) == [[0]]
