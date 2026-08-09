"""
Phase-5 geometric threshold cache — hybrid invalidation, exact fallback.

The O(n²) emergent spread threshold (median−σ of pairwise keyword distances)
is cached on (subset ids, GEO_EPOCH). Any apply_pull bumps the epoch and any
subset change alters the key → exact recompute. sleep()/load() clear the cache.
"""

from __future__ import annotations

import time

from metacog import geometry
from metacog.defaults import SimpleEncoder
from metacog.geometry import apply_pull, geometric_spread
from metacog.memory import Memory


def _mem(n=12):
    m = Memory(encoder=SimpleEncoder())
    words = ["alpha", "beta", "gamma", "delta", "iran", "oil"]
    for i in range(n):
        m.ingest(" ".join(words[(i + j) % len(words)] for j in range(4)),
                 kind="FACT", id=f"P{i}")
    return m


def test_cache_hit_identical_result_on_frozen_manifold():
    m = _mem()
    geometry.clear_geo_cache()
    pts = list(m.points)
    seeds = pts[:2]
    r1 = geometric_spread(seeds, pts, 100.0)       # t explicite (figé)
    assert geometry._SPREAD_THR_CACHE               # threshold cached
    r2 = geometric_spread(seeds, pts, 100.0)       # hit -> identical
    assert [(d, p.id) for d, p in r1] == [(d, p.id) for d, p in r2]


def test_pull_bumps_epoch_and_invalidates():
    m = _mem()
    geometry.clear_geo_cache()
    pts = list(m.points)
    geometric_spread(pts[:1], pts, 100.0)
    (key, (epoch, _thr)), = list(geometry._SPREAD_THR_CACHE.items())
    apply_pull(pts[0], pts[1], +1.0, 101.0)        # structural mutation
    assert geometry.GEO_EPOCH > epoch               # epoch bumped
    geometric_spread(pts[:1], pts, 102.0)
    assert geometry._SPREAD_THR_CACHE[key][0] == geometry.GEO_EPOCH  # recomputed


def test_subset_change_changes_key():
    m = _mem()
    geometry.clear_geo_cache()
    pts = list(m.points)
    geometric_spread(pts[:1], pts, 100.0)
    n1 = len(geometry._SPREAD_THR_CACHE)
    m.ingest("a brand new point about oil", kind="FACT", id="NEW")
    geometric_spread(pts[:1], list(m.points), 100.0)
    assert len(geometry._SPREAD_THR_CACHE) > n1     # different key -> new entry


def test_load_clears_cache(tmp_path):
    path = str(tmp_path / "m.pkl")
    m = _mem()
    m.storage_path = path
    geometric_spread(m.points[:1], list(m.points), 100.0)
    assert geometry._SPREAD_THR_CACHE
    m.save()
    m.load()
    assert geometry._SPREAD_THR_CACHE == {}


def test_perf_cached_repeat_calls_faster():
    m = _mem(n=80)
    geometry.clear_geo_cache()
    pts = list(m.points)
    t0 = time.time()
    geometric_spread(pts[:2], pts, 100.0)           # cold : pays O(n²)
    cold = time.time() - t0
    t0 = time.time()
    for _ in range(10):
        geometric_spread(pts[:2], pts, 100.0)       # hits : O(n) only
    warm_each = (time.time() - t0) / 10
    assert warm_each < cold                          # strictly cheaper per call
