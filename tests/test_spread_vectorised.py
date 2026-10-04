"""
TAC-940 — geometric_spread vectorised with numpy.

V1  the numpy (median − σ) threshold equals the scalar all-pairs reference
    to 1e-9 on a small population
V2  geometric_spread returns the same neighbours, in the same order, with
    the same distances (1e-9) as the scalar reference loop
V3  a cache miss on n = 1 830 points of dimension 384 (the `global`
    context) costs under 3 s — the scalar loop took ~170 s
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from metacog import Point, PointKind
from metacog import geometry
from metacog.geometry import (
    _pairwise_spread_threshold,
    distance,
    effective_keyword_embedding,
    geometric_spread,
)


def _points(n: int, dim: int, seed: int) -> list:
    rng = random.Random(seed)
    pts = []
    for i in range(n):
        v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in v))
        kw = tuple(x / norm for x in v)
        pts.append(Point(
            id=f"P{i}", content=f"p{i}", embedding_orig=kw,
            kind=PointKind.FACT, keywords_embedding=kw,
            delta_active=tuple(rng.gauss(0.0, 0.05) for _ in range(dim)),
            delta_latent=tuple(rng.gauss(0.0, 0.05) for _ in range(dim)),
            t_last_obs=float(i % 7),
        ))
    return pts


def _scalar_threshold(embs: list) -> float:
    """The pre-TAC-940 statistic, verbatim : all pairs in pure Python."""
    dists = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            dists.append(distance(embs[i], embs[j]))
    dists.sort()
    median = dists[len(dists) // 2]
    mean = sum(dists) / len(dists)
    sigma = math.sqrt(sum((d - mean) ** 2 for d in dists) / len(dists))
    return max(0.0, median - sigma)


def _scalar_spread(seeds, pts, t_now):
    embs = {p.id: effective_keyword_embedding(p, t_now) for p in pts}
    thr = _scalar_threshold([embs[p.id] for p in pts])
    seed_ids = {p.id for p in seeds}
    found: dict = {}
    for s in seeds:
        es = embs.get(s.id)
        if es is None:
            continue
        for p in pts:
            if p.id in seed_ids:
                continue
            d = distance(es, embs[p.id])
            if d < thr and (p.id not in found or d < found[p.id]):
                found[p.id] = d
    return sorted(found.items(), key=lambda x: x[1])


def test_numpy_threshold_matches_scalar_reference():
    pts = _points(60, 32, seed=7)
    embs = [effective_keyword_embedding(p, 10.0) for p in pts]
    ref = _scalar_threshold(embs)
    got = _pairwise_spread_threshold(np.asarray(embs, dtype=np.float64))
    assert abs(got - ref) < 1e-9


def test_spread_matches_scalar_reference():
    pts = _points(80, 32, seed=11)
    seeds = pts[:5]
    geometry.clear_geo_cache()
    got = geometric_spread(seeds, pts, 10.0)
    ref = _scalar_spread(seeds, pts, 10.0)
    assert ref                                      # non-trivial population
    assert [p.id for _d, p in got] == [pid for pid, _d in ref]
    for (d_got, _p), (_pid, d_ref) in zip(got, ref):
        assert abs(d_got - d_ref) < 1e-9


def test_cache_miss_on_global_size_under_3s():
    pts = _points(1830, 384, seed=3)
    geometry.clear_geo_cache()
    t0 = time.perf_counter()
    geometric_spread(pts[:7], pts, 10.0)
    assert time.perf_counter() - t0 < 3.0
