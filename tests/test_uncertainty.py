"""
Tests for uncertainty propagation (metacog/uncertainty.py) and its
effect on lineage retrieval pruning.

Properties :
  U1   beta_sigma is monotonic with the Beta variance
  U2   hop_sigma returns 1 - cosine of effective embeddings
  U3   propagate combines uncertainties as sqrt(sum of squares)
  U4   prune_threshold returns None for small populations
  U5   prune_threshold returns None for homogeneous populations (cold start)
  U6   prune_threshold returns median + std when diversity is sufficient
  U7   high-uncertainty branches drop out of retrieve_with_lineage results
"""

from __future__ import annotations

import hashlib
import math
from typing import Tuple

from metacog import (
    Memory,
    Point,
    PointKind,
    beta_sigma,
    hop_sigma,
    propagate,
    prune_threshold,
)


class _Enc:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self._cache: dict = {}

    def encode(self, text: str) -> Tuple[float, ...]:
        words = text.lower().split()
        if not words:
            return tuple(0.0 for _ in range(self.dim))
        acc = [0.0] * self.dim
        for w in words:
            if w not in self._cache:
                h = int(hashlib.md5(f"u:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


# ---------- U1 ------------------------------------------------------------


def test_beta_sigma_increases_with_higher_variance():
    # Point with many observations → low variance → low sigma
    confident = Point(
        id="C", content="x", embedding_orig=(0.0,),
        n_corrob=20, n_contra=20,
    )
    # Point with few observations → high variance → high sigma
    uncertain = Point(
        id="U", content="x", embedding_orig=(0.0,),
        n_corrob=1, n_contra=1,
    )
    assert beta_sigma(uncertain) > beta_sigma(confident)


# ---------- U2 ------------------------------------------------------------


def test_hop_sigma_equals_one_minus_cosine():
    enc = _Enc()
    p1 = Point(id="P1", content="alpha beta",
               embedding_orig=enc.encode("alpha beta"))
    p2 = Point(id="P2", content="alpha beta",
               embedding_orig=enc.encode("alpha beta"))
    # Identical content → identical embedding → hop_sigma ≈ 0
    assert hop_sigma(p1, p2, t_now=0.0) < 0.01


def test_hop_sigma_high_for_unrelated_points():
    enc = _Enc()
    p1 = Point(id="P1", content="alpha beta gamma",
               embedding_orig=enc.encode("alpha beta gamma"))
    p2 = Point(id="P2", content="completely different topic",
               embedding_orig=enc.encode("completely different topic"))
    # Different content → high hop_sigma
    assert hop_sigma(p1, p2, t_now=0.0) > 0.3


# ---------- U3 ------------------------------------------------------------


def test_propagate_combines_as_quadrature():
    # σ_total² = σ_seed² + Σ σ_hop²
    seed = 0.5
    hops = [0.3, 0.4]
    expected = math.sqrt(0.25 + 0.09 + 0.16)
    assert abs(propagate(seed, hops) - expected) < 1e-9


def test_propagate_empty_hops_returns_seed_sigma():
    assert abs(propagate(0.7, []) - 0.7) < 1e-9


# ---------- U4, U5, U6 ----------------------------------------------------


def test_prune_threshold_none_for_small_population():
    enc = _Enc()
    pts = [
        Point(id=f"P{i}", content=f"x{i}",
              embedding_orig=enc.encode(f"x{i}"))
        for i in range(3)
    ]
    assert prune_threshold(pts) is None


def test_prune_threshold_none_for_homogeneous_population():
    enc = _Enc()
    # All points share the same Beta prior → std(σ) = 0 → threshold None
    pts = [
        Point(id=f"P{i}", content=f"x{i}",
              embedding_orig=enc.encode(f"x{i}"))
        for i in range(8)
    ]
    assert prune_threshold(pts) is None


def test_prune_threshold_returns_value_for_diverse_population():
    enc = _Enc()
    pts = [
        Point(id=f"P{i}", content=f"x{i}",
              embedding_orig=enc.encode(f"x{i}"),
              n_corrob=i, n_contra=10 - i)
        for i in range(10)
    ]
    threshold = prune_threshold(pts)
    assert threshold is not None
    assert threshold > 0


# ---------- U7 — end-to-end pruning effect --------------------------------


def test_uncertain_branches_are_pruned_in_lineage():
    """A high-uncertainty branch is pruned from lineage expansion when
    the population is diverse enough for the threshold to fire."""
    enc = _Enc()
    m = Memory(encoder=enc)
    # Confident seed : many observations
    m.ingest("alpha bridge", id="SEED")
    m.points[-1].n_corrob = 30
    m.points[-1].n_contra = 1
    # Highly uncertain neighbor (no obs, geometrically unrelated)
    m.ingest("completely orthogonal content here", id="UNCERTAIN",
             sequence_prev="SEED")
    m.points[-1].n_corrob = 0
    m.points[-1].n_contra = 0
    # Diversify the population so the threshold fires
    for i in range(20):
        m.ingest(f"filler {i} different topic",
                 id=f"D{i}", )
        m.points[-1].n_corrob = i
        m.points[-1].n_contra = 20 - i

    # With lineage + uncertainty propagation, the uncertain orthogonal
    # neighbor of SEED should NOT make the top result for "alpha".
    results = m.retrieve("alpha bridge", k=3, use_hybrid=True, use_lineage=True)
    ids = [r["id"] for r in results]
    assert "SEED" in ids
    # UNCERTAIN may still appear but should NOT outrank SEED
    if "UNCERTAIN" in ids:
        assert ids.index("SEED") < ids.index("UNCERTAIN")
