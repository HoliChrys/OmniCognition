"""
Uncertainty propagation for retrieval pruning.

Reference : Propagation des incertitudes (Bureau International des
Poids et Mesures, GUM 1995).

Each Point carries its own epistemic uncertainty derived from the
Beta posterior on its counters :

    σ_point  =  sqrt(Var(Beta(1 + n_corrob, 1 + n_contra)))

When the BFS traversal hops from a seed to a neighbor, the hop adds
its own uncertainty derived from the geometric distance between the
two points :

    σ_hop  =  1 − cosine(effective_embedding_seed,
                         effective_embedding_neighbor)

Standard uncertainty propagation for independent contributions :

    σ_path²  =  σ_seed² + Σ σ_hop²

A neighbor is PRUNED (the BFS branch stops there) when its
propagated σ exceeds an emergent threshold computed from the
population's own σ distribution :

    threshold  =  median(σ_points) + std(σ_points)

This is metacog-canonical : zero hyperparameter, the threshold
emerges from the data, deeper branches naturally truncate when
uncertainty compounds beyond what the population already supports.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


def beta_sigma(point: "Point") -> float:  # noqa: F821
    """Standard deviation of the Beta(α, β) posterior on the point's
    counters, where α = 1 + n_corrob and β = 1 + n_contra.
    """
    return math.sqrt(max(0.0, point.uncertainty))


def hop_sigma(parent: "Point", child: "Point", t_now: float) -> float:  # noqa: F821
    """Uncertainty added by a single BFS hop.

    Defined as 1 − cosine(effective_parent, effective_child). Returns
    a value in [0, 2] : 0 for geometrically identical points, ~0.3 for
    semantically related, > 1 for nearly opposite.
    """
    from metacog.geometry import cosine, effective_embedding

    e1 = effective_embedding(parent, t_now)
    e2 = effective_embedding(child, t_now)
    return max(0.0, 1.0 - cosine(e1, e2))


def propagate(seed_sigma: float, hop_sigmas: Sequence[float]) -> float:
    """Combined uncertainty for independent contributions :

        σ_total² = σ_seed² + Σ σ_hop²
    """
    return math.sqrt(seed_sigma * seed_sigma + sum(s * s for s in hop_sigmas))


def prune_threshold(
    points: Sequence["Point"],  # noqa: F821
    *,
    min_population: int = 4,
    min_diversity: float = 1e-3,
) -> Optional[float]:
    """Emergent pruning threshold from the population σ distribution.

    Returns median(σ) + std(σ), or None when :
      - the population is too small (< min_population), or
      - the population is homogeneous (std(σ) < min_diversity).
        In a cold-start scenario where every point still has the same
        Beta prior, std collapses to 0 and the threshold would prune
        every branch ; returning None lets the BFS use depth-only
        limiting until enough observations have diversified the
        population.
    """
    if len(points) < min_population:
        return None
    sigmas: List[float] = [beta_sigma(p) for p in points]
    n = len(sigmas)
    mean = sum(sigmas) / n
    var = sum((s - mean) ** 2 for s in sigmas) / n
    std = math.sqrt(var)
    if std < min_diversity:
        return None
    sorted_s = sorted(sigmas)
    median = sorted_s[n // 2]
    return median + std
