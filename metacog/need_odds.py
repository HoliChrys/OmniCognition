"""
Need-odds decay (Anderson-Schooler 1991) — the ONE fitted hyperparameter, the
mnema model (architecture.md §11).

A memory's "need odds" — the odds it will be needed again soon — follow a
power-law of the ages of its past accesses :

    need_odds(node) = Σ over past accesses of (now - t_access) ** (-d)

where `d` is the decay exponent (~0.5 in the human-memory literature). It is the
system's ONLY learned parameter : everything else scales for free with the
models. The access history comes from the append-only journal
(`Journal.access_timestamps`) and the supervision from `mark_useful`
(`Journal.useful_retrievals`).

`fit_exponent` grid-searches `d` to best SEPARATE useful nodes from useless ones
by an AUC objective (scale-invariant, so different `d` are comparable). This is
the closed L3 feedback loop : the agent's 0/1/2 scores recalibrate the forget
speed. `blend` is the optional ranking modulator (a convex mix of a base score
and the normalized need-odds), off by default (recency_weight=0)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_EXPONENT = 0.5
DEFAULT_GRID: Tuple[float, ...] = tuple(round(0.1 + 0.1 * i, 2) for i in range(15))


def need_odds(access_timestamps: Sequence[float], now: float,
              exponent: float = DEFAULT_EXPONENT) -> float:
    """Power-law need-odds. Empty history → 0.0. Ages ≤ 0 (same instant) are
    clamped to 1.0 to avoid a divide-by-zero blow-up."""
    total = 0.0
    for ts in access_timestamps:
        age = now - ts
        if age <= 0:
            age = 1.0
        total += age ** (-exponent)
    return total


def _auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Probability a random positive scores above a random negative (ties=0.5).
    0.5 = no separation. O(len(pos)·len(neg))."""
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def fit_exponent(
    pos_histories: Sequence[Sequence[float]],
    neg_histories: Sequence[Sequence[float]],
    now: float,
    grid: Sequence[float] = DEFAULT_GRID,
) -> Dict[str, object]:
    """Grid-search the decay exponent that best separates useful nodes (pos)
    from useless ones (neg) by need-odds AUC. Returns
    {exponent, auc, n_pos, n_neg}. Falls back to DEFAULT_EXPONENT with auc=0.5
    when either class is empty (nothing to learn from yet)."""
    if not pos_histories or not neg_histories:
        return {"exponent": DEFAULT_EXPONENT, "auc": 0.5,
                "n_pos": len(pos_histories), "n_neg": len(neg_histories)}
    best_d, best_auc = DEFAULT_EXPONENT, -1.0
    for d in grid:
        pos = [need_odds(h, now, d) for h in pos_histories]
        neg = [need_odds(h, now, d) for h in neg_histories]
        auc = _auc(pos, neg)
        if auc > best_auc:
            best_auc, best_d = auc, d
    return {"exponent": best_d, "auc": best_auc,
            "n_pos": len(pos_histories), "n_neg": len(neg_histories)}


def _minmax(xs: Sequence[float]) -> List[float]:
    """Min-max normalize to [0,1], PRESERVING the relative spacing of the values
    (unlike a sigmoid, which compresses them). All-equal -> all 0.0 (no signal)."""
    xs = list(xs)
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return [0.0 for _ in xs]
    span = hi - lo
    return [(x - lo) / span for x in xs]


def blend(base_scores: Sequence[float], need_values: Sequence[float],
          recency_weight: float) -> List[float]:
    """Convex mix of a base relevance score and the need-odds modulator.
    recency_weight 0.0 → pure base ; 1.0 → pure need-odds. BOTH sides are
    min-max normalized over the candidate set so they live in [0,1] while
    keeping their internal spacing — a dominant base hit stays near 1.0 and
    resists being flipped, while near-tied candidates are the ones need-odds can
    reorder. (Earlier this squashed the base through a sigmoid, which crushed the
    discriminative cosine gaps and let tiny weights swing the ranking; calibration
    showed that degrades recall.)"""
    if not 0.0 <= recency_weight <= 1.0:
        raise ValueError(f"recency_weight must be in [0,1], got {recency_weight}")
    base = _minmax(base_scores)
    if recency_weight == 0.0 or not need_values:
        return base
    need = _minmax(need_values)
    return [(1.0 - recency_weight) * b + recency_weight * n
            for b, n in zip(base, need)]
