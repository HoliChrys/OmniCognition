"""
MetaCog-Mem — retrospective compression (Chasles-style).

When a reasoning trajectory A → B → C → D ... lands on a near-straight
geodesic — that is, the sum of consecutive distances is close to the
direct distance from start to end — the intermediates are merely
transit. They can be merged into a single point M that sits between
the trajectory boundaries, so future trajectories take the shortcut
A → M → D.

This is the analog of path compression in union-find, but in a
continuous manifold and triggered by usage rather than configuration.

Activation : EAGER (called after each successful trajectory).
Geometry   : SOUPLE (one apply_pull per anchor — positioning emerges,
             never forced to the exact midpoint).

The compression reuses `resolve_collision` from metacog.collision —
the intermediates are passed as the `intermediates` argument and the
trajectory boundaries (A, D, ...) as `anchors`.

Threshold for straight-enough : EMERGENT from the trajectory's own
distribution of subchain straight-ratios — no tuning constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from metacog.collision import (
    CollisionResult,
    Encoder,
    LLMExtractor,
    resolve_collision,
)
from metacog.epistemic import Point
from metacog.geometry import distance, effective_embedding


_EPS = 1e-12


# ---------------------------------------------------------------------------
# Straight-ratio
# ---------------------------------------------------------------------------


def straight_ratio(chain: Sequence[Point], t_now: float) -> float:
    """Return dist(start, end) / Σ dist(consecutives) ∈ (0, 1].

    Close to 1.0 → the chain is almost straight (intermediates are
                   transit-only — compression opportunity).
    Lower         → the chain detours through points that contribute
                   semantically.
    1.0 by convention for chains of length < 3 (no compression possible).
    """
    if len(chain) < 3:
        return 1.0
    embeddings = [effective_embedding(p, t_now) for p in chain]
    path = sum(
        distance(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    )
    if path < _EPS:
        return 1.0
    direct = distance(embeddings[0], embeddings[-1])
    return direct / path


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressionOpportunity:
    anchor_a: Point
    intermediates: Tuple[Point, ...]
    anchor_d: Point
    ratio: float


def detect_chasles_opportunities(
    trajectory: Sequence[Point],
    t_now: float,
) -> List[CompressionOpportunity]:
    """Scan a trajectory for subchains with anomalously high straight-ratio.

    Strategy : enumerate every contiguous subchain of length ≥ 4
    (so there are at least 2 intermediates), compute straight_ratio,
    then keep those above the EMERGENT threshold (median + σ over the
    distribution of ratios within this trajectory).

    Returns the opportunities sorted by descending ratio (strongest
    candidates first), de-duplicated by intermediate-set so the same
    intermediates are not compressed twice in one pass.

    Constraint : intermediates must share the same kind as their
    neighbors (intra-kind compression only).
    """
    n = len(trajectory)
    if n < 4:
        return []

    raw: List[CompressionOpportunity] = []
    for start in range(n - 3):
        for end in range(start + 3, n):
            chain = trajectory[start:end + 1]
            # All intermediates must share kind among themselves
            intermed = chain[1:-1]
            kinds = {p.kind for p in intermed}
            if len(kinds) != 1:
                continue
            ratio = straight_ratio(chain, t_now)
            raw.append(CompressionOpportunity(
                anchor_a=chain[0],
                intermediates=tuple(intermed),
                anchor_d=chain[-1],
                ratio=ratio,
            ))

    if not raw:
        return []

    ratios = sorted(o.ratio for o in raw)
    n_r = len(ratios)
    median = ratios[n_r // 2]
    mean = sum(ratios) / n_r
    var = sum((r - mean) ** 2 for r in ratios) / n_r
    sigma = math.sqrt(var)
    # Emergent threshold : at least median + σ AND > 0.5 (otherwise
    # there isn't enough "straightness" to claim a compression).
    threshold = max(median + sigma, 0.5)

    candidates = [o for o in raw if o.ratio >= threshold]
    candidates.sort(key=lambda o: o.ratio, reverse=True)

    # De-duplicate : prefer the longest chain when intermediate-sets
    # overlap, since a longer compression subsumes a shorter one.
    selected: List[CompressionOpportunity] = []
    used_ids: set[str] = set()
    for opp in candidates:
        ids = {p.id for p in opp.intermediates}
        if ids & used_ids:
            continue
        selected.append(opp)
        used_ids |= ids
    return selected


# ---------------------------------------------------------------------------
# Compression (single chain)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressionEvent:
    timestamp: float
    anchor_a_id: str
    intermediate_ids: Tuple[str, ...]
    anchor_d_id: str
    child_id: str
    ratio_before: float


@dataclass
class CompressionResult:
    child: Point
    parents: List[Point]
    anchors: List[Point]
    event: CompressionEvent
    raw_collision: CollisionResult  # the underlying resolve_collision result


def compress_chain(
    anchor_a: Point,
    intermediates: Sequence[Point],
    anchor_d: Point,
    llm: LLMExtractor,
    encoder: Encoder,
    t_now: float,
    *,
    ratio_before: float = 0.0,
) -> Optional[CompressionResult]:
    """Collapse `intermediates` into a single child M, then pull M
    toward the anchors so it lies between them geometrically.

    Returns None if the underlying resolve_collision produces no child
    (no common content, etc.) — the chain cannot be safely compressed.

    Geometry is SOUPLE : one apply_pull per anchor (handled inside
    resolve_collision when `anchors=` is passed). The child's exact
    position emerges from the encoder output PLUS the pulls — never
    forced to a precise midpoint.
    """
    collision = resolve_collision(
        intermediates, llm, encoder, t_now,
        anchors=[anchor_a, anchor_d],
        trigger_distance=0.0,
        threshold=0.0,
    )
    if collision is None:
        return None

    event = CompressionEvent(
        timestamp=t_now,
        anchor_a_id=anchor_a.id,
        intermediate_ids=tuple(p.id for p in intermediates),
        anchor_d_id=anchor_d.id,
        child_id=collision.child.id,
        ratio_before=ratio_before,
    )
    return CompressionResult(
        child=collision.child,
        parents=collision.parents,
        anchors=[anchor_a, anchor_d],
        event=event,
        raw_collision=collision,
    )


# ---------------------------------------------------------------------------
# Eager activation : called after each successful trajectory
# ---------------------------------------------------------------------------


@dataclass
class EagerCompressionReport:
    opportunities_detected: int = 0
    compressions_applied: int = 0
    new_children: List[Point] = field(default_factory=list)
    events: List[CompressionEvent] = field(default_factory=list)


def compress_trajectory(
    trajectory: Sequence[Point],
    points: List[Point],
    llm: LLMExtractor,
    encoder: Encoder,
    t_now: float,
) -> EagerCompressionReport:
    """EAGER activation : detect every Chasles opportunity in this
    trajectory and apply compression for each. The new children are
    appended to `points`. No deletion — original intermediates remain
    in the cloud (revised via Option Z, with n_revision++).
    """
    report = EagerCompressionReport()
    opportunities = detect_chasles_opportunities(trajectory, t_now)
    report.opportunities_detected = len(opportunities)
    for opp in opportunities:
        result = compress_chain(
            opp.anchor_a, list(opp.intermediates), opp.anchor_d,
            llm, encoder, t_now,
            ratio_before=opp.ratio,
        )
        if result is None:
            continue
        points.append(result.child)
        report.new_children.append(result.child)
        report.events.append(result.event)
        report.compressions_applied += 1
    return report
