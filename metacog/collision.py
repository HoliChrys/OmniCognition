"""
MetaCog-Mem — collision resolution (lineage-by-fission).

When two points in the manifold become anomalously close, they're
factorized: a CHILD point captures their common content, and each
parent is trimmed to retain only its distinctive part.

Design choices (validated upstream):
  - Threshold        : emergent (1σ below the median pairwise distance).
  - Parent identity  : Option Z — counters KEPT, `n_revision` += 1,
                       uncertainty rises via Beta dilution.
  - LLM usage        : content production only (∈ P), never as evidence.
                       Detection (distance) = COMPUTATION.
  - Cascade          : max 2 iterations of detect→resolve per cycle.

Trace : every collision is logged as a `CollisionEvent` with explicit
source classes for both detection and content generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

from metacog.epistemic import (
    EpistemicState,
    Point,
    SourceClass,
)
from metacog.geometry import (
    distance,
    effective_embedding,
    vec_norm,
)


# ---------------------------------------------------------------------------
# LLM and encoder stubs (Protocols — real implementations injected by caller)
# ---------------------------------------------------------------------------


class LLMExtractor(Protocol):
    """LLM-backed content surgery. All outputs are typed GENERATOR — used
    ONLY as content, never as observations."""

    def extract_common(self, a: str, b: str) -> str: ...
    def remove_overlap(self, full: str, common: str) -> str: ...


class Encoder(Protocol):
    """Deterministic text → vector. Treated as COMPUTATION (same input
    → same output, no stochastic sampling at inference time)."""

    def encode(self, text: str) -> Tuple[float, ...]: ...


# ---------------------------------------------------------------------------
# Collision audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollisionEvent:
    """One row in the collision audit log."""

    timestamp: float
    parent_a_id: str
    parent_b_id: str
    child_id: str
    trigger_distance: float
    threshold: float
    cycle_iteration: int
    detection_class: SourceClass = SourceClass.COMPUTATION
    content_generation_class: SourceClass = SourceClass.GENERATOR


@dataclass
class CollisionResult:
    child: Point
    parent_a: Point  # mutated in place (Option Z)
    parent_b: Point  # mutated in place
    event: CollisionEvent


# ---------------------------------------------------------------------------
# Threshold and detection
# ---------------------------------------------------------------------------


def collision_threshold(points: Sequence[Point], t_now: float) -> Optional[float]:
    """Emergent threshold : (median - σ) of all pairwise distances.

    Returns None if the population is too small to have meaningful
    statistics (< 4 points → < 6 pairs).
    """
    if len(points) < 4:
        return None
    distances: List[float] = []
    for i in range(len(points)):
        eff_i = effective_embedding(points[i], t_now)
        for j in range(i + 1, len(points)):
            eff_j = effective_embedding(points[j], t_now)
            distances.append(distance(eff_i, eff_j))
    if not distances:
        return None
    n = len(distances)
    distances.sort()
    median = distances[n // 2]
    mean = sum(distances) / n
    var = sum((d - mean) ** 2 for d in distances) / n
    sigma = math.sqrt(var)
    return max(0.0, median - sigma)


def detect_collisions(
    points: Sequence[Point], t_now: float
) -> List[Tuple[Point, Point, float]]:
    """Return all pairs whose effective distance is below the emergent
    threshold. Each tuple is `(p_a, p_b, distance)`.
    Eligible points have state ∉ {DEPRECATED, INVALID}.
    """
    threshold = collision_threshold(points, t_now)
    if threshold is None:
        return []
    eligible = [
        p for p in points
        if p.state not in {EpistemicState.DEPRECATED, EpistemicState.INVALID}
    ]
    pairs: List[Tuple[Point, Point, float]] = []
    for i in range(len(eligible)):
        eff_i = effective_embedding(eligible[i], t_now)
        for j in range(i + 1, len(eligible)):
            eff_j = effective_embedding(eligible[j], t_now)
            d = distance(eff_i, eff_j)
            if d < threshold:
                pairs.append((eligible[i], eligible[j], d))
    # Sort closest first — most urgent collisions resolved first.
    pairs.sort(key=lambda triple: triple[2])
    return pairs


# ---------------------------------------------------------------------------
# Resolution (the actual fission)
# ---------------------------------------------------------------------------


def _make_child_id(a_id: str, b_id: str, t_now: float) -> str:
    return f"collision({a_id}+{b_id})@{t_now:.3f}"


def resolve_collision(
    p_a: Point,
    p_b: Point,
    llm: LLMExtractor,
    encoder: Encoder,
    t_now: float,
    trigger_distance: float,
    threshold: float,
    cycle_iteration: int = 1,
) -> Optional[CollisionResult]:
    """Factorize two close points.

    Returns None if LLM produces an empty common (no shared content) —
    in that case the apparent closeness is geometric noise, not real
    redundancy, and we leave the points alone.
    """
    common_text = llm.extract_common(p_a.content, p_b.content).strip()
    if not common_text:
        return None

    trimmed_a = llm.remove_overlap(p_a.content, common_text).strip()
    trimmed_b = llm.remove_overlap(p_b.content, common_text).strip()

    if not trimmed_a or not trimmed_b:
        # The "common" covers the entirety of one parent — that parent
        # IS the common content. Skip rather than create an empty point.
        return None

    child_id = _make_child_id(p_a.id, p_b.id, t_now)
    child = Point(
        id=child_id,
        content=common_text,
        embedding_orig=tuple(encoder.encode(common_text)),
        state=EpistemicState.CONJECTURE,
        parents=[p_a.id, p_b.id],
        lineage_depth=max(p_a.lineage_depth, p_b.lineage_depth) + 1,
    )

    # Mutate parents in place (Option Z) :
    # - keep id, counters, update_log, parents lineage
    # - content + embedding_orig refreshed
    # - deltas reset (the geometric history was anchored to the old
    #   position; in the new coordinate system it doesn't transfer)
    # - n_revision += 1 (Beta dilution rises uncertainty)
    # - children gets the new child appended
    d = len(child.embedding_orig)
    for parent, trimmed in ((p_a, trimmed_a), (p_b, trimmed_b)):
        parent.content = trimmed
        parent.embedding_orig = tuple(encoder.encode(trimmed))
        parent.delta_active = tuple(0.0 for _ in range(d))
        parent.delta_latent = tuple(0.0 for _ in range(d))
        parent.n_revision += 1
        parent.children = list(parent.children) + [child_id]

    event = CollisionEvent(
        timestamp=t_now,
        parent_a_id=p_a.id,
        parent_b_id=p_b.id,
        child_id=child_id,
        trigger_distance=trigger_distance,
        threshold=threshold,
        cycle_iteration=cycle_iteration,
    )
    return CollisionResult(child=child, parent_a=p_a, parent_b=p_b, event=event)


# ---------------------------------------------------------------------------
# Sleep cycle orchestration
# ---------------------------------------------------------------------------


MAX_CASCADE_ITERATIONS = 2  # validated design choice — not a tuning knob


@dataclass
class SleepCycleReport:
    iterations: int = 0
    resolved: List[CollisionEvent] = field(default_factory=list)
    new_children: List[Point] = field(default_factory=list)
    aborted_for_cascade_limit: bool = False


def sleep_cycle_collisions(
    points: List[Point],
    llm: LLMExtractor,
    encoder: Encoder,
    t_now: float,
) -> SleepCycleReport:
    """One full sleep cycle of collision resolution.

    Runs at most `MAX_CASCADE_ITERATIONS` rounds of detect→resolve.
    Any collisions still present after that wait for the next cycle.

    Mutates `points` in place : new children are appended, parents are
    revised. Returns a structured report for audit.
    """
    report = SleepCycleReport()
    threshold_for_logging = collision_threshold(points, t_now) or 0.0

    for iteration in range(1, MAX_CASCADE_ITERATIONS + 1):
        collisions = detect_collisions(points, t_now)
        if not collisions:
            break
        report.iterations = iteration

        resolved_ids_this_round: set[str] = set()
        for p_a, p_b, d in collisions:
            # Don't touch the same point twice within a single round.
            if p_a.id in resolved_ids_this_round or p_b.id in resolved_ids_this_round:
                continue
            result = resolve_collision(
                p_a, p_b, llm, encoder, t_now,
                trigger_distance=d,
                threshold=threshold_for_logging,
                cycle_iteration=iteration,
            )
            if result is None:
                continue
            points.append(result.child)
            report.new_children.append(result.child)
            report.resolved.append(result.event)
            resolved_ids_this_round.add(p_a.id)
            resolved_ids_this_round.add(p_b.id)
    else:
        # Loop ran to MAX without break — check if collisions remain.
        if detect_collisions(points, t_now):
            report.aborted_for_cascade_limit = True

    return report
