"""
MetaCog-Mem — collision resolution (lineage-by-fission).

Two use cases now supported under a unified API :

  1. PROXIMITY COLLISION (original)
       Two (or more) points anomalously close in the manifold get
       factorized : a CHILD point captures their common content, each
       parent is trimmed to retain only its distinctive part.

  2. CHASLES COMPRESSION (new)
       A chain of intermediate points along a reasoning trajectory
       (A → B → C → D ...) where the path is almost-straight gets
       compressed : B, C, ... merge into a single child M that is
       geometrically pulled toward the boundary anchors A and D,
       so future trajectories take the shortcut A → M → D.

Design choices (validated upstream) :
  - Threshold        : emergent (1σ below the median pairwise distance).
  - Parent identity  : Option Z — counters KEPT, `n_revision` += 1,
                       uncertainty rises via Beta dilution.
  - LLM usage        : content production only (∈ P), never as evidence.
                       Detection (distance) = COMPUTATION.
  - Cascade          : max 2 iterations of detect→resolve per cycle.
  - Cross-kind       : (when PointKind is added) collisions restricted
                       to same-kind pairs only.

Trace : every collision is logged as a `CollisionEvent` carrying
parent_ids (≥ 2) and anchor_ids (≥ 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

from metacog.epistemic import (
    EpistemicState,
    Point,
    PointKind,
    SourceClass,
)
from metacog.geometry import (
    apply_pull,
    cosine,
    distance,
    effective_embedding,
    effective_keyword_embedding,
)

# "Really the same information" is the extreme top tail of semantic
# similarity : near-identical full-content meaning. Empirically (LoCoMo)
# content-cosine ≥ 0.95 isolates genuine restatements with no false
# positives, while merely-related adjacent turns sit well below. This is a
# semantic-IDENTITY cutoff (deliberately high), not a relatedness knob.
MERGE_COSINE_IDENTITY = 0.95


# ---------------------------------------------------------------------------
# LLM and encoder stubs (Protocols — real implementations injected by caller)
# ---------------------------------------------------------------------------


class LLMExtractor(Protocol):
    """LLM-backed content surgery. All outputs are typed GENERATOR — used
    ONLY as content, never as observations.

    Generalized to N inputs : extract_common takes a sequence of texts.
    """

    def extract_common(self, texts: Sequence[str]) -> str: ...
    def remove_overlap(self, full: str, common: str) -> str: ...


class Encoder(Protocol):
    """Deterministic text → vector. Treated as COMPUTATION."""

    def encode(self, text: str) -> Tuple[float, ...]: ...


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollisionEvent:
    """One row in the collision audit log.

    `parent_ids` carries N ≥ 2 ids (the intermediates that were merged).
    `anchor_ids` carries M ≥ 0 ids (the boundary points the child was
    pulled toward — non-empty for Chasles compression).
    """

    timestamp: float
    parent_ids: Tuple[str, ...]
    anchor_ids: Tuple[str, ...]
    child_id: str
    trigger_distance: float
    threshold: float
    cycle_iteration: int
    detection_class: SourceClass = SourceClass.COMPUTATION
    content_generation_class: SourceClass = SourceClass.GENERATOR


@dataclass
class CollisionResult:
    child: Point
    parents: List[Point]    # mutated in place (Option Z)
    anchors: List[Point]    # untouched in their counters, just pulled toward
    event: CollisionEvent


# ---------------------------------------------------------------------------
# Threshold and detection
# ---------------------------------------------------------------------------


def collision_threshold(points: Sequence[Point], t_now: float) -> Optional[float]:
    """Emergent threshold : (median - σ) of all pairwise distances.

    Returns None if the population is too small (< 4 points).
    """
    if len(points) < 4:
        return None
    distances: List[float] = []
    for i in range(len(points)):
        eff_i = effective_keyword_embedding(points[i], t_now)
        for j in range(i + 1, len(points)):
            eff_j = effective_keyword_embedding(points[j], t_now)
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
    threshold AND that share the same `kind` (FACT/THOUGHT/ACTION).

    Cross-kind proximity is preserved as a feature of the manifold —
    it expresses semantic relatedness — but cross-kind merge is
    ontologically incorrect (a FACT and an ACTION cannot be distilled
    into a common parent).

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
        eff_i = effective_keyword_embedding(eligible[i], t_now)
        for j in range(i + 1, len(eligible)):
            # Intra-kind only
            if eligible[i].kind != eligible[j].kind:
                continue
            eff_j = effective_keyword_embedding(eligible[j], t_now)
            d = distance(eff_i, eff_j)
            if d < threshold:
                pairs.append((eligible[i], eligible[j], d))
    pairs.sort(key=lambda triple: triple[2])
    return pairs


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _make_child_id(parent_ids: Sequence[str], t_now: float) -> str:
    return f"collision({'+'.join(parent_ids)})@{t_now:.3f}"


def resolve_collision(
    intermediates: Sequence[Point],
    llm: LLMExtractor,
    encoder: Encoder,
    t_now: float,
    *,
    anchors: Optional[Sequence[Point]] = None,
    trigger_distance: float = 0.0,
    threshold: float = 0.0,
    cycle_iteration: int = 1,
) -> Optional[CollisionResult]:
    """Factorize N ≥ 2 intermediates into a single child.

    `intermediates` : the points being merged. Their content gets trimmed
                      (Option Z) ; the child holds the common content.

    `anchors`       : optional boundary points. When provided, after the
                      child is created we apply_pull(child, anchor, +1)
                      for each anchor — this positions the new child
                      geometrically between the anchors (Chasles-style).

    Returns None if the LLM produces an empty common (no shared content)
    or if any parent's trimmed text would be empty.
    """
    if len(intermediates) < 2:
        raise ValueError(
            f"resolve_collision requires ≥ 2 intermediates, got {len(intermediates)}"
        )
    # Intra-kind invariant : all intermediates must share the same kind.
    kinds = {p.kind for p in intermediates}
    if len(kinds) != 1:
        raise ValueError(
            f"resolve_collision requires intermediates of a single kind, got {kinds}"
        )
    common_kind = next(iter(kinds))

    texts = [p.content for p in intermediates]
    common_text = llm.extract_common(texts).strip()
    if not common_text:
        return None

    trimmed: List[str] = []
    for p in intermediates:
        t = llm.remove_overlap(p.content, common_text).strip()
        if not t:
            return None
        trimmed.append(t)

    parent_ids = [p.id for p in intermediates]
    child_id = _make_child_id(parent_ids, t_now)
    child = Point(
        id=child_id,
        content=common_text,
        embedding_orig=tuple(encoder.encode(common_text)),
        kind=common_kind,  # child inherits the kind of its parents
        state=EpistemicState.CONJECTURE,
        parents=list(parent_ids),
        lineage_depth=max(p.lineage_depth for p in intermediates) + 1,
    )

    # Mutate parents in place (Option Z)
    d = len(child.embedding_orig)
    for parent, trimmed_text in zip(intermediates, trimmed):
        parent.content = trimmed_text
        parent.embedding_orig = tuple(encoder.encode(trimmed_text))
        parent.delta_active = tuple(0.0 for _ in range(d))
        parent.delta_latent = tuple(0.0 for _ in range(d))
        parent.n_revision += 1
        parent.children = list(parent.children) + [child_id]

    # Chasles : pull child toward boundary anchors (positions the new
    # child geometrically between them, so future kNN finds it on the
    # direct route). One-way : the anchors themselves don't move.
    anchor_list = list(anchors) if anchors else []
    if anchor_list:
        for anchor in anchor_list:
            apply_pull(child, anchor, polarity=+1.0, t_now=t_now,
                       bidirectional=False)

    event = CollisionEvent(
        timestamp=t_now,
        parent_ids=tuple(parent_ids),
        anchor_ids=tuple(a.id for a in anchor_list),
        child_id=child_id,
        trigger_distance=trigger_distance,
        threshold=threshold,
        cycle_iteration=cycle_iteration,
    )
    return CollisionResult(
        child=child, parents=list(intermediates), anchors=anchor_list, event=event
    )


# ---------------------------------------------------------------------------
# Sleep cycle (proximity-based, no anchors)
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
    """One full sleep cycle of proximity-driven collision resolution.

    Mutates `points` in place : new children are appended, parents are
    revised. Pairs are resolved closest-first ; a point cannot be touched
    twice within a single round.
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
            if p_a.id in resolved_ids_this_round or p_b.id in resolved_ids_this_round:
                continue
            result = resolve_collision(
                [p_a, p_b], llm, encoder, t_now,
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
        if detect_collisions(points, t_now):
            report.aborted_for_cascade_limit = True

    return report


# ---------------------------------------------------------------------------
# True merge (deduplication) — "quand l'info est pareille"
# ---------------------------------------------------------------------------
#
# Distinct from fission above : when two same-kind points carry the SAME
# information they collapse into a SINGLE node — the duplicate is dropped,
# NOT kept as a trimmed parent. A genuine duplicate is corroboration, so the
# survivor absorbs the other's counters and gains one corroboration. Pure
# COMPUTATION : no LLM (the manifold's own near-identity is the signal).


@dataclass(frozen=True)
class MergeEvent:
    """One row in the merge audit log : `absorbed_id` folded into
    `keeper_id` because their information was the same."""

    timestamp: float
    keeper_id: str
    absorbed_id: str
    trigger_distance: float
    threshold: float
    detection_class: SourceClass = SourceClass.COMPUTATION


def _normalize_content(text: str) -> str:
    return " ".join((text or "").lower().split())


def detect_identical(
    points: Sequence[Point], t_now: float,
    *, cosine_identity: float = MERGE_COSINE_IDENTITY,
) -> List[Tuple[Point, Point, float]]:
    """Same-kind, eligible pairs whose information is the SAME : identical
    normalized content, OR full-CONTENT-embedding cosine ≥ `cosine_identity`
    (semantic restatement). High precision — merely-related turns are well
    below the cutoff. Sorted most-similar-first."""
    eligible = [
        p for p in points
        if p.state not in {EpistemicState.DEPRECATED, EpistemicState.INVALID}
    ]
    embs = [effective_embedding(p, t_now) for p in eligible]
    norms = [_normalize_content(p.content) for p in eligible]
    pairs: List[Tuple[Point, Point, float]] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            if eligible[i].kind != eligible[j].kind:
                continue
            same_text = norms[i] and norms[i] == norms[j]
            c = cosine(embs[i], embs[j])
            if same_text or c >= cosine_identity:
                pairs.append((eligible[i], eligible[j], c))
    # most-similar first (highest cosine)
    pairs.sort(key=lambda triple: triple[2], reverse=True)
    return pairs


def resolve_merge(keeper: Point, absorbed: Point, t_now: float,
                  *, trigger_distance: float = 0.0,
                  threshold: float = 0.0) -> MergeEvent:
    """Fold `absorbed` into `keeper` (Option Z-free : single surviving node).

    The duplicate IS corroboration : keeper absorbs the other's counters
    and gains one extra corroboration. The caller removes `absorbed` from
    the point list and records the id alias."""
    keeper.n_corrob += absorbed.n_corrob + 1
    keeper.n_contra += absorbed.n_contra
    keeper.n_uses += absorbed.n_uses
    keeper.t_last_obs = max(keeper.t_last_obs, absorbed.t_last_obs, t_now)
    return MergeEvent(
        timestamp=t_now,
        keeper_id=keeper.id,
        absorbed_id=absorbed.id,
        trigger_distance=trigger_distance,
        threshold=threshold,
    )


@dataclass
class MergeReport:
    merged: List[MergeEvent] = field(default_factory=list)
    aliases: dict = field(default_factory=dict)  # absorbed_id -> keeper_id


def merge_duplicates(points: List[Point], t_now: float) -> MergeReport:
    """One pass of true-merge deduplication, closest-first. Mutates
    `points` in place : absorbed duplicates are REMOVED (no parent
    retention). A point is touched at most once per pass."""
    report = MergeReport()
    pairs = detect_identical(points, t_now)
    touched: set[str] = set()
    to_remove: List[Point] = []
    for keeper, absorbed, sim in pairs:
        if keeper.id in touched or absorbed.id in touched:
            continue
        ev = resolve_merge(keeper, absorbed, t_now,
                           trigger_distance=sim,
                           threshold=MERGE_COSINE_IDENTITY)
        report.merged.append(ev)
        report.aliases[absorbed.id] = keeper.id
        touched.add(keeper.id)
        touched.add(absorbed.id)
        to_remove.append(absorbed)
    remove_ids = {p.id for p in to_remove}
    points[:] = [p for p in points if p.id not in remove_ids]
    return report
