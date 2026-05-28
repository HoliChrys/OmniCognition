"""
MetaCog-Mem — epistemic core (manifold version).

Replaces the previous edge-based `Node` with a `Point` in the embedding
manifold. Relations are no longer stored as edges; they emerge from
proximity in `effective_embedding(point, t)` via kNN.

The function A(·) still depends ONLY on integer counters and the
population's confidence distribution. A(·) ⊥ P holds.

GENERATOR sources cannot construct an Observation — enforced at type
level. Corollary 5 of Romanchuk & Bondar 2026.

POLICY : ZERO DELETION.
    No code path in MetaCog-Mem removes a point from the cloud. Points
    that fall into latent states (INVALID, DEPRECATED) are excluded
    from active recall by `retrieve()`, but remain in the manifold for
    audit and for RESURRECTION — A(·) is reversible, so a previously
    INVALID point that accumulates new corroborations can transition
    back to CORROBORATED automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

Vector = Tuple[float, ...]

DEFAULT_OBSERVATOR_ID = "default"


class SourceClass(Enum):
    """Epistemic class of an observation source (Def. 7)."""

    OBSERVER = "OBSERVER"        # external world state (user turn, clock)
    COMPUTATION = "COMPUTATION"  # deterministic transform of OBSERVERs
    GENERATOR = "GENERATOR"      # LLM-produced (forbidden as evidence)


class PointKind(Enum):
    """Ontological kind of a Point — three coexisting types in the same DB.

    All three share the same schema, A(·), and lifecycle. They differ
    only in what they REPRESENT and in how they typically get
    corroborated :

      FACT     external observation, world-anchored
      THOUGHT  internal derivation / reasoning step
      ACTION   contextualized intervention (replaces the old `skill`
               concept — actions live in the manifold, not in a
               separate registry)
    """

    FACT = "FACT"
    THOUGHT = "THOUGHT"
    ACTION = "ACTION"


class EpistemicState(Enum):
    CONJECTURE = "CONJECTURE"
    CORROBORATED = "CORROBORATED"
    WARRANTED = "WARRANTED"
    SUSPECT = "SUSPECT"
    INVALID = "INVALID"
    DEPRECATED = "DEPRECATED"


class LaunderingError(Exception):
    """Raised when a GENERATOR source tries to enter the observation set O."""


@dataclass(frozen=True)
class Observation:
    """A signal that may update a point's counters AND drive geometric
    pulls. Construction rejects GENERATOR sources — Corollary 5.

    `observator_id` carries the AUTHORIAL attribution. The default
    observator aggregates every observation regardless of attribution;
    a named observator only sees observations it emitted.
    """

    source: SourceClass
    signal_type: str
    polarity: float
    target_node_ids: Sequence[str]
    timestamp: float
    raw_content: Optional[str] = None
    observator_id: str = DEFAULT_OBSERVATOR_ID

    def __post_init__(self) -> None:
        if self.source == SourceClass.GENERATOR:
            raise LaunderingError(
                f"GENERATOR cannot produce Observation (signal={self.signal_type!r}). "
                "Corollary 5 of Romanchuk & Bondar 2026: GENERATOR results must be "
                "excluded from the observation set O."
            )
        if not -1.0 <= self.polarity <= 1.0:
            raise ValueError(f"polarity must be in [-1, +1], got {self.polarity}")


@dataclass
class AuditEntry:
    """Every counter update is logged with its source observation."""

    node_id: str
    observation: Observation
    confidence_before: float
    confidence_after: float
    state_before: EpistemicState
    state_after: EpistemicState


@dataclass
class Point:
    """A point in the manifold. Replaces the previous `Node` concept.

    There are no stored edges. Relations emerge from kNN on the
    `effective_embedding` view computed at query time.

    Fields ∈ P (the LLM proposition space):
        `content`, optionally `keywords` (if extracted by LLM).

    Fields ∈ ℝ^d (vectors — COMPUTATION):
        `embedding_orig` (frozen at ingest, mutated only on collision
        revision under Option Z),
        `delta_active`, `delta_latent` (metacognitive offsets).

    Fields ∈ ℕ (pure counters — read by A(·)):
        `n_corrob`, `n_contra`, `n_uses`, `n_revision`.

    Lineage (collision metadata) :
        `parents`, `children`, `lineage_depth`.
    """

    id: str
    content: str
    embedding_orig: Vector
    kind: PointKind = PointKind.FACT  # FACT / THOUGHT / ACTION
    delta_active: Vector = field(default_factory=tuple)
    delta_latent: Vector = field(default_factory=tuple)
    t_last_obs: float = 0.0
    n_corrob: int = 0
    n_contra: int = 0
    n_uses: int = 0
    n_revision: int = 0  # Option Z : incremented on collision revision
    state: EpistemicState = EpistemicState.CONJECTURE
    keywords: List[str] = field(default_factory=list)
    keywords_source: Optional[SourceClass] = None
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    lineage_depth: int = 0
    # Sequence neighbors for ordered ingestion (conversation turns,
    # document paragraphs, code lines, …). Optional ; set by the
    # ingester when the source data has intrinsic sequential order.
    sequence_prev: Optional[str] = None
    sequence_next: Optional[str] = None
    update_log: List[AuditEntry] = field(default_factory=list)
    execution_log: list = field(default_factory=list)  # for kind=ACTION
    # Per-observator views (default view = the point's own counters).
    # Created on demand the first time a named observator touches the point.
    observator_views: Dict[str, "object"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Zero-initialize deltas to match embedding dimension if not given.
        d = len(self.embedding_orig)
        if not self.delta_active:
            self.delta_active = tuple(0.0 for _ in range(d))
        if not self.delta_latent:
            self.delta_latent = tuple(0.0 for _ in range(d))
        if len(self.delta_active) != d or len(self.delta_latent) != d:
            raise ValueError(
                f"delta dimensions must match embedding_orig (d={d}), got "
                f"active={len(self.delta_active)}, latent={len(self.delta_latent)}"
            )

    @property
    def confidence(self) -> float:
        """Beta-mean with revision dilution (Option Z).

        Effective alpha/beta are divided by (1 + n_revision) so that
        each revision INCREASES the uncertainty without losing the
        ratio between corroborations and contradictions.
        """
        scale = 1.0 / (1.0 + self.n_revision)
        alpha = 1.0 + self.n_corrob * scale
        beta = 1.0 + self.n_contra * scale
        return alpha / (alpha + beta)

    @property
    def uncertainty(self) -> float:
        """Beta-variance with revision dilution (Option Z)."""
        scale = 1.0 / (1.0 + self.n_revision)
        alpha = 1.0 + self.n_corrob * scale
        beta = 1.0 + self.n_contra * scale
        s = alpha + beta
        return (alpha * beta) / (s * s * (s + 1))


def assign_status(
    point: Point,
    population: Sequence[Point],
    *,
    observator_id: str = DEFAULT_OBSERVATOR_ID,
) -> EpistemicState:
    """A(·) — reads ONLY integer counters and the population's confidence
    distribution. Never reads `content`, `embedding_orig`, or any delta.
    Therefore A(·) ⊥ P.

    When `observator_id` is "default", reads the point's main counters.
    For a named observator, reads the corresponding ObservatorView (or
    empty counters if the view doesn't exist yet).
    """
    if observator_id == DEFAULT_OBSERVATOR_ID:
        n_corrob = point.n_corrob
        n_contra = point.n_contra
        c = point.confidence
    else:
        view = point.observator_views.get(observator_id)
        if view is None:
            n_corrob = 0
            n_contra = 0
        else:
            n_corrob = view.n_corrob
            n_contra = view.n_contra
        scale = 1.0 / (1.0 + (view.n_revision if view else 0))
        alpha = 1.0 + n_corrob * scale
        beta = 1.0 + n_contra * scale
        c = alpha / (alpha + beta)
    n_obs = n_corrob + n_contra

    if len(population) < 5:
        if n_contra >= 2 and n_corrob < n_contra:
            return EpistemicState.INVALID
        if n_contra > n_corrob:
            return EpistemicState.SUSPECT
        if n_corrob >= 1:
            return EpistemicState.CORROBORATED
        return EpistemicState.CONJECTURE

    confidences = sorted(w.confidence for w in population)
    n = len(confidences)
    med = confidences[n // 2]
    mean_c = sum(confidences) / n
    var_c = sum((x - mean_c) ** 2 for x in confidences) / n
    sigma = math.sqrt(var_c)
    threshold_obs = max(2, int(med * 10))

    if n_contra >= 2 and n_corrob < n_contra:
        return EpistemicState.INVALID
    if c > med + sigma and n_obs >= threshold_obs:
        return EpistemicState.WARRANTED
    if n_contra > n_corrob:
        return EpistemicState.SUSPECT
    if n_corrob >= 1:
        return EpistemicState.CORROBORATED
    return EpistemicState.CONJECTURE


def apply_observation(
    point: Point, observation: Observation, population: Sequence[Point]
) -> AuditEntry:
    """Single-point counter and state update.

    To trigger geometric pulls between multiple points participating in
    the same observation, call `metacog.geometry.apply_pull` BEFORE
    this function (it reads `t_last_obs` which this function overwrites).
    See `process_observation` for the full orchestration.
    """
    if point.id not in observation.target_node_ids:
        raise ValueError(f"observation does not target point {point.id}")

    c_before = point.confidence
    s_before = point.state

    # --- default view (aggregates every observation) ---
    if observation.polarity > 0:
        point.n_corrob += 1
    elif observation.polarity < 0:
        point.n_contra += 1
    # polarity == 0 → ambiguous, no counter change

    point.t_last_obs = observation.timestamp
    point.state = assign_status(point, population)

    # --- named observator view (only its own observations) ---
    if observation.observator_id != DEFAULT_OBSERVATOR_ID:
        from metacog.observator import ObservatorView
        view = point.observator_views.get(observation.observator_id)
        if view is None:
            view = ObservatorView()
            point.observator_views[observation.observator_id] = view
        if observation.polarity > 0:
            view.n_corrob += 1
        elif observation.polarity < 0:
            view.n_contra += 1
        view.state = assign_status(
            point, population,
            observator_id=observation.observator_id,
        )

    # Geometric exile : the éloignement that REPLACES suppression.
    # Latent points are pushed away from the active centroid, but
    # never deleted (ZERO-DELETION policy).
    if point.state in {EpistemicState.INVALID, EpistemicState.DEPRECATED}:
        from metacog.geometry import apply_exile
        apply_exile(point, population, observation.timestamp)

    entry = AuditEntry(
        node_id=point.id,
        observation=observation,
        confidence_before=c_before,
        confidence_after=point.confidence,
        state_before=s_before,
        state_after=point.state,
    )
    point.update_log.append(entry)
    return entry


def process_observation(
    points: Sequence[Point],
    observation: Observation,
    population: Sequence[Point],
) -> List[AuditEntry]:
    """Full orchestration: apply pairwise geometric pulls between every
    pair of target points, then update counters and state for each.

    Geometric pulls use `t_last_obs` for lazy decay — they MUST run
    before `apply_observation` overwrites that field.
    """
    from metacog.geometry import apply_pull

    targets = [p for p in points if p.id in observation.target_node_ids]
    if len(targets) != len(observation.target_node_ids):
        missing = set(observation.target_node_ids) - {p.id for p in targets}
        raise ValueError(f"missing target points: {missing}")

    # Pairwise geometric pulls (uses old t_last_obs)
    if len(targets) >= 2:
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                apply_pull(targets[i], targets[j], observation.polarity, observation.timestamp)

    # Per-point counter and state updates (overwrites t_last_obs)
    return [apply_observation(p, observation, population) for p in targets]
