"""
MetaCog-Mem — Observators (pluralisme épistémique).

When multiple sources of expertise use the same substrate, they pull
the same information in opposite directions and the global counters
oscillate without ever reaching equilibrium. The fix : give each
source its OWN view over each point.

  Observator       authorial entity (id, name, expertise keywords)
  ObservatorView   per-(point, observator) counters and state
  default observator      aggregates ALL observations regardless of
                          observator_id, so the default view always
                          reflects the global consensus

Routing (PHASE O3) : `select_observators` ranks observators by cosine
between the query embedding and the observator's keyword embedding.

Audit invariants : an Observator is an AUTHOR, not an epistemic class.
The OBSERVER/COMPUTATION/GENERATOR typing on Observations is untouched.
Cor. 5 still forbids GENERATOR-sourced observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

from metacog.epistemic import DEFAULT_OBSERVATOR_ID, EpistemicState, Point


class Encoder(Protocol):
    def encode(self, text: str) -> Tuple[float, ...]: ...


@dataclass
class Observator:
    """An authorial source of observations.

    The default observator (`id="default"`) is implicit : every system
    automatically has one and it aggregates everything. Named
    observators must be created explicitly by the caller (PHASE O1) or
    automatically by polarization detection (deferred to PHASE O2).
    """

    id: str
    name: str = ""
    keywords: List[str] = field(default_factory=list)
    keywords_embedding: Optional[Tuple[float, ...]] = None
    parent_observator_id: Optional[str] = None
    created_at: float = 0.0
    n_observations_emitted: int = 0

    def ensure_keywords_embedding(self, encoder: Encoder) -> None:
        """Compute and cache keywords_embedding from `keywords`."""
        if self.keywords_embedding is None and self.keywords:
            text = " ".join(self.keywords)
            self.keywords_embedding = tuple(encoder.encode(text))


@dataclass
class ObservatorView:
    """A per-(point, observator) view of the point's epistemic state.

    Counters reflect ONLY observations emitted under this observator's
    name. The view's `state` is derived from these counters via the
    same A(·) used for the default view.
    """

    n_corrob: int = 0
    n_contra: int = 0
    n_uses: int = 0
    n_revision: int = 0
    state: EpistemicState = EpistemicState.CONJECTURE

    @property
    def confidence(self) -> float:
        """Beta-mean with revision dilution — same formula as Point."""
        scale = 1.0 / (1.0 + self.n_revision)
        alpha = 1.0 + self.n_corrob * scale
        beta = 1.0 + self.n_contra * scale
        return alpha / (alpha + beta)


def select_observators(
    query_embedding: Tuple[float, ...],
    observators: Sequence[Observator],
    k: int = 1,
    encoder: Optional[Encoder] = None,
) -> List[Tuple[float, Observator]]:
    """Top-k observators by cosine between query and observator keywords.

    Skips observators that have no keywords (and no pre-computed
    embedding). The caller can use `encoder` to lazily build the
    keyword embedding on the fly.
    """
    from metacog.geometry import cosine

    scored: List[Tuple[float, Observator]] = []
    for obs in observators:
        if obs.keywords_embedding is None and encoder is not None:
            obs.ensure_keywords_embedding(encoder)
        if obs.keywords_embedding is None:
            continue
        score = cosine(query_embedding, obs.keywords_embedding)
        scored.append((score, obs))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# PHASE O2 — polarization detection + auto-spawn
# ---------------------------------------------------------------------------


def polarization_score(point: Point) -> float:
    """Symmetric polarization metric ∈ [0, 0.5] over the point's
    update_log. 0 = all observations same sign, 0.5 = perfect split.

    Formula : min(pos, neg) / (pos + neg).
    """
    pos = sum(1 for e in point.update_log if e.observation.polarity > 0)
    neg = sum(1 for e in point.update_log if e.observation.polarity < 0)
    total = pos + neg
    if total == 0:
        return 0.0
    return min(pos, neg) / total


def detect_polarization(point: Point, *, min_per_side: int = 2) -> bool:
    """A point is considered polarized when at least `min_per_side`
    observations exist in BOTH directions (positive and negative)."""
    pos = sum(1 for e in point.update_log if e.observation.polarity > 0)
    neg = sum(1 for e in point.update_log if e.observation.polarity < 0)
    return pos >= min_per_side and neg >= min_per_side


def spawn_observators_from_polarization(
    point: Point, t_now: float,
) -> List[Observator]:
    """If `point` is polarized and observations were emitted under the
    default observator, auto-spawn one observator per polarity cluster
    and create the matching ObservatorView.

    No-op if the polarization is already attributed to ≥ 2 named
    observators (the conflict is already represented).

    Returns the list of newly spawned Observators (possibly empty).
    """
    if not detect_polarization(point):
        return []

    attributed_ids = {
        e.observation.observator_id for e in point.update_log
        if e.observation.observator_id != DEFAULT_OBSERVATOR_ID
    }
    if len(attributed_ids) >= 2:
        return []

    pos_count = sum(
        1 for e in point.update_log
        if e.observation.polarity > 0
        and e.observation.observator_id == DEFAULT_OBSERVATOR_ID
    )
    neg_count = sum(
        1 for e in point.update_log
        if e.observation.polarity < 0
        and e.observation.observator_id == DEFAULT_OBSERVATOR_ID
    )
    if pos_count == 0 or neg_count == 0:
        return []

    spawned: List[Observator] = []

    pro_id = f"auto-pro@{point.id}@{t_now:.6f}"
    pro = Observator(
        id=pro_id,
        name=f"auto-pro on {point.id}",
        parent_observator_id=DEFAULT_OBSERVATOR_ID,
        created_at=t_now,
    )
    point.observator_views[pro_id] = ObservatorView(n_corrob=pos_count)
    spawned.append(pro)

    con_id = f"auto-con@{point.id}@{t_now:.6f}"
    con = Observator(
        id=con_id,
        name=f"auto-con on {point.id}",
        parent_observator_id=DEFAULT_OBSERVATOR_ID,
        created_at=t_now,
    )
    point.observator_views[con_id] = ObservatorView(n_contra=neg_count)
    spawned.append(con)

    return spawned


# ---------------------------------------------------------------------------
# PHASE O5 — inter-observator delegation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelegationEvent:
    """One row in the delegation audit log."""

    from_observator_id: str
    to_observator_id: str
    timestamp: float
    chain: Tuple[str, ...]
    aborted: bool = False
    abort_reason: str = ""


def delegate_query(
    from_observator: Observator,
    to_observator: Observator,
    query_embedding: Tuple[float, ...],
    points: Sequence[Point],
    k: int,
    t_now: float,
    *,
    max_chain_depth: int = 3,
    chain: Optional[List[str]] = None,
) -> Tuple[List[Tuple[float, Point]], DelegationEvent]:
    """Delegate a query from one observator to another.

    The target observator handles the retrieval using ITS view of each
    point's state. Returns the retrieval results plus an audit event.

    Safety :
      - cycle protection : if `to_observator.id` already appears in
        `chain`, returns an empty result with aborted=True.
      - depth bound      : returns empty if chain length ≥ max_chain_depth.

    The max_chain_depth default of 3 is a math-convention bound (the
    "small-world" tier — most useful delegations are within 3 hops),
    not a tuning knob. Callers can pass a different bound if needed.
    """
    chain = list(chain) if chain else [from_observator.id]

    if to_observator.id in chain:
        return [], DelegationEvent(
            from_observator_id=from_observator.id,
            to_observator_id=to_observator.id,
            timestamp=t_now,
            chain=tuple(chain),
            aborted=True,
            abort_reason="cycle",
        )

    if len(chain) >= max_chain_depth:
        return [], DelegationEvent(
            from_observator_id=from_observator.id,
            to_observator_id=to_observator.id,
            timestamp=t_now,
            chain=tuple(chain),
            aborted=True,
            abort_reason="depth_limit",
        )

    chain.append(to_observator.id)

    from metacog.geometry import retrieve_for_observator
    results = retrieve_for_observator(
        query_embedding, points, k=k, t_now=t_now,
        observator_id=to_observator.id,
    )

    to_observator.n_observations_emitted += 1

    event = DelegationEvent(
        from_observator_id=from_observator.id,
        to_observator_id=to_observator.id,
        timestamp=t_now,
        chain=tuple(chain),
        aborted=False,
    )
    return results, event
