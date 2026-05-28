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

from metacog.epistemic import EpistemicState


DEFAULT_OBSERVATOR_ID = "default"


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
