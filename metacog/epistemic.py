"""
MetaCog-Mem — epistemic core.

Implements the function A(·) of the metacognitive layer such that
A(·) ⊥ P (the space of LLM-generated propositions).

Reference: Romanchuk & Bondar, "Semantic Laundering in AI Agent
Architectures" (arXiv:2601.08333). Corollary 5 is enforced at the
type level: GENERATOR-sourced Observations cannot be constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence


class SourceClass(Enum):
    """Epistemic class of an observation source (Def. 7 of the paper)."""

    OBSERVER = "OBSERVER"        # external world state (user turn, clock)
    COMPUTATION = "COMPUTATION"  # deterministic transform of OBSERVERs
    GENERATOR = "GENERATOR"      # LLM-produced proposition (forbidden as obs.)


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
    """A signal that may update node confidence.

    Construction REJECTS GENERATOR at runtime — Corollary 5.
    """

    source: SourceClass
    signal_type: str           # 'reformulation', 'contradiction', ...
    polarity: float            # in [-1, +1]
    target_node_ids: Sequence[str]
    timestamp: float
    raw_content: Optional[str] = None  # the user turn (OBSERVER) or formula (COMPUTATION)

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
    """Every confidence update is logged with its source observation."""

    node_id: str
    observation: Observation
    confidence_before: float
    confidence_after: float
    state_before: EpistemicState
    state_after: EpistemicState


@dataclass
class Node:
    id: str
    content: str            # in P — LLM-generated text
    n_corrob: int = 0       # in ℕ — counter
    n_contra: int = 0       # in ℕ — counter
    n_uses: int = 0         # in ℕ — counter
    state: EpistemicState = EpistemicState.CONJECTURE
    update_log: List[AuditEntry] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Beta-mean. Pure COMPUTATION on internal counters in ℕ."""
        alpha = 1 + self.n_corrob
        beta = 1 + self.n_contra
        return alpha / (alpha + beta)

    @property
    def uncertainty(self) -> float:
        """Beta-variance."""
        alpha = 1 + self.n_corrob
        beta = 1 + self.n_contra
        s = alpha + beta
        return (alpha * beta) / (s * s * (s + 1))


def assign_status(node: Node, population: Sequence[Node]) -> EpistemicState:
    """The function A(·) from §A.2.

    Domain: Node × Population.
    Codomain: EpistemicState.

    Reads ONLY: integer counters and the distribution of confidences
    across population. Never reads any `content` field — A(·) ⊥ P.
    """
    n_corrob = node.n_corrob
    n_contra = node.n_contra
    n_obs = n_corrob + n_contra
    c = node.confidence

    # Cold start: population too small to derive emergent thresholds.
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
    node: Node, observation: Observation, population: Sequence[Node]
) -> AuditEntry:
    """Apply an observation to a node.

    Because Observation() rejects GENERATOR at construction, by the time
    we reach this function, source ∈ {OBSERVER, COMPUTATION}. The audit
    entry records the source for downstream verification.
    """
    if node.id not in observation.target_node_ids:
        raise ValueError(f"observation does not target node {node.id}")

    c_before = node.confidence
    s_before = node.state

    if observation.polarity > 0:
        node.n_corrob += 1
    elif observation.polarity < 0:
        node.n_contra += 1
    # polarity == 0 → no counter change (signal ambiguous)

    node.state = assign_status(node, population)

    entry = AuditEntry(
        node_id=node.id,
        observation=observation,
        confidence_before=c_before,
        confidence_after=node.confidence,
        state_before=s_before,
        state_after=node.state,
    )
    node.update_log.append(entry)
    return entry
