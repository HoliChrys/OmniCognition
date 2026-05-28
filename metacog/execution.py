"""
MetaCog-Mem — ACTION execution.

Executes a Point of kind=ACTION via an injected Executor, observes
the outcome, and feeds it back as an Observation on the action
(EXECUTION_SUCCESS or EXECUTION_FAILURE). The result content can be
captured as a new FACT linked to the action via lineage.

Epistemic typing :
  - The Executor is treated as OBSERVER : its outputs (outcome +
    content) report what HAPPENED in the world. They are not
    GENERATOR — they describe an effect we caused, not a proposition
    we invented.
  - The resulting Observation has source=OBSERVER.
  - The resulting FACT (if created) carries content ∈ P, but is
    typed as a FACT kind (anchored in observed outcome).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol, Tuple

from metacog.epistemic import (
    Observation,
    Point,
    PointKind,
    SourceClass,
    apply_observation,
)


class ExecutionOutcome(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"


@dataclass
class ExecutionResult:
    """Returned by an ExecutorProtocol.execute() call."""

    outcome: ExecutionOutcome
    result_content: str = ""
    error: Optional[str] = None


@dataclass
class ExecutionRecord:
    """One row in a Point's execution_log (for kind=ACTION)."""

    timestamp: float
    outcome: ExecutionOutcome
    result_content: str
    result_fact_id: Optional[str] = None
    error: Optional[str] = None


class ExecutorProtocol(Protocol):
    """Injected executor interfacing the external world.

    Treated as an OBSERVER : `execute()` reports what happened, not
    what we wished. The polarity (success/failure) is therefore a
    valid evidence channel for the ACTION's counters.
    """

    def execute(self, action_content: str) -> ExecutionResult: ...


class Encoder(Protocol):
    def encode(self, text: str) -> Tuple[float, ...]: ...


def execute_action(
    action: Point,
    executor: ExecutorProtocol,
    points: List[Point],
    t_now: float,
    *,
    encoder: Optional[Encoder] = None,
) -> Tuple[Observation, Optional[Point]]:
    """Execute an ACTION point.

    Steps :
      1. Call executor.execute(action.content) → ExecutionResult.
      2. Append an ExecutionRecord to action.execution_log.
      3. If `encoder` is provided AND there is a result_content,
         create a FACT point with `parents=[action.id]`.
      4. Apply an EXECUTION_SUCCESS / EXECUTION_FAILURE Observation
         on the action (updates counters, may trigger state change
         and geometric exile).

    Returns (observation, result_fact) — `result_fact` is None if no
    encoder was provided or the result_content was empty.

    Raises ValueError if action.kind is not PointKind.ACTION.
    """
    if action.kind != PointKind.ACTION:
        raise ValueError(
            f"execute_action requires kind=ACTION, got {action.kind}"
        )

    result = executor.execute(action.content)

    polarity = +1.0 if result.outcome == ExecutionOutcome.SUCCESS else -1.0
    signal = (
        "execution_success" if result.outcome == ExecutionOutcome.SUCCESS
        else "execution_failure"
    )

    # Optional result FACT (linked via lineage)
    result_fact: Optional[Point] = None
    if encoder is not None and result.result_content:
        result_fact_id = f"result({action.id})@{t_now:.3f}"
        result_fact = Point(
            id=result_fact_id,
            content=result.result_content,
            embedding_orig=tuple(encoder.encode(result.result_content)),
            kind=PointKind.FACT,
            parents=[action.id],
            lineage_depth=action.lineage_depth + 1,
        )
        points.append(result_fact)
        action.children = list(action.children) + [result_fact_id]

    # Log
    record = ExecutionRecord(
        timestamp=t_now,
        outcome=result.outcome,
        result_content=result.result_content,
        result_fact_id=result_fact.id if result_fact else None,
        error=result.error,
    )
    action.execution_log.append(record)

    # Observation : the world (OBSERVER) reports the result
    obs = Observation(
        source=SourceClass.OBSERVER,
        signal_type=signal,
        polarity=polarity,
        target_node_ids=[action.id],
        timestamp=t_now,
        raw_content=result.result_content,
    )
    apply_observation(action, obs, population=points)

    return obs, result_fact
