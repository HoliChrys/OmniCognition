"""
Tests for ACTION execution.

Properties :
  X1   Calling execute_action on a non-ACTION point raises ValueError.
  X2   A successful execution increments n_corrob on the action.
  X3   A failed execution increments n_contra on the action.
  X4   Repeated successes → action state moves to CORROBORATED/WARRANTED.
  X5   Repeated failures → action state moves to INVALID.
  X6   On failure, the action gets exiled geometrically (delta_active ≠ 0).
  X7   When encoder is provided AND result has content, a FACT point is
       created with parents=[action.id].
  X8   ExecutionRecord is appended to action.execution_log.
  X9   No-laundering : the observation source is OBSERVER, not GENERATOR.
"""

from __future__ import annotations

from typing import Tuple

import pytest

from metacog import (
    EpistemicState,
    ExecutionOutcome,
    ExecutionResult,
    Point,
    PointKind,
    SourceClass,
    assert_no_laundering,
    execute_action,
    vec_norm,
)


class _Encoder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self._cache: dict = {}

    def encode(self, text: str) -> Tuple[float, ...]:
        words = text.lower().split()
        if not words:
            return tuple(0.0 for _ in range(self.dim))
        acc = [0.0] * self.dim
        for w in words:
            if w not in self._cache:
                h = abs(hash(("e", w)))
                self._cache[w] = tuple(
                    ((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        return tuple(v / len(words) for v in acc)


class _SuccessExecutor:
    def execute(self, action_content: str) -> ExecutionResult:
        return ExecutionResult(
            outcome=ExecutionOutcome.SUCCESS,
            result_content=f"ok: {action_content}",
        )


class _FailureExecutor:
    def execute(self, action_content: str) -> ExecutionResult:
        return ExecutionResult(
            outcome=ExecutionOutcome.FAILURE,
            result_content="",
            error="simulated failure",
        )


# ---------- X1 -------------------------------------------------------------


def test_execute_action_rejects_non_action_kind():
    enc = _Encoder()
    fact = Point(id="F1", content="some fact",
                 embedding_orig=enc.encode("some fact"),
                 kind=PointKind.FACT)
    with pytest.raises(ValueError):
        execute_action(fact, _SuccessExecutor(), [fact], t_now=0.0)


# ---------- X2, X3 --------------------------------------------------------


def test_success_increments_n_corrob():
    enc = _Encoder()
    action = Point(id="A1", content="do thing",
                   embedding_orig=enc.encode("do thing"),
                   kind=PointKind.ACTION)
    pop = [action]
    obs, _ = execute_action(action, _SuccessExecutor(), pop, t_now=1.0)
    assert action.n_corrob == 1
    assert obs.polarity > 0
    assert obs.signal_type == "execution_success"


def test_failure_increments_n_contra():
    enc = _Encoder()
    action = Point(id="A1", content="do thing",
                   embedding_orig=enc.encode("do thing"),
                   kind=PointKind.ACTION)
    pop = [action]
    obs, _ = execute_action(action, _FailureExecutor(), pop, t_now=1.0)
    assert action.n_contra == 1
    assert obs.polarity < 0
    assert obs.signal_type == "execution_failure"


# ---------- X4 -------------------------------------------------------------


def test_repeated_success_promotes_state():
    enc = _Encoder()
    pop = [
        Point(id=f"P{i}", content=f"filler {i}",
              embedding_orig=enc.encode(f"filler {i}"))
        for i in range(5)
    ]
    action = Point(id="A1", content="do thing",
                   embedding_orig=enc.encode("do thing"),
                   kind=PointKind.ACTION)
    pop.append(action)
    for t in range(10):
        execute_action(action, _SuccessExecutor(), pop, t_now=float(t))
    assert action.state in {EpistemicState.CORROBORATED, EpistemicState.WARRANTED}
    assert action.n_corrob == 10
    assert action.n_contra == 0


# ---------- X5, X6 --------------------------------------------------------


def test_repeated_failure_invalidates_and_exiles():
    enc = _Encoder()
    pop = [
        Point(id=f"P{i}", content=f"filler {i}",
              embedding_orig=enc.encode(f"filler {i}"))
        for i in range(5)
    ]
    action = Point(id="A1", content="do thing",
                   embedding_orig=enc.encode("do thing"),
                   kind=PointKind.ACTION)
    pop.append(action)
    for t in range(5):
        execute_action(action, _FailureExecutor(), pop, t_now=float(t))
    assert action.state == EpistemicState.INVALID
    # Exile pushed delta_active away from centroid
    assert vec_norm(action.delta_active) > 0


# ---------- X7 -------------------------------------------------------------


def test_result_fact_created_with_lineage():
    enc = _Encoder()
    action = Point(id="A1", content="search paris weather",
                   embedding_orig=enc.encode("search paris weather"),
                   kind=PointKind.ACTION)
    pop = [action]
    obs, result_fact = execute_action(
        action, _SuccessExecutor(), pop, t_now=1.0, encoder=enc,
    )
    assert result_fact is not None
    assert result_fact.kind == PointKind.FACT
    assert action.id in result_fact.parents
    assert result_fact.id in action.children
    assert result_fact in pop


def test_no_result_fact_without_encoder():
    enc = _Encoder()
    action = Point(id="A1", content="do thing",
                   embedding_orig=enc.encode("do thing"),
                   kind=PointKind.ACTION)
    pop = [action]
    _obs, result_fact = execute_action(
        action, _SuccessExecutor(), pop, t_now=1.0, encoder=None,
    )
    assert result_fact is None


# ---------- X8 -------------------------------------------------------------


def test_execution_log_records_calls():
    enc = _Encoder()
    action = Point(id="A1", content="do thing",
                   embedding_orig=enc.encode("do thing"),
                   kind=PointKind.ACTION)
    pop = [action]
    for t in range(3):
        execute_action(action, _SuccessExecutor(), pop, t_now=float(t),
                       encoder=enc)
    assert len(action.execution_log) == 3
    for rec in action.execution_log:
        assert rec.outcome == ExecutionOutcome.SUCCESS
        assert rec.result_fact_id is not None


# ---------- X9 -------------------------------------------------------------


def test_no_laundering_after_executions():
    enc = _Encoder()
    pop = [
        Point(id=f"P{i}", content=f"f{i}",
              embedding_orig=enc.encode(f"f{i}"))
        for i in range(4)
    ]
    action = Point(id="A1", content="x",
                   embedding_orig=enc.encode("x"),
                   kind=PointKind.ACTION)
    pop.append(action)
    for t in range(4):
        execute_action(action, _SuccessExecutor(), pop, t_now=float(t),
                       encoder=enc)
    for t in range(2):
        execute_action(action, _FailureExecutor(), pop, t_now=float(t + 10))
    report = assert_no_laundering(pop)
    assert report.ok
    # All execution observations are OBSERVER-class
    for entry in action.update_log:
        assert entry.observation.source == SourceClass.OBSERVER
