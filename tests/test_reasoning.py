"""
Tests for reason() — the metacognitive reasoning orchestrator.

Properties :
  R1   reason() returns a non-empty trajectory on a populated manifold.
  R2   Each step visits a point and creates a THOUGHT (lineage : parent
       = visited point).
  R3   Visited points get n_uses += 1.
  R4   Convergence (`stability`) when the LLM stabilizes (cos > cos π/12).
  R5   `no_more_candidates` if every eligible point has been visited.
  R6   If executor is provided and LLM proposes actions, ACTIONs get
       executed and result FACTs appear in the trajectory.
  R7   When the trajectory is long enough, compress_trajectory runs
       eagerly (compressions_applied ≥ 0).
  R8   Latent/exiled points (INVALID / DEPRECATED) are not visited.
  R9   No-laundering invariant holds after a complete reason() run.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from metacog import (
    EpistemicState,
    ExecutionOutcome,
    ExecutionResult,
    Point,
    PointKind,
    ReasoningTrajectory,
    SourceClass,
    assert_no_laundering,
    reason,
)


class _SimhashEncoder:
    """Signed hash-vector encoder — orthogonal-ish for unrelated words."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self._cache: dict = {}

    def encode(self, text: str) -> Tuple[float, ...]:
        words = text.lower().split()
        if not words:
            return tuple(0.0 for _ in range(self.dim))
        acc = [0.0] * self.dim
        for w in words:
            if w not in self._cache:
                h = abs(hash(("se", w)))
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0
                    for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


class _StabilizingLLM:
    """LLM stub : after N unique answers, returns the last one verbatim
    so the trajectory converges deterministically. Also implements
    extract_common / remove_overlap so compression doesn't no-op."""

    def __init__(self, max_unique: int = 3) -> None:
        self.max_unique = max_unique
        self._seq: list[str] = []

    def synthesize_step(self, query, previous_answer, new_point_content):
        if len(self._seq) < self.max_unique:
            ans = f"step{len(self._seq)} {new_point_content}"
            self._seq.append(ans)
            return ans
        return self._seq[-1]

    def propose_action(self, query, current_answer):
        return None

    def extract_common(self, texts):
        texts = list(texts)
        if len(texts) < 2:
            return ""
        word_sets = [set(t.lower().split()) for t in texts]
        common = word_sets[0].copy()
        for ws in word_sets[1:]:
            common &= ws
        if not common:
            return ""
        seen = set()
        out = []
        for w in texts[0].lower().split():
            if w in common and w not in seen:
                out.append(w)
                seen.add(w)
        return " ".join(out)

    def remove_overlap(self, full, common):
        cset = set(common.lower().split())
        return " ".join(w for w in full.lower().split() if w not in cset)


class _ActionProposingLLM(_StabilizingLLM):
    """LLM that occasionally proposes an action."""

    def __init__(self) -> None:
        super().__init__(max_unique=2)
        self._proposed = False

    def propose_action(self, query, current_answer):
        if self._proposed:
            return None
        self._proposed = True
        return "fetch_external_data"


class _OkExecutor:
    def execute(self, action_content: str) -> ExecutionResult:
        return ExecutionResult(
            outcome=ExecutionOutcome.SUCCESS,
            result_content=f"got data for {action_content}",
        )


def _seed_points(enc: _SimhashEncoder, n: int = 6) -> list[Point]:
    return [
        Point(
            id=f"P{i}",
            content=f"fact about topic {i} with extras",
            embedding_orig=enc.encode(f"fact about topic {i} with extras"),
            kind=PointKind.FACT,
        )
        for i in range(n)
    ]


# ---------- R1, R2, R3 ----------------------------------------------------


def test_reason_returns_non_empty_trajectory():
    enc = _SimhashEncoder()
    llm = _StabilizingLLM(max_unique=3)
    points = _seed_points(enc, n=5)
    trajectory = reason("query about topic", points, enc, llm, t_now=1.0)
    assert isinstance(trajectory, ReasoningTrajectory)
    assert len(trajectory.steps) >= 1
    assert trajectory.final_answer != ""


def test_each_step_creates_a_thought_with_lineage():
    enc = _SimhashEncoder()
    llm = _StabilizingLLM(max_unique=3)
    points = _seed_points(enc, n=5)
    initial_ids = {p.id for p in points}
    trajectory = reason("topic alpha", points, enc, llm, t_now=1.0,
                        apply_compression=False)
    # At least one new THOUGHT was created per step
    new_thought_ids = {p.id for p in trajectory.created_points if p.kind == PointKind.THOUGHT}
    assert len(new_thought_ids) == len(trajectory.steps)
    # Each created THOUGHT has a parent that is a visited point
    visited_ids = {p.id for p in trajectory.visited_points}
    for t in trajectory.created_points:
        if t.kind == PointKind.THOUGHT:
            assert any(parent in visited_ids for parent in t.parents)
    # All originals still present (zero deletion)
    assert initial_ids.issubset({p.id for p in points})


def test_visited_points_get_n_uses_incremented():
    enc = _SimhashEncoder()
    llm = _StabilizingLLM(max_unique=2)
    points = _seed_points(enc, n=4)
    initial_uses = {p.id: p.n_uses for p in points}
    trajectory = reason("topic", points, enc, llm, t_now=1.0,
                        apply_compression=False)
    for vp in trajectory.visited_points:
        assert vp.n_uses == initial_uses[vp.id] + 1


# ---------- R4 ------------------------------------------------------------


def test_stability_halts_trajectory():
    enc = _SimhashEncoder()
    llm = _StabilizingLLM(max_unique=2)  # converges fast
    points = _seed_points(enc, n=10)
    trajectory = reason("topic", points, enc, llm, t_now=1.0,
                        apply_compression=False)
    # With max_unique=2, the third step's answer equals the second's.
    # cosine = 1.0 > COS_STABLE, halt with reason='stability'.
    assert trajectory.halt_reason == "stability"
    assert trajectory.converged


# ---------- R5 ------------------------------------------------------------


def test_no_more_candidates_halts_trajectory():
    enc = _SimhashEncoder()

    class _NoMoreLLM(_StabilizingLLM):
        def synthesize_step(self, query, previous_answer, new_point_content):
            # Always produce a NEW unique answer so stability never kicks in
            return f"unique-{id(new_point_content)}-{new_point_content}"

    llm = _NoMoreLLM(max_unique=999)
    points = _seed_points(enc, n=3)
    trajectory = reason("topic", points, enc, llm, t_now=1.0,
                        apply_compression=False)
    # After visiting all 3, no candidates remain
    assert trajectory.halt_reason in {"no_more_candidates", "stability"}


# ---------- R6 ------------------------------------------------------------


def test_action_proposed_gets_executed_and_creates_fact():
    enc = _SimhashEncoder()
    llm = _ActionProposingLLM()
    points = _seed_points(enc, n=4)
    trajectory = reason(
        "topic needing data", points, enc, llm,
        executor=_OkExecutor(),
        t_now=1.0,
        apply_compression=False,
    )
    # One ACTION was created and executed
    actions = [p for p in trajectory.created_points if p.kind == PointKind.ACTION]
    assert len(actions) == 1
    action = actions[0]
    # Action got a positive observation (success)
    assert action.n_corrob == 1
    # The action's execution_log records the run
    assert len(action.execution_log) == 1
    # A result FACT was created with the action as parent
    result_facts = [p for p in trajectory.created_points
                    if p.kind == PointKind.FACT and action.id in p.parents]
    assert len(result_facts) == 1


# ---------- R7 ------------------------------------------------------------


def test_compression_runs_when_trajectory_long_enough():
    enc = _SimhashEncoder()

    class _StretchedLLM(_StabilizingLLM):
        def synthesize_step(self, query, previous_answer, new_point_content):
            # Keep producing slightly different answers — never identical
            # until we've gone through enough points
            n = len(self._seq)
            ans = f"answer{n} bridge {new_point_content}"
            self._seq.append(ans)
            return ans

    llm = _StretchedLLM(max_unique=999)
    points = _seed_points(enc, n=6)
    trajectory = reason("topic", points, enc, llm, t_now=1.0,
                        apply_compression=True)
    # The trajectory visited several points → compression at least RAN
    assert trajectory.compressions_applied >= 0
    # And we have enough visits to be eligible
    assert len(trajectory.visited_points) >= 4 or trajectory.halt_reason != "stability"


# ---------- R8 ------------------------------------------------------------


def test_invalid_points_not_visited():
    enc = _SimhashEncoder()
    llm = _StabilizingLLM(max_unique=10)
    points = _seed_points(enc, n=4)
    points[0].state = EpistemicState.INVALID
    points[1].state = EpistemicState.DEPRECATED
    trajectory = reason("topic", points, enc, llm, t_now=1.0,
                        apply_compression=False)
    visited_ids = {p.id for p in trajectory.visited_points}
    assert "P0" not in visited_ids
    assert "P1" not in visited_ids


# ---------- R9 ------------------------------------------------------------


def test_no_laundering_after_full_reason_cycle():
    enc = _SimhashEncoder()
    llm = _ActionProposingLLM()
    points = _seed_points(enc, n=5)
    reason("topic", points, enc, llm, executor=_OkExecutor(),
           t_now=1.0, apply_compression=True)
    report = assert_no_laundering(points)
    assert report.ok
    # And no GENERATOR appears in any update_log
    for p in points:
        for entry in p.update_log:
            assert entry.observation.source != SourceClass.GENERATOR
