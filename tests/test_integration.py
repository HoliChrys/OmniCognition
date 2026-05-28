"""
End-to-end integration test.

Stitches the whole MetaCog-Mem pipeline together on a single scenario:

  1. INGEST   a handful of FACTs into the manifold.
  2. CONVERSE process user turns — detectors fire, observations update
              counters and geometry (pulls, exile).
  3. SLEEP    run a sleep cycle of collisions on the resulting cloud.
  4. REASON   call reason() on a query, with an executor that handles
              proposed ACTIONs.
  5. AUDIT    verify zero deletion, no laundering, A(·) ⊥ P invariants
              hold across the full pipeline.

This is the smoke-test of the system : if the modules don't compose,
it fails. If it passes, the system is end-to-end functional.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Tuple

from metacog import (
    ConversationLog,
    EpistemicState,
    ExecutionOutcome,
    ExecutionResult,
    Observation,
    Point,
    PointKind,
    SourceClass,
    TurnRecord,
    analyze_user_turn,
    apply_observation,
    assert_no_laundering,
    assign_status,
    inputs_of_A,
    process_observation,
    reason,
    sleep_cycle_collisions,
)


# ---------------------------------------------------------------------------
# Stubs (production code would inject real implementations)
# ---------------------------------------------------------------------------


class _Encoder:
    """Signed simhash : unrelated words → orthogonal-ish vectors."""

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
                h = int(hashlib.md5(f"e2e:{w}".encode()).hexdigest(), 16)
                self._cache[w] = tuple(
                    1.0 if (h >> i) & 1 else -1.0 for i in range(self.dim)
                )
            for i in range(self.dim):
                acc[i] += self._cache[w][i]
        norm = math.sqrt(sum(x * x for x in acc))
        if norm < 1e-12:
            return tuple(0.0 for _ in range(self.dim))
        return tuple(x / norm for x in acc)


class _LLM:
    """Unified stub : reasoning + compression + collision LLM methods."""

    def __init__(self) -> None:
        self._uniq = 0
        self._proposed = False

    def synthesize_step(
        self, query: str, previous_answer: Optional[str], new_point_content: str
    ) -> str:
        self._uniq += 1
        if self._uniq >= 3:
            # converge after 3 unique answers — return the last verbatim
            return previous_answer or new_point_content
        return f"step{self._uniq}: {new_point_content}"

    def propose_action(
        self, query: str, current_answer: str
    ) -> Optional[str]:
        if self._proposed:
            return None
        self._proposed = True
        return "lookup external"

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


class _Executor:
    def execute(self, action_content: str) -> ExecutionResult:
        return ExecutionResult(
            outcome=ExecutionOutcome.SUCCESS,
            result_content=f"resolved: {action_content}",
        )


# ---------------------------------------------------------------------------
# End-to-end scenario
# ---------------------------------------------------------------------------


def test_end_to_end_pipeline_runs_and_preserves_invariants():
    enc = _Encoder()
    llm = _LLM()
    executor = _Executor()

    # STEP 1 — INGEST initial FACTs
    facts_text = [
        "dr sarah met at adoption conference 2023",
        "dr sarah encouraged counseling career",
        "user lives in berkeley california",
        "user attended therapy training program",
        "user enjoys outdoor activities hiking",
    ]
    points = [
        Point(
            id=f"FACT_{i}",
            content=text,
            embedding_orig=enc.encode(text),
            kind=PointKind.FACT,
        )
        for i, text in enumerate(facts_text)
    ]
    initial_ids = {p.id for p in points}
    initial_count = len(points)

    # STEP 2 — CONVERSE : process several user turns
    log = ConversationLog()
    turns = [
        (1.0, "user", "who is dr sarah counseling career", []),
        (2.0, "assistant", "dr sarah encouraged counseling career",
         ["FACT_1"]),
        # corroboration : shares "dr sarah counseling career" + extends with
        # "adoption conference" → both overlap and extension
        (3.0, "user", "dr sarah counseling career and also adoption conference",
         []),
        (4.0, "assistant", "you met dr sarah at adoption conference 2023",
         ["FACT_0"]),
        (5.0, "user", "where do i live california",
         []),
        (6.0, "assistant", "user lives berkeley california",
         ["FACT_2"]),
        # contradiction : "no" + overlap on "berkeley california"
        (7.0, "user", "no not berkeley california wrong",
         []),
    ]
    for t, speaker, text, retrieved in turns:
        turn = TurnRecord(
            timestamp=t,
            speaker=speaker,
            text=text,
            embedding=enc.encode(text),
            retrieved_point_ids=list(retrieved),
        )
        log.record(turn)
        if speaker == "user":
            observations = analyze_user_turn(turn, log, points, encoder=enc)
            for obs in observations:
                process_observation(points, obs, population=points)

    # Some FACTs should now have non-zero counters from detected signals
    touched_counters_total = sum(p.n_corrob + p.n_contra for p in points)
    assert touched_counters_total > 0

    # STEP 3 — SLEEP cycle (proximity collisions, if any)
    report = sleep_cycle_collisions(points, llm, enc, t_now=10.0)
    # Sleep cycle should not error out — report is well-formed
    assert isinstance(report.iterations, int)
    # Original points still all present (zero deletion)
    current_ids = {p.id for p in points}
    assert initial_ids.issubset(current_ids)

    # STEP 4 — REASON on a multi-hop query
    trajectory = reason(
        "where did i first meet dr sarah",
        points, enc, llm,
        executor=executor,
        t_now=20.0,
        apply_compression=True,
    )
    # Trajectory was generated
    assert len(trajectory.steps) >= 1
    # THOUGHTs were created
    thoughts = [p for p in trajectory.created_points
                if p.kind == PointKind.THOUGHT]
    assert len(thoughts) == len(trajectory.steps)
    # Possibly an ACTION was proposed and executed (LLM proposes once)
    actions = [p for p in trajectory.created_points
               if p.kind == PointKind.ACTION]
    if actions:
        # Action executed → has a corroboration
        assert actions[0].n_corrob >= 1 or actions[0].n_contra >= 1
        # And there's a result FACT linked back
        result_facts = [p for p in trajectory.created_points
                        if p.kind == PointKind.FACT
                        and actions[0].id in p.parents]
        assert len(result_facts) >= 1

    # STEP 5 — AUDIT invariants

    # No laundering across the entire pipeline
    audit_report = assert_no_laundering(points)
    assert audit_report.ok
    assert audit_report.by_source.get(SourceClass.GENERATOR, 0) == 0

    # Zero deletion : all originals still present
    final_ids = {p.id for p in points}
    assert initial_ids.issubset(final_ids)
    # The cloud only GREW (thoughts, possibly actions, possibly children)
    assert len(points) >= initial_count

    # A(·) ⊥ P : the function only reads integer counters
    sample = points[0]
    inputs = inputs_of_A(sample)
    for k, v in inputs.items():
        assert isinstance(v, int), f"A(·) input {k!r} is not an int: {v!r}"
    assert "content" not in inputs
    assert "embedding_orig" not in inputs
    assert "delta_active" not in inputs
    assert "delta_latent" not in inputs

    # State transitions happened : at least one point is no longer
    # CONJECTURE (some signal was detected).
    states_observed = {p.state for p in points if p.id in initial_ids}
    assert states_observed != {EpistemicState.CONJECTURE}


# ---------------------------------------------------------------------------
# Smoke test — every public symbol importable
# ---------------------------------------------------------------------------


def test_public_api_is_importable():
    """If a module import is broken, this will fail at collection time."""
    from metacog import (  # noqa: F401
        AuditEntry, EpistemicState, LaunderingError, Observation, Point,
        PointKind, SourceClass, apply_observation, assign_status,
        process_observation,
        # geometry
        apply_exile, apply_pull, compute_centroid, cosine, decay_factor,
        distance, effective_embedding, k_nearest, retrieve, vec_norm,
        # collision
        CollisionEvent, CollisionResult, Encoder, LLMExtractor,
        MAX_CASCADE_ITERATIONS, SleepCycleReport, collision_threshold,
        detect_collisions, resolve_collision, sleep_cycle_collisions,
        # compression
        CompressionEvent, CompressionOpportunity, CompressionResult,
        EagerCompressionReport, compress_chain, compress_trajectory,
        detect_chasles_opportunities, straight_ratio,
        # detectors
        ConversationLog, TurnRecord, analyze_user_turn,
        detect_contradiction, detect_continuation, detect_corroboration,
        detect_reformulation, detect_repetition_of_need,
        # execution
        ExecutionOutcome, ExecutionRecord, ExecutionResult,
        ExecutorProtocol, execute_action,
        # reasoning
        COS_STABLE, ReasoningLLM, ReasoningStep, ReasoningTrajectory,
        reason,
        # audit
        NoLaunderingInvariant, assert_no_laundering, audit, inputs_of_A,
    )
