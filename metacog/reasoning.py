"""
MetaCog-Mem — reasoning trajectory (the reason() orchestrator).

Greedy traversal of the manifold until output convenable :

  CRITÈRE D'ARRÊT (composite, parameter-free) :
    1. STABILITÉ            cos(answer_t, answer_{t-1}) > cos(π/12)
    2. NO MORE CANDIDATES   tous les points éligibles déjà visités
    3. QUALITY DROP         le prochain candidat scorerait négativement

  À CHAQUE PAS :
    - greedy kNN autour de l'ancre courante (query au début, puis
      embedding de la réponse en cours)
    - visite du point top-1 → n_uses += 1
    - synthèse via le LLM injecté → nouveau THOUGHT point créé
    - éventuellement, propose+exécute une ACTION (si executor fourni)

  À LA FIN (EAGER) :
    - compress_trajectory pour appliquer Chasles sur le chemin parcouru
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

from metacog.compression import compress_trajectory
from metacog.epistemic import (
    EpistemicState,
    Point,
    PointKind,
)
from metacog.execution import (
    ExecutorProtocol,
    execute_action,
)
from metacog.geometry import cosine, effective_embedding


# Angular stability convention — same family as other cosine thresholds
COS_STABLE = math.cos(math.pi / 12)  # ≈ 0.966 (15°)


class Encoder(Protocol):
    def encode(self, text: str) -> Tuple[float, ...]: ...


class ReasoningLLM(Protocol):
    """LLM responsible for synthesizing each step + optionally proposing
    ACTION descriptions.

    All outputs are GENERATOR — they become content of new THOUGHT or
    ACTION points, but never enter the observation set O.

    To also enable Chasles compression at the end of the trajectory,
    the same object should implement `extract_common` and
    `remove_overlap` (i.e. the LLMExtractor protocol).
    """

    def synthesize_step(
        self,
        query: str,
        previous_answer: Optional[str],
        new_point_content: str,
    ) -> str: ...

    def propose_action(
        self, query: str, current_answer: str
    ) -> Optional[str]: ...


@dataclass
class ReasoningStep:
    iteration: int
    visited_point_id: str
    visited_kind: PointKind
    new_thought_id: str
    tentative_answer: str
    answer_embedding: Tuple[float, ...]
    cosine_to_previous: Optional[float]
    score_to_anchor: float
    timestamp: float


@dataclass
class ReasoningTrajectory:
    query: str
    query_embedding: Tuple[float, ...]
    steps: List[ReasoningStep] = field(default_factory=list)
    visited_points: List[Point] = field(default_factory=list)
    created_points: List[Point] = field(default_factory=list)
    final_answer: str = ""
    final_answer_embedding: Tuple[float, ...] = ()
    converged: bool = False
    halt_reason: str = ""
    compressions_applied: int = 0


def reason(
    query: str,
    points: List[Point],
    encoder: Encoder,
    llm: ReasoningLLM,
    *,
    executor: Optional[ExecutorProtocol] = None,
    t_now: float = 0.0,
    apply_compression: bool = True,
) -> ReasoningTrajectory:
    """Run a greedy reasoning trajectory until output convenable.

    Mutates `points` in place :
      - newly created THOUGHTs are appended
      - if an executor is provided and the LLM proposes actions, new
        ACTION points are created and executed (their result FACTs
        are also appended)
      - visited points get n_uses += 1
      - at the end (if apply_compression), compress_trajectory may
        spawn compression-children on near-straight subchains
    """
    q_emb = tuple(encoder.encode(query))
    visited_ids: set[str] = set()
    visited_in_order: List[Point] = []
    created: List[Point] = []
    steps: List[ReasoningStep] = []

    answer: Optional[str] = None
    answer_emb: Optional[Tuple[float, ...]] = None
    halt_reason = "no_more_candidates"
    converged = False

    while True:
        anchor = answer_emb if answer_emb is not None else q_emb

        # Greedy : top-1 not yet visited, not latent
        candidates: List[Tuple[float, Point]] = []
        for p in points:
            if p.id in visited_ids:
                continue
            if p.state in {EpistemicState.INVALID, EpistemicState.DEPRECATED}:
                continue
            score = cosine(anchor, effective_embedding(p, t_now))
            candidates.append((score, p))
        if not candidates:
            halt_reason = "no_more_candidates"
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        score, next_point = candidates[0]

        # Quality drop : strictly negative cosine means the next-best
        # candidate points away from our state → no more useful info.
        if answer_emb is not None and score < 0.0:
            halt_reason = "quality_drop"
            break

        # Visit
        visited_ids.add(next_point.id)
        visited_in_order.append(next_point)
        next_point.n_uses += 1

        # Synthesize new answer (LLM = GENERATOR producing content)
        new_answer = llm.synthesize_step(query, answer, next_point.content)
        new_answer_emb = tuple(encoder.encode(new_answer))

        # Create the THOUGHT (lineage : parent = visited point)
        iteration = len(steps)
        thought_id = f"thought({iteration})@{t_now:.6f}"
        thought = Point(
            id=thought_id,
            content=new_answer,
            embedding_orig=new_answer_emb,
            kind=PointKind.THOUGHT,
            parents=[next_point.id],
            lineage_depth=next_point.lineage_depth + 1,
        )
        points.append(thought)
        created.append(thought)
        # Don't revisit our own thought
        visited_ids.add(thought_id)

        # Optionally propose + execute an ACTION
        if executor is not None:
            action_desc = llm.propose_action(query, new_answer)
            if action_desc:
                action_id = f"action({iteration})@{t_now:.6f}"
                action = Point(
                    id=action_id,
                    content=action_desc,
                    embedding_orig=tuple(encoder.encode(action_desc)),
                    kind=PointKind.ACTION,
                    parents=[thought_id],
                    lineage_depth=thought.lineage_depth + 1,
                )
                points.append(action)
                created.append(action)
                visited_ids.add(action_id)
                # Execute and capture result FACT
                _obs, result_fact = execute_action(
                    action, executor, points, t_now, encoder=encoder,
                )
                if result_fact is not None:
                    created.append(result_fact)
                    visited_ids.add(result_fact.id)

        # Stability check
        sim_to_prev: Optional[float] = None
        if answer_emb is not None:
            sim_to_prev = cosine(new_answer_emb, answer_emb)

        steps.append(ReasoningStep(
            iteration=iteration,
            visited_point_id=next_point.id,
            visited_kind=next_point.kind,
            new_thought_id=thought_id,
            tentative_answer=new_answer,
            answer_embedding=new_answer_emb,
            cosine_to_previous=sim_to_prev,
            score_to_anchor=score,
            timestamp=t_now,
        ))

        answer = new_answer
        answer_emb = new_answer_emb

        if sim_to_prev is not None and sim_to_prev > COS_STABLE:
            halt_reason = "stability"
            converged = True
            break

    trajectory = ReasoningTrajectory(
        query=query,
        query_embedding=q_emb,
        steps=steps,
        visited_points=visited_in_order,
        created_points=created,
        final_answer=answer or "",
        final_answer_embedding=answer_emb or q_emb,
        converged=converged,
        halt_reason=halt_reason,
    )

    # Eager compression on the visited path (Chasles)
    if apply_compression and len(visited_in_order) >= 4:
        try:
            report = compress_trajectory(
                visited_in_order, points, llm, encoder, t_now,
            )
            trajectory.compressions_applied = report.compressions_applied
            for child in report.new_children:
                created.append(child)
        except Exception:
            # If the LLM doesn't implement extract_common/remove_overlap
            # (compression deps), skip compression — the trajectory
            # still has value.
            pass

    return trajectory
