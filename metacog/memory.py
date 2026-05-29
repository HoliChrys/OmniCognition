"""
MetaCog-Mem — top-level Memory wrapper.

This is the public API surface. All the low-level primitives in
epistemic / geometry / collision / compression / detectors / execution
/ reasoning / observator are still available individually for advanced
callers ; Memory composes them into a single coherent service.

  m = Memory()                              # in-memory only
  m = Memory(storage_path="memory.pkl")     # persistent
  m.ingest("Dr Sarah lives in Berkeley")
  m.process_turn("user", "tell me about Dr Sarah")
  trajectory = m.reason("where does Dr Sarah live?")
  m.save()

The Memory takes care of timestamps, default observators, default
encoder/LLM/executor, persistence, and audit reporting.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from metacog.audit import audit, assert_no_laundering, inputs_of_A
from metacog.collision import sleep_cycle_collisions
from metacog.compression import compress_trajectory
from metacog.defaults import NoOpExecutor, SimpleEncoder, SimpleLLM
from metacog.detectors import ConversationLog, TurnRecord, analyze_user_turn
from metacog.epistemic import (
    DEFAULT_OBSERVATOR_ID,
    EpistemicState,
    Observation,
    Point,
    PointKind,
    SourceClass,
    apply_observation,
    process_observation,
)
from metacog.execution import execute_action
from metacog.geometry import (
    effective_embedding,
    retrieve,
    retrieve_for_observator,
    retrieve_hybrid,
    retrieve_with_lineage,
)
from metacog.keywords import KeywordExtractor, SimpleKeywordExtractor
from metacog.observator import (
    Observator,
    delegate_query,
    detect_polarization,
    select_observators,
    spawn_observators_by_clustering,
    spawn_observators_from_polarization,
)
from metacog.reasoning import ReasoningTrajectory, reason


@dataclass
class Memory:
    """High-level service object that orchestrates the whole pipeline."""

    encoder: Any = field(default_factory=SimpleEncoder)
    llm: Any = field(default_factory=SimpleLLM)
    executor: Any = field(default_factory=NoOpExecutor)
    extractor: Any = field(default_factory=SimpleKeywordExtractor)
    storage_path: Optional[str] = None

    points: List[Point] = field(default_factory=list)
    observators: Dict[str, Observator] = field(default_factory=dict)
    conversation_log: ConversationLog = field(default_factory=ConversationLog)
    _t_clock: float = 0.0

    def __post_init__(self) -> None:
        if self.storage_path:
            try:
                self.load()
            except FileNotFoundError:
                pass  # first run

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    def _now(self, t: Optional[float] = None) -> float:
        if t is not None:
            self._t_clock = max(self._t_clock, t)
            return t
        self._t_clock += 1.0
        return self._t_clock

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        content: str,
        *,
        kind: str = "FACT",
        id: Optional[str] = None,
        parents: Optional[List[str]] = None,
        sequence_prev: Optional[str] = None,
        sequence_next: Optional[str] = None,
    ) -> Point:
        """Create and store a new Point. Returns the Point.

        `sequence_prev` / `sequence_next` link the new point to its
        ordered neighbors (e.g. previous and next conversation turn,
        adjacent paragraphs). They power retrieve_with_lineage.
        """
        if id is None:
            id = f"{kind.lower()}_{uuid.uuid4().hex[:8]}"
        # Extract keywords + their embedding for hybrid retrieval.
        kws: List[str] = []
        kw_emb = None
        kw_src = None
        if self.extractor is not None:
            kws = self.extractor.extract(content, n=5)
            if kws:
                kw_emb = tuple(self.encoder.encode(" ".join(kws)))
                kw_src = getattr(self.extractor, "source", SourceClass.COMPUTATION)
        point = Point(
            id=id,
            content=content,
            embedding_orig=tuple(self.encoder.encode(content)),
            kind=PointKind[kind.upper()],
            parents=list(parents) if parents else [],
            lineage_depth=0 if not parents else 1,
            sequence_prev=sequence_prev,
            sequence_next=sequence_next,
            keywords=kws,
            keywords_embedding=kw_emb,
            keywords_source=kw_src,
        )
        self.points.append(point)
        # Backfill the previous point's sequence_next if we know it
        if sequence_prev:
            for p in self.points:
                if p.id == sequence_prev and p.sequence_next is None:
                    p.sequence_next = id
                    break
        # Backfill children references on the new point's parents
        if parents:
            parent_set = set(parents)
            for p in self.points:
                if p.id in parent_set and id not in p.children:
                    p.children = list(p.children) + [id]
        return point

    def ingest_action(
        self,
        description: str,
        *,
        id: Optional[str] = None,
        parents: Optional[List[str]] = None,
    ) -> Point:
        """Convenience : ingest an ACTION point."""
        return self.ingest(description, kind="ACTION", id=id, parents=parents)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(
        self,
        target_ids: List[str],
        polarity: float,
        *,
        source: str = "OBSERVER",
        signal_type: str = "external",
        observator_id: str = DEFAULT_OBSERVATOR_ID,
        raw_content: Optional[str] = None,
        t: Optional[float] = None,
    ) -> Observation:
        """Apply an Observation to one or more points.

        `source` is one of "OBSERVER" / "COMPUTATION" (GENERATOR is
        rejected at construction — Cor. 5).
        """
        t_now = self._now(t)
        obs = Observation(
            source=SourceClass[source.upper()],
            signal_type=signal_type,
            polarity=polarity,
            target_node_ids=list(target_ids),
            timestamp=t_now,
            raw_content=raw_content,
            observator_id=observator_id,
        )
        process_observation(self.points, obs, population=self.points)
        return obs

    # ------------------------------------------------------------------
    # Conversation turn
    # ------------------------------------------------------------------

    def process_turn(
        self,
        text: str,
        *,
        speaker: str = "user",
        retrieved_point_ids: Optional[List[str]] = None,
        t: Optional[float] = None,
    ) -> List[Observation]:
        """Record a conversation turn and (if user-speaker) run all
        detectors to produce Observations on already-retrieved points.
        """
        t_now = self._now(t)
        turn = TurnRecord(
            timestamp=t_now,
            speaker=speaker,
            text=text,
            embedding=tuple(self.encoder.encode(text)),
            retrieved_point_ids=list(retrieved_point_ids) if retrieved_point_ids else [],
        )
        self.conversation_log.record(turn)
        if speaker != "user":
            return []
        observations = analyze_user_turn(turn, self.conversation_log, self.points, encoder=self.encoder)
        for obs in observations:
            process_observation(self.points, obs, population=self.points)
        return observations

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        k: int = 7,
        observator_id: Optional[str] = None,
        use_lineage: bool = False,
        use_hybrid: bool = False,
        lineage_depth: int = 7,
        t: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k points.

        Modes :
          - default              cosine kNN over effective embedding
          - use_lineage=True     cosine + lineage RRF
          - use_hybrid=True      cosine on keywords + BM25 on content
                                 + RRF (+ optional lineage)
          - observator_id=…      route through that observator's view

        Default k=7 (≈ matches LoCoMo / typical agentic context budget).
        """
        t_now = self._now(t)
        if observator_id and observator_id != DEFAULT_OBSERVATOR_ID:
            q_emb = tuple(self.encoder.encode(query))
            results = retrieve_for_observator(
                q_emb, self.points, k, t_now, observator_id,
            )
        elif use_hybrid:
            results = retrieve_hybrid(
                query, self.points, k, t_now,
                encoder=self.encoder,
                extractor=self.extractor,
                use_lineage=use_lineage,
                lineage_depth=lineage_depth,
            )
        elif use_lineage:
            q_emb = tuple(self.encoder.encode(query))
            results = retrieve_with_lineage(
                q_emb, self.points, k, t_now,
                lineage_depth=lineage_depth,
            )
        else:
            q_emb = tuple(self.encoder.encode(query))
            results = retrieve(q_emb, self.points, k, t_now)
        return [
            {
                "id": p.id,
                "content": p.content,
                "kind": p.kind.value,
                "state": p.state.value,
                "score": score,
                "confidence": p.confidence,
                "n_corrob": p.n_corrob,
                "n_contra": p.n_contra,
            }
            for score, p in results
        ]

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def reason(
        self,
        query: str,
        *,
        with_executor: bool = True,
        apply_compression: bool = True,
        t: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a full reasoning trajectory and return a summary."""
        t_now = self._now(t)
        traj: ReasoningTrajectory = reason(
            query, self.points, self.encoder, self.llm,
            executor=self.executor if with_executor else None,
            t_now=t_now,
            apply_compression=apply_compression,
        )
        return {
            "query": traj.query,
            "final_answer": traj.final_answer,
            "converged": traj.converged,
            "halt_reason": traj.halt_reason,
            "iterations": len(traj.steps),
            "visited_point_ids": [p.id for p in traj.visited_points],
            "created_point_ids": [p.id for p in traj.created_points],
            "compressions_applied": traj.compressions_applied,
        }

    # ------------------------------------------------------------------
    # Sleep cycle
    # ------------------------------------------------------------------

    def sleep(self, t: Optional[float] = None) -> Dict[str, Any]:
        """Run a sleep cycle of collision resolution."""
        t_now = self._now(t)
        report = sleep_cycle_collisions(
            self.points, self.llm, self.encoder, t_now=t_now,
        )
        return {
            "iterations": report.iterations,
            "resolved_count": len(report.resolved),
            "new_children_ids": [p.id for p in report.new_children],
            "aborted_for_cascade_limit": report.aborted_for_cascade_limit,
        }

    # ------------------------------------------------------------------
    # Observators
    # ------------------------------------------------------------------

    def declare_observator(
        self,
        id: str,
        *,
        name: str = "",
        keywords: Optional[List[str]] = None,
    ) -> Observator:
        """Explicitly create an Observator."""
        obs = Observator(
            id=id,
            name=name,
            keywords=list(keywords) if keywords else [],
            created_at=self._now(),
        )
        obs.ensure_keywords_embedding(self.encoder)
        self.observators[id] = obs
        return obs

    def detect_polarized_points(self) -> List[str]:
        return [p.id for p in self.points if detect_polarization(p)]

    def spawn_observators(
        self,
        point_id: str,
        *,
        strategy: str = "clustering",
        t: Optional[float] = None,
    ) -> List[str]:
        """Spawn observators for a polarized point.
        `strategy` ∈ {"clustering", "polarity"}.
        Returns the ids of newly created observators.
        """
        t_now = self._now(t)
        p = next((q for q in self.points if q.id == point_id), None)
        if p is None:
            return []
        if strategy == "polarity":
            spawned = spawn_observators_from_polarization(p, t_now=t_now)
        else:
            spawned = spawn_observators_by_clustering(p, self.encoder, t_now=t_now)
        for o in spawned:
            self.observators[o.id] = o
        return [o.id for o in spawned]

    def route(self, query: str, k: int = 1) -> List[Dict[str, Any]]:
        """Pick the top-k observators most aligned with the query."""
        q_emb = tuple(self.encoder.encode(query))
        top = select_observators(q_emb, list(self.observators.values()), k=k, encoder=self.encoder)
        return [
            {"id": obs.id, "name": obs.name, "keywords": list(obs.keywords), "score": score}
            for score, obs in top
        ]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def inspect(self, point_id: str) -> Optional[Dict[str, Any]]:
        p = next((q for q in self.points if q.id == point_id), None)
        if p is None:
            return None
        return {
            "id": p.id,
            "content": p.content,
            "kind": p.kind.value,
            "state": p.state.value,
            "n_corrob": p.n_corrob,
            "n_contra": p.n_contra,
            "n_uses": p.n_uses,
            "n_revision": p.n_revision,
            "confidence": p.confidence,
            "uncertainty": p.uncertainty,
            "parents": list(p.parents),
            "children": list(p.children),
            "lineage_depth": p.lineage_depth,
            "observator_views": {
                oid: {
                    "n_corrob": v.n_corrob,
                    "n_contra": v.n_contra,
                    "state": v.state.value,
                    "confidence": v.confidence,
                }
                for oid, v in p.observator_views.items()
            },
            "update_log_size": len(p.update_log),
            "execution_log_size": len(p.execution_log),
        }

    def audit(self) -> Dict[str, Any]:
        report = audit(self.points)
        return {
            "total_updates": report.total_updates,
            "by_source": {k.value: v for k, v in report.by_source.items()},
            "violations": list(report.violations),
            "ok": report.ok,
        }

    def stats(self) -> Dict[str, Any]:
        kind_counts: Dict[str, int] = {}
        state_counts: Dict[str, int] = {}
        for p in self.points:
            kind_counts[p.kind.value] = kind_counts.get(p.kind.value, 0) + 1
            state_counts[p.state.value] = state_counts.get(p.state.value, 0) + 1
        return {
            "n_points": len(self.points),
            "by_kind": kind_counts,
            "by_state": state_counts,
            "n_observators": len(self.observators),
            "n_turns": len(self.conversation_log.turns),
            "clock": self._t_clock,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        import pickle
        target = path or self.storage_path
        if target is None:
            raise ValueError("no storage_path configured and none provided")
        snapshot = {
            "points": self.points,
            "observators": self.observators,
            "conversation_log": self.conversation_log,
            "_t_clock": self._t_clock,
        }
        with open(target, "wb") as f:
            pickle.dump(snapshot, f)

    def load(self, path: Optional[str] = None) -> None:
        import pickle
        source = path or self.storage_path
        if source is None:
            raise ValueError("no storage_path configured and none provided")
        with open(source, "rb") as f:
            snapshot = pickle.load(f)
        self.points = snapshot.get("points", [])
        self.observators = snapshot.get("observators", {})
        self.conversation_log = snapshot.get("conversation_log", ConversationLog())
        self._t_clock = snapshot.get("_t_clock", 0.0)
