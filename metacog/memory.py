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
from metacog.collision import merge_duplicates, sleep_cycle_collisions
from metacog.compression import compress_trajectory
from metacog.defaults import NoOpExecutor, SimpleEncoder
from metacog.llm import ClaudeLLM
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
    apply_pull,
    effective_embedding,
    retrieve,
    retrieve_for_observator,
    retrieve_hybrid,
    retrieve_with_lineage,
)
from metacog.keywords import (
    KeywordExtractor,
    SimpleKeywordExtractor,
    position_weighted_keyword_embedding,
)
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
    llm: Any = field(default_factory=ClaudeLLM)
    executor: Any = field(default_factory=NoOpExecutor)
    extractor: Any = field(default_factory=SimpleKeywordExtractor)
    # Opt-in LLM entity extractor. None = current behavior unchanged.
    # When set, each ingested FACT spawns tagged entity beacon nodes that
    # are geometrically pulled onto it (the edge-free "edges-equivalent").
    entity_extractor: Any = None
    # Opt-in mem0-style atomic-fact extractor. When set, each ingested FACT
    # spawns clean self-contained atomic FACTs (parents=[source dia_id],
    # GENERATOR) that act as retrieval handles ; retrieval resolves an
    # atomic hit back to its source turn (dia_id) and dedups.
    atomic_extractor: Any = None
    storage_path: Optional[str] = None

    points: List[Point] = field(default_factory=list)
    observators: Dict[str, Observator] = field(default_factory=dict)
    conversation_log: ConversationLog = field(default_factory=ConversationLog)
    # atomic-fact id -> source turn id (for retrieval resolution).
    _atom_parent: Dict[str, str] = field(default_factory=dict)
    # absorbed point id -> surviving node id, from consolidate_duplicates().
    _merge_aliases: Dict[str, str] = field(default_factory=dict)
    # Total number of multi-hop transitions recorded by record_hop ;
    # drives the Poisson baseline used by spike_threshold(). Persisted.
    _spike_total_hops: int = 0
    _t_clock: float = 0.0
    # Diversity-weighted co-retrieval ledger driving LATERAL collision
    # (metacog.lateral). Accumulated via record_retrieval() ; consumed by
    # lateral_collapse(). Opt-in : recording is a no-op unless enabled.
    _lateral_ledger: Any = None
    lateral_enabled: bool = False
    # Action-recurrence ledger driving SKILL/TOOL crystallization
    # (metacog.skills). Accumulated via record_action_generation() ;
    # consumed by crystallize_skills(). Crystallized tools are NORMAL nodes
    # (kind=ACTION, tagged "tool") in self.points, so they persist with the
    # cloud and the walk finds them recursively. Opt-in.
    _skill_ledger: Any = None
    skills_enabled: bool = False
    # Resolution ledger driving the LATENT SKILL DISTILLER (run in sleep()).
    # Each entry records a solved task — (query, the walk's resolution path
    # point-ids, the output) — so the distiller can replay it afterwards and
    # crystallize a theoretical tool linked to the semantic facts/thoughts/
    # actions that explicate it, so the next time the task recurs it is
    # retrieved fast WITHOUT forced metacognition. `_distill_cursor` marks
    # how far the distiller has consumed the ledger.
    _resolution_ledger: List[dict] = field(default_factory=list)
    _distill_cursor: int = 0

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
                kw_emb = position_weighted_keyword_embedding(kws, self.encoder)
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
        # Spawn entity beacon nodes (opt-in). Only from genuine FACTs —
        # never recurse on entity-derived facts (their ids start "entity_").
        if (
            self.entity_extractor is not None
            and kind.upper() == "FACT"
            and not id.startswith("entity_")
        ):
            try:
                self._spawn_entities(point)
            except Exception:
                # A flaky extractor must never break ingestion.
                pass
        if (
            self.atomic_extractor is not None
            and kind.upper() == "FACT"
            and not id.startswith(("entity_", "atom_"))
        ):
            try:
                self._spawn_atomics(point)
            except Exception:
                pass
        return point

    def _spawn_atomics(self, source_fact: Point) -> None:
        """Decompose a turn into clean atomic FACTs (mem0-style). Each is a
        retrieval handle : GENERATOR-sourced, parents=[source id], pulled
        onto the source turn ; retrieval resolves it back to the dia_id."""
        # speaker / text from the "[date] Speaker: text" content
        body = source_fact.content
        spk, txt = "", body
        if "]" in body:
            body = body.split("]", 1)[1]
        if ":" in body[:40]:
            spk, txt = body.split(":", 1)[0].strip(), body.split(":", 1)[1].strip()
        atoms = self.atomic_extractor.extract_atoms(txt, speaker=spk)
        t_now = self._now()
        for k, a in enumerate(atoms):
            kws = self.extractor.extract(a, n=6) if self.extractor else []
            kw_emb = (position_weighted_keyword_embedding(kws, self.encoder)
                      if kws else None)
            atom = Point(
                id=f"atom_{source_fact.id}_{k}",
                content=a,
                embedding_orig=tuple(self.encoder.encode(a)),
                kind=PointKind.FACT,
                keywords=kws,
                keywords_embedding=kw_emb,
                keywords_source=SourceClass.GENERATOR,
                parents=[source_fact.id],
                tags=["atomic"],
            )
            self.points.append(atom)
            self._atom_parent[atom.id] = source_fact.id
            apply_pull(atom, source_fact, +1.0, t_now)

    # ------------------------------------------------------------------
    # Entity beacons (edge-free "edges" via geometric pull)
    # ------------------------------------------------------------------

    def ingest_entity(
        self,
        value: str,
        *,
        source_fact: Point,
        tags: Optional[List[str]] = None,
        parent_entity: Optional[Point] = None,
        t_now: Optional[float] = None,
    ) -> Point:
        """Create a tagged entity beacon node and relate it to its source
        fact GEOMETRICALLY — there are no stored edges.

        The beacon is a PointKind.FACT carrying a clean entity value +
        type tags so a query matches it easily ; `apply_pull` then drags
        BOTH the beacon and `source_fact` together in the manifold. The
        fact's effective (content) embedding shifts toward the entity
        value, so a query for that value retrieves the real fact directly
        — this is the edges-equivalent. Date components are also pulled
        onto their `parent_entity` (the shared `date` beacon).

        Per-fact (NOT deduplicated across facts) : the same entity in two
        facts spawns two beacons, each co-located with its own fact, so a
        query surfaces ALL facts that mention it (parallel paths).

        Cor. 5 : the beacon is GENERATOR-sourced and never produces an
        Observation — apply_pull is called directly.
        """
        tags = tags or []
        if t_now is None:
            t_now = self._now()
        kws = [value] + [t for t in tags if t and t != value]
        kw_emb = position_weighted_keyword_embedding(kws, self.encoder)
        beacon = Point(
            id=f"entity_{uuid.uuid4().hex[:8]}",
            content=(":".join(tags) + " " + value).strip() if tags else value,
            embedding_orig=tuple(self.encoder.encode(value)),
            kind=PointKind.FACT,
            keywords=kws,
            keywords_embedding=kw_emb,
            keywords_source=SourceClass.GENERATOR,
            tags=list(tags) + ["entity"],
        )
        self.points.append(beacon)
        # Geometric "edges" : pull the beacon onto its source fact, and
        # date components onto their parent date. Shared t_now so intra-fact
        # decay never wipes the accumulating pulls.
        apply_pull(beacon, source_fact, +1.0, t_now)
        if parent_entity is not None:
            apply_pull(beacon, parent_entity, +1.0, t_now)
        return beacon

    def _spawn_entities(self, source_fact: Point) -> None:
        """Extract entities from a freshly ingested FACT and spawn their
        beacon nodes. A date yields a full-date beacon plus day/month/year
        component beacons — all tagged "date" and all pulled onto the SAME
        source fact, so they cluster together near it (their geometric
        "part_of" link is this shared co-location, not a separate pull,
        which would only fight the source pull). One frozen t_now per fact
        so pulls accumulate without inter-pull decay."""
        ents = self.entity_extractor.extract_entities(source_fact.content)
        if not ents:
            return
        t_now = self._now()
        for e in ents:
            if e.etype == "date":
                self.ingest_entity(
                    e.value, source_fact=source_fact,
                    tags=["date"], t_now=t_now,
                )
                for part in ("day", "month", "year"):
                    val = (e.date_parts or {}).get(part)
                    if not val:
                        continue
                    # Bare value ("20"/"january"/"2023") is the matchable
                    # keyword ; the part lives in tags.
                    self.ingest_entity(
                        val, source_fact=source_fact,
                        tags=["date", part], t_now=t_now,
                    )
            else:
                self.ingest_entity(
                    e.value, source_fact=source_fact,
                    tags=[e.etype], t_now=t_now,
                )

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
    # Skills : a named "theoretical directory" of tools, ingested from a
    # nested JSON tree, re-indexed as normal tagged points.
    # ------------------------------------------------------------------

    def ingest_skill(
        self,
        tree: Dict[str, Any],
        *,
        name: str,
        user_id: str,
        session_id: str,
        date: Optional[str] = None,
        t_now: Optional[float] = None,
    ) -> List[Point]:
        """Ingest a NAMED skill — a theoretical directory tree of tools —
        as ordinary tagged points, returning every created point (root
        first).

        `tree` is a nested dict : a directory maps a name to a sub-dict, a
        file maps a name to its description string :
            {"retrieval": {"hybrid.py": "RRF over 4 signals",
                           "bm25.py":   "content-first BM25"},
             "walk.py":    "uncertainty-governed depth"}

        Each node becomes a `PointKind.FACT` carrying the usual format, so
        the walk retrieves it like any other point (gold = semantic fact OR
        tool fact). The directory **topology** is encoded two ways that
        agree :
          - the edge-free lineage link (`parents = [parent_point_id]`), so
            lineage spread traverses the tree ;
          - typed `ref:skill:*` keyword tokens — `ref:skill:<name>:path:<p>`
            and `ref:skill:<name>:parent:<pp>` — so the topology is itself
            retrievable / co-locating (the same trick as `ref:date:*`).

        Every point is tagged so a later double-query can pre-filter the
        SECTION (this user's this-session this-skill) before the free
        semantic query :
            skill · tool · name:<name> · user:<id> · session:<id>
            · date:<indexation>
        plus `skill_dir` / `skill_file` for the node kind.

        Cor. 5 : skill content is agent/LLM-produced → keywords_source =
        GENERATOR ; ids are prefixed `skill_` so they never collide with a
        real evidence id.
        """
        if t_now is None:
            t_now = self._now()
        if date is None:
            from datetime import date as _date
            date = _date.today().isoformat()
        base_tags = [
            "skill", "tool", f"name:{name}",
            f"user:{user_id}", f"session:{session_id}", f"date:{date}",
        ]
        created: List[Point] = []

        def _tok(s: str) -> List[str]:
            import re as _re
            return [w.lower() for w in _re.findall(r"[A-Za-z0-9]{2,}", s or "")]

        def _node(node_name: str, value: Any, path: str,
                  parent_path: str, parent_id: Optional[str]) -> Point:
            is_dir = isinstance(value, dict)
            desc = "" if is_dir else str(value)
            content = (
                f"skill:{name} {path}" + (f" — {desc}" if desc else "")
            )
            kws: List[str] = []
            seen: set = set()
            for k in (_tok(name) + _tok(node_name)
                      + [f"ref:skill:{name}:path:{path}"]
                      + ([f"ref:skill:{name}:parent:{parent_path}"]
                         if parent_path else [])
                      + _tok(desc)[:8]):
                if k and k not in seen:
                    seen.add(k)
                    kws.append(k)
            kw_emb = position_weighted_keyword_embedding(kws, self.encoder)
            p = Point(
                id=f"skill_{uuid.uuid4().hex[:8]}",
                content=content,
                embedding_orig=tuple(self.encoder.encode(content)),
                kind=PointKind.FACT,
                parents=[parent_id] if parent_id else [],
                lineage_depth=0 if parent_id is None else 1,
                keywords=kws,
                keywords_embedding=kw_emb,
                keywords_source=SourceClass.GENERATOR,
                tags=list(base_tags)
                + ["skill_dir" if is_dir else "skill_file"],
            )
            self.points.append(p)
            created.append(p)
            # Geometric topology : pull each child onto its parent so the
            # tree clusters in the manifold (edges-equivalent), mirroring
            # the lineage link above.
            if parent_id is not None:
                parent_pt = next((q for q in self.points
                                  if q.id == parent_id), None)
                if parent_pt is not None:
                    apply_pull(p, parent_pt, +1.0, t_now)
            return p

        # Root = the skill itself ; children hang under it.
        root = _node(name, tree, name, "", None)

        def _walk(subtree: Dict[str, Any], parent_path: str,
                  parent_id: str) -> None:
            for child_name, value in subtree.items():
                path = f"{parent_path}/{child_name}" if parent_path else child_name
                pt = _node(child_name, value, path, parent_path, parent_id)
                if isinstance(value, dict):
                    _walk(value, path, pt.id)

        _walk(tree, name, root.id)
        return created

    def build_skill(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str,
        n_stages: int = 8,
    ) -> Dict[str, Any]:
        """End-to-end skill construction (task mode). Runs a TASK-mode walk
        — gold is a tool fact as readily as a semantic fact, and the walk's
        depth gate is "have we gathered enough tools to elaborate the
        complete workflow?" — then synthesises a NAMED skill folder from the
        collected tools and re-indexes it via `ingest_skill`.

        The query's section (this user / session) is given a retrieval rank
        boost through the double-query `section_filter`. Returns the skill
        JSON `{"name", "tree"}` (also persisted as points). Records the
        resolution for the latent distiller (§ sleep)."""
        from metacog.meta_walk import (
            MetaWalker, synthesize_skill_json,
        )
        section = {f"user:{user_id}", f"session:{session_id}"}
        walker = MetaWalker(
            query, self, n_stages=n_stages, commit=False,
            task_mode=True, section_filter=section,
        )
        for _ in range(n_stages):
            out = walker.step()
            if out.done:
                break
        tools = list(getattr(walker, "_tools_collected", []))
        skill = synthesize_skill_json(query, tools, self.llm, name_hint=query)
        self.ingest_skill(
            skill.get("tree", {}), name=skill.get("name", "skill"),
            user_id=user_id, session_id=session_id,
        )
        self.record_resolution(
            query, [p.id for p in tools], skill.get("name", "skill"),
            user_id=user_id, session_id=session_id,
        )
        return skill

    # ------------------------------------------------------------------
    # Latent skill distiller (replayed in sleep())
    # ------------------------------------------------------------------

    def record_resolution(
        self,
        query: str,
        path_ids: Sequence[str],
        output: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Record a solved task for the latent distiller. Cheap, always on
        (the distiller itself is gated by `skills_enabled` in sleep)."""
        self._resolution_ledger.append({
            "query": query,
            "path_ids": list(path_ids),
            "output": output,
            "user_id": user_id,
            "session_id": session_id,
            "t": self._now(),
        })

    def distill_skills(self, t: Optional[float] = None) -> Dict[str, Any]:
        """LATENT distiller : replay the resolutions recorded since the last
        pass and crystallize, for each, a theoretical TOOL node linked to
        the semantic facts/thoughts/actions on its resolution path. The tool
        is a normal `ACTION` point tagged `tool`+`skill`+`distilled`, with
        `parents` set to its path points — so the next time the same task
        recurs the walk retrieves the tool DIRECTLY (and chains through it),
        no forced metacognition. apply_pull co-locates it with its path.

        Idempotent via `_distill_cursor`; deterministic; no LLM required."""
        t_now = self._now(t)
        by_id = {p.id: p for p in self.points}
        made: List[str] = []
        ledger = self._resolution_ledger
        while self._distill_cursor < len(ledger):
            r = ledger[self._distill_cursor]
            self._distill_cursor += 1
            path = [by_id[pid] for pid in r["path_ids"] if pid in by_id]
            if not path:
                continue
            tags = ["tool", "skill", "distilled"]
            if r.get("user_id"):
                tags.append(f"user:{r['user_id']}")
            if r.get("session_id"):
                tags.append(f"session:{r['session_id']}")
            tags.append(f"name:{r['output']}")
            # keywords : query terms + the path points' keywords (the
            # semantic facts/actions/thoughts that explicate the tool).
            import re as _re
            kws: List[str] = []
            seen: set = set()
            for w in _re.findall(r"[A-Za-z0-9]{3,}", r["query"]):
                wl = w.lower()
                if wl not in seen:
                    seen.add(wl); kws.append(wl)
            for p in path:
                for k in (p.keywords or [])[:3]:
                    if k and k not in seen:
                        seen.add(k); kws.append(k)
            kw_emb = position_weighted_keyword_embedding(kws[:12], self.encoder)
            content = f"tool[{r['output']}] for: {r['query']}"
            tool = Point(
                id=f"skill_{uuid.uuid4().hex[:8]}",
                content=content,
                embedding_orig=tuple(self.encoder.encode(content)),
                kind=PointKind.ACTION,
                parents=[p.id for p in path],   # linked to the explicating facts
                lineage_depth=1,
                keywords=kws[:12],
                keywords_embedding=kw_emb,
                keywords_source=SourceClass.GENERATOR,
                tags=tags,
            )
            self.points.append(tool)
            made.append(tool.id)
            for p in path:                      # co-locate with its path
                apply_pull(tool, p, +1.0, t_now)
        return {"distilled": len(made), "tool_ids": made}

    # ------------------------------------------------------------------
    # Tools : tool-tagged ACTION points + sandboxed execution.
    # ------------------------------------------------------------------

    def ingest_tool(
        self,
        content: str,
        code: str,
        *,
        lang: str = "python",
        id: Optional[str] = None,
    ) -> Point:
        """Create an executable tool — an ACTION point with the "tool"
        tag and a populated exec_spec. Discoverable via find_tools()
        (semantic match) or directly by Memory.execute_tool(id, args).

        Convention : `code` must define `def run(args: dict) -> JSON`.
        """
        p = self.ingest(content, kind="ACTION", id=id)
        p.exec_spec = {"lang": lang, "code": code}
        p.add_tag("tool")
        return p

    def find_tools(self, query: str, k: int = 5) -> List[Point]:
        """Top-k tool-tagged ACTION points most semantically aligned
        with `query`. Pure discovery — does NOT execute anything."""
        results = self.retrieve(query, k=max(k * 2, k), use_hybrid=True)
        by_id = {p.id: p for p in self.points}
        out: List[Point] = []
        for r in results:
            p = by_id.get(r["id"])
            if p is not None and p.has_tag("tool") and p not in out:
                out.append(p)
                if len(out) >= k:
                    break
        return out

    def execute_tool(
        self,
        tool_id: str,
        args: Dict[str, Any],
        executor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run a tool-tagged ACTION through the sandboxed executor.

        Returns {ok, result, fact_id} on success — a FACT child is
        created in memory (parents=[tool_id], tag "executed") whose
        content is the str(result). On failure : {ok=False, error,
        fact_id=None} and NO fact is created.
        """
        from metacog.executor import PyExecutor
        tool = next((p for p in self.points if p.id == tool_id), None)
        if tool is None:
            raise ValueError(f"unknown tool id {tool_id!r}")
        if not tool.has_tag("tool"):
            raise ValueError(f"point {tool_id!r} is not tagged 'tool'")
        if not tool.exec_spec:
            raise ValueError(f"point {tool_id!r} has no exec_spec")
        exe = executor if executor is not None else PyExecutor()
        out = exe.execute(tool.exec_spec, args)
        if not out.get("ok"):
            return {"ok": False, "error": out.get("error", "unknown"),
                    "fact_id": None}
        result_fact = self.ingest(
            content=str(out["result"]), kind="FACT", parents=[tool_id],
        )
        result_fact.add_tag("executed")
        return {"ok": True, "result": out["result"], "fact_id": result_fact.id}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _refresh_keywords(self, point: Point, additional_context: str = "") -> None:
        """Re-extract keywords for a revisited point.

        Called from observe / process_turn / process_observation
        whenever a point is touched by an observation. Lets the
        keywords drift as the system learns new context around the
        point (Hebbian-style entity refresh).

        If `additional_context` is provided, keywords are extracted
        from `content + " " + additional_context` so the local context
        (e.g. the user turn that triggered the observation) can
        influence the entity tags.
        """
        if self.extractor is None:
            return
        source_text = point.content
        if additional_context:
            source_text = f"{point.content} {additional_context}"
        new_kws = self.extractor.extract(source_text, n=5)
        if not new_kws:
            return
        if new_kws == list(point.keywords):
            return  # no change, skip embedding recompute
        point.keywords = new_kws
        point.keywords_embedding = position_weighted_keyword_embedding(
            new_kws, self.encoder,
        )
        # source class follows the extractor
        point.keywords_source = getattr(
            self.extractor, "source", SourceClass.COMPUTATION,
        )

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
        # Refresh keywords for every touched point (entity drift via context)
        for tid in target_ids:
            p = next((q for q in self.points if q.id == tid), None)
            if p is not None:
                self._refresh_keywords(p, additional_context=raw_content or "")
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
            # Refresh keywords for points touched by detected observations
            for tid in obs.target_node_ids:
                p = next((q for q in self.points if q.id == tid), None)
                if p is not None:
                    self._refresh_keywords(p, additional_context=text)
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
        use_spreading: bool = False,
        lineage_depth: int = 7,
        prefer_kind: Optional[str] = None,
        t: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k points.

        Modes :
          - default              cosine kNN over effective embedding
          - use_lineage=True     cosine + lineage RRF
          - use_hybrid=True      cosine on keywords + BM25 on content
                                 + RRF (+ optional lineage)
          - observator_id=…      route through that observator's view
          - prefer_kind=…        boost a PointKind in hybrid RRF (étape 5)
                                 accepts "FACT" / "THOUGHT" / "ACTION" or
                                 a PointKind ; ignored unless use_hybrid.

        Default k=7 (≈ matches LoCoMo / typical agentic context budget).
        """
        t_now = self._now(t)
        # When entity beacons exist they act as ingest-time pull agents :
        # their geometric pull already shifted the real facts, so we drop
        # them from the RETURNED ids (over-fetching to backfill to k) — the
        # recall metric then measures real facts only, and beacons never
        # displace evidence. No-op / unchanged when no extractor is set.
        beacons = self.entity_extractor is not None
        atomics = self.atomic_extractor is not None or bool(self._atom_parent)
        # Entity beacons do their work AT INGESTION : apply_pull (first step
        # = 1/(1+0) = 1.0) already shifted the real source facts toward each
        # entity/topic value, so a matching query finds the shifted REAL fact
        # directly. At query time the beacons are dead weight that would only
        # bloat the search 10x (spreading activation is superlinear), so we
        # exclude them from the SEARCH POOL entirely — the pull benefit
        # persists in the real facts' positions. (Atomics keep the old joint-
        # pool path, which already has its own entity_/atom_ handling below.)
        search_pts = self.points
        if beacons and not atomics:
            search_pts = [p for p in self.points
                          if not p.id.startswith("entity_")]
        # Over-fetch more when atomic facts are present : many atoms resolve
        # to the same source turn, so we need headroom to dedup to k turns.
        k_fetch = k
        if beacons:
            k_fetch = k * 2 + 10
        if atomics:
            k_fetch = max(k_fetch, k * 5 + 20)
        if observator_id and observator_id != DEFAULT_OBSERVATOR_ID:
            q_emb = tuple(self.encoder.encode(query))
            results = retrieve_for_observator(
                q_emb, search_pts, k_fetch, t_now, observator_id,
            )
        elif use_hybrid:
            kind_filter: Optional[PointKind] = None
            if prefer_kind is not None:
                kind_filter = (
                    prefer_kind if isinstance(prefer_kind, PointKind)
                    else PointKind[prefer_kind.upper()]
                )
            results = retrieve_hybrid(
                query, search_pts, k_fetch, t_now,
                encoder=self.encoder,
                extractor=self.extractor,
                use_lineage=use_lineage,
                use_spreading=use_spreading,
                lineage_depth=lineage_depth,
                prefer_kind=kind_filter,
            )
            if atomics:
                # Second pass over RAW turns only — atoms flood the joint
                # pool and bury strong raw evidence below the over-fetch
                # cutoff, so we need the TRUE raw ranking to interleave with.
                raw_pts = [p for p in self.points
                           if not p.id.startswith(("atom_", "entity_"))]
                self._raw_results = retrieve_hybrid(
                    query, raw_pts, k, t_now,
                    encoder=self.encoder, extractor=self.extractor,
                    use_lineage=use_lineage, use_spreading=use_spreading,
                    lineage_depth=lineage_depth, prefer_kind=kind_filter,
                )
        elif use_lineage:
            q_emb = tuple(self.encoder.encode(query))
            results = retrieve_with_lineage(
                q_emb, search_pts, k_fetch, t_now,
                lineage_depth=lineage_depth,
            )
        else:
            q_emb = tuple(self.encoder.encode(query))
            results = retrieve(q_emb, search_pts, k_fetch, t_now)
        if beacons or atomics:
            by_id = {p.id: p for p in self.points}
            # Two streams, both resolved to source turns : raw-turn hits and
            # atom-derived hits. Atoms make many turns competitive, so pure
            # score order lets atoms displace strong raw evidence (multi-hop
            # recall drops). INTERLEAVING the streams keeps recall >= the
            # better of {raw-only, atom-only} per query : the raw stream
            # preserves the baseline (e.g. enumeration turns), the atom
            # stream adds the entity-lookup turns the raw extractor missed.
            # Atom stream : atom-derived turns from the joint pool.
            atom_stream = [
                (s, self._atom_parent.get(p.id, p.id))
                for s, p in results if p.id.startswith("atom_")
            ]
            # Raw stream : prefer the dedicated raw-only ranking (true
            # baseline, computed above) ; fall back to the joint pool's
            # non-atom hits (beacons-only case).
            raw_src = getattr(self, "_raw_results", None)
            if raw_src is not None:
                raw_stream = [(s, p.id) for s, p in raw_src]
                self._raw_results = None
            else:
                raw_stream = [(s, p.id) for s, p in results
                              if not p.id.startswith(("atom_", "entity_"))]
            deduped, seen = [], set()
            ri = ai = 0
            while len(deduped) < k and (ri < len(raw_stream) or ai < len(atom_stream)):
                for stream, idx_name in ((raw_stream, "r"), (atom_stream, "a")):
                    i = ri if idx_name == "r" else ai
                    while i < len(stream) and stream[i][1] in seen:
                        i += 1
                    if i < len(stream):
                        s, rid = stream[i]
                        seen.add(rid)
                        deduped.append((s, by_id.get(rid)))
                        i += 1
                    if idx_name == "r":
                        ri = i
                    else:
                        ai = i
                    if len(deduped) >= k:
                        break
            results = [(s, p) for s, p in deduped if p is not None]
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
        """Run a sleep cycle of collision resolution.

        After geometric (proximity-fission) collisions, also run one
        LATERAL collision pass : functionally-redundant nodes that the
        co-retrieval ledger has shown to always surface together under
        diverse queries collapse into a single keeper. No-op unless
        lateral collision is enabled and the cloud is past the gate."""
        t_now = self._now(t)
        report = sleep_cycle_collisions(
            self.points, self.llm, self.encoder, t_now=t_now,
        )
        out = {
            "iterations": report.iterations,
            "resolved_count": len(report.resolved),
            "new_children_ids": [p.id for p in report.new_children],
            "aborted_for_cascade_limit": report.aborted_for_cascade_limit,
        }
        lat = self.lateral_collapse(t_now)
        out["lateral_collided_groups"] = lat["collided_groups"]
        out["lateral_aliases"] = lat["aliases"]
        # LATENT skill distiller : replay resolutions recorded since the
        # last sleep and crystallize theoretical tools linked to their
        # explicating facts. Opt-in (skills_enabled), idempotent.
        if self.skills_enabled and self._resolution_ledger:
            out["distilled"] = self.distill_skills(t_now)["distilled"]
        return out

    def compress_chasles(self) -> List[Dict[str, Any]]:
        """Auto-detect spike-driven Chasles paths and compress them.

        Each path of >= 4 same-kind spiking nodes (start, ≥2 intermediates,
        end) triggers resolve_collision on the intermediates with start /
        end as anchors. Reset n_spike to 0 on every node of a fired path
        (refractory period). Returns the list of CollisionEvent dicts."""
        from metacog.spike import auto_compress_chasles
        events = auto_compress_chasles(self, self.llm, self.encoder)
        return [
            {
                "child_id": ev.child_id,
                "parent_ids": list(ev.parent_ids),
                "anchor_ids": list(ev.anchor_ids),
                "timestamp": ev.timestamp,
            }
            for ev in events
        ]

    def consolidate_duplicates(self, t: Optional[float] = None) -> Dict[str, Any]:
        """True-merge deduplication : same-kind points carrying the SAME
        information collapse into a single node (the duplicate is dropped,
        its corroboration absorbed). Records id aliases so a dropped id can
        still be resolved to its survivor (e.g. evidence-id scoring)."""
        t_now = self._now(t)
        report = merge_duplicates(self.points, t_now)
        # Chain aliases through the existing map so older absorbed ids still
        # resolve to the final survivor.
        for absorbed, keeper in report.aliases.items():
            self._merge_aliases[absorbed] = self._merge_aliases.get(keeper, keeper)
        return {
            "merged_count": len(report.merged),
            "aliases": dict(report.aliases),
            "n_points": len(self.points),
        }

    def record_retrieval(
        self, ranked_ids: Sequence[str], query_emb: Optional[Sequence[float]] = None,
    ) -> None:
        """Fold one retrieval's ranked result ids into the lateral
        co-retrieval ledger. No-op unless `lateral_enabled`. Cheap
        (O(k·window)) so it can sit on the retrieval hot path."""
        if not self.lateral_enabled:
            return
        from metacog.lateral import LateralLedger, record_coretrieval
        if self._lateral_ledger is None:
            self._lateral_ledger = LateralLedger()
        record_coretrieval(
            self._lateral_ledger, list(ranked_ids),
            tuple(query_emb) if query_emb is not None else None,
        )

    def lateral_collapse(self, t: Optional[float] = None) -> Dict[str, Any]:
        """LATERAL collision : nodes that the co-retrieval ledger shows to
        be functionally redundant (always surfaced together by DIVERSE
        queries) collapse into a single keeper. Gated on a large, tag-rich
        cloud. Records absorbed->keeper aliases (chained through the merge
        map) so dropped ids still resolve. No-op when disabled or below
        the gate."""
        t_now = self._now(t)
        if self._lateral_ledger is None:
            return {"collided_groups": 0, "aliases": {}, "n_points": len(self.points)}
        from metacog.lateral import lateral_collapse as _collapse
        report = _collapse(
            self.points, self._lateral_ledger, self.encoder, t_now,
        )
        for absorbed, keeper in report.aliases.items():
            self._merge_aliases[absorbed] = self._merge_aliases.get(keeper, keeper)
        return {
            "collided_groups": len(report.collided),
            "aliases": dict(report.aliases),
            "n_points": len(self.points),
        }

    def record_action_generation(
        self, action: Point, query_emb: Optional[Sequence[float]],
        query_text: str, facts: Sequence[Point],
    ) -> None:
        """Register one re-derived ACTION into the skill-recurrence ledger,
        scoped by the query that triggered it and the grounding facts'
        keywords. No-op unless `skills_enabled`."""
        if not self.skills_enabled or action is None:
            return
        from metacog.skills import SkillLedger, record_action
        if self._skill_ledger is None:
            self._skill_ledger = SkillLedger()
        fact_kw: List[str] = []
        for f in facts[:5]:
            fact_kw.extend(f.keywords or [])
        record_action(
            self._skill_ledger, action,
            tuple(query_emb) if query_emb is not None else None,
            query_text, fact_kw,
        )

    def crystallize_skills(self, t: Optional[float] = None) -> Dict[str, Any]:
        """Crystallize recurring actions into persistent TOOL nodes — normal
        kind=ACTION Points tagged "tool", scoped by keywords, added to the
        cloud (so the walk finds them recursively and they persist on save).
        Gated on the emergent recurrence threshold. No-op when disabled."""
        t_now = self._now(t)
        if self._skill_ledger is None:
            return {"crystallized": 0, "tool_ids": [], "n_points": len(self.points)}
        from metacog.skills import detect_skill_candidates, synthesize_tool
        existing = list(self.points)
        cands = detect_skill_candidates(self._skill_ledger, existing_tools=existing)
        new_ids: List[str] = []
        for sig, trace in cands:
            tool = synthesize_tool(
                sig, trace, self.llm, self.encoder, t_now,
            )
            if tool is not None:
                self.points.append(tool)
                new_ids.append(tool.id)
        return {
            "crystallized": len(new_ids),
            "tool_ids": new_ids,
            "n_points": len(self.points),
        }

    def match_tool(self, query: str) -> Optional[Dict[str, Any]]:
        """Capability-cache lookup : does a previously-generated TOOL node
        already cover `query` ? Returns the tool summary + match score, or
        None (the agent must think from scratch). The walk also surfaces
        these tool nodes natively, so this is the explicit fast-path."""
        from metacog.skills import match_tool as _match
        q_kw = self.extractor.extract(query, n=8) if self.extractor else []
        if q_kw:
            q_emb = position_weighted_keyword_embedding(q_kw, self.encoder)
        else:
            q_emb = tuple(self.encoder.encode(query))
        hit = _match(q_emb, self.points)
        if hit is None:
            return None
        tool, score = hit
        return {
            "id": tool.id, "content": tool.content,
            "keywords": list(tool.keywords or []), "tags": list(tool.tags or []),
            "score": round(score, 4),
        }

    def ensure_tool(
        self, query: str, *, how: Optional[str] = None, genre: str = "command",
    ) -> Dict[str, Any]:
        """The 'no tool → generate it → proceed' step. Looks up a covering
        tool for `query` ; if one exists it is REUSED (no think phase) ;
        otherwise a tool is GENERATED now from the query + the deduced
        approach (`how`) and added to the cloud. Generation is
        unconstrained — this never blocks the agent, it only ever grows the
        self-built tool set. Returns {tool, reused}."""
        existing = self.match_tool(query)
        if existing is not None:
            tid = existing["id"]
            for p in self.points:
                if p.id == tid:
                    p.n_uses += 1
                    break
            return {"tool": existing, "reused": True}
        from metacog.skills import synthesize_tool_from_intent
        t_now = self._now()
        tool = synthesize_tool_from_intent(
            query, how or query, self.llm, self.encoder, t_now,
            genre=genre, extractor=self.extractor,
        )
        if tool is None:
            return {"tool": None, "reused": False}
        self.points.append(tool)
        return {
            "tool": {
                "id": tool.id, "content": tool.content,
                "keywords": list(tool.keywords or []), "tags": list(tool.tags or []),
            },
            "reused": False,
        }

    def resolve_alias(self, point_id: str) -> str:
        """Resolve an id to its canonical node : an atomic-fact id maps to
        its source turn (dia_id), and a merge-absorbed id to its survivor."""
        point_id = self._atom_parent.get(point_id, point_id)
        seen = set()
        while point_id in self._merge_aliases and point_id not in seen:
            seen.add(point_id)
            point_id = self._merge_aliases[point_id]
        return point_id

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

    def auto_cluster_observators(
        self,
        *,
        min_cluster_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect Level-1 communities over the FACTs and instantiate one
        named observator per community (à la GraphRAG / LightRAG).

        Single-pass Louvain on the cleaned-content similarity graph.
        THOUGHTs, ACTIONs, and entity beacons are excluded ; FACTs not in
        a large-enough community stay attached only to the default
        observator.

        Each point in a community gets an ObservatorView marker so
        retrieve_for_observator / select_observators can route to it.
        Observators can call each other via observator.delegate_query
        (cycle- and depth-bounded).

        Returns a list of {id, name, keywords, point_ids} dicts.
        """
        from metacog.communities import detect_level1_communities
        from metacog.observator import ObservatorView

        comms = detect_level1_communities(
            self.points, self.encoder,
            min_cluster_size=min_cluster_size,
        )
        out: List[Dict[str, Any]] = []
        for c in comms:
            obs = self.declare_observator(
                c.id,
                name=" / ".join(c.keywords[:3]) if c.keywords else c.id,
                keywords=c.keywords,
            )
            member_set = set(c.point_ids)
            for p in self.points:
                if p.id in member_set and obs.id not in p.observator_views:
                    p.observator_views[obs.id] = ObservatorView()
            out.append({
                "id": obs.id,
                "name": obs.name,
                "keywords": c.keywords,
                "point_ids": c.point_ids,
                "n_points": len(c.point_ids),
            })
        return out

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
            "keywords": list(p.keywords or []),
            "tags": list(p.tags or []),
            "n_spike": p.n_spike,
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
            "_spike_total_hops": self._spike_total_hops,
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
        self._spike_total_hops = snapshot.get("_spike_total_hops", 0)
