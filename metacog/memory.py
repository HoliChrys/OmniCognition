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
import re
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


_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def _session_date_tags(content: str) -> List[str]:
    """Deterministic date tags from a turn's leading "[<date>]" prefix.

    Every LoCoMo turn (and any dated ingest) carries its session date in a
    bracket prefix — "[20 April 2022] Speaker: text". That date is STRUCTURED
    metadata, so it must be tagged deterministically on EVERY turn, not left to
    the LLM keyword/entity extractor (which silently skips it on most turns, so
    a date-scoped search misses them). Returns the hierarchical date tags
    `time:year:2022`, `time:month:april`, `time:date:2022-04-20`. Empty when no
    parseable date prefix. Edge-free : these are labels, never relations."""
    m = re.match(r"\s*\[([^\]]+)\]", content or "")
    if not m:
        return []
    inner = m.group(1)
    dm = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", inner)   # 20 April 2022
    if dm:
        day, mon, year = dm.group(1), dm.group(2).lower(), dm.group(3)
    else:
        dm = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", inner)  # April 20, 2022
        if not dm:
            return []
        mon, day, year = dm.group(1).lower(), dm.group(2), dm.group(3)
    if mon not in _MONTHS:
        return []
    return [
        f"time:year:{year}",
        f"time:month:{mon}",
        f"time:date:{year}-{_MONTHS[mon]}-{int(day):02d}",
    ]


def query_date_tags(question: str) -> List[str]:
    """Deterministic date-CONSTRAINT tags from a QUESTION, e.g. "during April
    2022" -> ['time:month:april', 'time:year:2022'] ; "in 2022" -> ['time:year:
    2022'] ; "in July" -> ['time:month:july']. This is the temporal anchor: a
    question that names a period should scope retrieval to that period's turns
    (the deterministic ingest date tags), so a walk cannot drift to another
    month. Sourced from the ORIGINAL question, it stays fixed across the
    agent's reformulated walks. Empty when the question names no period."""
    q = (question or "").lower()
    tags: List[str] = []
    for mon in _MONTHS:
        if re.search(r"\b" + mon + r"\b", q):
            tags.append(f"time:month:{mon}")
            break
    ym = re.search(r"\b(19|20)\d{2}\b", q)
    if ym:
        tags.append(f"time:year:{ym.group(0)}")
    return tags


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
    # Opt-in LLM EVENT detector. When set, each ingested FACT is scanned for a
    # notable event ; a PointKind.EVENT hub is created/deduped (ingest_event)
    # and the turn gravitates onto it. None = current behavior unchanged.
    event_extractor: Any = None
    storage_path: Optional[str] = None

    points: List[Point] = field(default_factory=list)
    observators: Dict[str, Observator] = field(default_factory=dict)
    conversation_log: ConversationLog = field(default_factory=ConversationLog)
    # atomic-fact id -> source turn id (for retrieval resolution).
    _atom_parent: Dict[str, str] = field(default_factory=dict)
    # EVENT hub registry : "<type>::<name>" -> event_id, for resolution/dedup
    # (a war mentioned in 10 turns must resolve to ONE hub, not 10). Rebuilt
    # from points on load(). The hub aggregates the facts that gravitate to it.
    _event_registry: Dict[str, str] = field(default_factory=dict)
    # RETRIEVE-mode BAG : an agent-curated, order-preserving list of node refs
    # the agent decides to collect across iterations (for "find all / list"
    # tasks). Rendered as the exhaustive list answer at the end. Seeded from an
    # event cluster, grown by the agent via bag_add.
    _bag: List[str] = field(default_factory=list)
    _bags: Dict[str, List[str]] = field(default_factory=dict)
    _bag_meta: Dict[str, dict] = field(default_factory=dict)
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
    # LATENT TAG REFINER (run in sleep()) : replays FACT phrase keywords and
    # crystallizes their hierarchical namespace tags ("fingers too big" ->
    # body:finger, health:condition:swelling) so inference queries can SCOPE
    # on the latent category. Opt-in, idempotent, LLM-backed. See
    # metacog.tag_refine.
    tags_refine_enabled: bool = False
    # Run the SAME hierarchical refinement at INGEST time for each created
    # FACT/ACTION (opt-in) instead of waiting for latent sleep. Costs one LLM
    # call per node on the hot path — off by default ; sleep is the cheap path.
    tags_refine_on_ingest: bool = False
    # Resolution ledger driving the LATENT SKILL DISTILLER (run in sleep()).
    # Each entry records a solved task — (query, the walk's resolution path
    # point-ids, the output) — so the distiller can replay it afterwards and
    # crystallize a theoretical tool linked to the semantic facts/thoughts/
    # actions that explicate it, so the next time the task recurs it is
    # retrieved fast WITHOUT forced metacognition. `_distill_cursor` marks
    # how far the distiller has consumed the ledger.
    _resolution_ledger: List[dict] = field(default_factory=list)
    _distill_cursor: int = 0
    # Episodic conversation index : the id of the last message ingested per
    # (user, session), so successive messages chain via sequence_prev and a
    # session reads back in order. Continuous indexation feeds this.
    _session_msg_chain: Dict[tuple, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.storage_path:
            try:
                self.load()
            except FileNotFoundError:
                pass  # first run

    # ------------------------------------------------------------------
    # Isolation
    # ------------------------------------------------------------------

    def snapshot(self) -> "Memory":
        """Return an ISOLATED copy of this memory for mutation-safe use.

        The heavy, stateless resources — `encoder`, `llm`, `executor`,
        `extractor`, `entity_extractor`, `atomic_extractor` — are SHARED
        by reference (they hold no per-memory state and are expensive to
        rebuild). Everything mutable — `points` (each `Point`), the
        observators, the conversation log, every ledger and counter — is
        DEEP-COPIED, so anything the caller does to the snapshot (ingest,
        observe, sleep, crystallize, save) cannot touch the original.

        This is what lets a read-only concurrent harness hand each worker
        the FULL tool surface (including the state-mutating tools) without
        the workers racing on one shared point cloud: give each its own
        `memory.snapshot()`.

        `storage_path` is cleared on the clone so a `save()` on a snapshot
        can never overwrite the original's on-disk store.
        """
        import copy
        # Seed the deepcopy memo with the shared resources keyed by id, so
        # deepcopy returns them as-is (shared) instead of cloning them.
        memo: Dict[int, Any] = {}
        for name in ("encoder", "llm", "executor", "extractor",
                     "entity_extractor", "atomic_extractor"):
            obj = getattr(self, name, None)
            if obj is not None:
                memo[id(obj)] = obj
        clone = copy.deepcopy(self, memo)
        clone.storage_path = None
        return clone

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
        # Deterministic session-date tags from the "[<date>]" content prefix —
        # so EVERY dated turn is date-filterable (time:year/month/date), never
        # depending on the LLM to notice the date. Enables date-scoped search.
        if kind.upper() == "FACT" and not id.startswith(("entity_", "atom_")):
            for dt in _session_date_tags(content):
                if dt not in point.tags:
                    point.tags.append(dt)
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
        # Detect/type an EVENT in the turn and gravitate it onto its hub (opt-in,
        # never on derived nodes ; a flaky extractor must not break ingestion).
        if (
            self.event_extractor is not None
            and kind.upper() == "FACT"
            and not id.startswith(("entity_", "atom_", "event_"))
        ):
            try:
                self._spawn_events(point)
            except Exception:
                pass
        # Optional INGEST-time hierarchical tag refinement for genuine
        # FACT/ACTION nodes (entity beacons / atomics are already typed). Off
        # by default — the latent-sleep pass is the cheap path ; this is the
        # opt-in eager path. The refined tags land in the tag glossary (the
        # registry) automatically since they are appended to point.tags.
        if (
            self.tags_refine_on_ingest
            and kind.upper() in ("FACT", "ACTION")
            and not id.startswith(("entity_", "atom_"))
            and "refined" not in point.tags
        ):
            try:
                from metacog.tag_refine import refine_tags as _refine
                fresh = [t for t in _refine(list(point.keywords or []), self.llm)
                         if t not in point.tags]
                point.tags.extend(fresh)
                if hasattr(self.llm, "generate"):
                    point.tags.append("refined")
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

    @staticmethod
    def _cos(a, b) -> float:
        if not a or not b:
            return 0.0
        num = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return num / (na * nb) if na and nb else 0.0

    def _same_event_llm(self, new_name: str, hub: Point) -> bool:
        """LLM arbitration : is `new_name` the SAME real-world event as the
        existing hub? Bounded (only called for same-type, mid-cosine
        candidates). Fails closed (False) so a flaky LLM never over-merges."""
        if not hasattr(self.llm, "generate"):
            return False
        existing = hub.content.split(" ", 1)[1] if " " in hub.content else hub.content
        try:
            r = self.llm.generate(
                "Are these two references to the SAME real-world event? "
                "Answer only yes or no.\n"
                f"A: {new_name}\nB: {existing}\nAnswer:", max_tokens=4)
            return (r or "").strip().lower().startswith("y")
        except Exception:
            return False

    def ingest_event(
        self,
        name: str,
        etype: str,
        *,
        source_facts: Sequence[Point] = (),
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        salience: float = 0.30,
        t_now: Optional[float] = None,
    ) -> Point:
        """Create (or RESOLVE to an existing) EVENT hub and gravitate the
        relevant source facts onto it.

        The EVENT is a `PointKind.EVENT` aggregator — NOT a fact — carrying a
        hierarchical `event:type:<etype>` schema tag and an interval
        [t_start, t_end?] (t_end absent ⇒ ongoing / right-open). It is the
        temporally-extended HUB the facts gravitate around.

        RESOLUTION / DEDUP (Graphiti recipe, edge-free) : a hub with the same
        type and a name that matches an existing hub (registry key OR cosine
        ≥ 0.85) is REUSED — the new facts are pulled onto the existing hub, so
        "the war" across many turns is ONE node, not many.

        SELECTIVE GRAVITATION (EventKG/MEAD : centrality = relevance) : a source
        fact is pulled onto the hub only when cos(fact, hub) ≥ `salience`, so
        the hub aggregates the salient facts, not noise.

        Cor. 5 : the hub is GENERATOR-sourced ; apply_pull is called directly,
        never via an Observation.
        """
        if t_now is None:
            t_now = self._now()
        nm = (name or "").strip()
        et = (etype or "").strip().lower()
        key = f"{et}::{nm.lower()}"
        emb = tuple(self.encoder.encode(nm or et))

        # ---- resolution / dedup (Graphiti recipe : cosine + LLM arbitration) ----
        hub: Optional[Point] = None
        if key in self._event_registry:
            hub = next((p for p in self.points
                        if p.id == self._event_registry[key]), None)
        if hub is None:
            best, best_s = None, 0.0
            for p in self.points:
                if p.kind is PointKind.EVENT and f"event:type:{et}" in p.tags:
                    s = self._cos(emb, p.embedding_orig)
                    if s > best_s:
                        best, best_s = p, s
            if best is not None:
                if best_s >= 0.85:
                    hub = best
                elif best_s >= 0.45 and self._same_event_llm(nm, best):
                    # same type, ambiguous name — let the LLM decide if it is the
                    # same real-world event ("the war" == "war between two
                    # nations"). This is what cosine-on-names alone misses.
                    hub = best
            if hub is not None:                 # alias the new name to the hub
                self._event_registry[key] = hub.id

        if hub is None:
            tags = ["event", f"event:type:{et}"]
            for dt in _session_date_tags(f"[{t_start}]") if t_start else []:
                tags.append(dt.replace("time:", "event:start:", 1)
                            if dt.startswith("time:") else dt)
            if t_end:
                for dt in _session_date_tags(f"[{t_end}]"):
                    tags.append(dt.replace("time:", "event:end:", 1))
            kws = [w for w in (nm.split() + [et]) if w]
            hub = Point(
                id=f"event_{uuid.uuid4().hex[:8]}",
                content=f"event:{et} {nm}".strip(),
                embedding_orig=emb,
                kind=PointKind.EVENT,
                keywords=kws,
                keywords_embedding=position_weighted_keyword_embedding(
                    kws, self.encoder),
                keywords_source=SourceClass.GENERATOR,
                tags=tags,
            )
            self.points.append(hub)
            self._event_registry[key] = hub.id

        # ---- selective gravitation ----
        # A fact that gravitates is also TAGGED with its hub membership
        # (event:in:<hub_id>) — edge-free, persists with the point, and lets
        # the schema later partition this event's gravitating CLUSTER.
        member = f"event:in:{hub.id}"
        for f in source_facts or ():
            if f is None or f.id == hub.id:
                continue
            if self._cos(f.embedding_orig, hub.embedding_orig) >= salience:
                apply_pull(hub, f, +1.0, t_now)
                if member not in f.tags:
                    f.tags.append(member)
        return hub

    def event_cluster(self, event_id: str) -> List[Point]:
        """The facts that gravitate to an event hub (its `event:in:<id>`
        members) — 'everything there is' about the event."""
        m = f"event:in:{event_id}"
        return [p for p in self.points if m in p.tags]

    def event_absorb(self, event_id: str, facts: Sequence[Point],
                     *, t_now: Optional[float] = None) -> int:
        """Fold newly-found facts INTO an event's gravitating cluster (= the
        context's knowledge base) : tag event:in:<id> and pull onto the hub.
        Used when a schema GAP is filled from the corpus — 'after the first
        search the result joins the knowledge base', so later slots and future
        queries see it. Edge-free ; returns the number of facts absorbed."""
        hub = next((p for p in self.points if p.id == event_id), None)
        if hub is None:
            return 0
        if t_now is None:
            t_now = self._now()
        member = f"event:in:{event_id}"
        n = 0
        for f in facts or ():
            if f is None or f.id == event_id or member in f.tags:
                continue
            f.tags.append(member)
            apply_pull(hub, f, +1.0, t_now)
            n += 1
        return n

    @staticmethod
    def _centroid(vecs: Sequence[Sequence[float]]) -> Tuple[float, ...]:
        """Mean (L2-normalised) of a set of embeddings — the AGGREGATE 'aboutness'
        vector. Mean-pooling denoises : each member's idiosyncratic tokens cancel,
        the shared latent theme survives. That is what an OBLIQUE query (latent
        relevance, no single member matching on surface) can lock onto."""
        vv = [v for v in vecs if v]
        if not vv:
            return ()
        dim = len(vv[0])
        acc = [0.0] * dim
        for v in vv:
            for i in range(dim):
                acc[i] += v[i]
        n = float(len(vv))
        m = [x / n for x in acc]
        nrm = sum(x * x for x in m) ** 0.5 or 1.0
        return tuple(x / nrm for x in m)

    def event_centroid(self, event_id: str) -> Tuple[float, ...]:
        """Centroid of an event's gravitating cluster (+ the hub name vector) —
        the event's latent-theme representation for oblique routing."""
        hub = next((p for p in self.points if p.id == event_id), None)
        vecs = [p.embedding_orig for p in self.event_cluster(event_id)]
        if hub is not None and hub.embedding_orig:
            vecs.append(hub.embedding_orig)
        return self._centroid(vecs)

    def context_members(self, ctx: str) -> List[Point]:
        """A CONTEXT = an event TYPE (war, earnings, …) : the thematic frame an
        oblique query is SITUATED in. Returns the UNION of facts gravitating to
        every event of that type — the broadest pool of latent evidence for the
        context (deduped, order-preserving)."""
        tag = f"event:type:{(ctx or '').strip().lower()}"
        seen: set = set()
        out: List[Point] = []
        for ev in [p for p in self.points
                   if p.kind is PointKind.EVENT and tag in p.tags]:
            for f in self.event_cluster(ev.id):
                if f.id not in seen:
                    seen.add(f.id)
                    out.append(f)
        return out

    def context_centroid(self, ctx: str) -> Tuple[float, ...]:
        """Centroid pooled over the WHOLE context (all events of the type and
        their clusters) — strictly more evidence than a single event's centroid,
        so it denoises harder and matches an oblique query's latent theme best."""
        tag = f"event:type:{(ctx or '').strip().lower()}"
        vecs = [p.embedding_orig for p in self.context_members(ctx)]
        vecs += [p.embedding_orig for p in self.points
                 if p.kind is PointKind.EVENT and tag in p.tags
                 and p.embedding_orig]
        return self._centroid(vecs)

    # ---- NAMED BAGS : retrieve-mode + cluster + schema-slot containers ------
    # The bag is the primitive container used everywhere parallel channels
    # collect ids — the agent's retrieve list ("default"), an event cluster
    # ("event:<hid>"), a schema slot ("event:<hid>:slot:<slot>"), a normal
    # search ("normal:<q>") — all live in `_bags`. Intersecting two bags is the
    # cheap, INTERPRETED join (a fact found by both channels is the high-
    # precision nugget). ACTION/THOUGHT generators read every non-empty bag for
    # context (an empty bag has no influence — the invariant of empty steps).
    def _bag_ref(self, name: str) -> List[str]:
        if not hasattr(self, "_bags") or self._bags is None:
            self._bags = {}
        return self._bags.setdefault(name or "default", [])

    def _bag_meta_ref(self) -> Dict[str, dict]:
        if not hasattr(self, "_bag_meta") or self._bag_meta is None:
            self._bag_meta = {}
        return self._bag_meta

    @staticmethod
    def _bag_describe(schema: Any) -> str:
        """Derive a description from a bag schema, so description tracks schema."""
        if not isinstance(schema, dict):
            return str(schema or "")
        kind = schema.get("kind", "bag")
        if kind == "event_cluster":
            return f"Exhaustive cluster of event {schema.get('event_id', '')}."
        if kind == "event_slot":
            tier = "core" if schema.get("core") else "peripheral"
            return f"Schema slot «{schema.get('slot', '')}» ({tier})."
        if kind == "event_schema":
            return "Schema roll-up: slots " + \
                   ", ".join(schema.get("slots", []) or []) + "."
        return f"{kind} bag."

    def bag_add(self, ids, *, bag: str = "default",
                description: Optional[str] = None, schema: Any = None) -> int:
        """Append node ref(s) to a NAMED bag (deduped, order-preserving).
        Default bag is the agent's retrieve list (backwards-compatible).
        `description` / `schema` annotate the bag so the agent can DECIDE which
        bag to use and how to render it (set/overwritten when provided)."""
        b = self._bag_ref(bag)
        if isinstance(ids, str):
            ids = [ids]
        for i in ids or []:
            if isinstance(i, str) and i and i not in b:
                b.append(i)
        if description is not None or schema is not None:
            m = self._bag_meta_ref().setdefault(bag, {})
            if schema is not None:
                m["schema"] = schema
                # description MUST track the schema : a schema change with no
                # explicit description re-derives the description from it.
                if description is None:
                    description = self._bag_describe(schema)
            if description is not None:
                m["description"] = description
        # mirror the default into the legacy `_bag` field so external readers
        # (format_bag_answer wiring, agent loop) keep working unchanged.
        if bag == "default":
            self._bag = list(b)
        return len(b)

    def bag_meta(self, name: str) -> dict:
        """A bag's decision metadata : description, schema, size, sample ids."""
        b = self._bag_ref(name)
        m = self._bag_meta_ref().get(name, {})
        return {"name": name, "size": len(b),
                "description": m.get("description", ""),
                "schema": m.get("schema"), "sample": b[:5]}

    def curated_bags(self) -> Dict[str, list]:
        """The agent's CURATED bags as a map {name: [(id, content)]} — the
        default retrieve list plus any named bags it collected. Internal channel
        bags (event:*) are excluded : they feed generators, not the surface
        answer. This is the map of lists the final generation / keepup injects."""
        return {n: self.bag_items(bag=n) for n in self.bag_names()
                if not n.startswith("event:")}

    def bag_overview(self) -> List[dict]:
        """Every non-empty bag with its decision metadata — what the agent reads
        to choose which list(s) to surface and how. Default + curated bags first,
        internal channel bags (event:*) after."""
        names = self.bag_names()
        names.sort(key=lambda n: (n.startswith("event:"), n != "default", n))
        return [self.bag_meta(n) for n in names]

    def bag_items(self, *, bag: str = "default"):
        """A bag as (id, content) pairs, in collection order."""
        b = self._bag_ref(bag)
        by_id = {p.id: p for p in self.points}
        return [(i, getattr(by_id.get(i), "content", "")) for i in b]

    def bag_clear(self, *, bag: Optional[str] = None) -> None:
        """Clear ONE named bag (default 'default') or, when `bag is None`, the
        default bag — backwards-compatible with the old no-arg form."""
        if not hasattr(self, "_bags") or self._bags is None:
            self._bags = {}
        name = bag if bag is not None else "default"
        self._bags[name] = []
        if name == "default":
            self._bag = []

    def bag_names(self) -> List[str]:
        if not hasattr(self, "_bags") or self._bags is None:
            self._bags = {}
        return [n for n, b in self._bags.items() if b]

    def bag_intersect(self, names: Sequence[str]) -> List[str]:
        """Intersection of the NAMED bags — the INTERPRETED join : an id that
        every listed channel surfaced. Order-preserving on the first bag."""
        names = [n for n in (names or []) if n]
        if not names:
            return []
        first = self._bag_ref(names[0])
        rest = [set(self._bag_ref(n)) for n in names[1:]]
        if not rest:
            return list(first)
        return [i for i in first if all(i in s for s in rest)]

    def bag_union(self, names: Sequence[str]) -> List[str]:
        """Union of the NAMED bags (dedup, order-preserving)."""
        seen: set = set()
        out: List[str] = []
        for n in (names or []):
            for i in self._bag_ref(n):
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        return out

    def bag_publish_cluster(self, event_id: str) -> int:
        """Publish an event's gravitating cluster as a NAMED bag — so the
        cluster becomes an intersect-able channel like any other bag. Carries a
        description + schema so the agent can decide how to use it."""
        cl = self.event_cluster(event_id)
        hub = next((p for p in self.points if p.id == event_id), None)
        nm = (hub.content if hub else event_id)
        return self.bag_add(
            [p.id for p in cl], bag=f"event:{event_id}",
            description=f"Everything gravitating to the event «{nm}» "
                        f"({len(cl)} nodes) — the exhaustive cluster.",
            schema={"kind": "event_cluster", "event_id": event_id,
                    "element": "node/fact"})

    def bag_publish_schema(self, event_id: str, fill: dict) -> int:
        """Publish each filled schema SLOT as its own bag, plus a roll-up
        'event:<id>:schema' = union of the slots. Each carries a description +
        schema (the slot role) so the agent can decide what to surface."""
        rolled: List[str] = []
        for slot, hits in ((fill or {}).get("filled") or {}).items():
            ids = [h["id"] for h in hits if isinstance(h, dict) and h.get("id")]
            if ids:
                core = slot in set((fill or {}).get("core") or [])
                self.bag_add(
                    ids, bag=f"event:{event_id}:slot:{slot}",
                    description=f"Schema slot «{slot}» "
                                f"({'core' if core else 'peripheral'}) of the "
                                f"event — the nodes filling this role.",
                    schema={"kind": "event_slot", "slot": slot, "core": core})
                rolled += [i for i in ids if i not in rolled]
        if rolled:
            self.bag_add(
                rolled, bag=f"event:{event_id}:schema",
                description="Union of all filled schema slots for the event.",
                schema={"kind": "event_schema",
                        "slots": list((fill or {}).get("slots") or []),
                        "core": list((fill or {}).get("core") or [])})
        return len(rolled)

    # ---- bi-temporal validity (Graphiti : t_valid / t_invalid) --------------
    # event_time lives in the time:* tags ; ingestion_time is t_last_obs. A fact
    # superseded by newer contradicting info is INVALIDATED (tagged
    # valid:until:<date>) rather than deleted — preserving history for "as-of"
    # queries. valid:until absent ⇒ still valid / ongoing (right-open).
    @staticmethod
    def is_valid(fact: Point, as_of: Optional[str] = None) -> bool:
        """True if `fact` holds (no valid:until, or `as_of` precedes it)."""
        until = next((t.split(":", 2)[2] for t in fact.tags
                      if t.startswith("valid:until:")), None)
        if not until:
            return True
        return bool(as_of) and as_of < until

    def invalidate(self, fact_id: str, until: str) -> bool:
        """Mark a fact invalid from `until` (a YYYY-MM-DD date), last-write-wins.
        Edge-free : a tag, never a deletion."""
        p = next((q for q in self.points if q.id == fact_id), None)
        if p is None:
            return False
        tag = f"valid:until:{until}"
        if not any(t.startswith("valid:until:") for t in p.tags):
            p.tags.append(tag)
            if "invalidated" not in p.tags:
                p.tags.append("invalidated")
            return True
        return False

    def invalidate_contradictions(self, new_fact: Point, *,
                                  threshold: float = 0.55,
                                  max_check: int = 5) -> List[str]:
        """Graphiti-style temporal invalidation : find prior FACTs semantically
        close to `new_fact`, ask the LLM if the new one CONTRADICTS/supersedes
        each, and invalidate the superseded ones at the new fact's event date
        (last-write-wins). Bounded (top `max_check` by cosine). Returns the
        invalidated ids. Failure-safe / no-op without an LLM."""
        if not hasattr(self.llm, "generate"):
            return []
        until = next((t.split(":", 2)[2] for t in new_fact.tags
                      if t.startswith("time:date:")), None) or "9999-99-99"
        cands = []
        for p in self.points:
            if (p.kind is PointKind.FACT and p.id != new_fact.id
                    and not p.id.startswith(("entity_", "atom_", "event_"))
                    and self.is_valid(p)):
                s = self._cos(new_fact.embedding_orig, p.embedding_orig)
                if s >= threshold:
                    cands.append((s, p))
        cands.sort(key=lambda x: -x[0])
        invalidated: List[str] = []
        for _, p in cands[:max_check]:
            try:
                r = self.llm.generate(
                    "Does NEW information CONTRADICT and supersede OLD (a later "
                    "update replacing it)? Answer only yes or no.\n"
                    f"OLD: {p.content[:160]}\nNEW: {new_fact.content[:160]}\n"
                    "Answer:", max_tokens=4)
            except Exception:
                r = ""
            if (r or "").strip().lower().startswith("y"):
                if self.invalidate(p.id, until):
                    invalidated.append(p.id)
        return invalidated

    _EVENT_STOP = {
        "the", "of", "and", "to", "in", "on", "for", "with", "from", "between",
        "over", "into", "this", "that", "their", "his", "her", "its", "new",
        "us", "a", "an", "by", "at", "as", "is", "are", "was",
    }

    def _event_tokens(self, hub: Point) -> set:
        """Salient entity tokens of an event hub's name (drop its type word and
        stopwords) — the participants that decide if two events are facets of
        the SAME situation."""
        et = next((t.split(":", 2)[2] for t in hub.tags
                   if t.startswith("event:type:")), "")
        name = hub.content.split(" ", 1)[1] if " " in hub.content else ""
        et_words = set(re.findall(r"[a-z]{3,}", et.lower()))
        return {w for w in re.findall(r"[a-z]{3,}", name.lower())
                if w not in self._EVENT_STOP and w not in et_words}

    def consolidate_events(
        self, *, min_shared: int = 1, name_cos: float = 0.85,
        cluster_overlap: int = 1, use_llm: bool = True,
    ) -> Dict[str, Any]:
        """Hierarchical aggregation : micro-event hubs that describe the SAME
        real-world situation are MERGED into a macro hub ; its cluster becomes
        the UNION of their gravitating facts. Fixes event fragmentation (27
        tweets → 31 micro-hubs → 1 'iran-israel war' macro).

        Multi-signal merge (mirrors the ingest dedup recipe, but post-hoc, so
        it ALSO catches dedup failures where the LLM gave the same event two
        very different names) — pair (a,b) merges when ANY of :
          * `len(toks(a) ∩ toks(b)) ≥ min_shared`         (shared entity)
          * same etype AND cos(name(a), name(b)) ≥ `name_cos`   (synonyms)
          * same etype AND `|cluster(a) ∩ cluster(b)| ≥ cluster_overlap`
              (the SAME facts gravitate to both : strongest signal)
          * same etype, ambiguous cosine (0.45-0.85), LLM arbitrates yes
              (Graphiti dedup recipe, bounded ; skipped when `use_llm=False`)
        Edge-free : facts re-tagged event:in:<macro> and pulled onto the macro,
        registry re-pointed. Last-write-wins on hub absorption ; deterministic
        choice of canonical = the hub with the largest gravitating cluster."""
        events = [p for p in self.points if p.kind is PointKind.EVENT]
        toks = {e.id: self._event_tokens(e) for e in events}
        etype = {e.id: next((t.split(":", 2)[2] for t in e.tags
                             if t.startswith("event:type:")), "") for e in events}
        cluster_ids = {e.id: {p.id for p in self.event_cluster(e.id)}
                       for e in events}
        parent = {e.id: e.id for e in events}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a_id, b_id):
            parent[find(a_id)] = find(b_id)

        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                a, b = events[i], events[j]
                if find(a.id) == find(b.id):
                    continue
                if len(toks[a.id] & toks[b.id]) >= min_shared:
                    union(a.id, b.id); continue
                if etype[a.id] and etype[a.id] == etype[b.id]:
                    if (cluster_ids[a.id] and cluster_ids[b.id]
                            and len(cluster_ids[a.id] & cluster_ids[b.id])
                                >= cluster_overlap):
                        union(a.id, b.id); continue
                    cos = self._cos(a.embedding_orig, b.embedding_orig)
                    if cos >= name_cos:
                        union(a.id, b.id); continue
                    if use_llm and cos >= 0.45 and self._same_event_llm(
                            a.content.split(" ", 1)[-1], b):
                        union(a.id, b.id)

        groups: Dict[str, List[Point]] = {}
        for e in events:
            groups.setdefault(find(e.id), []).append(e)

        t_now = self._now()
        merged = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            # canonical = the hub with the largest gravitating cluster
            canon = max(members, key=lambda e: len(self.event_cluster(e.id)))
            cmem = f"event:in:{canon.id}"
            for e in members:
                if e.id == canon.id:
                    continue
                for f in self.event_cluster(e.id):
                    if cmem not in f.tags:
                        f.tags.append(cmem)
                        apply_pull(canon, f, +1.0, t_now)
                # re-point the registry entries of the merged hub to the canon
                for k, v in list(self._event_registry.items()):
                    if v == e.id:
                        self._event_registry[k] = canon.id
                if cmem not in e.tags:           # the micro stays as a sub-event
                    e.tags.append(cmem)
                merged += 1
        return {"events": len(events), "groups": sum(
            1 for m in groups.values() if len(m) >= 2), "merged": merged}

    def event_tag_signature(self, event_id: str, *, top: int = 12) -> List[str]:
        """The TAG SIGNATURE of an event : the meaningful tags shared across its
        gravitating cluster, most-frequent first (structural tags dropped). A
        compact, discriminative fingerprint of what the event is about — far
        less to reason over than the raw facts."""
        from collections import Counter
        cl = self.event_cluster(event_id)
        cnt: Counter = Counter()
        for p in cl:
            for t in p.tags:                    # hierarchical tags (if refined)
                if (":" in t and not t.startswith(("event:in:", "valid:",
                                                   "time:"))
                        and t not in ("fact", "refined", "invalidated")):
                    cnt[t] += 1
            for kw in (p.keywords or []):        # + the facts' keywords
                k = (kw or "").strip().lower()
                if len(k) >= 3:
                    cnt[k] += 1
        # the SHARED signal = tags/keywords recurring across the cluster.
        return [t for t, c in cnt.most_common(top) if c >= 2] or \
               [t for t, _ in cnt.most_common(top)]

    def _route_event_llm(self, query: str, cands: List[Point]) -> Optional[Point]:
        """LLM event routing : of these candidate events, which one is the query
        REALLY about? Uses reasoning, so it bridges the obliqueness cosine can't
        (a query sharing no words with its event). Each candidate is shown by
        its TAG SIGNATURE (the cluster's shared tags) — compact and
        discriminative, much less to process than the raw facts. Returns the
        chosen Point or None (declines / no LLM). Fails closed."""
        if not hasattr(self.llm, "generate") or not cands:
            return None
        lines = []
        for i, m in enumerate(cands, 1):
            et = next((t.split(":", 2)[2] for t in m.tags
                       if t.startswith("event:type:")), "")
            nm = m.content.split(" ", 1)[1] if " " in m.content else m.content
            sig = ", ".join(self.event_tag_signature(m.id, top=10))
            lines.append(f"{i}. {nm} [{et}] — tags: {sig}")
        try:
            r = self.llm.generate(
                "Which numbered EVENT is the QUESTION really about? It may share "
                "NO words with the event (e.g. 'mocking branding while wars "
                "intensify' is about a WAR). Reason about the referent. Answer "
                "ONLY the number, or 0 if none.\n\n"
                f"QUESTION: {query}\n\nEVENTS:\n" + "\n".join(lines) + "\nNumber:",
                max_tokens=6)
        except Exception:
            return None
        m = re.search(r"\d+", r or "")
        if not m:
            return None
        n = int(m.group(0))
        return cands[n - 1] if 1 <= n <= len(cands) else None

    def detect_event_type(self, query: str, *, threshold: float = 0.42
                          ) -> Optional[dict]:
        """Decide whether a QUERY is about an event, and of which TYPE.

        Two signals (grounded first) : (1) the EVENT hubs are first-class
        retrievable nodes — the query's nearest hub (cosine ≥ threshold) carries
        the type, because the hub embeds near its gravitating facts ; (2)
        fallback to the LLM event extractor on the query for events not yet
        hubbed. Returns `{name, etype, event_id, score, via}` or None (then the
        caller runs the normal walk/clue_search)."""
        q = (query or "").strip()
        if not q:
            return None
        try:
            qe = tuple(self.encoder.encode(q))
        except Exception:
            return None
        # CLUSTER-AWARE multi-event routing : score each MACRO event not just by
        # its name's cosine to the query but by how well its gravitating CLUSTER
        # matches — so a query routes to the event whose facts it is really
        # about, not merely the nearest hub name (multi-event disambiguation).
        ev_by_id = {p.id: p for p in self.points if p.kind is PointKind.EVENT}

        def _macro(hub: Point) -> Point:
            seen: set = set()
            while True:
                mid = next((t.split(":", 2)[2] for t in hub.tags
                            if t.startswith("event:in:")
                            and t.split(":", 2)[2] in ev_by_id
                            and t.split(":", 2)[2] != hub.id), None)
                if not mid or mid in seen:
                    return hub
                seen.add(mid)
                hub = ev_by_id[mid]

        macro_score: Dict[str, float] = {}
        macro_pt: Dict[str, Point] = {}
        for p in ev_by_id.values():
            m = _macro(p)
            name_s = self._cos(qe, m.embedding_orig)
            clus = self.event_cluster(m.id)
            # best single member — catches a SPECIFIC query naming one fact.
            clus_s = max((self._cos(qe, f.embedding_orig)
                          for f in clus[:40]), default=0.0)
            # CONTEXT centroid — the latent-theme signal. An OBLIQUE query shares
            # no surface words with any single member (so name_s and clus_s are
            # both low), but is close to the mean of the whole context : the
            # denoised 'aboutness' vector. This is the path that avoids '0 clue'.
            et = next((t.split(":", 2)[2] for t in m.tags
                       if t.startswith("event:type:")), "")
            cent_s = self._cos(qe, self.context_centroid(et)) if et else 0.0
            s = max(name_s, 0.5 * name_s + 0.5 * clus_s, cent_s)
            if s > macro_score.get(m.id, -1.0):
                macro_score[m.id] = s
                macro_pt[m.id] = m
        if macro_score:
            ranked = sorted(macro_score, key=macro_score.get, reverse=True)
            best_id = ranked[0]
            via = "hub"
            # LLM ROUTING for oblique disambiguation. When several events
            # compete, cosine cannot route a query that shares NO surface words
            # with its own evidence ("mocking branding while wars intensify" ->
            # a war). Give the LLM the event glossary and let it pick by
            # reasoning (cosine only narrows the candidate list). Cosine is the
            # fallback (single event, or the LLM declines).
            cands = [macro_pt[i] for i in ranked[:8]]
            if len(cands) > 1:
                pick = self._route_event_llm(q, cands)
                if pick is not None:
                    best_id, via = pick.id, "llm"
            if via == "llm" or macro_score[best_id] >= threshold:
                best = macro_pt[best_id]
                et = next((t.split(":", 2)[2] for t in best.tags
                           if t.startswith("event:type:")), "")
                nm = best.content.split(" ", 1)[1] if " " in best.content else ""
                return {"name": nm, "etype": et, "event_id": best.id,
                        "score": round(macro_score[best_id], 3), "via": via}
        if self.event_extractor is not None:
            try:
                evs = self.event_extractor.extract_events(q)
            except Exception:
                evs = []
            if evs:
                return {"name": evs[0].name, "etype": evs[0].etype,
                        "event_id": None, "score": 0.0, "via": "llm"}
        return None

    def _spawn_events(self, source_fact: Point) -> None:
        """Detect events in a freshly ingested FACT and gravitate it onto each
        event's hub (create/dedup the hub via ingest_event). The selective
        gravitation gate in ingest_event keeps only turns actually about the
        event."""
        evs = self.event_extractor.extract_events(source_fact.content)
        if not evs:
            return
        t_now = self._now()
        for e in evs:
            self.ingest_event(
                e.name, e.etype, source_facts=[source_fact],
                t_start=getattr(e, "t_start", None), t_now=t_now,
            )

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
    # Episodic conversation index : EVERY message of a session (user AND
    # agent), indexed continuously, with its timestamp preserved. The
    # indexation is ASYNC and NON-BLOCKING w.r.t. search — a background
    # worker does the (LLM-keyworded) ingest while queries read the cloud.
    # ------------------------------------------------------------------

    def _ensure_indexer(self) -> None:
        """Lazily start the background indexing worker (daemon thread). The
        queue/thread are plain instance attributes — never pickled (save()
        snapshots points only)."""
        if getattr(self, "_index_thread", None) is not None:
            return
        import queue as _queue
        import threading
        self._index_queue = _queue.Queue()
        self._index_lock = threading.Lock()
        t = threading.Thread(target=self._index_worker, daemon=True,
                             name="metacog-indexer")
        self._index_thread = t
        t.start()

    def _index_worker(self) -> None:
        while True:
            job = self._index_queue.get()
            try:
                if job is not None:
                    self._ingest_message_sync(**job)
            except Exception:
                pass  # never let a bad message kill the indexer
            finally:
                self._index_queue.task_done()

    def ingest_message(
        self,
        content: str,
        *,
        role: str,
        user_id: str,
        session_id: str,
        timestamp: Optional[str] = None,
        block: bool = False,
    ):
        """Index ONE conversation message (role = "user" or "agent") of a
        session, with its timestamp. Call it for EVERY message on both
        sides — this is the continuous episodic feed.

        Async by default : the message is queued and indexed by the
        background worker, so the call returns immediately and never blocks
        a concurrent search. Pass `block=True` (or call `flush_index()`) to
        index synchronously / wait — used in tests and at shutdown.

        Returns the created Point when `block=True`, else a queued-ack dict."""
        job = dict(content=content, role=role, user_id=user_id,
                   session_id=session_id, timestamp=timestamp)
        if block:
            return self._ingest_message_sync(**job)
        self._ensure_indexer()
        self._index_queue.put(job)
        return {"queued": True}

    def _ingest_message_sync(
        self, *, content: str, role: str, user_id: str,
        session_id: str, timestamp: Optional[str],
    ) -> Point:
        role = (role or "user").lower()
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()
        ts = str(timestamp)
        date = ts.split("T")[0] if "T" in ts else (ts.split()[0] if ts else "")
        # Prefix with [timestamp] role: so temporal retrieval + the
        # relative-date machinery have an absolute anchor (episodic system :
        # the horodatage of each message is preserved verbatim).
        prefixed = f"[{ts}] {role}: {content}"
        lock = getattr(self, "_index_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            prev = self._session_msg_chain.get((user_id, session_id))
            p = self.ingest(prefixed, kind="FACT", sequence_prev=prev)
            p.add_tag("episode", "message", f"role:{role}",
                      f"user:{user_id}", f"session:{session_id}",
                      f"ts:{ts}", f"date:{date}")
            self._session_msg_chain[(user_id, session_id)] = p.id
        finally:
            if lock is not None:
                lock.release()
        return p

    def flush_index(self, timeout: Optional[float] = None) -> None:
        """Block until the async index queue is drained (tests / shutdown)."""
        q = getattr(self, "_index_queue", None)
        if q is not None:
            q.join()

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

    def capture_code_tool(
        self,
        code: str,
        *,
        context: str = "",
        lang: str = "python",
        user_id: str = "",
        session_id: str = "",
        date: Optional[str] = None,
    ) -> Optional[Point]:
        """Chat-side hook : whenever code is GENERATED, ask whether it is a
        reusable TOOL and, if so, FEED IT INTO THE RAG.

        Self-assessment uses `assess_tool_intent` (surface markers + LLM
        adjudication) over the code + its context. When it reads as an
        executable capability, the code is re-indexed as a normal tool node
        — an `ACTION` point with `exec_spec` (executable) tagged
        `tool · skill · code · kind:<…> · name:<…> · user:<id> ·
        session:<id> · date:<…>`. Being a normal node it is retrieved by
        the walk like any fact (gold = tool fact OR semantic fact), reused
        via match_tool, and persisted with the cloud.

        Returns the created tool Point, or None when the code is judged not
        a reusable tool. Fail-safe : any error → None (chat never blocked)."""
        from metacog.skills import assess_tool_intent
        code = (code or "").strip()
        if not code:
            return None
        try:
            intent = assess_tool_intent(f"{context}\n\n{code[:600]}", self.llm)
        except Exception:
            return None
        if not intent.is_executable:
            return None
        if date is None:
            from datetime import date as _date
            date = _date.today().isoformat()
        import re as _re
        mname = (_re.search(r"\bdef\s+([A-Za-z_]\w*)", code)
                 or _re.search(r"\bfunction\s+([A-Za-z_]\w*)", code))
        name = (mname.group(1) if mname else
                (_re.sub(r"[^a-z0-9]+", "_", (context or "tool").lower())
                 .strip("_")[:40] or "tool"))
        desc = context.strip() or f"tool {name} ({intent.kind})"
        extra = []
        if user_id:
            extra.append(f"user:{user_id}")
        if session_id:
            extra.append(f"session:{session_id}")
        extra.append(f"date:{date}")
        return self._make_tool_node(
            code, name=name, kind=intent.kind, desc=desc, lang=lang,
            extra_tags=extra,
        )

    def _make_tool_node(
        self, code: str, *, name: str, kind: str, desc: str,
        lang: str = "python", extra_tags: Sequence[str] = (),
    ) -> Point:
        """Create an executable tool node (ACTION + exec_spec) tagged
        tool·skill·code·kind:<…>·name:<…> plus any extra tags. Shared by
        capture_code_tool and push_code."""
        import re as _re
        content = f"tool[{name}] {desc}"
        kws: List[str] = []
        seen: set = set()
        for w in (_re.findall(r"[A-Za-z_]\w{2,}", name + " " + desc)
                  + _re.findall(r"[A-Za-z_]\w{3,}", code)[:10]):
            wl = w.lower()
            if wl not in seen:
                seen.add(wl); kws.append(wl)
        kw_emb = position_weighted_keyword_embedding(kws[:12], self.encoder)
        tags = (["tool", "skill", "code", f"kind:{kind}", f"name:{name}"]
                + list(extra_tags))
        tool = Point(
            id=f"skill_{uuid.uuid4().hex[:8]}",
            content=content,
            embedding_orig=tuple(self.encoder.encode(content)),
            kind=PointKind.ACTION,
            keywords=kws[:12],
            keywords_embedding=kw_emb,
            keywords_source=SourceClass.GENERATOR,
            tags=tags,
        )
        tool.exec_spec = {"lang": lang, "code": code}
        self.points.append(tool)
        return tool

    # ------------------------------------------------------------------
    # push_code : evaluate generated code → project-documentation FACT
    # and/or a metacognitively-rewritten executable TOOL, linked
    # bidirectionally via ref: tags when it is both.
    # ------------------------------------------------------------------

    def _evaluate_code(self, code: str, context: str) -> Dict[str, Any]:
        """Classify generated code. Returns
        {"project": bool, "tool": bool, "kind": str, "name": str}.
        LLM-adjudicated with a deterministic fallback (markers + def-name)."""
        import re as _re
        mname = (_re.search(r"\bdef\s+([A-Za-z_]\w*)", code)
                 or _re.search(r"\bfunction\s+([A-Za-z_]\w*)", code))
        name = mname.group(1) if mname else (
            _re.sub(r"[^a-z0-9]+", "_", (context or "tool").lower())
            .strip("_")[:40] or "tool")
        out = {"project": False, "tool": False, "kind": "code", "name": name}
        # tool verdict via the shared intent assessor (markers + LLM).
        try:
            from metacog.skills import assess_tool_intent
            intent = assess_tool_intent(f"{context}\n\n{code[:600]}", self.llm)
            out["tool"] = bool(intent.is_executable)
            out["kind"] = intent.kind if intent.is_executable else "code"
        except Exception:
            pass
        # project / doc verdict (one short LLM line ; default no).
        if hasattr(self.llm, "generate"):
            try:
                raw = (self.llm.generate(
                    "Is this code part of a real software PROJECT (worth keeping "
                    "as documentation to learn about that project)? And could it "
                    "be a reusable TOOL? Reply one line:\n"
                    "PROJECT: <yes|no> TOOL: <yes|no> NAME: <snake_case|->\n\n"
                    f"CONTEXT: {context[:200]}\nCODE: {code[:500]}",
                    max_tokens=24) or "").lower()
                if "project: yes" in raw or "project:yes" in raw:
                    out["project"] = True
                if "tool: yes" in raw or "tool:yes" in raw:
                    out["tool"] = True
                mn = _re.search(r"name:\s*([a-z_]\w*)", raw)
                if mn and mn.group(1) not in ("-", "none"):
                    out["name"] = mn.group(1)
            except Exception:
                pass
        return out

    def _rewrite_as_tool(self, code: str, context: str) -> str:
        """Metacognitive interpretation : rewrite code as a standalone,
        reusable tool exposing `def run(args: dict)`. Falls back to the
        original code on any failure (never blocks)."""
        if not hasattr(self.llm, "generate"):
            return code
        try:
            raw = (self.llm.generate(
                "Rewrite the code as a STANDALONE reusable tool exposing "
                "`def run(args: dict)`. Keep the behaviour, strip "
                "project-specific coupling. Output ONLY the code, no prose.\n\n"
                f"CONTEXT: {context[:200]}\nCODE:\n{code[:1200]}",
                max_tokens=400) or "").strip()
            import re as _re
            m = _re.search(r"```(?:\w+)?\s*(.+?)```", raw, _re.DOTALL)
            if m:
                raw = m.group(1).strip()
            return raw if ("def run" in raw or "function run" in raw) else code
        except Exception:
            return code

    def _make_doc_node(self, code: str, context: str,
                       tags: Sequence[str], *, name: str) -> Point:
        """A semantic FACT node documenting project code — retrievable like
        any fact, treated as documentation that teaches about the project."""
        import re as _re
        content = (f"[doc:{name}] " + (context.strip() + "\n" if context else "")
                   + code.strip())[:2000]
        kws: List[str] = []
        seen: set = set()
        for w in (_re.findall(r"[A-Za-z_]\w{2,}", name + " " + context)
                  + _re.findall(r"[A-Za-z_]\w{3,}", code)[:12]
                  + [t for t in tags if t.startswith(("project:", "branch:"))]):
            wl = w.lower()
            if wl not in seen:
                seen.add(wl); kws.append(wl)
        kw_emb = position_weighted_keyword_embedding(kws[:14], self.encoder)
        doc = Point(
            id=f"skill_{uuid.uuid4().hex[:8]}",
            content=content,
            embedding_orig=tuple(self.encoder.encode(content)),
            kind=PointKind.FACT,
            keywords=kws[:14],
            keywords_embedding=kw_emb,
            keywords_source=SourceClass.GENERATOR,
            tags=list(tags),
        )
        self.points.append(doc)
        return doc

    def push_code(
        self,
        code: str,
        *,
        context: str = "",
        project: str = "",
        branch: str = "",
        github_user: str = "",
        lang: str = "python",
        user_id: str = "",
        session_id: str = "",
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Push generated code into the RAG. The server EVALUATES it and
        routes it :

          - PROJECT code → a semantic FACT node, tagged
            `doc·code·project:<…>·branch:<…>·github:<user>·user:·session:·
            date:`, treated as DOCUMENTATION that teaches about the project ;
          - TOOL-able code → a metacognitive rewrite (`_rewrite_as_tool`,
            exposing `def run(args)`) re-indexed as an executable tool node ;
          - BOTH → create both and keep a BIDIRECTIONAL filiation via
            `ref:` tags (`ref:doc:<id>` on the tool, `ref:tool:<id>` on the
            doc) plus co-location, so the tool remembers the project it came
            from and the project doc points at the derived tool.

        Fail-safe : on any error the code is still captured as a plain tool
        if it looks executable (chat is never blocked). Returns a summary."""
        code = (code or "").strip()
        if not code:
            return {"pushed": False}
        if date is None:
            from datetime import date as _date
            date = _date.today().isoformat()
        ev = self._evaluate_code(code, context)
        is_project = ev["project"] or bool(project or branch or github_user)
        is_tool = ev["tool"]
        name = ev["name"]
        meta: List[str] = ["code"]
        for k, v in (("project", project), ("branch", branch),
                     ("github", github_user), ("user", user_id),
                     ("session", session_id)):
            if v:
                meta.append(f"{k}:{v}")
        meta.append(f"date:{date}")

        doc = tool = None
        if is_project:
            doc = self._make_doc_node(code, context, ["doc"] + meta, name=name)
        if is_tool:
            rewritten = self._rewrite_as_tool(code, context) or code
            tool = self._make_tool_node(
                rewritten, name=name, kind=ev["kind"],
                desc=context or f"tool {name}", lang=lang, extra_tags=meta,
            )
        # Bidirectional project↔tool filiation via ref: tags.
        if doc is not None and tool is not None:
            tool.keywords.append(f"ref:doc:{doc.id}")
            doc.keywords.append(f"ref:tool:{tool.id}")
            tool.keywords_embedding = position_weighted_keyword_embedding(
                tool.keywords[:16], self.encoder)
            doc.keywords_embedding = position_weighted_keyword_embedding(
                doc.keywords[:16], self.encoder)
            tool.parents = [doc.id]
            apply_pull(tool, doc, +1.0, self._now())
        return {
            "pushed": bool(doc or tool),
            "is_project": is_project, "is_tool": is_tool, "name": name,
            "doc_id": doc.id if doc else None,
            "tool_id": tool.id if tool else None,
        }

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
        # event-action beacons : event_action_enrich plants a GENERATOR ACTION
        # that re-anchors the WALK on the schema's findings. Like entity beacons
        # their pull already shifted the real facts ; they are dead weight in a
        # cosine retrieve and would only displace evidence (and aren't answers).
        search_pts = [p for p in search_pts
                      if "event:action" not in (p.tags or ())]
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

    def answer_keepup(
        self,
        query: str,
        *,
        n_stages: int = 16,
        user_id: str = "",
        session_id: str = "",
    ):
        """KEEPUP mode — organic, continuously-rewriting answer generation.

        Yields one snapshot per walk stage (see MetaWalker.keepup) : a
        provisional answer that is re-written as evidence accumulates, the
        live thought, and the depth-gate state. The walk runs in a single
        uncertainty-governed pass ; because a provisional answer is emitted
        from the first walk and refined each stage, the moment the σ /
        coverage depth-gate validates (done=True) the answer is already
        final — there is no separate final-generation wait. Consume it over
        SSE to stream a message that rewrites itself until validated."""
        from metacog.meta_walk import MetaWalker
        section = {f"{k}:{v}" for k, v in
                   (("user", user_id), ("session", session_id)) if v} or None
        walker = MetaWalker(
            query, self, n_stages=n_stages, commit=False,
            section_filter=section,
        )
        yield from walker.keepup()

    @staticmethod
    def _point_date(p: Point) -> Optional[str]:
        """A point's event date (YYYY-MM-DD) from its time:date tag, or None."""
        return next((t.split(":", 2)[2] for t in p.tags
                     if t.startswith("time:date:")), None)

    def filter_list(
        self,
        *,
        tags: Optional[Sequence[str]] = None,
        match: str = "exact",
        event_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        kind: Optional[str] = None,
        valid_only: bool = False,
        as_of: Optional[str] = None,
    ) -> List[Point]:
        """NON-kNN FILTERED LISTING : return ALL points matching the PARAMETERS
        as a plain list — a scan, with no query embedding, no walk, no top-k.

        This is the "refer an exhaustive set" primitive the walk's ACTION
        launches once meta-cognition sees the input relates to an event : a
        targeted, filtered knowledge-base query by `event_id` (the gravitating
        cluster), inclusive `date_from`/`date_to` range (ISO dates, lexical),
        hierarchical `tags`, and `kind`. Ordered by event date then id, so the
        listing is deterministic and exhaustive (every match, not a similarity
        ranking)."""
        from metacog.tags import filter_points
        if tags:
            keep = set(filter_points(self.points, list(tags), mode=match))
            pts = [p for p in self.points if p.id in keep]
        else:
            pts = list(self.points)
        if event_id:
            ev = {p.id for p in self.event_cluster(event_id)}
            pts = [p for p in pts if p.id in ev]
        if kind:
            kn = str(kind).upper()
            pts = [p for p in pts
                   if getattr(p.kind, "name", str(p.kind)).upper() == kn]
        if date_from or date_to:
            ranged = []
            for p in pts:
                d = self._point_date(p)
                if d is None:
                    continue
                if date_from and d < date_from:
                    continue
                if date_to and d > date_to:
                    continue
                ranged.append(p)
            pts = ranged
        if valid_only:
            pts = [p for p in pts if self.is_valid(p, as_of=as_of)]
        pts.sort(key=lambda p: (self._point_date(p) or "9999-99-99", p.id))
        return pts

    def scoped_answer(
        self,
        query: str,
        *,
        tags: Sequence[str],
        knowledge_base: bool = False,
        n_stages: int = 16,
        match: str = "exact",
        list_only: bool = False,
        event_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tag-filtered retrieval as a TWO-PHASE cascade.

        Phase 1 — SCOPED. Run a full uncertainty-governed walk **hard-
        restricted** to the points carrying ALL `tags` (e.g. a discussion /
        `session:<id>`). This establishes *what we are talking about* inside
        that filtered set and produces a scoped answer + its evidence.

        Phase 2 — KNOWLEDGE BASE (only if `knowledge_base=True`). Seed a
        SECOND walk over the WHOLE memory with the original query enriched by
        Phase-1's finding, so the discussion context drives the global
        search. When `knowledge_base=False` the answer stays strictly within
        the filtered set.

        `match` selects how `tags` are resolved on each Point:
          - "exact" (default) — equality OR hierarchical ancestry, so
            filtering on `ref:date` includes points carrying
            `ref:date:2022`. Drop-in for the previous `issubset` behaviour
            with the added ancestor convenience.
          - "fuzzy" — segment-wise Levenshtein (typos / morphological
            variants on tag names).
          - "regex" — `re.search` on each raw tag (case-insensitive),
            e.g. `r"^ref:date:202[0-9]$"`.

        Returns `{"scoped_answer", "scoped_evidence", "knowledge_base",
        ["global_answer", "global_evidence"]}`."""
        from metacog.meta_walk import MetaWalker, provisional_answer
        from metacog.tags import filter_points

        # NON-kNN listing : the walk's ACTION saw the input relates to an event
        # and asks for the exhaustive set by parameters, not a similarity walk.
        if list_only:
            pts = self.filter_list(tags=tags, match=match, event_id=event_id,
                                   date_from=date_from, date_to=date_to)
            return {
                "scoped_list": [{"id": p.id, "content": p.content} for p in pts],
                "scoped_ids": [p.id for p in pts],
                "knowledge_base": bool(knowledge_base), "list_only": True,
            }

        scoped_ids = filter_points(self.points, list(tags or []), mode=match)
        # Targeted filter : narrow the kNN walk's pool by event / date too.
        if event_id or date_from or date_to:
            filt = {p.id for p in self.filter_list(
                event_id=event_id, date_from=date_from, date_to=date_to)}
            scoped_ids = [i for i in (scoped_ids or
                          [p.id for p in self.points]) if i in filt]

        def _run(q, *, restrict):
            w = MetaWalker(q, self, n_stages=n_stages, commit=False,
                           restrict_ids=restrict)
            for _ in range(n_stages):
                if w.step().done:
                    break
            ev = w._composable_evidence()
            chain = [t.content for t in w._thought_chain]
            return provisional_answer(q, ev, chain, self.llm), ev

        ans1, ev1 = _run(query, restrict=scoped_ids)
        out: Dict[str, Any] = {
            "scoped_answer": ans1, "scoped_evidence": ev1,
            "knowledge_base": bool(knowledge_base),
        }
        if not knowledge_base:
            return out
        # Phase-1's finding seeds the global search.
        seed = (f"{query} {ans1}").strip() if ans1 else query
        ans2, ev2 = _run(seed, restrict=None)
        out["global_answer"] = ans2
        out["global_evidence"] = ev2
        return out

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
        # LATENT tag refiner : decompose FACT phrase keywords into
        # hierarchical namespace tags. Opt-in (tags_refine_enabled), idempotent.
        if self.tags_refine_enabled:
            out["tags_refined"] = self.refine_tags()["refined_points"]
        return out

    def refine_tags(self) -> Dict[str, Any]:
        """Decompose FACT phrase keywords into hierarchical namespace tags.

        For every FACT not yet refined, run the LLM taxonomy pass over its
        keyword phrases and APPEND the resulting hierarchical tags (deduped,
        normalised lowercase by Point.__post_init__). A "refined" marker tag
        makes the pass idempotent : a point is visited once across sleeps.

        Tags are INDEXING metadata (Cor. 5 : provenance GENERATOR) — appended
        directly, never via an Observation. Fully failure-safe : a flaky LLM
        leaves the cloud unchanged. Returns the count of points refined and
        the union of new tags added."""
        from metacog.tag_refine import refine_tags as _refine
        from metacog.epistemic import PointKind

        refined_points = 0
        added: List[str] = []
        for p in self.points:
            if p.kind != PointKind.FACT or "refined" in p.tags:
                continue
            if p.id.startswith("entity_"):      # entity beacons are already typed
                continue
            try:
                new_tags = _refine(list(p.keywords or []), self.llm)
            except Exception:
                new_tags = []
            existing = set(p.tags)
            fresh = [t for t in new_tags if t not in existing]
            if fresh:
                p.tags.extend(fresh)
                for t in fresh:
                    if t not in added:
                        added.append(t)
            # Mark visited regardless (so a phrase the LLM declined is not
            # retried every sleep) — but only once we actually reached the LLM
            # path ; a missing LLM leaves the point unmarked for a later run.
            if hasattr(self.llm, "generate"):
                if "refined" not in p.tags:
                    p.tags.append("refined")
                refined_points += 1
        return {"refined_points": refined_points, "tags_added": added}

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
        # Rebuild the EVENT-hub registry from the restored points (it is not
        # serialised — the points carry the canonical event:type tags).
        self._event_registry = {}
        for p in self.points:
            if p.kind is PointKind.EVENT:
                et = next((t.split(":", 2)[2] for t in p.tags
                           if t.startswith("event:type:")), "")
                nm = p.content.split(" ", 1)[1] if " " in p.content else ""
                self._event_registry[f"{et}::{nm.lower()}"] = p.id
