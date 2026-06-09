"""
Event-type SCHEMA induction + schema-driven sub-questions.

An event TYPE (e.g. "war") defines a RECURRENT typology of roles/slots and
sub-events — war ⇒ belligerents, territory/lands, locations (fronts, camps),
casus belli, timeline, casualties, treaties. That schema is the inventory an
instance of the type should answer (FrameNet core/non-core frame elements; ACE/
MAVEN argument roles; LLM event-schema induction — Li et al., ACL 2023).

Two functions, both LLM-backed, cached, fully failure-safe (mirroring
metacog.clues / metacog.tag_refine):

  • `induce_event_schema(etype, llm)` — the recurrent slots of the type, each
    marked CORE (essential to the type) or peripheral. Induced once per type
    and cached.
  • `slot_subquestions(event_name, schema)` — one concrete sub-question per
    slot, so retrieval can fill the schema by gathering the facts that
    gravitate to the event (schema-driven query decomposition; the fixed slot
    count calibrates the sub-query budget vs free decomposition that
    over-generates — Question Decomposition for RAG).

NEVER cache an empty result (one 529 must not disable schema induction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

_SCHEMA_CACHE: dict = {}
# Cross-instance slot statistics per type : etype -> {slot: [filled, seen]}.
# Reinforces the recurrent slots (A.3) : slots filled across many instances of a
# type are promoted to core ; slots never filled are dropped.
_SCHEMA_STATS: dict = {}


@dataclass
class EventSchema:
    """The recurrent slots of an event type. `core` slots are essential to the
    type (required frame elements); the rest are peripheral descriptors."""
    etype: str = ""
    slots: List[str] = field(default_factory=list)      # ordered, core first
    core: List[str] = field(default_factory=list)       # subset of `slots`

    def is_empty(self) -> bool:
        return not self.slots


_PROMPT = (
    "An EVENT of type \"{etype}\" recurrently involves the same kinds of "
    "participants, places, sub-events and attributes. List that RECURRENT "
    "schema — the slots a typical instance of a {etype} should let you answer. "
    "Give SHORT slot names (1-3 words, lowercase). Mark each CORE (essential to "
    "the very notion of a {etype}) or PERIPHERAL. For example a 'war' ⇒ "
    "belligerents (core), territory (core), fronts (core), casus belli (core), "
    "timeline (core), casualties (peripheral), treaties (peripheral). "
    "Output one slot per line as `slot | core` or `slot | peripheral`, at most "
    "10 slots, no prose, no numbering."
)


def induce_event_schema(etype: str, llm: Any) -> EventSchema:
    """Return the recurrent slot schema for `etype` (cached, failure-safe)."""
    et = (etype or "").strip().lower()
    if not et:
        return EventSchema()
    if et in _SCHEMA_CACHE:
        s = _SCHEMA_CACHE[et]
        return EventSchema(etype=et, slots=list(s.slots), core=list(s.core))
    if not hasattr(llm, "generate"):
        return EventSchema(etype=et)
    try:
        raw = (llm.generate(_PROMPT.format(etype=et), max_tokens=200) or "").strip()
    except Exception:
        return EventSchema(etype=et)
    slots: List[str] = []
    core: List[str] = []
    for ln in raw.splitlines():
        s = ln.strip().lstrip("-*0123456789.) \t").strip()
        if not s or "|" not in s:
            continue
        name, _, flag = s.partition("|")
        name = name.strip().lower().strip(".,:;")
        if not name or name in slots:
            continue
        slots.append(name)
        if "core" in flag.strip().lower():
            core.append(name)
    slots = slots[:10]
    core = [c for c in core if c in slots]
    out = EventSchema(etype=et, slots=slots, core=core)
    if slots:                               # never cache an empty result
        _SCHEMA_CACHE[et] = EventSchema(etype=et, slots=list(slots),
                                        core=list(core))
    return out


def slot_subquestions(event_name: str, schema: EventSchema
                      ) -> List[Tuple[str, str]]:
    """One concrete sub-question per slot of the schema, for an instance named
    `event_name`. Returns [(slot, question)] — the fixed-budget decomposition
    that drives slot-filling retrieval. Deterministic (no LLM needed)."""
    name = (event_name or "the event").strip()
    out: List[Tuple[str, str]] = []
    for slot in schema.slots:
        out.append((slot, f"What is the {slot} of {name}?"))
    return out


def _scoped_slot(memory: Any, q: str, scope_tag: str, *, n_stages: int = 4
                 ) -> Tuple[List[dict], List[dict]]:
    """A NECESSARY slot's filtered scoped search, with `knowledge_base=True`.

    Phase 1 is HARD-FILTERED to the context (`scope_tag` = event:in:<hub>) ;
    Phase 2 (knowledge_base) runs a SECOND walk over the WHOLE knowledge base,
    seeded by the Phase-1 finding — 'after the first search the result can join
    the KB'. Returns (scoped_hits, kb_hits) as id/content/score dicts."""
    try:
        res = memory.scoped_answer(q, tags=[scope_tag], knowledge_base=True,
                                   n_stages=n_stages)
    except Exception:
        return [], []

    def _ev(key: str) -> List[dict]:
        return [{"id": e["id"], "content": e.get("content", ""), "score": 1.0}
                for e in (res.get(key) or []) if isinstance(e, dict)
                and e.get("id")]
    return _ev("scoped_evidence"), _ev("global_evidence")


def _retrieve_scoped(memory: Any, q: str, k: int, scope: Any, *,
                     exclude: Any = None) -> List[dict]:
    exclude = exclude or set()
    orig = memory.points
    try:
        if scope is not None:
            memory.points = scope
        # over-fetch so excluding ids already taken by earlier slots still
        # leaves k fresh ones (slots diversify coverage, not repeat the top).
        hits = memory.retrieve(q, k=max(1, k + len(exclude)))
    except Exception:
        hits = []
    finally:
        memory.points = orig
    out = [{"id": h["id"], "content": h["content"], "score": h["score"]}
           for h in hits if h["id"] not in exclude]
    return out[:k]


def fill_event_schema(memory: Any, event_name: str, etype: str, *,
                      k_per_slot: int = 5, gap_fill: bool = True,
                      restrict_ids: Any = None, scope_tag: Any = None) -> dict:
    """Schema-driven slot-filling for one event instance — the two design loops.

    USAGE (open the questions to find everything) : induce the type schema, then
    PARTITION the event's gravitating CLUSTER (the facts pulled onto its hub)
    into roles — one sub-question PER SLOT, scoped to the cluster. This is the
    exhaustive, fixed-budget alternative to stochastic answer-space expansion.

    GAP detection (the "open" question) : a CORE slot with no fact in the
    cluster is a GAP — `gap_fill` re-asks it against the FULL corpus to fetch
    the missing piece (additive, only for empty core slots, per the Graphiti
    'scope must be additive/gated' lesson).

    Returns `{etype, slots, core, cluster_size, filled:{slot:[hits]},
    gaps:[core-slot…], fact_ids}`. `restrict_ids` overrides the cluster scope
    (e.g. the event interval). Failure-safe : empty schema -> empty fill."""
    schema = induce_event_schema(etype, getattr(memory, "llm", None))
    out = {"etype": schema.etype, "slots": list(schema.slots),
           "core": list(schema.core), "cluster_size": 0, "cluster_ids": [],
           "filled": {}, "gaps": [], "fact_ids": []}
    if schema.is_empty():
        return out

    # the gravitating cluster = "everything there is" about the event = the BAG
    # the enumeration/retrieve mode returns wholesale.
    cluster = None
    if restrict_ids is not None:
        cluster = [p for p in memory.points if p.id in set(restrict_ids)]
    elif hasattr(memory, "_event_registry") and hasattr(memory, "event_cluster"):
        hub_id = memory._event_registry.get(
            f"{schema.etype}::{(event_name or '').strip().lower()}")
        if hub_id:
            cluster = memory.event_cluster(hub_id)
    out["cluster_size"] = len(cluster) if cluster is not None else 0
    out["cluster_ids"] = [p.id for p in cluster] if cluster is not None else []

    ids: List[str] = []
    core = set(schema.core)
    # A NECESSARY (core) slot is answered by the FILTERED scoped search with
    # knowledge_base=True (when a scope tag + scoped_answer are available) :
    # Phase 1 hard-filtered to the context, Phase 2 over the whole KB. Phase-2
    # finds are absorbed back INTO the context (tag event:in + pull), so they
    # join the knowledge base for later slots / future queries.
    use_scoped = (scope_tag is not None and hasattr(memory, "scoped_answer"))
    hub_id = (scope_tag.split("event:in:", 1)[1]
              if isinstance(scope_tag, str)
              and scope_tag.startswith("event:in:") else None)

    def _absorb(hits):
        if not (hub_id and hits and hasattr(memory, "event_absorb")):
            return
        by_id = {p.id: p for p in memory.points}
        memory.event_absorb(
            hub_id, [by_id[h["id"]] for h in hits if h["id"] in by_id])

    # ids already assigned to earlier slots — EXCLUDED from later slots so the
    # schema DIVERSIFIES coverage instead of every slot repeating the same top
    # facts (the redundancy that capped slot-fill recall below the cluster).
    taken: set = set()
    for slot, q in slot_subquestions(event_name, schema):
        if use_scoped and slot in core:
            scoped_hits, kb_hits = _scoped_slot(memory, q, scope_tag)
            _absorb(kb_hits)                       # KB finds join the context
            hits = [h for h in scoped_hits
                    if h["id"] not in taken][:k_per_slot]
            if not hits:                           # GAP -> filled by the KB phase
                hits = [h for h in kb_hits
                        if h["id"] not in taken][:k_per_slot]
                out["gaps"].append(slot)
        else:
            # (1) partition the gravitating cluster (excluding taken ids)
            hits = _retrieve_scoped(memory, q, k_per_slot, cluster,
                                    exclude=taken)
            # (2) a CORE slot empty in the cluster is a GAP -> corpus search
            if not hits and slot in core and gap_fill:
                hits = _retrieve_scoped(memory, q, k_per_slot, None,
                                        exclude=taken)
                if hits:
                    out["gaps"].append(slot)
            elif not hits and slot in core:
                out["gaps"].append(slot)
        out["filled"][slot] = hits
        for h in hits:
            if h["id"] not in ids:
                ids.append(h["id"])
                taken.add(h["id"])
    out["fact_ids"] = ids
    return out


def enrich_schema(memory: Any, etype: str, event_name: str, fill: dict
                  ) -> List[str]:
    """A.2 — instance-overlay enrichment. Cluster facts that filled NO slot are
    candidate NEW slots : ask the LLM what recurrent role of an `etype` they
    describe and ADD it to the type schema (cached), so future fills cover it.
    Returns the new slot names added. Failure-safe."""
    llm = getattr(memory, "llm", None)
    if not hasattr(llm, "generate"):
        return []
    eid = (memory._event_registry.get(f"{etype}::{(event_name or '').lower()}")
           if hasattr(memory, "_event_registry") else None)
    if not eid or not hasattr(memory, "event_cluster"):
        return []
    covered = set(fill.get("fact_ids") or [])
    leftover = [p for p in memory.event_cluster(eid) if p.id not in covered]
    if not leftover:
        return []
    sample = " / ".join((p.content or "")[:80] for p in leftover[:5])
    existing = ", ".join(fill.get("slots") or [])
    try:
        raw = (llm.generate(
            f"For an event of type \"{etype}\", these facts did NOT fit any of "
            f"the known slots [{existing}]:\n{sample}\nName up to 2 NEW recurrent "
            "slots (1-3 lowercase words each) they fill, comma-separated, or "
            "'none'.", max_tokens=30) or "").strip().lower()
    except Exception:
        return []
    if "none" in raw:
        return []
    new = [s.strip() for s in raw.replace("\n", ",").split(",")
           if s.strip() and s.strip() not in (fill.get("slots") or [])][:2]
    if new and etype in _SCHEMA_CACHE:
        s = _SCHEMA_CACHE[etype]
        s.slots = list(s.slots) + [n for n in new if n not in s.slots]
    return new


def record_fill(etype: str, schema: EventSchema, fill: dict) -> None:
    """A.3 — accumulate cross-instance slot statistics (which slots get filled
    across instances of a type)."""
    et = (etype or "").lower()
    if not et or schema.is_empty():
        return
    stats = _SCHEMA_STATS.setdefault(et, {})
    filled_slots = {s for s, hits in (fill.get("filled") or {}).items() if hits}
    for slot in schema.slots:
        cell = stats.setdefault(slot, [0, 0])
        cell[1] += 1                         # seen
        if slot in filled_slots:
            cell[0] += 1                     # filled


def reinforce_schema(etype: str, *, min_instances: int = 3,
                     core_frac: float = 0.6) -> EventSchema:
    """A.3 — refine a type's cached schema from accumulated statistics : promote
    slots filled in ≥ `core_frac` of instances to CORE, drop slots never filled.
    No-op until `min_instances` observed. Returns the (possibly updated) schema."""
    et = (etype or "").lower()
    stats = _SCHEMA_STATS.get(et)
    sc = _SCHEMA_CACHE.get(et)
    if not stats or sc is None:
        return sc or EventSchema(etype=et)
    seen_max = max((v[1] for v in stats.values()), default=0)
    if seen_max < min_instances:
        return sc
    kept, core = [], []
    for slot in sc.slots:
        filled, seen = stats.get(slot, [0, 0])
        if seen and filled == 0:
            continue                          # never filled -> drop
        kept.append(slot)
        if seen and filled / seen >= core_frac:
            core.append(slot)
    sc.slots, sc.core = kept, core
    return sc


def event_anchor(memory: Any, query: str):
    """Build a query anchor ENRICHED with the detected event's schema elements
    — the convergence of the event subsystem and the drift-resistance anchor.

    When the query is about an event, the anchor's RELATIVE half is seeded with
    the event vocabulary (event name tokens + schema slot names + filled slot
    leaf values) so retrieval/the walk is pulled toward ALL facets of the event
    — recovering oblique facts that share the event's vocabulary but NONE of the
    query's surface words ("bomb Iran during peace negotiations" -> belligerents
    + timeline). Returns (anchor, detection) or (None, None) when not an event."""
    from metacog.query_anchor import build_query_anchor
    det = memory.detect_event_type(query) if hasattr(
        memory, "detect_event_type") else None
    if not det:
        return None, None
    schema = induce_event_schema(det["etype"], getattr(memory, "llm", None))
    fill = fill_event_schema(memory, det["name"], det["etype"], k_per_slot=2)
    terms: List[str] = [w for w in det["name"].split() if len(w) >= 3]
    terms += [s for s in schema.slots]
    for hits in fill.get("filled", {}).values():       # filled slot evidence
        for h in hits[:1]:
            terms += [w for w in (h.get("content") or "").split()[:4]
                      if len(w) >= 4]
    try:
        anchor = build_query_anchor(
            query, encoder=getattr(memory, "encoder", None),
            corpus_texts=[p.content for p in memory.points])
        anchor.with_relative(list(dict.fromkeys(terms))[:16],
                             getattr(memory, "encoder", None))
    except Exception:
        return None, det
    return anchor, det


def event_action_enrich(memory: Any, event_id: str, fill: dict
                        ) -> Optional[str]:
    """Reuse the META-COGNITION ACTION system to fold the schema result back
    into the cluster — and, by the same node, bridge it to the normal search.

    The schema's filled facts seed `generate_action` : a GENERATOR-sourced
    ACTION node (Cor.5 safe) whose `parents` are those facts (lineage) and whose
    keywords are anchored on their entities (no drift). It is co-located with
    the event hub (`apply_pull`) and pulls the schema facts toward itself, so —
    because the walk RE-ANCHORS on the nearest ACTION and SPREADS from it — the
    action becomes a beacon that drags the schema findings into co-retrieval
    whenever the context is touched (geometric, edge-free enrichment, no bespoke
    plumbing). Living in the shared manifold, the SAME action is also a global
    ACTION node the NORMAL walk picks up via nearest_by_kind — so the normal
    channel 'regresses' back onto what the event channel found. Returns the
    action id, or None when there is nothing to fold (an empty step has no
    influence)."""
    from metacog.meta_walk import generate_action
    from metacog.geometry import apply_pull
    by_id = {p.id: p for p in memory.points}
    facts = [by_id[i] for i in (fill.get("fact_ids") or []) if i in by_id]
    if not facts or not hasattr(memory, "_now"):
        return None
    t_now = memory._now()
    act = generate_action(facts, getattr(memory, "llm", None),
                          getattr(memory, "encoder", None),
                          getattr(memory, "extractor", None), t_now)
    if act is None:
        return None
    act.tags = list(act.tags) + [f"event:in:{event_id}", "event:action"]
    memory.points.append(act)
    hub = by_id.get(event_id)
    if hub is not None:
        apply_pull(hub, act, +1.0, t_now)           # the action joins the cluster
    for f in facts:                                  # re-anchor facts on the find
        apply_pull(act, f, +1.0, t_now)
    return act.id


def event_search(memory: Any, query: str, *, k_per_slot: int = 5,
                 enrich: bool = True, action_bridge: bool = True
                 ) -> Optional[dict]:
    """End-to-end event-schema retrieval for a QUERY : detect the event type,
    fill the schema by partitioning the gravitating cluster (+ gap-fill core
    slots), enrich the schema from leftover cluster facts (A.2), and record
    cross-instance stats (A.3). Returns the fill result (+ event/new_slots) or
    None when the query is not about an event (caller falls back to the walk)."""
    det = memory.detect_event_type(query) if hasattr(
        memory, "detect_event_type") else None
    if not det:
        return None
    # CONTEXT channel : the query is situated in a CONTEXT (= the event:type) ;
    # its knowledge base is the UNION of facts gravitating to EVERY event of the
    # type (broadest latent evidence). The schema's NECESSARY sub-questions are
    # then answered by a FILTERED retrieval SCOPED to that context KB — not the
    # whole corpus.
    context_ids: List[str] = []
    if hasattr(memory, "context_members"):
        try:
            context_ids = [p.id for p in memory.context_members(det["etype"])]
        except Exception:
            context_ids = []
    # the context filter for the scoped NECESSARY-slot search : the macro hub's
    # event:in tag (post-consolidation it covers the whole context). knowledge_
    # base=True (inside fill) then expands beyond it.
    hub_id = det.get("event_id") or (
        memory._event_registry.get(
            f"{det['etype']}::{(det['name'] or '').strip().lower()}")
        if hasattr(memory, "_event_registry") else None)
    scope_tag = f"event:in:{hub_id}" if hub_id else None
    fill = fill_event_schema(memory, det["name"], det["etype"],
                             k_per_slot=k_per_slot,
                             restrict_ids=context_ids or None,
                             scope_tag=scope_tag)
    fill["event"] = det
    fill["context_ids"] = context_ids

    # REUSE the schema result via a meta-cognitive ACTION : fold it into the
    # cluster AND bridge it to the normal search (one beacon, both directions).
    if action_bridge and hub_id:
        try:
            fill["action_id"] = event_action_enrich(memory, hub_id, fill)
        except Exception:
            fill["action_id"] = None

    # COLLECT each NECESSARY (core) slot's clue into the BAG : the schema's list
    # of required roles IS a retrieve-bag. Parallel — it only feeds the final
    # list rendering, the rest of the agent process is unchanged.
    if hasattr(memory, "bag_add"):
        core = set(fill.get("core") or [])
        need = [h["id"] for slot, hits in (fill.get("filled") or {}).items()
                if slot in core for h in hits]
        if need:
            memory.bag_add(need)

    # ENUMERATION / retrieve mode : the context IS the bag — return it as a list.
    from metacog.enumeration import is_enumeration_query, format_bag_answer
    if is_enumeration_query(query) and fill.get("cluster_ids"):
        if hasattr(memory, "bag_add"):
            memory.bag_add(fill["cluster_ids"])
        by_id = {p.id: p for p in memory.points}
        bag = [(cid, getattr(by_id.get(cid), "content", ""))
               for cid in fill["cluster_ids"]]
        fill["bag_answer"] = format_bag_answer(query, bag,
                                               getattr(memory, "llm", None))
    if enrich:
        try:
            fill["new_slots"] = enrich_schema(
                memory, det["etype"], det["name"], fill)
        except Exception:
            fill["new_slots"] = []
        record_fill(det["etype"],
                    induce_event_schema(det["etype"], getattr(memory, "llm", None)),
                    fill)
    return fill
