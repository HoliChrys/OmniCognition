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


def _retrieve_scoped(memory: Any, q: str, k: int, scope: Any) -> List[dict]:
    orig = memory.points
    try:
        if scope is not None:
            memory.points = scope
        hits = memory.retrieve(q, k=max(1, k))
    except Exception:
        hits = []
    finally:
        memory.points = orig
    return [{"id": h["id"], "content": h["content"], "score": h["score"]}
            for h in hits]


def fill_event_schema(memory: Any, event_name: str, etype: str, *,
                      k_per_slot: int = 3, gap_fill: bool = True,
                      restrict_ids: Any = None) -> dict:
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
           "core": list(schema.core), "cluster_size": 0,
           "filled": {}, "gaps": [], "fact_ids": []}
    if schema.is_empty():
        return out

    # the gravitating cluster = "everything there is" about the event
    cluster = None
    if restrict_ids is not None:
        cluster = [p for p in memory.points if p.id in set(restrict_ids)]
    elif hasattr(memory, "_event_registry") and hasattr(memory, "event_cluster"):
        hub_id = memory._event_registry.get(
            f"{schema.etype}::{(event_name or '').strip().lower()}")
        if hub_id:
            cluster = memory.event_cluster(hub_id)
    out["cluster_size"] = len(cluster) if cluster is not None else 0

    ids: List[str] = []
    core = set(schema.core)
    for slot, q in slot_subquestions(event_name, schema):
        # (1) partition the gravitating cluster
        hits = _retrieve_scoped(memory, q, k_per_slot, cluster)
        # (2) a CORE slot empty in the cluster is a GAP -> additive corpus search
        if not hits and slot in core and gap_fill:
            hits = _retrieve_scoped(memory, q, k_per_slot, None)
            if hits:
                out["gaps"].append(slot)
        elif not hits and slot in core:
            out["gaps"].append(slot)
        out["filled"][slot] = hits
        for h in hits:
            if h["id"] not in ids:
                ids.append(h["id"])
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


def event_search(memory: Any, query: str, *, k_per_slot: int = 3,
                 enrich: bool = True) -> Optional[dict]:
    """End-to-end event-schema retrieval for a QUERY : detect the event type,
    fill the schema by partitioning the gravitating cluster (+ gap-fill core
    slots), enrich the schema from leftover cluster facts (A.2), and record
    cross-instance stats (A.3). Returns the fill result (+ event/new_slots) or
    None when the query is not about an event (caller falls back to the walk)."""
    det = memory.detect_event_type(query) if hasattr(
        memory, "detect_event_type") else None
    if not det:
        return None
    fill = fill_event_schema(memory, det["name"], det["etype"],
                             k_per_slot=k_per_slot)
    fill["event"] = det
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
