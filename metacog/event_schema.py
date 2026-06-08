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
from typing import Any, List, Tuple

_SCHEMA_CACHE: dict = {}


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
