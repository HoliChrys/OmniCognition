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


def fill_event_schema(memory: Any, event_name: str, etype: str, *,
                      k_per_slot: int = 3, restrict_ids: Any = None
                      ) -> dict:
    """Schema-driven slot-filling retrieval for one event instance.

    Induce the type's schema, decompose into one sub-question PER SLOT, retrieve
    each (optionally scoped to `restrict_ids` — e.g. the event's interval), and
    aggregate the gathered facts. This is the EXHAUSTIVE, fixed-budget
    alternative to stochastic answer-space expansion : every recurrent slot of
    the type is queried, so the latent sub-questions of the event are covered by
    construction. Returns `{etype, slots, core, filled:{slot:[hits]}, fact_ids}`.
    Failure-safe : an empty schema yields an empty fill."""
    schema = induce_event_schema(etype, getattr(memory, "llm", None))
    out = {"etype": schema.etype, "slots": list(schema.slots),
           "core": list(schema.core), "filled": {}, "fact_ids": []}
    if schema.is_empty():
        return out
    scope = None
    if restrict_ids is not None:
        scope = [p for p in memory.points if p.id in set(restrict_ids)]
    ids: List[str] = []
    orig = memory.points
    try:
        if scope is not None:
            memory.points = scope
        for slot, q in slot_subquestions(event_name, schema):
            try:
                hits = memory.retrieve(q, k=max(1, k_per_slot))
            except Exception:
                hits = []
            trimmed = [{"id": h["id"], "content": h["content"],
                        "score": h["score"]} for h in hits]
            out["filled"][slot] = trimmed
            for h in trimmed:
                if h["id"] not in ids:
                    ids.append(h["id"])
    finally:
        memory.points = orig
    out["fact_ids"] = ids
    return out
