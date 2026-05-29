"""
Meta-cognitive walk — coordinated ACTION / FACT / THOUGHT retrieval.

Replaces the externally-driven ReAct agent with a walk that lives
INSIDE the memory : at each stage we retrieve the nearest FACTs and
ACTIONs to a query, generate any kind that is missing (kept under a
strict 50-token budget — Cor. 5 : these generations enter as Points
with `source=GENERATOR`, never as observations), then synthesize a
THOUGHT that fuses the keyword sets of the most-certain fact and the
chosen action. The THOUGHT's enriched keywords drive the next stage.

Algorithm — three coordinated kinds traversing the manifold :

    stage 0
      facts_0    = kNN(query, kind=FACT)            # parallel
      actions_0  = kNN(query, kind=ACTION)
      if no ACTION exists yet : Generate(facts_0)   # kind=ACTION,
                                                    # source=GENERATOR
      pick fact* = argmin σ over facts_0            # uncertainty filter

      thought_0  = MetaThought(fact*.keywords
                               ∪ actions_0[0].keywords)   # kind=THOUGHT
                                                    # source=GENERATOR
                                                    # ≤ 50 tokens
    stage k → k+1
      kws_new        = thought_k.keywords
      facts_{k+1}    = kNN(kws_new, kind=FACT, on keyword embeddings)
      actions_{k+1}  = nearest ACTION to actions_k[0] OR Generate
      pick fact*     = argmin σ_propagated over facts_{k+1}
      thought_{k+1}  = MetaThought(fact*.keywords ∪ action.keywords)

The walk feeds back into all the existing mechanisms :
  - apply_pull fires on co-activated (fact, action) pairs, dragging
    them together in the manifold (Hebbian-style consolidation).
  - generated THOUGHTs become candidates for COLLISION fission on the
    next sleep cycle if the system re-derives the same pensée later.
  - the walk can be routed through an OBSERVATOR id, so different
    perspectives produce different walks over the same cloud.

Returns a `MetaWalkTrajectory` carrying the visited fact ids (the
cumulative effective retrieval the answerer sees) and the generated
THOUGHT/ACTION Points (which were appended to the memory).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from metacog.epistemic import (
    DEFAULT_OBSERVATOR_ID,
    EpistemicState,
    Point,
    PointKind,
    SourceClass,
)
from metacog.geometry import (
    apply_pull,
    cosine,
    effective_keyword_embedding,
    geometric_spread,
)
from metacog.keywords import position_weighted_keyword_embedding
from metacog.uncertainty import beta_sigma, node_sigma


# Hard caps per generation step. Not tuning knobs : token budget per
# generation (50) is the contract stated by the design ; n_stages=3
# keeps the worst-case generation cost at 3·2·50 = 300 output tokens.
_GENERATION_TOKEN_BUDGET = 50
_DEFAULT_STAGES = 3
_FACTS_PER_STAGE = 5
_ACTIONS_PER_STAGE = 3


@dataclass
class StageRecord:
    """One stage of the walk."""

    stage: int
    fact_ids: List[str] = field(default_factory=list)
    action_ids: List[str] = field(default_factory=list)
    chosen_fact_id: Optional[str] = None
    chosen_action_id: Optional[str] = None
    thought_id: Optional[str] = None
    generated_action: bool = False
    generated_thought: bool = False


@dataclass
class MetaWalkTrajectory:
    """Full trace of a meta-cognitive walk."""

    query: str
    stages: List[StageRecord] = field(default_factory=list)
    # Cumulative ids of every FACT seen during the walk — this is what
    # the answerer (or the F1 scorer downstream) treats as "evidence".
    fact_ids_cumulative: List[str] = field(default_factory=list)
    generated_point_ids: List[str] = field(default_factory=list)


def _node_payload(p: Point) -> dict:
    """Compact JSON-ready dump of a node for an external (MCP) caller.

    Always carries the INDEXED CONTENT so the agent driving the walk
    over MCP can reason on the actual text, not just ids — even though
    retrieval itself runs on keyword kNN.
    """
    return {
        "id": p.id,
        "kind": p.kind.value,
        "content": p.content,
        "keywords": list(p.keywords),
        "confidence": round(p.confidence, 3),
        "uncertainty": round(beta_sigma(p), 3),
        "n_corrob": p.n_corrob,
        "n_contra": p.n_contra,
        "n_uses": p.n_uses,
        "state": p.state.value,
        "source": p.keywords_source.value if p.keywords_source else None,
    }


@dataclass
class StageOutput:
    """One stage's result as returned to an MCP caller.

    Carries full content of every retrieved node (not just ids) plus a
    'done' flag that lets the agent decide whether to call walk_next
    or move on to synthesis.
    """

    stage: int
    facts: List[dict]
    actions: List[dict]
    chosen_fact: Optional[dict]
    chosen_action: Optional[dict]
    thought: Optional[dict]
    generated_action: bool
    generated_thought: bool
    fact_ids_cumulative: List[str]
    done: bool

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "facts": self.facts,
            "actions": self.actions,
            "chosen_fact": self.chosen_fact,
            "chosen_action": self.chosen_action,
            "thought": self.thought,
            "generated_action": self.generated_action,
            "generated_thought": self.generated_thought,
            "fact_ids_cumulative": list(self.fact_ids_cumulative),
            "done": self.done,
        }


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def nearest_by_kind(
    query_emb: Tuple[float, ...],
    points: Sequence[Point],
    kind: PointKind,
    k: int,
    t_now: float,
    *,
    exclude_ids: Optional[set] = None,
) -> List[Tuple[float, Point]]:
    """Top-k points of a given kind by cosine on the keyword embedding."""
    exclude_ids = exclude_ids or set()
    scored: List[Tuple[float, Point]] = []
    for p in points:
        if p.kind != kind or p.id in exclude_ids:
            continue
        eff = effective_keyword_embedding(p, t_now)
        scored.append((cosine(query_emb, eff), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def nearest_facts_with_fallback(
    query_emb: Tuple[float, ...],
    query_text: str,
    points: Sequence[Point],
    k: int,
    t_now: float,
    *,
    exclude_ids: Optional[set] = None,
    encoder: Any = None,
    extractor: Any = None,
) -> List[Point]:
    """FACT retrieval — delegates to the FULL hybrid pipeline.

    Earlier this hand-rolled keyword-cosine + BM25, which proved much
    weaker than `retrieve_hybrid` (it lacked content-cosine, fuzzy
    Levenshtein, geometric spreading and lineage). We now route FACT
    retrieval through `retrieve_hybrid(restrict_kind=FACT, …)` so the
    walk benefits from every retrieval signal the rest of the system
    has. The keyword `query_emb` is unused here (the hybrid pipeline
    re-derives its own signals from `query_text`) but kept in the
    signature for callers that still pass it.

    When `encoder`/`extractor` are not supplied we fall back to the old
    keyword-cosine + BM25 path (used by unit tests that call this
    helper directly without a full Memory).
    """
    exclude_ids = exclude_ids or set()

    if encoder is not None and extractor is not None:
        from metacog.geometry import retrieve_hybrid
        results = retrieve_hybrid(
            query_text, points, k + len(exclude_ids), t_now,
            encoder=encoder, extractor=extractor,
            use_lineage=True, use_spreading=True, use_fuzzy=True,
            restrict_kind=PointKind.FACT,
        )
        out: List[Point] = []
        for _s, p in results:
            if p.id in exclude_ids:
                continue
            out.append(p)
            if len(out) >= k:
                break
        return out

    # --- Fallback path (no encoder/extractor : direct callers/tests) ---
    fact_pts = [p for p in points
                if p.kind == PointKind.FACT and p.id not in exclude_ids]
    knn = nearest_by_kind(
        query_emb, fact_pts, PointKind.FACT, k, t_now,
        exclude_ids=exclude_ids,
    )
    relevant = [p for s, p in knn if s > 0.0]
    seen: set = {p.id for p in relevant}
    if len(relevant) >= k:
        return relevant[:k]
    from metacog.bm25 import bm25_score
    fallback = bm25_score(query_text, fact_pts, k_pool=k * 2)
    for _s, p in fallback:
        if p.id in seen:
            continue
        relevant.append(p)
        seen.add(p.id)
        if len(relevant) >= k:
            break
    return relevant[:k]


def least_uncertain(facts: Sequence[Point]) -> Optional[Point]:
    """Return the most epistemically certain fact, using the COMBINED
    σ : β-uncertainty AND keyword-order uncertainty (node_sigma).

    A fact with strong counters but weak/few keywords is correctly
    penalized — the keyword list is its projection onto the entity
    manifold ; a weak projection means a weak retrieval handle, so it
    should not win the "least uncertain" vote at this stage.
    """
    if not facts:
        return None
    return min(facts, key=node_sigma)


def nearest_action_to(
    anchor: Point,
    points: Sequence[Point],
    t_now: float,
    *,
    exclude_ids: Optional[set] = None,
) -> Optional[Point]:
    """Nearest ACTION to a given anchor (typically the previous stage's
    chosen action) by keyword-embedding cosine."""
    exclude_ids = exclude_ids or {anchor.id}
    eff_a = effective_keyword_embedding(anchor, t_now)
    best: Optional[Tuple[float, Point]] = None
    for p in points:
        if p.kind != PointKind.ACTION or p.id in exclude_ids:
            continue
        eff_p = effective_keyword_embedding(p, t_now)
        s = cosine(eff_a, eff_p)
        if best is None or s > best[0]:
            best = (s, p)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Metacognition (the edge-free analog of HeLa's Reflective Agent /
# Hebbian Distillation : read the integer counters and the manifold
# density around the point. A(·) ⊥ P holds — we consume only counters
# and vector distances, never P-content of other points.)
# ---------------------------------------------------------------------------


def _meta_state(point: Point, population: Sequence[Point]) -> dict:
    """Read the point's metacognitive state.

    Returns a small dict with the four quantities a THOUGHT needs to
    reason ABOUT the chosen fact (not just over its content) :

      confidence   β-mean (n_corrob, n_contra)
      uncertainty  β-σ (epistemic spread)
      n_uses       how often this point has been recruited
      is_hub       emergent : n_uses above (median + σ) of active
                   population — marks a Hebbian consolidation candidate

    No hyperparameter : the hub threshold is the same emergent shape
    we use everywhere (median + σ over active counters).
    """
    import math
    sigma = beta_sigma(point)
    uses = [
        q.n_uses for q in population
        if q.state not in {EpistemicState.DEPRECATED, EpistemicState.INVALID}
    ]
    is_hub = False
    if len(uses) >= 4:
        sorted_uses = sorted(uses)
        median = sorted_uses[len(sorted_uses) // 2]
        mean = sum(uses) / len(uses)
        var = sum((u - mean) ** 2 for u in uses) / len(uses)
        std = math.sqrt(var)
        is_hub = point.n_uses > median + std
    return {
        "confidence": round(point.confidence, 3),
        "uncertainty": round(sigma, 3),
        "n_uses": point.n_uses,
        "n_corrob": point.n_corrob,
        "n_contra": point.n_contra,
        "is_hub": is_hub,
    }


def _reflective_context(
    hub: Point,
    population: Sequence[Point],
    t_now: float,
    *,
    max_neighbours: int = 3,
) -> List[str]:
    """Edge-free analog of HeLa's reflective distillation.

    When a fact is a HUB, gather its closest manifold neighbours and
    return their content as a condensed cluster — to be folded into the
    THOUGHT's generation prompt so the resulting reflection consolidates
    the cluster, not just the single point.

    Neighbour membership reuses geometric_spread's emergent (median−σ)
    keyword-distance cutoff — the same threshold the collision machine
    uses, so no new hyperparameter is introduced.
    """
    spread = geometric_spread([hub], population, t_now)
    return [p.content for _d, p in spread[:max_neighbours]]


# ---------------------------------------------------------------------------
# Generation (Cor. 5 : source=GENERATOR, kind=ACTION or THOUGHT)
# ---------------------------------------------------------------------------


def _gen_id(prefix: str, t_now: float) -> str:
    return f"{prefix}@{t_now:.3f}"


def generate_action(
    facts: Sequence[Point],
    llm: Any,
    encoder: Any,
    extractor: Any,
    t_now: float,
) -> Optional[Point]:
    """Generate an ACTION from a set of FACTs.

    The action describes what one would DO given the facts. Bounded
    by `_GENERATION_TOKEN_BUDGET` tokens. The returned Point is
    source=GENERATOR, kind=ACTION, parents=fact_ids.
    """
    if not facts:
        return None
    if not hasattr(llm, "generate"):
        return None
    context = "\n".join(f"- {p.content}" for p in facts[:5])
    prompt = (
        "Given these facts, propose ONE concrete action (an imperative "
        "phrase, ≤ 12 words). No explanation, just the action.\n\n"
        f"{context}"
    )
    try:
        text = llm.generate(prompt, max_tokens=_GENERATION_TOKEN_BUDGET).strip()
    except Exception:
        return None
    if not text:
        return None
    kws = extractor.extract(text, n=5) if extractor else []
    kw_emb = position_weighted_keyword_embedding(kws, encoder)
    return Point(
        id=_gen_id("act_gen", t_now),
        content=text,
        embedding_orig=tuple(encoder.encode(text)),
        kind=PointKind.ACTION,
        state=EpistemicState.CONJECTURE,
        parents=[p.id for p in facts[:5]],
        lineage_depth=max((p.lineage_depth for p in facts[:5]), default=0) + 1,
        keywords=kws,
        keywords_embedding=kw_emb,
        keywords_source=SourceClass.GENERATOR,
    )


def meta_thought(
    fact: Point,
    action: Point,
    population: Sequence[Point],
    llm: Any,
    encoder: Any,
    extractor: Any,
    t_now: float,
) -> Optional[Point]:
    """Generate a meta-cognitive THOUGHT that fuses the keyword axes of
    a FACT and an ACTION, INFORMED by the fact's metacognitive state.

    The THOUGHT now consumes :
      - content of FACT + ACTION
      - keyword sets of both
      - META-STATE of the fact (confidence, σ, n_uses, hub status)
      - if the fact is a HUB, a small cluster of its manifold neighbours
        (edge-free analog of HeLa's reflective Hebbian distillation)

    A(·) ⊥ P : the meta-state is computed from integer counters and the
    population's distance distribution only — no foreign P-content
    influences the reflection except the fact's own.

    Bounded by `_GENERATION_TOKEN_BUDGET` tokens. Cor. 5 :
    source=GENERATOR.
    """
    if not hasattr(llm, "generate"):
        return None
    fact_kws = ", ".join(fact.keywords[:5]) or "(none)"
    action_kws = ", ".join(action.keywords[:5]) or "(none)"
    state = _meta_state(fact, population)
    meta_line = (
        f"FACT-STATE conf={state['confidence']} sigma={state['uncertainty']} "
        f"uses={state['n_uses']} corr={state['n_corrob']} "
        f"contra={state['n_contra']} hub={state['is_hub']}"
    )
    cluster_lines: List[str] = []
    if state["is_hub"]:
        cluster_lines = _reflective_context(fact, population, t_now)
    cluster_block = ""
    if cluster_lines:
        cluster_block = (
            "\nCLUSTER (manifold neighbours of this hub, for consolidation) :\n"
            + "\n".join(f"  - {c}" for c in cluster_lines)
        )
    prompt = (
        "Reflect on the relation between the FACT and the ACTION, taking "
        "the FACT's epistemic state into account. Output ONE short "
        "reflection (≤ 15 words) naming the entities/properties/relations "
        "that connect them. No prose, just the reflection.\n\n"
        f"{meta_line}\n"
        f"FACT [{fact_kws}] : {fact.content}\n"
        f"ACTION [{action_kws}] : {action.content}"
        f"{cluster_block}"
    )
    try:
        text = llm.generate(prompt, max_tokens=_GENERATION_TOKEN_BUDGET).strip()
    except Exception:
        return None
    if not text:
        return None
    # Enriched keyword set = thought-extracted keywords ∪ parent keywords
    base_kws = extractor.extract(text, n=8) if extractor else []
    parent_kws = list(fact.keywords) + list(action.keywords)
    merged: List[str] = []
    seen: set = set()
    for kw in base_kws + parent_kws:
        if kw and kw not in seen:
            seen.add(kw)
            merged.append(kw)
    kws = merged[:8]
    kw_emb = position_weighted_keyword_embedding(kws, encoder)
    return Point(
        id=_gen_id("thought_meta", t_now),
        content=text,
        embedding_orig=tuple(encoder.encode(text)),
        kind=PointKind.THOUGHT,
        state=EpistemicState.CONJECTURE,
        parents=[fact.id, action.id],
        lineage_depth=max(fact.lineage_depth, action.lineage_depth) + 1,
        keywords=kws,
        keywords_embedding=kw_emb,
        keywords_source=SourceClass.GENERATOR,
    )


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def meta_walk(
    query: str,
    memory: Any,
    *,
    n_stages: int = _DEFAULT_STAGES,
    facts_per_stage: int = _FACTS_PER_STAGE,
    actions_per_stage: int = _ACTIONS_PER_STAGE,
    t_now: Optional[float] = None,
    pull_strength: float = 1.0,
) -> MetaWalkTrajectory:
    """Run the meta-cognitive walk and return the trajectory.

    SIDE-EFFECTS on the memory (intended) :
      - Generated ACTIONs and THOUGHTs are appended to `memory.points`
        (with `source=GENERATOR`, never observations — Cor. 5 holds).
      - For each stage, apply_pull(fact*, action*) fires (Hebbian-style
        co-activation : the chosen fact and action are dragged together
        in the manifold). This feeds the collision machinery downstream.

    The cumulative `fact_ids_cumulative` is what the answerer should
    use as effective retrieval — it is what HeLa would call the
    "spreading-augmented set" but built without edges.
    """
def synthesize_answer_from_walk(
    query: str,
    trajectory: "MetaWalkTrajectory",
    memory: Any,
    *,
    max_tokens: int = 120,
) -> str:
    """Build the final answer by following the walk's filiation.

    The walk has produced a chain of (fact*, action*, thought) per
    stage. For multi-hop the answer must respect that chain — not just
    collect a flat bag of evidence. We expose the stages IN ORDER so
    the LLM constructs the answer by composing across them, exactly
    matching the filiation encoded in `thought.parents`.

    Single LLM call, budget-capped. Stays Cor. 5 compliant : the
    synthesis is the answer to a question, it does NOT enter the
    memory as an observation.
    """
    if not trajectory.stages:
        return ""
    if not hasattr(memory.llm, "generate"):
        # Fallback : concatenate the chosen facts.
        return " ".join(
            next((p.content for p in memory.points if p.id == s.chosen_fact_id), "")
            for s in trajectory.stages if s.chosen_fact_id
        )
    by_id = {p.id: p for p in memory.points}
    lines: List[str] = []
    for s in trajectory.stages:
        f = by_id.get(s.chosen_fact_id or "")
        a = by_id.get(s.chosen_action_id or "")
        t = by_id.get(s.thought_id or "")
        if f:
            lines.append(f"  Fact[{s.stage}] : {f.content}")
        if a:
            lines.append(f"  Action[{s.stage}] : {a.content}")
        if t:
            lines.append(f"  Thought[{s.stage}] : {t.content}")
    prompt = (
        "Answer the question by composing across the stages (multi-hop "
        "filiation). Reply with the bare value matching the gold style "
        "(date only / place only / short noun phrase / Not mentioned). "
        "No prose.\n\n"
        f"QUERY : {query}\n"
        "WALK :\n" + "\n".join(lines)
    )
    try:
        return memory.llm.generate(prompt, max_tokens=max_tokens).strip()
    except Exception:
        return ""


def meta_walk(
    query: str,
    memory: Any,
    *,
    n_stages: int = _DEFAULT_STAGES,
    facts_per_stage: int = _FACTS_PER_STAGE,
    actions_per_stage: int = _ACTIONS_PER_STAGE,
    t_now: Optional[float] = None,
    pull_strength: float = 1.0,
) -> MetaWalkTrajectory:
    if t_now is None:
        t_now = memory._now() if hasattr(memory, "_now") else 0.0

    enc = memory.encoder
    extractor = memory.extractor
    llm = memory.llm

    traj = MetaWalkTrajectory(query=query)

    # Stage 0 : seed from the raw query — position-weighted on its
    # extracted keywords so the most salient one drives the embedding.
    query_kws = extractor.extract(query, n=8) if extractor else []
    pwe = position_weighted_keyword_embedding(query_kws, enc) if query_kws else None
    query_emb = pwe if pwe is not None else tuple(enc.encode(query))

    facts = nearest_facts_with_fallback(
        query_emb, query, memory.points, facts_per_stage, t_now,
        encoder=enc, extractor=extractor,
    )
    actions = [p for _s, p in nearest_by_kind(
        query_emb, memory.points, PointKind.ACTION, actions_per_stage, t_now,
    )]
    record = StageRecord(stage=0,
                         fact_ids=[p.id for p in facts],
                         action_ids=[p.id for p in actions])

    if not actions and facts:
        generated = generate_action(facts, llm, enc, extractor, t_now)
        if generated is not None:
            memory.points.append(generated)
            actions = [generated]
            record.action_ids = [generated.id]
            record.generated_action = True
            traj.generated_point_ids.append(generated.id)

    fact_star = least_uncertain(facts)
    action_star = actions[0] if actions else None
    record.chosen_fact_id = fact_star.id if fact_star else None
    record.chosen_action_id = action_star.id if action_star else None
    traj.fact_ids_cumulative.extend(record.fact_ids)
    traj.stages.append(record)

    if fact_star is None or action_star is None:
        return traj

    # Hebbian co-activation : drag the chosen fact and action together.
    # apply_pull mutates delta_active/latent (COMPUTATION on vectors).
    apply_pull(fact_star, action_star, polarity=+pull_strength, t_now=t_now)
    fact_star.n_uses += 1
    action_star.n_uses += 1

    # Stage 0 thought (the bridge that produces the enriched keyword set)
    thought = meta_thought(
        fact_star, action_star, memory.points,
        llm, enc, extractor, t_now,
    )
    if thought is not None:
        memory.points.append(thought)
        traj.stages[-1].thought_id = thought.id
        traj.stages[-1].generated_thought = True
        traj.generated_point_ids.append(thought.id)
    else:
        return traj

    # --- Iterate ---
    visited_fact_ids: set = set(record.fact_ids)
    visited_action_ids: set = set(record.action_ids)
    prev_action = action_star
    cur_thought = thought

    for stage in range(1, n_stages):
        kws_new = cur_thought.keywords or query_kws
        pwe_n = position_weighted_keyword_embedding(kws_new, enc) if kws_new else None
        new_emb = pwe_n if pwe_n is not None else query_emb

        # FACT retrieval with BM25 fallback when keyword kNN runs dry.
        # The fallback query text combines the new keyword set so BM25
        # can match its rare verbatim tokens on the full content.
        fallback_query = " ".join(kws_new) if kws_new else query
        facts_next = nearest_facts_with_fallback(
            new_emb, fallback_query, memory.points, facts_per_stage, t_now,
            exclude_ids=visited_fact_ids,
            encoder=enc, extractor=extractor,
        )
        action_next = nearest_action_to(
            prev_action, memory.points, t_now,
            exclude_ids=visited_action_ids,
        )
        rec = StageRecord(
            stage=stage,
            fact_ids=[p.id for p in facts_next],
            action_ids=[action_next.id] if action_next else [],
        )

        if action_next is None and facts_next:
            generated = generate_action(facts_next, llm, enc, extractor, t_now)
            if generated is not None:
                memory.points.append(generated)
                action_next = generated
                rec.action_ids = [generated.id]
                rec.generated_action = True
                traj.generated_point_ids.append(generated.id)

        fact_star_next = least_uncertain(facts_next)
        rec.chosen_fact_id = fact_star_next.id if fact_star_next else None
        rec.chosen_action_id = action_next.id if action_next else None
        traj.fact_ids_cumulative.extend(rec.fact_ids)
        traj.stages.append(rec)

        if fact_star_next is None or action_next is None:
            return traj

        visited_fact_ids.update(rec.fact_ids)
        visited_action_ids.add(action_next.id)

        apply_pull(fact_star_next, action_next,
                   polarity=+pull_strength, t_now=t_now)
        fact_star_next.n_uses += 1
        action_next.n_uses += 1

        thought_next = meta_thought(
            fact_star_next, action_next, memory.points,
            llm, enc, extractor, t_now,
        )
        if thought_next is None:
            return traj
        memory.points.append(thought_next)
        rec.thought_id = thought_next.id
        rec.generated_thought = True
        traj.generated_point_ids.append(thought_next.id)
        prev_action = action_next
        cur_thought = thought_next

    return traj


# ---------------------------------------------------------------------------
# Stateful walker for MCP-driven step-by-step traversal
# ---------------------------------------------------------------------------


class MetaWalker:
    """Stateful one-step-at-a-time meta-cognitive walker.

    Designed for an external agent driving the walk over MCP : the agent
    calls `step()` after each stage, sees the full content of every
    retrieved node, decides whether to keep going or to synthesize.

    Construction runs stage 0 immediately (so the first `step()` returns
    stage 0). Each subsequent `step()` either advances or signals
    `done=True` if the walk is exhausted (no new facts, no new action,
    or the configured `n_stages` ceiling reached).

    The walker holds a reference to the live `memory`, so generated
    THOUGHT/ACTION points are appended to it as they are produced —
    they participate in collision/observator mechanics like any other
    Point. Cor. 5 : all generated points carry `source=GENERATOR`.
    """

    def __init__(
        self,
        query: str,
        memory: Any,
        *,
        n_stages: int = _DEFAULT_STAGES,
        facts_per_stage: int = _FACTS_PER_STAGE,
        actions_per_stage: int = _ACTIONS_PER_STAGE,
        pull_strength: float = 1.0,
        t_now: Optional[float] = None,
    ) -> None:
        self.query = query
        self.memory = memory
        self.n_stages = n_stages
        self.facts_per_stage = facts_per_stage
        self.actions_per_stage = actions_per_stage
        self.pull_strength = pull_strength
        self.t_now = (
            t_now if t_now is not None
            else (memory._now() if hasattr(memory, "_now") else 0.0)
        )

        self._enc = memory.encoder
        self._extr = memory.extractor
        self._llm = memory.llm

        # Stage-0 seed embedding from the query keywords. Position-
        # weighted (1/(i+1) decay) so the top keyword drives the vector.
        q_kws = self._extr.extract(query, n=8) if self._extr else []
        self._query_keywords = q_kws
        pwe0 = position_weighted_keyword_embedding(q_kws, self._enc) if q_kws else None
        self._cur_emb = pwe0 if pwe0 is not None else tuple(self._enc.encode(query))

        self._fact_ids_cum: List[str] = []
        self._visited_fact_ids: set = set()
        self._visited_action_ids: set = set()
        self._prev_action: Optional[Point] = None
        self._cur_thought: Optional[Point] = None
        self._stage_idx = 0
        self._done = False
        self._generated_ids: List[str] = []

    # ----------------------------------------------------------------

    def walk(self):
        """Generator : yield one StageOutput per stage until exhausted.

        Lets a caller validate / inspect each step as it happens :

            for stage in walker.walk():
                ...inspect stage...   # decide to break early if wanted
        """
        while not self._done:
            yield self.step()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def generated_point_ids(self) -> List[str]:
        return list(self._generated_ids)

    # ----------------------------------------------------------------
    # The step
    # ----------------------------------------------------------------

    def step(self) -> StageOutput:
        """Advance one stage and return the StageOutput. Sets done=True
        on the returned output when the walk cannot continue further."""
        if self._done:
            # Idempotent : repeated calls after exhaustion return an
            # empty done-stage instead of raising.
            return StageOutput(
                stage=self._stage_idx,
                facts=[], actions=[],
                chosen_fact=None, chosen_action=None, thought=None,
                generated_action=False, generated_thought=False,
                fact_ids_cumulative=list(self._fact_ids_cum),
                done=True,
            )

        # Retrieval — for stage 0 we seed from the query embedding ; for
        # later stages from the current thought's enriched keywords (set
        # at the end of the previous step).
        if self._stage_idx == 0:
            seed_query = self.query
        else:
            seed_kws = self._cur_thought.keywords if self._cur_thought else []
            seed_query = " ".join(seed_kws) if seed_kws else self.query

        facts = nearest_facts_with_fallback(
            self._cur_emb, seed_query, self.memory.points,
            self.facts_per_stage, self.t_now,
            exclude_ids=self._visited_fact_ids,
            encoder=self._enc, extractor=self._extr,
        )

        # ACTIONs : at stage 0 nearest by query embedding ; at later
        # stages nearest to the previous action.
        if self._stage_idx == 0 or self._prev_action is None:
            actions = [
                p for _s, p in nearest_by_kind(
                    self._cur_emb, self.memory.points, PointKind.ACTION,
                    self.actions_per_stage, self.t_now,
                    exclude_ids=self._visited_action_ids,
                )
            ]
        else:
            a = nearest_action_to(
                self._prev_action, self.memory.points, self.t_now,
                exclude_ids=self._visited_action_ids,
            )
            actions = [a] if a else []

        gen_action = False
        if not actions and facts:
            generated = generate_action(
                facts, self._llm, self._enc, self._extr, self.t_now,
            )
            if generated is not None:
                self.memory.points.append(generated)
                actions = [generated]
                gen_action = True
                self._generated_ids.append(generated.id)

        fact_star = least_uncertain(facts)
        action_star = actions[0] if actions else None

        # Accumulate FACT ids (this is the "effective recall" the
        # answerer sees across the whole walk).
        for f in facts:
            if f.id not in self._visited_fact_ids:
                self._fact_ids_cum.append(f.id)
                self._visited_fact_ids.add(f.id)
        if action_star and action_star.id not in self._visited_action_ids:
            self._visited_action_ids.add(action_star.id)

        gen_thought = False
        thought: Optional[Point] = None
        if fact_star is not None and action_star is not None:
            # Hebbian co-activation + use bump.
            apply_pull(fact_star, action_star,
                       polarity=+self.pull_strength, t_now=self.t_now)
            fact_star.n_uses += 1
            action_star.n_uses += 1

            thought = meta_thought(
                fact_star, action_star, self.memory.points,
                self._llm, self._enc, self._extr, self.t_now,
            )
            if thought is not None:
                self.memory.points.append(thought)
                gen_thought = True
                self._generated_ids.append(thought.id)
                self._cur_thought = thought
                # Refresh the query embedding for the NEXT stage from
                # the thought's enriched keywords.
                kws = thought.keywords or self._query_keywords
                if kws:
                    pwe = position_weighted_keyword_embedding(kws, self._enc)
                    if pwe is not None:
                        self._cur_emb = pwe

        out = StageOutput(
            stage=self._stage_idx,
            facts=[_node_payload(p) for p in facts],
            actions=[_node_payload(p) for p in actions if p is not None],
            chosen_fact=_node_payload(fact_star) if fact_star else None,
            chosen_action=_node_payload(action_star) if action_star else None,
            thought=_node_payload(thought) if thought else None,
            generated_action=gen_action,
            generated_thought=gen_thought,
            fact_ids_cumulative=list(self._fact_ids_cum),
            done=False,
        )

        self._prev_action = action_star

        # Terminal conditions for the NEXT step.
        next_idx = self._stage_idx + 1
        cant_continue = (
            fact_star is None
            or action_star is None
            or thought is None
        )
        if cant_continue or next_idx >= self.n_stages:
            self._done = True
            out.done = True

        self._stage_idx = next_idx
        return out


# ---------------------------------------------------------------------------
# In-process walker registry (per-MCP-server) for tool-call continuity
# ---------------------------------------------------------------------------


class WalkerRegistry:
    """Holds active MetaWalker instances by walk_id so a sequence of
    MCP tool calls (walk_start → walk_next → walk_next → …) can drive
    the same walk."""

    def __init__(self) -> None:
        self._walkers: dict = {}
        self._counter: int = 0

    def open(self, walker: MetaWalker) -> str:
        self._counter += 1
        walk_id = f"walk_{self._counter}"
        self._walkers[walk_id] = walker
        return walk_id

    def get(self, walk_id: str) -> Optional[MetaWalker]:
        return self._walkers.get(walk_id)

    def close(self, walk_id: str) -> None:
        self._walkers.pop(walk_id, None)
