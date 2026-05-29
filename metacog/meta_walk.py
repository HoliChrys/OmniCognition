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
from metacog.uncertainty import beta_sigma


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
) -> List[Point]:
    """FACT retrieval with BM25 fallback on the node's full content.

    Primary signal : keyword-embedding cosine kNN. When that returns
    fewer than `k` non-zero-cosine FACTs (no meaningful entity overlap),
    we BACKFILL from BM25 on the full content text. BM25 catches rare
    verbatim tokens — dates, names, numbers — that the keyword
    embedding misses when the query doesn't share entity keywords with
    the relevant turn.

    The fallback is parameter-free : a cosine of exactly 0 (or below)
    means the query and the point's keyword vector are orthogonal —
    no entity signal at all. BM25 fills the slots up to k.
    """
    exclude_ids = exclude_ids or set()
    fact_pts = [p for p in points
                if p.kind == PointKind.FACT and p.id not in exclude_ids]

    # Primary : keyword-embedding cosine
    knn = nearest_by_kind(
        query_emb, fact_pts, PointKind.FACT, k, t_now,
        exclude_ids=exclude_ids,
    )
    relevant = [p for s, p in knn if s > 0.0]
    seen: set = {p.id for p in relevant}

    if len(relevant) >= k:
        return relevant[:k]

    # Fallback : BM25 on full content. Pulls in rare-token matches the
    # keyword embedding cannot see.
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
    """Return the most epistemically certain fact (lowest β-σ)."""
    if not facts:
        return None
    return min(facts, key=beta_sigma)


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
    kw_emb = tuple(encoder.encode(" ".join(kws))) if kws else None
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
    kw_emb = tuple(encoder.encode(" ".join(kws))) if kws else None
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

    # Stage 0 : seed from the raw query
    query_kws = extractor.extract(query, n=8) if extractor else []
    query_kw_text = " ".join(query_kws) if query_kws else query
    query_emb = tuple(enc.encode(query_kw_text))

    facts = nearest_facts_with_fallback(
        query_emb, query, memory.points, facts_per_stage, t_now,
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
        new_emb = tuple(enc.encode(" ".join(kws_new))) if kws_new else query_emb

        # FACT retrieval with BM25 fallback when keyword kNN runs dry.
        # The fallback query text combines the new keyword set so BM25
        # can match its rare verbatim tokens on the full content.
        fallback_query = " ".join(kws_new) if kws_new else query
        facts_next = nearest_facts_with_fallback(
            new_emb, fallback_query, memory.points, facts_per_stage, t_now,
            exclude_ids=visited_fact_ids,
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
