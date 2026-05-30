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

import math
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
# Match the system retrieval budget (k=7) so a walk's stage-0 facts are
# the same top-k a single-shot retrieve would return — the walk can then
# only ADD evidence across later stages, never lose what single-shot found.
_FACTS_PER_STAGE = 7
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


def _node_payload(p: Point, relevance: Optional[str] = None) -> dict:
    """Compact JSON-ready dump of a node for an external (MCP) caller.

    Always carries the INDEXED CONTENT so the agent driving the walk
    over MCP can reason on the actual text, not just ids — even though
    retrieval itself runs on keyword kNN.

    `relevance` (Chain-of-Note) annotates how the retrieved fact relates
    to the query : relevant | partial | irrelevant | contradicts. It is
    a READING NOTE for the agent and the THOUGHT prompt — it never drops
    the node from the cumulative recall set, so recall is preserved while
    the reasoning gains precision.
    """
    out = {
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
    if relevance is not None:
        out["relevance"] = relevance
    return out


@dataclass
class StageOutput:
    """One stage's result as returned to an MCP caller.

    Carries full content of every retrieved node (not just ids) plus a
    'done' flag that lets the agent decide whether to call walk_next
    or move on to synthesis.

    sigma_path : cumulative σ propagated in quadrature over the walk's
    embedding hops. High sigma_path signals that depth is exhausted and
    the agent should pivot to a breadth walk_start with a reformulated
    query targeting the missing aspect.

    n_relevant : Chain-of-Note count of facts that read
    relevant/partial/contradicts (i.e. on-target) at this stage.
    drifted : True when n_relevant == 0 — every retrieved fact reads
    irrelevant, so the walk has wandered off the query. This is a SOFT
    hint : it does NOT terminate the walk (that would risk recall), it
    tells the agent to pivot breadth with a reformulated query.
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
    sigma_path: float = 0.0
    n_relevant: int = 0
    drifted: bool = False
    # MAP-REDUCE deliverable : the successively-collected ON-TARGET facts
    # across all stages (relevant/partial/contradicts), each {id,
    # content, relevance}. This is the reduced evidence the answerer
    # should compose over — it never loses a prior stage's bridging fact.
    relevant_collected: List[dict] = field(default_factory=list)

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
            "sigma_path": round(self.sigma_path, 4),
            "n_relevant": self.n_relevant,
            "drifted": self.drifted,
            "relevant_collected": self.relevant_collected,
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
        # Over-fetch : entity beacons (id "entity_*") are FACT-kind so they
        # score here, but they are ingest-time pull agents — their pull
        # already lifted the real facts. Drop them from the returned set so
        # the walk reasons over (and agent_recall counts) real facts only.
        results = retrieve_hybrid(
            query_text, points, (k + len(exclude_ids)) * 5 + 20, t_now,
            encoder=encoder, extractor=extractor,
            use_lineage=True, use_spreading=True, use_fuzzy=True,
            restrict_kind=PointKind.FACT,
        )
        by_id = {p.id: p for p in points}
        out: List[Point] = []
        seen: set = set()
        for _s, p in results:
            if p.id.startswith("entity_"):
                continue                       # beacon : drop
            # Resolve an atomic-fact hit ("atom_<dia>_<k>") back to its
            # source turn (parents[0]) and dedup, so the agent reasons over
            # real turns (with [date] prefix) and agent_recall counts dia_ids.
            tgt = p
            if p.id.startswith("atom_") and p.parents:
                tgt = by_id.get(p.parents[0], p)
            if tgt.id in exclude_ids or tgt.id in seen:
                continue
            seen.add(tgt.id)
            out.append(tgt)
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


# Chain-of-Note relevance labels. Categorical (not a threshold) — no
# hyperparameter. "contradicts" is surfaced rather than dropped so the
# THOUGHT can reason about the conflict (it also feeds the existing
# polarization / collision machinery downstream when committed).
_CON_LABELS = ("relevant", "partial", "contradicts", "irrelevant")


def chain_of_note(
    query: str,
    facts: Sequence[Point],
    llm: Any,
    *,
    collected: Optional[Sequence[Point]] = None,
    current_thought: Optional[str] = None,
) -> List[str]:
    """Annotate each retrieved FACT with its relevance to the query —
    judged AS A MULTI-HOP CHAIN, not as a static keyword match.

    Returns a list of labels (one per fact, same order) drawn from
    {relevant, partial, contradicts, irrelevant}. This is the
    Chain-of-Note reading pass (Yu et al., EMNLP 2024) adapted for an
    iterative walk : before the THOUGHT is composed, the walk reads
    every retrieved fact and notes its role IN THE CHAIN toward the
    answer. Crucially a fact that does NOT directly answer but BRIDGES
    to the next hop (a stepping-stone) is `relevant`, not `irrelevant` —
    going deeper does not make a bridging fact off-target. Only genuine
    different-topic chitchat is `irrelevant` (the #1 LoCoMo failure
    mode : adjacent-but-wrong turns).

    `collected` — the facts already gathered as on-target in PRIOR
    stages (the map-reduce accumulator). Passed as context so a new
    fact is judged for what it ADDS to the chain, and so the model
    understands the bridge it must continue.
    `current_thought` — the latest reflection, the "where we are" of the
    walk.

    A(·) ⊥ P is preserved at the EPISTEMIC level : this note does NOT
    update any point's epistemic state from foreign content ; it is a
    transient reasoning annotation used to focus the THOUGHT and the
    fact* choice. The cumulative recall set keeps every fact regardless
    of its label, so recall is never traded away for this precision.

    On any LLM failure we fail OPEN — every fact is labelled "relevant"
    so the walk degrades to its pre-Chain-of-Note behaviour rather than
    silently dropping evidence.
    """
    n = len(facts)
    if n == 0:
        return []
    if not hasattr(llm, "generate"):
        return ["relevant"] * n

    lines = []
    for i, p in enumerate(facts):
        lines.append(f"[{i}] {p.content}")
    listing = "\n".join(lines)

    # Context block : what the chain has already established, so the
    # model judges each new fact's role relative to where the walk is.
    ctx_parts: List[str] = []
    if collected:
        gathered = "\n".join(f"  - {p.content}" for p in list(collected)[-6:])
        ctx_parts.append("ALREADY GATHERED (on-target so far) :\n" + gathered)
    if current_thought:
        ctx_parts.append(f"CURRENT REFLECTION : {current_thought}")
    ctx_block = ("\n" + "\n".join(ctx_parts) + "\n") if ctx_parts else ""

    prompt = (
        "You are reading retrieved memory turns during a MULTI-HOP walk "
        "toward answering the QUERY. The answer may require chaining "
        "several facts. For EACH numbered fact output one label :\n"
        "  relevant    — helps answer the query, OR is a stepping-stone "
        "that bridges toward the answer (links people/events/topics the "
        "next hop needs). A fact need NOT answer directly to be relevant.\n"
        "  partial     — gives useful context but is neither answer nor "
        "a clear bridge\n"
        "  contradicts — conflicts with the gathered facts about the query\n"
        "  irrelevant  — a genuinely DIFFERENT topic / off-target chitchat "
        "that neither answers nor bridges\n\n"
        "Be generous with `relevant` for bridging facts : going deeper "
        "does not make an earlier on-topic fact irrelevant.\n"
        "Output ONLY lines of the form `i: label`, one per fact, nothing "
        "else.\n\n"
        f"QUERY : {query}\n"
        f"{ctx_block}"
        f"FACTS :\n{listing}"
    )
    try:
        # ~6 tokens per fact line is plenty for "i: label".
        raw = llm.generate(prompt, max_tokens=max(16, 8 * n)).strip()
    except Exception:
        return ["relevant"] * n
    if not raw:
        return ["relevant"] * n

    labels: List[Optional[str]] = [None] * n
    for ln in raw.splitlines():
        ln = ln.strip().lstrip("-•* ").strip()
        if ":" not in ln:
            continue
        idx_part, _, label_part = ln.partition(":")
        idx_part = idx_part.strip().strip("[]")
        if not idx_part.isdigit():
            continue
        idx = int(idx_part)
        if not (0 <= idx < n):
            continue
        label_norm = label_part.strip().lower()
        match = next((L for L in _CON_LABELS if label_norm.startswith(L)), None)
        if match is not None:
            labels[idx] = match
    # Any fact the model didn't label → fail open to "relevant".
    return [lab if lab is not None else "relevant" for lab in labels]


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
    # Anchor the action on the facts' OWN entities so it doesn't drift
    # into unrelated concepts (e.g. inventing "camping" / "beach").
    fact_entities = []
    seen_e: set = set()
    for p in facts[:5]:
        for kw in p.keywords:
            if kw and kw not in seen_e:
                seen_e.add(kw)
                fact_entities.append(kw)
    anchor = ", ".join(fact_entities[:8]) or "(the facts above)"
    prompt = (
        "Given these facts, propose ONE concrete action (an imperative "
        "phrase, ≤ 12 words) that stays ABOUT these entities — do not "
        f"introduce new topics. Entities: {anchor}.\n"
        "No explanation, just the action.\n\n"
        f"{context}"
    )
    try:
        text = llm.generate(prompt, max_tokens=_GENERATION_TOKEN_BUDGET).strip()
    except Exception:
        return None
    if not text:
        return None
    # KEYWORDS — anchored, like the THOUGHT : start from the facts'
    # entities the action text actually mentions, reorder by that, then
    # enrich only with concrete tokens shared between the action text and
    # the fact content. Never an unrelated free-extraction.
    act_tokens = {w.strip(".,;:!?\"'()").lower() for w in text.split()}
    content_vocab = {
        w.strip(".,;:!?\"'()").lower()
        for p in facts[:5] for w in p.content.split() if len(w) >= 4
    }
    emphasised = [k for k in fact_entities if k.lower() in act_tokens]
    rest = [k for k in fact_entities if k.lower() not in act_tokens]
    enrich = [w for w in act_tokens
              if w in content_vocab and w not in seen_e]
    kws = (emphasised + rest + enrich)[:5]
    if not kws:
        kws = (extractor.extract(text, n=5) if extractor else [])
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
        tags=["generated"],
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
        "Write ONE short RETROSPECTIVE reflection (≤ 15 words) on the "
        "situation — what the FACT means given the ACTION, taking the "
        "fact's epistemic state into account. This is a thought looking "
        "BACK on the fact and action it is born from.\n\n"
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

    # KEYWORDS — never generated from zero. The thought ENRICHES and
    # REORDERS the parent (fact + action) keywords ; the reflection only
    # decides which of them matter most for the next hop. We additionally
    # allow ENRICHMENT with tokens that appear in the fact/action CONTENT
    # (concrete entities) — never the abstract relational vocabulary of
    # the reflection itself.
    parent_kws: List[str] = []
    seen: set = set()
    for kw in list(fact.keywords) + list(action.keywords):
        if kw and kw not in seen:
            seen.add(kw)
            parent_kws.append(kw)

    # Tokens the reflection emphasises (lowercased word set).
    refl_tokens = {w.strip(".,;:!?\"'()").lower() for w in text.split()}
    # Concrete vocabulary the parents are actually made of.
    content_vocab = {
        w.strip(".,;:!?\"'()").lower()
        for w in (fact.content + " " + action.content).split()
        if len(w) >= 4
    }

    # 1. Parent keywords the reflection emphasised come FIRST (reordered
    #    by metacognitive relevance), then the remaining parent keywords.
    emphasised = [k for k in parent_kws if k.lower() in refl_tokens]
    rest = [k for k in parent_kws if k.lower() not in refl_tokens]
    # 2. ENRICH : concrete content tokens the reflection surfaced that are
    #    not already keywords (grounded entities, never abstract prose).
    enrich = [
        w for w in refl_tokens
        if w in content_vocab and w not in seen
    ]
    kws = (emphasised + rest + enrich)[:8]
    if not kws:
        kws = parent_kws[:8]
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
        tags=["generated"],
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
        # Re-anchor : original query keywords FIRST (so position weighting
        # keeps them dominant), then the thought's enrichment.
        thought_kws = cur_thought.keywords if cur_thought else []
        kws_new = list(query_kws) + list(thought_kws)
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
        commit: bool = True,
        use_chain_of_note: bool = True,
        observator_id: Optional[str] = None,
    ) -> None:
        """`commit` controls whether the walk MUTATES the shared memory.

        commit=True  (live use) : generated ACTION/THOUGHT points are
            appended to `memory.points`, apply_pull fires, n_uses is
            bumped — the walk feeds collision/observator dynamics.
        commit=False (benchmark / concurrent eval) : generated points
            live in a per-walk local list (visible only to this walk's
            retrieval), no pull, no counter bump. This makes concurrent
            walks over a shared memory fully ISOLATED and reproducible —
            essential when many QAs run in parallel against one memory.

        `use_chain_of_note` (default True) : run a relevance reading pass
            over each stage's retrieved facts. Irrelevant facts are kept
            in the cumulative recall set (recall preserved) but excluded
            from the THOUGHT seed and the fact* choice (precision gained).
            When EVERY fact at a stage reads irrelevant, the walk is
            drifting → it stops so the agent can pivot breadth.
        """
        self.query = query
        self.memory = memory
        self.n_stages = n_stages
        self.facts_per_stage = facts_per_stage
        self.actions_per_stage = actions_per_stage
        self.pull_strength = pull_strength
        self.commit = commit
        self.use_chain_of_note = use_chain_of_note
        # Optional Level-1 community activation : restricts FACT retrieval
        # to this observator's members (see _all_points).
        self.observator_id = observator_id
        # Per-walk scratch for generated points when commit=False.
        self._local_points: List[Point] = []
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
        # MAP-REDUCE accumulator : the successively-collected set of
        # ON-TARGET facts (relevant/partial/contradicts per Chain-of-Note)
        # across ALL stages. A bridging fact found at état -1 PERSISTS
        # here into état k — going deeper never discards prior relevant
        # evidence. This reduced collection is what seeds each THOUGHT
        # and what the answerer composes over.
        self._relevant_cum: List[Point] = []
        self._relevant_ids: set = set()
        self._relevant_label: dict = {}
        # σ-propagation depth-stop state.
        self._sigma_path: float = 0.0
        self._prev_seed_emb: Optional[tuple] = None
        # Walk-local emergent threshold : median + std of pairwise cosine
        # distances between stage-0 retrieved facts. Set once after the
        # first retrieval. Replaces prune_threshold(points) which returns
        # None for cold-start memories (uniform beta_sigma).
        self._walk_sigma_cutoff: Optional[float] = None

    # ----------------------------------------------------------------

    def _all_points(self) -> List[Point]:
        """The point set this walk retrieves over : the shared memory
        plus this walk's locally-generated points (when commit=False).
        When commit=True the generated points are already in
        memory.points, so the local list is empty and this is just the
        shared list.

        When `observator_id` is set (the agent activated a Level-1
        community), the candidate FACTs are restricted to that community's
        members. Entity beacons (id "entity_*") and walk scaffolding
        (THOUGHT / ACTION) are ALWAYS kept — beacons are shared anchors and
        scaffolding is community-agnostic."""
        base = (list(self.memory.points) + self._local_points
                if self._local_points else list(self.memory.points))
        oid = getattr(self, "observator_id", None)
        if not oid:
            return base
        out: List[Point] = []
        for p in base:
            if p.kind != PointKind.FACT or p.id.startswith("entity_"):
                out.append(p)              # scaffolding + shared beacons
            elif oid in p.observator_views:
                out.append(p)              # community member FACT
        # If activation matched nothing (bad id), fall back to the full set
        # rather than starve the walk.
        return out if any(
            p.kind == PointKind.FACT and not p.id.startswith("entity_")
            for p in out
        ) else base

    def _add_generated(self, point: Point) -> None:
        """Store a generated ACTION/THOUGHT : into shared memory when
        committing, otherwise into the per-walk scratch."""
        if self.commit:
            self.memory.points.append(point)
        else:
            self._local_points.append(point)
        self._generated_ids.append(point.id)

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
                sigma_path=self._sigma_path,
            )

        # Retrieval — stage 0 seeds from the query ; later stages from
        # the thought's enriched keywords ALWAYS combined with the
        # original query keywords (re-anchoring). The thought enriches /
        # advances the search, but the question stays in the signal so a
        # drifting generated action cannot pull the walk off-topic.
        if self._stage_idx == 0:
            seed_query = self.query
            seed_emb = self._cur_emb
        else:
            thought_kws = self._cur_thought.keywords if self._cur_thought else []
            anchored_kws = list(self._query_keywords) + list(thought_kws)
            seed_query = " ".join(anchored_kws) if anchored_kws else self.query
            pwe = position_weighted_keyword_embedding(anchored_kws, self._enc)
            seed_emb = pwe if pwe is not None else self._cur_emb

        # σ-propagation depth-stop — the cumulative embedding drift
        # between successive walk seeds, propagated in GUM quadrature.
        # When the total drift exceeds the walk-local emergent threshold
        # (median + std of stage-0 fact pairwise distances), depth is
        # exhausted ; the agent should pivot breadth via a new walk_start
        # with a query targeting the missing aspect.
        if self._stage_idx > 0 and self._prev_seed_emb is not None:
            hop = max(0.0, 1.0 - cosine(self._prev_seed_emb, seed_emb))
            self._sigma_path = math.sqrt(self._sigma_path ** 2 + hop ** 2)
            cutoff = self._walk_sigma_cutoff
            if cutoff is not None and self._sigma_path > cutoff:
                self._done = True
                return StageOutput(
                    stage=self._stage_idx,
                    facts=[], actions=[],
                    chosen_fact=None, chosen_action=None, thought=None,
                    generated_action=False, generated_thought=False,
                    fact_ids_cumulative=list(self._fact_ids_cum),
                    done=True,
                    sigma_path=self._sigma_path,
                )
        self._prev_seed_emb = seed_emb

        pts = self._all_points()
        facts = nearest_facts_with_fallback(
            seed_emb, seed_query, pts,
            self.facts_per_stage, self.t_now,
            exclude_ids=self._visited_fact_ids,
            encoder=self._enc, extractor=self._extr,
        )

        # Calibrate the walk-local σ threshold once from stage-0 facts.
        # median + std of pairwise cosine distances between the initial
        # retrieved facts = "typical manifold resolution" at this query.
        # Used instead of prune_threshold(points) which fails for
        # cold-start memories where all beta_sigma are identical.
        if self._stage_idx == 0 and self._walk_sigma_cutoff is None:
            fact_embs = [
                effective_keyword_embedding(f, self.t_now) for f in facts
            ]
            fact_embs = [e for e in fact_embs if e is not None]
            if len(fact_embs) >= 3:
                dists = [
                    max(0.0, 1.0 - cosine(fact_embs[i], fact_embs[j]))
                    for i in range(len(fact_embs))
                    for j in range(i + 1, len(fact_embs))
                ]
                if dists:
                    mean_d = sum(dists) / len(dists)
                    std_d = math.sqrt(
                        sum((d - mean_d) ** 2 for d in dists) / len(dists)
                    )
                    median_d = sorted(dists)[len(dists) // 2]
                    self._walk_sigma_cutoff = median_d + std_d

        # Chain-of-Note reading pass (MAP) — annotate each retrieved fact
        # with its relevance to the ORIGINAL query, judged as part of the
        # multi-hop CHAIN : a bridging fact is relevant even if it does
        # not answer directly. Context = the map-reduce accumulator so
        # far + the current reflection, so the model judges what each new
        # fact ADDS to the chain. Labels focus the THOUGHT/fact* WITHOUT
        # dropping any fact from the recall set : precision up, recall held.
        relevance: List[str] = ["relevant"] * len(facts)
        if self.use_chain_of_note and facts:
            relevance = chain_of_note(
                self.query, facts, self._llm,
                collected=self._relevant_cum,
                current_thought=(
                    self._cur_thought.content if self._cur_thought else None
                ),
            )
        rel_by_id = {f.id: relevance[i] for i, f in enumerate(facts)}

        # REDUCE — fold this stage's on-target facts into the persistent
        # accumulator (dedup, order-preserving). Bridging evidence from
        # état -1 survives into état k ; going deeper never loses it.
        for i, f in enumerate(facts):
            if relevance[i] != "irrelevant" and f.id not in self._relevant_ids:
                self._relevant_cum.append(f)
                self._relevant_ids.add(f.id)
                self._relevant_label[f.id] = relevance[i]

        # The subset the reasoning trusts : everything except clearly
        # off-target turns. "contradicts" is kept (the THOUGHT must see
        # the conflict). Fall back to ALL facts if the note left nothing.
        focus_facts = [
            f for i, f in enumerate(facts)
            if relevance[i] != "irrelevant"
        ] or list(facts)

        # ACTIONs : at stage 0 nearest by query embedding ; at later
        # stages nearest to the previous action.
        if self._stage_idx == 0 or self._prev_action is None:
            actions = [
                p for _s, p in nearest_by_kind(
                    self._cur_emb, pts, PointKind.ACTION,
                    self.actions_per_stage, self.t_now,
                    exclude_ids=self._visited_action_ids,
                )
            ]
        else:
            a = nearest_action_to(
                self._prev_action, pts, self.t_now,
                exclude_ids=self._visited_action_ids,
            )
            actions = [a] if a else []

        gen_action = False
        if not actions and focus_facts:
            # Generate the action from the FOCUSED facts so it doesn't
            # anchor on an adjacent-but-wrong turn.
            generated = generate_action(
                focus_facts, self._llm, self._enc, self._extr, self.t_now,
            )
            if generated is not None:
                self._add_generated(generated)
                actions = [generated]
                gen_action = True

        # fact* is chosen among the FOCUSED facts (Chain-of-Note has
        # filtered the off-target turns) so the THOUGHT is seeded by a
        # fact that actually bears on the query.
        fact_star = least_uncertain(focus_facts)
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
            # Hebbian co-activation + use bump — only when committing to
            # the live memory (skipped during isolated benchmark walks).
            if self.commit:
                apply_pull(fact_star, action_star,
                           polarity=+self.pull_strength, t_now=self.t_now)
                fact_star.n_uses += 1
                action_star.n_uses += 1

            thought = meta_thought(
                fact_star, action_star, pts,
                self._llm, self._enc, self._extr, self.t_now,
            )
            if thought is not None:
                self._add_generated(thought)
                gen_thought = True
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
            facts=[_node_payload(p, rel_by_id.get(p.id)) for p in facts],
            actions=[_node_payload(p) for p in actions if p is not None],
            chosen_fact=(
                _node_payload(fact_star, rel_by_id.get(fact_star.id))
                if fact_star else None
            ),
            chosen_action=_node_payload(action_star) if action_star else None,
            thought=_node_payload(thought) if thought else None,
            generated_action=gen_action,
            generated_thought=gen_thought,
            fact_ids_cumulative=list(self._fact_ids_cum),
            done=False,
            sigma_path=self._sigma_path,
            relevant_collected=[
                {"id": p.id, "content": p.content,
                 "relevance": self._relevant_label.get(p.id, "relevant")}
                for p in self._relevant_cum
            ],
        )

        self._prev_action = action_star

        # Chain-of-Note drift signal — SOFT. If not a single retrieved
        # fact at this stage reads relevant/partial/contradicts (all
        # off-target), the walk has drifted. We EXPOSE this as a hint
        # (out.drifted) so the agent can pivot breadth with a
        # reformulated query, but we deliberately do NOT force done from
        # it : forcing termination here would cut the walk short and risk
        # dropping gold the next stage might still surface. CoN therefore
        # only ADDS precision (focused THOUGHT/fact*) and a pivot hint —
        # it never reduces the number of stages, so recall can only stay
        # equal or improve versus the pre-CoN walk.
        on_target = sum(
            1 for r in relevance
            if r in ("relevant", "partial", "contradicts")
        )
        out.n_relevant = on_target
        out.drifted = self.use_chain_of_note and bool(facts) and on_target == 0

        # Terminal conditions for the NEXT step (drift NOT among them).
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


# ---------------------------------------------------------------------------
# Semantic walk (kind-agnostic perception layer — PR n°1).
#
# Coexists with `MetaWalker` (the staged fact*->action*->thought reasoning
# walk). This function does NOT route or compose — it only PERCEIVES :
# returns a top-K of points enriched with behavior scores against the
# four ConceptAnchors. The downstream orchestrator (PR n°3) will dispatch
# one of three flat paths (executable / factual / reasoning) on top.
# ---------------------------------------------------------------------------


def walk_semantic(
    query: str,
    memory: Any,
    *,
    k: int = 7,
    encoder: Any = None,
    extractor: Any = None,
    anchors: Any = None,
) -> List[Dict[str, Any]]:
    """Kind-agnostic semantic perception of the manifold for `query`.

    Retrieves top-`k` points via the hybrid pipeline WITHOUT restricting
    by PointKind — FACT / ACTION / THOUGHT are all candidates — then
    scores each candidate against the four ConceptAnchors (factual /
    reasoning / executable / topical). The orchestrator decides what to
    do with the result ; this function is pure perception.

    Entity beacons (`entity_*`) and atomic facts (`atom_*`) are NOT
    filtered out — perception is raw, the caller can strip later.

    Returns : list of dicts ordered by retrieval score :
        {id, kind, tags, content, score, behavior_scores}
    """
    if not memory.points:
        return []

    enc = encoder if encoder is not None else memory.encoder
    ext = extractor if extractor is not None else memory.extractor
    t_now = memory._now() if hasattr(memory, "_now") else 0.0

    if anchors is None:
        from metacog.anchors import ConceptAnchors
        anchors = ConceptAnchors(enc)

    from metacog.geometry import retrieve_hybrid
    results = retrieve_hybrid(
        query, memory.points, k, t_now,
        encoder=enc, extractor=ext,
        use_lineage=True, use_spreading=True, use_fuzzy=True,
    )

    out: List[Dict[str, Any]] = []
    for score, point in results:
        out.append({
            "id": point.id,
            "kind": point.kind.value,
            "tags": list(point.tags),
            "content": point.content,
            "score": score,
            "behavior_scores": anchors.score(point, t_now),
        })
    return out
