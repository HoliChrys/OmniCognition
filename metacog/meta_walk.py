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

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# Token budget per generation (50) is the contract stated by the design.
_GENERATION_TOKEN_BUDGET = 50

# DEPTH is governed by uncertainty propagation, not by a stage count :
#  - `_MIN_STAGES` : the FLOOR. The walk always runs at least this many
#    hops before σ-propagation is allowed to terminate it — always explore
#    the multi-hop neighbourhood.
#  - `_DEFAULT_STAGES` : a GENEROUS safety bound, NOT the controller. The
#    σ-cap (combined uncertainty exceeds the manifold's local resolution)
#    and the gold-retrieval gate decide when the walk actually stops ; this
#    number only prevents an unbounded loop if σ never trips.
_MIN_STAGES = 3
_DEFAULT_STAGES = 16

# Token discipline : the synthesis / agent never needs the full MAP-REDUCE
# set (it can reach dozens of facts and only bloats the prompt without
# improving the answer — recall is already captured in fact_ids_cumulative).
# We expose at most this many on-target facts as the composable evidence,
# ranked by relevance label then keyword overlap with the walk's own
# (query + thought-chain) vocabulary so vocabulary-distant gold the chain
# bridged to is kept, not the raw query terms.
_MAX_EVIDENCE = 15

# LINEAGE NEIGHBOUR EXPANSION (the "offer multiple possibilities" rule).
# When a walk's on-target grounding is THIN — the proxy for "the gold is
# probably outside the top-k the seed could reach" — the answer turn is
# often sitting one or two turns away from a turn the walk DID retrieve
# (measured on LoCoMo cat3 : the literal seed lands on D5:2/3/6/8 but the
# wealth-implying gold D5:5 stays just out of reach). Rather than commit to
# the single best chain, the walk then OFFERS the sequence-adjacent turns
# (±_NEIGHBOUR_WINDOW along the conversation chain) of everything it
# retrieved, as extra candidate evidence. Deterministic, no LLM, bounded.
_NEIGHBOUR_WINDOW = 2          # how many turns out, each direction (edge)
_MAX_NEIGHBOURS = 12          # hard cap on offered possibilities
_MAX_FILL_SPAN = 12           # max contiguous turns to fill between two hits
_NEIGHBOUR_GATE = 3           # promote into compose set when relevant_cum < this

# COMPOSE-SET FALLBACK — when the Chain-of-Note labelled almost nothing
# `relevant` (a vocab-gap walk : the seed register and the evidence register
# do not share content words), the agent would otherwise see an empty
# compose set and abstain. Retrieval has often touched the gold across the
# stages (it sits in `_fact_ids_cum`, just unlabelled). We then re-rank the
# unlabelled candidates by cosine similarity to the query embedding and
# promote the top `_FALLBACK_K` with label "retrieved" — a weaker confidence
# than "relevant", so the prompt knows.
_FALLBACK_RELEVANT_GATE = 2
_FALLBACK_K = 5


# ----------------------------------------------------------------------
# HyDE — Hypothetical Document Embeddings (Gao et al., 2022), as an
# ADDITIVE retrieval channel for the walk's stage-0 seed.
#
# Why this exists : LoCoMo "inference" questions (category 3) are
# abstract — "What might John's financial status be?" — while the
# evidence that answers them is concrete and lexically disjoint —
# "my kids have so much". The raw question embeds FAR from its own
# evidence, so keyword/BM25/content-cosine all miss it (r7 = 0).
#
# HyDE closes that gap with ZERO hardcoded vocabulary : we ask the LLM
# to imagine a concrete diary/chat line that WOULD answer the question,
# then retrieve against THAT. The hypothetical "John's family never
# worries about money; his kids have everything" lands right next to
# the real evidence. We keep the original-query channel too and only
# ADD the hypothetical's hits — so factual/temporal questions
# (categories 1,2,4) are never destabilised : their exact-match channel
# is untouched and any spurious HyDE hit is dropped by Chain-of-Note.
# ----------------------------------------------------------------------

# Default ON — HyDE is now a THIRD signal RRF-merged with the main
# retrieval and the original-question anchor (rrf_k=60), no longer an
# append. The earlier displacement issue (conv-26 cat3 Caroline went from
# recall 1.0→0.5 in append mode) is resolved by RRF: the original-query
# results keep their top ranks and HyDE only lifts items that ALSO match
# the hypothetical. Set METACOG_HYDE=0 to disable for ablation.
def _hyde_enabled() -> bool:
    return os.environ.get("METACOG_HYDE", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Process-global cache : one hypothetical per distinct query string.
# Walks re-issue the same seed query across stages and the agent issues
# the same question across retries — generate the passage once.
_HYDE_CACHE: Dict[str, str] = {}


def hyde_passage(query: str, llm: Any) -> str:
    """Generate (and cache) a concrete hypothetical evidence passage for
    `query` — the text a person might actually have written that would
    directly answer it. Returns "" if no usable LLM or on any failure
    (HyDE is strictly additive : a miss simply means no extra channel).

    Multi-angle: we ask the LLM to brainstorm 3 distinct evidence forms
    in different angles (direct mention / indirect signal / related
    activity) and concatenate them. A single hypothesis commits to one
    angle and misses when the real evidence uses a different one (e.g.
    "C.S. Lewis books" misses "Harry Potter fan" though both point to
    fantasy reading). Concatenation lets each angle contribute its own
    vocabulary tokens to the embedding."""
    q = (query or "").strip()
    if not q:
        return ""
    if q in _HYDE_CACHE:
        return _HYDE_CACHE[q]
    passage = ""
    if hasattr(llm, "generate"):
        prompt = (
            "Someone is searching a personal conversation memory (diary / "
            "chat logs) to answer this QUESTION:\n"
            f"  {q}\n\n"
            "Brainstorm THREE distinct chat lines that could each answer "
            "this question, from DIFFERENT angles. The evidence might be:\n"
            " - a direct statement on the topic, OR\n"
            " - an indirect signal / related activity that lets one infer "
            "the answer (different vocabulary, same conclusion), OR\n"
            " - a story / anecdote that implies it.\n"
            "Use concrete, everyday wording a real person would say — "
            "specific nouns and plain facts, first person where natural. "
            "Diversify proper nouns / activities across the three lines. "
            "Do NOT restate or reference the question, do NOT hedge, do "
            "NOT explain. Output exactly three lines, one per line, no "
            "numbering, no bullets."
        )
        try:
            raw = (llm.generate(prompt, max_tokens=180) or "").strip()
            # Concatenate the lines into one passage. Strip numbering /
            # bullets defensively, drop empties. Keep order: each line
            # contributes its tokens to the averaged sentence-embedding.
            lines = []
            for ln in raw.splitlines():
                s = ln.strip().lstrip("-*0123456789.) \t").strip()
                if s:
                    lines.append(s)
            passage = " ".join(lines[:3])
        except Exception:
            passage = ""
    # Only cache successful, non-empty passages. A transient 529 or empty
    # response must not permanently disable HyDE for this query for the
    # rest of the run.
    if passage:
        _HYDE_CACHE[q] = passage
    return passage
def lineage_neighbors(
    seed_ids: Sequence[str],
    points: Sequence["Point"],            # noqa: F821
    *,
    window: int = _NEIGHBOUR_WINDOW,
    cap: int = _MAX_NEIGHBOURS,
) -> List[dict]:
    """The "multiple possibilities" offered when the gold is likely just out
    of a query's reach : the sequence-adjacent turns of everything retrieved.

    Two mechanisms, both deterministic / no LLM :
      (a) GAP FILL — when retrieval landed on the spaced ENDS of a stretch
          (e.g. D5:2 and D5:8) the answer often sits in the MIDDLE (D5:5),
          out of any fixed ±window. So for each conversation chain the seeds
          touched, fill the contiguous turns between the lowest and highest
          retrieved position (bounded by `_MAX_FILL_SPAN`).
      (b) EDGE WINDOW — also offer ±`window` turns beyond every seed.

    Ingestion sets only `sequence_prev`, so the forward link is derived
    (the turn whose prev is X is X's next). Returns the turns NOT already in
    `seed_ids`, as `{id, content, relevance:"neighbor"}`, nearest-first,
    capped at `cap`. Used by the walk (over its retrieved set) and by
    `clue_search` (over its merged clue hits)."""
    seen = {s for s in (seed_ids or []) if isinstance(s, str)}
    if not seen:
        return []
    by_id = {p.id: p for p in points}
    nxt: dict = {}
    for p in points:
        if p.sequence_prev:
            nxt.setdefault(p.sequence_prev, p.id)

    ordered: List[str] = []
    seen_out: set = set()

    def _add(nid: Optional[str]) -> None:
        if nid and nid not in seen and nid not in seen_out:
            seen_out.add(nid)
            ordered.append(nid)

    # (a) gap fill per conversation chain
    chains: dict = {}
    for fid in list(seen):
        path = []
        cur = fid
        guard = 0
        while cur is not None and guard < 2000:
            path.append(cur)
            p = by_id.get(cur)
            cur = p.sequence_prev if p else None
            guard += 1
        root = path[-1] if path else fid
        chains.setdefault(root, {})[len(path) - 1] = fid
    for root, pos_map in chains.items():
        if len(pos_map) < 2:
            continue
        lo0, hi0 = min(pos_map), max(pos_map)
        if hi0 - lo0 > _MAX_FILL_SPAN:
            continue
        # PAD the fill by ±window : the gold is often just BEYOND the span the
        # clues hit (they land on D5:2/D5:3, the answer is D5:5). Filling only
        # BETWEEN the hits misses it, and the edge-window then loses the cap
        # race against dist-1 of many seeds. Padding the contiguous fill makes
        # the gold deterministically surface whenever a hit is within `window`
        # of it, and (being added before the edge-window) it survives the cap.
        lo, hi = max(0, lo0 - window), hi0 + window
        seq: List[str] = []
        cur = root
        guard = 0
        while cur is not None and guard < (hi + 2):
            seq.append(cur)
            cur = nxt.get(cur)
            guard += 1
        for pos in range(lo, hi + 1):
            if pos < len(seq):
                _add(seq[pos])

    # (b) edge window
    for dist in range(1, window + 1):
        for fid in list(seen):
            cur = fid
            for _ in range(dist):
                p = by_id.get(cur)
                cur = p.sequence_prev if p else None
                if cur is None:
                    break
            _add(cur)
            cur = fid
            for _ in range(dist):
                cur = nxt.get(cur)
                if cur is None:
                    break
            _add(cur)

    out: List[dict] = []
    for nid in ordered:
        p = by_id.get(nid)
        if p is not None and p.kind == PointKind.FACT \
                and not p.id.startswith(("entity_", "atom_")):
            out.append({"id": p.id, "content": p.content,
                        "relevance": "neighbor"})
        if len(out) >= cap:
            break
    return out


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
    anchor_query: Optional[str] = None,
    hyde_passage_text: Optional[str] = None,
    section_filter: Optional[set] = None,
    query_anchor: Any = None,
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

    anchor_query — the ORIGINAL question text (unchanged across stages).
    At stage 1+ the walk's `query_text` is a composed THOUGHT whose
    keywords may have drifted from the question's verbatim terms
    (proper nouns, abbreviations). A second BM25 pass on `anchor_query`
    anchors every stage on the verbatim question tokens and is RRF-merged
    with the main results. This is the "mix agentic recall + BM25"
    invariant: semantic drift is fine for the embedding channel but BM25
    must always stay anchored on the original question.

    hyde_passage_text — a hypothetical concrete answer to the original
    question, pre-generated by `hyde_passage()`. Closes the cat3 inference
    vocabulary gap: abstract questions ("financial status") embed far from
    their evidence ("my kids have so much"), but a hypothetical concrete
    answer ("John's family never worries about money") lands next to the
    evidence. The hypothetical drives a 3rd hybrid-retrieve pass that is
    RRF-merged with the main and anchor results (rrf_k=60). Stable across
    stages: same hypothetical, same question — used every stage like the
    BM25 anchor.

    When `encoder`/`extractor` are not supplied we fall back to the old
    keyword-cosine + BM25 path (used by unit tests that call this
    helper directly without a full Memory).
    """
    exclude_ids = exclude_ids or set()

    if encoder is not None and extractor is not None:
        from metacog.geometry import retrieve_hybrid
        from metacog.bm25 import bm25_score
        # Entity beacons (id "entity_*") are ingest-time pull agents : their
        # pull (first step 1/(1+0)=1.0) already lifted the real facts into
        # position, so they are never wanted in walk results. Crucially they
        # also BLOAT the cloud ~10x (5+ beacons/turn) and spreading activation
        # is superlinear — so we EXCLUDE them from the search pool, not just
        # from results. The pull benefit persists in the real facts' shifted
        # positions ; the walk stays fast. Atoms (id "atom_*") are kept — they
        # resolve to source turns below — so over-fetch headroom remains.
        # Also drop nodes folded under a LATERAL keeper : they were tagged
        # out of the pool so the keeper represents them in one kNN slot ;
        # their content survives in the cloud as the keeper's children.
        search_points = [p for p in points
                         if not p.id.startswith("entity_")
                         and "lateral_absorbed" not in p.tags]
        overfetch = (k + len(exclude_ids)) * 5 + 20
        results = retrieve_hybrid(
            query_text, search_points, overfetch, t_now,
            encoder=encoder, extractor=extractor,
            use_lineage=True, use_spreading=True, use_fuzzy=True,
            restrict_kind=PointKind.FACT,
        )

        # Multi-signal RRF — fuse three retrieval channels:
        #   (1) `results`             — semantic+BM25 hybrid on the
        #                                walk's current seed (`query_text`)
        #   (2) BM25 anchor on `anchor_query` — original question's
        #                                verbatim tokens (proper nouns,
        #                                abbreviations) preserved across
        #                                THOUGHT drift
        #   (3) HyDE hypothetical-answer hybrid — closes the cat3
        #                                inference vocabulary gap by
        #                                embedding a concrete answer
        # rrf_k=60 (Cormack et al. 2009 canonical). The original-question
        # channel keeps its top ranks (so factual cat1/2/4 are stable);
        # HyDE only lifts items that ALSO match the hypothetical.
        rrf_k = 60
        has_anchor = bool(anchor_query and anchor_query != query_text)
        has_hyde = bool(hyde_passage_text)
        # DOUBLE QUERY (skill sections) : when a section_filter is given
        # (e.g. {"session:s7", "user:u42"} or a skill name tag), run a
        # SECOND retrieval restricted to the points carrying ALL those tags
        # — the "section" — and RRF-merge it with the free query. This is
        # the pre-filtered-on-the-session ⊕ free-semantic-query the skill
        # system uses : the section's own tools/files get a rank boost
        # without excluding anything from the free channel.
        has_section = bool(section_filter)
        # SLICE C — typed query-alignment anchor (ColBERT MaxSim + IDF),
        # fired at EVERY stage (a fixed-fraction anchor per the multi-hop
        # drift literature), not only on drift like the BM25 channel. RRF-
        # merged, so it can only PROMOTE on-question facts, never drop any
        # (recall preserved) — additive, not substitutive.
        has_align = bool(
            query_anchor is not None and not query_anchor.is_empty())
        if has_anchor or has_hyde or has_section or has_align:
            rrf_scores: dict = {}
            pid_map: dict = {}
            for rank, (_, p) in enumerate(results):
                rrf_scores[p.id] = rrf_scores.get(p.id, 0.0) + 1.0 / (rrf_k + rank)
                pid_map.setdefault(p.id, p)
            if has_section:
                # Tags are normalised to lower-case on the Point, so the
                # filter must match case-insensitively (a user/session id
                # with capitals would otherwise silently match nothing).
                sect_lc = {s.lower() for s in section_filter}
                sect = [p for p in search_points
                        if sect_lc.issubset({t.lower() for t in p.tags})]
                if sect:
                    sect_pool = retrieve_hybrid(
                        query_text, sect, overfetch, t_now,
                        encoder=encoder, extractor=extractor,
                        use_lineage=True, use_spreading=True, use_fuzzy=True,
                        restrict_kind=PointKind.FACT,
                    )
                    for rank, (_, p) in enumerate(sect_pool):
                        rrf_scores[p.id] = rrf_scores.get(p.id, 0.0) + 1.0 / (rrf_k + rank)
                        pid_map.setdefault(p.id, p)
            if has_anchor:
                anchor_pool = bm25_score(
                    anchor_query,
                    [p for p in search_points if p.kind == PointKind.FACT],
                    k_pool=overfetch,
                )
                for rank, (_, p) in enumerate(anchor_pool):
                    rrf_scores[p.id] = rrf_scores.get(p.id, 0.0) + 1.0 / (rrf_k + rank)
                    pid_map.setdefault(p.id, p)
            if has_hyde:
                hyde_pool = retrieve_hybrid(
                    hyde_passage_text, search_points, overfetch, t_now,
                    encoder=encoder, extractor=extractor,
                    use_lineage=True, use_spreading=True, use_fuzzy=True,
                    restrict_kind=PointKind.FACT,
                )
                for rank, (_, p) in enumerate(hyde_pool):
                    rrf_scores[p.id] = rrf_scores.get(p.id, 0.0) + 1.0 / (rrf_k + rank)
                    pid_map.setdefault(p.id, p)
            if has_align:
                from metacog.query_anchor import alignment_score
                fact_pts = [p for p in search_points
                            if p.kind == PointKind.FACT]
                # The anchor is INPUT + RELATIVE-thought, added. The INPUT's
                # embedding cosine is DROPPED (use_input_sem=False) : it
                # re-injects the dominant-topic bias (cos(question, fact)
                # favours the co-present topic) — that is what drifted Caroline
                # to recall 0. The precise IDF exact-match on the input's
                # salient terms is kept (lex), and the RELATIVE thought
                # abstraction's cosine is ADDED (use_relative_sem) : it is the
                # inference bridge — "fingers too big" ranks near stress for the
                # input (cos .07) but FIRST for the thought "obesity" (cos .15).
                # Scored against the fact's frozen embedding_orig (no re-encode).
                scored_al = [
                    (alignment_score(query_anchor, p.content,
                                     p.embedding_orig,
                                     use_input_sem=False,
                                     use_relative_sem=True), p)
                    for p in fact_pts
                ]
                scored_al.sort(key=lambda x: -x[0])
                for rank, (al, p) in enumerate(scored_al[:overfetch]):
                    if al <= 0.0:
                        break
                    rrf_scores[p.id] = rrf_scores.get(p.id, 0.0) + 1.0 / (rrf_k + rank)
                    pid_map.setdefault(p.id, p)
            merged = sorted(rrf_scores.items(), key=lambda x: -x[1])
            results = [(sc, pid_map[pid]) for pid, sc in merged if pid in pid_map]

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
    *,
    prev_thoughts: Sequence[Point] = (),
    prev_facts: Sequence[Point] = (),
    prev_actions: Sequence[Point] = (),
    query: Optional[str] = None,
) -> Optional[Point]:
    """Generate a meta-cognitive THOUGHT that fuses the keyword axes of
    a FACT and an ACTION, INFORMED by the fact's metacognitive state.

    The THOUGHT now consumes :
      - content of FACT + ACTION
      - keyword sets of both
      - META-STATE of the fact (confidence, σ, n_uses, hub status)
      - if the fact is a HUB, a small cluster of its manifold neighbours
        (edge-free analog of HeLa's reflective Hebbian distillation)
      - **REDUCE chain** : prior reflections from earlier stages, so the
        new thought *extends* the reasoning rather than restarting it.
        Each thought = fold(prev_thoughts, new fact, new action).
        Without this, depth contributes parallel reflections that the
        next retrieval can't compose — a thought that doesn't enrich the
        next hop's vocabulary is dead weight.

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
    chain_block = ""
    if prev_thoughts:
        chain_lines = [f"  - {t.content}" for t in list(prev_thoughts)[-3:]]
        chain_block = (
            "PRIOR (chain so far) :\n" + "\n".join(chain_lines) + "\n"
        )
    # ACCUMULATED visited elements — the whole point of a THOUGHT is to add
    # context about each VISITED element RELATIVE to the walk : not just the
    # new pair, but what has been gathered so far, tagged by its ROLE (chosen
    # as FACT vs ACTION) and its RELATIVE meta-state (confidence / σ). This is
    # what lets the reflection notice the chain is drifting (all low-relevance,
    # same topic) instead of blindly extending it — and it carries that
    # relative reading into the anchor via the thought's keywords.
    accum_block = ""
    chain_pairs = list(zip(list(prev_facts), list(prev_actions)))[-4:]
    accum_lines: List[str] = []
    for pf, pa in chain_pairs:
        if pf is not None:
            st = _meta_state(pf, population)
            accum_lines.append(
                f"  [FACT conf={st['confidence']} sigma={st['uncertainty']}] "
                f"{pf.content[:80]}")
        if pa is not None:
            accum_lines.append(f"  [ACTION] {pa.content[:80]}")
    if accum_lines:
        accum_block = (
            "VISITED (gathered so far, by role + relative state) :\n"
            + "\n".join(accum_lines) + "\n"
        )
    query_block = f"QUERY : {query}\n" if query else ""
    prompt = (
        "Write ONE short reflection (≤ 15 words) on the NEW fact+action "
        "that EXTENDS the prior chain toward the QUERY. Weigh it against the "
        "VISITED elements and their relative state — if they keep missing the "
        "QUERY, reflect on what kind of evidence is still MISSING. Plain "
        "sentence, no headers, no markdown.\n\n"
        f"{query_block}"
        f"{chain_block}"
        f"{accum_block}"
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
    # WALKED carry-over : the THOUGHT's keyword pool is built ONLY from
    # PREEXISTING keywords of points the walk has already touched —
    # prior fact★, prior action★, prior thoughts — never from the LLM's
    # free reasoning vocabulary. This is what keeps the chain grounded :
    # each thought reorders an established index of walked tokens, never
    # invents abstract relational words that drift retrieval off-source.
    chain_kws: List[str] = []
    for src_list in (
        list(prev_facts)[-3:],
        list(prev_actions)[-3:],
        list(prev_thoughts)[-3:],
    ):
        for p in src_list:
            for kw in p.keywords or []:
                if kw and kw not in seen:
                    seen.add(kw)
                    chain_kws.append(kw)

    # Tokens the reflection emphasises (lowercased word set).
    refl_tokens = {w.strip(".,;:!?\"'()").lower() for w in text.split()}
    # Concrete vocabulary the WALKED points are actually made of.
    walked_content = " ".join(
        p.content for p in
        list(prev_facts)[-3:] + list(prev_actions)[-3:] + list(prev_thoughts)[-3:]
    )
    content_vocab = {
        w.strip(".,;:!?\"'()").lower()
        for w in (fact.content + " " + action.content + " " + walked_content).split()
        if len(w) >= 4
    }

    # 1. Walked keywords (parents + chain) the reflection emphasised come
    #    FIRST, reordered by metacognitive relevance ; remaining parent
    #    keywords next ; chain keywords not emphasised last.
    pool = parent_kws + chain_kws
    emphasised = [k for k in pool if k.lower() in refl_tokens]
    rest_parent = [k for k in parent_kws if k.lower() not in refl_tokens]
    rest_chain = [k for k in chain_kws if k.lower() not in refl_tokens]
    # 2. ENRICH : concrete content tokens (from WALKED content) the
    #    reflection surfaced that are not already keywords. Bounded by
    #    content_vocab membership, so the LLM's abstract relational prose
    #    can never leak in — only words that actually appear in the
    #    walked fact/action/thought content qualify.
    enrich = [
        w for w in refl_tokens
        if w in content_vocab and w not in seen
    ]
    kws = (emphasised + rest_parent + enrich + rest_chain)[:8]
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
    fact_lines: List[str] = []
    chain_lines: List[str] = []
    for s in trajectory.stages:
        f = by_id.get(s.chosen_fact_id or "")
        t = by_id.get(s.thought_id or "")
        if f:
            fact_lines.append(f"  [{s.stage}] {f.content}")
        if t:
            chain_lines.append(f"  [{s.stage}] {t.content}")
    # Expose the THOUGHT chain as a continuous reasoning trace (the
    # reduce-fold the walk produced) AND the evidence facts separately.
    # The chain tells the answerer where the walk's reasoning pointed ;
    # the facts are the raw evidence to ground the final value on.
    chain_block = ""
    if chain_lines:
        chain_block = (
            "REASONING CHAIN (each stage extends the prior) :\n"
            + "\n".join(chain_lines) + "\n\n"
        )
    prompt = (
        "Answer the question by composing across the walk. Use the "
        "REASONING CHAIN as your reasoning thread and the EVIDENCE as "
        "ground.\n\n"
        "Format the answer in the BARE gold style. PICK by the question's "
        "FIRST INTERROGATIVE WORD :\n"
        "  When/date     → absolute date only ('7 May 2023' / 'May 2023' /\n"
        "                  '2022'). Resolve relative dates against the\n"
        "                  evidence turn's [session date].\n"
        "  Where         → place only ('Sweden').\n"
        "  How long      → '<n> <unit>' ('4 years').\n"
        "  What/Which/Who/Why/How → short noun phrase or value :\n"
        "                  'adoption agencies', 'Liberal', 'counselor'.\n"
        "                  ONE word when possible. NO 'Likely yes/no' here.\n"
        "  Would …? / Is …? / Does …? / Was …? (true yes/no inference)\n"
        "        → 'Likely yes, <one short grounded clause>' OR\n"
        "          'Likely no, <one short grounded clause>'. NEVER\n"
        "          'Not mentioned' when the chain gives any signal.\n"
        "  genuinely absent / adversarial → exactly 'Not mentioned'.\n"
        "No preamble, no prose, no markdown — output ONLY the bare value.\n\n"
        f"QUERY : {query}\n\n"
        f"{chain_block}"
        "EVIDENCE :\n" + "\n".join(fact_lines)
    )
    try:
        return memory.llm.generate(prompt, max_tokens=max_tokens).strip()
    except Exception:
        return ""


def _query_year_range(query: str):
    """A YYYY year in the query -> its inclusive (date_from, date_to) range, so
    the event scan can be date-bounded ; (None, None) otherwise."""
    m = re.search(r"\b(?:19|20)\d{2}\b", query or "")
    if not m:
        return None, None
    y = m.group(0)
    return f"{y}-01-01", f"{y}-12-31"


def provisional_answer(query: str, evidence: Sequence[dict],
                       chain: Sequence[str], llm: Any,
                       *, max_tokens: int = 80) -> str:
    """A best-effort answer from the evidence gathered SO FAR — the unit
    the keepup stream re-writes each stage. Same bare-value style as the
    final synthesis ; cheap (one short LLM call). Empty on failure (the
    stream simply keeps the previous snapshot)."""
    if not hasattr(llm, "generate") or not evidence:
        return ""
    ev = "\n".join(f"  - {e.get('content','')}" for e in list(evidence)[:15])
    chain_block = ""
    if chain:
        chain_block = ("REASONING SO FAR :\n"
                       + "\n".join(f"  - {c}" for c in list(chain)[-4:]) + "\n\n")
    prompt = (
        "Answer the QUERY from the evidence gathered so far. This is a "
        "PROVISIONAL answer that may be refined as more evidence arrives. "
        "Reply with the bare value in the gold style (date / place / short "
        "noun phrase / 'Likely yes|no, <clause>' / 'Not mentioned'). No "
        "preamble, no prose.\n\n"
        f"QUERY : {query}\n\n"
        f"{chain_block}"
        f"EVIDENCE :\n{ev}"
    )
    try:
        return (llm.generate(prompt, max_tokens=max_tokens) or "").strip()
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
        prev_thoughts=(),
        query=query,
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
    thought_chain: List[Point] = [thought]
    fact_chain: List[Point] = [fact_star]
    action_chain: List[Point] = [action_star]

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
            prev_thoughts=thought_chain,
            prev_facts=fact_chain,
            prev_actions=action_chain,
            query=query,
        )
        if thought_next is None:
            return traj
        memory.points.append(thought_next)
        rec.thought_id = thought_next.id
        rec.generated_thought = True
        traj.generated_point_ids.append(thought_next.id)
        prev_action = action_next
        cur_thought = thought_next
        thought_chain.append(thought_next)
        fact_chain.append(fact_star_next)
        action_chain.append(action_next)

    return traj


# ---------------------------------------------------------------------------
# Skill / task mode : tools-in-the-walk depth gate + skill-JSON synthesis
# ---------------------------------------------------------------------------

def is_tool_point(p: Point) -> bool:
    """A point that represents a tool / skill ingredient — tagged `tool`
    or `skill`, or a crystallized tool ACTION."""
    t = set(getattr(p, "tags", []) or [])
    return bool(t & {"tool", "skill", "tool_workflow"})


def enough_tools_for_workflow(query: str, tools: Sequence[Point],
                              llm: Any) -> bool:
    """Task-mode DEPTH gate : do the gathered TOOL nodes suffice to
    elaborate the COMPLETE workflow that solves `query`? Returns True to
    stop the walk, False to keep collecting.

    Fails CLOSED (False) on any error / empty tool set, so a transient LLM
    hiccup only makes the walk gather MORE tools, never stop short. No
    tools yet → trivially not enough.
    """
    if not tools or not hasattr(llm, "generate"):
        return False
    listing = "\n".join(f"- {p.content}" for p in list(tools)[-20:])
    prompt = (
        "We are assembling a COMPLETE workflow to solve a TASK from the "
        "tools gathered so far. Answer with exactly YES if the tools are "
        "sufficient to build the whole workflow end-to-end, or NO if a "
        "step is still missing.\n\n"
        f"TASK : {query}\n"
        f"TOOLS GATHERED :\n{listing}"
    )
    try:
        raw = (llm.generate(prompt, max_tokens=8) or "").strip().lower()
    except Exception:
        return False
    return raw.startswith("yes")


def synthesize_skill_json(query: str, tools: Sequence[Point],
                          llm: Any, *, name_hint: str = "") -> Dict[str, Any]:
    """Produce the NAMED skill folder — `{"name": str, "tree": {...}}` —
    from the gathered tools. `tree` is the nested theoretical directory the
    skill ingestion (Memory.ingest_skill) consumes. Deterministic fallback
    on any failure : a flat tree of the tool contents under a slugged name,
    so a skill is ALWAYS produced and ingestable.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", (name_hint or query).lower()).strip("_")[:40]
    slug = slug or "skill"
    fallback = {
        "name": slug,
        "tree": {f"step_{i+1}.txt": (p.content or "")[:160]
                 for i, p in enumerate(list(tools)[:12])},
    }
    if not hasattr(llm, "generate"):
        return fallback
    listing = "\n".join(f"- {p.content}" for p in list(tools)[:20])
    prompt = (
        "Turn the gathered tools into a SKILL FOLDER that solves the task. "
        "Output STRICT JSON: an object with \"name\" (a short snake_case "
        "skill name) and \"tree\" (a nested directory: a folder maps a name "
        "to a sub-object, a file maps a name to a one-line description). "
        "No prose, JSON only.\n\n"
        f"TASK : {query}\n"
        f"TOOLS :\n{listing}"
    )
    try:
        raw = (llm.generate(prompt, max_tokens=400) or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return fallback
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict) or not isinstance(obj.get("tree"), dict):
            return fallback
        obj.setdefault("name", slug)
        if not isinstance(obj["name"], str) or not obj["name"].strip():
            obj["name"] = slug
        return obj
    except Exception:
        return fallback


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
        task_mode: bool = False,
        section_filter: Optional[set] = None,
        restrict_ids: Optional[set] = None,
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
        # SKILL / TASK mode. When task_mode is on, the walk is assembling a
        # workflow from TOOL nodes (skill/tool-tagged FACTs retrieved
        # alongside semantic facts) rather than answering a question : its
        # depth gate becomes "have we collected enough tools to elaborate
        # the complete workflow?" (an LLM check per depth). `section_filter`
        # is the double-query pre-filter — a set of tags (session:/user:/
        # name:) whose section gets a retrieval rank boost.
        self.task_mode = task_mode
        self.section_filter = section_filter
        # HARD search-pool restriction (scoped walk) : set of point ids the
        # walk may retrieve FACTs from. None = the whole memory.
        self.restrict_ids = restrict_ids
        # Tools gathered across the walk (skill/tool-tagged), the workflow
        # ingredients in task mode.
        self._tools_collected: List[Point] = []
        # Per-walk scratch for generated points when commit=False.
        self._local_points: List[Point] = []
        self.t_now = (
            t_now if t_now is not None
            else (memory._now() if hasattr(memory, "_now") else 0.0)
        )

        self._enc = memory.encoder
        self._extr = memory.extractor
        self._llm = memory.llm
        # HyDE — pre-generate one hypothetical answer to the original
        # question. Used as a THIRD RRF signal in every stage's retrieval
        # (see nearest_facts_with_fallback). One LLM call per walk (cached).
        self._use_hyde = _hyde_enabled()
        self._hyde_passage: str = ""
        if self._use_hyde and self._llm is not None:
            try:
                self._hyde_passage = hyde_passage(self.query, self._llm)
            except Exception:
                self._hyde_passage = ""

        # SLICE C — typed query-alignment anchor, built ONCE per walk and
        # fed to every stage's retrieval (no per-stage cost beyond scoring).
        # Uses the keyword extractor + any entity extractor + IDF over the
        # corpus ; entity extraction is skipped when no extractor is wired
        # (then it degrades to keyword/raw-token alignment, no extra LLM).
        self._query_anchor = None
        try:
            from metacog.query_anchor import build_query_anchor
            self._query_anchor = build_query_anchor(
                self.query,
                entity_extractor=getattr(self.memory, "entity_extractor", None),
                keyword_extractor=self._extr,
                encoder=self._enc,
                corpus_texts=[p.content for p in self.memory.points],
            )
        except Exception:
            self._query_anchor = None

        # Stage-0 seed embedding from the query keywords. Position-
        # weighted (1/(i+1) decay) so the top keyword drives the vector.
        q_kws = self._extr.extract(query, n=8) if self._extr else []
        self._query_keywords = q_kws
        pwe0 = position_weighted_keyword_embedding(q_kws, self._enc) if q_kws else None
        self._cur_emb = pwe0 if pwe0 is not None else tuple(self._enc.encode(query))

        self._fact_ids_cum: List[str] = []
        # Spike-and-Path Chasles : track the previous stage's fact_star id
        # so we can call record_hop on each multi-hop transition. Only
        # used when commit=True (benchmark walks stay reproducible).
        self._prev_fact_star_id: Optional[str] = None
        self._visited_fact_ids: set = set()
        self._visited_action_ids: set = set()
        self._prev_action: Optional[Point] = None
        self._cur_thought: Optional[Point] = None
        # REDUCE chain across stages : prior fact★ / action★ / thought
        # are all preserved so the next thought's keyword pool is built
        # from PREEXISTING tokens of walked points only — no abstract
        # vocabulary leaks in. Each thought reorders the walked index.
        # Bounded slice (~last 3) passed to the LLM to cap prompt growth.
        self._thought_chain: List[Point] = []
        self._fact_star_chain: List[Point] = []
        self._action_star_chain: List[Point] = []
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
        # σ-propagation state : cumulative combined uncertainty over the
        # evidence chain's hops (fact★→fact★, GUM quadrature). Accumulated
        # at the end of each stage. See the depth block in step() for how
        # the FLOOR / CAP / gold-gate use it.
        self._sigma_path: float = 0.0
        self._prev_fact_star_emb: Optional[tuple] = None
        # DEPTH is governed by UNCERTAINTY PROPAGATION (GUM quadrature),
        # not by a fixed stage count and not by an LLM "can I answer" gate.
        # Both parameters are emergent / structural (no magic numbers) :
        #  - a hard FLOOR of `_MIN_STAGES` hops always runs before σ may
        #    terminate (always explore the multi-hop neighbourhood) ;
        #  - the σ-cutoff (median + std of stage-0 pairwise distances) is
        #    the CAP : once propagated uncertainty over the fact★→fact★
        #    evidence chain exceeds the manifold's local resolution the
        #    chain has wandered out of the coherent neighbourhood, so the
        #    walk stops. Gold-retrieval extension is intrinsic to σ (a
        #    coherent trail = small hops = slow σ = deep walk), so no
        #    separate relevance gate is needed. `n_stages` is only a
        #    generous safety bound.
        # `_last_on_target` is kept only to expose `out.drifted` downstream.
        self._last_on_target: int = 1
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
        # HARD restriction (scoped walk) : only points in `restrict_ids`
        # (plus this walk's own generated scaffolding) are searchable. Used
        # by the scoped→knowledge-base cascade : the first walk runs over a
        # filtered set (a discussion / session) before any global search.
        rids = getattr(self, "restrict_ids", None)
        if rids:
            gen = set(self._generated_ids)
            base = [p for p in base
                    if p.id in rids or p.id in gen
                    or p.kind != PointKind.FACT or p.id.startswith("entity_")]
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

    def _composable_evidence(self) -> List[dict]:
        """Top-`_MAX_EVIDENCE` on-target facts as the COMPOSABLE evidence
        for the agent / synthesis — bounded so a bloated MAP-REDUCE set
        (CoN sometimes labels dozens of turns relevant) doesn't drown the
        answer prompt or blow up the agent's context tokens.

        Ranking is KEYWORD-ORIENTED and deterministic (no LLM) :
          1. relevance label priority  (relevant > contradicts > partial)
          2. keyword overlap with the walk's own vocabulary — the union of
             the query keywords and the thought-chain keywords. Using the
             CHAIN vocabulary (not the raw query) keeps vocabulary-distant
             gold the walk bridged to, instead of demoting it.
        Full recall is unaffected : `fact_ids_cumulative` still lists every
        fact seen ; this only bounds what is COMPOSED over.
        """
        cum = list(self._relevant_cum)

        # Vocab-gap fallback : when the CoN labelled almost nothing relevant,
        # promote the top-`_FALLBACK_K` unlabelled retrieved facts (re-ranked
        # by cosine similarity to the query embedding) as `retrieved`, so the
        # answerer sees something on-topic when the seed was off-register
        # (cat1 "What did X research?" with evidence "Researching adoption
        # agencies" — the gold sits in fact_ids_cumulative but the academic-
        # register walk never CoN-labelled it relevant).
        seen_ids = {p.id for p in cum}
        if (len(cum) < _FALLBACK_RELEVANT_GATE
                and len(self._fact_ids_cum) > _FALLBACK_K
                and getattr(self.memory, "encoder", None) is not None
                and (self.query or "").strip()):
            by_id = {p.id: p for p in self.memory.points}
            candidates: List[Point] = []
            for fid in self._fact_ids_cum:
                if fid in seen_ids or fid not in by_id:
                    continue
                p = by_id[fid]
                if p.kind != PointKind.FACT \
                        or p.id.startswith(("entity_", "atom_")):
                    continue
                candidates.append(p)
            if candidates:
                try:
                    import numpy as np
                    q_emb = np.asarray(
                        self.memory.encoder.encode(self.query), dtype=float)
                    q_norm = float(np.linalg.norm(q_emb)) + 1e-12
                    ranked: List[Tuple[float, Point]] = []
                    for p in candidates:
                        e = p.embedding_orig
                        if e is None:
                            continue
                        pe = np.asarray(e, dtype=float)
                        denom = q_norm * (float(np.linalg.norm(pe)) + 1e-12)
                        sim = float(np.dot(q_emb, pe) / denom)
                        ranked.append((sim, p))
                    ranked.sort(reverse=True, key=lambda x: x[0])
                    for _, p in ranked[:_FALLBACK_K]:
                        self._relevant_label.setdefault(p.id, "retrieved")
                        cum.append(p)
                        seen_ids.add(p.id)
                except Exception:
                    pass

        if len(cum) <= _MAX_EVIDENCE:
            return [
                {"id": p.id, "content": p.content,
                 "relevance": self._relevant_label.get(p.id, "relevant")}
                for p in cum
            ]
        # Walk vocabulary : query kws ∪ chain-thought kws (lowercased).
        vocab: set = {k.lower() for k in (self._query_keywords or [])}
        for t in self._thought_chain:
            for k in (t.keywords or []):
                vocab.add(k.lower())
        _label_rank = {"relevant": 0, "contradicts": 1, "partial": 2}

        def _score(p):
            label = self._relevant_label.get(p.id, "relevant")
            pk = {k.lower() for k in (p.keywords or [])}
            overlap = len(pk & vocab)
            return (_label_rank.get(label, 3), -overlap)

        ranked = sorted(cum, key=_score)[:_MAX_EVIDENCE]
        return [
            {"id": p.id, "content": p.content,
             "relevance": self._relevant_label.get(p.id, "relevant")}
            for p in ranked
        ]

    def neighbor_possibilities(
        self,
        *,
        window: int = _NEIGHBOUR_WINDOW,
        cap: int = _MAX_NEIGHBOURS,
    ) -> List[dict]:
        """Sequence-adjacent turns of everything the walk retrieved — the
        "multiple possibilities" offered when the gold is likely just out of
        the seed's reach. Delegates to the module-level `lineage_neighbors`
        over the walk's own retrieved set (`_fact_ids_cum`)."""
        return lineage_neighbors(
            self._fact_ids_cum, self.memory.points,
            window=window, cap=cap,
        )

    def grounding_is_thin(self) -> bool:
        """True when the walk collected fewer than `_NEIGHBOUR_GATE` on-target
        facts — the proxy for "the gold is likely out of the seed's reach",
        used to decide whether to offer the lineage neighbours."""
        return len(self._relevant_cum) < _NEIGHBOUR_GATE

    def _query_covered(self) -> bool:
        """True when every query keyword appears in the gathered evidence's
        keywords or content (case-insensitive, stem-tolerant prefix match).
        Deterministic, no LLM. Used as a metacognitive depth-stop only once
        the relevant set is substantial."""
        q = [k.lower() for k in (self._query_keywords or []) if len(k) >= 3]
        if not q:
            return False
        bag: set = set()
        for p in self._relevant_cum:
            for k in (p.keywords or []):
                bag.add(k.lower())
            for w in (p.content or "").lower().split():
                w = w.strip(".,;:!?\"'()")
                if len(w) >= 3:
                    bag.add(w)
        bag_list = list(bag)
        for term in q:
            # exact, or prefix either way (cheap stem tolerance)
            if term in bag:
                continue
            if any(term[:5] == b[:5] and (term in b or b in term)
                   for b in bag_list):
                continue
            return False
        return True

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
            # Stage 1+ : anchor on query keywords (position-weighted, kept
            # dominant) + the LATEST thought's keywords only. The chain's
            # cumulative reasoning is already folded INSIDE that latest
            # thought (meta_thought consumes prev_thoughts when generating
            # it), so we don't pile multiple thoughts' kws into the seed —
            # that would drown the query and pull retrieval away from gold
            # vocabulary on multi-hop comparison questions.
            thought_kws = self._cur_thought.keywords if self._cur_thought else []
            anchored_kws = list(self._query_keywords) + list(thought_kws)
            seed_query = " ".join(anchored_kws) if anchored_kws else self.query
            pwe = position_weighted_keyword_embedding(anchored_kws, self._enc)
            seed_emb = pwe if pwe is not None else self._cur_emb

        # ── DEPTH = UNCERTAINTY PROPAGATION ──────────────────────────────
        # The walk's depth is governed by GUM-style propagation of
        # uncertainty (https://en.wikipedia.org/wiki/Propagation_of_
        # uncertainty) over the EVIDENCE CHAIN, NOT by a fixed stage count.
        # `_sigma_path` is accumulated at the END of each stage as the
        # quadrature sum of the hop uncertainties between consecutive
        # chosen facts : σ_path = √(Σ (1 − cos(fact★ₖ₋₁, fact★ₖ))²)
        # (uncertainty.py, GUM 1995). Seed-to-seed drift is NOT used — the
        # seed is re-anchored on the query each stage so it barely moves ;
        # the real uncertainty is how far the EVIDENCE travels.
        #
        #   FLOOR  — the first `_MIN_STAGES` hops ALWAYS run ; σ may not
        #            terminate the walk before then (always explore the
        #            multi-hop neighbourhood).
        #   CAP    — once σ_path exceeds the walk-local emergent threshold
        #            (median + std of stage-0 pairwise distances = the
        #            manifold's local resolution), the evidence chain has
        #            travelled beyond the coherent neighbourhood and the
        #            walk stops. There is NO fixed maximum stage count.
        #
        # GOLD-RETRIEVAL IS INTRINSIC TO σ, NOT A SEPARATE GATE. Because σ
        # is the quadrature sum of fact★→fact★ hops, a coherent evidence
        # chain (the gold trail) has SMALL consecutive hops, so σ grows
        # slowly and the walk goes deep on its own ; wandering into
        # unrelated turns adds a LARGE hop, so σ trips quickly. The earlier
        # Chain-of-Note "still finding new relevant" gate was removed : it
        # made depth hostage to a stochastic, bimodal LLM relevance signal
        # (one run labels every turn relevant → walk never plateaus → 16
        # stages / 80 facts ; the next labels none relevant → walk
        # strangled at the floor). The σ measure already encodes
        # "extend on a coherent trail, cap when wandering" — deterministically.
        if (self._stage_idx >= _MIN_STAGES
                and self._walk_sigma_cutoff is not None
                and self._sigma_path > self._walk_sigma_cutoff):
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

        # ── METACOGNITIVE STOP : query-keyword COVERAGE ──────────────────
        # Triggered only once the evidence is SUBSTANTIAL (past the floor
        # and a non-trivial relevant set gathered) — a cheap, deterministic,
        # KEYWORD-ORIENTED "is more search necessary?" check, no LLM call.
        # If every keyword of the query is already covered by the keywords /
        # content of the gathered evidence, the walk has turns bearing on
        # every facet of the question — continuing only burns tokens, so
        # stop. Vocabulary-gap questions (cat3 : "political leaning" vs
        # "LGBTQ advocacy") never reach full coverage, so this never cuts
        # them short — σ governs those.
        if (self._stage_idx >= _MIN_STAGES
                and len(self._relevant_cum) >= _MIN_STAGES
                and self._query_keywords
                and self._query_covered()):
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

        # TASK-MODE depth gate : once past the floor, stop as soon as the
        # tools gathered suffice to elaborate the complete workflow (LLM
        # check per depth). Fails closed (keep collecting) on any error.
        if (self.task_mode
                and self._stage_idx >= _MIN_STAGES
                and self._tools_collected
                and enough_tools_for_workflow(
                    self.query, self._tools_collected, self._llm)):
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

        # Refresh the anchor's RELATIVE half from the latest thought — the
        # "pensée distancielle" that lifts surface facts into the answer
        # register. From stage 1 on, the thought's abstraction ("overweight")
        # is ADDED to the input anchor so Slice C can rank the oblique
        # evidence ("fingers too big") above the ambient topic (stress).
        if self._query_anchor is not None and self._cur_thought is not None:
            try:
                self._query_anchor.with_relative(
                    self._cur_thought.keywords, self._enc)
            except Exception:
                pass

        pts = self._all_points()
        facts = nearest_facts_with_fallback(
            seed_emb, seed_query, pts,
            self.facts_per_stage, self.t_now,
            exclude_ids=self._visited_fact_ids,
            encoder=self._enc, extractor=self._extr,
            anchor_query=self.query if seed_query != self.query else None,
            hyde_passage_text=self._hyde_passage or None,
            section_filter=self.section_filter,
            query_anchor=self._query_anchor,
        )

        # Feed the LATERAL co-retrieval ledger : this ranked result, under
        # this query direction, is one diversity-weighted vote about which
        # facts keep surfacing together. No-op unless the memory has lateral
        # collision enabled. Recorded only for committed (live) walks so
        # isolated benchmark walks stay side-effect-free.
        if self.commit and hasattr(self.memory, "record_retrieval"):
            self.memory.record_retrieval([f.id for f in facts], seed_emb)

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
            # Pass the FULL thought chain content (concatenated last 3)
            # so the relevance reading reflects what the walk has already
            # reasoned, not just the last reflection in isolation.
            chain_text = " | ".join(
                t.content for t in self._thought_chain[-3:]
            ) or (self._cur_thought.content if self._cur_thought else None)
            relevance = chain_of_note(
                self.query, facts, self._llm,
                collected=self._relevant_cum,
                current_thought=chain_text,
            )
        rel_by_id = {f.id: relevance[i] for i, f in enumerate(facts)}

        # LATERAL EXPANSION (on-demand) : a lateral keeper stands in for a
        # folded group in the kNN. Once it is judged ON-TARGET, expand it to
        # its absorbed children so the agent reasons over the exact folded
        # detail (e.g. the specific evidence turn), not just the
        # representative. Off-target keepers are never expanded, so the kNN
        # stays deduplicated and no bloat leaks back in. Expanded children
        # inherit the keeper's relevance label.
        from metacog.lateral import lateral_children
        by_id_all = {p.id: p for p in pts}
        expanded: List[Point] = []
        exp_label: dict = {}
        for i, f in enumerate(facts):
            if relevance[i] == "irrelevant":
                continue
            for child in lateral_children(f, by_id_all):
                if child.id in rel_by_id or child.id in exp_label:
                    continue
                expanded.append(child)
                exp_label[child.id] = relevance[i]
        if expanded:
            facts = list(facts) + expanded
            relevance = list(relevance) + [exp_label[c.id] for c in expanded]
            rel_by_id.update(exp_label)

        # REDUCE — fold this stage's on-target facts into the persistent
        # accumulator (dedup, order-preserving). Bridging evidence from
        # état -1 survives into état k ; going deeper never loses it.
        # (Depth is no longer gated on this count — σ over the evidence
        # chain governs the cap ; see the depth block above.)
        for i, f in enumerate(facts):
            if relevance[i] != "irrelevant" and f.id not in self._relevant_ids:
                self._relevant_cum.append(f)
                self._relevant_ids.add(f.id)
                self._relevant_label[f.id] = relevance[i]

        # TASK MODE : collect the tool/skill nodes surfaced this stage —
        # the workflow ingredients. Gold in task mode is a tool fact as
        # readily as a semantic fact, so they ride the same retrieval.
        if self.task_mode:
            have = {p.id for p in self._tools_collected}
            for f in facts:
                if f.id not in have and is_tool_point(f):
                    self._tools_collected.append(f)

        # The subset the reasoning trusts : everything except clearly
        # off-target turns. "contradicts" is kept (the THOUGHT must see
        # the conflict). Fall back to ALL facts if the note left nothing.
        focus_facts = [
            f for i, f in enumerate(facts)
            if relevance[i] != "irrelevant"
        ] or list(facts)

        # META-COGNITION : if the focused facts reveal the input relates to an
        # EVENT and the request refers to an exhaustive set, the ACTION launches
        # the NON-kNN filtered scan over that event and folds it into evidence —
        # a targeted filtered KB query instead of a parallel one.
        self._maybe_event_scan(focus_facts)

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
                # SKILL crystallization signal : this re-derived action,
                # scoped by the query + grounding facts, feeds the
                # recurrence ledger. When the same action recurs across
                # diverse queries it crystallizes into a persistent TOOL
                # node. Only on committed (live) walks.
                if self.commit and hasattr(self.memory, "record_action_generation"):
                    self.memory.record_action_generation(
                        generated, seed_emb, self.query, focus_facts,
                    )

        # fact* is chosen among the FOCUSED facts (Chain-of-Note has
        # filtered the off-target turns) so the THOUGHT is seeded by a
        # fact that actually bears on the query.
        fact_star = least_uncertain(focus_facts)
        action_star = actions[0] if actions else None

        # ── Accumulate the evidence-chain uncertainty (GUM quadrature) ──
        # The hop signal measures CHAIN COHERENCE, not the volatility of
        # the `least_uncertain` choice. Earlier we tried hop =
        # 1 − cos(prev_fact★, fact★) ; that made σ track the seismic jumps
        # of `least_uncertain` from one stage's most-certain fact to the
        # next, which is roughly constant (~0.25/hop) regardless of whether
        # the chain stayed coherent or wandered — so σ never adapted.
        #
        # Better signal : the hop is the SMALLEST distance from the previous
        # fact★ to ANY fact this stage retrieved (focus subset). Reading :
        #   small min-distance → there IS a fact near where the chain just
        #     stood ; the trail can extend smoothly → low hop → σ grows
        #     slowly → walk goes deep on coherent trails (cat3 included).
        #   large min-distance → nothing this stage retrieved is close to
        #     the chain's last position ; the walk has WANDERED off the
        #     coherent neighbourhood → big hop → σ trips fast → cap fires.
        # The choice of `least_uncertain` no longer dominates the signal ;
        # only the geometry of what was REACHABLE this stage does. Pure
        # COMPUTATION, no LLM-relevance dependency.
        if fact_star is not None:
            e_cur = effective_keyword_embedding(fact_star, self.t_now)
            if e_cur is not None:
                if self._prev_fact_star_emb is not None and facts:
                    min_dist = None
                    for f in facts:
                        e_f = effective_keyword_embedding(f, self.t_now)
                        if e_f is None:
                            continue
                        d = max(0.0, 1.0 - cosine(self._prev_fact_star_emb, e_f))
                        if min_dist is None or d < min_dist:
                            min_dist = d
                    if min_dist is not None:
                        self._sigma_path = math.sqrt(
                            self._sigma_path ** 2 + min_dist ** 2
                        )
                self._prev_fact_star_emb = e_cur

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
                # Spike-and-Path : record the multi-hop transition. The
                # very first stage has no predecessor and is skipped.
                from metacog.spike import record_hop
                if self._prev_fact_star_id is not None:
                    prev_fact = next(
                        (p for p in self.memory.points
                         if p.id == self._prev_fact_star_id),
                        None,
                    )
                    if prev_fact is not None:
                        record_hop(prev_fact, fact_star, self.memory)
                self._prev_fact_star_id = fact_star.id
                # WORKFLOW hop : also record the ACTION -> ACTION transition
                # (the chosen action of the previous stage led to this one).
                # This charges the ACTION chain so the in-depth Chasles
                # compression can later shorten a recurring multi-step
                # workflow (incl. tool nodes, which are kind=ACTION) into a
                # single optimized step. self._prev_action still holds the
                # previous stage's action here (updated at end of step).
                if (self._prev_action is not None
                        and self._prev_action.id != action_star.id):
                    record_hop(self._prev_action, action_star, self.memory)

            thought = meta_thought(
                fact_star, action_star, pts,
                self._llm, self._enc, self._extr, self.t_now,
                prev_thoughts=self._thought_chain,
                prev_facts=self._fact_star_chain,
                prev_actions=self._action_star_chain,
                query=self.query,
            )
            if thought is not None:
                self._add_generated(thought)
                gen_thought = True
                self._cur_thought = thought
                self._thought_chain.append(thought)
                self._fact_star_chain.append(fact_star)
                self._action_star_chain.append(action_star)
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
            relevant_collected=self._composable_evidence(),
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
        # Record progress for the NEXT stage's progress-gated σ-stop.
        self._last_on_target = on_target

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

    # ----------------------------------------------------------------
    # Keepup : organic streaming. Emit a provisional answer from the very
    # first walk and re-write it every stage until the depth-stop validates
    # it — the last snapshot IS the final answer (no separate final pass).
    # ----------------------------------------------------------------

    def _maybe_event_scan(self, focus_facts):
        """Meta-cognition + ACTION : when the focused facts show the input relates
        to an EVENT and the request refers to an exhaustive SET, launch the
        NON-kNN filtered scan (Memory.filter_list) over that event — optionally
        date-bounded by a year in the query — and FOLD the result into the walk's
        evidence. This is the targeted filtered knowledge-base query that
        replaces a separate parallel one.

        Gated to RECALL intent : a focused query wins on the walk alone (the
        walk-vs-event finding), so the scan only fires for set-style requests.
        Once per event ; fully failure-safe / no-op without the primitive."""
        mem = self.memory
        if not hasattr(mem, "filter_list"):
            return
        try:
            from metacog.enumeration import is_enumeration_query
            if not is_enumeration_query(self.query):
                return
        except Exception:
            return
        if not hasattr(self, "_scanned_events"):
            self._scanned_events = set()
        eid = None
        for f in focus_facts or ():
            for t in getattr(f, "tags", ()) or ():
                if t.startswith("event:in:"):
                    eid = t.split(":", 2)[2]
                    break
            if eid:
                break
        if not eid or eid in self._scanned_events:
            return
        self._scanned_events.add(eid)
        df, dt = _query_year_range(self.query)
        try:
            pts = mem.filter_list(event_id=eid, date_from=df, date_to=dt)
        except Exception:
            return
        for p in pts:
            if p.id not in self._relevant_ids:
                self._relevant_cum.append(p)
                self._relevant_ids.add(p.id)
                self._relevant_label[p.id] = "relevant"
            if p.id not in self._visited_fact_ids:
                self._fact_ids_cum.append(p.id)
                self._visited_fact_ids.add(p.id)
        if pts and hasattr(mem, "bag_add"):
            try:
                mem.bag_add([p.id for p in pts], bag=f"event:{eid}",
                            description=f"Exhaustive filtered scan of event "
                                        f"{eid}.",
                            schema={"kind": "event_cluster", "event_id": eid})
            except Exception:
                pass

    def keepup(self):
        """Generator yielding one snapshot per stage :

            {"stage", "thought", "answer", "evidence_count", "done",
             "sigma_path"}

        The `answer` is a PROVISIONAL answer synthesised from the evidence
        gathered up to this stage ; a consumer re-writes the displayed
        message with each yield. Generation is therefore CONTINUOUS — by
        the time the walk's σ / coverage depth-gate fires (done=True) the
        provisional answer has already converged, so there is nothing extra
        to compute : the depth-validation and the answer are produced
        together. If walk k+1 only confirms walk k, the answer was already
        right and the user saw it k stages early.

        Cheap-by-design : a provisional answer is only (re)synthesised once
        the floor is reached and the evidence set actually grew, so easy
        questions emit one or two snapshots, not n.
        """
        last_answer = ""
        last_ev_count = -1
        while not self._done:
            out = self.step()
            ev = self._composable_evidence()
            chain = [t.content for t in self._thought_chain]
            # (Re)synthesise past the floor when evidence changed, AND
            # always on the terminal stage if we have evidence but no
            # answer yet (a walk that stops before the floor must still
            # emit a final answer — keepup must never end empty-handed).
            need = (out.stage + 1 >= _MIN_STAGES and len(ev) != last_ev_count)
            if not need and out.done and ev and not last_answer:
                need = True
            if ev and need:
                ans = provisional_answer(self.query, ev, chain, self._llm)
                if ans:
                    last_answer = ans
                last_ev_count = len(ev)
            yield {
                "stage": out.stage,
                "thought": (out.thought or {}).get("content")
                if isinstance(out.thought, dict) else None,
                "answer": last_answer,
                "evidence_count": len(ev),
                "done": bool(out.done),
                "sigma_path": round(out.sigma_path, 4),
            }
            if out.done:
                break


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
