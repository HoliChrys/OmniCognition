"""
MetaCog-Mem — MEMORY SKILLS (capability crystallization).

The third application of the same emergent law that drives lateral
collision and spike-Chasles compression :

    lateral collision : facts that co-recur under diverse queries
                        -> one keeper        (dedup RETRIEVAL)
    spike / Chasles    : hops that recur along a path
                        -> one shortcut      (dedup REASONING PATH)
    memory skills      : ACTIONS the system keeps re-deriving to answer
                        diverse queries
                        -> one persistent TOOL (dedup CAPABILITY)

When a user asks for something the system does not yet have a tool for
(e.g. "search the scientific articles about X"), the meta-walk THINKS :
it generates an ACTION describing HOW to do it. Today that thinking is
thrown away after the turn. A SKILL captures it : once the SAME kind of
action has been re-derived across enough DIVERSE queries, the system
crystallizes it into a named, persistent capability — semantically
SCOPED by the facts that grounded it, the user queries that triggered
it, and the system's own metacognitive deductions (the action text).

Next time a query falls in that skill's semantic scope, `match_skill`
returns it directly : the agent can invoke the cached capability instead
of re-running the whole think-from-scratch phase.

This module builds the foundation — recurrence tracking, the emergent
firing threshold, the candidate detector, LLM synthesis of the skill
descriptor, and the semantic fast-path matcher. The skill `body` is a
prompt/approach template (safe : no arbitrary code is executed here) ;
binding a body to an executor or to generated code is the guarded next
step, deliberately left to the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from metacog.epistemic import Point, PointKind
from metacog.geometry import cosine

Vector = Tuple[float, ...]


# ---------------------------------------------------------------------------
# A tool is a NORMAL node, not a special object
# ---------------------------------------------------------------------------
#
# A crystallized capability is a Point in the same manifold as everything
# else — kind=ACTION (the executable kind, which replaced the old "skill")
# — distinguished only by tags :
#
#   "tool"            : this node is an invokable capability
#   "tool_<genre>"    : its genre — tool_code / tool_command / tool_api
#   <context tags>    : the grounding context it was distilled from
#
# and SCOPED by its KEYWORDS (when it applies) exactly like any other node.
# Being a normal node means a tool is auto-persisted with the cloud,
# retrieved by the same kNN/keyword machinery, and subject to lateral
# collision / spike like everything else — no parallel storage, no special
# case. The fast-path "do I already know how to do this?" is just ordinary
# retrieval restricted to tool-tagged nodes.

TOOL_TAG = "tool"


def _genre_tag(kind: str) -> str:
    return f"tool_{kind}" if kind in ("code", "command", "api") else "tool_command"


def is_tool(p: Point) -> bool:
    return TOOL_TAG in (p.tags or [])


# ---------------------------------------------------------------------------
# Recurrence ledger (diversity-weighted, like the lateral ledger)
# ---------------------------------------------------------------------------


@dataclass
class _ActionTrace:
    """Accumulated evidence for one action SIGNATURE (a recurring task)."""

    weight: float = 0.0                       # diversity-weighted recurrence
    dirs: List[Vector] = field(default_factory=list)   # query directions seen
    query_samples: List[str] = field(default_factory=list)
    action_samples: List[str] = field(default_factory=list)
    scope_kw: Dict[str, int] = field(default_factory=dict)  # grounding kw -> count
    n: int = 0


@dataclass
class SkillLedger:
    """Tracks, per action-signature, how often and how DIVERSELY the system
    has re-derived that action — plus the queries and grounding keywords
    that framed it. The diversity weighting (shared with lateral collision)
    is the point : an action re-derived only for near-identical queries is
    over-specialised, not a reusable skill ; one re-derived across many
    DISTANT queries is a genuine capability."""

    traces: Dict[str, _ActionTrace] = field(default_factory=dict)
    total_weight: float = 0.0
    n_events: int = 0


# Operational bounds (not statistical thresholds — firing is emergent).
_NOVELTY_MIN = 0.15     # a re-derivation whose query repeats a counted
                        # direction adds < this : ignored.
_MAX_DIRS = 8           # cap on stored query directions per signature.
_MAX_SAMPLES = 4        # cap on stored sample texts per signature.
_MIN_SIGNATURES = 4     # Poisson baseline needs a few distinct signatures.
_SIG_KEYWORDS = 3       # an action signature = its top-N keywords.


def _signature(action: Point) -> str:
    """Stable key for an action's TASK identity : its top keywords (the
    verb/object of the imperative), order-independent. Actions that
    describe the same task collapse to the same signature."""
    kws = [k.lower() for k in (action.keywords or []) if k]
    if not kws:
        kws = [w.lower() for w in (action.content or "").split()][:_SIG_KEYWORDS]
    return "|".join(sorted(kws[:_SIG_KEYWORDS]))


def record_action(
    ledger: SkillLedger,
    action: Point,
    query_emb: Optional[Vector],
    query_text: str,
    fact_keywords: Sequence[str],
    *,
    novelty_min: float = _NOVELTY_MIN,
    max_dirs: int = _MAX_DIRS,
) -> None:
    """Register one re-derivation of an action. The recurrence weight is
    discounted by how redundant `query_emb` is with the query directions
    already counted for this signature (so only DIVERSE re-derivations
    build a skill). Grounding keywords accumulate the skill's scope."""
    sig = _signature(action)
    if not sig:
        return
    ledger.n_events += 1
    tr = ledger.traces.get(sig)
    if tr is None:
        tr = _ActionTrace()
        ledger.traces[sig] = tr
    if query_emb is None or not tr.dirs:
        novelty = 1.0
    else:
        novelty = 1.0 - max(cosine(query_emb, d) for d in tr.dirs)
    if novelty < novelty_min:
        # Still record the grounding/sample (cheap) but no weight.
        novelty = 0.0
    tr.n += 1
    tr.weight += novelty
    ledger.total_weight += novelty
    if novelty > 0.0 and query_emb is not None and len(tr.dirs) < max_dirs:
        tr.dirs.append(tuple(query_emb))
    if query_text and len(tr.query_samples) < _MAX_SAMPLES:
        if query_text not in tr.query_samples:
            tr.query_samples.append(query_text)
    if action.content and len(tr.action_samples) < _MAX_SAMPLES:
        if action.content not in tr.action_samples:
            tr.action_samples.append(action.content)
    for k in fact_keywords:
        kl = (k or "").lower()
        if kl:
            tr.scope_kw[kl] = tr.scope_kw.get(kl, 0) + 1


# ---------------------------------------------------------------------------
# Emergent threshold (Poisson — same law as spike / lateral)
# ---------------------------------------------------------------------------


def skill_threshold(ledger: SkillLedger) -> Optional[float]:
    """Recurrence weight above which a signature is a crystallization
    candidate : lambda + sqrt(lambda), lambda = total_weight / n_signatures.
    None when too few signatures exist for the baseline to be meaningful."""
    n = len(ledger.traces)
    if n < _MIN_SIGNATURES or ledger.total_weight <= 0.0:
        return None
    lam = ledger.total_weight / n
    return lam + math.sqrt(lam)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _tool_signature(p: Point) -> str:
    for tg in (p.tags or []):
        if tg.startswith("sig_"):
            return tg[len("sig_"):]
    return ""


def detect_skill_candidates(
    ledger: SkillLedger,
    *,
    existing_tools: Optional[Sequence[Point]] = None,
) -> List[Tuple[str, _ActionTrace]]:
    """Signatures whose diversity-weighted recurrence clears the emergent
    threshold — the actions worth crystallizing into persistent tool nodes.
    Signatures already covered by an existing tool node (matching `sig_`
    tag) are skipped. Returned strongest-first."""
    thr = skill_threshold(ledger)
    if thr is None:
        return []
    covered = {
        _tool_signature(p) for p in (existing_tools or []) if is_tool(p)
    }
    covered.discard("")
    out = [
        (sig, tr) for sig, tr in ledger.traces.items()
        if tr.weight > thr and sig.replace("|", "_")[:44] not in covered
        and sig not in covered
    ]
    out.sort(key=lambda st: st[1].weight, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Synthesis (LLM generalizes the recurring actions into a skill)
# ---------------------------------------------------------------------------


def _scope_keywords(trace: _ActionTrace, *, top: int = 8) -> List[str]:
    return [k for k, _ in sorted(
        trace.scope_kw.items(), key=lambda kv: kv[1], reverse=True)][:top]


def synthesize_tool(
    signature: str,
    trace: _ActionTrace,
    llm: Any,
    encoder: Any,
    t_now: float,
    *,
    genre: str = "command",
    tool_id: Optional[str] = None,
) -> Optional[Point]:
    """Crystallize a recurring action cluster into a TOOL — a normal node.

    Returns a Point of kind=ACTION (the executable kind) tagged TOOL_TAG +
    a genre tag + the grounding CONTEXT tags, with the grounding KEYWORDS as
    its semantic scope and the generalized approach as its content. Being a
    normal node, it lands in the shared cloud : the walk discovers it
    recursively through the same action/thought/fact retrieval, it persists
    with the cloud, and lateral collision / spike apply to it like anything
    else. The LLM is CONTENT-only (Cor. 5)."""
    scope_kw = _scope_keywords(trace)
    from metacog.keywords import position_weighted_keyword_embedding

    name = signature.replace("|", "_") or "tool"
    when = "; ".join(trace.query_samples[:2])[:160]
    how = "; ".join(trace.action_samples[:2])[:200]
    if hasattr(llm, "generate"):
        prompt = (
            "The system has repeatedly derived the same kind of ACTION to "
            "answer DIFFERENT user queries. Crystallize it into ONE reusable "
            "tool.\n\n"
            "Recurring actions:\n" +
            "\n".join(f"  - {a}" for a in trace.action_samples) +
            "\n\nTriggered by queries like:\n" +
            "\n".join(f"  - {q}" for q in trace.query_samples) +
            f"\n\nGrounding topics: {', '.join(scope_kw)}\n\n"
            "Reply in 3 lines exactly:\n"
            "NAME: <short snake_case tool name>\n"
            "WHEN: <one line — when this tool applies>\n"
            "HOW: <one line — the reusable approach / command to perform it>"
        )
        try:
            raw = (llm.generate(prompt, max_tokens=160) or "").strip()
        except Exception:
            raw = ""
        for line in raw.splitlines():
            ls = line.strip()
            if ls.upper().startswith("NAME:"):
                cand = ls.split(":", 1)[1].strip().replace(" ", "_")
                if cand:
                    name = cand
            elif ls.upper().startswith("WHEN:"):
                when = ls.split(":", 1)[1].strip() or when
            elif ls.upper().startswith("HOW:"):
                how = ls.split(":", 1)[1].strip() or how

    # Content = the crystallized capability ("how"). Scope = keywords.
    content = how or when or name
    # Keywords : the grounding scope PLUS the tool's own name tokens, so the
    # walk matches it both on the task domain and on the capability name.
    kws = list(dict.fromkeys([w for w in name.split("_") if w] + scope_kw))[:8]
    kw_emb = (position_weighted_keyword_embedding(kws, encoder)
              if kws and encoder is not None else None)

    tid = tool_id or f"tool_{name}@{t_now:.3f}"
    tool = Point(
        id=tid,
        content=content,
        embedding_orig=tuple(encoder.encode(content)) if encoder is not None else (),
        kind=PointKind.ACTION,
        keywords=kws,
        keywords_embedding=kw_emb,
    )
    # Tags : the universal "tool" type, its genre, and grounding context.
    tool.add_tag(TOOL_TAG, _genre_tag(genre))
    for ctx in scope_kw[:5]:
        tool.add_tag(ctx)
    # Provenance lives in tags too (audit) without polluting keywords.
    tool.add_tag(f"sig_{signature}".replace("|", "_")[:48])
    return tool


def synthesize_tool_from_intent(
    query: str,
    how: str,
    llm: Any,
    encoder: Any,
    t_now: float,
    *,
    genre: str = "command",
    scope_keywords: Optional[Sequence[str]] = None,
    extractor: Any = None,
    tool_id: Optional[str] = None,
) -> Optional[Point]:
    """EAGER single-shot crystallization : build a tool node NOW from one
    query + the approach (`how`) the system just deduced, without waiting
    for the action to recur. This is the 'no tool → generate it → proceed'
    step : because generation is unconstrained it never blocks the agent,
    it only ever ADDS a tool to the closed self-built set.

    `how` is the metacognitive deduction (the generated action / approach).
    Scope keywords come from the query (+ any supplied grounding)."""
    if not (query or how):
        return None
    from metacog.keywords import position_weighted_keyword_embedding

    scope_kw = list(scope_keywords or [])
    if extractor is not None:
        scope_kw = list(dict.fromkeys(
            scope_kw + (extractor.extract(query, n=6) or [])))
    if not scope_kw:
        scope_kw = [w.lower() for w in (query or "").split()][:6]

    name = "tool"
    content = (how or query).strip()
    if hasattr(llm, "generate"):
        prompt = (
            "The agent needs a capability it does not yet have. Crystallize "
            "it into ONE reusable tool.\n"
            f"Request: {query}\n"
            f"Intended approach: {how}\n\n"
            "Reply in 2 lines exactly:\n"
            "NAME: <short snake_case tool name>\n"
            "HOW: <one line — the reusable approach / command to perform it>"
        )
        try:
            raw = (llm.generate(prompt, max_tokens=120) or "").strip()
        except Exception:
            raw = ""
        for line in raw.splitlines():
            ls = line.strip()
            if ls.upper().startswith("NAME:"):
                cand = ls.split(":", 1)[1].strip().replace(" ", "_")
                if cand:
                    name = cand
            elif ls.upper().startswith("HOW:"):
                content = ls.split(":", 1)[1].strip() or content

    kws = list(dict.fromkeys([w for w in name.split("_") if w] + scope_kw))[:8]
    kw_emb = (position_weighted_keyword_embedding(kws, encoder)
              if kws and encoder is not None else None)
    tid = tool_id or f"tool_{name}@{t_now:.3f}"
    tool = Point(
        id=tid,
        content=content,
        embedding_orig=tuple(encoder.encode(content)) if encoder is not None else (),
        kind=PointKind.ACTION,
        keywords=kws,
        keywords_embedding=kw_emb,
    )
    tool.add_tag(TOOL_TAG, _genre_tag(genre))
    for ctx in scope_kw[:5]:
        tool.add_tag(ctx)
    tool.add_tag("eager")           # provenance : generated on first need
    return tool


# ---------------------------------------------------------------------------
# Explicit tool-intent metacognition
# ---------------------------------------------------------------------------
#
# Crystallization has TWO triggers :
#
#   1. RECURRENCE (above) : the same action re-derived across diverse
#      queries — a reusable capability earned over time.
#
#   2. EXECUTABLE INTENT (here) : the moment a metacognitive THOUGHT or a
#      generated response implies running CODE or a COMMAND underneath,
#      there is, by definition, a tool to generate. The principle the host
#      enforces : the agent may ONLY invoke tools it has previously
#      generated — so any executable need MUST first crystallize into a
#      tool, which then persists and is reused (no second think-phase).
#
# `assess_tool_intent` is that explicit metacognition : it reads the
# generated text and decides whether it carries executable material worth
# turning into a tool, and of what kind.


# Surface markers that an action/thought entails underlying execution.
_EXEC_MARKERS = (
    "run ", "execute", "compute", "calculate", "query ", "fetch", "call ",
    "search", "scrape", "download", "parse", "api", "endpoint", "sql",
    "command", "script", "code", "function", "request", "http", "shell",
    "pip ", "install", "import ", "def ", "select ", "curl",
)


@dataclass(frozen=True)
class ToolIntent:
    """Verdict of the explicit tool-intent metacognition."""

    is_executable: bool
    kind: str            # "code" | "command" | "api" | "none"
    rationale: str = ""


def assess_tool_intent(
    text: str,
    llm: Any = None,
    *,
    use_llm: bool = True,
) -> ToolIntent:
    """Decide whether `text` (a metacognitive thought or a generated
    response/action) implies an EXECUTABLE capability — i.e. there is code
    or a command to run underneath, which means a tool should be generated.

    A cheap surface-marker pass screens obviously-executable text ; when an
    LLM is available it adjudicates the ambiguous middle (so a purely
    descriptive sentence that merely mentions 'code' is not mistaken for an
    executable need). Pure COMPUTATION + optional LLM judgment ; no
    execution happens here."""
    t = (text or "").lower()
    if not t.strip():
        return ToolIntent(False, "none", "empty")
    surface = any(m in t for m in _EXEC_MARKERS)
    if not surface:
        return ToolIntent(False, "none", "no execution markers")
    if not (use_llm and hasattr(llm, "generate")):
        # Marker-only mode : trust the surface signal, default kind "command".
        kind = ("code" if any(m in t for m in ("def ", "import ", "code", "script", "function"))
                else "api" if any(m in t for m in ("api", "endpoint", "http", "request", "curl"))
                else "command")
        return ToolIntent(True, kind, "surface markers")
    prompt = (
        "Does the following imply running CODE or a COMMAND / API call "
        "underneath (i.e. a tool would be needed to actually do it), or is "
        "it purely descriptive? Answer exactly one line:\n"
        "VERDICT: <yes|no> KIND: <code|command|api|none>\n\n"
        f"TEXT: {text[:400]}"
    )
    try:
        raw = (llm.generate(prompt, max_tokens=24) or "").strip().lower()
    except Exception:
        raw = ""
    is_exec = "yes" in raw
    kind = "none"
    for k in ("code", "command", "api"):
        if k in raw:
            kind = k
            break
    if is_exec and kind == "none":
        kind = "command"
    return ToolIntent(is_exec, kind if is_exec else "none", raw[:80])


# ---------------------------------------------------------------------------
# Fast-path matcher (skip re-thinking when a tool already covers the query)
# ---------------------------------------------------------------------------


_MATCH_THRESHOLD = 0.60   # query must be at least this close to a tool's
                          # scope to reuse it instead of thinking afresh.


def tools_in(points: Sequence[Point]) -> List[Point]:
    """All tool nodes in a point set (tagged TOOL_TAG)."""
    return [p for p in points if is_tool(p)]


def match_tool(
    query_emb: Optional[Vector],
    points: Sequence[Point],
    *,
    threshold: float = _MATCH_THRESHOLD,
) -> Optional[Tuple[Point, float]]:
    """Return the best TOOL node whose semantic scope (its keyword
    embedding) covers `query_emb`, with the match score — or None if the
    query falls outside every tool's scope (so the agent must think from
    scratch). This is the explicit capability-cache lookup ; the walk
    ALSO surfaces the same tool nodes natively through its ACTION
    retrieval, so a tool can additionally be found recursively mid-walk.
    A hit means 'you already generated how to do this — reuse it'."""
    if query_emb is None:
        return None
    best: Optional[Tuple[Point, float]] = None
    for p in points:
        if not is_tool(p) or p.keywords_embedding is None:
            continue
        s = cosine(query_emb, p.keywords_embedding)
        if s >= threshold and (best is None or s > best[1]):
            best = (p, s)
    return best
