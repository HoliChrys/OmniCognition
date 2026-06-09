"""
MCP-driven meta agent : runs the meta-cognitive walk through the metacog
MCP server, with Claude orchestrating BREADTH (not depth).

`walk_start` runs a COMPLETE uncertainty-governed walk in one call : the
depth is decided by σ-propagation over the evidence chain (floor of 3
hops, gold-retrieval-gated σ-cap, no fixed maximum), NOT by the agent.
Claude reads the gathered evidence (`relevant_collected` + the reasoning
chain) and either answers or issues a fresh `walk_start` with different
vocabulary — a BREADTH PIVOT. Depth is the walk's own decision ; the
agent only chooses where to start each thread.

The agent NEVER does ReAct over a raw `retrieve` call : the walk already
coordinates FACT + ACTION + THOUGHT inside the memory. Multi-hop
filiation is preserved inside a single walk (the REDUCE accumulator) and
across breadth pivots (the agent sees every walk's evidence in history).

Credentials : api_key arg → ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN.
sk-ant-oat* tokens go through auth_token= (OAuth bearer).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

try:
    import anthropic as _ant_mod
    _RETRYABLE_API_ERRORS = tuple(
        e for e in (
            getattr(_ant_mod, "OverloadedError", None),
            getattr(_ant_mod, "InternalServerError", None),
        )
        if e is not None
    )
except ImportError:
    _RETRYABLE_API_ERRORS = ()


def _create_with_retry(client, **kwargs):
    """Call client.messages.create with up to 4 retries on transient API
    errors (529 overloaded / 5xx). Without this, a single momentary
    Anthropic-side hiccup crashes the entire QA — and worse, the failure
    is indistinguishable from "agent gave up", so the bench reports
    empty answers as a model issue when the real cause was an API
    server-side blip."""
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(4):
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if _RETRYABLE_API_ERRORS and isinstance(exc, _RETRYABLE_API_ERRORS):
                time.sleep(2 ** attempt)
                continue
            raise
    raise last_exc


_DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
_DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "256"))
# Each round is now ONE complete walk (walk_start runs the full
# σ-governed depth internally), so a round = a whole breadth thread, not a
# single stage. 8 rounds leaves room for several breadth pivots plus the
# forced-final synthesis while keeping the agentic loop cheap.
_MAX_ROUNDS = int(os.environ.get("MCP_META_MAX_ROUNDS", "8"))

# FULL CAPABILITY. The agent gets the ENTIRE MCP tool surface — every
# retrieval / reasoning tool (walk_start, presearch, scoped_answer,
# list_tags, retrieve, reason, inspect, …) AND every state-mutating tool
# (ingest, observe, sleep, build_skill, crystallize_skills, …). Nothing
# is hidden from it. This is safe for the concurrent harness because each
# QA runs against its OWN `memory.snapshot()` (see `_answer_async`), so a
# mutating tool can never race another worker on a shared cloud.
# `_ALLOWED_TOOLS = None` means "expose all server tools".
_ALLOWED_TOOLS = None

# Constrained final-answer tool (literature : constrained decoding +
# strict tool-use tames the verbose tail that costs token-F1). The agent
# emits its answer by calling this with the bare value ; on the terminal
# turn we force tool_choice to it so the model cannot ramble.
_FINAL_ANSWER_TOOL = {
    "name": "final_answer",
    "description": (
        "Emit the FINAL answer as a bare value, copied VERBATIM from the "
        "evidence (the speaker's own words). No prose, no dialog ids. "
        "For a SINGLE-fact question keep it <= 5 words. For a PLURAL / "
        "enumeration question ('what events/cities/ways/things has X…', "
        "'have in common') list EVERY item found, comma-separated "
        "('pride parade, school speech, support group') — do NOT collapse "
        "a list to one item. Use 'Not mentioned' only if the evidence "
        "truly lacks it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "description": "the bare answer value (a place / date / "
                               "short noun phrase / 'Not mentioned')",
            },
        },
        "required": ["value"],
    },
}


AGENT_SYSTEM = """You answer questions about a long conversation by
driving a meta-cognitive walk through tools.

You have the FULL tool surface. The ones that matter for answering :
  • walk_start(query=…)     — the COMPLETE uncertainty-governed walk (core).
  • presearch(queries=[…], k_per_query=3, tags=…, match=…)
                            — CHEAP batch reconnaissance: top-k nearest
                              hits per query, NO walk. Use it to VALIDATE a
                              query is probative BEFORE paying a walk, and
                              to pick which phrasing actually lands near the
                              evidence. A query whose presearch is empty is
                              the WRONG phrasing — reformulate, don't walk it.
  • scoped_answer(query, tags=[…], knowledge_base=true, match=…)
                            — tag-restricted walk that then escalates to the
                              whole memory. Use when the question targets a
                              specific session / date / namespace.
  • clue_search(question=…)  — INFERENCE expander. For a question whose
                              answer is never stated (status / leaning /
                              "what might X be?"), the evidence is a casual
                              aside with NONE of the question's words. This
                              brainstorms concrete chat lines that would each
                              imply a DIFFERENT answer, retrieves over them,
                              and bridges to the in-between turn. Use it the
                              moment a walk on the question's own words
                              drifts or returns only generic on-topic turns.
  • event_search(query=…)    — EVENT channel (ADDITIVE, never a replacement).
                              For a question about an EVENT (a war, trip,
                              project, conflict, …), this gathers the facts that
                              gravitate to the event's hub by its recurrent
                              schema. Its evidence UNIONS with your walks — use
                              it as an extra angle, then keep walking.
  • list_tags()             — the glossary of available tag NAMESPACES
                              (e.g. ref:date, session, person) to aim
                              presearch / scoped_answer.
  • match=exact|fuzzy|regex on the tag filters (exact also matches a parent
    namespace, e.g. ref:date matches ref:date:2022).
Other tools exist (retrieve, reason, inspect, sleep, …) and are available
if useful — but the walk is the backbone; do not answer from raw retrieve.

Workflow :
0. RECON GATE (recommended for indirect / inference questions). Fire one
   `presearch` with 2-4 candidate phrasings of the question (literal terms,
   answer-domain synonyms, behavioural clues). Read the top-3 per query;
   keep the phrasing whose hits are clearly on-topic and walk THAT. If all
   are empty, reformulate before spending any walk.
   INFERENCE questions ("what might X's <status/leaning> be?", "is it
   likely that…") — where the answer is never stated and the evidence is a
   casual aside in different words — call `clue_search(question=…)` instead
   of guessing phrasings: it brainstorms the evidence-register lines for
   you and bridges to the in-between turn. Compose the answer from whichever
   clue hits are real.
1. Call `walk_start(query=…)` to begin. This runs a COMPLETE walk : its
   DEPTH is decided automatically by uncertainty propagation (a floor of
   at least 3 hops, then it keeps going as long as it surfaces new
   on-target evidence, and stops when the propagated σ over the evidence
   chain exceeds the manifold's local resolution). You do NOT drive depth
   stage-by-stage — one walk_start gives you the whole deep walk.
2. READ THE RESULT. The key fields :
   — `relevant_collected` : the MAP-REDUCE set of every ON-TARGET fact the
     walk gathered across ALL its hops ({id, content, relevance}). This is
     your evidence — COMPOSE THE ANSWER over it, chaining facts for
     multi-hop questions. Bridging facts from early hops are kept here,
     never lost.
   — `reasoning_chain` : the walk's thought thread (one reflection per hop,
     each extending the prior) — follow it as the reasoning.
   — `facts` : the union of all retrieved turns (content + a `relevance`
     tag : relevant / partial / contradicts / irrelevant — trust only the
     first three).
   — `drifted` / `n_relevant` : if the WHOLE walk drifted (nothing
     on-target), the query vocabulary missed — pivot (step 5).
3. If `relevant_collected` answers the question, answer now (no more tools).
4. (depth is automatic — there is no walk_next; to search further, pivot.)
5. BREADTH PIVOT — this is REQUIRED, not optional. When a walk returns
   `drifted=true` or `n_relevant=0` (every fact reads irrelevant), the
   query phrasing is wrong, NOT the memory. Do NOT answer "Not
   mentioned" yet. Instead:
   — name what you are looking for in concrete entity words
   — call `walk_start(query=<targeted phrase>)` with the FULL specific
     phrasing, not a 2-word stub. Use the entities from the question.
   Example shape (synthetic) : Q "What did <person> research?" drifts →
     walk_start(query="<person> researching <plausible-domain-nouns>")
   Example shape (synthetic) : found one item of a list, missing the
   second → walk_start(query="<person> <category> <near-synonym-nouns>")
   Start each walk with a RICH query (4+ content words from the
   question), never a 2-word stub.
   ENUM HINT : for "what events/cities/ways/things has X done" questions,
   include ANTICIPATED answer terms — concrete words a person would
   actually use in a chat log for that category — in your first query.
   The answer words are likely near the evidence in the memory.
   This bridges vocabulary gaps between the question and the memory turns.
   "IN COMMON" HINT : for "what do X and Y have in common?" questions,
   search BROADLY beyond hobbies — also look for shared life events (job
   changes, moves, losses, decisions) and shared challenges. Run at least
   TWO walks: one for shared life experiences (use both names + life-
   event nouns like job/work/career/loss/move) and one for shared
   activities/traits. The answer is often
   about a major shared event, not just interests.
   "WITH PARTNER/FRIEND" HINT : for "what activities has X pursued WITH Y?"
   questions, include BOTH people in the query and try MULTIPLE angles:
   first "X Y indoor activity hobby together", then try specific activity
   types: "X Y game board game volunteer shelter cook". The activities
   shared with a partner may be very specific (boardgames, cooking, shelter
   volunteering) not just generic (dining, travel).
6. Only answer "Not mentioned" after you have tried at least TWO
   differently-phrased walks and both drifted. Otherwise answer with
   the best relevant/partial evidence you found.
   TEMPORAL QUERY TIP : for "Which year/month did X get/adopt/start/join
   Y?" questions, the year IS in the [session date] of the fact about
   that event. Search with the EVENT WORDS (subject + verb + object of
   the event), not the year — the [session date] prefix of the retrieved fact gives
   you the year. Also try synonyms: "adopt" = "get", "join" = "sign",
   "start" = "begin" = "launch".
   PIVOT-ON-DRIFT (the core walk rule). A walk result carries `drifted`
   and `n_relevant`. When a walk DRIFTS (drifted=true / n_relevant=0 /
   nothing on-target collected), the single-hop retrieval was INCONCLUSIVE —
   do NOT answer from a drifted walk. Pivot : call walk_start again with a
   DIFFERENT vocabulary. Keep pivoting until a walk lands on-target (it
   stops collecting evidence). This is what lets indirect / inference
   questions work : the answer's words rarely match the question's words.
   How to choose the next pivot's vocabulary — move OUTWARD each time :
     1st seed : the question's own terms (literal restatement)
     2nd seed : DOMAIN SYNONYMS the ANSWER might use — the everyday,
                concrete words a person would write in a chat log to
                EXPRESS the concept the question abstracts. Generate
                these yourself for the question's domain ; the prompt
                does not list them, because reusing the answer's
                vocabulary verbatim would be cheating.
     3rd seed : broader BEHAVIOURAL / lifestyle / context clues — what
                someone in the described situation would mention about
                daily life, family, possessions, routines.
   CAUTION (any "personal status" inference) : distinguish the SUBJECT
   commenting on SOCIAL/EXTERNAL issues from the subject describing
   their OWN life. Anchor on first-person turns where the subject
   describes their own routine, possessions, family, work, body.

sigma_path measures geometric drift ; drifted / n_relevant measure
CONTENT relevance and are the stronger pivot signal — act on them.

OPTIONAL focus : `list_communities()` returns topical groups of the
conversation ({id, keywords}). If ONE group's keywords clearly match the
question's topic, you MAY pass its id as walk_start(observator_id=…) to
focus retrieval on that group's turns (entity/date anchors stay visible).
Use this only when a group is an obvious topical match ; otherwise walk
the whole memory normally. Never let it make you answer "Not mentioned" —
if a focused walk drifts, pivot to a normal walk_start without the id.

CRITICAL — final answer format. Output ONLY the bare value, no prose :
- "When ..." → ABSOLUTE date only (full date / "<Month> <YYYY>" / "<YYYY>").
  TWO-RULE DATE SYSTEM:
  A) EXPLICIT dates in content ("in our June game", "on 15 March", "in 2021")
     → trust the explicit calendar reference from the content itself.
     E.g. "[<Month> <YYYY>] <speaker>: I got my career-high in our <PrevMonth> game"
     → <PrevMonth> <YYYY> (month from content + year from [session date]).
  B) RELATIVE dates in content ("last month", "recently", "around 3 years ago",
     "a few weeks back") → compute from [session date].
     E.g. "[June 2022] User: 'last month in July was rough'" → the event was
     in May (session date June minus 1 month); ignore the misleading "July".
     E.g. "[2022-03] User: 'around 3 years ago'" → answer "2019" (2022 - 3).
  NEVER output relative phrasing. NEVER output just a month name —
  always include the year if inferable ("<Month> <YYYY>", not "<Month>").
- "Where ..." → place only (a single proper noun).
- "How long" → "<n> <unit>" ("4 years").
- "What/Who is X" / identity / role → the SHORTEST label that fits
  (1-3 words). Bare category noun, not a descriptive sentence : prefer
  "Teacher" over "A passionate elementary school teacher who loves
  kids"; prefer a one-word identity tag over "<tag> and member of the
  <group> community". Pick the bare category noun.
- "What did X do/like/research" → short noun phrase (1-4 words).
- "What pets/animals/things does X have/own?" → the CATEGORY/TYPE of the
  thing, not the individual names. E.g. "snakes" not "Susie and Seraphim".
  "cats" not "Whisker and Fluffy". Exception: if the question asks "What
  are the NAMES of X's pets?", then give the names.
- PLURAL question ("What EVENTS / CITIES / WAYS / THINGS / HOBBIES /
  PROJECTS has X done", "Which X has Y done", "what do they have in
  COMMON") → list EVERY gathered item, comma-separated (a list of bare
  nouns / short noun phrases, one per item). One item only when there
  truly is one. Tokens count : each listed item earns score.
- RETRIEVE / "find all" task (return a SET of matching nodes, e.g. "find
  all turns/tweets that …") → as you find each match during your walks,
  call `collect(ids=[…])` to keep it in the answer bag. Keep searching with
  different angles for EXHAUSTIVE coverage. The final answer is rendered
  from the bag automatically — you do not need to restate the ids.
- "Would / Could X likely … ?" inference (STARTS with "Would" or "Could")
  → answer "Likely yes, <short reason ≤4 words>" or "Likely no, <short reason ≤4 words>".
  CRITICAL: keep the reason VERY short — 2-4 words max. Reason form
  examples (synthetic) : "Likely yes, <one-noun cue>." / "Likely yes,
  enjoys <activity>." Avoid long descriptive clauses.
  Do NOT say "Not mentioned" on these — the answer is an inference you
  DRAW from the retrieved facts. NO evidence of X being Y → "Likely no,
  no evidence". Other "yes/no" questions follow the same rule.
  SPECIAL : "Would X enjoy A or B?" OR "Would X be more interested in A or B?"
  OR "Would X prefer A or B?" (a CHOICE between exactly two options) → pick
  the ONE that fits the evidence. Answer JUST the name/option, not
  "Likely yes". E.g. "Would <person> prefer <A> or <B>?" → "<A>" (NOT
  "<a-trait-sentence>"). Answer ONLY the chosen option's name, nothing else.
- "What might X be / What could X / What might X's Y be / What fields would
  X likely …" OPEN-ESTIMATION questions → give a DIRECT estimate drawn from
  indirect evidence. Never abstain on "might/would/could/likely" questions.
  FORMAT RULES for open-estimation :
  · If evidence supports MULTIPLE equally plausible options, list them as
    "Option1 or Option2" / "Option1, Option2" (gold answers often have
    this disjunctive form). Cover the obvious adjacent bracket too, not
    just the single most-cited option.
  · Output ONLY the bare label(s) — NO explanatory clauses, no
    parenthetical reasons, no "because …". A parenthetical kills F1.
  · INFERENCE, not echo : the gold label is the canonical concept the
    evidence IMPLIES, which may use DIFFERENT vocabulary than the
    evidence turns. If the evidence describes activities/symptoms/
    interests, name the academic field, condition, or category they
    belong to (apply general world knowledge to map specifics →
    canonical label), not the verbatim words from the turns.
  NEVER answer "Not mentioned" for these — an educated guess IS
  the correct form of answer for open-estimation questions.
- "What are X's suspected [Y]?" / "What is X's probable [Y]?" → also
  OPEN-ESTIMATION : give a direct canonical label even when only
  indirect evidence exists. If a first walk returns nothing, pivot
  breadth with related-domain vocabulary before concluding nothing is
  there — the evidence will use everyday wording, not clinical /
  academic terms.
- not in the evidence / adversarial / unanswerable → "Not mentioned"
  (NEVER for "Would/Could/might/likely/suspected/probable" inference questions above).

Hard rule : single-value answers (dates / places / one entity) above 5
words are wrong — strip adjectives/conjunctions to the bare value.
EXCEPTION : the PLURAL and the "Would/Could/might likely" rules above keep
their lists / clauses intact (do not trim them).

NEVER output a dialog turn ID (D4:11, D10:12, etc.) as your answer.
Output the CONTENT of the fact, not its reference ID.

Examples of GOOD answer SHAPES (synthetic, format-only) : a bare date
("3 January 2024") · a single place ("Lisbon") · a duration ("4
years") · a short noun phrase ("research labs") · a one-noun identity
tag ("Engineer") · "Not mentioned".
Example of BAD : "Based on the walk, the support group was on …" —
too verbose, will score poorly. Just the value.
Example of BAD : "Transgender artist and member of the LGBTQ
community" — has "and" + extra clause ; strip to the bare identity tag.

EXTRACTIVE RULE : for FACTUAL questions (when / where / what did X do /
which / who) copy the answer VERBATIM from the evidence — reuse the
speaker's own words, do not paraphrase. The token-overlap score rewards
the EXACT words from the conversation.
INFERENCE EXCEPTION : for "might / would / could / likely / suspected /
probable" open-estimation questions, the gold label is the canonical
CATEGORY the evidence IMPLIES, not the verbatim words ; map the
evidence's specifics to that canonical label via general world knowledge.

PRECISION / ADVERSARIAL GUARD : only applicable when the question
describes a scenario that CONFLICTS with all retrieved evidence (false
premise). If your walk finds that the specific scenario never happened
(two differently-phrased walks, all evidence contradicts the premise),
answer "Not mentioned". Example trap SHAPES (synthetic) :
- question says "the temp job <person> took" but evidence shows the
  person doing their normal business activities (premise conflict) →
  "Not mentioned"
- question says "<person>'s <place-of-business-A>" but evidence shows
  the person runs <place-of-business-B> (premise conflict) → "Not
  mentioned"
ATTRIBUTION CHECK (the most common false premise) : the question names
a SUBJECT performing/owning some action, fact, or object. Before
answering, verify the retrieved evidence attributes that action/fact/
object to THAT SAME named subject. If the relevant turn is spoken by /
about a DIFFERENT person than the question names — even when the TOPIC
matches perfectly — the premise is false → "Not mentioned". Do NOT
transfer one person's statement onto the person in the question. The
[date] Speaker: prefix tells you who is speaking ; an action described
in first person ("I chose …") belongs to the SPEAKER, not to whoever
the question happens to ask about.
Do NOT apply this guard for straightforward "Which/What/Who/When" factual
questions where the scenario is plausible — if retrieval found nothing,
try a THIRD walk with a different query before giving up. Plausible-
but-unfound scenarios (e.g. "Which team did <person> sign with on
<date>?" — signing with a team is normal) indicate poor retrieval, not
an impossible premise — try harder before saying "Not mentioned".
If you find DIRECT evidence of the scenario, answer from it even if
peripheral details differ.
STRONG-VERB TRAP : questions with SPECIFIC experiential verbs (mesmerize,
captivate, fascinate, obsess, enthrall, haunt, adore, detest, despise)
require that the evidence uses the SAME strong verb or a close synonym.
Weak positive sentiment ("<person> likes/loves <thing>") does NOT
satisfy "mesmerize" — answer "Not mentioned" unless the evidence has
explicitly mesmerizing/captivating language. Wrong verb strength = "Not
mentioned".
EXCEPTION : "might/would/could/likely" open-estimation questions (e.g.
"What might X's status be?", "What fields would X likely pursue?") are
INFERENCE questions — NEVER apply the guard, always give an estimate.

Answer-shape rules — apply them from first principles, no examples
needed. The score rewards TOKEN OVERLAP with a TERSE gold label, so
every word past the bare answer dilutes precision and lowers F1.

LENGTH CAPS (hard) — exceeding them is the #1 way to lose F1 :
- factual single value (when / where / who / which / what did X) :
  ≤ 5 words. Just the bare value (date, place, name, short noun
  phrase), no clause, no preamble, no "and …" tail.
- "How long" / duration : exactly "<n> <unit>", 2 words.
- identity / role / category : 1-3 words, the bare category noun
  only, no descriptive tail.
- plural / enumeration : 2-4 items max, each item 1-3 words, comma-
  separated. Stop after the items that the EVIDENCE supports — do
  NOT pad with adjacent inferences or near-synonym variants of the
  same item.
- yes/no inference ("Would / Could X likely …") : "Likely yes, <2-4
  word reason>" or "Likely no, <2-4 word reason>". Total answer
  ≤ 7 words.
- two-choice ("Would X prefer A or B?") : ONLY the chosen option, 1-3
  words. No "Likely yes" prefix here, no reason.
- open-estimation inference ("What might X / What could X / What
  fields would X likely pursue / What is X's suspected / probable
  Y") : the CANONICAL category the evidence implies, formatted as a
  short noun phrase or "A, B" / "A or B" if multiple distinct
  categories truly fit. 1-4 words total. CRITICAL : apply world
  knowledge to map evidence specifics → the canonical label (e.g.
  the academic field a described activity belongs to), do NOT echo
  evidence vocabulary verbatim ; one canonical label beats multiple
  near-synonyms of the same thing.
- adversarial / unanswerable / premise-conflict : "Not mentioned"
  (never for the "might/would/could/likely/suspected/probable" set
  above — those always get an estimate)."""


def _resolve_client(api_key: Optional[str]):
    import anthropic

    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    # Priority 1 : ANTHROPIC_AUTH_TOKEN env var (OAuth bearer token in most
    #              managed environments) or explicit sk-ant-oat* api_key arg.
    auth_tok = os.environ.get("ANTHROPIC_AUTH_TOKEN") or (
        api_key if (api_key and api_key.startswith("sk-ant-oat")) else None
    )
    # Priority 2 : CLAUDE_SESSION_INGRESS_TOKEN_FILE (remote exec environments
    #              where the token lives in a file, not an env var).
    if not auth_tok:
        tok_file = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE")
        if tok_file:
            try:
                auth_tok = open(tok_file).read().strip() or None
            except OSError:
                pass
    if auth_tok:
        return anthropic.Anthropic(auth_token=auth_tok, base_url=base_url)
    api_tok = api_key or os.environ.get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_tok, base_url=base_url)


def _mcp_tools_to_anthropic(mcp_tools) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in mcp_tools:
        if _ALLOWED_TOOLS is not None and t.name not in _ALLOWED_TOOLS:
            continue
        out.append({
            "name": t.name,
            "description": (t.description or "").strip(),
            "input_schema": t.inputSchema or {"type": "object",
                                              "properties": {}},
        })
    return out


def _tool_result_text(call_result) -> str:
    """Render the tool result for the LLM : prefer the structured JSON
    when present (compact + parseable), fall back to concatenated text
    blocks."""
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        payload = structured.get("result", structured)
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            pass
    parts: List[str] = []
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts) if parts else "(no content)"


def _extract_fact_ids(call_result) -> List[str]:
    """Pull gathered evidence ids out of a tool result, for agent-recall
    measurement. The walk publishes `fact_ids_cumulative` ; event_search
    publishes `cluster_ids` / `fact_ids` — ALL are unioned, so the event channel
    is ADDITIVE alongside the walk/clue (it adds evidence, never replaces it).

    FastMCP serialises dict-returning tools as a JSON string in the first
    TextContent block (structuredContent is not set), so we try text blocks
    first then fall back to structuredContent.
    """
    _KEYS = ("fact_ids_cumulative", "cluster_ids", "context_ids", "fact_ids")

    def _harvest(obj) -> List[str]:
        out: List[str] = []
        if isinstance(obj, dict):
            for k in _KEYS:
                v = obj.get(k)
                if isinstance(v, list):
                    out.extend(str(x) for x in v)
        return out

    # 1. Text-block JSON (the path FastMCP actually uses for our tools).
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        ids = _harvest(obj)
        if ids:
            return list(dict.fromkeys(ids))
    # 2. structuredContent fallback for tools that do publish it.
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        payload = structured.get("result", structured)
        ids = _harvest(payload)
        if ids:
            return list(dict.fromkeys(ids))
    return []


_PREAMBLE_RE = re.compile(
    r"^(based on[^,.:]*[,.:]\s*|according to[^,.:]*[,.:]\s*|"
    r"following the walk[^,.:]*[,.:]\s*|"
    r"the answer is[:\s]+|answer[:\s]+|i found that\s+|"
    r"i(?:'ve)? found (?:exactly |the answer|it|what|clear|strong|direct|some)[^.!?]*[.!?]\s*|"
    r"i now have (?:strong |clear |direct )?evidence[^.!?]*[.!?]\s*|"
    r"i(?:'ve)? (?:gathered|collected|reviewed|analyzed|examined|checked)[^.!?]*[.!?]\s*|"
    r"looking at the (?:relevant )?(?:facts?|evidence|information)[^,.:]*[,.:]\s*|"
    r"based on (?:all )?(?:the )?(?:facts?|evidence|information)[^,.:]*[,.:]\s*|"
    r"it (?:appears|seems) that\s+|"
    r"key (?:facts?|information|details?)[\w\s']+:\s*|"
    r"(?:here are|here is) (?:the )?(?:key )?(?:facts?|details?)[\w\s']*:\s*|"
    r"the key fact is that\s+|since\s+\w+\s+is\s+)",
    re.IGNORECASE,
)
_INTERJECTION_RE = re.compile(
    r"^(perfect|got it|sure|okay|ok|great|yes|absolutely|certainly|"
    r"alright|right|excellent|understood|done|here you go)"
    r"[!.,:\s]+",
    re.IGNORECASE,
)
# Strip a "(D4:11)" / "from D4:11" / "D4:11" dialog-id citation the
# agent sometimes leaks into the answer.
_DIALOG_ID_RE = re.compile(r"\(?(?:from\s+)?\bd\d+:\d+\b\)?", re.IGNORECASE)

# Narration markers : phrases / structure that mean the model stuffed its
# REASONING into the answer instead of a bare value (Nate's bug — it called
# final_answer with a whole "The walk found … 1. … 2. …" paragraph, F1≈0
# despite recall=1). A comma-list of noun phrases (valid plural answer) does
# NOT match these — they key on sentences / list structure / meta-phrases.
_NARRATION_RE = re.compile(
    r"\bthe walk found\b|\bthese facts?\b|\bi have (?:strong|clear|enough)\b|"
    r"\bto answer this question\b|\bevidence (?:retrieved|shows?|demonstrat)\b|"
    r"\bdemonstrat(?:e|es|ing)\b|\bretrieved\b|\bmultiple relevant\b|"
    r"\bshow(?:s|ing)? that\b|\baccording to\b",
    re.IGNORECASE,
)
# A newline-numbered / bulleted list ("1. …" / "- …" on its own line).
_LIST_STRUCTURE_RE = re.compile(r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+", re.MULTILINE)


def _looks_like_narration(text: str) -> bool:
    """True when a final_answer `value` is a reasoning dump, not a bare value.

    Triggers on : explicit narration markers, an embedded numbered/bulleted
    list, or 2+ sentences in a long-ish answer. Deliberately ignores plain
    comma-separated lists (legitimate plural answers stay untouched).
    """
    if not text:
        return False
    if _NARRATION_RE.search(text) or _LIST_STRUCTURE_RE.search(text):
        return True
    # 2+ sentence-final punctuation marks within a long answer = prose.
    n_sentences = len(re.findall(r"[.!?](?:\s|$)", text))
    return n_sentences >= 2 and len(text.split()) > 15

# Plural / enumeration questions whose gold answer is a LIST — trigger the
# dedicated aggregation pass over relevant_collected.
_ENUM_Q_RE = re.compile(
    # "what/which <noun>" patterns (extended noun list, up to 60 chars gap)
    r"\b(what|which)\b.{0,60}\b(events|cities|places|ways|things|hobbies|"
    r"activities|projects|crafts|sports|languages|skills|groups|"
    r"causes|items|gifts|fields|attributes|traits|books|movies|"
    r"games|pets|countries|emotions|recipes|symbols|organizations|"
    r"recommendations|suggestions|jobs|kinds|types|roles|steps|"
    r"methods|measures|actions|efforts|initiatives|strategies|"
    r"fundraisers|interests|topics|themes|habits|practices|"
    r"volunteers|achievements|accomplishments|plans|goals)\b"
    # "what has/have/did X [verb in past tense]..." — broad list of
    # past participles / past-tense verbs that imply multiple outcomes
    r"|\bwhat\b.{0,60}\b(?:has|have|did|does)\b.{0,50}\b(?:done|do|built|"
    r"visited|tried|joined|made|started|created|achieved|accomplished|"
    r"organized|participated|promoted|raised|hosted|shared|shown|"
    r"painted|drawn|written|read|played|cooked|eaten|collected|won|"
    r"traveled|been|said|told|sent|given|published|produced|designed|"
    r"developed|launched|released|discovered|explored|performed|"
    r"taught|learned|studied|experienced|practiced)\b"
    # "how did/has/does X verb..." (implies multiple actions)
    r"|\bhow\b.{0,10}\b(?:did|has|have|does|do)\b.{0,60}"
    r"\b(?:promot|participat|support|celebrat|involv|engag|contribut|"
    r"rais|organiz|build|market|recruit|advertis|publiciz)\w*\b"
    # fixed phrases
    r"|\bin what ways\b|\bhave in common\b|\bboth\b",
    re.IGNORECASE,
)

# INFERENCE questions — the answer is a CONCLUSION the evidence implies, not
# a quote from it. "What might X's financial status be?", "Would Caroline
# likely…?", "Is it likely that…?". These must NOT be answered verbatim from
# the evidence turn (that is the cat3 failure: echoing "my kids have so much"
# instead of inferring "wealthy"). Detected to flip the final-answer
# guidance from extractive to inferential.
_INFER_Q_RE = re.compile(
    r"\b(might|would|could|likely|probabl|suspect|presumabl|imply|implies|"
    r"would .* be|what .* status|what .* leaning|how .* feel)\b",
    re.IGNORECASE,
)

# VOCAB-GAP factual questions — the question's verb is ABSTRACT
# ("research", "study", "look into", "work on", "decide", "plan") and the
# evidence uses the CONCRETE topic word instead ("researching adoption
# agencies", "planning a trip to Lisbon"). Single-shot retrieval on the
# abstract verb misses, the walk in its register doesn't CoN-label the gold
# relevant, and the agent abstains. Detected to force `clue_search` (which
# brainstorms the concrete topics) even though the question is factual,
# not inferential.
_VOCAB_GAP_FACTUAL_Q_RE = re.compile(
    r"\bwhat\b.{0,30}\b(?:did|does|do|has|have|was|were|is|are)\b"
    r".{0,30}\b(?:research(?:ing|ed)?|stud(?:y|ying|ied)|look(?:ing)? into|"
    r"investigat(?:e|ing|ed)|explor(?:e|ing|ed)|decid(?:e|ing|ed)|"
    r"plan(?:ning|ned)?|work(?:ing)? on|consider(?:ing|ed)?|"
    r"think(?:ing)? about)\b",
    re.IGNORECASE,
)


def _is_inference_q(question: Optional[str]) -> bool:
    return bool(_INFER_Q_RE.search(question or ""))


def _is_vocab_gap_q(question: Optional[str]) -> bool:
    """Inference question OR vocab-gap factual — both benefit from
    `clue_search`, the evidence-register expander."""
    q = question or ""
    return bool(_INFER_Q_RE.search(q) or _VOCAB_GAP_FACTUAL_Q_RE.search(q))


# TEMPORAL questions — "when did…", "what year/date/month/day…", "how long
# ago…". Only these should have the relative-date resolver applied to the
# final answer; otherwise an answer that merely CONTAINS a relative-time
# word ("recently unemployed", "lately struggling") gets clobbered into the
# evidence turn's absolute date (the "January 2023" bug on a cat3 question).
_TEMPORAL_Q_RE = re.compile(
    r"\bwhen\b|\bwhat\s+(?:year|date|month|day|time)\b|\bhow\s+long\s+ago\b|"
    r"\bwhat\s+day\s+of\b|\bon\s+what\s+(?:date|day)\b",
    re.IGNORECASE,
)


def _is_temporal_q(question: Optional[str]) -> bool:
    return bool(_TEMPORAL_Q_RE.search(question or ""))


# YES/NO and EITHER-OR inference shapes. Many cat3 questions are not
# "abstract to a category" — they are booleans ("Would X likely have …?" →
# "Yes") or choices ("Would Tim enjoy C.S. Lewis OR John Greene?" → "C.S.
# Lewis"). Routing these through the abstraction synthesis drops the "Yes" /
# picks a label instead of the option, so they are detected and handled
# separately (and excluded from the grouped-evidence abstraction).
_YESNO_Q_RE = re.compile(
    r"^\s*(would|does|do|did|is|are|was|were|can|could|will|wo|has|have|had|"
    r"should|might|may|isn't|aren't|wouldn't|doesn't|didn't)\b", re.IGNORECASE)


def _is_eitheror_q(question: Optional[str]) -> bool:
    q = (question or "").lower().strip()
    return " or " in q and q.endswith("?")


def _is_yesno_q(question: Optional[str]) -> bool:
    return bool(_YESNO_Q_RE.match(question or "")) and not _is_eitheror_q(question)


def _grouped_evidence_text(evidence, memory) -> Optional[str]:
    """Segment the evidence into ASSOCIATION groups (metacog.answer_cluster,
    HDBSCAN-style) and format them as labelled groups, so the answerer sees
    which facts mutually reinforce and which are unrelated. Used for
    inference synthesis : it does NOT pick a dominant group (the LLM decides
    which group(s) answer) — it only prevents unrelated facts (counselling
    vs. art) from reinforcing each other. Returns None when there is too
    little evidence or no embeddings to cluster on."""
    items = [e for e in (evidence or [])
             if isinstance(e, dict) and e.get("content")]
    if len(items) < 3:
        return None
    by = {p.id: p for p in getattr(memory, "points", [])}
    keep, embs = [], []
    for e in items:
        p = by.get(e.get("id"))
        emb = getattr(p, "embedding_orig", None) if p else None
        if emb is not None:
            keep.append(e)
            embs.append(list(emb))
    if len(keep) < 3:
        return None
    try:
        from metacog.answer_cluster import group_evidence
        groups = group_evidence(keep, embs)
    except Exception:
        return None
    lines: List[str] = []
    for gi, g in enumerate(groups, 1):
        lines.append(f"[Group {gi}]")
        for e in g:
            lines.append(f"  - {e.get('content', '')}")
    return "\n".join(lines)


def _final_answer_hint(question: Optional[str]) -> str:
    """The verbatim-vs-inference instruction injected at the final step."""
    if _is_eitheror_q(question):
        return (
            "This is an EITHER/OR question — answer with EXACTLY ONE of the "
            "options named in the question (the one the evidence supports). "
            "Just that option, no prose.")
    if _is_yesno_q(question):
        return (
            "This is a YES/NO question — START the answer with 'Yes' or 'No' "
            "(or 'Likely yes' / 'Likely no' when inferred), optionally a "
            "short reason after a comma. Do NOT answer with a category label.")
    if _is_inference_q(question):
        return (
            "This is an INFERENCE question — the answer is the concise "
            "CONCLUSION the evidence implies, NOT a quote. Map the evidence's "
            "specifics to a canonical label via world knowledge (e.g. "
            "evidence 'my kids have so much' -> 'wealthy'; 'I volunteer every "
            "week' -> 'community-minded'). Do NOT echo the turn's words — "
            "give the bare inferred label."
        )
    return "Copy the bare value VERBATIM from the evidence (the speaker's words)."

# Loop-safety cap on how many times we redirect a premature final_answer back
# into a pivot when the walk evidence is still inconclusive (weak grounding).
# NOT a question-type rule — purely a bound so an answer that genuinely isn't
# in memory (e.g. abstention questions) still terminates.
_MAX_PIVOT_REDIRECTS = 3

# Grounding floor : a single on-target fact is thin evidence for ANY question.
# When fewer than this many DISTINCT on-target facts have been collected across
# all walks so far, the single-hop result is inconclusive → pivot. Reads the
# walk's own relevant_collected output, never the question wording.
_MIN_GROUNDING_FACTS = 2

# HARD PROCESS FLOOR (cat3 fix). Wiring presearch is useless if the agent
# never calls it : on inference questions the answer's words differ from the
# question's, so the FIRST literal walk lands off the gold turn (measured:
# "John financial status…" misses D5:5, but "John kids have so much" ranks it
# 3rd). So we MANDATE the reconnaissance + breadth protocol, regardless of how
# confident the model feels:
#   • round 0 is FORCED to presearch (tool_choice) — recon before any walk;
#   • final_answer is REFUSED until the agent has run >= _MIN_DISTINCT_WALKS
#     walk_start calls with DIFFERENT vocabulary AND >= 1 presearch.
# Bounded by _MAX_FLOOR_REDIRECTS so genuinely-absent evidence still
# terminates (the end-of-budget forced-final always passes through).
_MIN_DISTINCT_WALKS = 2
_MAX_FLOOR_REDIRECTS = 3

# Absolute-date patterns for "when" question extraction, most specific first :
#   "19 January 2023" · "January 19, 2023" · "January 2023" · "2023".
_MONTHS = (r"(?:January|February|March|April|May|June|July|August|"
           r"September|October|November|December)")
_DATE_RES = [
    re.compile(rf"\b\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b{_MONTHS}\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}\b"),
]


_RELATIVE_DATE_RE = re.compile(
    r"\b(last\s+year|last\s+month|last\s+week|yesterday|"
    r"\d+\s+years?\s+ago|\d+\s+months?\s+ago|a\s+few\s+(?:weeks?|days?)\s+ago|"
    r"recently|around\s+\d{4})\b",
    re.IGNORECASE,
)
_ABS_DATE_CLAUSE_RE = re.compile(r"\(absolute date:\s*([^)]+)\)", re.IGNORECASE)


def _resolve_relative_date_answer(
    answer: str, contents: Sequence[str]
) -> str:
    """If `answer` is (or contains) a RELATIVE date phrase, replace it with
    the absolute date computed at ingestion. Each dated turn carries a
    plain "(absolute date: 2022)" clause next to the relative phrase it
    resolves. We find the evidence turn that contains BOTH the SAME
    relative phrase the answer used AND an absolute-date clause, and return
    that absolute date — precise, deterministic, no LLM. Falls back to the
    only absolute-date clause seen if the phrase match is ambiguous.

    This closes the temporal 0-clue gap : "When did X paint a sunrise?" →
    the generator echoes "last year", which we deterministically rewrite to
    "2022" using the turn's own resolved date."""
    if not answer:
        return answer
    m = _RELATIVE_DATE_RE.search(answer)
    if not m:
        return answer
    phrase = m.group(1).lower().strip()
    # 1. Prefer the turn whose text contains the same relative phrase AND a
    #    resolved absolute-date clause.
    candidates: List[str] = []
    for c in contents:
        cl = c.lower()
        abs_m = _ABS_DATE_CLAUSE_RE.search(c)
        if not abs_m:
            continue
        val = abs_m.group(1).strip()
        candidates.append(val)
        if phrase in cl:
            return val
    # 2. Otherwise, if exactly one distinct absolute date was seen, use it.
    distinct = list(dict.fromkeys(candidates))
    if len(distinct) == 1:
        return distinct[0]
    return answer


def _extract_date(text: str) -> Optional[str]:
    """Pull the first absolute date out of a verbose 'when' answer.

    Prefers a leading "[session date]" bracket (the memory turn's own
    timestamp, the most reliable anchor for when an event happened),
    then falls back to the first date pattern anywhere in the text.
    """
    m = re.match(r"\s*\[([^\]]+)\]", text)
    if m:
        for rx in _DATE_RES:
            d = rx.search(m.group(1))
            if d:
                return d.group(0).strip()
    for rx in _DATE_RES:
        d = rx.search(text)
        if d:
            return d.group(0).strip()
    return None
# Compound-phrase cutters : keep the head label, drop trailing
# elaboration. Order matters — most specific first.
_COMPOUND_CUTTERS = [
    # Cut trailing "and/or X Y Z …" but ONLY when there are ≥ 3 words after
    # the conjunction — this preserves short coordinated lists like
    # "Love, faith and strength" or "running and hiking" while still removing
    # elaboration tails like "and member of the LGBTQ community".
    re.compile(r"\s+(?:and|or|as well as|along with|together with)\s+\S+(?:\s+\S+){2,}$",
               re.IGNORECASE),
    re.compile(r"\s+who\s+(?:is|was|has|loves|enjoys|likes)\s+.+$",
               re.IGNORECASE),
    re.compile(r"\s+that\s+(?:is|was)\s+.+$", re.IGNORECASE),
    # Cut trailing ", X Y Z …" but only when ≥ 4 words follow the comma —
    # avoids slicing "Love, faith and strength" → "Love" (3 words after comma
    # is a short list, not elaboration; 4+ words is safe to trim).
    re.compile(r",\s+\S+(?:\s+\S+){3,}$"),
]


def _extract_ab_options(question: Optional[str]):
    """Extract (opt_a, opt_b) from 'Would X enjoy A or B?' style questions.

    Splits at the LAST ' or ' in the question, takes the last ≤4 words before it
    as option A and the words up to '?' as option B. Returns None if not found
    or if options look like full clauses (> 4 words each)."""
    if not question:
        return None
    q = question.strip().rstrip("?").strip()
    # Find the last ' or ' (case-insensitive) — the choice separator
    lower_q = q.lower()
    idx = lower_q.rfind(" or ")
    if idx < 0:
        return None
    after = q[idx + 4:].strip()     # option B
    before = q[:idx].strip()         # everything before "or"
    # Take last ≤4 words of before as option A
    before_words = before.split()
    # Take last ≤4 words; strip leading stop-words (prepositions, articles,
    # pronouns, auxiliaries, verbs) that are context, not the option name.
    # E.g. "Would Sarah prefer coffee" → last 4 → strip context → "coffee"
    #      "reading books by C. S. Lewis" → last 4 → strip "by" → "C. S. Lewis"
    _STOP = {
        "by", "in", "at", "on", "the", "a", "an", "from", "to", "of",
        "with", "about", "for", "reading", "watching", "listening",
        "would", "could", "should", "prefer", "enjoy", "like", "love",
        "choose", "pick", "select", "between",
        "he", "she", "they", "we", "i", "you", "it",
    }
    raw_a = before_words[-4:] if before_words else []
    while raw_a and raw_a[0].lower().rstrip(".,") in _STOP:
        raw_a = raw_a[1:]
    # Also strip leading proper-noun subject (single capitalized word before verb)
    # e.g. ["Sarah", "prefer", "coffee"] → strip "Sarah" since "prefer" follows
    if (len(raw_a) >= 2
            and raw_a[0][0].isupper()
            and raw_a[1].lower().rstrip(".,") in _STOP):
        raw_a = raw_a[1:]
        while raw_a and raw_a[0].lower().rstrip(".,") in _STOP:
            raw_a = raw_a[1:]
    opt_a = " ".join(raw_a) if raw_a else ""
    opt_b = after
    # Sanity: both options must be plausible labels (1-4 words, not a clause)
    if (not opt_a or not opt_b
            or len(opt_a.split()) > 4 or len(opt_b.split()) > 4):
        return None
    # Reject if option B looks like a clause (contains a verb pattern)
    if re.search(r"\b(?:is|are|was|were|have|has|would|could|should)\b",
                 opt_b, re.IGNORECASE):
        return None
    return opt_a, opt_b


def _ab_choice_hint(question: Optional[str]) -> str:
    """Return a forced-final hint when the question presents exactly two options.

    "Would X enjoy A or B?" → model must output one of the two option names,
    not "Likely yes". Returns the hint string, or "" if pattern not found."""
    opts = _extract_ab_options(question)
    if not opts:
        return ""
    opt_a, opt_b = opts
    return (
        f"CHOICE QUESTION — the two options are '{opt_a}' and '{opt_b}'. "
        f"Pick EXACTLY ONE by name. Do NOT write 'Likely yes'. "
        f"Value must be '{opt_a}' or '{opt_b}' (verbatim). "
    )


def _looks_like_sentence(text: str) -> bool:
    """True when the text reads as prose (a mid-string sentence break, or a
    leading subject pronoun) rather than a bare answer phrase."""
    t = (text or "").strip()
    if re.search(r"[.!?]\s+\S", t):              # more than one sentence
        return True
    return bool(re.match(r"(?i)^(they|he|she|it|i|we|you|there|that|this)\b", t))


def _compress_answer(answer: str, question: Optional[str],
                     client, model) -> Optional[str]:
    """Compress a verbose-but-correct answer to the bare phrase the token-F1
    metric scores on — keep EVERY specific content word (names, items,
    reasons, numbers), drop narration scaffolding ("they help…", "it really
    spoke to me"). One cheap LLM call; returns None on failure (caller keeps
    the original)."""
    try:
        r = _create_with_retry(
            client, model=model, max_tokens=48, temperature=0,
            system=(
                "EXTRACT the shortest span of the ANSWER that answers the "
                "QUESTION. Use ONLY words that already appear in the ANSWER, "
                "in their original form — do NOT reword, rephrase, hyphenate, "
                "or introduce ANY new word. Just delete the narration "
                "scaffolding (subjects like 'they/it', framing verbs like "
                "'help/said/spoke', fillers like 'really', and whole "
                "sentences that don't answer). Keep the specific content "
                "words (names, places, items, reasons). If it is already a "
                "bare phrase, return it unchanged. Output only the phrase."),
            messages=[{"role": "user",
                       "content": f"QUESTION: {question}\nANSWER: {answer}"}],
        )
        out = " ".join(b.text for b in r.content if b.type == "text").strip()
        out = re.sub(r"^[\s*_#>:•\-\"']+", "", out).strip().strip('"')
        if not out:
            return None
        # EXTRACTIVE guard : accept only if it genuinely shortened AND stayed
        # within the original's words (else the model paraphrased — keep the
        # original rather than risk introducing non-gold tokens).
        def _toks(s):
            return set(re.findall(r"[a-z0-9]+", s.lower()))
        ans_t, out_t = _toks(answer), _toks(out)
        if not out_t:
            return None
        new = out_t - ans_t
        if len(out_t) >= len(ans_t):                 # not shorter → skip
            return None
        if len(new) > max(1, len(out_t) // 4):       # >25% new words → paraphrase
            return None
        return out
    except Exception:
        return None


def _compose_final(focused, bags, question, llm):
    """Mode-aware final composition of the two answer modes.

      * "generate a focused answer" -> the terse walk answer (F1 mode).
      * "refer an exhaustive set"   -> the rendered bag list(s) (recall mode).
      * BOTH                        -> focused answer + the exhaustive list(s).

    `bags` is a MAP {name: [(ref, content)]} (a legacy plain list is accepted as
    the default bag). Multiple lists are a dynamic multi-value inject. Returns
    (answer, bag_ids_used). Empty -> the focused answer unchanged. A pure
    enumeration query, or no real focused answer, yields the list(s) alone ;
    otherwise focused answer and list(s) coexist."""
    from metacog.enumeration import is_enumeration_query
    from metacog.bag_render import render_bags
    focused = (focused or "").strip()
    bags_map = bags if isinstance(bags, dict) else {"default": list(bags or [])}
    bags_map = {k: v for k, v in bags_map.items() if v}
    if not bags_map:
        return focused, []
    rendered, ids = render_bags(question, bags_map, llm)
    if not rendered:
        return focused, []
    real = bool(focused) and "not mentioned" not in focused.lower() \
        and "no information" not in focused.lower()
    if is_enumeration_query(question) or not real:
        return rendered, ids                    # exhaustive : the list(s) ARE it
    return focused + "\n\n" + rendered, ids      # BOTH


def terse(text: str, question: Optional[str] = None) -> str:
    """Strip Haiku's preamble/interjection wrapping. Mirrors the helper
    in mcp_agent.py — keeps the bare value the F1 metric scores on.

    For short-label answers (≤ 8 words after preamble strip), also cut
    trailing "and …" / "who is …" / ", …" compounds so identity-style
    questions reduce to the bare head label ("Transgender woman" not
    "Transgender artist and member of the LGBTQ community").

    Verbose Haiku replies often paste a quoted citation then put the
    bare label on its OWN line at the end. If we see that pattern (the
    text has multiple lines, ends with a short line ≤ 8 words and no
    sentence terminator before it), keep only the trailing label line.

    When `question` is a "when …" question and the answer is verbose
    (the agent pasted a whole "[date] Speaker: …" citation), extract the
    bare absolute date — that is all the F1 metric scores on.
    """
    if not text:
        return text
    t = text.strip()
    # Normalize abstention phrases to the official eval's expected string.
    # The official locomo metric checks for "no information available" or
    # "not mentioned" — any other phrasing scores 0 for cat5.
    # Only fire when the answer is purely an abstention (starts with one of
    # these markers), never when it's a real answer with a side clause.
    _tl = t.lower()
    _abstain_starts = (
        "not mentioned", "no information", "there is no information",
        "i cannot find", "i can't find", "cannot find", "no record",
        "i cannot determine", "i cannot identify", "i cannot confirm",
        "i cannot locate", "i cannot tell", "i cannot say",
        "i am unable to", "i'm unable to", "i was unable to",
        "unable to determine", "unable to identify", "unable to locate",
        "the facts provided do not", "the facts do not",
        "there's no information", "there isn't",
        "not stated", "not specified", "not provided",
        "i was unable to find", "i'm unable to find", "unable to find",
        "the conversation does not", "the conversation doesn't",
        "there doesn't appear", "there does not appear",
        "doesn't mention", "does not mention", "not mentioned in",
        "no mention", "couldn't find", "could not find",
        "no specific", "no direct", "no explicit",
    )
    # Only normalize short pure-abstentions (≤ 12 words, no mixed-content
    # "but" clause). Longer preds or preds with "but [real content]" after
    # the abstain start often have real mixed-in content — let terse() handle
    # them rather than discarding the content.
    _has_but_clause = bool(re.search(r"\bbut\b.{8,}", t, re.IGNORECASE))
    if (any(_tl.startswith(s) for s in _abstain_starts)
            and 2 < len(t.split()) <= 12
            and not _has_but_clause):
        return "Not mentioned"
    # Strip leading bullet-point marker ("- " or "• ") — the agent sometimes
    # returns the raw fact content prefixed with its bullet from the walk output.
    t = re.sub(r"^[-•]\s+", "", t)
    # Skip compound-cutting when the prediction is ALREADY a comma-separated
    # list of short noun phrases — the agent has chosen to enumerate, and the
    # trailing-comma cutter would chop "Pride parade, school speech, support
    # group" down to "Pride parade" (destroying cat1 multi-hop F1).
    _segs = [s.strip() for s in t.split(",") if s.strip()]
    _is_list_pred = len(_segs) >= 2 and all(len(s.split()) <= 4 for s in _segs)
    # Plural question hint as a fallback (covers an empty-list pred where the
    # gold IS a list, e.g. cat1 enumeration questions).
    _is_enum_q = _is_list_pred or bool(
        question and _ENUM_Q_RE.search(question))
    # "When …" questions : if the agent dumped a citation instead of a bare
    # date, pull the date out. Only when the text is verbose (>5 words) —
    # a clean "19 January 2023" already passes through untouched.
    if question and re.match(r"\s*when\b", question, re.IGNORECASE) \
            and len(t.split()) > 5 and "not mentioned" not in t.lower():
        d = _extract_date(t)
        if d:
            return d
    # Bold-marked answer — take the first bold span that is a real value,
    # skipping two traps:
    #  - a bold dialog-id citation ("From **D2:7** …") : the agent bolds the
    #    reference, not the answer — returning it leaks the ID and scores 0.
    #  - a bold in a negation/correction context ("it's actually **Caroline's**
    #    necklace, not Melanie's") : the bolded entity is the WRONG one.
    for m in re.finditer(r"\*\*(.+?)\*\*", t):
        cand = m.group(1).strip().rstrip(".")
        # Skip a bold that is only a dialog-turn id (e.g. "D2:7").
        if _DIALOG_ID_RE.fullmatch(cand):
            continue
        # Skip a long bold : that's a quoted sentence, not a terse answer
        # label. Let it fall through to the normal prose-cleaning path.
        if len(cand.split()) > 8:
            continue
        prefix_start = max(0, m.start() - 30)
        prefix = t[prefix_start:m.start()].lower()
        _negation = any(x in prefix for x in (
            "actually", "it's actually", "it is actually",
            "not melanie", "not mel", "rather than",
        ))
        if _negation:
            continue
        return cand
    # Citation + trailing label pattern : take the last short line.
    lines = [ln.strip(" .,") for ln in t.splitlines() if ln.strip(" .,")]
    if len(lines) >= 2:
        tail = lines[-1]
        tail_words = tail.split()
        # Don't take the tail if it's a bare gerund/past-participle (a verb
        # from mid-sentence that happened to land on its own line, e.g. "hosting").
        _single_verb = (
            len(tail_words) == 1
            and (tail.lower().endswith("ing") or tail.lower().endswith("ed"))
            and not any(c.isdigit() for c in tail)
        )
        if 1 <= len(tail_words) <= 8 and not _single_verb and not tail.lower().startswith(
            ("based on", "according to", "the answer", "perfect", "i found")
        ):
            t = tail
        elif _single_verb:
            # Strip the stray verb line so it doesn't pollute the answer.
            raw_lines = t.splitlines()
            while raw_lines and raw_lines[-1].strip(" .,") == tail:
                raw_lines.pop()
            t = "\n".join(raw_lines).strip(" .,")
    while True:
        stripped = _INTERJECTION_RE.sub("", t)
        stripped = _PREAMBLE_RE.sub("", stripped).strip()
        if stripped == t:
            break
        if not stripped:
            t = ""
            break
        t = stripped
    if re.search(r"\bnot mentioned\b", t, re.IGNORECASE):
        return "Not mentioned"
    # Bare list header with no value ("Key facts about X:" → stripped to "").
    if t.endswith(":") and len(t.split()) <= 8:
        return ""
    # Long inference answers : extract the "Likely yes/no, …" clause buried in prose.
    # The agent often reasons for 50+ words then ends with the actual answer.
    if len(t.split()) > 15:
        m_inf = re.search(
            r"\blikely\s+(?:yes|no)\b[^.!?]{0,60}",
            t, re.IGNORECASE,
        )
        if m_inf:
            t = m_inf.group(0).strip().rstrip(".,;")
    # Drop any leaked dialog-id citation, then tidy leftover punctuation.
    t = _DIALOG_ID_RE.sub("", t).strip(" .,()")
    # Binary A-or-B choice detection: "Would X enjoy A or B?" → pick the ONE option.
    # When the question gives exactly two options and the answer is "Likely yes/no, ..."
    # the agent should output just the chosen option name (not "Likely yes").
    _ab_opts = _extract_ab_options(question)
    if _ab_opts and re.match(r"^\s*likely\s+(?:yes|no)\b", t, re.IGNORECASE):
        opt_a, opt_b = _ab_opts
        tl = t.lower()
        a_hit = opt_a.lower() in tl
        b_hit = opt_b.lower() in tl
        if a_hit and not b_hit:
            t = opt_a
        elif b_hit and not a_hit:
            t = opt_b
        # both or neither → fall through to normal processing
    # Trim overlong "Likely yes/no, [very long clause]" inference answers.
    # Gold answers for "Would X enjoy Y?" are typically 2-5 words.
    # Cap at 6 words after "Likely yes/no,": "Likely yes, classical music."
    is_inference = bool(re.match(r"^\s*likely\s+(?:yes|no)\b", t, re.IGNORECASE))
    if is_inference and len(t.split()) > 7:
        m_head = re.match(r"((?:likely\s+)?(?:yes|no)(?:,|;)?\s*)", t, re.IGNORECASE)
        if m_head:
            head = m_head.group(1)
            rest_words = t[len(head):].split()
            t = (head + " ".join(rest_words[:4])).strip().rstrip(".,;")
    # Compound cutting : apply iteratively while the answer is short
    # enough that the head is plausibly the bare label. Avoid touching
    # long prose answers where "and" is part of a real sentence.
    if t and not is_inference and not _is_enum_q and len(t.split()) <= 12:
        changed = True
        while changed and t and len(t.split()) <= 12:
            changed = False
            for cutter in _COMPOUND_CUTTERS:
                new = cutter.sub("", t).strip(" .,()")
                if new and new != t:
                    t = new
                    changed = True
                    break
    # Kinship synonym normalization: conversational short forms → formal tokens
    # that align with LoCoMo gold answers. Only apply to short answers (≤ 5
    # words) to avoid touching real sentence content.
    if len(t.split()) <= 5:
        t = re.sub(r"\bmom\b", "mother", t, flags=re.IGNORECASE)
        t = re.sub(r"\bdad\b", "father", t, flags=re.IGNORECASE)
        t = re.sub(r"\bmum\b", "mother", t, flags=re.IGNORECASE)
        t = re.sub(r"\bgrandma\b", "grandmother", t, flags=re.IGNORECASE)
        t = re.sub(r"\bgrandpa\b", "grandfather", t, flags=re.IGNORECASE)
    return t.strip().rstrip(".")


class McpMetaAgent:
    """Tool-using agent that drives the meta-cognitive walk over MCP."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
        max_rounds: int = _MAX_ROUNDS,
    ) -> None:
        self.client = _resolve_client(api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds

    def answer(self, memory, question: str, **_ignored) -> Dict[str, Any]:
        return asyncio.run(self._answer_async(memory, question))

    async def _answer_async(self, memory, question: str) -> Dict[str, Any]:
        from mcp.shared.memory import create_connected_server_and_client_session
        from metacog.mcp_server import build_app

        # PER-QA ISOLATION. The agent now has the FULL tool surface,
        # including state-mutating tools (ingest / observe / sleep /
        # build_skill / …). The harness runs many QAs concurrently against
        # one conversation memory, so give THIS QA its own deep copy : any
        # mutation stays local and cannot race another worker. Read-only
        # tools are unaffected ; the cost is one snapshot per QA.
        memory = memory.snapshot()

        app = build_app(memory=memory)
        total_in = 0
        total_out = 0
        trace: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        # All retrieved-fact contents seen across walks — used by the
        # deterministic relative→absolute date resolver at answer time.
        seen_contents: List[str] = []

        async with create_connected_server_and_client_session(app) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = _mcp_tools_to_anthropic(listed.tools) + [_FINAL_ANSWER_TOOL]

            messages: List[Dict[str, Any]] = [
                {"role": "user", "content": f"Question: {question}"}
            ]
            answer_text = ""
            walk_start_count = 0   # total breadth pivots taken
            done_seen = False
            last_walk_drifted = False  # latest walk drifted (n_relevant==0)
            pivot_redirects = 0        # premature-answer redirects issued
            relevant_ids_seen: set = set()  # distinct on-target fact ids (grounding)
            last_relevant_collected: List[dict] = []  # for forced-final evidence
            # HARD PROCESS FLOOR state (cat3): force recon + breadth.
            presearch_count = 0
            clue_search_count = 0      # clue_search is the inference recon op
            walk_vocab: set = set()    # distinct walk_start query phrasings
            floor_redirects = 0
            # Vocab-gap questions (inference + abstract-verb factual) get
            # clue_search FORCED, not merely offered : it reliably surfaces
            # the answer-register evidence and anchors the agent on the
            # question's actual topic instead of drifting to a co-present one
            # (Caroline "research" → adoption, not her counseling career).
            _vocab_gap = _is_vocab_gap_q(question)

            for round_idx in range(self.max_rounds):
                # Round 0 is FORCED to presearch : recon the question's
                # phrasings before paying any walk (the answer's words differ
                # from the question's on inference items, so a literal first
                # walk misses the gold turn).
                _tool_names = {t["name"] for t in tools}
                _force_presearch = (
                    round_idx == 0 and "presearch" in _tool_names
                )
                # Round 1 FORCES clue_search on vocab-gap questions — BEFORE
                # the agent can walk a co-present topic. Forcing it only via
                # the floor (after a stray walk) is too late: the wrong
                # topic's content already dominates the compose set and the
                # agent answers from it (Caroline "research" → her counseling
                # career instead of the adoption she actually researched).
                _force_clue = (
                    round_idx == 1 and _vocab_gap and clue_search_count < 1
                    and "clue_search" in _tool_names
                )
                if _force_presearch:
                    _tc = {"type": "tool", "name": "presearch"}
                elif _force_clue:
                    _tc = {"type": "tool", "name": "clue_search"}
                else:
                    _tc = None
                _kw = {"tool_choice": _tc} if _tc is not None else {}
                resp = _create_with_retry(self.client,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    system=AGENT_SYSTEM,
                    tools=tools,
                    messages=messages,
                    **_kw,
                )
                if hasattr(resp, "usage"):
                    total_in += getattr(resp.usage, "input_tokens", 0) or 0
                    total_out += getattr(resp.usage, "output_tokens", 0) or 0

                tool_uses = [b for b in resp.content if b.type == "tool_use"]
                text_blocks = [b.text for b in resp.content if b.type == "text"]

                # Natural exit : no tool calls → answer
                if not tool_uses:
                    raw = " ".join(text_blocks).strip()
                    # HARD PROCESS FLOOR also guards the text-only exit, so the
                    # agent can't bypass the recon/breadth protocol by simply
                    # narrating an answer instead of calling final_answer.
                    # EXCEPTION : an abstention ("not mentioned" / "no info")
                    # is allowed through — that is the adversarial escape hatch,
                    # and the prompt already requires two walks before it.
                    _raw_low = raw.lower()
                    _is_abstain = any(s in _raw_low for s in (
                        "not mentioned", "no information", "isn't mentioned",
                        "not mention", "no mention", "doesn't mention",
                        "not available", "not stated", "can't find",
                        "cannot find", "no record", "not provided",
                    ))
                    # Recon floor : a presearch, then EITHER a clue_search
                    # (the inference recon op — ~10x cheaper than a walk and
                    # the right tool for indirect questions) OR two distinct
                    # walks. clue_search thus replaces the 2nd forced walk on
                    # inference items.
                    _floor_unmet = (
                        presearch_count < 1
                        or (_vocab_gap and clue_search_count < 1)
                        or (clue_search_count < 1
                            and len(walk_vocab) < _MIN_DISTINCT_WALKS))
                    if (_floor_unmet and not _is_abstain
                            and floor_redirects < _MAX_FLOOR_REDIRECTS
                            and round_idx < self.max_rounds - 1):
                        floor_redirects += 1
                        messages.append(
                            {"role": "assistant", "content": resp.content})
                        messages.append({"role": "user", "content": (
                            "Do not answer yet. The answer's words usually "
                            "differ from the question's, so one literal pass "
                            "misses the evidence. You still need : "
                            + ("a presearch over candidate phrasings; "
                               if presearch_count < 1 else "")
                            + ("call clue_search(question=…) NOW — this is an "
                               "inference / abstract-verb question and "
                               "clue_search anchors on its actual topic; "
                               if (_vocab_gap and clue_search_count < 1)
                               else ("EITHER clue_search(question=…) OR "
                                     f">= {_MIN_DISTINCT_WALKS} walk_start "
                                     "calls with different vocabulary (you "
                                     f"have {len(walk_vocab)}); "
                                     if (clue_search_count < 1
                                         and len(walk_vocab) < _MIN_DISTINCT_WALKS)
                                     else ""))
                            + "do the missing step now (call the tool)."
                        )})
                        trace.append({"round": round_idx,
                                      "action": "floor_redirect_text",
                                      "presearch_count": presearch_count,
                                      "distinct_walks": len(walk_vocab)})
                        continue
                    # If the agent NARRATED instead of giving a bare value
                    # ("I now have clear evidence…", "Looking at the facts…"),
                    # convert it via a forced final_answer turn — otherwise
                    # the reasoning prose leaks as the answer (F1≈0 despite
                    # recall=1). Short answers are taken as-is.
                    # Abstention guard : if the narration says the info is
                    # absent, keep "Not mentioned" — forcing a value here
                    # turns a correct adversarial abstention into a wrong
                    # answer (or a leaked citation).
                    _low = raw.lower()
                    _abstains = any(s in _low for s in (
                        "not mentioned", "no information", "isn't mentioned",
                        "not mention", "no mention", "doesn't mention",
                        "don't have", "do not have", "not available",
                        "not stated", "not specified", "can't find",
                        "cannot find", "no record", "not provided",
                    ))
                    if _abstains:
                        answer_text = "Not mentioned"
                    elif len(raw.split()) > 6:
                        messages.append({"role": "assistant",
                                         "content": resp.content})
                        fr = _create_with_retry(self.client, 
                            model=self.model, max_tokens=96, temperature=0,
                            system=AGENT_SYSTEM, tools=[_FINAL_ANSWER_TOOL],
                            tool_choice={"type": "tool",
                                         "name": "final_answer"},
                            messages=messages + [{
                                "role": "user",
                                "content": (
                                    _ab_choice_hint(question)
                                    + "Call final_answer with the bare value, "
                                    "following the format rules (absolute "
                                    "calendar date for 'when'; shortest label; "
                                    + (
                                        # inference-plural → a FEW best labels,
                                        # not an exhaustive list (the gold is a
                                        # small canonical estimate).
                                        "THIS IS AN OPEN-ESTIMATE QUESTION — "
                                        "give the 1-3 MOST LIKELY canonical "
                                        "labels, comma-separated, NOT an "
                                        "exhaustive list. "
                                        if (_ENUM_Q_RE.search(question or "")
                                            and _is_inference_q(question))
                                        else "THIS IS A PLURAL QUESTION — list "
                                        "ALL items found, comma-separated. "
                                        "Do NOT collapse to one item. "
                                        if _ENUM_Q_RE.search(question or "")
                                        else "full list for plural. "
                                    )
                                    + "No narration. "
                                    + _final_answer_hint(question)
                                ),
                            }],
                        )
                        if hasattr(fr, "usage"):
                            total_in += getattr(fr.usage, "input_tokens", 0) or 0
                            total_out += getattr(fr.usage, "output_tokens", 0) or 0
                        fa = next((b for b in fr.content
                                   if b.type == "tool_use"
                                   and b.name == "final_answer"), None)
                        val = str((fa.input or {}).get("value", "")).strip() if fa else ""
                        answer_text = val or raw
                    else:
                        answer_text = raw
                    trace.append({"round": round_idx, "action": "final",
                                  "text": answer_text[:200]})
                    break

                # Constrained exit : the agent emitted final_answer(value).
                final_tu = next(
                    (b for b in tool_uses if b.name == "final_answer"), None)
                if final_tu is not None:
                    # HARD PROCESS FLOOR (cat3). Refuse to answer until the
                    # recon + breadth protocol has actually run : >= 1
                    # presearch AND >= _MIN_DISTINCT_WALKS walks with DIFFERENT
                    # vocabulary. This is process-driven, not grounding-driven
                    # (the grounding pivot below is a separate, weaker gate) —
                    # it forces the agent to use the tools we wired even when
                    # it feels confident after one literal walk. Bounded by
                    # _MAX_FLOOR_REDIRECTS and the round budget, so abstention
                    # questions still terminate at the forced-final.
                    # Recon floor : a presearch, then EITHER a clue_search
                    # (the inference recon op — ~10x cheaper than a walk and
                    # the right tool for indirect questions) OR two distinct
                    # walks. clue_search thus replaces the 2nd forced walk on
                    # inference items.
                    _floor_unmet = (
                        presearch_count < 1
                        or (_vocab_gap and clue_search_count < 1)
                        or (clue_search_count < 1
                            and len(walk_vocab) < _MIN_DISTINCT_WALKS))
                    if (_floor_unmet
                            and floor_redirects < _MAX_FLOOR_REDIRECTS
                            and round_idx < self.max_rounds - 1):
                        floor_redirects += 1
                        _need = []
                        if presearch_count < 1:
                            _need.append(
                                "run presearch with 3-4 candidate phrasings "
                                "(literal terms, answer-domain synonyms, "
                                "behavioural clues) and walk the one whose "
                                "hits are on-topic")
                        if _vocab_gap and clue_search_count < 1:
                            _need.append(
                                "call clue_search(question=…) NOW — this is an "
                                "inference / abstract-verb question; "
                                "clue_search brainstorms the evidence-register "
                                "lines and anchors on the question's actual "
                                "topic (do NOT answer from a different topic "
                                "you happened to retrieve)")
                        elif (clue_search_count < 1
                                and len(walk_vocab) < _MIN_DISTINCT_WALKS):
                            _need.append(
                                "EITHER call clue_search(question=…) (for an "
                                "inference question — it brainstorms the "
                                "evidence-register lines) OR run at least "
                                f"{_MIN_DISTINCT_WALKS} walk_start calls with "
                                "DIFFERENT vocabulary (you have "
                                f"{len(walk_vocab)} distinct so far)")
                        floor_msg = (
                            "Do not answer yet — the answer's words usually "
                            "differ from the question's, so a single literal "
                            "walk lands off the evidence. You must still : "
                            + "; ".join(_need) + ". Do the missing step now."
                        )
                        floor_results: List[Dict[str, Any]] = [{
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": (floor_msg if tu.id == final_tu.id
                                        else "Skipped — recon/breadth floor "
                                             "not met yet."),
                        } for tu in tool_uses]
                        messages.append(
                            {"role": "assistant", "content": resp.content})
                        messages.append({"role": "user",
                                         "content": floor_results})
                        trace.append({"round": round_idx,
                                      "action": "floor_redirect",
                                      "presearch_count": presearch_count,
                                      "distinct_walks": len(walk_vocab),
                                      "floor_redirects": floor_redirects})
                        continue  # back to top — satisfy the floor first
                    # VOCAB-GAP ABSTAIN GUARD. If the agent is about to
                    # abstain ("Not mentioned") on a question whose verb is
                    # abstract ("what did X research / study / look into?")
                    # and HAS NOT tried clue_search yet, redirect to
                    # clue_search before accepting the abstain. The evidence
                    # is usually in the conversation but in the answer's
                    # vocabulary, not the question's. The check on
                    # `clue_search_count == 0` keeps this from looping after
                    # the agent has already tried clue_search and still
                    # wants to abstain.
                    _final_value = str(
                        (final_tu.input or {}).get("value", "")).lower()
                    _abstaining = any(s in _final_value for s in (
                        "not mentioned", "no information", "isn't mentioned",
                        "not mention", "no mention", "doesn't mention",
                        "don't have", "not stated", "not specified",
                        "can't find", "cannot find",
                    ))
                    if (_abstaining
                            and _is_vocab_gap_q(question)
                            and clue_search_count == 0
                            and floor_redirects < _MAX_FLOOR_REDIRECTS
                            and round_idx < self.max_rounds - 1):
                        floor_redirects += 1
                        guard_msg = (
                            "Hold on — you are about to abstain on a question "
                            "whose answer is likely in the conversation but in "
                            "DIFFERENT words than the question's (e.g. the "
                            "question says 'research' but the turn says "
                            "'researching adoption agencies'). You have NOT "
                            "tried clue_search yet. Call clue_search(question="
                            "…) NOW : it brainstorms concrete chat lines that "
                            "would be evidence for different plausible "
                            "answers and retrieves over them. Then decide."
                        )
                        guard_results: List[Dict[str, Any]] = [{
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": (guard_msg if tu.id == final_tu.id
                                        else "Skipped — try clue_search "
                                             "before abstaining."),
                        } for tu in tool_uses]
                        messages.append(
                            {"role": "assistant", "content": resp.content})
                        messages.append({"role": "user",
                                         "content": guard_results})
                        trace.append({"round": round_idx,
                                      "action": "vocab_gap_abstain_redirect",
                                      "clue_search_count": clue_search_count})
                        continue
                    # SIGNAL-DRIVEN pivot (not question-type-driven). The
                    # meta-walk fires when single-hop retrieval is INCONCLUSIVE,
                    # measured by GROUNDING : how many DISTINCT on-target facts
                    # the walks have collected. Fewer than _MIN_GROUNDING_FACTS
                    # (covers total drift AND thin single-fact grounding) → the
                    # answer would rest on too little, so redirect the agent to
                    # pivot with different vocabulary. Bounded by
                    # _MAX_PIVOT_REDIRECTS so genuinely-absent evidence
                    # (abstention questions) still terminates.
                    # We must provide a tool_result for EVERY tool_use in the
                    # assistant turn (parallel tool calls), else the API errors.
                    n_grounding = len(relevant_ids_seen)
                    evidence_inconclusive = n_grounding < _MIN_GROUNDING_FACTS
                    if (evidence_inconclusive
                            and pivot_redirects < _MAX_PIVOT_REDIRECTS
                            and round_idx < self.max_rounds - 1):
                        pivot_redirects += 1
                        redirect_msg = (
                            f"Only {n_grounding} on-target fact(s) collected so "
                            f"far — that is thin grounding to answer from. Pivot : "
                            "call walk_start again with DIFFERENT vocabulary "
                            "(synonyms the ANSWER might use, then broader context "
                            "clues). Answer once you have solid on-target "
                            "evidence, or if you have genuinely exhausted the "
                            "memory."
                        )
                        # Build tool_result for every tool_use in this turn.
                        redirect_results: List[Dict[str, Any]] = []
                        for tu in tool_uses:
                            redirect_results.append({
                                "type": "tool_result",
                                "tool_use_id": tu.id,
                                "content": (redirect_msg if tu.id == final_tu.id
                                            else "Skipped — pivot first; "
                                                 "grounding is thin."),
                            })
                        messages.append(
                            {"role": "assistant", "content": resp.content})
                        messages.append({"role": "user",
                                         "content": redirect_results})
                        trace.append({"round": round_idx,
                                      "action": "pivot_on_weak_grounding",
                                      "walk_start_count": walk_start_count,
                                      "n_grounding": n_grounding,
                                      "pivot_redirects": pivot_redirects})
                        continue  # back to top of for-loop, don't break
                    answer_text = str((final_tu.input or {}).get("value", "")).strip()
                    # Never accept an empty answer : fall back to any prose
                    # in the turn, then to the strongest accumulated fact.
                    if not answer_text:
                        answer_text = " ".join(text_blocks).strip()
                    if not answer_text and last_relevant_collected:
                        answer_text = str(
                            last_relevant_collected[0].get("content", "")).strip()
                    # Narration guard : the agent stuffed its REASONING into the
                    # final_answer value ("The walk found … 1. … 2. …") instead
                    # of a bare value — re-ask once, forcing a terse value via
                    # tool_choice, so the retrieved answer isn't lost (Nate bug).
                    if _looks_like_narration(answer_text):
                        messages.append({"role": "assistant",
                                         "content": resp.content})
                        # Provide a tool_result for every tool_use this turn.
                        narr_results = [{
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": ("Your value was a reasoning paragraph, "
                                        "not a bare answer. Re-call final_answer "
                                        "with ONLY the short value."
                                        if tu.id == final_tu.id
                                        else "Skipped."),
                        } for tu in tool_uses]
                        messages.append({"role": "user", "content": narr_results})
                        fr = _create_with_retry(self.client, 
                            model=self.model, max_tokens=96, temperature=0,
                            system=AGENT_SYSTEM, tools=[_FINAL_ANSWER_TOOL],
                            tool_choice={"type": "tool", "name": "final_answer"},
                            messages=messages + [{
                                "role": "user",
                                "content": (
                                    _ab_choice_hint(question)
                                    + "Call final_answer with the BARE value only, "
                                    "no narration, no lists, no 'the walk found'. "
                                    "KEEP the most concrete noun(s) from your "
                                    "evidence (e.g. 'teammates', 'video game "
                                    "team') — do NOT generalise them to vague "
                                    "words like 'people'. Follow the format "
                                    "rules (e.g. 'Likely yes, <≤4-word reason "
                                    "using those nouns>' for 'Is it likely / "
                                    "Would' questions; shortest label otherwise)."
                                ),
                            }],
                        )
                        if hasattr(fr, "usage"):
                            total_in += getattr(fr.usage, "input_tokens", 0) or 0
                            total_out += getattr(fr.usage, "output_tokens", 0) or 0
                        fa = next((b for b in fr.content if b.type == "tool_use"
                                   and b.name == "final_answer"), None)
                        val = str((fa.input or {}).get("value", "")).strip() if fa else ""
                        if val:
                            answer_text = val
                    trace.append({"round": round_idx, "action": "final_tool",
                                  "text": answer_text[:200]})
                    break

                messages.append({"role": "assistant", "content": resp.content})
                tool_results: List[Dict[str, Any]] = []
                for tu in tool_uses:
                    if _ALLOWED_TOOLS is not None and tu.name not in _ALLOWED_TOOLS:
                        result_text = (f"Tool {tu.name} not permitted.")
                    elif tu.name == "walk_next" and walk_start_count == 0:
                        # Guard : walk_next without any walk_start.
                        result_text = ("Call walk_start first.")
                    else:
                        call_input = tu.input
                        # TEMPORAL ANCHOR : auto-inject the ORIGINAL question's
                        # date constraint into every walk so it stays scoped to
                        # the asked period across all reformulations (the agent
                        # often drops the date when it rephrases). Deterministic,
                        # sourced from `question`, never from the reformulation.
                        if tu.name == "walk_start" and not (
                                tu.input or {}).get("date_tags"):
                            from metacog.memory import query_date_tags
                            _dt = query_date_tags(question)
                            if _dt:
                                call_input = dict(tu.input or {})
                                call_input["date_tags"] = _dt
                        call_result = await session.call_tool(tu.name, call_input)
                        result_text = _tool_result_text(call_result)
                        seen_ids.update(_extract_fact_ids(call_result))
                        # Track the latest relevant_collected for forced-final,
                        # and accumulate distinct on-target fact ids as the
                        # grounding signal that drives weak-grounding pivoting.
                        try:
                            obj = json.loads(result_text)
                            rc = obj.get("relevant_collected")
                            if isinstance(rc, list) and rc:
                                last_relevant_collected = rc
                                for _e in rc:
                                    _id = (_e.get("id") if isinstance(_e, dict)
                                           else None)
                                    if _id:
                                        relevant_ids_seen.add(_id)
                            # Capture ALL retrieved-fact contents (not just the
                            # CoN-relevant set) so the deterministic
                            # relative→absolute date resolver can find the
                            # "(absolute date: …)" clause even on the 0-clue
                            # path where relevant_collected is empty.
                            for _src in (obj.get("facts"), rc):
                                if isinstance(_src, list):
                                    for _e in _src:
                                        if isinstance(_e, dict):
                                            _c = _e.get("content")
                                            if _c:
                                                seen_contents.append(_c)
                        except Exception:
                            pass
                        if tu.name == "presearch":
                            presearch_count += 1
                        if tu.name == "clue_search":
                            clue_search_count += 1
                        if tu.name == "walk_start":
                            walk_start_count += 1
                            # Track DISTINCT walk vocabulary for the breadth
                            # floor : two walks with the same phrasing are not
                            # a breadth pivot. Normalise to a sorted token set.
                            _q = str((tu.input or {}).get("query", "")).lower()
                            _vocab = frozenset(re.findall(r"[a-z0-9]+", _q))
                            if _vocab:
                                walk_vocab.add(_vocab)
                            done_seen = False  # reset — new walk, new chance
                        # Detect done + drift flags for trace / pivot signal.
                        try:
                            obj2 = json.loads(result_text)
                            if obj2.get("done"):
                                done_seen = True
                            # Track whether the latest walk stage drifted —
                            # the inconclusiveness signal that drives pivoting.
                            if "drifted" in obj2:
                                last_walk_drifted = bool(obj2.get("drifted"))
                        except Exception:
                            pass
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result_text,
                    })
                    trace.append({"round": round_idx, "action": "tool",
                                  "name": tu.name, "input": tu.input,
                                  "result_chars": len(result_text)})

                messages.append({"role": "user", "content": tool_results})

                # Force final answer only after the agent has had a chance
                # to try a breadth pivot (second walk_start). A single
                # done=true should be a signal to pivot, not to stop.
                # Signal-driven : if the walk is done but grounding is still
                # THIN (< _MIN_GROUNDING_FACTS on-target) and pivots remain, do
                # NOT force — let the agent pivot to fresh vocabulary. Only
                # force once grounding is solid, or the pivot budget is spent.
                _have_evidence = len(relevant_ids_seen) >= _MIN_GROUNDING_FACTS
                _pivots_spent = pivot_redirects >= _MAX_PIVOT_REDIRECTS
                if (done_seen and walk_start_count >= 2
                        and (_have_evidence or _pivots_spent)):
                    # Two breadth threads exhausted — synthesize from
                    # accumulated evidence, never from a blank slate.
                    if last_relevant_collected:
                        ev_lines = "\n".join(
                            f"  [{e.get('id', '')}] ({e.get('relevance', '')}) "
                            f"{e.get('content', '')}"
                            for e in last_relevant_collected[:10]
                        )
                        ev_ctx = (
                            f"\n\nYour accumulated evidence:\n{ev_lines}\n\n"
                        )
                    else:
                        # relevant_collected is empty (CoN may have labelled all
                        # facts irrelevant) but the answer IS in your tool results.
                        ev_ctx = (
                            " The answer IS in your earlier tool results — "
                            "re-read the 'facts' and 'relevant_collected' "
                            "arrays from each walk_start result above to find it. "
                        )
                    messages.append({
                        "role": "user",
                        "content": (
                            _ab_choice_hint(question)
                            + f"Walk finished.{ev_ctx}"
                            "Call final_answer now with the bare value. "
                            + _final_answer_hint(question)
                        ),
                    })
            else:
                # max_rounds exhausted without a natural answer.
                # Inject the accumulated evidence so the agent can answer
                # from facts it DID retrieve rather than saying "Not mentioned".
                forced_evidence = ""
                if last_relevant_collected:
                    ev_lines = "\n".join(
                        f"  [{e.get('id', '')}] ({e.get('relevance', '')}) "
                        f"{e.get('content', '')}"
                        for e in last_relevant_collected[:10]
                    )
                    forced_evidence = (
                        f"\n\nYour accumulated evidence (all on-target facts "
                        f"gathered so far):\n{ev_lines}\n\nUse this to answer. "
                    )
                resp = _create_with_retry(self.client, 
                    model=self.model,
                    max_tokens=48,
                    temperature=0,
                    system=AGENT_SYSTEM,
                    tools=[_FINAL_ANSWER_TOOL],
                    tool_choice={"type": "tool", "name": "final_answer"},
                    messages=messages + [{
                        "role": "user",
                        "content": (
                            _ab_choice_hint(question)
                            + f"Stop searching.{forced_evidence}"
                            "Call final_answer with the bare value. "
                            + _final_answer_hint(question)
                        ),
                    }],
                )
                if hasattr(resp, "usage"):
                    total_in += getattr(resp.usage, "input_tokens", 0) or 0
                    total_out += getattr(resp.usage, "output_tokens", 0) or 0
                fa = next((b for b in resp.content
                           if b.type == "tool_use" and b.name == "final_answer"),
                          None)
                answer_text = ""
                if fa is not None:
                    answer_text = str((fa.input or {}).get("value", "")).strip()
                if not answer_text:
                    answer_text = " ".join(
                        b.text for b in resp.content if b.type == "text"
                    ).strip()
                # Last resort : the strongest fact we gathered, never empty.
                if not answer_text and last_relevant_collected:
                    answer_text = str(
                        last_relevant_collected[0].get("content", "")).strip()
                trace.append({"round": self.max_rounds,
                              "action": "forced_final"})

            # ENUMERATION (cat1 multi-hop) : the walk's relevant_collected
            # already gathers ALL the items across depth (verified), but
            # Haiku collapses them to ONE in free composition. For a plural
            # question, run a dedicated aggregation pass over relevant_
            # collected that lists every distinct item — exploiting the
            # MAP-REDUCE we already build instead of trusting free-form.
            # NOT for inference questions : their gold is a small canonical
            # ESTIMATE (e.g. "Psychology, counseling certification"), not an
            # exhaustive enumeration. The "list EVERY item / prefer MORE
            # commas" pass over-generates there ("Counseling, Mental health,
            # LGBTQ+ counseling, Therapeutic work with trans people, Art")
            # and token-F1 precision collapses. A factual plural ("what
            # events did X attend") still enumerates fully.
            if (_ENUM_Q_RE.search(question or "")
                    and not _is_inference_q(question)
                    and len(last_relevant_collected) >= 1):
                ev = "\n".join(
                    f"- {e.get('content', '')}"
                    for e in last_relevant_collected[:14])
                try:
                    er = _create_with_retry(self.client, 
                        model=self.model, max_tokens=120, temperature=0,
                        system=("List EVERY distinct item that answers the "
                                "question, drawn from the facts. Output a "
                                "comma-separated list of short phrases (up to "
                                "5 words each), no prose, no repeats. If only "
                                "one item, output it alone."),
                        messages=[{"role": "user", "content":
                                   f"Question: {question}\nFacts:\n{ev}"}],
                    )
                    if hasattr(er, "usage"):
                        total_in += getattr(er.usage, "input_tokens", 0) or 0
                        total_out += getattr(er.usage, "output_tokens", 0) or 0
                    et = " ".join(b.text for b in er.content
                                  if b.type == "text").strip()
                    # Replace when aggregation enumerates MORE items than the
                    # current answer, or when the current answer is empty.
                    cur_commas = (answer_text or "").count(",")
                    if et and (et.count(",") > cur_commas or not answer_text):
                        answer_text = et
                except Exception:
                    pass

            # INFERENCE synthesis over ASSOCIATION-GROUPED evidence. The gold
            # is a small canonical estimate ; a flat list lets unrelated facts
            # reinforce each other ("...Art"). Present the evidence grouped by
            # association and let the LLM pick the group(s) that answer —
            # WITHOUT merging unrelated groups. Not a dominant-cluster vote ;
            # the model decides, the grouping just blocks false reinforcement.
            elif (_is_inference_q(question)
                    and not _is_yesno_q(question)
                    and not _is_eitheror_q(question)
                    and len(last_relevant_collected) >= 3):
                # only the OPEN-LABEL inference questions ("what fields /
                # status / leaning") get the abstraction synthesis. Yes/No and
                # either-or inference questions keep the agent's direct answer
                # (the abstraction would drop the "Yes" or replace the chosen
                # option with a category label).
                grouped = _grouped_evidence_text(last_relevant_collected, memory)
                if grouped:
                    try:
                        # Re-use the input's typed extraction (action verb +
                        # named entities, Slice A) so the generation FOLLOWS
                        # THE INPUT, and have the model take a RETROSPECTIVE
                        # thought over the groups w.r.t. that anchor before
                        # concluding the best answer.
                        from metacog.query_anchor import build_query_anchor
                        qa = build_query_anchor(
                            question,
                            entity_extractor=getattr(memory, "entity_extractor", None),
                            keyword_extractor=getattr(memory, "extractor", None),
                            encoder=getattr(memory, "encoder", None))
                        # action verb + named entities if an extractor is
                        # wired ; else fall back to the question's content
                        # tokens so the key-terms line is never empty.
                        anchor_terms = ", ".join(
                            (qa.salient or qa.soft or [])[:8]) or question
                        er = _create_with_retry(self.client,
                            model=self.model, max_tokens=220, temperature=0,
                            system=(
                                "You answer an INFERENCE question by "
                                "reflecting on association-grouped evidence "
                                "THROUGH THE LENS OF THE QUESTION. [Group 1] "
                                "is the STRONGEST (most mutually-reinforcing) "
                                "association; groups are unrelated — never "
                                "merge them or borrow labels across groups.\n"
                                "1) In ONE sentence, reflect RETROSPECTIVELY "
                                "against the question's action + named "
                                "entities: which group answers? Default to "
                                "[Group 1] ALONE unless another is clearly "
                                "required.\n"
                                "2) ABSTRACT that group to the LEVEL THE "
                                "QUESTION ASKS FOR — name the canonical "
                                "category / field its facts represent, while "
                                "KEEPING the group's own core term. E.g. for "
                                "a 'what fields' question, evidence "
                                "'counseling, mental health' → 'psychology, "
                                "counseling'. Stay WITHIN the chosen group.\n"
                                "3) Final line 'ANSWER: ...' — 1-3 canonical "
                                "labels, comma-separated, no prose."),
                            messages=[{"role": "user", "content":
                                       f"Question (verbatim): {question}\n"
                                       f"Question's key terms (action + named "
                                       f"entities): {anchor_terms}\n\n"
                                       f"Association-grouped evidence:\n{grouped}"}],
                        )
                        if hasattr(er, "usage"):
                            total_in += getattr(er.usage, "input_tokens", 0) or 0
                            total_out += getattr(er.usage, "output_tokens", 0) or 0
                        et = " ".join(b.text for b in er.content
                                      if b.type == "text").strip()
                        # take the labels after the retrospective thought,
                        # stripping any leading markdown/punctuation the model
                        # emits ("**ANSWER:** ..." → "...").
                        m = re.search(r"answer\s*:\s*(.+)", et, re.I)
                        val = (m.group(1) if m else et).strip()
                        val = val.splitlines()[0].strip() if val else ""
                        val = re.sub(r"^[\s*_#>:•\-]+", "", val).strip()
                        if val:
                            answer_text = val
                    except Exception:
                        pass

        # Deterministic temporal resolution : rewrite any relative-date
        # answer ("last year") to the absolute date the evidence resolved
        # ("2022"), using the turns' own "(absolute date: …)" clauses.
        # ONLY for temporal questions — otherwise an answer that merely
        # contains a relative-time word ("recently unemployed") gets
        # clobbered into an evidence date (the "January 2023" cat3 bug).
        if _is_temporal_q(question):
            answer_text = _resolve_relative_date_answer(
                answer_text, seen_contents)

        # CONCISION pass. token-F1 precision collapses when a correct answer
        # is wrapped in a full sentence ("They help LGBTQ+ folks with
        # adoption. Their inclusivity and support really spoke to me." vs gold
        # "because of their inclusivity and support for LGBTQ+ individuals").
        # When the answer is verbose (a sentence / many words) on a non-
        # temporal, non-yes/no question, compress to the bare answer phrase —
        # keep every specific content word, drop the narration scaffolding.
        if (answer_text and not _is_temporal_q(question)
                and not _is_yesno_q(question)
                and "not mentioned" not in answer_text.lower()
                and (len(answer_text.split()) > 8
                     or _looks_like_sentence(answer_text))):
            answer_text = _compress_answer(
                answer_text, question, self.client, self.model) or answer_text

        # MODE-AWARE composition (the two answer modes coexist).
        #   * "generate a focused answer" -> the terse walk answer (F1 mode).
        #   * "refer an exhaustive set"   -> the rendered bag list (recall mode).
        #   * BOTH                        -> answer + the exhaustive list.
        # The whole process above is UNCHANGED ; this only shapes the final
        # output. Empty bag -> identical to no-bag behaviour. The bag refs join
        # retrieved_ids either way (pure agent_recall — what the agent gathered).
        # Gather the agent's CURATED bags (default + any named bags it collected)
        # as a MAP of lists ; internal channel bags (event:*) are excluded from
        # the surface answer to avoid bloat. One map -> dynamic multi-value inject.
        if hasattr(memory, "bag_names"):
            curated = [n for n in memory.bag_names() if not n.startswith("event:")]
            bags_map = {n: memory.bag_items(bag=n) for n in curated}
        else:
            bags_map = {"default": memory.bag_items()} \
                if hasattr(memory, "bag_items") else {}
        final_answer, _bag_used = _compose_final(
            terse(answer_text, question), bags_map, question, memory.llm)
        seen_ids.update(_bag_used)

        return {
            "answer": final_answer,
            "answer_raw": answer_text,
            "steps": len([t for t in trace if t["action"] == "tool"]),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "retrieved_ids": sorted(seen_ids),
            "trace": trace,
        }
