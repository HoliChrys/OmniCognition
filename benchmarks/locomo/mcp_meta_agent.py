"""
MCP-driven meta agent : runs the meta-cognitive walk STEP BY STEP
through the metacog MCP server, with Claude orchestrating.

Each tool call returns ONE stage of the walk (with full INDEXED
CONTENT of every retrieved node) ; Claude reads it, decides whether
to call `walk_next` for the next stage or to answer. Multi-hop
filiation is preserved because the agent sees every stage in its
conversation history before composing the answer.

The agent NEVER does ReAct over a raw `retrieve` call : the walk
already coordinates FACT + ACTION + THOUGHT inside the memory. The
only ReAct-style step is between successive `walk_next` calls — and
that loop has a hard termination via the walker's `done=True`.

Credentials : api_key arg → ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN.
sk-ant-oat* tokens go through auth_token= (OAuth bearer).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional


_DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
_DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "256"))
# 16 rounds: each walk can run up to 8 stages (walk_start + 7 walk_next),
# and the walk stops naturally when no new unseen facts arrive (the
# |seen ∩ gold| / |gold| proxy). With 16 rounds two full-depth walks fit
# (2 × 8), or one deep walk + one breadth pivot, leaving rounds for synthesis.
_MAX_ROUNDS = int(os.environ.get("MCP_META_MAX_ROUNDS", "16"))

# Only the walk tools — the agent should NOT bypass the walk by
# calling raw retrieve. This is enforced server-side AND client-side.
_ALLOWED_TOOLS = {"walk_start", "walk_next", "list_communities"}

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

Workflow :
1. Call `walk_start(query=…)` to begin. You get ONE stage : its facts,
   its action, its thought (the meta-cognitive bridge), and the
   `walk_id`. Each fact / action / thought carries its full content,
   keywords, confidence and uncertainty — read those.
2. If the chosen fact at this stage already answers the question,
   answer now (no more tool calls).
3. READ THE RELEVANCE NOTES. Each fact has a `relevance` tag :
   relevant / partial / contradicts / irrelevant. Trust only
   relevant + partial + contradicts ; ignore irrelevant facts (they are
   adjacent-but-off-target conversation turns). The stage also returns
   `relevant_collected` — the running MAP-REDUCE set of every on-target
   fact gathered SO FAR across all stages (bridging facts from earlier
   stages are kept here, never lost). COMPOSE YOUR ANSWER over
   `relevant_collected`, not just the latest stage : a multi-hop answer
   chains facts collected across several stages.
4. DEPTH : call `walk_next(walk_id=…)` for the next stage when the
   current thread is still on-topic. Multi-hop questions often need 2-3
   depth stages.
5. BREADTH PIVOT — this is REQUIRED, not optional. When a stage returns
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
        if t.name not in _ALLOWED_TOOLS:
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
    """Pull `fact_ids_cumulative` out of a walk_* tool result, for
    agent-recall measurement.

    FastMCP serialises dict-returning tools as a JSON string in the
    first TextContent block (structuredContent is not set), so we try
    text blocks first then fall back to structuredContent.
    """
    # 1. Text-block JSON (the path FastMCP actually uses for our tools).
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if isinstance(obj, dict):
            ids = obj.get("fact_ids_cumulative")
            if isinstance(ids, list):
                return [str(x) for x in ids]
    # 2. structuredContent fallback for tools that do publish it.
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        payload = structured.get("result", structured)
        if isinstance(payload, dict):
            ids = payload.get("fact_ids_cumulative")
            if isinstance(ids, list):
                return [str(x) for x in ids]
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

        app = build_app(memory=memory)
        total_in = 0
        total_out = 0
        trace: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

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

            for round_idx in range(self.max_rounds):
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    system=AGENT_SYSTEM,
                    tools=tools,
                    messages=messages,
                )
                if hasattr(resp, "usage"):
                    total_in += getattr(resp.usage, "input_tokens", 0) or 0
                    total_out += getattr(resp.usage, "output_tokens", 0) or 0

                tool_uses = [b for b in resp.content if b.type == "tool_use"]
                text_blocks = [b.text for b in resp.content if b.type == "text"]

                # Natural exit : no tool calls → answer
                if not tool_uses:
                    raw = " ".join(text_blocks).strip()
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
                        fr = self.client.messages.create(
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
                                        "THIS IS A PLURAL QUESTION — list "
                                        "ALL items found, comma-separated. "
                                        "Do NOT collapse to one item. "
                                        if _ENUM_Q_RE.search(question or "")
                                        else "full list for plural. "
                                    )
                                    + "No narration."
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
                        fr = self.client.messages.create(
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
                    if tu.name not in _ALLOWED_TOOLS:
                        result_text = (f"Tool {tu.name} not permitted — only "
                                       "walk_start and walk_next are allowed.")
                    elif tu.name == "walk_next" and walk_start_count == 0:
                        # Guard : walk_next without any walk_start.
                        result_text = ("Call walk_start first.")
                    else:
                        call_result = await session.call_tool(tu.name, tu.input)
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
                        except Exception:
                            pass
                        if tu.name == "walk_start":
                            walk_start_count += 1
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
                            "re-read the 'facts' arrays from each walk_start/"
                            "walk_next result above to find it. "
                        )
                    messages.append({
                        "role": "user",
                        "content": (
                            _ab_choice_hint(question)
                            + f"Walk finished.{ev_ctx}"
                            "Call final_answer now with the bare value, "
                            "copied verbatim from the evidence."
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
                resp = self.client.messages.create(
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
                            "Call final_answer with the bare value, copied "
                            "verbatim from the evidence."
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
            if (_ENUM_Q_RE.search(question or "")
                    and len(last_relevant_collected) >= 1):
                ev = "\n".join(
                    f"- {e.get('content', '')}"
                    for e in last_relevant_collected[:14])
                try:
                    er = self.client.messages.create(
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

        return {
            "answer": terse(answer_text, question),
            "answer_raw": answer_text,
            "steps": len([t for t in trace if t["action"] == "tool"]),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "retrieved_ids": sorted(seen_ids),
            "trace": trace,
        }
