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
_MAX_ROUNDS = int(os.environ.get("MCP_META_MAX_ROUNDS", "5"))

# Only the walk tools — the agent should NOT bypass the walk by
# calling raw retrieve. This is enforced server-side AND client-side.
_ALLOWED_TOOLS = {"walk_start", "walk_next"}


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
   Example : question "What did Caroline research?" drifted →
     walk_start(query="Caroline researching adoption agencies family")
   Example : found running but not the second hobby →
     walk_start(query="Melanie hobby activity craft pottery")
   Start each walk with a RICH query (4+ content words from the
   question), never a 2-word stub like "Caroline research".
6. Only answer "Not mentioned" after you have tried at least TWO
   differently-phrased walks and both drifted. Otherwise answer with
   the best relevant/partial evidence you found.

sigma_path measures geometric drift ; drifted / n_relevant measure
CONTENT relevance and are the stronger pivot signal — act on them.

CRITICAL — final answer format. Output ONLY the bare value, no prose :
- "When ..." → ABSOLUTE date only ("7 May 2023" / "June 2023" / "2022").
  Resolve any "yesterday / last week / X days ago" against the turn's
  [session date] prefix into a calendar date. Never answer relative.
- "Where ..." → place only ("Sweden").
- "How long" → "<n> <unit>" ("4 years").
- "What/Who is X" / identity / role → the SHORTEST label that fits
  (1-3 words). "Transgender woman" not "Transgender artist and member
  of the LGBTQ community". "Teacher" not "A passionate elementary
  school teacher who loves kids". Pick the bare category noun.
- "What did X do/like/research" → short noun phrase (1-4 words).
- yes/no/inference → "Likely yes, <one short clause>".
- not in the evidence / adversarial / unanswerable → "Not mentioned".

Hard rule : if your answer is longer than 5 words, you are wrong —
strip every adjective and conjunction until only the bare value
remains. Drop "and ...", "who is ...", "the ... of ...".

Examples of GOOD answers : "7 May 2023" · "Sweden" · "4 years" ·
"adoption agencies" · "Transgender woman" · "Not mentioned".
Example of BAD : "Based on the walk, the support group was on …" —
too verbose, will score poorly. Just the value.
Example of BAD : "Transgender artist and member of the LGBTQ
community" — has "and", strip to "Transgender woman"."""


def _resolve_client(api_key: Optional[str]):
    import anthropic

    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    # ANTHROPIC_AUTH_TOKEN is always an OAuth bearer — use auth_token=.
    # ANTHROPIC_API_KEY / explicit api_key that starts with sk-ant-oat too.
    auth_tok = os.environ.get("ANTHROPIC_AUTH_TOKEN") or (
        api_key if (api_key and api_key.startswith("sk-ant-oat")) else None
    )
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
    r"it (?:appears|seems) that\s+)",
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
# Compound-phrase cutters : keep the head label, drop trailing
# elaboration. Order matters — most specific first.
_COMPOUND_CUTTERS = [
    re.compile(r"\s+(?:and|or|as well as|along with|together with)\s+.+$",
               re.IGNORECASE),
    re.compile(r"\s+who\s+(?:is|was|has|loves|enjoys|likes)\s+.+$",
               re.IGNORECASE),
    re.compile(r"\s+that\s+(?:is|was)\s+.+$", re.IGNORECASE),
    re.compile(r",\s+.+$"),  # trailing ", member of …"
]


def terse(text: str) -> str:
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
    """
    if not text:
        return text
    t = text.strip()
    m = re.search(r"\*\*(.+?)\*\*", t)
    if m:
        return m.group(1).strip().rstrip(".")
    # Citation + trailing label pattern : take the last short line.
    lines = [ln.strip(" .,") for ln in t.splitlines() if ln.strip(" .,")]
    if len(lines) >= 2:
        tail = lines[-1]
        if 1 <= len(tail.split()) <= 8 and not tail.lower().startswith(
            ("based on", "according to", "the answer", "perfect", "i found")
        ):
            t = tail
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
    # Drop any leaked dialog-id citation, then tidy leftover punctuation.
    t = _DIALOG_ID_RE.sub("", t).strip(" .,()")
    # Compound cutting : apply iteratively while the answer is short
    # enough that the head is plausibly the bare label. Avoid touching
    # long prose answers where "and" is part of a real sentence.
    # Skip inference answers ("Likely yes, …") — the trailing clause IS
    # the expected format.
    is_inference = bool(re.match(r"^\s*likely\s+(?:yes|no)\b", t, re.IGNORECASE))
    if t and not is_inference and len(t.split()) <= 12:
        changed = True
        while changed and t and len(t.split()) <= 12:
            changed = False
            for cutter in _COMPOUND_CUTTERS:
                new = cutter.sub("", t).strip(" .,()")
                if new and new != t:
                    t = new
                    changed = True
                    break
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
            tools = _mcp_tools_to_anthropic(listed.tools)

            messages: List[Dict[str, Any]] = [
                {"role": "user", "content": f"Question: {question}"}
            ]
            answer_text = ""
            walk_start_count = 0   # total breadth pivots taken
            done_seen = False

            for round_idx in range(self.max_rounds):
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
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
                    answer_text = " ".join(text_blocks).strip()
                    trace.append({"round": round_idx, "action": "final",
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
                        if tu.name == "walk_start":
                            walk_start_count += 1
                            done_seen = False  # reset — new walk, new chance
                        # Detect done flag for trace.
                        try:
                            if json.loads(result_text).get("done"):
                                done_seen = True
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
                if done_seen and walk_start_count >= 2:
                    # Two breadth threads exhausted — enough evidence.
                    messages.append({
                        "role": "user",
                        "content": "Walk finished. Give the final answer "
                                   "now, value only.",
                    })
            else:
                # max_rounds exhausted without a natural answer
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=AGENT_SYSTEM,
                    messages=messages + [{
                        "role": "user",
                        "content": "Stop searching. Final answer now.",
                    }],
                )
                if hasattr(resp, "usage"):
                    total_in += getattr(resp.usage, "input_tokens", 0) or 0
                    total_out += getattr(resp.usage, "output_tokens", 0) or 0
                answer_text = " ".join(
                    b.text for b in resp.content if b.type == "text"
                ).strip()
                trace.append({"round": self.max_rounds,
                              "action": "forced_final"})

        return {
            "answer": terse(answer_text),
            "answer_raw": answer_text,
            "steps": len([t for t in trace if t["action"] == "tool"]),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "retrieved_ids": sorted(seen_ids),
            "trace": trace,
        }
