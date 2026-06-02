"""
Real ReAct agent over the metacog MCP server (in-process transport).

Unlike `claude_react.ClaudeReactAnswerer` — which pre-dumps retrieved
chunks into a single prompt and fakes ReAct with a JSON action field —
this answerer is a genuine tool-using agent :

  - it connects to the metacog MCP server (the Memory of the
    conversation, exposed as MCP tools) over the in-memory transport ;
  - Claude drives the loop itself via native tool-use : it decides to
    call `retrieve` (and may refine the query, call `inspect`, etc.),
    reads the tool results, and answers when it has enough evidence.

LOOP EXIT — three terminal conditions, checked every round :

  1. NATURAL    the model returns a text response with no tool_use
                block (stop_reason != "tool_use"). That text is the
                answer. This is the normal exit.
  2. MAX_ROUNDS a hard cap on tool-call rounds (default 6). On hitting
                it we issue ONE final call with tools disabled and a
                "answer now" instruction, then take whatever text comes
                back. Guarantees termination.
  3. NO_PROGRESS the model repeats a tool call it already made this
                turn (same name + same arguments). We stop expanding
                and force the final answer as in (2). Prevents the
                agent from looping on an unproductive query.

Token spend is unbounded by design (a real agent re-reads tool
results) ; we track and report tokens_in / tokens_out so the cost is
visible.

Credentials : api_key arg → ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN.
OAuth access tokens (sk-ant-oat…) are routed via auth_token=.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

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
    """Up to 4 retries with exponential backoff on transient API errors
    (529 overloaded, 5xx). Prevents a single Anthropic-side blip from
    crashing the whole QA agentic loop."""
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

# NOTE: `mcp` and `metacog.mcp_server` are imported lazily inside
# _answer_async so that the lightweight helpers (terse, credential
# resolution) remain importable in environments without the mcp client
# installed (e.g. a minimal test runner).


_DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
_DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "256"))
_MAX_ROUNDS = int(os.environ.get("MCP_AGENT_MAX_ROUNDS", "6"))

# Read-only tools the QA agent is allowed to drive. We deliberately
# withhold the mutating tools (ingest / observe / sleep / …) so the
# benchmark agent cannot alter the memory it is being scored against.
_ALLOWED_TOOLS = {"retrieve", "inspect", "stats"}


AGENT_SYSTEM = """You answer questions about a long conversation by
searching a metacognitive memory through tools.

Workflow :
- Call `retrieve` with a focused query to fetch candidate turns. Each
  turn is prefixed with its [session date]. If the first results don't
  contain the answer, search AGAIN with different keywords (synonyms,
  the other speaker's name, related events) before giving up — 2-3
  searches are normal for multi-hop questions. Use `inspect` to see a
  point's full state when useful.
- Stop searching as soon as you can answer, then reply with the answer
  ONLY (no tool call).

CRITICAL — final answer format. Output ONLY the bare value. NO preamble,
NO "Based on", NO explanation, NO markdown, NO full sentence. Just the
value, matching the gold style :
- "When ..." → ABSOLUTE date only : "7 May 2023" or "June 2023" or
  "2022". Resolve any "yesterday/last week/X days/months ago" against
  the turn's [session date] into a calendar date. Never answer relative.
- "Where ..." → place only : "Sweden".
- "How long" → "<n> <unit>" : "4 years".
- "What did X do/like/research" → short noun phrase : "adoption agencies".
- yes/no/inference → "Likely yes, <one short clause>".
- not in the evidence / adversarial / unanswerable → exactly "Not mentioned".

Examples of GOOD answers : "7 May 2023" · "Sweden" · "4 years" ·
"adoption agencies" · "Not mentioned".
Example of BAD answer : "Based on the search results, she went on 7 May
2023." — too verbose, would score poorly. Output just "7 May 2023"."""


EXTRACT_SYSTEM = """Extract the single answer value from the text, in the
briefest gold form : a date ("7 May 2023" / "June 2023" / "2022"), a place
("Sweden"), a duration ("4 years"), or a short noun phrase ("adoption
agencies"). If the text says nothing was found, output "Not mentioned".
Output ONLY the value — no sentence, no preamble, no punctuation around it."""


def _looks_verbose(text: str) -> bool:
    """Heuristic : does this answer still carry sentence-like wrapping
    that would dilute token-F1 ? Triggers the extraction call."""
    if not text:
        return False
    if len(text) > 60:
        return True
    low = text.lower()
    return any(p in low for p in (
        "i found", "mentioned", "according to", "the conversation",
        "based on", "in the ", "she said", "he said", "states that",
        "it appears", "the answer",
    ))


def _resolve_client(api_key: Optional[str], model: str):
    import anthropic

    token = (
        api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    if token and token.startswith("sk-ant-oat"):
        return anthropic.Anthropic(auth_token=token)
    return anthropic.Anthropic(api_key=token)


def _mcp_tools_to_anthropic(mcp_tools) -> List[Dict[str, Any]]:
    """Bridge MCP tool descriptors → Anthropic tool schema."""
    out: List[Dict[str, Any]] = []
    for t in mcp_tools:
        if t.name not in _ALLOWED_TOOLS:
            continue
        schema = t.inputSchema or {"type": "object", "properties": {}}
        out.append({
            "name": t.name,
            "description": (t.description or "").strip(),
            "input_schema": schema,
        })
    return out


# Leading conversational interjections the model prepends to its answer.
_INTERJECTION_RE = re.compile(
    r"^(perfect|got it|sure|okay|ok|great|yes|absolutely|certainly|"
    r"alright|right|excellent|understood|done|here you go|here's the answer)"
    r"[!.,:\s]+",
    re.IGNORECASE,
)
_PREAMBLE_RE = re.compile(
    r"^(based on[^,.:]*[,.:]\s*|according to[^,.:]*[,.:]\s*|"
    r"the answer is[:\s]+|answer[:\s]+|i found that\s+|"
    r"it (?:appears|seems) that\s+)",
    re.IGNORECASE,
)


def terse(text: str) -> str:
    """Strip the verbose wrapping Haiku tends to add, leaving the value
    the token-overlap F1 wants.

    Steps :
      1. A bolded **value** → take it.
      2. Strip leading interjections ("Perfect!", "Sure,", …) and
         preambles ("Based on …, / The answer is …"), repeatedly.
      3. "Not mentioned" detection.

    Deliberately does NOT pick "the shortest sentence" — that rule used
    to reduce "Perfect! Mel collects leaves on hikes." to "Perfect!",
    destroying correct answers. A moderately verbose answer that still
    contains the gold tokens scores far better than an interjection.
    """
    if not text:
        return text
    t = text.strip()

    m = re.search(r"\*\*(.+?)\*\*", t)
    if m:
        return m.group(1).strip().rstrip(".")

    # Strip stacked interjections + preambles ("Perfect! Based on …, …").
    while True:
        stripped = _INTERJECTION_RE.sub("", t)
        stripped = _PREAMBLE_RE.sub("", stripped).strip()
        if stripped == t:
            break
        if not stripped:
            # The entire text was interjection/preamble — no real answer.
            t = ""
            break
        t = stripped

    if re.search(r"\bnot mentioned\b", t, re.IGNORECASE):
        return "Not mentioned"

    return t.strip().rstrip(".")


def _ids_from_call_result(call_result) -> List[str]:
    """Extract point ids from a `retrieve` MCP result.

    FastMCP returns the list as `structuredContent={'result': [...]}`
    AND as one TextContent block per item. Prefer structuredContent ;
    fall back to parsing each text block as a standalone JSON object.
    Defensive : returns [] on any failure.
    """
    ids: List[str] = []
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        items = structured.get("result")
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and "id" in it:
                    ids.append(it["id"])
            if ids:
                return ids
    # Fallback : one JSON object per content block
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "id" in obj:
                ids.append(obj["id"])
        except Exception:
            continue
    return ids


def _tool_result_text(call_result) -> str:
    """Flatten an MCP CallToolResult's content blocks into text."""
    parts: List[str] = []
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(str(block))
    return "\n".join(parts) if parts else "(no content)"


class McpReactAgent:
    """Tool-using ReAct answerer connected to the metacog MCP server."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
        max_rounds: int = _MAX_ROUNDS,
    ) -> None:
        self.client = _resolve_client(api_key, model)
        self.model = model
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds

    # -- sync entry point (matches ClaudeReactAnswerer.answer signature) --
    def answer(self, memory, question: str, **_ignored) -> Dict[str, Any]:
        return asyncio.run(self._answer_async(memory, question))

    async def _answer_async(self, memory, question: str) -> Dict[str, Any]:
        from mcp.shared.memory import create_connected_server_and_client_session
        from metacog.mcp_server import build_app

        app = build_app(memory=memory)
        total_in = 0
        total_out = 0
        trace: List[Dict[str, Any]] = []

        async with create_connected_server_and_client_session(app) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = _mcp_tools_to_anthropic(listed.tools)

            messages: List[Dict[str, Any]] = [
                {"role": "user", "content": f"Question: {question}"}
            ]
            seen_calls: set[str] = set()
            seen_ids: set[str] = set()  # all point ids the agent retrieved
            answer_text = ""

            for round_idx in range(self.max_rounds):
                resp = _create_with_retry(self.client, 
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

                # EXIT 1 — natural : no tool calls → final answer
                if not tool_uses:
                    answer_text = " ".join(text_blocks).strip()
                    trace.append({"round": round_idx, "action": "final",
                                  "text": answer_text[:200]})
                    break

                # Record the assistant turn (with its tool_use blocks)
                messages.append({"role": "assistant", "content": resp.content})

                # EXIT 3 — no-progress : a repeated (name,args) call
                repeated = False
                tool_results: List[Dict[str, Any]] = []
                for tu in tool_uses:
                    sig = f"{tu.name}:{json.dumps(tu.input, sort_keys=True)}"
                    if sig in seen_calls:
                        repeated = True
                    seen_calls.add(sig)
                    if tu.name not in _ALLOWED_TOOLS:
                        result_text = f"tool {tu.name} not permitted"
                    else:
                        call_result = await session.call_tool(tu.name, tu.input)
                        result_text = _tool_result_text(call_result)
                        # Accumulate retrieved ids for effective-recall.
                        if tu.name == "retrieve":
                            seen_ids.update(_ids_from_call_result(call_result))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result_text,
                    })
                    trace.append({"round": round_idx, "action": "tool",
                                  "name": tu.name, "input": tu.input,
                                  "result_chars": len(result_text)})

                messages.append({"role": "user", "content": tool_results})

                if repeated:
                    answer_text = await self._force_final(
                        messages, trace, round_idx,
                    )
                    r = self._count_force(answer_text)
                    total_in += r[0]
                    total_out += r[1]
                    answer_text = r[2]
                    break
            else:
                # EXIT 2 — max rounds reached without a natural answer
                r = self._count_force(
                    await self._force_final(messages, trace, self.max_rounds)
                )
                total_in += r[0]
                total_out += r[1]
                answer_text = r[2]

        # Finalize : the agent's free-form answer is often verbose
        # ("I found the answer. John mentioned on Nov 6 2023 that…"),
        # which tanks token-overlap F1. One cheap extraction call distills
        # it to the bare gold-style value. Deterministic terse() then
        # mops up any remaining wrapper.
        final = terse(answer_text)
        if _looks_verbose(final):
            extracted, ei, eo = self._extract_value(question, answer_text)
            total_in += ei
            total_out += eo
            if extracted:
                final = terse(extracted)

        return {
            "answer": final,
            "answer_raw": answer_text,
            "steps": len([t for t in trace if t["action"] == "tool"]),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "retrieved_ids": sorted(seen_ids),  # cumulative across queries
            "trace": trace,
        }

    def _extract_value(self, question: str, verbose_answer: str):
        """One small call : distill a verbose answer to the bare value."""
        try:
            resp = _create_with_retry(self.client, 
                model=self.model,
                max_tokens=40,
                system=EXTRACT_SYSTEM,
                messages=[{"role": "user", "content":
                           f"Q: {question}\nText: {verbose_answer}\nValue:"}],
            )
            ti = getattr(getattr(resp, "usage", None), "input_tokens", 0) or 0
            to = getattr(getattr(resp, "usage", None), "output_tokens", 0) or 0
            text = " ".join(b.text for b in resp.content
                            if b.type == "text").strip()
            return text, ti, to
        except Exception:
            return "", 0, 0

    async def _force_final(self, messages, trace, round_idx) -> Any:
        """Issue ONE final call with tools disabled, demanding the answer.
        Returns the raw response so the caller can count tokens."""
        messages.append({
            "role": "user",
            "content": "Stop searching. Give the final answer now, text only.",
        })
        resp = _create_with_retry(self.client, 
            model=self.model,
            max_tokens=self.max_tokens,
            system=AGENT_SYSTEM,
            messages=messages,
        )
        trace.append({"round": round_idx, "action": "forced_final"})
        return resp

    @staticmethod
    def _count_force(resp):
        ti = getattr(getattr(resp, "usage", None), "input_tokens", 0) or 0
        to = getattr(getattr(resp, "usage", None), "output_tokens", 0) or 0
        text = " ".join(b.text for b in resp.content if b.type == "text").strip()
        return ti, to, text
