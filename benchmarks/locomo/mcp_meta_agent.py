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
3. Otherwise call `walk_next(walk_id=…)` to receive the NEXT stage —
   it follows the filiation of the previous thought. Multi-hop
   questions usually need 2-3 stages.
4. Stop calling `walk_next` once `done=true` is returned, or once you
   have enough evidence across the stages you have seen.

CRITICAL — final answer format. Output ONLY the bare value, no prose :
- "When ..." → ABSOLUTE date only ("7 May 2023" / "June 2023" / "2022").
  Resolve any "yesterday / last week / X days ago" against the turn's
  [session date] prefix into a calendar date. Never answer relative.
- "Where ..." → place only ("Sweden").
- "How long" → "<n> <unit>" ("4 years").
- "What did X do/like/research" → short noun phrase.
- yes/no/inference → "Likely yes, <one short clause>".
- not in the evidence / adversarial / unanswerable → "Not mentioned".

Examples of GOOD answers : "7 May 2023" · "Sweden" · "4 years" ·
"adoption agencies" · "Not mentioned".
Example of BAD : "Based on the walk, the support group was on …" —
too verbose, will score poorly. Just the value."""


def _resolve_client(api_key: Optional[str]):
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


def terse(text: str) -> str:
    """Strip Haiku's preamble/interjection wrapping. Mirrors the helper
    in mcp_agent.py — keeps the bare value the F1 metric scores on."""
    if not text:
        return text
    t = text.strip()
    m = re.search(r"\*\*(.+?)\*\*", t)
    if m:
        return m.group(1).strip().rstrip(".")
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
            walk_started = False
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
                    elif tu.name == "walk_next" and not walk_started:
                        # Guard : walk_next without walk_start.
                        result_text = ("Call walk_start first.")
                    else:
                        call_result = await session.call_tool(tu.name, tu.input)
                        result_text = _tool_result_text(call_result)
                        seen_ids.update(_extract_fact_ids(call_result))
                        if tu.name == "walk_start":
                            walk_started = True
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

                if done_seen:
                    # Walk exhausted — force a final answer next round.
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
