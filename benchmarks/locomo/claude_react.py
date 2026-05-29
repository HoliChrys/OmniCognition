"""
Claude-API-backed ReAct answerer for the LoCoMo benchmark.

Designed for a strict token budget : target ≤ 500 tokens / QA with k=7.

Layout per QA (target) :
  system        ~ 80 tokens  (format rules only, no examples)
  user          ~ 250 tokens (question + 7 chunks truncated to ~120 chars)
  output        ~  25 tokens (single JSON object)
  ────────────────────────────
  total         ~ 355 tokens

ReAct loop is OPT-IN (max_steps default = 1). Pass max_steps>1 to
allow the model to issue refined searches at the cost of more tokens.

Set ANTHROPIC_API_KEY. Optional knobs :
  CLAUDE_MODEL         (default : claude-haiku-4-5-20251001)
  CLAUDE_MAX_TOKENS    (default : 128 — bounds output spend)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


_DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
_DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "128"))

# Hard cap on each evidence chunk's CONTENT character length when
# rendered into the user message. 100 chars ≈ 25 tokens. With the
# enrichment prefix [id|kw1,kw2,kw3], total per chunk ≈ 130 chars
# ≈ 32 tokens. 7 × 32 = 224 evidence tokens.
_CHUNK_CHAR_LIMIT = 100
# Compact keyword hint cap : at most 3 keywords, comma-separated
_KEYWORDS_HINT_MAX = 3


REACT_SYSTEM = """Answer concisely matching the gold style :
- "When ..." → date only ("7 May 2023" or "June 2023")
- "Where ..." → place only ("Sweden")
- "What did X do/research/like" → noun phrase ("adoption agencies")
- "How long" → "<n> <unit>" ("4 years")
- YES/NO/inference → "Likely yes/no, <one clause>"
- adversarial → "Not mentioned"

Use D<session>:<turn> ids and "yesterday/last week/this month" relative
to the session date to anchor temporal answers.

Output ONE JSON object, no prose, no code fences :
  {"action": "answer", "answer": "..."}
or {"action": "search", "query": "..."}"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of the model's reply."""
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
                break
    return None


def _format_evidence(
    chunks: List[Dict[str, Any]],
    char_limit: int,
    memory=None,
) -> str:
    """Render chunks as compact enriched one-liners.

    Format per chunk :
      [<id>|<kw1>,<kw2>,<kw3>] <truncated_content>

    Enrichments :
      - id           : dialog id ; for collision children, resolves to
                       <parent_a>↔<parent_b> when possible.
      - kw1..kw3     : top keywords already extracted at ingest
                       (compact entity signal, ~20 chars total).
      - content      : raw text, truncated to `char_limit` chars
                       and suffixed with "…" if truncated.

    Falls back to plain `[id] content` when keywords or memory are
    unavailable.
    """
    points_by_id = {p.id: p for p in memory.points} if memory else {}
    lines = []
    for r in chunks:
        rid = r["id"]
        # Resolve collision child id to its parents when possible
        display_id = rid
        if rid.startswith("collision("):
            p = points_by_id.get(rid)
            if p is not None and p.parents:
                display_id = "↔".join(p.parents[:2])

        # Compact keyword hint from the live point
        kw_hint = ""
        p = points_by_id.get(rid)
        if p is not None and p.keywords:
            kw_hint = "|" + ",".join(p.keywords[:_KEYWORDS_HINT_MAX])

        content = r.get("content", "")
        if len(content) > char_limit:
            content = content[:char_limit].rstrip() + "…"
        lines.append(f"[{display_id}{kw_hint}] {content}")
    return "\n".join(lines)


class ClaudeReactAnswerer:
    """Token-budgeted ReAct answerer using the Anthropic SDK."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
        chunk_char_limit: int = _CHUNK_CHAR_LIMIT,
    ) -> None:
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.model = model
        self.max_tokens = max_tokens
        self.chunk_char_limit = chunk_char_limit

    def answer(
        self,
        memory,
        question: str,
        *,
        k: int = 7,
        max_steps: int = 1,
        use_lineage: bool = True,
        use_hybrid: bool = True,
    ) -> Dict[str, Any]:
        """Run one (or up to max_steps) ReAct rounds.

        Defaults : single shot. The full evidence pool (k=7) is shown to
        the model on the first message ; further rounds only happen if
        Claude explicitly emits {"action": "search", ...}.
        """
        evidence: List[Dict[str, Any]] = []
        evidence_ids: set[str] = set()
        trace: List[Dict[str, Any]] = []
        total_in = 0
        total_out = 0
        query = question

        for step in range(max_steps):
            results = memory.retrieve(
                query, k=k,
                use_hybrid=use_hybrid,
                use_lineage=use_lineage,
            )
            for r in results:
                if r["id"] not in evidence_ids:
                    evidence.append(r)
                    evidence_ids.add(r["id"])

            evidence_text = _format_evidence(evidence, self.chunk_char_limit, memory=memory)
            user_msg = f"Q: {question}\n{evidence_text}"

            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=REACT_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text if resp.content else ""
            if hasattr(resp, "usage"):
                total_in += getattr(resp.usage, "input_tokens", 0) or 0
                total_out += getattr(resp.usage, "output_tokens", 0) or 0
            obj = _extract_json(text)
            trace.append({
                "step": step,
                "query": query,
                "n_evidence": len(evidence),
                "raw_reply": text,
                "parsed": obj,
            })

            if obj is None:
                return {
                    "answer": text.strip(),
                    "steps": step + 1,
                    "evidence_ids": sorted(evidence_ids),
                    "trace": trace,
                    "tokens_in": total_in,
                    "tokens_out": total_out,
                }
            action = obj.get("action")
            if action == "answer":
                return {
                    "answer": str(obj.get("answer", "")).strip(),
                    "steps": step + 1,
                    "evidence_ids": sorted(evidence_ids),
                    "trace": trace,
                    "tokens_in": total_in,
                    "tokens_out": total_out,
                }
            if action == "search":
                new_q = str(obj.get("query", "")).strip() or question
                if new_q == query and step > 0:
                    break
                query = new_q
                continue

        # Forced final answer (max_steps exhausted on a search action)
        evidence_text = _format_evidence(evidence, self.chunk_char_limit)
        user_msg = (
            f"Q: {question}\n{evidence_text}\n"
            f'Budget exhausted. Output {{"action": "answer", "answer": "..."}}.'
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=REACT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text if resp.content else ""
        if hasattr(resp, "usage"):
            total_in += getattr(resp.usage, "input_tokens", 0) or 0
            total_out += getattr(resp.usage, "output_tokens", 0) or 0
        obj = _extract_json(text) or {}
        return {
            "answer": str(obj.get("answer", text)).strip(),
            "steps": max_steps,
            "evidence_ids": sorted(evidence_ids),
            "trace": trace,
            "tokens_in": total_in,
            "tokens_out": total_out,
        }
