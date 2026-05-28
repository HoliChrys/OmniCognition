"""
Claude-API-backed ReAct answerer for the LoCoMo benchmark.

Each step :
  1. retrieve(query, k) over the Memory (with lineage if enabled)
  2. accumulate evidence chunks
  3. ask Claude to decide :
       {"action": "search", "query": "<refined>"}
       {"action": "answer", "answer": "<final concise>"}
  4. loop until "answer" or max_steps

Set ANTHROPIC_API_KEY in the environment. Optional knobs :
  CLAUDE_MODEL          (default : claude-haiku-4-5-20251001 — cheap + fast)
  CLAUDE_MAX_TOKENS     (default : 512)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


_DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
_DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "512"))


REACT_SYSTEM = """You are answering questions over a memory of conversation turns.

You proceed step by step (ReAct). At each step you see the original
question and all evidence chunks retrieved so far (each labeled with
its dialog id like D3:13).

At each step, output EXACTLY ONE JSON object, no prose, of the form :

  {"action": "search", "query": "<refined search terms>"}
    — when the evidence is insufficient and you want more chunks.
    The query should be different from previous queries — try
    entity-level or temporal-anchor reformulations.

  {"action": "answer", "answer": "<concise final answer>"}
    — when you have enough evidence. The answer must match the
    style of the gold answer : a date like "7 May 2023" for "when"
    questions, a noun phrase like "adoption agencies" for "what"
    questions, a one-word place name like "Sweden" for "where"
    questions. Do not add explanations.

Rules :
- Max 3 search rounds. After that, give your best answer.
- Be concise — the evaluator scores by token-overlap F1.
- For inference questions ("would she likely..."), start the answer
  with "Likely yes" or "Likely no" then a brief reason.
"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of the model's reply."""
    if not text:
        return None
    text = text.strip()
    # Strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    # Direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback : find first { ... last matching }
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


class ClaudeReactAnswerer:
    """Persistent ReAct answerer using the Anthropic SDK."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
    ) -> None:
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.model = model
        self.max_tokens = max_tokens

    def answer(
        self,
        memory,
        question: str,
        *,
        k: int = 10,
        max_steps: int = 3,
        use_lineage: bool = True,
    ) -> Dict[str, Any]:
        evidence: List[Dict[str, Any]] = []
        evidence_ids: set[str] = set()
        trace: List[Dict[str, Any]] = []
        query = question

        for step in range(max_steps):
            results = memory.retrieve(query, k=k, use_lineage=use_lineage)
            for r in results:
                if r["id"] not in evidence_ids:
                    evidence.append(r)
                    evidence_ids.add(r["id"])

            evidence_text = "\n".join(
                f"[{r['id']}] {r['content']}" for r in evidence
            )
            user_msg = (
                f"Question: {question}\n\n"
                f"Evidence so far ({len(evidence)} chunks):\n{evidence_text}\n\n"
                f"Output the JSON object."
            )
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=REACT_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text if resp.content else ""
            obj = _extract_json(text)
            trace.append({
                "step": step,
                "query": query,
                "n_evidence": len(evidence),
                "raw_reply": text,
                "parsed": obj,
            })

            if obj is None:
                # Fall back : treat the raw text as the answer
                return {
                    "answer": text.strip(),
                    "steps": step + 1,
                    "evidence_ids": sorted(evidence_ids),
                    "trace": trace,
                }
            action = obj.get("action")
            if action == "answer":
                return {
                    "answer": str(obj.get("answer", "")).strip(),
                    "steps": step + 1,
                    "evidence_ids": sorted(evidence_ids),
                    "trace": trace,
                }
            if action == "search":
                new_q = str(obj.get("query", "")).strip() or question
                # Avoid infinite loop : if same query twice, abort
                if new_q == query and step > 0:
                    break
                query = new_q
                continue

        # Final fallback : ask Claude to answer with the evidence we have
        evidence_text = "\n".join(
            f"[{r['id']}] {r['content']}" for r in evidence
        )
        user_msg = (
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"You have exhausted search rounds. "
            f"Output {{\"action\": \"answer\", \"answer\": \"<your best answer>\"}}."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=REACT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text if resp.content else ""
        obj = _extract_json(text) or {}
        return {
            "answer": str(obj.get("answer", text)).strip(),
            "steps": max_steps,
            "evidence_ids": sorted(evidence_ids),
            "trace": trace,
        }
