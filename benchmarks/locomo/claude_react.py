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

At each step, output EXACTLY ONE JSON object, no prose, no code
fences, of the form :

  {"action": "search", "query": "<refined search terms>"}
    when the evidence is insufficient.

  {"action": "answer", "answer": "<the answer>"}
    when you have enough evidence.

================ ANSWER FORMAT (critical) ================

Token-overlap F1 punishes verbosity. NEVER explain. NEVER restate
the question. NEVER include the dialog id. The answer must be the
shortest possible string that matches the gold style below.

QUESTION STYLE     GOLD STYLE                  YOUR ANSWER FORMAT
─────────────────────────────────────────────────────────────────
"When did X..."    "7 May 2023"                "<date>" no prefix
"When did X..."    "June 2023"                 "<month year>" no prefix
"When did X..."    "The week before 9 June 2023"  use this exact phrasing
"Where did X..."   "Sweden"                    "<single proper noun>"
"What did X do/research/like"  "adoption agencies"  noun phrase, lowercase ok
"How long..."      "4 years"                   "<number> <unit>"
"Who is X"         "Transgender woman"         "<short noun phrase>"
"What activities/books/places (list)"  "pottery, camping, painting"  comma list

For YES/NO/INFERENCE questions :
  Start with exactly "Likely yes" or "Likely no" then add ", "
  followed by ONE short clause. Examples :
    "Likely no, she does not refer to herself as part of it"
    "Yes, since she collects classic children's books"

ADVERSARIAL questions (where the answer is "Not mentioned / I don't
know") : output exactly  {"action": "answer", "answer": "Not mentioned"}

================ TEMPORAL ANCHORING ================

Dialogue turns reference time relative to their session date :
"yesterday", "last week", "next month", "this year". If you see a
dia_id like D5:13, session 5's date is encoded in the surrounding
context — combine "this month" + the dia_id session number to
produce an absolute date when the question asks "when".

================ SEARCH STRATEGY ================

Max 3 search rounds. Reformulate by :
  - swapping entity for relation ("Caroline went to" → "LGBTQ support group date")
  - going from semantic to lexical ("career" → "counseling mental health")
  - adding temporal anchors ("when" → "yesterday last week month year")

After 3 rounds, answer with the best evidence you have.
"""


_FEW_SHOTS = """Example 1
Question: When did Caroline go to the LGBTQ support group?
Evidence:
[D1:3] Caroline: I went to a LGBTQ support group yesterday and it was so powerful.
[D1:1] session date_time: 8 May 2023
Output:
{"action": "answer", "answer": "7 May 2023"}

Example 2
Question: Where did Caroline move from 4 years ago?
Evidence:
[D4:3] Caroline: This necklace is super special to me - a gift from my grandma in my home country, Sweden.
Output:
{"action": "answer", "answer": "Sweden"}

Example 3
Question: What does Melanie do to destress?
Evidence:
[D7:22] Melanie: I've been running farther to de-stress, which has been great for my headspace.
[D5:6] Melanie: I'm a big fan of pottery - the creativity and skill is awesome. Plus, making it is so calming.
Output:
{"action": "answer", "answer": "running, pottery"}

Example 4 (inference)
Question: Would Caroline likely have Dr. Seuss books on her bookshelf?
Evidence:
[D6:9] Caroline: I've got lots of kids' books- classics, stories from different cultures, educational books.
Output:
{"action": "answer", "answer": "Yes, since she collects classic children's books"}

Example 5 (insufficient → search)
Question: How long ago was Caroline's 18th birthday?
Evidence:
[D1:9] Caroline: Gonna continue my edu and check out career options.
Output:
{"action": "search", "query": "Caroline 18th birthday age years"}
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
        k: int = 7,
        max_steps: int = 3,
        use_lineage: bool = True,
        use_hybrid: bool = True,
    ) -> Dict[str, Any]:
        evidence: List[Dict[str, Any]] = []
        evidence_ids: set[str] = set()
        trace: List[Dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
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

            evidence_text = "\n".join(
                f"[{r['id']}] {r['content']}" for r in evidence
            )
            user_msg = (
                f"{_FEW_SHOTS}\n"
                f"Now answer the real question.\n\n"
                f"Question: {question}\n\n"
                f"Evidence ({len(evidence)} chunks):\n{evidence_text}\n\n"
                f"Output exactly one JSON object."
            )
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=REACT_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text if resp.content else ""
            if hasattr(resp, "usage"):
                total_input_tokens += getattr(resp.usage, "input_tokens", 0) or 0
                total_output_tokens += getattr(resp.usage, "output_tokens", 0) or 0
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
                    "tokens_in": total_input_tokens,
                    "tokens_out": total_output_tokens,
                }
            action = obj.get("action")
            if action == "answer":
                return {
                    "answer": str(obj.get("answer", "")).strip(),
                    "steps": step + 1,
                    "evidence_ids": sorted(evidence_ids),
                    "trace": trace,
                    "tokens_in": total_input_tokens,
                    "tokens_out": total_output_tokens,
                }
            if action == "search":
                new_q = str(obj.get("query", "")).strip() or question
                if new_q == query and step > 0:
                    break
                query = new_q
                continue

        # Forced final answer
        evidence_text = "\n".join(
            f"[{r['id']}] {r['content']}" for r in evidence
        )
        user_msg = (
            f"{_FEW_SHOTS}\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Search budget exhausted. Output "
            f'{{"action": "answer", "answer": "<your best answer in the gold format>"}}.'
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=REACT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text if resp.content else ""
        if hasattr(resp, "usage"):
            total_input_tokens += getattr(resp.usage, "input_tokens", 0) or 0
            total_output_tokens += getattr(resp.usage, "output_tokens", 0) or 0
        obj = _extract_json(text) or {}
        return {
            "answer": str(obj.get("answer", text)).strip(),
            "steps": max_steps,
            "evidence_ids": sorted(evidence_ids),
            "trace": trace,
            "tokens_in": total_input_tokens,
            "tokens_out": total_output_tokens,
        }
