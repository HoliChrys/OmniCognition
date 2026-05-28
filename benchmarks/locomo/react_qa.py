"""ReAct-style QA over MetaCog retrievals using extractive roberta.

Two-step loop :
  step 0  retrieve(question, k) → extract_answer(question, context)
  step 1  if confidence < 0.5 : refined query = question + extracted
          → retrieve → extract again
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


_MODEL_NAME = "deepset/roberta-base-squad2"


@lru_cache(maxsize=1)
def _qa_model():
    tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(_MODEL_NAME)
    model.eval()
    return tok, model


@torch.no_grad()
def extract_answer(question: str, context: str) -> Tuple[str, float]:
    """Extract answer span from context. Returns (text, score)."""
    if not context or not question:
        return ("", 0.0)
    tok, model = _qa_model()
    inputs = tok(
        question, context,
        return_tensors="pt",
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
    )
    offsets = inputs.pop("offset_mapping")[0]
    out = model(**inputs)
    start = out.start_logits[0]
    end = out.end_logits[0]
    start_probs = torch.softmax(start, dim=0)
    end_probs = torch.softmax(end, dim=0)
    best_score = 0.0
    best_span = (0, 0)
    max_len = 30
    n = len(start)
    for s in range(n):
        for e in range(s, min(s + max_len, n)):
            sc = float(start_probs[s] * end_probs[e])
            if sc > best_score:
                best_score = sc
                best_span = (s, e)
    s, e = best_span
    if s == 0 and e == 0:
        return ("", 0.0)
    char_start = int(offsets[s][0])
    char_end = int(offsets[e][1])
    return (context[char_start:char_end].strip(), best_score)


def react_answer(memory, question: str, *, max_steps: int = 2,
                 k_retrieve: int = 10) -> dict:
    """Run a ReAct loop over a Memory instance.

    Returns {answer, score, evidence_ids, trace}.
    """
    trace = []
    evidence_ids: list[str] = []
    final_answer = ""
    final_score = 0.0

    query = question
    for step in range(max_steps):
        results = memory.retrieve(query, k=k_retrieve)
        for r in results:
            if r["id"] not in evidence_ids:
                evidence_ids.append(r["id"])
        context = " ".join(r["content"] for r in results)
        if not context:
            trace.append({"step": step, "halt": "no_context"})
            break
        ans, score = extract_answer(question, context)
        trace.append({
            "step": step,
            "query": query,
            "extracted": ans,
            "score": round(score, 4),
        })
        if ans:
            final_answer, final_score = ans, score
        if score > 0.5 and ans:
            break
        query = f"{question} {ans}".strip() if ans else question

    return {
        "answer": final_answer,
        "score": final_score,
        "evidence_ids": evidence_ids,
        "trace": trace,
    }
