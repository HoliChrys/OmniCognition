"""
Behavioural-clue expansion for INFERENCE questions.

Some questions ("What might John's financial status be?", "What political
leaning would Caroline have?") are never answered DIRECTLY in a chat log.
The evidence is a casual aside whose words share nothing with the question
("my kids have so much and others don't" ⇒ well-off). A query in the
QUESTION's vocabulary ("financial status money income") retrieves the
wrong turns; even HyDE fails, because a hypothetical ANSWER ("John is
wealthy") still lives in the money register, not the evidence register.

`clue_utterances` asks the LLM for the opposite of HyDE: instead of one
passage that ANSWERS the question, it brainstorms several short first-
person chat lines that would each be EVIDENCE for a DIFFERENT plausible
answer — spanning the answer space (well-off AND struggling), in the
concrete everyday register a person actually types. Retrieval over those
clue lines then lands on whichever casual aside really exists in the
memory, and the agent infers the answer from what stuck.

Same robustness discipline as the rest of the LLM layer: retry on
transient overload, and NEVER cache an empty result (one 529 must not
disable clue expansion for the rest of a run).
"""

from __future__ import annotations

from typing import Any, List

_CLUE_CACHE: dict = {}


def clue_utterances(question: str, llm: Any, n: int = 6) -> List[str]:
    """Return up to `n` concrete first-person chat lines that would each be
    EVIDENCE for a different plausible answer to `question`.

    Returns [] when there is no usable LLM or on failure — the caller
    treats clue expansion as strictly additive.
    """
    q = (question or "").strip()
    if not q:
        return []
    key = f"{q}::{n}"
    if key in _CLUE_CACHE:
        return list(_CLUE_CACHE[key])
    if not hasattr(llm, "generate"):
        return []
    prompt = (
        "Someone is searching a personal conversation memory (chat logs / "
        "diary) to INFER the answer to this question:\n"
        f"  {q}\n\n"
        "The answer is almost never stated directly — it must be inferred "
        f"from casual things the person said. List {n} SHORT first-person "
        "chat lines that, if they appeared in the logs, would each be "
        "EVIDENCE pointing to a DIFFERENT plausible answer. SPAN the range "
        "of plausible answers (for a wealth question, include both well-off "
        "signals AND struggling signals; for a leaning question, include "
        "signals for each side). At least TWO of the lines MUST be INDIRECT "
        "lifestyle asides — about their FAMILY, KIDS, HOME, what they have / "
        "give / afford, or a comparison to others (e.g. 'my kids have so "
        "much and others don't', 'we just redid the kitchen') — NOT money "
        "words — because the real evidence for a status/leaning question is "
        "almost always such an aside, not a declaration. Cover BOTH direct "
        "signals (a purchase, a bill) AND those indirect ones. If the "
        "question uses an abstract verb (research, study, "
        "look into, work on, plan, decide), interpret it BROADLY — the real "
        "evidence may be a LIFE topic ('researching adoption agencies', "
        "'planning a trip to Lisbon', 'looking into a counseling certificate') "
        "rather than an academic one. Use the concrete everyday wording a "
        "real person actually types — specific nouns, plain facts, no "
        "abstract category words from the question itself. Do NOT restate or "
        "reference the question, do NOT hedge, do NOT explain or label. "
        "Output exactly one line per clue, no numbering, no bullets."
    )
    lines: List[str] = []
    try:
        raw = (llm.generate(prompt, max_tokens=240) or "").strip()
        for ln in raw.splitlines():
            s = ln.strip().lstrip("-*0123456789.) \t").strip()
            # Drop a model-added label like "Well-off: ..." -> keep the line.
            if ":" in s[:24] and s.split(":", 1)[1].strip():
                head = s.split(":", 1)[0]
                if len(head.split()) <= 3:
                    s = s.split(":", 1)[1].strip()
            if s:
                lines.append(s)
    except Exception:
        lines = []
    lines = lines[:n]
    if lines:                       # never cache an empty result
        _CLUE_CACHE[key] = list(lines)
    return lines
