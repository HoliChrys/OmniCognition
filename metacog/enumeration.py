"""
Enumeration / retrieve mode — answer as a LIST (a "bag" of element refs).

Some requests don't want a synthesised answer, they want to RETRIEVE an
exhaustive set ("find all tweets that …", "list every city X visited", "which
turns mention …"). For those the right output is the BAG : the collected refs
of the matching elements, returned as a list — not a one-line answer.

`is_enumeration_query` detects the intent. `format_bag_answer` renders the bag :
the LLM writes a SHORT framing sentence with ONE placeholder it names itself
(`{{whatever}}`), and we inject the formatted list of bag items into that
placeholder — so the model controls the wording and the system controls the
exhaustive list. Failure-safe : no LLM / no placeholder -> a plain list header.
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple

# A set-noun the request enumerates (tweets, posts, messages, …) — used both in
# the verb+noun pattern below and elsewhere as "what an exhaustive set is made
# of". Kept broad : these are the recall-mode units.
_SET_NOUNS = (r"tweets|posts|messages|turns|docs|documents|items|examples|"
              r"instances|cases|results|ones|comments|replies|entries|"
              r"statements|quotes|mentions|threads")
_ENUM_RE = re.compile(
    r"\b(find all|list (all|every)?|which (" + _SET_NOUNS + r")|all the|"
    r"every |enumerate|retrieve all|how many|name all|give me all|"
    # "Find tweets where …", "show me the posts that …", "gather all examples"
    r"(find|show(?: me)?|get|gather|collect|surface|fetch|return|identify) "
    r"(all |every |the )?(" + _SET_NOUNS + r"))\b", re.IGNORECASE)


def is_enumeration_query(question: str) -> bool:
    """True when the question asks to RETRIEVE/enumerate a set rather than to
    answer with a single value."""
    return bool(_ENUM_RE.search(question or ""))


def _render_list(items: Sequence[Tuple[str, str]]) -> str:
    out = []
    for ref, content in items:
        c = (content or "").strip().replace("\n", " ")
        out.append(f"- [{ref}] {c[:140]}" if c else f"- [{ref}]")
    return "\n".join(out)


def format_bag_answer(question: str, bag: Sequence[Tuple[str, str]],
                      llm: Any) -> str:
    """Render an enumeration answer from the BAG of (ref, content) items.

    The LLM emits a one-line framing with a single `{{placeholder}}` it names ;
    the system substitutes the exhaustive list. Returns the filled string.
    Empty bag -> "" (caller falls back to the normal answer path)."""
    items = [(str(r), c) for r, c in (bag or []) if r]
    if not items:
        return ""
    listing = _render_list(items)
    frame = ""
    if hasattr(llm, "generate"):
        try:
            frame = (llm.generate(
                "The user asked to RETRIEVE/list items. Write ONE short framing "
                "sentence for the result, containing EXACTLY ONE placeholder of "
                "the form {{name}} (you choose the name) where the list of "
                f"{len(items)} found items will be inserted. Output only the "
                f"sentence.\n\nRequest: {question}\nSentence:", max_tokens=40)
                or "").strip()
        except Exception:
            frame = ""
    m = re.search(r"\{\{[^}]+\}\}", frame)
    if m:
        return frame[:m.start()] + "\n" + listing + "\n" + frame[m.end():]
    return f"Found {len(items)} matching items:\n{listing}"
