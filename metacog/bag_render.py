"""
Bag rendering strategies — the agent decides HOW to present the referenced nodes.

When an answer must "refer an exhaustive set" (the bag), the raw list is rarely
the best surface form. The agent chooses among :

  * raw        — the full list of nodes, untouched.
  * extract    — each node truncated to a preview : first `head` tokens + " ... "
                 + last `tail` tokens (the "20 … 20" ellipsis).
  * interpret  — one short INTERPRETATION per node, the exhaustive list of those
                 parallel re-interpretations.
  * mapreduce  — a rolling BATCH summary : batches of `batch` nodes, each summary
                 injected into the next batch (cumulative, nothing dropped).

and PLACEMENT : `inject` weaves the rendering into a one-line framing the LLM
writes (a single `{{placeholder}}` it names), or `bare` returns the rendering
alone with no surrounding message. `auto` lets the LLM (the agent) decide.

Failure-safe throughout : no LLM / errors fall back to deterministic forms.
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple

Item = Tuple[str, str]                                   # (ref, content)


def _clean(c: str) -> str:
    return (c or "").strip().replace("\n", " ")


def render_raw(items: Sequence[Item]) -> str:
    """The full list of nodes, untouched."""
    out = []
    for ref, c in items:
        c = _clean(c)
        out.append(f"- [{ref}] {c}" if c else f"- [{ref}]")
    return "\n".join(out)


def _ellipsis(text: str, head: int, tail: int) -> str:
    toks = text.split()
    if len(toks) <= head + tail:
        return " ".join(toks)
    return " ".join(toks[:head]) + " ... " + " ".join(toks[-tail:])


def render_extract(items: Sequence[Item], head: int = 20, tail: int = 20) -> str:
    """Each node truncated : `head` tokens + ' ... ' + `tail` tokens."""
    out = []
    for ref, c in items:
        c = _ellipsis(_clean(c), head, tail)
        out.append(f"- [{ref}] {c}" if c else f"- [{ref}]")
    return "\n".join(out)


def render_interpretations(items: Sequence[Item], llm: Any, question: str) -> str:
    """One short interpretation per node — the exhaustive list of parallel
    re-interpretations of how each node bears on the request."""
    out = []
    for ref, c in items:
        interp = ""
        if hasattr(llm, "generate"):
            try:
                interp = (llm.generate(
                    "In ONE short clause, how does this item bear on the "
                    f"request? No preamble.\nRequest: {question}\nItem: "
                    f"{_clean(c)[:300]}\nClause:", max_tokens=40) or "").strip()
            except Exception:
                interp = ""
        out.append(f"- [{ref}] {interp or _clean(c)[:80]}")
    return "\n".join(out)


def render_mapreduce(items: Sequence[Item], llm: Any, question: str,
                     batch: int = 4) -> str:
    """Rolling batch summary : summarise batches of `batch` nodes, injecting the
    running summary into the next batch so nothing earlier is dropped."""
    if not hasattr(llm, "generate"):
        return render_extract(items)
    summary = ""
    for i in range(0, len(items), max(1, batch)):
        chunk = items[i:i + max(1, batch)]
        try:
            nxt = (llm.generate(
                "Summarise these items for the request, CONTINUING the running "
                "summary (keep earlier points, add the new ones).\n"
                f"Request: {question}\nRunning summary so far: "
                f"{summary or '(none)'}\nNew items:\n{render_raw(chunk)}\n"
                "Updated summary:", max_tokens=220) or "").strip()
            summary = nxt or summary
        except Exception:
            pass
    return summary or render_extract(items)


_STRATEGIES = ("raw", "extract", "interpret", "mapreduce")


def choose_strategy(question: str, items: Sequence[Item], llm: Any) -> str:
    """The agent picks the rendering strategy (LLM ; heuristic fallback)."""
    if hasattr(llm, "generate"):
        try:
            r = (llm.generate(
                "Choose how to present a retrieved list to serve the request. "
                "Options: raw (full list), extract (truncated previews), "
                "interpret (one interpretation per item), mapreduce (a rolling "
                "batch summary for many items). Answer ONE word.\n"
                f"Request: {question}\nItems: {len(items)}\nChoice:",
                max_tokens=4) or "").strip().lower()
            for k in _STRATEGIES:
                if k in r:
                    return k
        except Exception:
            pass
    return "raw" if len(items) <= 8 else "extract"      # size heuristic


def render_list(question: str, items: Sequence[Item], llm: Any, *,
                strategy: str = "auto", head: int = 20, tail: int = 20,
                batch: int = 4) -> str:
    """Apply the chosen rendering strategy. `auto` lets the agent decide."""
    items = [(str(r), c) for r, c in (items or []) if r]
    if not items:
        return ""
    if strategy == "auto":
        strategy = choose_strategy(question, items, llm)
    if strategy == "extract":
        return render_extract(items, head, tail)
    if strategy == "interpret":
        return render_interpretations(items, llm, question)
    if strategy == "mapreduce":
        return render_mapreduce(items, llm, question, batch)
    return render_raw(items)


def _choose_placement(question: str, llm: Any) -> str:
    if hasattr(llm, "generate"):
        try:
            r = (llm.generate(
                "Should the retrieved list be WOVEN into a sentence (inject) or "
                "returned BARE as just the list (bare)? Answer one word.\n"
                f"Request: {question}\nChoice:", max_tokens=4) or "").strip().lower()
            if "bare" in r:
                return "bare"
            if "inject" in r:
                return "inject"
        except Exception:
            pass
    from metacog.enumeration import is_enumeration_query
    return "bare" if is_enumeration_query(question) else "inject"


def _inject(question: str, listing: str, n: int, llm: Any) -> str:
    frame = ""
    if hasattr(llm, "generate"):
        try:
            frame = (llm.generate(
                "Write ONE short framing sentence for a retrieved result, "
                "containing EXACTLY ONE placeholder {{name}} (you choose the "
                f"name) where the {n} found items will be inserted. Output only "
                f"the sentence.\nRequest: {question}\nSentence:",
                max_tokens=40) or "").strip()
        except Exception:
            frame = ""
    m = re.search(r"\{\{[^}]+\}\}", frame)
    if m:
        return frame[:m.start()] + "\n" + listing + "\n" + frame[m.end():]
    return f"Found {n} matching items:\n{listing}"


def render_bag(question: str, items: Sequence[Item], llm: Any, *,
               strategy: str = "auto", placement: str = "auto",
               head: int = 20, tail: int = 20, batch: int = 4) -> str:
    """Render the bag (chosen strategy) and place it (inject / bare). Empty bag
    -> "". This is the agent's full control over how the referenced nodes are
    surfaced in the final answer."""
    items = [(str(r), c) for r, c in (items or []) if r]
    if not items:
        return ""
    listing = render_list(question, items, llm, strategy=strategy,
                          head=head, tail=tail, batch=batch)
    if not listing:
        return ""
    if placement == "auto":
        placement = _choose_placement(question, llm)
    if placement == "bare":
        return listing
    return _inject(question, listing, len(items), llm)
