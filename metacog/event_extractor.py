"""
LLM-backed EVENT detector/typer for the event-hub mechanism.

At ingestion we ask the LLM whether a turn describes or refers to a notable
EVENT (a war, election, trip, wedding, accident, project, adoption, …) and, if
so, its short NAME and its TYPE. `metacog.memory` then creates/dedups a
`PointKind.EVENT` hub (`ingest_event`) and gravitates the turn onto it
(apply_pull) — so the facts that mention the same event cluster on ONE hub.

Same discipline as the rest of the LLM layer: open type vocabulary, retry on
transient overload, GENERATOR-sourced, and NEVER cache an empty result. Opt-in
(wired only when `Memory.event_extractor` is set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol

from metacog.epistemic import SourceClass

_EVENT_CACHE: dict = {}


@dataclass
class ExtractedEvent:
    """One event the LLM found. `name` = short instance name ("the border
    war"); `etype` = open-vocabulary one-word type ("war"); `t_start` =
    optional textual start date if the turn states one."""
    name: str
    etype: str
    t_start: Optional[str] = None


class EventExtractor(Protocol):
    def extract_events(self, text: str) -> List[ExtractedEvent]: ...


_PROMPT = (
    "Read this chat/diary line. Does it describe, mention, or refer to a "
    "notable EVENT — something with a beginning that unfolds over time (a war, "
    "election, trip, wedding, move, accident, illness, project, adoption, "
    "graduation, deal, …)? NOT a generic feeling or a one-off mundane act.\n"
    "If yes, output the event(s), one per line as `name | type | start` where "
    "name is a SHORT instance name built ONLY from words in the line (do NOT "
    "invent specifics, real-world names, countries or details not present — if "
    "the line just says 'the war', name it 'the war'), type is ONE lowercase "
    "word, and start is a date if the line states one (else leave blank). If "
    "none, output nothing. Do NOT explain.\n\nLine: {text}\nEvents:"
)


class LLMEventExtractor:
    """Claude-backed event detector. Reuses ClaudeLLM credential handling."""

    source: SourceClass = SourceClass.GENERATOR

    def __init__(self, llm: Any = None) -> None:
        if llm is None:
            from metacog.llm import ClaudeLLM
            llm = ClaudeLLM()
        self.llm = llm

    def extract_events(self, text: str) -> List[ExtractedEvent]:
        t = (text or "").strip()
        if not t:
            return []
        if t in _EVENT_CACHE:
            return list(_EVENT_CACHE[t])
        if not hasattr(self.llm, "generate"):
            return []
        try:
            raw = (self.llm.generate(_PROMPT.format(text=t),
                                     max_tokens=120) or "").strip()
        except Exception:
            return []
        out: List[ExtractedEvent] = []
        for ln in raw.splitlines():
            s = ln.strip().lstrip("-*0123456789.) \t").strip()
            if not s or "|" not in s:
                continue
            parts = [p.strip() for p in s.split("|")]
            name = parts[0].lower().strip(".,:;\"'")
            etype = (parts[1].lower().split()[0].strip(".,:;\"'")
                     if len(parts) > 1 and parts[1].strip() else "")
            start = parts[2] if len(parts) > 2 and parts[2].strip() else None
            if name and etype:
                out.append(ExtractedEvent(name=name, etype=etype, t_start=start))
        out = out[:3]
        if out:                                 # never cache an empty result
            _EVENT_CACHE[t] = list(out)
        return out
