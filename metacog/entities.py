"""
LLM-backed ENTITY extractor for the edge-free entity-node mechanism.

At indexation we ask the LLM to read a turn/fact and emit the salient
entities it contains (people, places, organizations, **dates**, …).  The
LLM decides autonomously which entities exist (open type vocabulary).

Dates are decomposed by the LLM ITSELF (pure-LLM decomposition) into
day / month / year fields — we do no date parsing on our side.

Cor. 5 : the extractor's output is GENERATOR data.  Entity nodes built
from it are tagged keywords_source=GENERATOR for audit ; they are used
as INDEXING beacons (their geometric pull co-locates the real source
fact) and never enter the observation set O.

The relation between an entity and its source fact is NOT stored here —
there are no edges.  `metacog.memory` simulates the edge geometrically
with `apply_pull`, which drags the entity node onto its source fact in
the keyword manifold.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from metacog.epistemic import SourceClass


@dataclass
class ExtractedEntity:
    """One entity the LLM found in a piece of text.

    `etype`   open-vocabulary type, lowercased ("person", "place",
              "organization", "date", …).
    `value`   canonical surface form, lowercased ("melanie", "sweden",
              "20 january 2023").
    `date_parts`  for dates only : any subset of {"day","month","year"},
              decomposed by the LLM.  None / empty for non-dates.
    """

    etype: str
    value: str
    date_parts: Optional[Dict[str, str]] = field(default=None)


class EntityExtractor(Protocol):
    """Protocol for entity extractors injected into Memory.

    Deliberately a distinct method name from the keyword extractor's
    `extract(text, n)` so the two can never be confused.
    """

    def extract_entities(self, text: str) -> List[ExtractedEntity]: ...


class LLMEntityExtractor:
    """Claude-API-backed entity extractor (pure-LLM date decomposition).

    Reuses `metacog.llm.ClaudeLLM` so it inherits the full credential
    resolution (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / sk-ant-oat).
    Caches per-text : ingesting the same turn twice costs one API call.
    Total failure-safety : any error → [] (ingestion stays usable).
    """

    source: SourceClass = SourceClass.GENERATOR

    _SYSTEM = (
        "You index a short conversational message for later retrieval. "
        "Emit two kinds of items.\n"
        "(1) NAMED ENTITIES actually present : people, places, "
        "organizations, dates, objects, events. "
        "For every DATE, decompose it into day / month / year fields "
        "(omit any part the text does not give).\n"
        "(2) IMPLIED TOPICS (type \"topic\") : up to 3 concepts the "
        "message REVEALS or IMPLIES about the speaker or subject that "
        "someone might later SEARCH for, even when the word is NOT in the "
        "text. Infer the underlying category. Examples : 'my kids have so "
        "much and others don't' -> topics wealth, comfort, privilege ; "
        "'keen on counseling or mental health' -> topics psychology, "
        "counseling ; 'won my video game tournament with my team' -> "
        "topics teammates, gaming, friends ; 'I've been so tired and "
        "down lately' -> topics depression, fatigue. Only add a topic "
        "you are confident the message genuinely implies ; never invent.\n"
        "Return ONLY a strict JSON array, no prose, no code fences. "
        "Each item: {\"type\":<lowercase>, \"value\":<lowercase>}. "
        "Date items add \"parts\": {\"day\":\"20\",\"month\":\"january\","
        "\"year\":\"2023\"} (any subset). "
        "Example: [{\"type\":\"person\",\"value\":\"melanie\"},"
        "{\"type\":\"place\",\"value\":\"sweden\"},"
        "{\"type\":\"topic\",\"value\":\"wealth\"},"
        "{\"type\":\"date\",\"value\":\"20 january 2023\","
        "\"parts\":{\"day\":\"20\",\"month\":\"january\",\"year\":\"2023\"}}]. "
        "If there is nothing salient, return []."
    )

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
        llm: Any = None,
    ) -> None:
        if llm is not None:
            self.llm = llm
        else:
            from metacog.llm import ClaudeLLM
            import os

            self.llm = ClaudeLLM(
                model=model or os.environ.get(
                    "CLAUDE_ENTITY_MODEL", "claude-haiku-4-5-20251001",
                ),
                max_tokens=max_tokens,
                api_key=api_key,
                system=self._SYSTEM,
            )
        self.max_tokens = max_tokens
        self._cache: Dict[str, List[ExtractedEntity]] = {}

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        if not text:
            return []
        if text in self._cache:
            return self._cache[text]

        raw = ""
        try:
            raw = self.llm.generate(text, max_tokens=self.max_tokens)
        except Exception:
            return []

        ents = self._parse(raw)
        self._cache[text] = ents
        return ents

    # ------------------------------------------------------------------
    # Parsing — TOTAL : never raises, drops anything malformed.
    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw: str) -> List[ExtractedEntity]:
        if not raw:
            return []
        raw = raw.strip()
        # Strip code fences if the model added them despite instructions.
        m = re.search(r"```(?:\w+)?\s*(.+?)\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        # Be liberal : grab the first JSON array in the text.
        if not raw.startswith("["):
            b = raw.find("[")
            e = raw.rfind("]")
            if b != -1 and e != -1 and e > b:
                raw = raw[b:e + 1]
        try:
            data = json.loads(raw)
        except Exception:
            return []
        if not isinstance(data, list):
            return []

        out: List[ExtractedEntity] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            etype = str(item.get("type", "")).strip().lower()
            value = str(item.get("value", "")).strip().lower()
            if not etype or not value:
                continue
            parts: Optional[Dict[str, str]] = None
            if etype == "date":
                p = item.get("parts")
                if isinstance(p, dict):
                    parts = {}
                    for k in ("day", "month", "year"):
                        v = p.get(k)
                        if v is not None and str(v).strip():
                            parts[k] = str(v).strip().lower()
                    if not parts:
                        parts = None
            out.append(ExtractedEntity(etype=etype, value=value,
                                       date_parts=parts))
        return out
