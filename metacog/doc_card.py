"""
Document Card — ONE structured LLM reading per document at ingestion.

Phase 1 of docs/ingest_index_plan.md. Replaces up to five separate ingest-time
LLM reads (keywords, entities, events, tone/gloss — and pre-computes the
stance + answerable-questions used by later phases) with a single call : the
document is read ONCE, coherently, and every index artifact derives from that
reading. Query-independent by construction — everything here is a property of
the document, amortized across all later queries.

Failure-safe : any parse/LLM failure returns None and the caller falls back to
the individual extractors. Cached ; an unparseable result is NEVER cached.
Cor. 5 : source = GENERATOR — card content can become content, never evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from metacog.entities import ExtractedEntity
from metacog.epistemic import SourceClass
from metacog.event_extractor import ExtractedEvent

_TONES = ("ironic", "hyperbole", "echo-register", "plain")

_PROMPT = (
    "Read this text ONCE and fill the card. Be faithful to the text ; do not "
    "invent details.\n"
    "KEYWORDS: 3-6 comma-separated content words from the text\n"
    "ENTITIES: `value|type` pairs separated by ` ; ` (types : person, place, "
    "organization, date, …) or NONE\n"
    "EVENT: `name | type | start` — a notable event the text refers to (short "
    "name built only from words present ; type = one lowercase word ; start = "
    "date if stated, else blank) or NONE\n"
    "TONE: one of ironic (irony/sarcasm/satire/absurd or false-premise/"
    "mock-innocent), hyperbole, echo-register (grandiose official/promotional "
    "wording echoed by an ordinary voice), plain — do not force irony where "
    "there is none\n"
    "INTENDED: one sentence — what the author TRULY means (for plain, restate "
    "plainly)\n"
    "STANCE: one sentence — the position the text takes (what it supports / "
    "opposes / mocks), or NONE if it takes none\n"
    "QUESTIONS: 2 questions this text answers, separated by ` ; ` (include an "
    "oblique phrasing when natural)\n\n"
    "TEXT : {text}\n"
    "Answer with exactly the 7 labelled lines."
)

_CARD_CACHE: Dict[str, "DocCard"] = {}


@dataclass
class DocCard:
    """One coherent reading of a document — every ingest artifact's source."""
    keywords: List[str] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    events: List[ExtractedEvent] = field(default_factory=list)
    tone: str = "plain"
    intended: str = ""
    stance: str = ""
    questions: List[str] = field(default_factory=list)


def _parse_card(raw: str) -> Optional[DocCard]:
    """Parse the 7 labelled lines. TONE is the minimum for validity — without
    it the reading is unusable and must NOT be cached."""
    card = DocCard()
    tone_seen = False
    for ln in (raw or "").splitlines():
        ls = ln.strip().lstrip("-•* ")
        low = ls.lower()
        if low.startswith("keywords:"):
            card.keywords = [w.strip() for w in ls.split(":", 1)[1].split(",")
                             if w.strip()][:6]
        elif low.startswith("entities:"):
            body = ls.split(":", 1)[1].strip()
            if body and body.lower() != "none":
                for pair in body.split(";"):
                    if "|" not in pair:
                        continue
                    val, _, et = pair.partition("|")
                    val, et = val.strip().lower(), et.strip().lower()
                    if val and et:
                        card.entities.append(
                            ExtractedEntity(etype=et, value=val))
        elif low.startswith("event:"):
            body = ls.split(":", 1)[1].strip()
            if body and body.lower() != "none":
                parts = [p.strip() for p in body.split("|")]
                if len(parts) >= 2 and parts[0] and parts[1]:
                    card.events.append(ExtractedEvent(
                        name=parts[0].lower(),
                        etype=parts[1].lower().split()[0],
                        t_start=(parts[2] or None) if len(parts) > 2 else None))
        elif low.startswith("tone:"):
            cand = ls.split(":", 1)[1].strip().lower()
            t = next((t for t in _TONES if t in cand), None)
            if t:
                card.tone, tone_seen = t, True
        elif low.startswith("intended:"):
            card.intended = ls.split(":", 1)[1].strip()
        elif low.startswith("stance:"):
            body = ls.split(":", 1)[1].strip()
            card.stance = "" if body.lower() == "none" else body
        elif low.startswith("questions:"):
            card.questions = [q.strip() for q in ls.split(":", 1)[1].split(";")
                              if q.strip()][:3]
    return card if tone_seen else None


class DocCardExtractor:
    """Claude-backed single-read card extractor (one LLM call per document)."""

    source: SourceClass = SourceClass.GENERATOR

    def __init__(self, llm: Any):
        self.llm = llm

    def extract_card(self, text: str) -> Optional[DocCard]:
        t = (text or "").strip()
        if not t or not hasattr(self.llm, "generate"):
            return None
        if t in _CARD_CACHE:
            return _CARD_CACHE[t]
        try:
            raw = self.llm.generate(_PROMPT.format(text=t[:600]),
                                    max_tokens=240) or ""
        except Exception:
            return None
        card = _parse_card(raw)
        if card is None:
            return None                          # unparseable -> never cache
        _CARD_CACHE[t] = card
        return card
