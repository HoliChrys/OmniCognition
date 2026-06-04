"""
Atomic-fact extractor (mem0-style) adapted to MetaCog-Mem.

Instead of indexing only the raw conversational turn ("Thanks Melanie!
This necklace is special — a gift from my grandma in my home country,
Sweden."), we ask the LLM to decompose each turn into self-contained
ATOMIC facts with pronouns/speaker resolved ("Caroline's home country is
Sweden."). These clean, discriminative units are the retrieval handles
that fix the keyword-extraction noise problem on raw turns.

Architecture fit :
  - Atomic facts are GENERATOR-sourced (Cor. 5) — indexing only, never
    observations.
  - They carry parents=[source dia_id] so retrieval can resolve an atomic
    hit back to the gold-evidence dia_id (LoCoMo scores recall on dia_ids).
  - apply_pull co-locates each atomic fact with its source turn.

Total failure-safety : any error → [] (ingestion stays usable). Cached.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from metacog.epistemic import SourceClass


class AtomicFactExtractor:
    """Claude-backed atomic-fact decomposer. Reuses ClaudeLLM credentials."""

    source: SourceClass = SourceClass.GENERATOR

    _SYSTEM = (
        "You decompose ONE conversational message into atomic facts. "
        "Each atomic fact is a SHORT standalone declarative sentence that "
        "is true on its own, with all pronouns and references RESOLVED to "
        "explicit names (use the speaker's name and the people/things "
        "mentioned). Keep the discriminative nouns (places, objects, "
        "events, dates). One fact per line, no numbering, no prose. "
        "If the message carries no factual content (pure greeting / filler "
        "/ question), output nothing.\n"
        "Example input  (speaker=Caroline): 'Thanks Mel! This necklace is "
        "super special - a gift from my grandma in my home country, "
        "Sweden.'\n"
        "Example output:\n"
        "Caroline's necklace is a gift from her grandma.\n"
        "Caroline's grandma is from Sweden.\n"
        "Caroline's home country is Sweden."
    )

    def __init__(self, model: Optional[str] = None, max_tokens: int = 200,
                 api_key: Optional[str] = None, llm: Any = None) -> None:
        if llm is not None:
            self.llm = llm
        else:
            import os
            from metacog.llm import ClaudeLLM
            self.llm = ClaudeLLM(
                model=model or os.environ.get(
                    "CLAUDE_ATOMIC_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=max_tokens, api_key=api_key, system=self._SYSTEM,
            )
        self.max_tokens = max_tokens
        self._cache: dict[str, List[str]] = {}

    def extract_atoms(self, text: str, *, speaker: str = "") -> List[str]:
        if not text:
            return []
        key = f"{speaker}::{text}"
        if key in self._cache:
            return self._cache[key]
        prompt = f"(speaker={speaker}) {text}" if speaker else text
        raw = ""
        try:
            raw = self.llm.generate(prompt, max_tokens=self.max_tokens)
        except Exception:
            return []
        atoms = []
        for line in (raw or "").splitlines():
            s = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
            if len(s.split()) >= 3:
                atoms.append(s)
        self._cache[key] = atoms
        return atoms
