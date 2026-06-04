"""
Keyword extractors for the hybrid retrieval pipeline.

Keywords are populated at ingest time and used by retrieve_hybrid to
compute cosine similarity at the entity level — distinct from BM25 on
full text (verbatim match) and from cosine on the full semantic
embedding.

Cor. 5 : if an LLM extractor is used, its output is tagged with
keywords_source=GENERATOR for audit but never enters the observation
set O.

Two implementations :

  SimpleKeywordExtractor   pure COMPUTATION, no LLM, no dep
                           frequency-based, stopword-filtered
                           good baseline, misses rare entities

  LLMKeywordExtractor      GENERATOR, uses Claude API
                           catches proper nouns, dates, named entities
                           cached in memory to amortize cost
                           requires ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Optional, Protocol

from metacog.epistemic import SourceClass


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")


_BASIC_STOPWORDS = frozenset({
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "with", "on", "at", "by", "from", "into",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "and", "or", "but", "not", "no", "so", "as", "if", "when",
    "where", "who", "what", "how", "why", "than", "then", "there",
    "their", "them", "his", "her", "its", "our", "your", "do", "does",
    "did", "done", "have", "has", "had", "having", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can",
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "à", "en",
    "pour", "par", "avec", "sans", "sur", "sous", "et", "ou", "ni",
    "mais", "donc", "or", "car", "ne", "pas", "non", "que", "qui",
    "quoi", "où", "comment", "pourquoi", "je", "tu", "il", "elle",
    "nous", "vous", "ils", "elles", "ce", "ça", "cette", "ces",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "est", "était", "étaient", "sont", "fut", "été",
})


def position_weighted_keyword_embedding(
    keywords, encoder,
):
    """Position-aware keyword embedding.

    Keywords are returned ORDERED by salience (top keyword = most
    discriminative entity). Encoding the joined string treats them as
    a bag, so order is washed out. We instead :

      emb = Σ_i  enc(kw_i) / (i+1)        # position weighting
      emb = emb / ||emb||                  # L2 normalize

    The weight 1/(i+1) is the same canonical decay used by RRF (and the
    apply_pull step size) — no hyperparameter. The top keyword
    contributes the full encoder output ; the i-th contributes 1/(i+1).
    Two nodes that share their TOP keywords land closer in the manifold
    than two nodes whose keywords overlap only at the tail.

    Returns None if the input is empty.
    """
    import math

    kws = [k for k in (keywords or []) if k]
    if not kws:
        return None
    acc = None
    for i, kw in enumerate(kws):
        v = encoder.encode(kw)
        w = 1.0 / (i + 1)
        if acc is None:
            acc = [x * w for x in v]
        else:
            for j, x in enumerate(v):
                acc[j] += x * w
    norm = math.sqrt(sum(x * x for x in acc))
    if norm < 1e-12:
        return tuple(acc)
    return tuple(x / norm for x in acc)


class KeywordExtractor(Protocol):
    """Protocol for keyword extractors injected into Memory."""

    def extract(self, text: str, n: int = 5) -> List[str]: ...


class SimpleKeywordExtractor:
    """Frequency-based extractor : top-N content words after stopword
    removal, ranked by (frequency desc, length desc) for stable
    ordering. Pure COMPUTATION, no LLM, no external dependency.
    """

    source: SourceClass = SourceClass.COMPUTATION

    def __init__(
        self,
        stopwords: frozenset = _BASIC_STOPWORDS,
        min_length: int = 3,
    ) -> None:
        self.stopwords = stopwords
        self.min_length = min_length

    # Calendar tokens carry no discriminative power (every dated turn has
    # them) and would crowd out real proper nouns.
    _CALENDAR = frozenset({
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday",
    })

    def extract(self, text: str, n: int = 5) -> List[str]:
        raw = text or ""
        # Drop a leading "[date] Speaker:" prefix so the date and the
        # (non-discriminative, every-turn) speaker name don't dominate.
        body = raw
        if "]" in body:
            body = body.split("]", 1)[1]
        if ":" in body[:40]:
            body = body.split(":", 1)[1]

        all_tokens = _TOKEN_RE.findall(body)
        # Proper nouns : capitalized, NOT sentence-initial, not a calendar
        # word — the discriminative entities a query usually targets
        # ("Sweden", "Vivaldi", "DoorDash"). These get priority.
        proper: List[str] = []
        for i, tok in enumerate(all_tokens):
            low = tok.lower()
            if (tok[:1].isupper() and i > 0
                    and low not in self.stopwords
                    and low not in self._CALENDAR
                    and len(low) >= self.min_length
                    and not all_tokens[i - 1].endswith((".", "!", "?"))):
                if low not in proper:
                    proper.append(low)

        tokens = [
            t.lower() for t in all_tokens
            if t.lower() not in self.stopwords
            and t.lower() not in self._CALENDAR
            and len(t) >= self.min_length
        ]
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        ordered = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
        freq_ranked = [w for w, _ in ordered]

        # Proper nouns first (capped to leave room), then fill by frequency.
        out: List[str] = []
        for w in proper[:max(1, n // 2 + 1)] + freq_ranked:
            if w not in out:
                out.append(w)
            if len(out) >= n:
                break
        return out


class LLMKeywordExtractor:
    """Claude-API-backed keyword extractor.

    Targets the same entities a human would tag : proper nouns, dates,
    rare named entities. Tagged GENERATOR for audit (Cor. 5) — the
    output is used as INDEXING data (drives keywords_embedding for
    cosine retrieval) but never enters the observation set O.

    Caches results in memory : ingesting the same text twice costs
    one API call.

    Default model : claude-haiku-4-5-20251001 (cheap + fast).
    Override via CLAUDE_KW_MODEL env or constructor kwarg.
    """

    source: SourceClass = SourceClass.GENERATOR

    _SYSTEM = (
        "Extract the most informative keywords from the text. "
        "Prefer proper nouns, dates, entities, rare terms. "
        "Skip stopwords and generic words. "
        "Return ONLY a comma-separated list, lowercase, no prose, no "
        "code fences. Output at most the number of items requested."
    )

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 64,
        api_key: Optional[str] = None,
    ) -> None:
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.model = model or os.environ.get(
            "CLAUDE_KW_MODEL", "claude-haiku-4-5-20251001",
        )
        self.max_tokens = max_tokens
        # Deterministic fallback when the LLM returns nothing (rate limit,
        # transient 529, or genuinely terse input it judges as having
        # no informative keywords — e.g. a question composed mostly of
        # stop words). Without this, retrieval falls back to the raw
        # sentence embedding, which mismatches the keyword-embedding
        # space the indexed points live in.
        self._fallback = SimpleKeywordExtractor()
        # Retryable transient API errors (overloaded / 5xx).
        try:
            self._retryable = tuple(
                e for e in (
                    getattr(anthropic, "OverloadedError", None),
                    getattr(anthropic, "InternalServerError", None),
                )
                if e is not None
            )
        except Exception:
            self._retryable = ()
        self._cache: dict[str, List[str]] = {}

    def extract(self, text: str, n: int = 5) -> List[str]:
        if not text:
            return []
        cache_key = f"{n}::{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        user_msg = (
            f"Extract up to {n} keywords from this text:\n{text}\n\n"
            f"Output: lowercase comma-separated, max {n} items."
        )
        raw = ""
        for attempt in range(4):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self._SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text if resp.content else ""
                break
            except Exception as exc:
                # Retry transient 5xx / overloaded ; on any other failure
                # bail to the deterministic fallback below.
                if self._retryable and isinstance(exc, self._retryable):
                    time.sleep(2 ** attempt)
                    continue
                raw = ""
                break

        # Strip code fences if any, then split on commas
        raw = raw.strip()
        m = re.search(r"```(?:\w+)?\s*(.+?)\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        kws = [w.strip().lower() for w in raw.split(",") if w.strip()]
        kws = [w.rstrip(".;:!?") for w in kws if w.rstrip(".;:!?")]
        kws = kws[:n]

        # If the LLM produced nothing usable, fall back to the
        # deterministic SimpleKeywordExtractor. Never CACHE an empty
        # result — a transient failure must not poison subsequent calls
        # for the same text (the original bug : one 529 would silently
        # zero out every query for that string for the rest of the run).
        if not kws:
            kws = self._fallback.extract(text, n=n)
            if not kws:
                return []
            return kws

        self._cache[cache_key] = kws
        return kws
