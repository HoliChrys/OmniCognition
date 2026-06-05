"""
Query-alignment anchor (Slices A/B/C of the drift-resistance design).

The retrieval signals that drive a walk or a `clue_search` are RELATIVE —
each turn is scored against the latest reflection, the σ-neighbourhood, or a
brainstormed clue. That relativity is what lets the walk go deep, but it
also lets it DRIFT off the original question (classic pseudo-relevance-
feedback *query drift*): a noisy clue ("ordered books on gardening") or a
co-present topic (Caroline's counselling persona) hijacks the ranking.

The literature's answer is an explicit ANCHOR to the original query, kept
ADDITIVELY alongside the relative signal (Rocchio's α·q_original term;
ReformIR's "always score relevance w.r.t. the original query"; multi-hop
RAG's "blend a fixed fraction of the original query at every hop"). The
anchor here is a ColBERT-style typed token alignment:

  • SALIENT terms — named entities (person/place/object/…) and the
    question's content keywords — are HIGH-IDF: matched by EXACT stem
    (ColBERT prefers exact lexical match for rare terms). "Researching
    adoption agencies" exact-matches the verb of "What did X research?".
  • SOFT terms — implied topics and the rest — are matched by cosine
    (semantic MaxSim) so a paraphrase still contributes.

`alignment_score(anchor, turn)` returns a value in [0, 1]; callers blend it
as `(1−α)·relative + α·alignment` (Slice B in clue_search merge, Slice C
per walk stage), never replacing the relative signal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# wh-words / auxiliaries / fillers that carry no retrieval signal.
_STOP = {
    "what", "when", "where", "who", "whom", "whose", "why", "how", "which",
    "did", "do", "does", "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
    "that", "this", "would", "could", "might", "may", "will", "shall", "can",
    "have", "has", "had", "about", "into", "from", "her", "his", "their",
    "them", "they", "she", "he", "it", "you", "your", "likely", "probably",
}

# Salient terms get the bulk of the weight (high-IDF, exact-match-preferred);
# soft terms contribute a softer semantic signal.
_W_SALIENT = 1.0
_W_SOFT = 0.4
_STEM = 5            # cheap prefix stem length for lexical matching


def _stem(w: str) -> str:
    return w.lower()[:_STEM]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]{3,}", (text or "").lower())


def _cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


@dataclass
class QueryAnchor:
    """Typed alignment target built once per question."""
    salient: List[str] = field(default_factory=list)   # exact-match terms
    soft: List[str] = field(default_factory=list)       # semantic terms
    _emb: Dict[str, tuple] = field(default_factory=dict)
    _idf: Dict[str, float] = field(default_factory=dict)  # rarity weight ∈ (0,1]
    _encoder: Any = None                                 # for turn-token MaxSim
    _tok_emb: Dict[str, tuple] = field(default_factory=dict)  # token emb cache
    question: str = ""

    def is_empty(self) -> bool:
        return not (self.salient or self.soft)

    def idf(self, term: str) -> float:
        """Rarity weight for `term` in (0, 1]. 1.0 when no corpus stats
        (everything counts) ; ubiquitous terms (the speaker's name) → ~0."""
        return self._idf.get(term, 1.0)

    def _token_embedding(self, token: str):
        """Cached embedding of a single TURN token (for ColBERT MaxSim)."""
        if self._encoder is None:
            return None
        e = self._tok_emb.get(token)
        if e is None and token not in self._tok_emb:
            try:
                e = tuple(self._encoder.encode(token))
            except Exception:
                e = None
            self._tok_emb[token] = e
        return e


def build_query_anchor(
    question: str,
    *,
    entity_extractor: Any = None,
    keyword_extractor: Any = None,
    encoder: Any = None,
    corpus_texts: Optional[Sequence[str]] = None,
) -> QueryAnchor:
    """Extract the typed alignment terms for `question`.

    SALIENT  = named-entity values (person/place/organization/object/event)
               ∪ the question's content keywords (the action verb + nouns).
    SOFT     = implied topics ∪ remaining content tokens.
    Embeddings for every term are precomputed (cached) for the semantic
    fallback. Fully failure-safe : a missing extractor just narrows the
    anchor, never raises."""
    q = question or ""
    salient: List[str] = []
    soft: List[str] = []

    # (1) typed entities / topics from the LLM entity extractor (reused).
    if entity_extractor is not None and hasattr(entity_extractor,
                                                "extract_entities"):
        try:
            for e in entity_extractor.extract_entities(q):
                v = (getattr(e, "value", "") or "").lower().strip()
                t = (getattr(e, "etype", "") or "").lower()
                if not v or t == "date":
                    continue
                if t == "topic":
                    soft.append(v)            # implied concept → semantic
                else:
                    salient.append(v)         # named entity → exact-match
        except Exception:
            pass

    # (2) content keywords (the question's verb + nouns) → salient.
    if keyword_extractor is not None and hasattr(keyword_extractor, "extract"):
        try:
            for k in keyword_extractor.extract(q, n=6):
                kl = (k or "").lower().strip()
                if kl and kl not in _STOP:
                    salient.append(kl)
        except Exception:
            pass

    # (3) fallback : raw content tokens not already captured.
    captured = {w for term in salient + soft for w in _tokens(term)}
    for w in _tokens(q):
        if w not in _STOP and w not in captured:
            soft.append(w)

    # dedupe preserving order
    salient = list(dict.fromkeys(salient))
    soft = [s for s in dict.fromkeys(soft) if s not in set(salient)]

    emb: Dict[str, tuple] = {}
    if encoder is not None:
        for term in set(salient) | set(soft):
            try:
                emb[term] = tuple(encoder.encode(term))
            except Exception:
                pass

    # IDF rarity weights — the heart of ColBERT's high-IDF preference.
    # A term in nearly every turn (the conversation's speaker name, common
    # words) carries no discriminative signal and must not inflate the
    # alignment of off-topic turns; a rare term (the question's verb / a
    # specific object) should dominate. weight = log(N/df) / log(N), in
    # (0, 1]. No corpus → all weights 1.0 (degrades to un-weighted MaxSim).
    idf: Dict[str, float] = {}
    if corpus_texts:
        docs = [{_stem(w) for w in _tokens(t)} for t in corpus_texts]
        n = len(docs) or 1
        log_n = math.log(n + 1)
        for term in set(salient) | set(soft):
            ts = {_stem(w) for w in _tokens(term)}
            if not ts:
                continue
            df = sum(1 for d in docs if ts <= d)
            idf[term] = max(0.0, math.log((n + 1) / (df + 1)) / log_n)
    return QueryAnchor(salient=salient, soft=soft, _emb=emb, _idf=idf,
                       _encoder=encoder, question=q)


def _maxsim(term: str, term_emb, turn_tokens: List[str],
            turn_stems: set, anchor: QueryAnchor,
            turn_embedding) -> float:
    """ColBERT MaxSim for one query `term` over a turn's tokens.

    The score is `max` over the turn's token embeddings of the cosine to the
    term — the late-interaction operator — with an EXACT (stem) lexical hit
    short-circuiting to 1.0 (ColBERT's high-IDF exact-match preference).
    Falls back to a turn-level cosine when per-token encoding is unavailable
    (so the operator degrades gracefully, never raises)."""
    t_stems = {_stem(w) for w in _tokens(term)}
    if t_stems and t_stems <= turn_stems:            # exact lexical MaxSim = 1
        return 1.0
    if term_emb is None:
        return 0.0
    best = 0.0
    saw_token = False
    for w in turn_tokens:                            # MaxSim over turn tokens
        te = anchor._token_embedding(w)
        if te is None:
            continue
        saw_token = True
        c = _cos(term_emb, te)
        if c > best:
            best = c
    if not saw_token:                                # no per-token emb → turn-level
        return max(0.0, _cos(term_emb, turn_embedding))
    return max(0.0, best)


def alignment_score(
    anchor: QueryAnchor,
    turn_text: str,
    turn_embedding: Optional[Sequence[float]] = None,
    *,
    lexical_only: bool = False,
) -> float:
    """ColBERT-style typed alignment of `anchor` against one turn, in [0, 1].

    For every query term, take the MaxSim over the turn's TOKENS (late
    interaction) — exact stem match short-circuits to 1.0 for the high-IDF
    salient terms, soft terms contribute their best token cosine. Weighted
    by `_W_SALIENT` / `_W_SOFT`, normalised by total weight so the result is
    comparable across turns and questions.

    `lexical_only` : skip the semantic token-MaxSim (no per-token encoding) —
    only EXACT stem matches contribute. O(tokens) set ops, no embedding work.
    Used by the walk's per-stage anchor (Slice C), where scoring every fact
    every stage with full MaxSim would mean thousands of transformer forward
    passes ; the exact-match-on-salient signal is what that channel needs (the
    semantic side is already covered by the embedding/HyDE RRF channels)."""
    if anchor.is_empty():
        return 0.0
    turn_tokens = _tokens(turn_text)
    turn_stems = {_stem(w) for w in turn_tokens}

    def _term(term: str) -> float:
        t_stems = {_stem(w) for w in _tokens(term)}
        if t_stems and t_stems <= turn_stems:
            return 1.0
        if lexical_only:
            return 0.0
        return _maxsim(term, anchor._emb.get(term), turn_tokens,
                       turn_stems, anchor, turn_embedding)

    score = 0.0
    wsum = 0.0
    for s in anchor.salient:
        w = _W_SALIENT * anchor.idf(s)               # IDF-weighted (high-IDF)
        wsum += w
        score += w * _term(s)
    for t in anchor.soft:
        w = _W_SOFT * anchor.idf(t)
        wsum += w
        score += w * _term(t)
    return score / wsum if wsum else 0.0
