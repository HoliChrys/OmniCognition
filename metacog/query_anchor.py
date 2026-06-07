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
    """Typed alignment target built once per question.

    Holds the original input ENCODED ONCE (`anchor_emb`) plus the typed
    salient/soft terms for the cheap lexical exact-match. Scoring a fact is
    then O(1): one cosine against the fact's already-stored embedding +
    set-membership on stems — no per-stage, per-token re-encoding."""
    salient: List[str] = field(default_factory=list)   # exact-match terms
    soft: List[str] = field(default_factory=list)       # semantic terms
    anchor_emb: Optional[tuple] = None                   # input encoded ONCE
    _idf: Dict[str, float] = field(default_factory=dict)  # rarity weight ∈ (0,1]
    question: str = ""
    # RELATIVE half of the anchor : the latest walk THOUGHT's keywords — the
    # abstraction the reasoning has reached ("overweight", "well-off"), kept
    # SEPARATE from the input and refreshed each stage. The input ranks the
    # turn "fingers too big" near STRESS (cos .07) ; only the thought's
    # abstraction "obesity" ranks it first (cos .15) — so the anchor is the
    # two ADDED. Encoded ONCE per refresh, scored by one cosine (no re-encode).
    relative: List[str] = field(default_factory=list)
    relative_emb: Optional[tuple] = None

    def is_empty(self) -> bool:
        return not (self.salient or self.soft)

    def idf(self, term: str) -> float:
        """Rarity weight for `term` in (0, 1]. 1.0 when no corpus stats
        (everything counts) ; ubiquitous terms (the speaker's name) → ~0."""
        return self._idf.get(term, 1.0)

    def with_relative(self, terms: Sequence[str], encoder: Any = None
                      ) -> "QueryAnchor":
        """Refresh the RELATIVE half from the latest thought's keywords.

        The thought is "une pensée plus distancielle des fact" — it lifts the
        surface vocabulary ("fingers too big", "go for a run") into the
        answer register ("overweight"). We keep only terms NOT already in the
        input's salient set (those add nothing) and that carry signal, encode
        the joined phrase ONCE, and store it. Mutates and returns self so the
        same anchor object stays wired into the walk. Fully failure-safe."""
        seen = {_stem(w) for s in self.salient for w in _tokens(s)}
        rel: List[str] = []
        for t in terms or []:
            tl = (t or "").lower().strip()
            if not tl or tl in _STOP:
                continue
            ts = {_stem(w) for w in _tokens(tl)}
            if not ts or ts <= seen:        # already covered by the input
                continue
            rel.append(tl)
        rel = list(dict.fromkeys(rel))[:8]
        self.relative = rel
        self.relative_emb = None
        if rel and encoder is not None:
            try:
                self.relative_emb = tuple(encoder.encode(" ".join(rel)))
            except Exception:
                self.relative_emb = None
        return self


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

    # The ORIGINAL input is encoded ONCE, as-is — NOT synthesised from the
    # extracted terms. This is the fixed anchor vector; scoring a fact is
    # then a single cosine against the fact's already-stored embedding (no
    # per-stage / per-token re-encoding). The typed terms are kept SEPARATE,
    # used only for the cheap lexical exact-match.
    anchor_emb: Optional[tuple] = None
    if encoder is not None and q.strip():
        try:
            anchor_emb = tuple(encoder.encode(q))
        except Exception:
            anchor_emb = None

    # IDF rarity weights — ColBERT's high-IDF preference, parameter-free.
    # A term in nearly every turn (the speaker name, common words) carries
    # no discriminative signal; a rare term (the question's verb / a
    # specific object) should dominate. weight = log(N/df)/log(N) ∈ (0,1].
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
    return QueryAnchor(salient=salient, soft=soft, anchor_emb=anchor_emb,
                       _idf=idf, question=q)


# Weight of the lexical exact-match vs the semantic anchor cosines in the
# combined alignment. Lexical (IDF-weighted exact stem hit on a salient term —
# ColBERT's high-IDF exact-match) is the precise, drift-safe core. The two
# semantic supports are the INPUT-embedding cosine and the RELATIVE
# thought-abstraction cosine.
_W_LEX = 0.7
_W_SEM = 0.3            # input-embedding cosine (drift-prone : off in the walk)
_W_REL = 0.4           # relative thought-abstraction cosine (the inference bridge)


def alignment_score(
    anchor: QueryAnchor,
    turn_text: str,
    turn_embedding: Optional[Sequence[float]] = None,
    *,
    lexical_only: bool = False,
    use_input_sem: bool = True,
    use_relative_sem: bool = True,
) -> float:
    """Typed alignment of `anchor` against one fact, in [0, 1] — O(1).

    The anchor is TWO things ADDED, per the design : the INPUT and the
    RELATIVE thought-abstraction. Three cheap signals, none re-encoding :
      • LEXICAL (precise) — IDF-weighted fraction of the salient/soft terms
        whose stem appears in the fact (ColBERT's high-IDF exact-match, set
        ops only). The ubiquitous speaker name is IDF-driven to ~0.
      • INPUT SEMANTIC — cosine of the ORIGINAL-input embedding (encoded once)
        against the fact's stored embedding. DRIFT-PRONE : cos(question, fact)
        favours the co-present dominant topic, so the walk turns it OFF
        (`use_input_sem=False`) and keeps only the precise lexical core.
      • RELATIVE SEMANTIC — cosine of the latest thought's abstraction
        (`relative_emb`, e.g. "overweight") against the fact. This is the
        inference BRIDGE : the input ranks "fingers too big" near stress
        (cos .07), only the abstraction "obesity" ranks it first (cos .15).
        SAFE to keep on in the walk : it is a TARGETED concept, not the broad
        question, so it pulls toward the reasoning's conclusion, not the
        ambient topic.

    `lexical_only` is a shortcut dropping BOTH semantic channels (callers with
    no fact embedding). Otherwise each channel is gated independently."""
    if anchor.is_empty():
        return 0.0
    turn_stems = {_stem(w) for w in _tokens(turn_text)}

    # Lexical : IDF-weighted exact-stem hit fraction.
    lex = 0.0
    wsum = 0.0
    for term, base in (
        [(s, _W_SALIENT) for s in anchor.salient]
        + [(t, _W_SOFT) for t in anchor.soft]
    ):
        w = base * anchor.idf(term)
        wsum += w
        t_stems = {_stem(x) for x in _tokens(term)}
        if t_stems and t_stems <= turn_stems:
            lex += w
    lex = lex / wsum if wsum else 0.0

    if lexical_only or turn_embedding is None:
        return lex
    score = _W_LEX * lex
    if use_input_sem and anchor.anchor_emb is not None:
        score += _W_SEM * max(0.0, _cos(anchor.anchor_emb, turn_embedding))
    if use_relative_sem and anchor.relative_emb is not None:
        score += _W_REL * max(0.0, _cos(anchor.relative_emb, turn_embedding))
    return score
