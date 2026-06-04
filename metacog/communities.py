"""
Edge-free community detection on the manifold (à la GraphRAG / LightRAG).

We do NOT store graph edges. At read-time we materialize a similarity
graph over the FACT points (cosine on keyword embeddings, emergent
threshold median + σ), run a single Louvain pass, and treat each
community ≥ `min_cluster_size` as a "Level-1 group" that becomes a
named observator the agent can activate at walk_start.

Single-pass (one Louvain level), parameter-free apart from a single
minimum-size floor which is itself emergent (max(3, ceil(log2(n)))).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from metacog.epistemic import Point, PointKind
from metacog.geometry import cosine

_WORD_RE = None
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "for", "in", "on",
    "at", "by", "with", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "should", "could", "may", "might", "must", "shall", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "it", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "its", "as", "if", "so", "no", "not", "yes", "what", "when",
    "where", "who", "how", "why", "which", "all", "any", "some", "just",
    "very", "more", "less", "than", "then", "there", "here",
    # months / days — conversation timestamps land in keywords
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "yesterday", "today", "tomorrow", "week", "month", "year",
    "morning", "evening", "night",
    # conversational politesses / fillers that swamp keyword frequency
    "hey", "hi", "hello", "wow", "oh", "yeah", "yes", "no", "ok", "okay",
    "really", "cool", "awesome", "good", "great", "nice", "thanks", "thank",
    "love", "like", "want", "know", "think", "just", "got", "get", "going",
    "make", "made", "way", "lot", "lots", "much", "many", "see", "saw",
    "look", "feel", "felt", "say", "said", "tell", "told",
    # contractions / possessives that frequency counts surface as topics
    "i'm", "i've", "i'll", "i'd", "we're", "we've", "we'll", "we'd",
    "you're", "you've", "you'll", "you'd", "he's", "she's", "it's",
    "that's", "there's", "here's", "what's", "who's", "let's", "don't",
    "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
    "isn't", "wasn't", "aren't", "weren't", "haven't", "hasn't",
    "hadn't", "shouldn't", "stuff", "kind", "lots", "out", "back",
    "fun", "amazing", "happy", "way", "ways",
}


def _content_tokens(text: str, *, drop: Sequence[str] = ()) -> List[str]:
    """Extract discriminative content words from `text`.

    Drops generic stopwords, calendar / politesse noise, and a caller-
    provided `drop` list (typically the conversation's speaker names).
    """
    import re
    global _WORD_RE
    if _WORD_RE is None:
        _WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
    drop_set = {d.lower() for d in drop}
    return [
        t.lower() for t in _WORD_RE.findall(text or "")
        if t.lower() not in _STOPWORDS
        and t.lower() not in drop_set
        and len(t) > 2
        and not t.isdigit()
    ]


def _detect_speakers(points: Sequence[Point]) -> List[str]:
    """Best-effort speaker discovery from FACT content : looks for the
    "[DATE] Speaker: ..." prefix the LoCoMo ingester writes. Returns the
    set of first-tokens that appear ≥ 5 times. Speakers carry zero
    discriminative power for community detection, only noise."""
    import re
    speaker_re = re.compile(r"^\s*(?:\[[^\]]+\]\s*)?(\w+):")
    counts: Counter[str] = Counter()
    for p in points:
        m = speaker_re.match(p.content or "")
        if m:
            counts[m.group(1).lower()] += 1
    return [s for s, n in counts.items() if n >= 5]


@dataclass
class Community:
    """One detected community of FACT points."""

    id: str                       # observator id, e.g. "auto-comm-3"
    point_ids: List[str]          # member point ids
    keywords: List[str]           # top frequency keywords across members
    summary_words: int = 0        # diagnostic : total content tokens scanned


def _similarity_threshold(scores: Sequence[float]) -> float:
    """Emergent edge threshold : median + σ over pairwise cosines.

    Returns 0.0 when scores is empty (no edges → only singleton
    communities, which the caller filters out by min_size)."""
    n = len(scores)
    if n == 0:
        return 0.0
    sorted_s = sorted(scores)
    median = sorted_s[n // 2]
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    sigma = math.sqrt(var)
    return median + sigma


def detect_level1_communities(
    points: Sequence[Point],
    encoder: Any,
    *,
    min_cluster_size: Optional[int] = None,
    threshold: Optional[float] = None,
    drop_tokens: Optional[Sequence[str]] = None,
) -> List[Community]:
    """Run a single-pass Louvain on the cleaned-content similarity graph
    over FACT points and return communities of size ≥ min_cluster_size.

    `min_cluster_size` defaults to max(3, ceil(log2(n_facts))) so it
    scales with corpus size without a tuning knob.

    Similarity is computed on a CLEANED content embedding (stopwords +
    months + politesses + auto-detected speakers stripped) so it tracks
    topical structure rather than calendar / speaker noise. The points'
    own keywords_embedding is NOT touched.
    """
    # Cluster ONLY on genuine conversation FACTs : THOUGHTs / ACTIONs are
    # walk-time scaffolding and entity beacons (id "entity_*") are common
    # to all clusters (shared structural anchors) — neither should be
    # assigned to one community.
    facts = [
        p for p in points
        if p.kind == PointKind.FACT and not p.id.startswith("entity_")
    ]
    n = len(facts)
    if n < 4:
        return []
    if min_cluster_size is None:
        min_cluster_size = max(3, int(math.ceil(math.log2(n))))

    speakers = list(drop_tokens) if drop_tokens else _detect_speakers(facts)
    # Cleaned-content embeddings — discriminative on topics, not speakers.
    cleaned_emb: List[Tuple[float, ...]] = []
    for p in facts:
        toks = _content_tokens(p.content, drop=speakers)
        joined = " ".join(toks[:32]) if toks else ""
        cleaned_emb.append(tuple(encoder.encode(joined)) if joined else None)

    edges: List[Tuple[int, int, float]] = []
    scores: List[float] = []
    for i in range(n):
        if cleaned_emb[i] is None:
            continue
        for j in range(i + 1, n):
            if cleaned_emb[j] is None:
                continue
            s = cosine(cleaned_emb[i], cleaned_emb[j])
            scores.append(s)
            edges.append((i, j, s))
    if not scores:
        return []
    thr = threshold if threshold is not None else _similarity_threshold(scores)

    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, j, s in edges:
        if s > thr:
            g.add_edge(i, j, weight=s)

    partition = louvain_communities(g, weight="weight", seed=0)

    out: List[Community] = []
    for k, comm in enumerate(partition):
        if len(comm) < min_cluster_size:
            continue
        members = [facts[i] for i in comm]
        counter: Counter[str] = Counter()
        scanned = 0
        for p in members:
            toks = _content_tokens(p.content, drop=speakers)
            scanned += len(toks)
            counter.update(toks)
        top = [w for w, _ in counter.most_common(8)]
        out.append(Community(
            id=f"auto-comm-{k}",
            point_ids=[p.id for p in members],
            keywords=top,
            summary_words=scanned,
        ))
    return out
