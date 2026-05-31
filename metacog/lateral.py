"""
MetaCog-Mem — LATERAL collision (co-retrieval-driven consolidation).

The three existing collision mechanisms (proximity fission, Chasles
compression, content-identity merge) are all GEOMETRIC : they fire on
where the nodes sit, or on how similar their content is. Lateral
collision is FUNCTIONAL : it fires on how the nodes BEHAVE under
retrieval.

Idea
----
If, across MANY different queries, the same one-or-few nodes keep coming
back TOGETHER and ADJACENT in the kNN / BM25 ranking, those nodes are
functionally interchangeable retrieval targets : they represent "the
same thing, give or take", and they wastefully squat several slots of
every result list. They deserve to be COLLIDED into a single
representative node.

The catch — query diversity
---------------------------
Two near-identical queries trivially return the same results ; that is
no evidence of redundancy. We only learn that nodes A and B are the same
retrieval target when DISTANT (low-cosine) queries ALL surface them
together. So every co-occurrence is weighted by the NOVELTY of its query
relative to the query directions already counted for that pair :

    weight(A,B) += 1 - max_cosine(q, dirs_already_seen(A,B))

A redundant near-duplicate query adds ~0 ; an orthogonal query adds ~1
and registers its direction. The stored direction set is capped, so the
ledger stays small ("garder ça de manière intelligente").

The threshold — spike activation, reused
-----------------------------------------
We reuse the Poisson emergent threshold of the spike system. If the
total accumulated weight were spread uniformly over the W distinct pairs
ever observed, each pair would expect lambda = total_weight / W. A pair
"laterally spikes" when its weight exceeds lambda + sqrt(lambda) — one
standard deviation above the uniform noise floor. No hyperparameter.

The gate — only when the cloud is big and tangled
-------------------------------------------------
With a handful of nodes, everything co-occurs in every kNN ; redundancy
is undefined. The pass only runs once the memory holds a substantial,
tag-rich population whose tags sit in genuinely different intersections.
Below the gate, detection returns nothing.

Resolution
----------
A qualifying group is folded into ONE keeper (the most-retrieved member,
tie-broken by keyword richness). The keeper ABSORBS the others' counters,
keywords and tags — so the single survivor still matches every query
angle the group used to — and the absorbed ids are removed and aliased
to the keeper (reusing the merge-alias machinery, so recall scoring still
resolves them). One node, one kNN slot, full retrieval coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from metacog.epistemic import EpistemicState, Point
from metacog.geometry import cosine

Vector = Tuple[float, ...]
Pair = Tuple[str, str]


def _pair_key(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass
class LateralLedger:
    """Accumulates diversity-weighted co-retrieval adjacency.

    `weight`      : pair -> total novelty-weighted co-occurrence weight.
    `dirs`        : pair -> capped list of representative query directions
                    already counted (used to discount redundant queries).
    `total_weight`: running sum of all `weight` values (for the Poisson
                    baseline).
    `n_events`    : number of retrievals recorded (diagnostics only).
    """

    weight: Dict[Pair, float] = field(default_factory=dict)
    dirs: Dict[Pair, List[Vector]] = field(default_factory=dict)
    total_weight: float = 0.0
    n_events: int = 0

    def pairs_seen(self) -> int:
        return len(self.weight)


# Default knobs. These are operational bounds (memory budget, gating
# floors), not statistical thresholds — the firing threshold itself is
# fully emergent (Poisson, below).
_WINDOW = 2            # "un ou plusieurs résultats à la suite" : adjacency
                       # span — pairs within this many ranks count.
_NOVELTY_MIN = 0.15    # a co-occurrence whose query adds < this novelty is
                       # ignored (redundant probing of the same direction).
_MAX_DIRS = 8          # cap on stored query directions per pair.
_MIN_NODES = 64        # gate : "un nombre assez conséquent de nodes".
_MIN_DISTINCT_TAGS = 12     # gate : tag vocabulary must be rich…
_MIN_MULTI_TAG_NODES = 16   # …and enough nodes must live in tag
                            # intersections (≥ 2 tags each).


def record_coretrieval(
    ledger: LateralLedger,
    ranked_ids: Sequence[str],
    query_emb: Optional[Vector],
    *,
    window: int = _WINDOW,
    novelty_min: float = _NOVELTY_MIN,
    max_dirs: int = _MAX_DIRS,
) -> None:
    """Fold one retrieval's ranked result into the ledger.

    For every pair of result ids within `window` ranks of each other,
    add a co-occurrence weight discounted by how redundant `query_emb`
    is with the directions already counted for that pair. `query_emb`
    None (no embedding available) falls back to undiscounted unit weight.
    """
    ledger.n_events += 1
    ids = [i for i in ranked_ids if i]
    if len(ids) < 2:
        return
    for i in range(len(ids)):
        for j in range(i + 1, min(i + window + 1, len(ids))):
            a, b = ids[i], ids[j]
            if a == b:
                continue
            key = _pair_key(a, b)
            seen = ledger.dirs.get(key)
            if query_emb is None:
                novelty = 1.0
            elif not seen:
                novelty = 1.0
            else:
                novelty = 1.0 - max(cosine(query_emb, d) for d in seen)
            if novelty < novelty_min:
                continue
            ledger.weight[key] = ledger.weight.get(key, 0.0) + novelty
            ledger.total_weight += novelty
            # Register this query direction if it widens the spread and we
            # still have room (keeps the ledger bounded).
            if query_emb is not None:
                if seen is None:
                    ledger.dirs[key] = [tuple(query_emb)]
                elif len(seen) < max_dirs:
                    seen.append(tuple(query_emb))


# ---------------------------------------------------------------------------
# Threshold (Poisson — same emergent law as spike_threshold)
# ---------------------------------------------------------------------------


def lateral_threshold(ledger: LateralLedger) -> Optional[float]:
    """Weight above which a pair is a statistically-significant co-retrieval
    cluster : lambda + sqrt(lambda), lambda = total_weight / pairs_seen.

    None when too few distinct pairs have been observed for the Poisson
    baseline to be meaningful."""
    w = ledger.pairs_seen()
    if w < 4 or ledger.total_weight <= 0.0:
        return None
    lam = ledger.total_weight / w
    return lam + math.sqrt(lam)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _facets(p: Point) -> set:
    """A node's position in the tag manifold : its tags UNION its keywords.

    The user's gate is phrased in terms of "tags in different
    intersections". In this system the keyword list is explicitly the
    node's projection onto the entity manifold (see node_sigma /
    effective_keyword_embedding), so keywords play the same intersection
    role as tags — and the default (no-entity) config carries only
    keywords. We treat both as facets. Structural `ref:` tags (spike
    back-references) are not semantic facets and are excluded."""
    facets = {t for t in (p.tags or []) if not t.startswith("ref:")}
    facets.update(k.lower() for k in (p.keywords or []))
    return facets


def gate_ready(
    points: Sequence[Point],
    *,
    min_nodes: int = _MIN_NODES,
    min_distinct_tags: int = _MIN_DISTINCT_TAGS,
    min_multi_tag_nodes: int = _MIN_MULTI_TAG_NODES,
) -> bool:
    """True once the cloud is big and tangled enough for redundancy to be
    well-defined : enough nodes, a rich FACET vocabulary (tags ∪
    keywords), and enough nodes sitting in facet INTERSECTIONS (carrying
    ≥ 2 facets). Below this, every node recurs in every kNN and
    'laterally redundant' is meaningless."""
    if len(points) < min_nodes:
        return False
    distinct: set = set()
    multi = 0
    for p in points:
        f = _facets(p)
        if len(f) >= 2:
            multi += 1
        distinct.update(f)
    return len(distinct) >= min_distinct_tags and multi >= min_multi_tag_nodes


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _eligible(p: Point) -> bool:
    if p.state in {EpistemicState.DEPRECATED, EpistemicState.INVALID}:
        return False
    # Structural/index nodes are never collided away.
    return not p.id.startswith(("entity_", "atom_", "lateral_"))


def detect_lateral_groups(
    ledger: LateralLedger,
    points: Sequence[Point],
    *,
    min_nodes: int = _MIN_NODES,
    require_gate: bool = True,
) -> List[List[str]]:
    """Return groups (each ≥ 2 ids) of laterally-redundant, same-kind nodes.

    A pair qualifies when its diversity-weighted co-retrieval weight
    exceeds the emergent Poisson threshold AND both endpoints are
    eligible same-kind nodes that exist in `points`. Qualifying pairs are
    transitively unioned into groups (one keeper per group downstream).
    """
    if require_gate and not gate_ready(points, min_nodes=min_nodes):
        return []
    thr = lateral_threshold(ledger)
    if thr is None:
        return []
    by_id = {p.id: p for p in points}

    # Union-find over qualifying pairs.
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for (a, b), w in ledger.weight.items():
        if w <= thr:
            continue
        pa, pb = by_id.get(a), by_id.get(b)
        if pa is None or pb is None:
            continue
        if not _eligible(pa) or not _eligible(pb):
            continue
        if pa.kind != pb.kind:           # intra-kind only, like every collision
            continue
        union(a, b)

    groups: Dict[str, List[str]] = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LateralEvent:
    timestamp: float
    keeper_id: str
    absorbed_ids: Tuple[str, ...]
    group_weight: float
    threshold: float


@dataclass
class LateralReport:
    collided: List[LateralEvent] = field(default_factory=list)
    aliases: Dict[str, str] = field(default_factory=dict)  # absorbed -> keeper


def _keeper_of(group: Sequence[Point]) -> Point:
    """Best representative : most-retrieved (n_uses), then richest keyword
    handle, then most corroborated. The survivor must be the strongest
    retrieval magnet so coverage is preserved."""
    return max(
        group,
        key=lambda p: (p.n_uses, len(p.keywords or []), p.n_corrob),
    )


def resolve_lateral(
    keeper: Point,
    absorbed: Sequence[Point],
    encoder: Any,
    t_now: float,
    *,
    group_weight: float = 0.0,
    threshold: float = 0.0,
) -> LateralEvent:
    """Fold `absorbed` into `keeper`. The keeper absorbs counters and —
    crucially — the UNION of keywords and tags, so the single survivor
    still matches every query angle the group covered. Its keyword
    embedding is recomputed from the enlarged handle."""
    from metacog.keywords import position_weighted_keyword_embedding

    merged_kws: List[str] = list(keeper.keywords or [])
    seen_kw = {k.lower() for k in merged_kws}
    for p in absorbed:
        keeper.n_corrob += p.n_corrob + 1     # a redundant target is corroboration
        keeper.n_contra += p.n_contra
        keeper.n_uses += p.n_uses
        keeper.t_last_obs = max(keeper.t_last_obs, p.t_last_obs, t_now)
        for k in (p.keywords or []):
            if k.lower() not in seen_kw:
                merged_kws.append(k)
                seen_kw.add(k.lower())
        for tg in (p.tags or []):
            if tg not in keeper.tags:
                keeper.tags.append(tg)
    keeper.keywords = merged_kws
    if merged_kws and encoder is not None:
        emb = position_weighted_keyword_embedding(merged_kws, encoder)
        if emb is not None:
            keeper.keywords_embedding = emb
    return LateralEvent(
        timestamp=t_now,
        keeper_id=keeper.id,
        absorbed_ids=tuple(p.id for p in absorbed),
        group_weight=group_weight,
        threshold=threshold,
    )


def lateral_collapse(
    points: List[Point],
    ledger: LateralLedger,
    encoder: Any,
    t_now: float,
    *,
    min_nodes: int = _MIN_NODES,
    require_gate: bool = True,
) -> LateralReport:
    """One pass of co-retrieval-driven lateral collision. Mutates `points`
    in place : each redundant group collapses into its keeper, the other
    members are removed, and absorbed->keeper aliases are recorded so
    downstream id resolution still works. No-op below the gate."""
    report = LateralReport()
    groups = detect_lateral_groups(
        ledger, points, min_nodes=min_nodes, require_gate=require_gate,
    )
    if not groups:
        return report
    thr = lateral_threshold(ledger) or 0.0
    by_id = {p.id: p for p in points}
    removed: set = set()
    for gids in groups:
        members = [by_id[i] for i in gids if i in by_id and i not in removed]
        if len(members) < 2:
            continue
        keeper = _keeper_of(members)
        absorbed = [p for p in members if p.id != keeper.id]
        gweight = sum(
            ledger.weight.get(_pair_key(keeper.id, p.id), 0.0) for p in absorbed
        )
        ev = resolve_lateral(
            keeper, absorbed, encoder, t_now,
            group_weight=gweight, threshold=thr,
        )
        report.collided.append(ev)
        for p in absorbed:
            report.aliases[p.id] = keeper.id
            removed.add(p.id)
    if removed:
        points[:] = [p for p in points if p.id not in removed]
    return report
