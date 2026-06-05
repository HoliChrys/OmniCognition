"""
Association segmentation of retrieved evidence for answer synthesis.

When the final answer is composed from a FLAT list of retrieved facts,
unrelated facts reinforce each other spuriously: for "what fields would
Caroline pursue?" the evidence holds a tight counselling / mental-health /
psychology group AND an unrelated "art" turn, and a linear composition
lists them all as equally supported ("...Art").

We SEGMENT the evidence into association groups so mutual reinforcement is
computed WITHIN a group, never across unrelated ones. We do NOT pick a
single dominant cluster — a complex question may legitimately draw on
several groups; that judgement is the LLM's, which is simply shown the
structure. Clustering is SOFT: a fact about two related things may belong
to two groups.

Method (literature-grounded, parameter-free, pure-Python on the small
evidence set — no sklearn dependency):

  • plain single-linkage on raw cosine CHAINS — "a single noise point in the
    wrong place acts as a bridge between islands" (the counselling—…—art
    risk). We instead use HDBSCAN's MUTUAL REACHABILITY similarity
    (Campello et al. 2013): s_mr(i,j) = min(core_i, core_j, sim_ij), where
    core_i is the k-th-highest similarity of i (k = `min_cluster_size`).
    A semantically isolated fact (few close neighbours → low core) is
    pushed apart from everything → it forms its OWN group and never bridges
    or reinforces an unrelated one (native outlier handling).
  • OVERLAP is link-community style (Ahn, Bagrow & Lehmann, Nature 2010):
    a fact also joins a second group when its mean similarity to that
    group clears the threshold — its associations genuinely span two
    groups.
  • the edge threshold is EMERGENT (median + σ of the mutual-reachability
    similarities — the manifold's own resolution), no tunable knob.

The incremental ("au fur et à mesure") accumulation of co-occurrence lives
upstream in the lateral co-retrieval ledger (memory.record_retrieval, fed
every walk stage); this module reads the gathered state.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple


def _cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


def _emergent_threshold(scores: Sequence[float]) -> float:
    """median + σ over the pairwise similarities — the manifold's local
    resolution (same rule as communities._similarity_threshold)."""
    n = len(scores)
    if n == 0:
        return 0.0
    s = sorted(scores)
    median = s[n // 2]
    mean = sum(scores) / n
    sigma = math.sqrt(sum((x - mean) ** 2 for x in scores) / n)
    return median + sigma


def _natural_break_threshold(scores: Sequence[float]) -> float:
    """Cut at the LARGEST gap in the sorted similarities (Jenks-style natural
    break), restricted to positive cuts.

    A single median+σ threshold is fragile when the distribution is BIMODAL
    (a tight cluster ≈0.6 plus unrelated items ≈0): σ then lifts the cut
    ABOVE the cluster's own internal similarities and nothing connects. The
    natural break instead finds the empty band SEPARATING the dense cluster
    from the rest (e.g. the gap from ≈0 to ≈0.4) and cuts there — robust to
    bimodality, still parameter-free."""
    s = sorted(scores)
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return max(s[0], 1e-9)
    best_gap = -1.0
    cut = s[-1]
    for i in range(n - 1):
        gap = s[i + 1] - s[i]
        # only a cut whose UPPER side is a positive association is useful
        if gap > best_gap and s[i + 1] > 0.0:
            best_gap = gap
            cut = s[i + 1]
    return cut


def _pairwise(embeddings: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(embeddings)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = _cos(embeddings[i], embeddings[j])
            sim[i][j] = sim[j][i] = c
    return sim


def associate_clusters(
    embeddings: Sequence[Sequence[float]],
    *,
    method: str = "mutual_reachability",
    min_cluster_size: int = 2,
    threshold: Optional[float] = None,
) -> List[List[int]]:
    """Soft association clustering of `embeddings`. Returns overlapping
    groups of item indices, strongest (most mutually-reinforcing) first.

    `method`:
      • "mutual_reachability" (default) — HDBSCAN-style. Robust on the small
        / sparse evidence sets we work with: native outlier isolation, no
        single-linkage chaining.
      • "link_communities" — Ahn/Bagrow/Lehmann edge clustering. Native
        overlap on dense graphs; kept for comparison.
    Both are non-parametric (emergent median+σ thresholds)."""
    n = len(embeddings)
    if n <= 1:
        return [[0]] if n == 1 else []
    if method == "link_communities":
        return _link_community_clusters(embeddings, threshold=threshold)
    return _mutual_reachability_clusters(
        embeddings, min_cluster_size=min_cluster_size, threshold=threshold)


def _mutual_reachability_clusters(
    embeddings: Sequence[Sequence[float]],
    *,
    min_cluster_size: int = 2,
    threshold: Optional[float] = None,
) -> List[List[int]]:
    """HDBSCAN-style: mutual-reachability similarity + soft link overlap."""
    n = len(embeddings)
    if n <= 1:
        return [[0]] if n == 1 else []

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = _cos(embeddings[i], embeddings[j])
            sim[i][j] = sim[j][i] = c

    # core similarity : the k-th HIGHEST similarity of each node (k =
    # min_cluster_size). Low core ⇒ semantically sparse ⇒ outlier.
    k = max(1, min(min_cluster_size, n - 1))
    core = []
    for i in range(n):
        nbr = sorted((sim[i][j] for j in range(n) if j != i), reverse=True)
        core.append(nbr[k - 1])

    # mutual reachability similarity (HDBSCAN, in similarity space): pulls
    # isolated points down so they cannot bridge dense groups.
    smr = [[0.0] * n for _ in range(n)]
    off: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = min(core[i], core[j], sim[i][j])
            smr[i][j] = smr[j][i] = v
            off.append(v)
    # natural-break cut on the mutual-reachability values — robust to the
    # bimodal (tight cluster + outliers) shape a median+σ cut breaks on.
    thr = _natural_break_threshold(off) if threshold is None else threshold

    # base groups : connected components with mutual-reachability >= thr.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # an edge needs mutual-reachability at/above the emergent threshold AND
    # a strictly positive association (orthogonal / unrelated items, sim ≈ 0,
    # never connect — so they stay singletons instead of all collapsing when
    # the threshold degenerates to ~0).
    for i in range(n):
        for j in range(i + 1, n):
            if smr[i][j] >= thr and smr[i][j] > 1e-9:
                union(i, j)

    base: dict = {}
    for i in range(n):
        base.setdefault(find(i), []).append(i)
    groups = [sorted(m) for m in base.values()]

    # SOFT overlap (link-community spirit): a node also joins another group
    # when its MEAN raw similarity to that group clears the threshold — its
    # associations genuinely span both. Isolated outliers (low sim to every
    # group) never overlap, so they cannot reinforce an unrelated group.
    for members in groups:
        mset = set(members)
        for x in range(n):
            if x in mset:
                continue
            ms = sum(sim[x][m] for m in members) / len(members)
            if ms >= thr and ms > 1e-9:
                members.append(x)
        members.sort()
    # dedupe identical groups produced by overlap
    uniq: List[List[int]] = []
    seen_sets = set()
    for g in groups:
        key = tuple(sorted(set(g)))
        if key not in seen_sets:
            seen_sets.add(key)
            uniq.append(sorted(set(g)))
    groups = uniq

    def _strength(g: List[int]) -> Tuple[float, int]:
        if len(g) <= 1:
            return (0.0, len(g))
        s = sum(sim[g[a]][g[b]]
                for a in range(len(g)) for b in range(a + 1, len(g)))
        return (s, len(g))

    groups.sort(key=_strength, reverse=True)
    return groups


def _link_community_clusters(
    embeddings: Sequence[Sequence[float]],
    *,
    threshold: Optional[float] = None,
) -> List[List[int]]:
    """Ahn/Bagrow/Lehmann (Nature 2010) link communities: cluster the EDGES
    by neighbourhood overlap, then nodes inherit membership from their edges
    (overlap is native — a node whose edges fall in two edge-clusters joins
    both). Kept for comparison; on dense graphs it gives clean overlaps, on
    small/sparse evidence sets it leaves most nodes edgeless (returned as
    singletons here). Non-parametric (emergent thresholds)."""
    n = len(embeddings)
    sim = _pairwise(embeddings)
    off = [sim[i][j] for i in range(n) for j in range(i + 1, n)]
    thr = _emergent_threshold(off) if threshold is None else threshold

    nbr = {i: {i} for i in range(n)}
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] >= thr and sim[i][j] > 1e-9:
                edges.append((i, j))
                nbr[i].add(j)
                nbr[j].add(i)

    def link_sim(e1, e2) -> float:
        shared = set(e1) & set(e2)
        if not shared:
            return 0.0
        k = next(iter(shared))
        a = e1[0] if e1[1] == k else e1[1]
        b = e2[0] if e2[1] == k else e2[1]
        inter = len(nbr[a] & nbr[b])
        union = len(nbr[a] | nbr[b])
        return inter / union if union else 0.0

    # single-linkage on edges, emergent link threshold.
    ep = {e: e for e in edges}

    def find(x):
        while ep[x] != x:
            ep[x] = ep[ep[x]]
            x = ep[x]
        return x

    ls = [link_sim(e1, e2)
          for a, e1 in enumerate(edges) for e2 in edges[a + 1:]
          if set(e1) & set(e2)]
    lthr = _emergent_threshold(ls) if ls else 0.0
    for a, e1 in enumerate(edges):
        for e2 in edges[a + 1:]:
            if (set(e1) & set(e2)) and link_sim(e1, e2) >= lthr:
                r1, r2 = find(e1), find(e2)
                if r1 != r2:
                    ep[r2] = r1

    comm: dict = {}
    for e in edges:
        comm.setdefault(find(e), set()).update(e)
    groups = [sorted(c) for c in comm.values()]
    # nodes with no edge → their own singletons (comparable output).
    covered = set().union(*groups) if groups else set()
    for i in range(n):
        if i not in covered:
            groups.append([i])

    def _strength(g):
        if len(g) <= 1:
            return (0.0, len(g))
        s = sum(sim[g[a]][g[b]]
                for a in range(len(g)) for b in range(a + 1, len(g)))
        return (s, len(g))

    groups.sort(key=_strength, reverse=True)
    return groups


def group_evidence(
    items: Sequence[Any],
    embeddings: Sequence[Sequence[float]],
    *,
    method: str = "mutual_reachability",
    min_cluster_size: int = 2,
    threshold: Optional[float] = None,
) -> List[List[Any]]:
    """`associate_clusters` mapped back onto the caller's items, strongest
    group first (overlap possible)."""
    if len(items) != len(embeddings):
        raise ValueError("items and embeddings length mismatch")
    idx_groups = associate_clusters(
        embeddings, method=method,
        min_cluster_size=min_cluster_size, threshold=threshold)
    return [[items[i] for i in g] for g in idx_groups]
