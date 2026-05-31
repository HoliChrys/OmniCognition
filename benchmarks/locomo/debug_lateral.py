"""Measure the effect of LATERAL collision on kNN redundancy.

Targeted, fast (no big bench). Indexes one conversation, fills the
co-retrieval ledger with a fixed bank of DIVERSE probe queries, then
compares retrieval BEFORE vs AFTER a lateral-enabled sleep on :

  - redundancy : mean pairwise content-cosine within each probe's top-k
    (higher = more near-duplicate slots wasted) ;
  - distinct-coverage : how many DISTINCT keepers fill the top-k ;
  - gold recall : the Caroline cat3 evidence must still be reachable
    (directly or via its keeper alias) — collapse must not lose it.

Run : DEBUG_NSESS=0 python -m benchmarks.locomo.debug_lateral
"""
from __future__ import annotations

import itertools
from typing import List, Tuple

from benchmarks.locomo.debug_walk import build_memory, PROBE_GOLD, PROBE_Q
from metacog.epistemic import PointKind
from metacog.geometry import cosine, retrieve_hybrid

# A diverse probe bank spanning the conversation's topics. Deliberately
# varied so the diversity-weighted ledger accumulates real signal.
PROBES: List[str] = [
    "Caroline LGBTQ support group transgender identity",
    "Melanie charity race running awareness mental health",
    "Caroline counseling mental health career field study",
    "Melanie adoption agency family kids inclusivity",
    "painting art sunrise colors hobby canvas",
    "camping outdoors summer trip nature lake",
    "birthday celebration daughter family event gift",
    "necklace grandma heirloom symbol memory",
    "volunteer shelter community helping others",
    "books reading children library stories",
    "pottery craft making hands creative",
    "travel cities visited places trip",
]


def _retrieve(mem, query: str, k: int):
    res = retrieve_hybrid(
        query, mem.points, k, mem._now(),
        encoder=mem.encoder, extractor=mem.extractor,
        use_lineage=True, use_spreading=True, use_fuzzy=True,
        restrict_kind=PointKind.FACT,
    )
    return [p for _s, p in res]


def _phase(mem, k: int):
    """One retrieval pass over the whole probe bank. Returns
    (per-probe top-k point lists, mean redundancy, mean distinct count,
    union of all retrieved ids)."""
    topk = {q: _retrieve(mem, q, k) for q in PROBES}
    reds, dists = [], []
    union: set = set()
    for q, pts in topk.items():
        embs = [p.embedding_orig for p in pts]
        pairs = list(itertools.combinations(range(len(embs)), 2))
        reds.append(
            sum(cosine(embs[i], embs[j]) for i, j in pairs) / len(pairs)
            if pairs else 0.0
        )
        dists.append(len({p.id for p in pts}))
        union.update(p.id for p in pts)
    return topk, sum(reds) / len(reds), sum(dists) / len(dists), union


def main() -> None:
    K = 8
    mem = build_memory(verbose=True)
    mem.lateral_enabled = True

    print("\n[before] filling co-retrieval ledger with diverse probes…")
    topk_before, red_before, cov_before, union_before = _phase(mem, K)
    for q, pts in topk_before.items():
        mem.record_retrieval([p.id for p in pts], tuple(mem.encoder.encode(q)))
    gold_before = {g: (g in union_before) for g in PROBE_GOLD}
    n_before = len(mem.points)

    print("\n[sleep] running lateral-enabled consolidation…")
    rep = mem.sleep()
    print(f"  lateral groups collided : {rep['lateral_collided_groups']}")
    print(f"  aliases : {rep['lateral_aliases']}")
    print(f"  points  : {n_before} -> {len(mem.points)}")

    _topk_after, red_after, cov_after, union_after = _phase(mem, K)
    gold_after = {
        g: (g in union_after or mem.resolve_alias(g) in union_after)
        for g in PROBE_GOLD
    }

    print("\n================ RESULT ================")
    print(f"top-k = {K}   probes = {len(PROBES)}")
    print(f"redundancy (mean pairwise cos in top-k) : "
          f"{red_before:.4f} -> {red_after:.4f}  "
          f"(Δ {red_after - red_before:+.4f})")
    print(f"distinct coverage (mean #distinct/top-k): "
          f"{cov_before:.2f} -> {cov_after:.2f}  "
          f"(Δ {cov_after - cov_before:+.2f})")
    print(f"gold reachable before : {gold_before}")
    print(f"gold reachable after  : {gold_after}")
    lost = [g for g in PROBE_GOLD if gold_before.get(g) and not gold_after.get(g)]
    print(f"gold LOST by collapse : {lost or 'none'}")
    print(f"probe Q : {PROBE_Q}")


if __name__ == "__main__":
    main()
