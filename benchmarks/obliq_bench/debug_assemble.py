"""
Per-gold autopsy of the assemble loop — WHY is each gold missed, which tool fixes it.

Instead of guessing, trace every gold's fate :
  * is it SURFACEABLE by each surface mode (sim / fuzzy / regex over content+tags)?
    at what RANK (would the top-k budget catch it)?
  * its EMBEDDING cosine to the query (would a semantic search mode catch it)?
  * after the real assemble loop, is it in the bag (HIT) or missed?

Then bucket the misses so the optimisation w.r.t. our tools is explicit :
  REJECTED        surfaced in top-k but Chain-of-Note dropped it   -> judge too strict
  BEYOND_K        surfaceable but ranked past k                    -> raise k
  NEEDS_SEMANTIC  no surface match, but high embedding cosine      -> add sim=embedding
  OBLIQUE_TAIL    no surface match AND low embedding cosine         -> genuinely latent

Usage: python -m benchmarks.obliq_bench.debug_assemble --query-id q0459 --bg 30
"""

from __future__ import annotations

import argparse
import math

from benchmarks.obliq_bench.event_step import build
from benchmarks.obliq_bench.batch_walk_vs_event import _docs


def _cos(a, b):
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


def _ranks(mem, query, gold, mode, k_full):
    """Full ranking under `mode` ; return {gold_id: rank} for gold that score>0."""
    ranked = mem.search_nodes(query, mode=mode, on="both", k=k_full)
    pos = {p.id: i for i, p in enumerate(ranked)}
    return {g: pos[g] for g in gold if g in pos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="descriptive/twitter")
    ap.add_argument("--query-id", default="q0459")
    ap.add_argument("--bg", type=int, default=30)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--max-rounds", type=int, default=14)
    args = ap.parse_args()

    mem, question, gold = build(args.track, args.query_id, args.bg)
    mem.consolidate_events(use_llm=False)
    gold = list(gold)
    gset = set(gold)
    pool_ids = {p.id for p in mem.points}
    n_full = len(mem.points)

    # surface-mode ranks + embedding cosine per gold
    ranks = {m: _ranks(mem, question, gold, m, n_full)
             for m in ("sim", "fuzzy", "regex")}
    qv = mem.encoder.encode(question)
    by_id = {p.id: p for p in mem.points}
    cos = {}
    for g in gold:
        p = by_id.get(g)
        cos[g] = _cos(qv, mem.encoder.encode(p.content)) if p else 0.0

    # run the real assemble loop (global, CoN judge)
    rep = mem.assemble_set(question, k=args.k, max_rounds=args.max_rounds,
                           bag="autopsy", auto_route=False)
    bag = _docs({i for i, _ in mem.bag_items(bag="autopsy")})

    print("\nQUERY [%s] gold=%d   assemble bag recall=%d/%d  rounds=%d %s" % (
        args.query_id, len(gold), len(bag & gset), len(gold),
        rep["rounds"], rep["added"]))
    print("  Q: %s\n" % question[:100])
    print("  %-10s %4s | %-22s | %5s | %s" % (
        "gold", "in?", "best surface rank (mode)", "emb", "verdict"))
    print("  " + "-" * 74)

    buckets = {"HIT": [], "REJECTED": [], "BEYOND_K": [],
               "NEEDS_SEMANTIC": [], "OBLIQUE_TAIL": []}
    for g in gold:
        best_rank, best_mode = None, "-"
        for m in ("sim", "fuzzy", "regex"):
            if g in ranks[m] and (best_rank is None or ranks[m][g] < best_rank):
                best_rank, best_mode = ranks[m][g], m
        hit = g in bag
        if hit:
            verdict = "HIT"
        elif best_rank is None:
            verdict = "NEEDS_SEMANTIC" if cos[g] >= 0.30 else "OBLIQUE_TAIL"
        elif best_rank < args.k:
            verdict = "REJECTED"            # surfaced in budget, judge dropped it
        else:
            verdict = "BEYOND_K"
        buckets[verdict].append(g)
        rk = ("#%d (%s)" % (best_rank, best_mode)) if best_rank is not None \
            else "none (no surface match)"
        print("  %-10s %4s | %-22s | %.2f | %s" % (
            g[:10], "Y" if hit else "·", rk, cos[g], verdict))

    print("\n  SUMMARY")
    for k, v in buckets.items():
        print("    %-15s %2d   %s" % (k, len(v), ", ".join(x[:8] for x in v)))
    print("\n  -> levers : REJECTED=loosen judge ; BEYOND_K=raise k ; "
          "NEEDS_SEMANTIC=add embedding mode ; OBLIQUE_TAIL=intrinsic.")


if __name__ == "__main__":
    main()
