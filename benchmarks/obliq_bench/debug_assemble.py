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

    # INSTRUMENTED loop : replicate assemble_set but record, per gold, the round
    # it was SURFACED to the judge and the CoN label it got — the actual decisions
    # (not inferred from rank). This distinguishes a k/search miss (never
    # surfaced) from a JUDGE miss (surfaced then labelled irrelevant).
    from metacog.meta_walk import chain_of_note
    seen: set = set()
    bag_ids: list = []
    fate = {g: {"surf": None, "label": None} for g in gold}
    for r in range(args.max_rounds):
        surfaced = 0
        for mode in ("sim", "fuzzy", "regex"):
            cand = mem.search_nodes(question, mode=mode, on="both",
                                    exclude_ids=seen, k=args.k)
            cand = [p for p in cand if p.id not in seen]
            if not cand:
                continue
            surfaced += len(cand)
            collected = [p for p in mem.points if p.id in set(bag_ids)]
            labels = chain_of_note(question, cand, mem.llm, collected=collected)
            for p, lab in zip(cand, labels):
                if p.id in fate and fate[p.id]["surf"] is None:
                    fate[p.id]["surf"] = (r, mode)
                    fate[p.id]["label"] = lab
                seen.add(p.id)
                if lab != "irrelevant":
                    bag_ids.append(p.id)
        if surfaced == 0:
            break
    bag = _docs(set(bag_ids))

    print("\nQUERY [%s] gold=%d   loop recall=%d/%d" % (
        args.query_id, len(gold), len(bag & gset), len(gold)))
    print("  Q: %s\n" % question[:100])
    print("  %-10s %4s | %-14s | %-11s | %5s | %s" % (
        "gold", "in?", "surfaced", "CoN label", "emb", "verdict"))
    print("  " + "-" * 72)

    buckets = {"HIT": [], "JUDGE_REJECTED": [], "NEVER_SURFACED": [],
               "NEEDS_SEMANTIC": [], "OBLIQUE_TAIL": []}
    for g in gold:
        hit = g in bag
        surf = fate[g]["surf"]
        lab = fate[g]["label"] or "-"
        if hit:
            verdict = "HIT"
        elif surf is not None:
            verdict = "JUDGE_REJECTED"           # surfaced to CoN, labelled irrelevant
        elif cos[g] >= 0.30:
            verdict = "NEEDS_SEMANTIC"           # never surfaced, but semantically close
        else:
            verdict = "OBLIQUE_TAIL"
        buckets[verdict].append(g)
        sf = ("r%d/%s" % (surf[0], surf[1])) if surf else "never"
        print("  %-10s %4s | %-14s | %-11s | %.2f | %s" % (
            g[:10], "Y" if hit else "·", sf, lab, cos[g], verdict))

    print("\n  SUMMARY")
    for k, v in buckets.items():
        print("    %-15s %2d   %s" % (k, len(v), ", ".join(x[:8] for x in v)))
    print("\n  -> levers : JUDGE_REJECTED=fix CoN on oblique relevance ; "
          "NEVER_SURFACED/NEEDS_SEMANTIC=add embedding search ; "
          "OBLIQUE_TAIL=intrinsic.")


if __name__ == "__main__":
    main()
