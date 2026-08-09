"""
Validate the agentic assemble_set loop on an OBLIQ query.

The cluster ceiling caps recall when the event subsystem fragments the gold
across micro-events (q0459: context = 13/23). The agentic loop searches the
FILTERED POOL by content+tags (sim/fuzzy/regex) with an LLM judge, drawing
WITHOUT REPLACEMENT — so it can recover the scattered gold the cluster missed.

This compares, with the REAL LLM judge:
  context             detect_event_type -> cluster ∪ context (the prior ceiling)
  assemble (cluster)  assemble_set auto-routed to the event cluster
  assemble (global)   assemble_set over the whole pool (auto_route=False)

Usage: python -m benchmarks.obliq_bench.validate_assemble --query-id q0459 --bg 30
"""

from __future__ import annotations

import argparse
import time

from benchmarks.obliq_bench.event_step import build, _recall
from benchmarks.obliq_bench.batch_walk_vs_event import _docs


def _bag_recall(mem, bag, gold):
    ids = _docs({i for i, _ in mem.bag_items(bag=bag)})
    g = set(gold)
    return _recall(ids, g), len(ids & g), len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="descriptive/twitter")
    ap.add_argument("--query-id", default="q0459")
    ap.add_argument("--bg", type=int, default=30)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--max-rounds", type=int, default=6)
    args = ap.parse_args()

    mem, question, gold = build(args.track, args.query_id, args.bg)
    mem.consolidate_events(use_llm=False)
    g = set(gold)
    print("\nQUERY [%s] gold=%d" % (args.query_id, len(gold)))
    print("  Q: %s" % question[:110])

    # ---- context ceiling (cluster ∪ context) ------------------------------
    det = mem.detect_event_type(question, threshold=0.30)
    ctx = set()
    if det:
        if det.get("event_id"):
            ctx |= {p.id for p in mem.event_cluster(det["event_id"])}
        try:
            ctx |= {p.id for p in mem.context_members(det["etype"])}
        except Exception:
            pass
    ctx = _docs(ctx)
    print("  [context]          recall=%.3f (%d/%d, n=%d)  route=%s" % (
        _recall(ctx, g), len(ctx & g), len(gold), len(ctx),
        (det or {}).get("etype")))

    # ---- assemble (cluster-scoped) ----------------------------------------
    t0 = time.time()
    rep1 = mem.assemble_set(question, k=args.k, max_rounds=args.max_rounds,
                            bag="asm_cluster")
    r, h, n = _bag_recall(mem, "asm_cluster", gold)
    print("  [assemble cluster] recall=%.3f (%d/%d, n=%d)  rounds=%d %s (%.0fs)" % (
        r, h, len(gold), n, rep1["rounds"], rep1["added"], time.time() - t0))

    # ---- assemble (global pool) -------------------------------------------
    t0 = time.time()
    rep2 = mem.assemble_set(question, k=args.k, max_rounds=args.max_rounds,
                            bag="asm_global", auto_route=False)
    r, h, n = _bag_recall(mem, "asm_global", gold)
    print("  [assemble global]  recall=%.3f (%d/%d, n=%d)  rounds=%d %s (%.0fs)" % (
        r, h, len(gold), n, rep2["rounds"], rep2["added"], time.time() - t0))


if __name__ == "__main__":
    main()
