"""
Step-by-step EVENT/CONTEXT pipeline debugger for OBLIQ-Bench.

OBLIQ queries are OBLIQUE : the attribute that decides relevance is LATENT,
with no surface expression ("tweets implicitly tying energy costs to overseas
conflict" — never the word 'war'). `clue_search` / a plain retriever returns
0 because nothing on the surface matches. This harness exercises the EVENT
subsystem's answer to that, stage by stage, on ONE oblique query :

  1. BUILD   : index the query's gold tweets + `--bg` distractors, with an
               LLMEventExtractor so each tweet auto-SPAWNS its events (a tweet
               about a strike on a refinery spawns a 'war' hub it gravitates to).
  2. CONTEXT : `consolidate_events` merges the fragmented micro-hubs into the
               macro CONTEXT (= the event:type, e.g. 'war') — the multi-signal
               merge (shared entity / cluster-overlap / name-cosine / LLM).
  3. ROUTE   : `detect_event_type(query)` — does the OBLIQUE query route to the
               war context VIA THE CONTEXT CENTROID (the denoised latent theme),
               when neither the hub name nor any single tweet matches on surface?
  4. FILL    : `event_search(query)` — the schema's NECESSARY (core) slots are
               answered by the FILTERED scoped search with knowledge_base=True
               (Phase-1 scoped to the context, Phase-2 over the whole KB), the
               clues collected into the bag, the gap-fills absorbed into the KB.
  5. RECALL  : how much gold the EVENT channel (cluster ∪ context ∪ slot fills)
               surfaces — i.e. 'find the war tweets WITHOUT 0 clue'.

Usage
-----
  python -m benchmarks.obliq_bench.event_step --query-id q0419 --bg 40
  python -m benchmarks.obliq_bench.event_step --query-id q0395 --bg 80 --no-llm-merge
"""

from __future__ import annotations

import argparse
import time
from typing import List

from benchmarks.obliq_bench.debug_qa import (
    _load_queries, _load_qrels, _dl,
)


def _recall(found: set, gold: set) -> float:
    g = set(gold)
    return (len(found & g) / len(g)) if g else 0.0


def _walk(mem, question, n=6):
    """The NORMAL meta-cognitive walk's cumulative facts — the right sonde for
    the action-bridge (it re-anchors on the nearest ACTION each stage and
    spreads from it, which retrieve@k cannot see)."""
    from metacog.meta_walk import MetaWalker
    w = MetaWalker(question, mem, n_stages=n, commit=False)
    for _ in range(n):
        if w.step().done:
            break
    return set(w._fact_ids_cum)


def build(track: str, query_id: str, bg: int, *, seed: int = 0):
    """Per-question memory with EVENT auto-spawn (event_extractor set)."""
    import json
    import random
    from metacog.memory import Memory
    from metacog.llm import ClaudeLLM
    from metacog.event_extractor import LLMEventExtractor
    from benchmarks.locomo.encoders import SemanticEncoder

    queries, qrels = _load_queries(track), _load_qrels(track)
    if query_id not in queries:
        query_id = sorted(queries)[0]
    question = queries[query_id]
    gold = sorted(qrels[query_id])

    corpus = {}
    for ln in open(_dl(track, "corpus/corpus.jsonl")):
        o = json.loads(ln)
        corpus[str(o["_id"])] = (o.get("text") or o.get("title") or "")
    rel = [g for g in gold if g in corpus]
    pool = [c for c in corpus if c not in set(rel)]
    random.Random(seed).shuffle(pool)
    keep = rel + pool[:max(0, bg)]

    llm = ClaudeLLM()
    mem = Memory(encoder=SemanticEncoder(), llm=llm,
                 event_extractor=LLMEventExtractor(llm))
    t0 = time.time()
    for cid in keep:
        mem.ingest(corpus[cid][:500], kind="FACT", id=cid)
    print(f"[build] {query_id}: {len(rel)} gold + {len(keep)-len(rel)} distractors"
          f" = {len(keep)} docs, events auto-spawned in {time.time()-t0:.1f}s")
    return mem, question, gold


def main() -> None:
    ap = argparse.ArgumentParser(description="OBLIQ event-pipeline step debugger")
    ap.add_argument("--track", default="descriptive/twitter")
    ap.add_argument("--query-id", default="q0419")
    ap.add_argument("--bg", type=int, default=40)
    ap.add_argument("--no-llm-merge", action="store_true",
                    help="skip LLM arbitration in consolidate_events (faster)")
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()

    from metacog.epistemic import PointKind
    mem, question, gold = build(args.track, args.query_id, args.bg)
    gold_set = set(gold)
    print(f"\n  QUERY [{args.query_id}]: {question}")
    print(f"  gold (relevant ids): {len(gold)}\n")

    # ---- baseline : plain retrieval (the '0 clue' symptom) -----------------
    base = mem.retrieve(question, k=max(10, len(gold)))
    base_ids = {h["id"] for h in base}
    print("  [1] BASELINE retrieve@%d   recall=%.3f  (hits %d/%d)" % (
        len(base), _recall(base_ids, gold_set),
        len(base_ids & gold_set), len(gold)))

    # ---- baseline WALK on the CLEAN manifold (before any event processing) --
    walk_base = _walk(mem, question)
    print("  [1b] BASELINE WALK         recall=%.3f  (hits %d/%d)" % (
        _recall(walk_base, gold_set), len(walk_base & gold_set), len(gold)))

    # ---- events spawned ----------------------------------------------------
    hubs = [p for p in mem.points if p.kind is PointKind.EVENT]
    from collections import Counter
    types = Counter(t.split(":", 2)[2] for h in hubs for t in h.tags
                    if t.startswith("event:type:"))
    print("  [2] EVENTS spawned: %d hubs, types=%s" % (
        len(hubs), dict(types.most_common(8))))

    # ---- consolidate into CONTEXTS ----------------------------------------
    rep = mem.consolidate_events(use_llm=not args.no_llm_merge)
    print("  [3] CONSOLIDATE -> %s" % rep)
    hubs = [p for p in mem.points if p.kind is PointKind.EVENT]
    # macro hubs = those not absorbed under another event:in
    ev_ids = {h.id for h in hubs}
    macros = [h for h in hubs if not any(
        t.startswith("event:in:") and t.split(":", 2)[2] in ev_ids
        and t.split(":", 2)[2] != h.id for t in h.tags)]
    macros.sort(key=lambda h: len(mem.event_cluster(h.id)), reverse=True)
    for h in macros[:6]:
        cl = mem.event_cluster(h.id)
        et = next((t.split(":", 2)[2] for t in h.tags
                   if t.startswith("event:type:")), "")
        print("        macro [%s] %-28s cluster=%-3d goldᶜ=%d" % (
            et, (h.content[:28]), len(cl),
            len({p.id for p in cl} & gold_set)))

    # ---- ROUTE via centroid ------------------------------------------------
    det = mem.detect_event_type(question, threshold=args.threshold)
    print("  [4] ROUTE detect_event_type -> %s" % det)
    if det:
        ctx = mem.context_members(det["etype"])
        ctx_ids = {p.id for p in ctx}
        print("        CONTEXT '%s': %d facts, recall=%.3f (gold %d/%d)" % (
            det["etype"], len(ctx_ids), _recall(ctx_ids, gold_set),
            len(ctx_ids & gold_set), len(gold)))

    # ---- FILL : schema slots, scoped+KB, bag (ACTION BRIDGE OFF for now) ----
    # We ablate the bridge cleanly : run event_search WITHOUT the beacon, walk
    # (walk_mid = event_search's general perturbation), THEN plant the beacon and
    # walk again (walk_post). Δ(post-mid) isolates the action bridge's marginal
    # effect on the NORMAL walk — the right sonde, not retrieve@k.
    from metacog.event_schema import event_search, event_action_enrich
    t0 = time.time()
    fill = event_search(mem, question, k_per_slot=5, action_bridge=False)
    print("  [5] EVENT_SEARCH (%.1fs) ->" % (time.time() - t0))
    if not fill:
        print("        no event detected (channel silent)")
    else:
        cl_ids = set(fill.get("cluster_ids") or [])
        cx_ids = set(fill.get("context_ids") or [])
        fa_ids = set(fill.get("fact_ids") or [])
        chan = cl_ids | cx_ids | fa_ids
        print("        core slots : %s" % fill.get("core"))
        print("        gaps       : %s" % fill.get("gaps"))
        print("        cluster=%d context=%d slot_facts=%d  bag=%d" % (
            len(cl_ids), len(cx_ids), len(fa_ids),
            len(mem.bag_items()) if hasattr(mem, "bag_items") else 0))
        inter = fill.get("bag_intersection_cluster_schema") or []
        print("        bag ∩ (cluster,schema) = %d ids, goldᶜ=%d" % (
            len(inter), len(set(inter) & gold_set)))
        print("        EVENT CHANNEL recall=%.3f (gold %d/%d)" % (
            _recall(chan, gold_set), len(chan & gold_set), len(gold)))

        # ---- ADDITIVE union with the baseline retrieval -------------------
        union = base_ids | chan
        print("  [6] ADDITIVE (baseline ∪ event) recall=%.3f (gold %d/%d)" % (
            _recall(union, gold_set), len(union & gold_set), len(gold)))

        # ---- VALIDATE THE BRIDGE VIA THE WALK (ablation) ------------------
        bw = len(walk_base & gold_set)
        walk_mid = _walk(mem, question)
        mw = len(walk_mid & gold_set)
        print("  [7] WALK after event_search (no beacon)  recall=%.3f (%d/%d)"
              "  Δvs[1b]=%+d" % (
                  _recall(walk_mid, gold_set), mw, len(gold), mw - bw))
        hub_id = (fill.get("event") or {}).get("event_id")
        aid = event_action_enrich(mem, hub_id, fill) if hub_id else None
        walk_post = _walk(mem, question)
        pw = len(walk_post & gold_set)
        print("  [8] WALK after ACTION BRIDGE (id=%s)     recall=%.3f (%d/%d)"
              "  Δvs[7]=%+d  (marginal bridge effect)" % (
                  aid, _recall(walk_post, gold_set), pw, len(gold), pw - mw))


if __name__ == "__main__":
    main()
