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


def _ingest_debug(args) -> None:
    """INGESTION mode : trace the per-document indexing pipeline step by step —
    the document card, every spawned node (tone gloss / stance / doc2query
    questions / entities / event hubs), the tags added, and the text-index
    artifacts. Ingests only `--n-docs` gold docs so it is fast and legible."""
    import json
    import os
    from metacog.memory import Memory
    from metacog.llm import ClaudeLLM
    from metacog.event_extractor import LLMEventExtractor
    from metacog.doc_card import DocCardExtractor
    from benchmarks.locomo.encoders import SemanticEncoder
    from benchmarks.obliq_bench.debug_qa import _load_qrels, _dl

    gold = sorted(_load_qrels(args.track)[args.query_id])
    corpus = {}
    for ln in open(_dl(args.track, "corpus/corpus.jsonl")):
        o = json.loads(ln)
        corpus[str(o["_id"])] = (o.get("text") or o.get("title") or "")
    docs = [(g, corpus[g]) for g in gold if g in corpus][:args.n_docs]

    llm = ClaudeLLM()
    card = None if os.environ.get("OBLIQ_NO_CARD") else DocCardExtractor(llm)
    mem = Memory(encoder=SemanticEncoder(), llm=llm, doc_card_extractor=card,
                 event_extractor=LLMEventExtractor(llm), tone_reading=True)

    print("\nINGEST DEBUG [%s] — %d docs, card=%s\n" % (
        args.query_id, len(docs), "OFF" if card is None else "ON"))
    for did, text in docs:
        print("=" * 78)
        print("DOC [%s]\n  %s" % (did, text[:160].replace("\n", " ")))
        if card is not None:
            c = card.extract_card(text[:500])
            if c:
                print("  CARD : tone=%s | kw=%s | ents=%s | event=%s" % (
                    c.tone, c.keywords,
                    [(e.value, e.etype) for e in c.entities],
                    [(e.name, e.etype) for e in c.events]))
                print("       intended: %s" % (c.intended or "-"))
                print("       stance  : %s" % (c.stance or "-"))
                print("       doc2query: %s" % (c.questions or []))
            else:
                print("  CARD : <unparseable -> fallback to extractors>")
        before = {p.id for p in mem.points}
        mem.ingest(text[:500], kind="FACT", id=did)
        new = [p for p in mem.points if p.id not in before]
        fact = next(p for p in mem.points if p.id == did)

        def _kind(p):
            return getattr(p.kind, "name", str(p.kind))
        print("  SPAWNED %d nodes:" % len(new))
        for p in sorted(new, key=lambda x: x.id):
            if p.id == did:
                continue
            print("    %-22s [%-7s] %s" % (
                p.id[:22], _kind(p), (p.content or "")[:60].replace("\n", " ")))
        cats = [t for t in fact.tags if t.startswith(
            ("tone:", "event:in:", "event:type:", "time:date:"))]
        print("  DOC TAGS : %s" % (cats or "-"))
        ti = mem._text_index
        rt = sorted(ti.raw_tokens(fact))
        print("  INDEX : raw_tokens=%s" % rt[:12])
        print("          bm25_doc=%s" % ti.bm25_doc(fact)[:12])
    print("=" * 78)
    print("postings sample :", dict(list(mem._text_index.postings.items())[:6]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("retrieve", "ingest"),
                    default="retrieve",
                    help="retrieve: the assemble/recall autopsy ; "
                         "ingest: trace the per-doc indexing pipeline")
    ap.add_argument("--track", default="descriptive/twitter")
    ap.add_argument("--query-id", default="q0459")
    ap.add_argument("--bg", type=int, default=30)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--max-rounds", type=int, default=14)
    ap.add_argument("--n-docs", type=int, default=4,
                    help="ingest mode: how many gold docs to trace")
    args = ap.parse_args()

    if args.mode == "ingest":
        _ingest_debug(args)
        return

    mem, question, gold = build(args.track, args.query_id, args.bg)
    mem.consolidate_events(use_llm=False)
    gold = list(gold)
    gset = set(gold)
    pool_ids = {p.id for p in mem.points}
    n_full = len(mem.points)

    # context ceiling (detect_event_type -> cluster ∪ context) for comparison
    det = mem.detect_event_type(question, threshold=0.30)
    ctx = set()
    if det:
        if det.get("event_id"):
            ctx |= {p.id for p in mem.event_cluster(det["event_id"])}
        try:
            ctx |= {p.id for p in mem.context_members(det["etype"])}
        except Exception:
            pass
    ctx_recall = len(_docs(ctx) & gset)

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
    seen: set = set()
    bag_ids: list = []
    fate = {g: {"surf": None, "label": None} for g in gold}
    for r in range(args.max_rounds):
        surfaced = 0
        for mode in ("semantic", "sim", "fuzzy", "regex"):
            cand = mem.search_nodes(question, mode=mode, on="both",
                                    exclude_ids=seen, k=args.k)
            cand = [p for p in cand if p.id not in seen]
            if not cand:
                continue
            surfaced += len(cand)
            collected = [p for p in mem.points if p.id in set(bag_ids)]
            labels = mem.oblique_labels(question, cand, collected=collected)
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

    print("\nQUERY [%s] gold=%d   loop recall=%d/%d   (context %d/%d)" % (
        args.query_id, len(gold), len(bag & gset), len(gold),
        ctx_recall, len(gold)))
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
