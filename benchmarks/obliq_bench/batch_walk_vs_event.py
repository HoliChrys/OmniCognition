"""
BATCH : walk-alone vs walk+event on OBLIQ — recall AND precision (over-selection?)

The q0419 ablation showed the meta-cognitive WALK scores 15/15 recall on the
clean manifold. But recall alone is misleading : the walk's `fact_ids_cumulative`
is EVERYTHING it swept across all stages, so a high recall can just be OVER-
SELECTION (15/15 gold buried in 45 returned facts = precision 0.33 = noise). The
walk's real answer-set is the BOUNDED, relevance-ranked `_composable_evidence`.

So this batch reports, per oblique query and per channel, recall AND precision
AND set size :

  retrieve@k     the naive retriever
  WALK sweep     fact_ids_cumulative — the full reachable set (recall ceiling,
                 low precision : the over-selection the user warned about)
  WALK evid      _composable_evidence — the walk's actual bounded answer-set
                 (the honest recall/precision the answerer sees)
  context        the event channel = detect -> cluster ∪ context

If WALK-evid already dominates context on F1, the event branch is cost without
gain. If the WALK only reaches gold via the noisy sweep (high sweep-recall, low
evid-recall/precision) while the context is tighter, the event branch earns it.

Usage
-----
  python -m benchmarks.obliq_bench.batch_walk_vs_event \
      --query-ids q0419,q0433,q0469,q0407,q0424 --bg 20
"""

from __future__ import annotations

import argparse

from benchmarks.obliq_bench.event_step import build

# Ingestion auto-spawns DERIVED nodes (atoms, entities, event hubs) and the walk
# generates actions ; none are corpus documents, none can be gold. They must be
# excluded from the measured set or n exceeds the pool size and precision is
# meaningless. Gold are corpus doc ids ; recall is unaffected, but a clean n
# matters for the over-selection read.
_DERIVED = ("entity_", "atom_", "event_", "act_", "lateral_", "thought_", "gen_")


def _docs(ids):
    return {i for i in ids if not (str(i).startswith(_DERIVED) or "@" in str(i))}


def _scores(found, gold):
    # OBLIQ is RECALL-only ; precision/F1/n are shown for cross-benchmark context
    # but DO NOT count for OBLIQ. The set is restricted to document ids first so
    # n never exceeds the pool.
    f, g = _docs(found), set(gold)
    hit = len(f & g)
    rec = hit / len(g) if g else 0.0
    prec = hit / len(f) if f else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(n=len(f), hit=hit, rec=rec, prec=prec, f1=f1)


def _walk_sets(mem, question, n=12):
    """Return (sweep_ids, evidence_ids).

    The walk STOPS on its own UNCERTAINTY (`step().done` — the σ/GUM gate), not
    a fixed cap ; `n` is only a ceiling. The answer-set is `_relevant_cum` — the
    CoN-relevant facts, UNCAPPED (no _MAX_EVIDENCE truncation) : the evidence the
    uncertainty-governed walk actually committed to. `sweep` (fact_ids_cum) is
    every fact touched, kept only as the over-selection diagnostic."""
    from metacog.meta_walk import MetaWalker
    w = MetaWalker(question, mem, n_stages=n, commit=False)
    for _ in range(n):
        if w.step().done:                         # uncertainty-driven stop
            break
    sweep = set(w._fact_ids_cum)
    relevant = {p.id for p in getattr(w, "_relevant_cum", [])}
    return sweep, relevant


def _context_channel(mem, question, threshold):
    det = mem.detect_event_type(question, threshold=threshold)
    if not det:
        return set(), None
    ids = set()
    if det.get("event_id"):
        ids |= {p.id for p in mem.event_cluster(det["event_id"])}
    try:
        ids |= {p.id for p in mem.context_members(det["etype"])}
    except Exception:
        pass
    return ids, det


def main() -> None:
    ap = argparse.ArgumentParser(description="OBLIQ batch : recall+precision")
    ap.add_argument("--track", default="descriptive/twitter")
    ap.add_argument("--query-ids", default="q0419,q0433,q0469,q0407,q0424")
    ap.add_argument("--bg", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--n-stages", type=int, default=40,
                    help="generous CEILING only — uncertainty (done) is the "
                         "real terminator ; a subject can have arbitrarily many "
                         "relevant items, so the set size must NOT be hard-capped")
    args = ap.parse_args()

    qids = [q.strip() for q in args.query_ids.split(",") if q.strip()]
    rows = []
    # TWO MODES, reported side by side :
    #   * RECALL  -> "refer an exhaustive set" (find ALL the tweets ; OBLIQ).
    #   * F1/prec -> "generate a focused answer" (no noise ; LoCoMo-style).
    # They can coexist (an answer WITH the exhaustive list = the bag). n is the
    # set size : it is EMERGENT (uncertainty-stopped, uncapped), never a fixed k.
    hdr = ("%-7s %4s | %-17s | %-17s | %-17s | %-17s" % (
        "query", "gold", "retrieve@R", "WALK sweep", "WALK relev", "context"))
    sub = ("%-7s %4s | %5s%5s%4s%4s | %5s%5s%4s%4s | %5s%5s%4s%4s | "
           "%5s%5s%4s%4s" % (
               "", "", "rec", "f1", "pr", "n", "rec", "f1", "pr", "n",
               "rec", "f1", "pr", "n", "rec", "f1", "pr", "n"))
    print("\n" + hdr + "\n" + sub + "\n" + "-" * len(sub))
    for qid in qids:
        try:
            mem, question, gold = build(args.track, qid, args.bg)
        except Exception as e:
            print("%-7s  build error: %s" % (qid, e))
            continue
        mem.consolidate_events(use_llm=False)
        # retrieve@R : conventional recall@|gold| ceiling for the NAIVE retriever
        # (reference only — the walk/context determine their own size).
        retr = {h["id"] for h in mem.retrieve(question, k=max(10, len(gold)))}
        sweep, evid = _walk_sets(mem, question, n=args.n_stages)
        ctx, det = _context_channel(mem, question, args.threshold)
        s = {k: _scores(v, gold) for k, v in
             dict(retr=retr, sweep=sweep, evid=evid, ctx=ctx).items()}
        s["qid"], s["gold"] = qid, len(gold)
        rows.append(s)

        def cell(d):
            return "%5.2f%5.2f%4.2f%4d" % (d["rec"], d["f1"], d["prec"], d["n"])
        print("%-7s %4d | %s | %s | %s | %s  route=%s" % (
            qid, len(gold), cell(s["retr"]), cell(s["sweep"]),
            cell(s["evid"]), cell(s["ctx"]), (det or {}).get("etype")))

    if rows:
        n = len(rows)
        def m(ch, k): return sum(r[ch][k] for r in rows) / n
        def cellm(ch):
            return "%5.2f%5.2f%4.2f%4.0f" % (
                m(ch, "rec"), m(ch, "f1"), m(ch, "prec"), m(ch, "n"))
        print("-" * len(sub))
        print("%-7s %4s | %s | %s | %s | %s" % (
            "MEAN", "", cellm("retr"), cellm("sweep"),
            cellm("evid"), cellm("ctx")))
        print("\nVERDICT (exhaustive/RECALL mode): WALK-relev %.3f (n=%.0f) vs "
              "context %.3f (n=%.0f) ; sweep %.3f (n=%.0f) = over-select ceiling."
              "\nVERDICT (answer/F1 mode):     WALK-relev %.3f vs context %.3f." % (
                  m("evid", "rec"), m("evid", "n"),
                  m("ctx", "rec"), m("ctx", "n"),
                  m("sweep", "rec"), m("sweep", "n"),
                  m("evid", "f1"), m("ctx", "f1")))


if __name__ == "__main__":
    main()
