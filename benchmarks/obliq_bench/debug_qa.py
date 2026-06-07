"""
Step-by-step QA debugger for OBLIQ-Bench (arXiv 2605.06235).

OBLIQ-Bench is an IR benchmark of OBLIQUE queries — the attributes that decide
relevance are LATENT, with little or no surface expression in the document
("tweets that implicitly link energy costs to overseas conflict", "chat logs
that demonstrate a failure mode", "an obscure passage from a lossy
recollection"). It exposes a retrieval/verification asymmetry: a reasoning LLM
recognises latent relevance once a document is surfaced, but retrievers fail to
surface the relevant document in the first place. That is exactly the regime
MetaCog-Mem's answer-space (clue_search), anchor and tag machinery target — so
we debug it with the SAME REPL as the LoCoMo path.

This driver reuses `benchmarks.locomo.debug_qa`'s `QADebugger` REPL verbatim
(presearch / clues / walk / auto / recall / step / …) ; it only differs in how
the memory is built : PER-QUESTION, indexing just one query's RELEVANT docs
(from qrels) plus `--bg` random corpus distractors — NOT the 72k-doc corpus —
so a single oblique query is a small, fast, honest ranking test. The query text
becomes the question and the relevant corpus ids become the gold, so the REPL's
recall / agent_recall tracking works unchanged.

Usage
-----
  # build (per-question) + step-by-step
  python -m benchmarks.obliq_bench.debug_qa --track descriptive/twitter \
      --query-id q0000 --bg 200 --script "gold; clues; recall"

  # let the full agent run and read agent_recall
  python -m benchmarks.obliq_bench.debug_qa --track descriptive/twitter \
      --query-id q0008 --bg 300 --script "auto; recall"

Tracks: descriptive/twitter, descriptive/wildchat, analogues/math,
analogues/writing, tip-of-tongue/congress.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from collections import defaultdict
from typing import List, Tuple

# Reuse the LoCoMo REPL verbatim — the debugger is benchmark-agnostic once a
# memory + question + gold ids are provided.
from benchmarks.locomo.debug_qa import QADebugger, _run

_REPO = "dianetc/OBLIQ-Bench"


def _dl(track: str, sub: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(_REPO, f"{track}/{sub}", repo_type="dataset")


def _load_queries(track: str) -> dict:
    out = {}
    for ln in open(_dl(track, "queries+qrels/queries.jsonl")):
        o = json.loads(ln)
        out[o["_id"]] = o["text"]
    return out


def _load_qrels(track: str) -> dict:
    """query-id -> set(relevant corpus-id) (graded score > 0)."""
    qrels: dict = defaultdict(set)
    for ln in open(_dl(track, "queries+qrels/qrels.tsv")):
        p = ln.rstrip().split("\t")
        if (p[0].startswith("q") and len(p) >= 3
                and p[2].lstrip("-").isdigit() and int(p[2]) > 0):
            qrels[p[0]].add(p[1])
    return qrels


def build_obliq_memory(track: str, query_id: str, bg: int, *,
                       seed: int = 0, rebuild: bool = False
                       ) -> Tuple[object, str, List[str]]:
    """Build a PER-QUESTION memory : the query's relevant docs + `bg` random
    corpus distractors. Returns (memory, question_text, gold_ids). Cached."""
    from metacog.memory import Memory
    from metacog.llm import ClaudeLLM
    from benchmarks.locomo.encoders import SemanticEncoder

    cache = f"/tmp/obliq_{track.replace('/', '_')}_{query_id}_{bg}.pkl"
    enc, llm = SemanticEncoder(), ClaudeLLM()

    queries = _load_queries(track)
    if query_id not in queries:
        query_id = sorted(queries)[0]
    question = queries[query_id]
    gold = sorted(_load_qrels(track)[query_id])

    if os.path.exists(cache) and not rebuild:
        mem = Memory(encoder=enc, llm=llm, storage_path=cache)
        mem.storage_path = None
        print(f"[cache] loaded {len(mem.points)} docs from {cache}")
        return mem, question, gold

    corpus = {}
    for ln in open(_dl(track, "corpus/corpus.jsonl")):
        o = json.loads(ln)
        corpus[str(o["_id"])] = (o.get("text") or o.get("title") or "")
    rel = [g for g in gold if g in corpus]
    pool = [c for c in corpus if c not in set(rel)]
    random.Random(seed).shuffle(pool)
    keep = rel + pool[:max(0, bg)]

    mem = Memory(encoder=enc, llm=llm)
    for cid in keep:
        mem.ingest(corpus[cid][:500], kind="FACT", id=cid)
    mem.storage_path = cache
    mem.save()
    mem.storage_path = None
    print(f"[build] {query_id} : {len(rel)} relevant + {len(keep) - len(rel)} "
          f"distractors = {len(keep)} docs (cached {cache})")
    return mem, question, gold


def main() -> None:
    ap = argparse.ArgumentParser(description="Step-by-step OBLIQ-Bench debugger")
    ap.add_argument("--track", default="descriptive/twitter",
                    help="descriptive/twitter | descriptive/wildchat | "
                         "analogues/math | analogues/writing | "
                         "tip-of-tongue/congress")
    ap.add_argument("--query-id", default="q0000")
    ap.add_argument("--bg", type=int, default=300,
                    help="number of random corpus distractors to index")
    ap.add_argument("--script", help="run ';'-separated commands then exit")
    ap.add_argument("--script-file")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    mem, question, gold = build_obliq_memory(
        args.track, args.query_id, args.bg, rebuild=args.rebuild)
    print(f"\n  OBLIQ query [{args.query_id}]: {question}")
    print(f"  gold (relevant ids): {len(gold)}")

    script = args.script
    if args.script_file:
        with open(args.script_file) as fh:
            script = fh.read()

    dbg = QADebugger(mem, question, gold, "", 3,
                     f"obliq:{args.track}:{args.query_id}")
    asyncio.run(_run(dbg, script))


if __name__ == "__main__":
    main()
