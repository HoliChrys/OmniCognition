"""The FROZEN query sets of the oracle — 50 per context (OO0-5, TAC2-1178).

    python -m benchmarks.storage_baseline.queries --freeze
        writes queries/<ctx>.json for the three built contexts
    python -m benchmarks.storage_baseline.queries --from-journal <copy of
        memory.pkl.journal.db> --ctx <ctx> --hand <hand.json> --out <file>
        a query set for a COPIED served context: the first 40 distinct
        `retrievals.query_text` of its journal (read-only, `mode=ro`) + 10
        hand-written ones

Built contexts (no retrieval history to draw from), deterministic from the
frozen corpus and `SEED`:

* 25 `window` — a 6-to-10-word window from the middle of a chunk the
  context holds (lexical + semantic overlap with one point);
* 10 `section` — the words of a section heading the context holds (topical,
  several points);
* 5 `held_out` — a window of `docs/research_event_schema_rag.md`, which no
  context holds (the gap / floor path);
* 10 `hand` — written for the context below, French and English, two of them
  off-topic (the right answer is nothing).

A query set is FROZEN: edited after a baseline = a new set, a new file name.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from typing import Any, Dict, List

from benchmarks.storage_baseline import corpus as _corpus

HERE = os.path.dirname(os.path.abspath(__file__))
QUERY_DIR = os.path.join(HERE, "queries")
SEED = 1146
VERSION = 1

#: Which corpus roles a context holds (what its `window` / `section` queries
#: are drawn from).
HELD = {
    "oracle.small": ["small"],
    "oracle.medium": ["medium"],
    "oracle.narrow": ["narrow_context", "narrow_account"],
}

HAND: Dict[str, List[str]] = {
    "oracle.small": [
        "how does the walk decide to stop",
        "what happens in the presearch round",
        "comment l'agent reformule-t-il la question de John",
        "which facts are collected into the bag",
        "pourquoi la recherche littérale échoue-t-elle",
        "what is the final answer about John",
        "étapes de la marche métacognitive",
        "deterministic embedding retrieval with no LLM cost",
        "recette du kouign-amann",
        "weather forecast for tomorrow in Brest",
    ],
    "oracle.medium": [
        "how do I install the MCP server",
        "what is the anti-laundering corollary",
        "comment lancer l'évaluation LoCoMo",
        "recall at 7 compared with the walk",
        "why are there no stored edges between points",
        "qu'est-ce que le plafond sigma de la marche",
        "hybrid retrieval with BM25 and reciprocal rank fusion",
        "how are observators used to route a query",
        "recette du kouign-amann",
        "le cours de l'action Total aujourd'hui",
    ],
    "oracle.narrow": [
        "what does test_context_isolation prove",
        "comment l'écriture du store reste-t-elle atomique",
        "what is the reranker relevance floor",
        "why must a test never need an API key",
        "quelle est la règle de l'index de champs OKF",
        "the D3 deep wiki bench protocol",
        "how is the event channel additive",
        "store corrupt raises CorruptStoreError",
        "recette du kouign-amann",
        "résultats du match de football hier soir",
    ],
}


def _window(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) <= 8:
        return " ".join(words)
    n = rng.randint(6, min(10, len(words)))
    mid = len(words) // 2
    start = max(0, min(len(words) - n, mid - n // 2 + rng.randint(-3, 3)))
    return " ".join(words[start:start + n])


def _chunks(c: Dict[str, Any], roles: List[str]) -> List[Dict[str, str]]:
    return [ch for r in roles for f in c["roles"][r] for ch in f["chunks"]]


def build_set(ctx: str, c: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(f"{SEED}:{ctx}")
    held = _chunks(c, HELD[ctx])
    out: List[Dict[str, str]] = []
    for ch in rng.sample(held, 25):
        out.append({"kind": "window", "text": _window(ch["text"], rng)})
    sections = sorted({ch["section"] for ch in held if ch["section"] != "top"})
    for sec in rng.sample(sections, min(10, len(sections))):
        out.append({"kind": "section", "text": sec.replace("-", " ")})
    for ch in rng.sample(_chunks(c, ["held_out"]), 5):
        out.append({"kind": "held_out", "text": _window(ch["text"], rng)})
    out += [{"kind": "hand", "text": t} for t in HAND[ctx]]
    seen: set = set()
    queries = []
    for q in out:
        if q["text"] in seen:
            continue
        seen.add(q["text"])
        queries.append({"qid": f"{ctx}#{len(queries):02d}", **q})
    return {"version": VERSION, "ctx": ctx, "seed": SEED,
            "corpus_commit": c["commit"], "queries": queries}


def from_journal(journal: str, ctx: str, hand: List[str], n: int = 40) -> Dict[str, Any]:
    uri = "file:" + os.path.abspath(journal) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT query_text FROM retrievals WHERE query_text <> '' "
            "GROUP BY query_text ORDER BY MIN(retrieval_id)").fetchall()
    finally:
        conn.close()
    picked = [r[0] for r in rows][:n]
    queries = [{"qid": f"{ctx}#{i:02d}", "kind": "journal", "text": t}
               for i, t in enumerate(picked)]
    queries += [{"qid": f"{ctx}#{len(picked) + i:02d}", "kind": "hand", "text": t}
                for i, t in enumerate(hand)]
    return {"version": VERSION, "ctx": ctx, "source": "journal",
            "journal_rows": len(rows), "queries": queries}


def path_for(ctx: str) -> str:
    return os.path.join(QUERY_DIR, f"{ctx}.v{VERSION}.json")


def load(ctx: str) -> Dict[str, Any]:
    with open(path_for(ctx), encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="freeze the oracle query sets")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--from-journal")
    ap.add_argument("--ctx")
    ap.add_argument("--hand", help="JSON list of hand-written queries")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.from_journal:
        if not (a.ctx and a.out):
            ap.error("--from-journal needs --ctx and --out")
        hand = json.load(open(a.hand, encoding="utf-8")) if a.hand else []
        qs = from_journal(a.from_journal, a.ctx, hand)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(qs, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"{a.ctx}: {len(qs['queries'])} queries "
              f"({qs['journal_rows']} distinct in the journal) -> {a.out}")
        return 0
    if not a.freeze:
        ap.error("--freeze or --from-journal")
    c = _corpus.load()
    os.makedirs(QUERY_DIR, exist_ok=True)
    for ctx in HELD:
        p = path_for(ctx)
        if os.path.exists(p):
            raise SystemExit(f"{p} exists: a frozen set is never rewritten "
                             "(bump VERSION for a new set)")
        qs = build_set(ctx, c)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(qs, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"{ctx}: {len(qs['queries'])} queries -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
