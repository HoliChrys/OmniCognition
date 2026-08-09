"""
Calibration probe for the ACT-R ranking levers (recency_weight / spreading_weight).

NOT a benchmark on a published dataset (none are vendored here). This is a
CONTROLLED probe : curated OBLIQUE (query, gold) pairs — the gold's surface text
is disjoint from the query, so plain cosine struggles — plus distractors, encoded
with the REAL SentenceTransformer model. A fair WARMUP builds journal history by
issuing each theme's DIRECT query (which the cosine does surface), logging access
+ co-retrieval. Then we sweep each lever and measure recall@k on the OBLIQUE
queries. It answers "does the mechanism help recall, and at what weight?" — the
gain is on THIS probe, not a claim about LoCoMo/OBLIQ.

    python -m benchmarks.calibrate_levers            # full sweep
    python -m benchmarks.calibrate_levers --k 10
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from metacog.memory import Memory
from benchmarks.locomo.encoders import SemanticEncoder


# (theme, direct_query, oblique_query, gold_doc, context_docs...)
# oblique_query shares MEANING but not WORDS with gold ; direct_query is the
# lexically-close phrasing used only to warm up history (as a real prior session
# would have surfaced the doc). context docs give co-retrieval anchors.
THEMES: List[dict] = [
    dict(theme="wealth",
         direct="signs that a person is wealthy or rich",
         oblique="how comfortable is his financial situation",
         gold="They own three vacation homes, a yacht, and fly private.",
         context=["Their household spends freely on luxury goods.",
                  "The family never worries about money at all."]),
    dict(theme="grief",
         direct="someone mourning the death of a loved one",
         oblique="why has she seemed so withdrawn and quiet lately",
         gold="She still sets a place at the table for her late husband.",
         context=["The funeral was held quietly last spring.",
                  "He passed away after a long illness."]),
    dict(theme="pregnancy",
         direct="signs that a woman is pregnant",
         oblique="what big life change is the couple preparing for",
         gold="She has been painting the spare room a soft yellow and buying tiny socks.",
         context=["The morning nausea finally started to fade.",
                  "They scheduled the twenty-week ultrasound."]),
    dict(theme="job_loss",
         direct="a person who lost their job or got laid off",
         oblique="why is he suddenly home every weekday",
         gold="His office badge no longer works and the desk is cleared out.",
         context=["The company announced a round of layoffs.",
                  "He updates his resume every morning now."]),
    dict(theme="new_relationship",
         direct="two people falling in love or dating",
         oblique="why does she keep smiling at her phone",
         gold="He leaves a rose on her windshield every Friday.",
         context=["They text each other good morning without fail.",
                  "Dinner reservations for two, again."]),
    dict(theme="illness",
         direct="a person who is seriously ill or sick",
         oblique="why did he cancel the hiking trip",
         gold="The chemotherapy leaves him exhausted by noon.",
         context=["Weekly appointments at the oncology ward.",
                  "He lost his appetite and a lot of weight."]),
    dict(theme="moving",
         direct="a family relocating to a new city",
         oblique="why are there boxes stacked in the hallway",
         gold="The lease on the new apartment across the country starts Monday.",
         context=["They hired movers for the long haul.",
                  "Forwarding address filed with the post office."]),
    dict(theme="retirement",
         direct="a person retiring from their career",
         oblique="why did they throw him a big party at the office",
         gold="After forty years he handed in his keys and cleared his calendar for good.",
         context=["A gold watch and a farewell speech.",
                  "He talks about finally sailing full-time."]),
    dict(theme="debt",
         direct="a person struggling with debt or money problems",
         oblique="why does he flinch when the phone rings",
         gold="Collection agencies call about the unpaid balances daily.",
         context=["The credit cards are all maxed out.",
                  "He skips meals to make the minimum payment."]),
    dict(theme="new_pet",
         direct="a family that just got a new dog or puppy",
         oblique="why is the backyard suddenly fenced",
         gold="Chewed slippers and a food bowl by the door greet everyone now.",
         context=["Vaccination records from the shelter.",
                  "Early morning walks around the block."]),
]


def _gold_rank(mem: Memory, th: dict, depth: int = 30) -> int:
    """Rank (0-based) of the theme's gold in the oblique retrieval, or depth if
    it falls outside the fetched window."""
    got = [h["id"] for h in mem.retrieve(th["oblique"], k=depth)]
    gid = f"gold::{th['theme']}"
    return got.index(gid) if gid in got else depth


def _recall_at_k(mem: Memory, k: int) -> float:
    return sum(_gold_rank(mem, th) < k for th in THEMES) / len(THEMES)


def build() -> Memory:
    from metacog.journal import Journal
    mem = Memory(encoder=SemanticEncoder(), journal=Journal())
    # corpus : gold + context docs for every theme (context of one theme is a
    # distractor for the others, so the corpus is genuinely competitive).
    for th in THEMES:
        mem.ingest(th["gold"], kind="FACT", id=f"gold::{th['theme']}")
        for i, c in enumerate(th["context"]):
            mem.ingest(c, kind="FACT", id=f"ctx::{th['theme']}::{i}")
    return mem


def warmup(mem: Memory, rounds: int = 3) -> None:
    """Fair history : issue each theme's DIRECT query (lexically close) so the
    journal accrues access history for the gold and co-retrieval between gold and
    its context — exactly what a prior session would have produced. The oblique
    test queries are NEVER used to warm up."""
    for _ in range(rounds):
        for th in THEMES:
            hits = [h["id"] for h in mem.retrieve(th["direct"], k=5)]
            # plain retrieve() does NOT log to the journal (only the walk does),
            # so record it explicitly — this is the access + co-retrieval history
            # the levers consume.
            mem.record_retrieval(hits, query_text=th["direct"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)
    k = args.k

    mem = build()
    warmup(mem)
    print(f"\ncorpus={len(mem.points)} docs, {len(THEMES)} oblique probes, recall@{k}\n")

    # per-theme baseline gold rank — shows where the levers have room to work
    print("  baseline gold ranks (oblique query, lower = better):")
    for th in THEMES:
        r = _gold_rank(mem, th)
        flag = "" if r < k else "  <- MISS"
        print(f"     {th['theme']:16s} rank {r}{flag}")
    print()

    base = _recall_at_k(mem, k)
    print(f"  baseline (both levers OFF)           recall@{k} = {base:.3f}")

    print("\n  -- need-odds (recency_weight) sweep --")
    for w in (0.0, 0.2, 0.4, 0.6, 0.8):
        mem.recency_weight = w
        print(f"     recency_weight={w:.1f}   recall@{k} = {_recall_at_k(mem, k):.3f}")
    mem.recency_weight = 0.0

    print("\n  -- spreading (spreading_weight) sweep --")
    for w in (0.0, 0.2, 0.4, 0.6, 0.8):
        mem.spreading_weight = w
        print(f"     spreading_weight={w:.1f} recall@{k} = {_recall_at_k(mem, k):.3f}")
    mem.spreading_weight = 0.0

    print("\n(Controlled probe with the real MiniLM encoder — mechanism efficacy,"
          "\n not a LoCoMo/OBLIQ score. On a NEUTRAL probe with roughly-uniform"
          "\n access history both levers tend to DEGRADE recall — empirically"
          "\n confirming the default-OFF choice. They target specific regimes"
          "\n (skewed access for need-odds ; genuine associative bridges for"
          "\n spreading) ; a real-dataset sweep is needed to find where they pay"
          "\n off. Note: need_odds.blend squashes the base cosine through a"
          "\n sigmoid, which compresses discriminative cosine gaps — a likely"
          "\n reason small weights swing recall so hard here.)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
