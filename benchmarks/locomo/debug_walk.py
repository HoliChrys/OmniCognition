"""Targeted, fast, step-by-step debugger for the meta-cognitive walk.

NOT a benchmark. Indexes a TINY slice (a couple of sessions of one
conversation) so the whole thing runs in seconds, then lets us inspect,
one stage at a time, exactly how indexation and the fact/thought/action
walk behave — the way you'd step through a debugger.

Usage :
    python -m benchmarks.locomo.debug_walk index      # ingest + dump points
    python -m benchmarks.locomo.debug_walk retrieve    # one-shot retrieval
    python -m benchmarks.locomo.debug_walk walk        # walk, stage by stage
    python -m benchmarks.locomo.debug_walk all         # everything

Slice + question are chosen so the gold evidence lives inside the slice,
so a MISS is a real retrieval bug, never "the turn wasn't indexed".
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.locomo.eval import ingest_conversation
from benchmarks.locomo.encoders import SemanticEncoder
from metacog.memory import Memory

DATA = "benchmarks/locomo/data/locomo10.json"
SAMPLE = os.environ.get("DEBUG_SAMPLE", "conv-26")
# Slice size : DEBUG_NSESS=0 (or unset large) ingests the WHOLE conversation
# (~400 turns) to reproduce real-bench competition ; small N for fast probes.
N_SESSIONS = int(os.environ.get("DEBUG_NSESS", "2"))

# Targeted probes. Pick via DEBUG_PROBE=caroline | john (default caroline).
# Both are cat3 INFERENCE questions whose evidence is lexically disjoint
# from the question — the exact failure mode we are debugging.
_PROBES = {
    "caroline": {
        "sample": "conv-26",
        "question": "What fields would Caroline be likely to pursue in her career?",
        "gold": ["D1:9", "D1:11"],
        "answer": "Psychology, counseling certification",
        "category": 3,
    },
    "john": {
        # John's "kids have so much" wealth-implying turn.
        "sample": "conv-41",
        "question": "What might John's financial status be?",
        "gold": ["D5:5"],
        "answer": "Middle-class or wealthy",
        "category": 3,
    },
}
_PROBE_KEY = os.environ.get("DEBUG_PROBE", "caroline").lower()
_PROBE = _PROBES.get(_PROBE_KEY, _PROBES["caroline"])
SAMPLE = _PROBE["sample"] if "DEBUG_SAMPLE" not in os.environ else SAMPLE
PROBE_Q = _PROBE["question"]
PROBE_GOLD = list(_PROBE["gold"])
PROBE_ANSWER = _PROBE["answer"]
PROBE_CATEGORY = _PROBE["category"]


def _slice_conversation() -> Dict[str, Any]:
    """Build a conversation dict containing only the first N_SESSIONS."""
    data = json.load(open(DATA))
    conv = next(c for c in data if c.get("sample_id") == SAMPLE)
    C = conv["conversation"]
    out: Dict[str, Any] = {
        "speaker_a": C.get("speaker_a"),
        "speaker_b": C.get("speaker_b"),
    }
    hi = N_SESSIONS if N_SESSIONS > 0 else 999
    for i in range(1, hi + 1):
        for key in (f"session_{i}", f"session_{i}_date_time"):
            if key in C:
                out[key] = C[key]
    return out


def build_memory(*, verbose: bool = True) -> Memory:
    enc = SemanticEncoder()
    mem = Memory(encoder=enc)  # default SimpleKeywordExtractor : fast, no LLM
    conv = _slice_conversation()
    n = ingest_conversation(mem, conv, with_dates=True)
    if verbose:
        print(f"[index] ingested {n} turns -> {len(mem.points)} points")
    return mem


def _by_id(mem: Memory) -> Dict[str, Any]:
    return {p.id: p for p in mem.points}


# ----------------------------------------------------------------------
# STEP 1 — indexation coherence
# ----------------------------------------------------------------------
def cmd_index() -> Memory:
    print("=" * 70)
    print("STEP 1 — INDEXATION COHERENCE")
    print("=" * 70)
    mem = build_memory()

    kinds: Dict[str, int] = {}
    for p in mem.points:
        kinds[p.kind.name] = kinds.get(p.kind.name, 0) + 1
    print(f"[index] point kinds: {kinds}")

    print("\n[index] first 8 points (id | kind | keywords | content) :")
    for p in mem.points[:8]:
        print(f"  {p.id:8} {p.kind.name:8} kw={list(p.keywords or [])}")
        print(f"           {p.content[:78]}")

    print(f"\n[index] GOLD evidence turns for probe ({PROBE_GOLD}) :")
    bid = _by_id(mem)
    for gid in PROBE_GOLD:
        p = bid.get(gid)
        if p is None:
            print(f"  !! {gid} NOT INDEXED")
            continue
        print(f"  {gid}: kw={list(p.keywords or [])}")
        print(f"        content : {p.content}")
        print(f"        has_kw_emb={p.keywords_embedding is not None} "
              f"has_content_emb={p.embedding_orig is not None}")
    return mem


# ----------------------------------------------------------------------
# STEP 2 — one-shot retrieval (no walk) : is the gold reachable at all ?
# ----------------------------------------------------------------------
def cmd_retrieve(mem: Optional[Memory] = None) -> Memory:
    print("\n" + "=" * 70)
    print("STEP 2 — ONE-SHOT RETRIEVAL (does the gold rank at all?)")
    print("=" * 70)
    mem = mem or build_memory(verbose=False)
    from metacog.geometry import retrieve_hybrid
    from metacog.epistemic import PointKind

    print(f"[probe] Q: {PROBE_Q}")
    print(f"[probe] gold: {PROBE_GOLD}  answer: {PROBE_ANSWER}")

    # Production query-keyword view (what the seed embedding is built from).
    qkw = mem.extractor.extract(PROBE_Q, n=8)
    print(f"[probe] query keywords (SimpleKeywordExtractor): {qkw}")

    results = retrieve_hybrid(
        PROBE_Q, mem.points, 12, mem._now(),
        encoder=mem.encoder, extractor=mem.extractor,
        use_lineage=True, use_spreading=True, use_fuzzy=True,
        restrict_kind=PointKind.FACT,
    )
    print("\n[retrieve] top-12 (rank | score | id | content) :")
    gold_ranks = {}
    for rank, (s, p) in enumerate(results, 1):
        hit = " <== GOLD" if p.id in PROBE_GOLD else ""
        if p.id in PROBE_GOLD:
            gold_ranks[p.id] = rank
        print(f"  {rank:2} {s:6.3f} {p.id:8} {p.content[:60]}{hit}")
    print(f"\n[retrieve] gold ranks: {gold_ranks or 'ALL MISSED in top-12'}")
    return mem


# ----------------------------------------------------------------------
# STEP 3 — the walk, stage by stage (agent_recall pas à pas)
# ----------------------------------------------------------------------
def cmd_walk(mem: Optional[Memory] = None) -> Memory:
    print("\n" + "=" * 70)
    print("STEP 3 — WALK, STAGE BY STAGE")
    print("=" * 70)
    mem = mem or build_memory(verbose=False)
    from metacog.meta_walk import MetaWalker

    walker = MetaWalker(PROBE_Q, mem, commit=False)
    print(f"[walk] use_hyde={walker._use_hyde}  query_kws={walker._query_keywords}")

    recall_hits: set = set()
    for _ in range(walker.n_stages):
        out = walker.step()
        print(f"\n--- stage {out.stage} (done={out.done}) ---")
        if out.facts:
            print("  retrieved facts (id | relevance | content) :")
            for f in out.facts:
                fid = f.get("id")
                lab = f.get("relevance", "?")
                gold = " <==GOLD" if fid in PROBE_GOLD else ""
                if fid in PROBE_GOLD:
                    recall_hits.add(fid)
                print(f"    {fid:8} {lab:11} {str(f.get('content',''))[:54]}{gold}")
        cf = out.chosen_fact
        print(f"  fact*  : {cf.get('id') if cf else None}")
        if out.thought:
            print(f"  thought: {str(out.thought.get('content',''))[:70]}")
            print(f"  thought.kw: {out.thought.get('keywords')}")
        if out.done:
            break

    print(f"\n[walk] cumulative fact ids: {walker._fact_ids_cum}")
    got = sorted(recall_hits)
    print(f"[walk] GOLD recall: {got or 'NONE'}  "
          f"({len(got)}/{len(PROBE_GOLD)})  "
          f"=> {'HIT' if got else 'MISS'}")
    return mem


# ----------------------------------------------------------------------
# STEP 4 — the FULL agent path (mcp_meta_agent.answer), end-to-end
# ----------------------------------------------------------------------
def cmd_agent(mem: Optional[Memory] = None) -> Memory:
    print("\n" + "=" * 70)
    print("STEP 4 — FULL AGENT PATH (answer + agent_recall + F1)")
    print("=" * 70)
    mem = mem or build_memory(verbose=False)
    from benchmarks.locomo.mcp_meta_agent import McpMetaAgent
    from benchmarks.locomo.eval import official_score

    agent = McpMetaAgent()
    result = agent.answer(mem, PROBE_Q, k=7, max_steps=3, use_lineage=False)
    pred = result.get("answer", "")
    retrieved = result.get("retrieved_ids", []) or []
    trace = result.get("trace") or []

    hits = sorted(set(retrieved) & set(PROBE_GOLD))
    recall = len(hits) / max(1, len(PROBE_GOLD))
    f1 = official_score(pred, PROBE_ANSWER, PROBE_CATEGORY)

    print(f"[agent] question : {PROBE_Q}")
    print(f"[agent] gold     : {PROBE_ANSWER}")
    print(f"[agent] PREDICTED: {pred!r}")
    print(f"[agent] F1            = {f1:.3f}")
    print(f"[agent] agent_recall  = {recall:.3f}  hits={hits}  gold={PROBE_GOLD}")
    print(f"[agent] retrieved_ids = {retrieved}")
    print(f"[agent] trace ({len(trace)} steps) :")
    for t in trace:
        print(f"   - {str(t)[:140]}")
    return mem


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "index":
        cmd_index()
    elif cmd == "retrieve":
        cmd_retrieve()
    elif cmd == "walk":
        cmd_walk()
    elif cmd == "agent":
        cmd_agent()
    else:
        mem = cmd_index()
        cmd_retrieve(mem)
        cmd_walk(mem)
        cmd_agent(mem)


if __name__ == "__main__":
    main()
