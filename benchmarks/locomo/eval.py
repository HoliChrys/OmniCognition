"""
LoCoMo-10 benchmark runner for MetaCog-Mem.

Usage :
  python -m benchmarks.locomo.eval --data data/locomo10.json [...]

Per conversation :
  1. Ingest every dialog turn as a FACT (id = dia_id,
     content = "<speaker>: <text>"), linked via sequence_prev/next.
  2. Per QA :
     - retrieve top-k by question (cosine or cosine+lineage RRF)
     - measure Recall@5 / Recall@10 vs gold evidence dia_ids
     - synthesize an answer (chunk dump, extractive ReAct, …)
     - measure token-overlap F1 vs gold

Categories (official LoCoMo convention, Maharana et al. 2024) :
  1 = multi-hop    2 = temporal    3 = open-domain
  4 = single-hop   5 = adversarial
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import defaultdict
from typing import Any, Dict, List

from metacog import Memory


# Official LoCoMo metric — imported VERBATIM from the vendored upstream
# (benchmarks/locomo/official_locomo_eval.py, copied from snap-research/
# locomo task_eval/evaluation.py). We do NOT reimplement scoring : the
# math and the per-category dispatch are the upstream code.
from benchmarks.locomo.official_locomo_eval import (  # noqa: E402
    f1_score,
    normalize_answer,
    score_qa as official_score,
)


def make_encoder(name: str):
    if name == "simple":
        from metacog.defaults import SimpleEncoder
        return SimpleEncoder()
    elif name == "semantic":
        from benchmarks.locomo.encoders import SemanticEncoder
        return SemanticEncoder()
    raise ValueError(f"unknown encoder: {name}")


def load_locomo(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


_DATE_AFTER_ON_RE = re.compile(r"\bon\s+(.+)$")


def _clean_session_date(raw: str) -> str:
    """'1:56 pm on 8 May, 2023' → '8 May 2023'.

    LoCoMo session timestamps carry a clock time prefix we don't need ;
    keep only the calendar date so it can anchor relative references
    ("yesterday", "last week") to an absolute date the gold expects.
    """
    if not raw:
        return ""
    m = _DATE_AFTER_ON_RE.search(raw)
    date = m.group(1) if m else raw
    return date.replace(",", "").strip()


def ingest_conversation(
    memory: Memory, conv: Dict[str, Any], *, with_dates: bool = True
) -> int:
    """Ingest each dialog turn with sequence_prev so lineage traversal
    can walk adjacent turns within a session.

    When `with_dates`, each turn's content is prefixed with its session
    date — `[8 May 2023] Caroline: ... yesterday` — so the answerer can
    resolve relative time references to the absolute dates the LoCoMo
    gold answers use. Without this anchor, temporal questions are
    unanswerable even with perfect retrieval (the turn only says
    "yesterday").
    """
    count = 0
    for key in conv:
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        turns = conv[key]
        if not isinstance(turns, list):
            continue
        session_date = _clean_session_date(conv.get(f"{key}_date_time", ""))
        prefix = f"[{session_date}] " if (with_dates and session_date) else ""
        prev_id = None
        for turn in turns:
            text = turn.get("text", "")
            speaker = turn.get("speaker", "")
            dia_id = turn.get("dia_id")
            if not text or not dia_id:
                continue
            try:
                memory.ingest(
                    content=f"{prefix}{speaker}: {text}",
                    kind="FACT",
                    id=dia_id,
                    sequence_prev=prev_id,
                )
                count += 1
                prev_id = dia_id
            except Exception:
                pass
    return count


def evaluate_sample(
    sample: Dict[str, Any],
    encoder_name: str,
    *,
    k_retrieve: int = 7,
    top_chunks_for_answer: int = 3,
    max_qa: int = None,
    answerer: str = "chunk",
    use_lineage: bool = False,
    use_hybrid: bool = False,
    with_dates: bool = True,
    agent_concurrency: int = 10,
    sample_seed: int = None,
    claude_answerer=None,
    debug_writer=None,
    entities: bool = False,
    merge: bool = False,
    per_category: int = None,
    auto_cluster: bool = False,
) -> Dict[str, Any]:
    qas = list(sample["qa"])
    if sample_seed is not None:
        # Deterministic random sample per conversation, so the category
        # mix reflects the true distribution instead of whatever the
        # first-N QAs happen to be.
        import random
        random.Random(sample_seed).shuffle(qas)
    if per_category is not None:
        # Stratified : up to `per_category` QAs from EACH category, so the
        # five categories are balanced (equal n) instead of following the
        # natural distribution. Shuffle above randomises which ones.
        from collections import defaultdict as _dd
        picked = _dd(list)
        for qa in qas:
            c = qa.get("category", 0)
            if len(picked[c]) < per_category:
                picked[c].append(qa)
        qas = [qa for c in sorted(picked) for qa in picked[c]]
    elif max_qa is not None:
        qas = qas[:max_qa]

    enc = make_encoder(encoder_name)
    entity_extractor = None
    if entities:
        from metacog.entities import LLMEntityExtractor
        entity_extractor = LLMEntityExtractor()
    memory = Memory(encoder=enc, entity_extractor=entity_extractor)
    ingested = ingest_conversation(
        memory, sample["conversation"], with_dates=with_dates,
    )
    if merge:
        # Collapse genuine same-info turns into single nodes (corroboration
        # absorbed). Recall scoring below resolves absorbed ids to survivors.
        mrep = memory.consolidate_duplicates()
        print(f"  merged {mrep['merged_count']} same-info turns "
              f"-> {mrep['n_points']} points")
    if auto_cluster:
        # Detect Level-1 communities and instantiate one observator each
        # (the agent can focus a walk on a community via observator_id).
        crep = memory.auto_cluster_observators()
        print(f"  auto-clustered {len(crep)} communities: "
              f"{[c['id'] for c in crep]}")

    per_cat = defaultdict(lambda: {
        "n": 0, "recall_at_5": 0.0, "recall_at_7": 0.0, "f1": 0.0,
        "agent_recall": 0.0, "agent_recall_n": 0,
        "tokens_in": 0, "tokens_out": 0, "steps": 0,
    })

    # For the MCP agent, precompute all answers concurrently. Each QA
    # runs the agent in its own thread (own asyncio loop + own in-process
    # MCP session) ; the conversation memory is read-only during QA, so
    # concurrent access is safe. This turns an otherwise multi-hour
    # sequential run into a parallel one.
    precomputed: Dict[int, Dict[str, Any]] = {}
    if answerer in ("mcp", "meta"):
        from concurrent.futures import ThreadPoolExecutor

        valid = [(i, qa) for i, qa in enumerate(qas) if qa.get("question")]

        def _run(idx_qa):
            idx, qa = idx_qa
            # Resilient : a transient per-QA API error (e.g. a DNS blip)
            # must not crash the whole run. Retry once, then degrade to an
            # empty answer for just this QA.
            for attempt in range(2):
                try:
                    return idx, claude_answerer.answer(
                        memory, qa["question"],
                        k=k_retrieve, max_steps=3, use_lineage=use_lineage,
                    )
                except Exception as exc:  # noqa: BLE001
                    if attempt == 0:
                        time.sleep(3)
                        continue
                    print(f"  [warn] QA {idx} failed twice ({exc}); "
                          "recording empty answer")
                    return idx, {"answer": "", "trace": [], "retrieved_ids": []}

        with ThreadPoolExecutor(max_workers=agent_concurrency) as ex:
            for idx, ca in ex.map(_run, valid):
                precomputed[idx] = ca

    for qa_index, qa in enumerate(qas):
        question = qa.get("question", "")
        # Category 5 (adversarial) stores its gold in `adversarial_answer`,
        # not `answer` (which is empty). Scoring those against "" forced
        # an F1 of 0 on every adversarial QA.
        gold_answer = str(qa.get("answer", "") or qa.get("adversarial_answer", ""))
        evidence = qa.get("evidence", []) or []
        category = qa.get("category", 0)
        if not question:
            continue

        results = memory.retrieve(
            question, k=k_retrieve,
            use_hybrid=use_hybrid, use_lineage=use_lineage,
        )
        retrieved_ids = [r["id"] for r in results]
        # Resolve absorbed (merged) ids to their surviving node so a survivor
        # still credits recall for an evidence turn that was folded into it.
        _rid = memory.resolve_alias
        top5 = {_rid(i) for i in retrieved_ids[:5]}
        top7 = {_rid(i) for i in retrieved_ids[:7]}
        ev_set = {_rid(e) for e in evidence}
        if ev_set:
            r5 = len(top5 & ev_set) / len(ev_set)
            r7 = len(top7 & ev_set) / len(ev_set)
        else:
            r5 = r7 = 0.0

        tokens_in = tokens_out = steps = 0
        react_trace: List[Dict[str, Any]] = []
        agent_recall = None        # cumulative recall across agent queries
        agent_retrieved_ids: List[str] = []
        if answerer == "extractive":
            from benchmarks.locomo.react_qa import react_answer
            ra = react_answer(memory, question, max_steps=2,
                              k_retrieve=k_retrieve)
            pred = ra["answer"]
        elif answerer in ("claude", "mcp", "meta"):
            if answerer in ("mcp", "meta"):
                ca = precomputed.get(qa_index, {"answer": ""})
            else:
                ca = claude_answerer.answer(
                    memory, question,
                    k=k_retrieve, max_steps=3, use_lineage=use_lineage,
                )
            pred = ca["answer"]
            tokens_in = ca.get("tokens_in", 0) or 0
            tokens_out = ca.get("tokens_out", 0) or 0
            steps = ca.get("steps", 0) or 0
            react_trace = ca.get("trace", []) or []
            agent_retrieved_ids = ca.get("retrieved_ids", []) or []
            if ev_set:
                agent_seen = {_rid(i) for i in agent_retrieved_ids}
                agent_recall = len(agent_seen & ev_set) / len(ev_set)
        else:  # "chunk"
            pred = " ".join(r["content"] for r in results[:top_chunks_for_answer])
        # Official per-category scoring (cat5 = abstention, cat1 = multi-hop
        # sub-answer F1, cat3 = first ';'-clause, cat2/4 = token-F1).
        f1 = official_score(pred, gold_answer, category)

        per_cat[category]["n"] += 1
        per_cat[category]["recall_at_5"] += r5
        per_cat[category]["recall_at_7"] += r7
        per_cat[category]["f1"] += f1
        if agent_recall is not None:
            per_cat[category]["agent_recall"] += agent_recall
            per_cat[category]["agent_recall_n"] += 1
        per_cat[category]["tokens_in"] += tokens_in
        per_cat[category]["tokens_out"] += tokens_out
        per_cat[category]["steps"] += steps

        if debug_writer is not None:
            ev_hits_5 = sorted(top5 & ev_set)
            ev_misses = sorted(ev_set - top7)
            # Compact dump of retrieved chunk content so we can read what
            # the answerer actually saw, not just the ids.
            top7_dump = [
                {"id": r["id"],
                 "kind": r.get("kind"),
                 "content": (r.get("content") or "")[:200]}
                for r in results[:7]
            ]
            # Keep ReAct trace compact. Two trace shapes are supported :
            # the claude_react shape (step/query/parsed/raw_reply) and the
            # mcp_agent shape (round/action/name/input/text).
            trace_dump = []
            for t in react_trace:
                if "action" in t:  # mcp_agent shape
                    trace_dump.append({
                        "round": t.get("round"),
                        "action": t.get("action"),
                        "name": t.get("name"),
                        "input": t.get("input"),
                        "text": (t.get("text") or "")[:200],
                    })
                else:  # claude_react shape
                    trace_dump.append({
                        "step": t.get("step"),
                        "query": t.get("query"),
                        "parsed": t.get("parsed"),
                        "raw_reply": (t.get("raw_reply") or "")[:300],
                    })
            debug_writer.write(json.dumps({
                "sample_id": sample.get("sample_id"),
                "category": category,
                "question": question,
                "gold_answer": gold_answer,
                "pred": pred,
                "f1": round(f1, 4),
                "r5": round(r5, 4),
                "r7": round(r7, 4),
                "agent_recall": (round(agent_recall, 4)
                                 if agent_recall is not None else None),
                "agent_retrieved_ids": agent_retrieved_ids,
                "retrieved_top7": top7_dump,
                "gold_evidence": evidence,
                "evidence_hits_in_top5": ev_hits_5,
                "evidence_missed_top7": ev_misses,
                "react_trace": trace_dump,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "steps": steps,
            }, ensure_ascii=False) + "\n")
            debug_writer.flush()

    summary: Dict[str, Any] = {
        "sample_id": sample.get("sample_id"),
        "encoder": encoder_name,
        "use_lineage": use_lineage,
        "answerer": answerer,
        "n_dialog_turns_ingested": ingested,
        "n_qa": len(qas),
        "by_category": {},
    }
    total = {"n": 0, "recall_at_5": 0.0, "recall_at_7": 0.0, "f1": 0.0,
             "agent_recall": 0.0, "agent_recall_n": 0,
             "tokens_in": 0, "tokens_out": 0, "steps": 0}
    for cat, stats in per_cat.items():
        n = stats["n"]
        if n == 0:
            continue
        cat_entry = {
            "n": n,
            "recall_at_5": round(stats["recall_at_5"] / n, 4),
            "recall_at_7": round(stats["recall_at_7"] / n, 4),
            "f1": round(stats["f1"] / n, 4),
        }
        if stats["agent_recall_n"]:
            cat_entry["agent_recall"] = round(
                stats["agent_recall"] / stats["agent_recall_n"], 4
            )
        if stats["tokens_in"] or stats["tokens_out"]:
            cat_entry["tokens_in_per_qa"] = round(stats["tokens_in"] / n, 1)
            cat_entry["tokens_out_per_qa"] = round(stats["tokens_out"] / n, 1)
            cat_entry["steps_per_qa"] = round(stats["steps"] / n, 2)
        summary["by_category"][cat] = cat_entry
        total["n"] += n
        total["recall_at_5"] += stats["recall_at_5"]
        total["recall_at_7"] += stats["recall_at_7"]
        total["f1"] += stats["f1"]
        total["agent_recall"] += stats["agent_recall"]
        total["agent_recall_n"] += stats["agent_recall_n"]
        total["tokens_in"] += stats["tokens_in"]
        total["tokens_out"] += stats["tokens_out"]
        total["steps"] += stats["steps"]
    if total["n"]:
        overall = {
            "n": total["n"],
            "recall_at_5": round(total["recall_at_5"] / total["n"], 4),
            "recall_at_7": round(total["recall_at_7"] / total["n"], 4),
            "f1": round(total["f1"] / total["n"], 4),
        }
        if total["agent_recall_n"]:
            overall["agent_recall"] = round(
                total["agent_recall"] / total["agent_recall_n"], 4
            )
        if total["tokens_in"] or total["tokens_out"]:
            overall["tokens_in_per_qa"] = round(total["tokens_in"] / total["n"], 1)
            overall["tokens_out_per_qa"] = round(total["tokens_out"] / total["n"], 1)
            overall["tokens_total_per_qa"] = round(
                (total["tokens_in"] + total["tokens_out"]) / total["n"], 1
            )
            overall["steps_per_qa"] = round(total["steps"] / total["n"], 2)
        summary["overall"] = overall
    return summary


def main():
    parser = argparse.ArgumentParser(prog="benchmarks.locomo.eval")
    parser.add_argument("--data", default="benchmarks/locomo/data/locomo10.json")
    parser.add_argument("--samples", type=int, default=10,
                        help="how many of the 10 conversations to evaluate")
    parser.add_argument("--max-qa", type=int, default=None,
                        help="cap QA pairs per sample")
    parser.add_argument("--per-category", type=int, default=None,
                        help="stratified : up to N QAs from EACH category "
                             "per sample (balanced n per category instead of "
                             "the natural distribution). Overrides --max-qa.")
    parser.add_argument("--k", type=int, default=7,
                        help="retrieval top-k (recall measured @5 and @7)")
    parser.add_argument("--top-chunks", type=int, default=3)
    parser.add_argument("--encoder", choices=["simple", "semantic"],
                        default="semantic")
    parser.add_argument("--answerer",
                        choices=["chunk", "extractive", "claude", "mcp", "meta"],
                        default="chunk",
                        help="how to produce the answer. 'claude' = single-shot "
                             "ReAct with pre-dumped chunks ; 'mcp' = real "
                             "tool-using ReAct agent connected to the metacog "
                             "MCP server (multi-round retrieval) ; 'meta' = "
                             "in-memory meta-cognitive walk (autonomous "
                             "FACT/ACTION/THOUGHT cycle, ReAct kicks in only "
                             "for tool use which LoCoMo doesn't need).")
    parser.add_argument("--claude-model", default=None,
                        help="Anthropic model id (default : env CLAUDE_MODEL or claude-haiku-4-5-20251001)")
    parser.add_argument("--react", action="store_true",
                        help="(deprecated alias for --answerer=extractive)")
    parser.add_argument("--lineage", action="store_true",
                        help="retrieve_with_lineage (cosine + adjacent + RRF)")
    parser.add_argument("--use-hybrid", action="store_true",
                        help="metric retrieval = hybrid (keyword-embedding "
                             "cosine + BM25 + RRF) instead of plain cosine kNN. "
                             "Matches what the claude answerer retrieves "
                             "internally. Works with the lightweight simhash "
                             "encoder (no torch).")
    parser.add_argument("--agent-concurrency", type=int, default=10,
                        help="number of MCP-agent QAs to run in parallel "
                             "(threads). Cuts wall-time for --answerer mcp.")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="deterministically shuffle each conversation's "
                             "QAs before --max-qa, so a capped run is a "
                             "representative random sample (category mix "
                             "matches the dataset) instead of the first-N.")
    parser.add_argument("--no-dates", action="store_true",
                        help="disable session-date prefix on ingested turns "
                             "(baseline for measuring the temporal-anchor "
                             "contribution). Default : dates ON.")
    parser.add_argument("--debug-jsonl", default=None,
                        help="per-QA full trace (question, gold, pred, retrieved "
                             "ids, hits, misses, ReAct steps, tokens) written one "
                             "JSON object per line. Use to root-cause low F1.")
    parser.add_argument("--entities", action="store_true",
                        help="opt-in : LLM entity extraction at ingest. Each FACT "
                             "spawns tagged entity beacon nodes (dates decomposed "
                             "into day/month/year) pulled onto it geometrically — "
                             "the edge-free 'edges-equivalent' that lifts the real "
                             "fact into recall. One LLM call per turn (cached).")
    parser.add_argument("--merge", action="store_true",
                        help="opt-in : after ingest, collapse genuine same-info "
                             "turns (content-cosine >= 0.95) into single nodes, "
                             "absorbing corroboration. Recall resolves absorbed "
                             "ids to survivors. No LLM (pure geometric).")
    parser.add_argument("--auto-cluster", action="store_true",
                        help="opt-in : after ingest, detect Level-1 topical "
                             "communities (Louvain, edge-free) and create one "
                             "observator per community. The agent can focus a "
                             "walk on a community via list_communities + "
                             "walk_start(observator_id=…). No LLM.")
    args = parser.parse_args()
    if args.react and args.answerer == "chunk":
        args.answerer = "extractive"

    data = load_locomo(args.data)
    print(f"Loaded {len(data)} conversations from {args.data}")

    claude_answerer = None
    if args.answerer == "claude":
        from benchmarks.locomo.claude_react import ClaudeReactAnswerer
        kwargs = {}
        if args.claude_model:
            kwargs["model"] = args.claude_model
        claude_answerer = ClaudeReactAnswerer(**kwargs)
        print(f"Claude answerer ready : model={claude_answerer.model}")
    elif args.answerer == "mcp":
        from benchmarks.locomo.mcp_agent import McpReactAgent
        kwargs = {}
        if args.claude_model:
            kwargs["model"] = args.claude_model
        claude_answerer = McpReactAgent(**kwargs)
        print(f"MCP ReAct agent ready : model={claude_answerer.model} "
              f"(in-process metacog MCP, max_rounds={claude_answerer.max_rounds})")
    elif args.answerer == "meta":
        from benchmarks.locomo.mcp_meta_agent import McpMetaAgent
        kwargs = {}
        if args.claude_model:
            kwargs["model"] = args.claude_model
        claude_answerer = McpMetaAgent(**kwargs)
        print(f"MCP meta-walk agent ready : model={claude_answerer.model} "
              f"(walks the metacog memory stage-by-stage, "
              f"max_rounds={claude_answerer.max_rounds})")

    debug_writer = None
    if args.debug_jsonl:
        debug_writer = open(args.debug_jsonl, "w", encoding="utf-8")
        print(f"Per-QA debug trace → {args.debug_jsonl}")

    results = []
    overall = {"n": 0, "recall_at_5": 0.0, "recall_at_7": 0.0, "f1": 0.0,
               "tokens_in": 0.0, "tokens_out": 0.0, "steps": 0.0}
    t0 = time.time()
    for i, sample in enumerate(data[: args.samples]):
        print(
            f"\n=== sample {i+1}/{args.samples}: {sample.get('sample_id')} "
            f"(encoder={args.encoder}, lineage={args.lineage}, answerer={args.answerer}) ==="
        )
        summary = evaluate_sample(
            sample, args.encoder,
            k_retrieve=args.k,
            top_chunks_for_answer=args.top_chunks,
            max_qa=args.max_qa,
            answerer=args.answerer,
            use_lineage=args.lineage,
            use_hybrid=args.use_hybrid,
            with_dates=not args.no_dates,
            agent_concurrency=args.agent_concurrency,
            sample_seed=args.shuffle_seed,
            claude_answerer=claude_answerer,
            debug_writer=debug_writer,
            entities=args.entities,
            merge=args.merge,
            per_category=args.per_category,
            auto_cluster=args.auto_cluster,
        )
        results.append(summary)
        print(json.dumps(summary, indent=2))
        if "overall" in summary:
            o = summary["overall"]
            overall["n"] += o["n"]
            overall["recall_at_5"] += o["recall_at_5"] * o["n"]
            overall["recall_at_7"] += o["recall_at_7"] * o["n"]
            overall["f1"] += o["f1"] * o["n"]
            if "tokens_in_per_qa" in o:
                overall["tokens_in"] += o["tokens_in_per_qa"] * o["n"]
                overall["tokens_out"] += o["tokens_out_per_qa"] * o["n"]
                overall["steps"] += o["steps_per_qa"] * o["n"]

    if overall["n"]:
        print("\n=== AGGREGATE ===")
        agg = {
            "n_samples": len(results),
            "n_qa_total": overall["n"],
            "encoder": args.encoder,
            "use_lineage": args.lineage,
            "answerer": args.answerer,
            "recall_at_5": round(overall["recall_at_5"] / overall["n"], 4),
            "recall_at_7": round(overall["recall_at_7"] / overall["n"], 4),
            "f1": round(overall["f1"] / overall["n"], 4),
            "elapsed_seconds": round(time.time() - t0, 2),
        }
        if overall["tokens_in"] or overall["tokens_out"]:
            n = overall["n"]
            agg["tokens_in_per_qa"] = round(overall["tokens_in"] / n, 1)
            agg["tokens_out_per_qa"] = round(overall["tokens_out"] / n, 1)
            agg["tokens_total_per_qa"] = round(
                (overall["tokens_in"] + overall["tokens_out"]) / n, 1
            )
            agg["steps_per_qa"] = round(overall["steps"] / n, 2)
        print(json.dumps(agg, indent=2))

    if debug_writer is not None:
        debug_writer.close()


if __name__ == "__main__":
    main()
