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

Categories (per LoCoMo paper) :
  1 = single-hop   2 = multi-hop   3 = temporal
  4 = open-domain  5 = adversarial
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from typing import Any, Dict, List

from metacog import Memory


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def f1_score(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    p_set: Dict[str, int] = {}
    g_set: Dict[str, int] = {}
    for t in p:
        p_set[t] = p_set.get(t, 0) + 1
    for t in g:
        g_set[t] = g_set.get(t, 0) + 1
    overlap = sum(min(p_set.get(t, 0), g_set.get(t, 0)) for t in p_set)
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p_set.values())
    recall = overlap / sum(g_set.values())
    return 2 * precision * recall / (precision + recall)


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


def ingest_conversation(memory: Memory, conv: Dict[str, Any]) -> int:
    """Ingest each dialog turn with sequence_prev so lineage traversal
    can walk adjacent turns within a session."""
    count = 0
    for key in conv:
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        turns = conv[key]
        if not isinstance(turns, list):
            continue
        prev_id = None
        for turn in turns:
            text = turn.get("text", "")
            speaker = turn.get("speaker", "")
            dia_id = turn.get("dia_id")
            if not text or not dia_id:
                continue
            try:
                memory.ingest(
                    content=f"{speaker}: {text}",
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
    k_retrieve: int = 10,
    top_chunks_for_answer: int = 3,
    max_qa: int = None,
    answerer: str = "chunk",
    use_lineage: bool = False,
    claude_answerer=None,
    debug_writer=None,
) -> Dict[str, Any]:
    qas = sample["qa"]
    if max_qa is not None:
        qas = qas[:max_qa]

    enc = make_encoder(encoder_name)
    memory = Memory(encoder=enc)
    ingested = ingest_conversation(memory, sample["conversation"])

    per_cat = defaultdict(lambda: {
        "n": 0, "recall_at_5": 0.0, "recall_at_10": 0.0, "f1": 0.0,
        "tokens_in": 0, "tokens_out": 0, "steps": 0,
    })

    for qa in qas:
        question = qa.get("question", "")
        gold_answer = str(qa.get("answer", ""))
        evidence = qa.get("evidence", []) or []
        category = qa.get("category", 0)
        if not question:
            continue

        results = memory.retrieve(question, k=k_retrieve, use_lineage=use_lineage)
        retrieved_ids = [r["id"] for r in results]
        top5 = set(retrieved_ids[:5])
        top10 = set(retrieved_ids[:10])
        ev_set = set(evidence)
        if ev_set:
            r5 = len(top5 & ev_set) / len(ev_set)
            r10 = len(top10 & ev_set) / len(ev_set)
        else:
            r5 = r10 = 0.0

        tokens_in = tokens_out = steps = 0
        react_trace: List[Dict[str, Any]] = []
        if answerer == "extractive":
            from benchmarks.locomo.react_qa import react_answer
            ra = react_answer(memory, question, max_steps=2,
                              k_retrieve=k_retrieve)
            pred = ra["answer"]
        elif answerer == "claude":
            ca = claude_answerer.answer(
                memory, question,
                k=k_retrieve, max_steps=3, use_lineage=use_lineage,
            )
            pred = ca["answer"]
            tokens_in = ca.get("tokens_in", 0) or 0
            tokens_out = ca.get("tokens_out", 0) or 0
            steps = ca.get("steps", 0) or 0
            react_trace = ca.get("trace", []) or []
        else:  # "chunk"
            pred = " ".join(r["content"] for r in results[:top_chunks_for_answer])
        f1 = f1_score(pred, gold_answer)

        per_cat[category]["n"] += 1
        per_cat[category]["recall_at_5"] += r5
        per_cat[category]["recall_at_10"] += r10
        per_cat[category]["f1"] += f1
        per_cat[category]["tokens_in"] += tokens_in
        per_cat[category]["tokens_out"] += tokens_out
        per_cat[category]["steps"] += steps

        if debug_writer is not None:
            ev_hits_5 = sorted(top5 & ev_set)
            ev_misses = sorted(ev_set - top10)
            # Compact dump of retrieved chunk content so we can read what
            # the answerer actually saw, not just the ids.
            top10_dump = [
                {"id": r["id"],
                 "kind": r.get("kind"),
                 "content": (r.get("content") or "")[:200]}
                for r in results[:10]
            ]
            # Keep ReAct trace compact : drop raw_reply > 300 chars
            trace_dump = []
            for t in react_trace:
                trace_dump.append({
                    "step": t.get("step"),
                    "query": t.get("query"),
                    "n_evidence": t.get("n_evidence"),
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
                "r10": round(r10, 4),
                "retrieved_top10": top10_dump,
                "gold_evidence": evidence,
                "evidence_hits_in_top5": ev_hits_5,
                "evidence_missed_top10": ev_misses,
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
    total = {"n": 0, "recall_at_5": 0.0, "recall_at_10": 0.0, "f1": 0.0,
             "tokens_in": 0, "tokens_out": 0, "steps": 0}
    for cat, stats in per_cat.items():
        n = stats["n"]
        if n == 0:
            continue
        cat_entry = {
            "n": n,
            "recall_at_5": round(stats["recall_at_5"] / n, 4),
            "recall_at_10": round(stats["recall_at_10"] / n, 4),
            "f1": round(stats["f1"] / n, 4),
        }
        if stats["tokens_in"] or stats["tokens_out"]:
            cat_entry["tokens_in_per_qa"] = round(stats["tokens_in"] / n, 1)
            cat_entry["tokens_out_per_qa"] = round(stats["tokens_out"] / n, 1)
            cat_entry["steps_per_qa"] = round(stats["steps"] / n, 2)
        summary["by_category"][cat] = cat_entry
        total["n"] += n
        total["recall_at_5"] += stats["recall_at_5"]
        total["recall_at_10"] += stats["recall_at_10"]
        total["f1"] += stats["f1"]
        total["tokens_in"] += stats["tokens_in"]
        total["tokens_out"] += stats["tokens_out"]
        total["steps"] += stats["steps"]
    if total["n"]:
        overall = {
            "n": total["n"],
            "recall_at_5": round(total["recall_at_5"] / total["n"], 4),
            "recall_at_10": round(total["recall_at_10"] / total["n"], 4),
            "f1": round(total["f1"] / total["n"], 4),
        }
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
    parser.add_argument("--k", type=int, default=10, help="retrieval top-k")
    parser.add_argument("--top-chunks", type=int, default=3)
    parser.add_argument("--encoder", choices=["simple", "semantic"],
                        default="semantic")
    parser.add_argument("--answerer", choices=["chunk", "extractive", "claude"],
                        default="chunk",
                        help="how to produce the answer from retrieved chunks")
    parser.add_argument("--claude-model", default=None,
                        help="Anthropic model id (default : env CLAUDE_MODEL or claude-haiku-4-5-20251001)")
    parser.add_argument("--react", action="store_true",
                        help="(deprecated alias for --answerer=extractive)")
    parser.add_argument("--lineage", action="store_true",
                        help="retrieve_with_lineage (cosine + adjacent + RRF)")
    parser.add_argument("--debug-jsonl", default=None,
                        help="per-QA full trace (question, gold, pred, retrieved "
                             "ids, hits, misses, ReAct steps, tokens) written one "
                             "JSON object per line. Use to root-cause low F1.")
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

    debug_writer = None
    if args.debug_jsonl:
        debug_writer = open(args.debug_jsonl, "w", encoding="utf-8")
        print(f"Per-QA debug trace → {args.debug_jsonl}")

    results = []
    overall = {"n": 0, "recall_at_5": 0.0, "recall_at_10": 0.0, "f1": 0.0,
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
            claude_answerer=claude_answerer,
            debug_writer=debug_writer,
        )
        results.append(summary)
        print(json.dumps(summary, indent=2))
        if "overall" in summary:
            o = summary["overall"]
            overall["n"] += o["n"]
            overall["recall_at_5"] += o["recall_at_5"] * o["n"]
            overall["recall_at_10"] += o["recall_at_10"] * o["n"]
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
            "recall_at_10": round(overall["recall_at_10"] / overall["n"], 4),
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
