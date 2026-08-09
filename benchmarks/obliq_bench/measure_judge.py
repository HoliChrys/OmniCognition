"""
A/B measurement of the JUDGE at retrieval — per-item vs funnel (this branch's
cost lever). Same memory, same candidate set ; counts LLM calls, prompt/output
chars (~4 chars/token), and wall-clock for each judge mode.

  per-item : one _judge_one LLM call per candidate (the pre-funnel cost).
  funnel   : embedding pre-accept (0 LLM) + 1 batch over the gloss + per-item
             second opinion on batch-rejects only.

Build is tone-only (fast) so the gloss exists ; without the card there is no
stance_ node, so the funnel's pre-accept runs on doc|gloss — representative of
the dominant lever.

Usage: python -m benchmarks.obliq_bench.measure_judge --query-id q0459 --bg 20
"""

from __future__ import annotations

import argparse
import time


class CountingLLM:
    """Wraps a real LLM, tallying calls + prompt/output chars + time."""
    def __init__(self, inner):
        self.inner = inner
        self.reset()

    def reset(self):
        self.calls = 0
        self.chars_in = 0
        self.chars_out = 0
        self.secs = 0.0

    def generate(self, prompt, max_tokens=128):
        self.calls += 1
        self.chars_in += len(prompt or "")
        t0 = time.time()
        out = self.inner.generate(prompt, max_tokens=max_tokens) or ""
        self.secs += time.time() - t0
        self.chars_out += len(out)
        return out

    def snap(self):
        return dict(calls=self.calls, tok_in=self.chars_in // 4,
                    tok_out=self.chars_out // 4, secs=round(self.secs, 1))


def _gold_hits(labels, cands, gold):
    keep = {p.id for p, lab in zip(cands, labels) if lab != "irrelevant"}
    return len(keep & set(gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-id", default="q0459")
    ap.add_argument("--track", default="descriptive/twitter")
    ap.add_argument("--bg", type=int, default=20)
    args = ap.parse_args()

    import os
    # Card mode follows OBLIQ_NO_CARD env (set externally) ; default = card ON
    # so the pre-accept has stance_ cards to work with — the true amortized cost.
    from benchmarks.obliq_bench.event_step import build
    mem, question, gold = build(args.track, args.query_id, args.bg)
    gold = set(gold)

    # the judge's candidate set : the document pool (gold + distractors)
    cands = [p for p in mem.points if not mem._is_derived(p.id)
             and getattr(p.kind, "name", "") == "FACT"]
    print("\nMEASURE judge [%s] : %d candidates, gold in pool=%d" % (
        args.query_id, len(cands), len({p.id for p in cands} & gold)))

    mem.llm = CountingLLM(mem.llm)
    prop = mem.distill_proposition(question)     # shared, exclude from the A/B

    # ---- A : per-item (pre-funnel cost) -----------------------------------
    mem.llm.reset()
    t0 = time.time()
    lab_pi = mem.oblique_labels(question, cands, proposition=prop, per_item=True)
    wall_pi = time.time() - t0
    s_pi = mem.llm.snap()

    # ---- B : funnel (batch + pre-accept + second opinion) -----------------
    mem.llm.reset()
    t0 = time.time()
    lab_fn = mem.oblique_labels(question, cands, proposition=prop,
                                per_item=False)
    wall_fn = time.time() - t0
    s_fn = mem.llm.snap()

    # ---- C : batch-only (best case : pre-accept + 1 batch, NO 2nd opinion) -
    mem.llm.reset()
    t0 = time.time()
    lab_bo = mem.oblique_labels(question, cands, proposition=prop,
                                per_item=False, second_opinion=False)
    wall_bo = time.time() - t0
    s_bo = mem.llm.snap()

    def row(name, s, wall, lab):
        print("  %-10s calls=%-4d tok_in=%-6d tok_out=%-5d llm_s=%-5s "
              "wall_s=%-5.1f  gold_kept=%d/%d" % (
                  name, s["calls"], s["tok_in"], s["tok_out"], s["secs"],
                  wall, _gold_hits(lab, cands, gold), len(gold)))

    print("\n  RESULT (same candidates, same proposition) :")
    row("per-item", s_pi, wall_pi, lab_pi)
    row("funnel", s_fn, wall_fn, lab_fn)
    row("batch-only", s_bo, wall_bo, lab_bo)

    def ratio(a, b):
        return "%.1fx" % (a / b) if b else "inf"
    print("\n  GAIN funnel     vs per-item : calls %s | tok_in %s | tok_out %s | llm_time %s" % (
        ratio(s_pi["calls"], max(1, s_fn["calls"])),
        ratio(s_pi["tok_in"], max(1, s_fn["tok_in"])),
        ratio(s_pi["tok_out"], max(1, s_fn["tok_out"])),
        ratio(s_pi["secs"], max(0.1, s_fn["secs"]))))
    print("  GAIN batch-only vs per-item : calls %s | tok_in %s | tok_out %s | llm_time %s" % (
        ratio(s_pi["calls"], max(1, s_bo["calls"])),
        ratio(s_pi["tok_in"], max(1, s_bo["tok_in"])),
        ratio(s_pi["tok_out"], max(1, s_bo["tok_out"])),
        ratio(s_pi["secs"], max(0.1, s_bo["secs"]))))


if __name__ == "__main__":
    main()
