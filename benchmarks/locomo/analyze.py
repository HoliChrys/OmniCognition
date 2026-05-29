"""
Analyse a LoCoMo debug-JSONL trace and compare per-category F1 / recall
against the HeLa-Mem reported scores.

Usage :
  python -m benchmarks.locomo.analyze /tmp/mcp_full.jsonl

Reads the per-QA records written by `eval.py --debug-jsonl` and prints :
  - per-category n, F1, recall@5, recall@7
  - overall F1 / recall
  - side-by-side vs HeLa-Mem (GPT-4o-mini, LoCoMo) with a ✓/✗ verdict
  - a "retrieval-bound" count : QAs where the gold was not in top-7
    (so F1 is capped by retrieval, not generation)
"""

from __future__ import annotations

import collections
import json
import sys
from typing import Dict, List

# Official LoCoMo category convention (Maharana et al. 2024) — verified
# against the data (cat 2 questions are all "When did…" = temporal).
CAT_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# HeLa-Mem reported F1 (GPT-4o-mini backbone, LoCoMo) — the bar to beat,
# keyed by the CORRECT category number.
HELA_F1 = {
    1: 0.401,   # multi-hop
    2: 0.473,   # temporal
    3: 0.297,   # open-domain
    4: 0.519,   # single-hop
}


def load(path: str) -> List[dict]:
    return [json.loads(ln) for ln in open(path) if ln.strip()]


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m benchmarks.locomo.analyze <debug.jsonl>")
        sys.exit(1)
    rows = load(sys.argv[1])
    if not rows:
        print("empty trace")
        sys.exit(0)

    cat: Dict[int, dict] = collections.defaultdict(
        lambda: {"n": 0, "f1": 0.0, "r5": 0.0, "r7": 0.0,
                 "ar": 0.0, "ar_n": 0, "ret_miss": 0}
    )
    for r in rows:
        c = r.get("category", 0)
        s = cat[c]
        s["n"] += 1
        s["f1"] += r.get("f1", 0.0)
        s["r5"] += r.get("r5", 0.0)
        s["r7"] += r.get("r7", 0.0)
        ar = r.get("agent_recall")
        if ar is not None:
            s["ar"] += ar
            s["ar_n"] += 1
        if r.get("gold_evidence") and r.get("r7", 0.0) == 0.0:
            s["ret_miss"] += 1

    print(f"\nTrace : {sys.argv[1]}  ({len(rows)} QAs)\n")
    hdr = (f"{'category':<13}{'n':>4}{'F1':>8}{'HeLa':>8}{'verdict':>9}"
           f"{'R@5':>7}{'R@7':>7}{'agentR':>8}{'ret-miss':>9}")
    print(hdr)
    print("-" * len(hdr))

    tot = {"n": 0, "f1": 0.0, "r5": 0.0, "r7": 0.0,
           "ar": 0.0, "ar_n": 0, "ret_miss": 0}
    beat = 0
    compared = 0
    for c in sorted(cat):
        s = cat[c]
        n = s["n"]
        f1 = s["f1"] / n
        r5 = s["r5"] / n
        r7 = s["r7"] / n
        ar = s["ar"] / s["ar_n"] if s["ar_n"] else 0.0
        ar_s = f"{ar:.3f}" if s["ar_n"] else "—"
        hela = HELA_F1.get(c)
        if hela is not None:
            compared += 1
            ok = f1 >= hela
            beat += int(ok)
            verdict = "✓ beat" if ok else "✗ below"
            hela_s = f"{hela:.3f}"
        else:
            verdict = "—"
            hela_s = "—"
        print(f"{CAT_NAMES.get(c, c):<13}{n:>4}{f1:>8.3f}{hela_s:>8}"
              f"{verdict:>9}{r5:>7.3f}{r7:>7.3f}{ar_s:>8}{s['ret_miss']:>9}")
        for kk in tot:
            tot[kk] += s[kk]

    n = tot["n"]
    tot_ar = f"{tot['ar']/tot['ar_n']:.3f}" if tot["ar_n"] else "—"
    print("-" * len(hdr))
    print(f"{'OVERALL':<13}{n:>4}{tot['f1']/n:>8.3f}{'':>8}"
          f"{'':>9}{tot['r5']/n:>7.3f}{tot['r7']/n:>7.3f}{tot_ar:>8}"
          f"{tot['ret_miss']:>9}")
    print(f"\nBeat HeLa on {beat}/{compared} compared categories.")
    print(f"Retrieval-bound QAs (gold not in top-7) : {tot['ret_miss']}/{n} "
          f"({100*tot['ret_miss']/n:.0f}%) — F1 ceiling from retrieval, "
          f"not generation.")


if __name__ == "__main__":
    main()
