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

CAT_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}

# HeLa-Mem reported F1 (GPT-4o-mini backbone, LoCoMo) — the bar to beat.
HELA_F1 = {
    1: 0.519,   # single-hop
    2: 0.401,   # multi-hop
    3: 0.473,   # temporal
    4: 0.297,   # open-domain
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
        lambda: {"n": 0, "f1": 0.0, "r5": 0.0, "r7": 0.0, "ret_miss": 0}
    )
    for r in rows:
        c = r.get("category", 0)
        s = cat[c]
        s["n"] += 1
        s["f1"] += r.get("f1", 0.0)
        s["r5"] += r.get("r5", 0.0)
        s["r7"] += r.get("r7", 0.0)
        if r.get("gold_evidence") and r.get("r7", 0.0) == 0.0:
            s["ret_miss"] += 1

    print(f"\nTrace : {sys.argv[1]}  ({len(rows)} QAs)\n")
    hdr = (f"{'category':<13}{'n':>4}{'F1':>8}{'HeLa':>8}{'verdict':>9}"
           f"{'R@5':>7}{'R@7':>7}{'ret-miss':>9}")
    print(hdr)
    print("-" * len(hdr))

    tot = {"n": 0, "f1": 0.0, "r5": 0.0, "r7": 0.0, "ret_miss": 0}
    beat = 0
    compared = 0
    for c in sorted(cat):
        s = cat[c]
        n = s["n"]
        f1 = s["f1"] / n
        r5 = s["r5"] / n
        r7 = s["r7"] / n
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
              f"{verdict:>9}{r5:>7.3f}{r7:>7.3f}{s['ret_miss']:>9}")
        for kk in tot:
            tot[kk] += s[kk]

    n = tot["n"]
    print("-" * len(hdr))
    print(f"{'OVERALL':<13}{n:>4}{tot['f1']/n:>8.3f}{'':>8}"
          f"{'':>9}{tot['r5']/n:>7.3f}{tot['r7']/n:>7.3f}{tot['ret_miss']:>9}")
    print(f"\nBeat HeLa on {beat}/{compared} compared categories.")
    print(f"Retrieval-bound QAs (gold not in top-7) : {tot['ret_miss']}/{n} "
          f"({100*tot['ret_miss']/n:.0f}%) — F1 ceiling from retrieval, "
          f"not generation.")


if __name__ == "__main__":
    main()
