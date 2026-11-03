"""Compare two oracle results: IDENTICAL, or DIFFERENT and where (OO0-5).

    python -m benchmarks.storage_baseline.compare <baseline.json> <candidate.json>
        [--atol 0] [--max-lines 40] [--json <report.json>]

Exit code: 0 identical, 1 different, 2 not comparable (oracle format,
query sets or context/view lists differ — a different question, not a
different answer).

What "identical" means, probe by probe (`retrieve_plain`, `retrieve`,
`recall`) and query by query: the same hits in the same order (ids, kinds,
states, content digests, gap notices) and the same scores (`score`,
`rerank_score`) — bit-for-bit by default, or within `--atol` when a backend
is KNOWN to round (e.g. float32 vectors), which the report then says.

The build fingerprints (content / vector digests of every store) are
compared too, and reported apart: a vector digest that differs while the
content digest is equal is a backend that rounds or re-encodes; a content
digest that differs is a different build, not a storage difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

PROBES = ("retrieve_plain", "retrieve", "recall")
SCORES = ("score", "rerank_score")


def _load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _hits_diff(a: List[Dict[str, Any]], b: List[Dict[str, Any]],
               atol: float) -> Tuple[List[str], float]:
    """Differences between two hit lists, and the max score delta seen."""
    why: List[str] = []
    max_delta = 0.0
    ida = [h.get("id") if not h.get("gap") else "<gap>" for h in a]
    idb = [h.get("id") if not h.get("gap") else "<gap>" for h in b]
    if ida != idb:
        gone = [i for i in ida if i not in idb]
        new = [i for i in idb if i not in ida]
        why.append(f"ids {ida} -> {idb}"
                   + (f" (lost {gone}, gained {new})" if gone or new else " (reordered)"))
        return why, max_delta
    for ha, hb in zip(a, b):
        for k in ("kind", "state", "content_sha12", "abstained"):
            if ha.get(k) != hb.get(k):
                why.append(f"{ha.get('id')}: {k} {ha.get(k)!r} -> {hb.get(k)!r}")
        for k in SCORES:
            va, vb = ha.get(k), hb.get(k)
            if (va is None) != (vb is None):
                why.append(f"{ha.get('id')}: {k} {va!r} -> {vb!r}")
            elif va is not None:
                delta = abs(float(va) - float(vb))
                max_delta = max(max_delta, delta)
                if delta > atol:
                    why.append(f"{ha.get('id')}: {k} {va!r} -> {vb!r} (Δ {delta:.3g})")
    return why, max_delta


def compare(base: Dict[str, Any], cand: Dict[str, Any],
            atol: float = 0.0) -> Dict[str, Any]:
    report: Dict[str, Any] = {"verdict": "identical", "atol": atol,
                              "incomparable": [], "build": [], "diffs": [],
                              "counts": {}, "max_score_delta": 0.0}
    if base.get("format") != cand.get("format"):
        report["incomparable"].append(
            f"format {base.get('format')} vs {cand.get('format')}")
    cb, cc = base.get("contexts", {}), cand.get("contexts", {})
    if sorted(cb) != sorted(cc):
        report["incomparable"].append(f"contexts {sorted(cb)} vs {sorted(cc)}")
    compared = differing = 0
    for ctx in sorted(set(cb) & set(cc)):
        a, b = cb[ctx], cc[ctx]
        if a.get("queries_sha256") != b.get("queries_sha256"):
            report["incomparable"].append(f"{ctx}: query sets differ")
            continue
        if sorted(a["views"]) != sorted(b["views"]):
            report["incomparable"].append(
                f"{ctx}: views {sorted(a['views'])} vs {sorted(b['views'])}")
            continue
        for vk in sorted(a["views"]):
            va, vb = a["views"][vk], b["views"][vk]
            fa, fb = va.get("fingerprint", {}), vb.get("fingerprint", {})
            for k in ("points", "content_sha256", "vectors_sha256"):
                if fa.get(k) != fb.get(k):
                    report["build"].append(f"{vk}: {k} {fa.get(k)} -> {fb.get(k)}")
            if va.get("dropped") or vb.get("dropped"):
                report["build"].append(
                    f"{vk}: witness drop {va.get('dropped')} -> {vb.get('dropped')}")
            n_view = 0
            for qid in sorted(va["results"]):
                for probe in PROBES:
                    compared += 1
                    why, delta = _hits_diff(va["results"][qid][probe],
                                            vb["results"][qid][probe], atol)
                    report["max_score_delta"] = max(report["max_score_delta"], delta)
                    if why:
                        differing += 1
                        n_view += 1
                        report["diffs"].append({"view": vk, "qid": qid,
                                                "probe": probe, "why": why})
            report["counts"][vk] = n_view
    report["compared"] = compared
    report["differing"] = differing
    if report["incomparable"]:
        report["verdict"] = "incomparable"
    elif differing:
        report["verdict"] = "different"
    return report


def render(report: Dict[str, Any], max_lines: int = 40) -> str:
    v = report["verdict"]
    head = {"identical": "IDENTIQUE", "different": "DIFFÉRENT",
            "incomparable": "NON COMPARABLE"}[v]
    lines = [f"{head} — {report.get('compared', 0)} (vue, requête, sonde) comparées, "
             f"{report.get('differing', 0)} différentes ; écart de score max "
             f"{report['max_score_delta']:.3g} (atol {report['atol']})"]
    for item in report["incomparable"]:
        lines.append(f"  non comparable : {item}")
    for item in report["build"]:
        lines.append(f"  build : {item}")
    for vk, n in sorted(report["counts"].items()):
        if n:
            lines.append(f"  {vk} : {n} différence(s)")
    for d in report["diffs"][:max_lines]:
        lines.append(f"  {d['view']} {d['qid']} {d['probe']} : " + " ; ".join(d["why"])[:400])
    if len(report["diffs"]) > max_lines:
        lines.append(f"  … {len(report['diffs']) - max_lines} de plus (--json pour tout)")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="compare two oracle results")
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--max-lines", type=int, default=40)
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args(argv)
    rep = compare(_load(a.baseline), _load(a.candidate), a.atol)
    print(render(rep, a.max_lines))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    return {"identical": 0, "different": 1, "incomparable": 2}[rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
