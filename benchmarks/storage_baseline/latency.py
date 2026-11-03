"""Latencies p50 / p95 of `recall`, `remember` and `save()` (OO0-6, TAC2-1179).

    python -m benchmarks.storage_baseline.latency --work <dir> --out <latency.json>

On a FRESH COPY of a built (or copied) root — the source is never written
(checked by digest) — and through the production tools (`harness.Driver`,
same encoder, reranker and gate memory as the oracle):

* `recall` — the MCP `recall` tool (k=5, reranker on) on the 50 frozen
  queries of the context, each view; the first call is reported apart
  (`first_ms`: ONNX warm-up) and included in the percentiles;
* `retrieve` — the same queries, `retrieve(k=7, hybrid+lineage+spreading,
  rerank=False)`: the engine without the cross-encoder;
* `remember` — the MCP `remember` tool on N held-out chunks (never in any
  context), each followed by the `save()` the tool does in production (a
  narrow account saves both its store and the context's);
* `save` — `Memory.save()` alone, repeated, on the store as it stands after
  the remembers (a full pickle rewrite + fsync + rename, under the lock).

Percentiles are nearest-rank on the measured wall times (perf_counter, ms).
The host's load is written with the numbers: the container also runs the
API and the gate, so these are this machine's numbers, not tk-omni's.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from benchmarks.storage_baseline import corpus as _corpus
from benchmarks.storage_baseline import harness as H
from benchmarks.storage_baseline import queries as Q
from benchmarks.storage_baseline.run import view_key


def pct(values: List[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100) of a non-empty list."""
    s = sorted(values)
    k = max(1, math.ceil(p / 100.0 * len(s)))
    return s[k - 1]


def summary(ms: List[float]) -> Dict[str, Any]:
    return {"n": len(ms), "p50_ms": round(pct(ms, 50), 1),
            "p95_ms": round(pct(ms, 95), 1), "max_ms": round(max(ms), 1),
            "first_ms": round(ms[0], 1)}


def _timed(fn, *args: Any, **kwargs: Any) -> float:
    t = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - t) * 1000.0


def measure(work: str, n_remember: int = 30, n_save: int = 20,
            factory_spec: Optional[str] = None,
            tmp_parent: Optional[str] = None) -> Dict[str, Any]:
    work = os.path.abspath(work)
    before = H.tree_digest(work)
    tmp = tempfile.mkdtemp(prefix="oracle-lat-", dir=tmp_parent)
    held = [ch["text"] for f in _corpus.load()["roles"]["held_out"]
            for ch in f["chunks"]]
    out: Dict[str, Any] = {}
    try:
        shutil.copytree(os.path.join(work, "stores"), os.path.join(tmp, "stores"))
        if os.path.isdir(os.path.join(work, "notes")):
            shutil.copytree(os.path.join(work, "notes"), os.path.join(tmp, "notes"))
        d = H.Driver(os.path.join(tmp, "stores"),
                     os.path.join(tmp, "notes", H.TREE),
                     H.resolve_factory(factory_spec))
        for ctx in H.CONTEXTS:
            qs = [q["text"] for q in Q.load(ctx)["queries"]]
            for vctx, account in H.VIEWS[ctx]:
                vk = view_key(vctx, account)
                rec: List[float] = []
                ret: List[float] = []
                for q in qs:
                    with d.under(vctx, account):
                        rec.append(_timed(d.tool, "recall", query=q, k=5))
                    with d.under(vctx, account):
                        ret.append(_timed(d.memory.retrieve, q, k=7,
                                          use_hybrid=True, use_lineage=True,
                                          use_spreading=True, rerank=False))
                rem: List[float] = []
                for i in range(n_remember):
                    text = held[i % len(held)] + f" [latency probe {vk} #{i}]"
                    with d.under(vctx, account):
                        rem.append(_timed(d.tool, "remember", content=text,
                                          tags=["latency-probe"]))
                sav: List[float] = []
                with d.under(vctx, account):
                    mem = d.mem()
                    points = len(mem.points)
                    for _ in range(n_save):
                        sav.append(_timed(mem.save))
                    size = os.path.getsize(mem.storage_path) if mem.storage_path else None
                out[vk] = {"points_after": points, "pkl_bytes_after": size,
                           "recall": summary(rec), "retrieve": summary(ret),
                           "remember": summary(rem), "save": summary(sav)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if H.tree_digest(work) != before:
        raise SystemExit("[latency] the SOURCE changed during the run")
    if H.LLMGuard.calls:
        raise SystemExit(f"[latency] {H.LLMGuard.calls} LLM call(s)")
    return {"format": 1,
            "taken_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "host": os.uname().nodename, "cpus": os.cpu_count(),
            "loadavg_after": os.getloadavg(), "code_commit": H.git_commit(),
            "memory_factory": factory_spec or "harness:default_memory_factory",
            "rerank_threads": os.environ.get("METACOG_RERANK_THREADS", "4 (default)"),
            "views": out}


def markdown(r: Dict[str, Any]) -> str:
    lines = [f"Latences — {r['taken_at']} — `{r['host']}` ({r['cpus']} CPU, "
             f"charge après {', '.join(f'{x:.1f}' for x in r['loadavg_after'])}) — "
             f"`{r['code_commit'][:10]}`", "",
             "| vue | points | recall p50 / p95 (ms) | retrieve sans reranker p50 / p95 | "
             "remember p50 / p95 | save() p50 / p95 | pkl après (Mo) |",
             "| -- | -- | -- | -- | -- | -- | -- |"]
    for vk, v in r["views"].items():
        def f(k: str) -> str:
            return f"{v[k]['p50_ms']} / {v[k]['p95_ms']}"
        lines.append(f"| `{vk}` | {v['points_after']} | {f('recall')} | {f('retrieve')} | "
                     f"{f('remember')} | {f('save')} | "
                     f"{(v['pkl_bytes_after'] or 0) / 1e6:.2f} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="recall / remember / save latencies")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remember", type=int, default=30)
    ap.add_argument("--save", type=int, default=20)
    ap.add_argument("--memory-factory", default=None)
    ap.add_argument("--tmp", default=None)
    a = ap.parse_args(argv)
    r = measure(a.work, a.remember, a.save, a.memory_factory, a.tmp)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1, sort_keys=True)
        f.write("\n")
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
