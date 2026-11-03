"""Run the frozen recall oracle on a built (or copied) storage root (OO0-5).

    python -m benchmarks.storage_baseline.run --work <dir> --out <result.json>

`<dir>` is what `build.py` wrote (`stores/`, `notes/oracle/`). The run works
on a FRESH COPY of it (a temporary directory, removed afterwards): a recall
advances the logical clock and appends to the journal, and the source must
stay byte-identical — checked, file by file, before and after.

For every context, every view (the narrow context is asked twice: as the
context's own account and as the narrow account `oracle-member`) and every
frozen query, in a fixed order, three probes:

* `retrieve_plain` — `retrieve(q, k=7, rerank=False)`: cosine on the
  EFFECTIVE vector (embedding_orig + time-decayed deltas), the geometry a
  storage backend must reproduce;
* `retrieve` — `retrieve(q, k=7, use_hybrid, use_lineage, use_spreading,
  rerank=False)`: the MCP `retrieve` defaults minus the cross-encoder
  (keywords, BM25, RRF, lineage, spreading);
* `recall` — the MCP `recall` tool (k=5): production, reranker, relevance
  floor, gap notice, journal write.

Each hit is kept as id, score, rerank score, kind, state and a digest of the
content. Wall-clock fields (`spread_ms`, `rerank_ms`) and the journal's
`retrieval_id` are dropped: they are not what a backend must preserve.

INJECTION: `--memory-factory module:callable` builds the memory from
`(storage_root, notes_root)` — see `harness.py`. The output names it.

POSITIVE WITNESS: `--drop-from-baseline <result.json>` removes, before any
query, one point per view — the first `recall` hit of the first query that
had one in that baseline — so `compare.py` MUST then say "different".
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from benchmarks.storage_baseline import harness as H
from benchmarks.storage_baseline import queries as Q

FORMAT = 1
#: Keys of a hit that are NOT part of the oracle (wall time, journal handles).
VOLATILE = {"spread_ms", "rerank_ms", "retrieval_id", "content", "confidence",
            "n_corrob", "n_contra"}


def _hit(h: Dict[str, Any]) -> Dict[str, Any]:
    if h.get("gap") or h.get("abstained"):
        return {"gap": True, "abstained": bool(h.get("abstained")),
                **{k: h[k] for k in ("pool_size", "rerank_n") if k in h}}
    out: Dict[str, Any] = {"id": h.get("id")}
    for k in ("score", "rerank_score"):
        if h.get(k) is not None:
            out[k] = float(h[k])
    for k in ("kind", "state", "pool_size", "rerank_n"):
        if k in h:
            out[k] = h[k]
    if h.get("content") is not None:
        out["content_sha12"] = hashlib.sha256(
            str(h["content"]).encode()).hexdigest()[:12]
    return out


def view_key(ctx: str, account: str) -> str:
    return ctx + (f"/accounts/{account}" if account else "")


def _drop(memory: Any, pid: str) -> bool:
    before = len(memory.points)
    memory.points = [p for p in memory.points if p.id != pid]
    from metacog.geometry import clear_geo_cache
    from metacog.text_index import TextIndex
    clear_geo_cache()
    memory._text_index = TextIndex()
    return len(memory.points) < before


def _witness_ids(baseline: str) -> Dict[str, str]:
    with open(baseline, encoding="utf-8") as f:
        base = json.load(f)
    picks: Dict[str, str] = {}
    for ctx in base["contexts"].values():
        for vk, view in ctx["views"].items():
            for qid in sorted(view["results"]):
                hits = [h for h in view["results"][qid]["recall"] if h.get("id")]
                if hits:
                    picks[vk] = hits[0]["id"]
                    break
    return picks


def run(work: str, factory_spec: Optional[str] = None,
        drop_from: Optional[str] = None, tmp_parent: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        allow_no_manifest: bool = False) -> Dict[str, Any]:
    work = os.path.abspath(work)
    src_stores = os.path.join(work, "stores")
    src_notes = os.path.join(work, "notes")
    if not os.path.isdir(src_stores):
        raise SystemExit(f"[run] {src_stores} missing: build first")
    if not allow_no_manifest and not os.path.exists(
            os.path.join(work, "build_manifest.json")):
        raise SystemExit(f"[run] {work} has no build_manifest.json: an "
                         "unfinished build is not a baseline (pass "
                         "--no-manifest for a copied served root)")
    digest_before = H.tree_digest(work)
    tmp = tempfile.mkdtemp(prefix="oracle-run-", dir=tmp_parent)
    try:
        stores = os.path.join(tmp, "stores")
        notes_root = os.path.join(tmp, "notes", H.TREE)
        shutil.copytree(src_stores, stores, symlinks=True)
        if os.path.isdir(src_notes):
            shutil.copytree(src_notes, os.path.join(tmp, "notes"), symlinks=True)
        d = H.Driver(stores, notes_root, H.resolve_factory(factory_spec))
        drops = _witness_ids(drop_from) if drop_from else {}
        result_ctx: Dict[str, Any] = {}
        t0 = time.time()
        for ctx in contexts or list(H.CONTEXTS):
            qs = Q.load(ctx)
            qsha = hashlib.sha256(json.dumps(qs, sort_keys=True).encode()).hexdigest()
            views: Dict[str, Any] = {}
            for vctx, account in H.VIEWS[ctx]:
                vk = view_key(vctx, account)
                with d.under(vctx, account):
                    mem = d.mem()
                    dropped = None
                    if vk in drops:
                        target = drops[vk]
                        # A narrow view's hit may be a NOTE of the context memory.
                        if not _drop(mem, target):
                            with d.under(vctx):
                                _drop(d.mem(), target)
                        dropped = target
                    fp = H.store_fingerprint(mem)
                results: Dict[str, Any] = {}
                for q in qs["queries"]:
                    with d.under(vctx, account):
                        plain = d.memory.retrieve(q["text"], k=7, rerank=False)
                    with d.under(vctx, account):
                        full = d.memory.retrieve(q["text"], k=7, use_hybrid=True,
                                                 use_lineage=True,
                                                 use_spreading=True, rerank=False)
                    with d.under(vctx, account):
                        rec = d.tool("recall", query=q["text"], k=5)
                    results[q["qid"]] = {
                        "retrieve_plain": [_hit(h) for h in plain],
                        "retrieve": [_hit(h) for h in full],
                        "recall": [_hit(h) for h in rec],
                    }
                views[vk] = {"fingerprint": fp, "dropped": dropped,
                             "results": results}
            result_ctx[ctx] = {"queries_version": qs["version"],
                               "queries_sha256": qsha,
                               "n_queries": len(qs["queries"]), "views": views}
        elapsed = time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    digest_after = H.tree_digest(work)
    if digest_after != digest_before:
        changed = sorted(k for k in set(digest_before) | set(digest_after)
                         if digest_before.get(k) != digest_after.get(k))
        raise SystemExit(f"[run] the SOURCE changed during the run: {changed}")
    enc, rr = H.shared_models()
    from metacog.defaults import encoder_id
    manifest = None
    mp = os.path.join(work, "build_manifest.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            manifest = json.load(f)
    out = {
        "format": FORMAT,
        "oracle": "storage_baseline",
        "run": {"taken_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "host": os.uname().nodename, "code_commit": H.git_commit(),
                "memory_factory": factory_spec or "harness:default_memory_factory",
                "encoder_id": encoder_id(enc),
                "reranker": getattr(rr, "model_name", type(rr).__name__),
                "llm_calls": H.LLMGuard.calls, "elapsed_s": round(elapsed, 1),
                "witness_drop_from": drop_from},
        "build": manifest,
        "contexts": result_ctx,
    }
    if H.LLMGuard.calls:
        raise SystemExit(f"[run] {H.LLMGuard.calls} LLM call(s): the oracle must be LLM-free")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="run the storage-baseline oracle")
    ap.add_argument("--work", required=True, help="directory written by build.py")
    ap.add_argument("--out", required=True, help="result JSON path")
    ap.add_argument("--memory-factory", default=None,
                    help="module:callable (storage_root, notes_root) -> memory")
    ap.add_argument("--drop-from-baseline", default=None,
                    help="positive witness: drop one served point per view")
    ap.add_argument("--tmp", default=None, help="parent dir of the run copy")
    ap.add_argument("--ctx", action="append", help="restrict to these contexts")
    ap.add_argument("--no-manifest", action="store_true",
                    help="accept a root without build_manifest.json")
    a = ap.parse_args(argv)
    out = run(a.work, a.memory_factory, a.drop_from_baseline, a.tmp, a.ctx,
              a.no_manifest)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    n = sum(c["n_queries"] * len(c["views"]) for c in out["contexts"].values())
    print(f"[run] {n} (view, query) pairs x 3 probes in {out['run']['elapsed_s']} s "
          f"-> {a.out} (llm_calls={out['run']['llm_calls']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
