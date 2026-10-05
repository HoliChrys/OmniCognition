"""D3 bench (Linear TAC-939) — the `sleep` and `walk` deep-wiki strategies,
compared per context on ONE corpus. Protocol: PROTOCOL.md (frozen before any
run); questions: questions.yaml (frozen, notes pinned by sha256).

Run on a host that holds the notes and stores, with the omni venv:

    python -m benchmarks.d3_wiki.run_d3 --scratch /tmp/d3-run \
        [--ctx global] [--out results/]

Never writes a live store: every store and notes folder is COPIED into
`--scratch` first, and an audit hook REFUSES (and records) every write the
bench process attempts under the live roots — a recorded write fails the run.
The sha256 of the live `memory.pkl` / journal is taken before and after, for
information only: the served gate legitimately writes them meanwhile
(`live_store_changed_by_other_process`). Decision Proxy, TAC-209, 2026-10-05.

The LLM is real or the bench does not start: one control call first, every
client failure counted (`ClaudeLLM.llm_errors`), at most `--llm-cap` calls
per context (beyond, the context stops and says so), and `build(sleep)`
capped at `--sleep-build-cap-s` (beyond, sleep is ineligible on that context
for build cost — a result, not a failure).

Pure parts (`cited_note`, `score`, `decide`) and the guards are unit-tested
in tests/test_d3_wiki_bench.py.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STRATEGIES = ("walk", "sleep")
#: Not a candidate of the decision rule: what the deployed API recall does
#: today (tachikoma OmniEngine.recall -> `retrieve`) on the un-slept store.
REFERENCE = "deployed"
NOISE_MAX = 0.05            # Proxy's rule: noise <= 5 % of answers
QUERY_P95_MAX_S = 2.0       # E1 budget: p95 <= 2 s
COVERAGE_TIE_POINTS = 10.0  # gap under 10 points -> walk (omni, active)
STORE_FILES = ("memory.pkl", "memory.pkl.journal.db")
LLM_CAP = 1000              # Proxy, TAC-209: LLM calls per context
SLEEP_BUILD_CAP_S = 5400    # Proxy, TAC-209: 90 min for build(sleep)
CONTROL_PROMPT = "Reply with the single word OK."


# ── guards: no write under the live roots, a real and bounded LLM ───────

_WRITE_FLAGS = (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC
                | getattr(os, "O_APPEND", 0))
#: Path-mutating audit events, besides `open` and `sqlite3.connect`.
_MUTATING_EVENTS = frozenset({
    "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.truncate",
    "os.chmod", "os.chown", "os.utime", "os.link", "os.symlink",
    "shutil.rmtree", "shutil.move"})
_GUARD: Dict[str, Any] = {"roots": (), "violations": None, "installed": False}
_IN_HOOK = threading.local()


def _path_of(arg: Any) -> Optional[str]:
    if arg is None or isinstance(arg, int):
        return None
    try:
        p = os.fsdecode(arg)
    except TypeError:
        return None
    if p.startswith("file:"):                       # sqlite URI
        p = p[len("file:"):].split("?", 1)[0]
    return p if p and p != ":memory:" else None


def _under_live(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    real = os.path.realpath(path)
    for root in _GUARD["roots"]:
        if real == root or real.startswith(root + os.sep):
            return real
    return None


def _audit(event: str, args: tuple) -> None:
    if not _GUARD["roots"] or getattr(_IN_HOOK, "on", False):
        return
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                 (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        if not writes:
            return
        candidates = [path]
    elif event == "sqlite3.connect":
        if args and isinstance(args[0], str) and "mode=ro" in args[0]:
            return
        candidates = list(args[:1])
    elif event in _MUTATING_EVENTS:
        candidates = list(args)
    else:
        return
    _IN_HOOK.on = True
    try:
        for c in candidates:
            hit = _under_live(_path_of(c))
            if hit is not None:
                _GUARD["violations"].append({"event": event, "path": hit})
                raise PermissionError(f"D3 bench: write under a live root refused "
                                      f"({event} {hit})")
    finally:
        _IN_HOOK.on = False


@contextmanager
def live_write_guard(roots: List[str]) -> Iterator[List[dict]]:
    """Refuse and record every write this process attempts under `roots`
    (audit hook on `open` in a write mode/flag, `sqlite3.connect` without
    `mode=ro`, and path-mutating os/shutil events). Yields the list of
    violations; a refused write is ALSO raised at the call site, but a
    failure-safe caller may swallow it — the list is the verdict."""
    if not _GUARD["installed"]:
        sys.addaudithook(_audit)                    # cannot be removed: idle
        _GUARD["installed"] = True                  # once `roots` is empty
    violations: List[dict] = []
    _GUARD["roots"] = tuple(sorted({os.path.realpath(r) for r in roots}))
    _GUARD["violations"] = violations
    try:
        yield violations
    finally:
        _GUARD["roots"], _GUARD["violations"] = (), None


class LLMCapReached(RuntimeError):
    """The per-context LLM call budget is spent: the call is NOT made."""


class BenchStopped(Exception):
    """A context stops early (LLM cap); the reason goes in the summary."""


class _BuildCapExceeded(BaseException):
    """Raised by SIGALRM inside build(sleep). BaseException so that no
    failure-safe `except Exception` of the library swallows it."""


def new_budget(cap: Optional[int]) -> Dict[str, Any]:
    return {"cap": cap, "used": 0, "errors": 0, "refused": 0, "reached": False}


def control_call(llm: Any) -> dict:
    """One real LLM call before anything else; SystemExit if it fails or
    answers nothing (the D3 run of 2026-10-05 measured 0 errors on an LLM
    that never answered)."""
    from metacog.llm import MissingCredential
    t0 = time.perf_counter()
    try:
        out = llm.generate(CONTROL_PROMPT, max_tokens=16)
    except MissingCredential as exc:
        raise SystemExit(f"LLM control call failed: {exc}")
    errors = getattr(llm, "llm_errors", 0)
    if errors or not (out or "").strip():
        raise SystemExit("LLM control call failed: "
                         f"{getattr(llm, 'last_error', None) or 'empty answer'}")
    return {"ok": True, "model": getattr(llm, "model", None),
            "answer": out.strip()[:40], "seconds": round(time.perf_counter() - t0, 2),
            "endpoint": urlparse(os.environ.get("ANTHROPIC_BASE_URL") or "").netloc
            or "api.anthropic.com"}


# ── pure scoring ────────────────────────────────────────────────────────

def cited_note(item_id: str, tags: Optional[List[str]] = None) -> Optional[str]:
    """The note doc id an item cites, or None. `notes:d#…` or tag `note:notes:d`."""
    if item_id.startswith("notes:"):
        return item_id.split("#", 1)[0]
    for t in tags or ():
        if t.startswith("note:notes:"):
            return t[len("note:"):]
    return None


def _pct(xs: List[float], q: float) -> float:
    """Nearest-rank percentile (no interpolation: a p95 is a measured value)."""
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, -(-int(q * 100) * len(s) // 100) - 1))]


def score(records: List[dict], questions: dict) -> dict:
    """Per (ctx, strategy): coverage, noise, off-topic items, query cost.

    A record: {ctx, strategy, qid, offtopic, items: [{id, note}], seconds,
    invalid_offtopic?}. In-topic answers count for coverage/noise; off-topic
    ones for the 0-item condition; every timed answer (both kinds) for cost.
    """
    out: Dict[str, Dict[str, dict]] = {}
    for r in records:
        cell = out.setdefault(r["ctx"], {}).setdefault(r["strategy"], {
            "n_intopic": 0, "covered": 0, "noisy": 0, "items": 0,
            "items_on_relevant": 0, "items_uncited": 0,
            "offtopic_asked": 0, "offtopic_items": 0, "offtopic_invalid": [],
            "query_llm_calls": 0, "query_llm_errors": 0, "seconds": []})
        cell["seconds"].append(r["seconds"])
        # An LLM error is a DEGRADED answer (rule 6): counted, never hidden.
        cell["query_llm_calls"] += r.get("llm_calls", 0)
        cell["query_llm_errors"] += r.get("llm_errors", 0)
        if r["offtopic"]:
            if r.get("invalid_offtopic"):
                cell["offtopic_invalid"].append(r["qid"])
                continue
            cell["offtopic_asked"] += 1
            cell["offtopic_items"] += len(r["items"])
            continue
        q = next(x for x in questions["contexts"][r["ctx"]]["questions"]
                 if x["id"] == r["qid"])
        notes = [i["note"] for i in r["items"]]
        cell["n_intopic"] += 1
        cell["items"] += len(notes)
        cell["items_uncited"] += sum(1 for n in notes if n is None)
        cell["items_on_relevant"] += sum(1 for n in notes if n in q["relevant"])
        if any(n in q["expected"] for n in notes):
            cell["covered"] += 1
        if any(n is not None and n not in q["relevant"] for n in notes):
            cell["noisy"] += 1
    for strategies in out.values():
        for c in strategies.values():
            n = c["n_intopic"]
            c["coverage"] = round(100.0 * c["covered"] / n, 1) if n else 0.0
            c["noise"] = round(c["noisy"] / n, 3) if n else 0.0
            cited = c["items"] - c["items_uncited"]
            c["precision_cited"] = (round(c["items_on_relevant"] / cited, 3)
                                    if cited else None)
            secs = c.pop("seconds")
            c["query_n"] = len(secs)
            c["query_p50_s"] = round(statistics.median(secs), 2) if secs else None
            c["query_p95_s"] = round(_pct(secs, 0.95), 2) if secs else None
            c["query_max_s"] = round(max(secs), 2) if secs else None
    return out


def decide(scores: Dict[str, Dict[str, dict]],
           ineligible: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, dict]:
    """Proxy's rule (TAC-190 decisions doc, 2026-10-04), per context.

    `ineligible` = {ctx: {strategy: reason}} set by the run itself (build cap
    exceeded, LLM cap reached — TAC-209): the rule then applies among the
    strategies that finished."""
    verdicts = {}
    for ctx in sorted(set(scores) | set(ineligible or {})):
        cells = scores.get(ctx, {})
        eligible, breaches = {}, {}
        for s in STRATEGIES:
            c = cells.get(s)
            if s in (ineligible or {}).get(ctx, {}):
                breaches[s] = [(ineligible or {})[ctx][s]]
                continue
            if c is None:
                breaches[s] = ["not measured"]
                continue
            why = []
            if c["noise"] > NOISE_MAX:
                why.append(f"noise {c['noise']:.1%} > 5 %")
            if c["offtopic_items"]:
                why.append(f"{c['offtopic_items']} item(s) on off-topic questions")
            if c["query_p95_s"] is None or c["query_p95_s"] > QUERY_P95_MAX_S:
                why.append(f"query p95 {c['query_p95_s']} s > 2 s")
            if why:
                breaches[s] = why
            else:
                eligible[s] = c["coverage"]
        if not eligible:
            kept, reason = "walk", "no strategy meets both conditions -> omni"
        elif len(eligible) == 1:
            kept = next(iter(eligible))
            reason = "the only strategy meeting both conditions"
        elif abs(eligible["sleep"] - eligible["walk"]) < COVERAGE_TIE_POINTS:
            kept, reason = "walk", "coverage gap under 10 points -> omni"
        else:
            kept = max(eligible, key=eligible.get)
            reason = "best coverage among eligible strategies"
        verdicts[ctx] = {"strategy": kept, "reason": reason,
                         "eligible": eligible, "breaches": breaches}
    return verdicts


# ── the corpus: pinned notes, copied stores ─────────────────────────────

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def note_manifest(folder: str) -> Dict[str, str]:
    """{doc_id: sha256} of every .md under `folder` — the LINK is followed
    (TAC-327: the notes folder is a symlink; a bare walk sees nothing)."""
    real = os.path.realpath(folder)
    out = {}
    for root, _dirs, files in os.walk(real):
        for f in files:
            if f.endswith(".md"):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, real)[:-3].replace(os.sep, "/")
                out[f"notes:{rel}"] = sha256_file(p)
    return out


def check_corpus(folder: str, pinned: Dict[str, str]) -> None:
    """Fail-closed: the questions were written on THESE notes."""
    got = note_manifest(folder)
    if got != pinned:
        missing = sorted(set(pinned) - set(got))
        extra = sorted(set(got) - set(pinned))
        changed = sorted(d for d in set(got) & set(pinned) if got[d] != pinned[d])
        raise SystemExit(f"corpus of {folder} differs from questions.yaml pins: "
                         f"missing={missing} extra={extra} changed={changed}")


class CountingLLM:
    """Counts every LLM call and its failures for the build/query cost, and
    spends the context's shared `budget`. A failure is an exception OR a
    client error the inner LLM absorbed into "" (`llm_errors` grew)."""

    def __init__(self, inner: Any, budget: Optional[Dict[str, Any]] = None):
        self._inner, self.calls, self.errors = inner, 0, 0
        self._budget = budget if budget is not None else new_budget(None)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def counted(*a, **kw):
            b = self._budget
            if b["cap"] is not None and b["used"] >= b["cap"]:
                b["reached"], b["refused"] = True, b["refused"] + 1
                raise LLMCapReached(f"{b['cap']} LLM calls spent on this context")
            b["used"] += 1
            self.calls += 1
            before = getattr(self._inner, "llm_errors", 0)
            failed = 0
            try:
                return attr(*a, **kw)
            except Exception:
                failed = 1
                raise
            finally:
                failed += max(0, getattr(self._inner, "llm_errors", 0) - before)
                self.errors += failed
                b["errors"] += failed
        return counted


@contextmanager
def _time_cap(seconds: Optional[float]) -> Iterator[None]:
    """SIGALRM after `seconds` raises _BuildCapExceeded in the main thread
    (where build runs). None or 0 = no cap."""
    if not seconds:
        yield
        return

    def _fire(_sig, _frame):
        raise _BuildCapExceeded()
    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def build(strategy: str, ctx: str, notes_src: str, store_src: Optional[str],
          scratch: str, encoder: Any, reranker: Any, llm: Any,
          budget: Optional[Dict[str, Any]] = None,
          cap_s: Optional[float] = None) -> tuple:
    """A memory for `strategy` on a scratch copy; returns (memory, cost),
    or (None, cost) with `cost["cap_exceeded_s"]` when a capped build ran
    past `cap_s` seconds (only `sleep` is capped — Proxy, TAC-209).

    Both strategies ingest the notes with the gate's own pass
    (`ContextualMemory._refresh_notes`: doc via import_okf + content point).
    `sleep` then seeds one query per note (its first heading) and runs
    `Memory.sleep()` (reconcile_wiki -> rerun_seeds -> absorb). `walk` and the
    reference stop after the ingest, as the deployed gate does.
    """
    counting = CountingLLM(llm, budget)
    t0 = time.perf_counter()
    try:
        with _time_cap(cap_s if strategy == "sleep" else None):
            return _build(strategy, ctx, notes_src, store_src, scratch,
                          encoder, reranker, counting)
    except _BuildCapExceeded:
        return None, {"cap_exceeded_s": cap_s,
                      "elapsed_s": round(time.perf_counter() - t0, 2),
                      "build_llm_calls": counting.calls,
                      "build_llm_errors": counting.errors}


def _build(strategy: str, ctx: str, notes_src: str, store_src: Optional[str],
           scratch: str, encoder: Any, reranker: Any, counting: CountingLLM) -> tuple:
    from metacog.memory import Memory
    from metacog.tachikoma_gate import ContextualMemory, notes_folder

    base = os.path.join(scratch, f"{ctx}.{strategy}.{'live' if store_src else 'empty'}")
    store_root, notes_root = os.path.join(base, "store"), os.path.join(base, "tachikoma")
    folder = notes_folder(notes_root, ctx)
    shutil.copytree(os.path.realpath(notes_src), folder)
    os.makedirs(os.path.join(store_root, ctx), exist_ok=True)
    path = os.path.join(store_root, ctx, "memory.pkl")
    if store_src:
        for f in STORE_FILES:
            src = os.path.join(store_src, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(store_root, ctx, f))
    t0 = time.perf_counter()
    m = Memory(storage_path=path, journal_path="auto", encoder=encoder,
               reranker=reranker, llm=counting)
    cost: Dict[str, Any] = {"load_s": round(time.perf_counter() - t0, 2),
                            "points_loaded": len(m.points)}
    t1 = time.perf_counter()
    report = ContextualMemory(store_root, notes_root)._refresh_notes(ctx, m)
    if report["state"] != "ok" or report["errors"]:
        raise SystemExit(f"{ctx}/{strategy}: notes not ingested: {report}")
    cost["ingest"] = {k: (len(v) if isinstance(v, list) else v)
                      for k, v in report.items() if k not in ("ctx", "folder")}
    if strategy == "sleep":
        seeded = 0
        for doc_id in sorted(note_manifest(folder)):
            doc = m.wiki_doc(doc_id) or ""
            head = next((ln.lstrip("# ").strip() for ln in doc.splitlines()
                         if ln.startswith("#")), doc_id.split("/")[-1])
            if m.add_seed(doc_id, head, target="*").get("seed_id"):
                seeded += 1
        cost["seeds"] = seeded
        cost["sleep_report"] = {k: v for k, v in m.sleep().items()
                                if isinstance(v, (int, float, str, bool))}
        m.save()
    cost["build_s"] = round(time.perf_counter() - t1, 2)
    cost["build_llm_calls"], cost["build_llm_errors"] = counting.calls, counting.errors
    counting.calls = counting.errors = 0
    m._d3_llm = counting
    return m, cost


# ── answering ───────────────────────────────────────────────────────────

ANSWER_TOOL = {"walk": "walk_start", "sleep": "recall", REFERENCE: "recall"}


def _items(strategy: str, payload: Any, m: Any) -> List[dict]:
    if strategy == "walk":
        raw = [e["id"] for e in (payload.get("relevant_collected") or [])]
    else:
        raw = [e["id"] for e in payload if isinstance(e, dict) and "id" in e
               and not e.get("abstained") and not e.get("gap")]
    tags = {p.id: list(p.tags) for p in m.points}
    return [{"id": i, "note": cited_note(i, tags.get(i))} for i in raw]


async def answer_all(strategy: str, m: Any, asks: List[tuple]) -> List[dict]:
    """Every question through the MCP tool of the strategy (surface external,
    as deployed), one uncounted warm-up first."""
    from mcp.shared.memory import create_connected_server_and_client_session
    from metacog.mcp_server import build_app

    tool = ANSWER_TOOL[strategy]
    arg = (lambda q: {"query": q}) if tool == "walk_start" else (lambda q: {"query": q, "k": 5})
    out: List[dict] = []
    stop: Optional[BaseException] = None     # raised AFTER the session closes:
    async with create_connected_server_and_client_session(   # inside, anyio
            build_app(memory=m, surface="external")) as s:   # wraps it in a group
        await s.initialize()
        await s.call_tool(tool, arg("warm-up"))
        for qid, q, offtopic, invalid in asks:
            m._d3_llm.calls = m._d3_llm.errors = 0
            t0 = time.perf_counter()
            r = await s.call_tool(tool, arg(q))
            secs = time.perf_counter() - t0
            if m._d3_llm._budget["reached"]:
                stop = BenchStopped(f"LLM cap reached while answering "
                                    f"{strategy} {qid}")
                break
            if r.isError:
                stop = SystemExit(f"{strategy} {qid}: tool error {r.content}")
                break
            payload = json.loads(r.content[0].text) if len(r.content) == 1 else \
                [json.loads(c.text) for c in r.content]
            if strategy != "walk" and isinstance(payload, dict):
                payload = [payload]
            out.append({"strategy": strategy, "qid": qid, "offtopic": offtopic,
                        "invalid_offtopic": invalid, "seconds": round(secs, 3),
                        "llm_calls": m._d3_llm.calls, "llm_errors": m._d3_llm.errors,
                        "items": _items(strategy, payload, m)})
    if stop is not None:
        raise stop
    return out


# ── main ────────────────────────────────────────────────────────────────

def live_roots(spec: dict) -> List[str]:
    """What the bench must never write: the live stores' root (the parent of
    the context's store, i.e. every context) and the source notes."""
    return [os.path.dirname(os.path.realpath(spec["store_source"])),
            os.path.realpath(spec["notes_source"])]


def run_context(ctx: str, spec: dict, offtopic: List[dict], scratch: str,
                models: tuple, llm_factory: Any, check_pins: bool = True,
                llm_cap: Optional[int] = LLM_CAP,
                sleep_cap_s: Optional[float] = SLEEP_BUILD_CAP_S) -> dict:
    if check_pins:
        check_corpus(spec["notes_source"], spec["corpus"])
    live = [os.path.join(spec["store_source"], f) for f in STORE_FILES
            if os.path.exists(os.path.join(spec["store_source"], f))]
    before = {p: sha256_file(p) for p in live}
    encoder, reranker = models
    budget = new_budget(llm_cap)
    result: Dict[str, Any] = {"ctx": ctx, "build_empty": {}, "build_live": {},
                              "ineligible": {}, "stopped": None, "records": []}

    def _cap_check(where: str) -> None:
        if budget["reached"]:
            raise BenchStopped(f"LLM cap reached during {where}")

    with live_write_guard(live_roots(spec)) as violations:
        try:
            for strategy in STRATEGIES:      # build cost from an EMPTY store
                m, cost = build(strategy, ctx, spec["notes_source"], None, scratch,
                                encoder, reranker, llm_factory(), budget, sleep_cap_s)
                result["build_empty"][strategy] = cost
                del m
                _cap_check(f"build_empty({strategy})")
                if "cap_exceeded_s" in cost:
                    result["ineligible"][strategy] = (
                        f"build({strategy}) exceeded the {cost['cap_exceeded_s']} s cap "
                        "(build cost)")
            for strategy in STRATEGIES:      # answers on a COPY of the live store
                if strategy in result["ineligible"]:
                    continue
                m, cost = build(strategy, ctx, spec["notes_source"],
                                spec["store_source"], scratch, encoder, reranker,
                                llm_factory(), budget, sleep_cap_s)
                result["build_live"][strategy] = cost
                _cap_check(f"build_live({strategy})")
                if m is None:
                    result["ineligible"][strategy] = (
                        f"build({strategy}) exceeded the {cost['cap_exceeded_s']} s cap "
                        "(build cost)")
                    continue
                asks = [(q["id"], q["q"], False, False) for q in spec["questions"]]
                for o in offtopic:
                    pat = re.compile(o["absent_marker"])
                    held = any(pat.search(p.content or "") for p in m.points)
                    asks.append((o["id"], o["q"], True, held))
                strategies = [strategy] + ([REFERENCE] if strategy == "walk" else [])
                for s in strategies:
                    for r in asyncio.run(answer_all(s, m, asks)):
                        result["records"].append({"ctx": ctx, **r})
                del m
        except BenchStopped as stop:
            # Proxy, TAC-209: beyond the cap the bench stops and says so. A
            # strategy whose answers did not all come back is not scored.
            result["stopped"] = str(stop)
            done = {r["strategy"] for r in result["records"]}
            for s in STRATEGIES:
                if s not in done:
                    result["ineligible"].setdefault(s, f"not measured: {stop}")
    after = {p: sha256_file(p) for p in live}
    result["llm"] = {"calls": budget["used"], "errors": budget["errors"],
                     "refused_over_cap": budget["refused"], "cap": budget["cap"],
                     "cap_reached": budget["reached"]}
    result["live_store_sha256_before"] = before
    result["live_store_sha256_after"] = after
    result["live_store_changed_by_other_process"] = before != after
    result["bench_writes_under_live_root"] = violations
    if violations:
        raise SystemExit(f"{ctx}: the bench tried to write under a live root: "
                         f"{violations}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.yaml"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--ctx", action="append")
    ap.add_argument("--out", default=".")
    ap.add_argument("--llm-cap", type=int, default=LLM_CAP,
                    help="LLM calls per context; beyond, the context stops")
    ap.add_argument("--sleep-build-cap-s", type=float, default=SLEEP_BUILD_CAP_S,
                    help="cap of build(sleep); beyond, sleep is ineligible")
    a = ap.parse_args()
    if os.path.exists(a.scratch) and os.listdir(a.scratch):
        raise SystemExit(f"--scratch {a.scratch} is not empty")
    questions = yaml.safe_load(open(a.questions, encoding="utf-8"))
    ctxs = a.ctx or list(questions["contexts"])
    roots = [r for c in ctxs for r in live_roots(questions["contexts"][c])]
    for where in (a.scratch, a.out):
        real = os.path.realpath(where)
        if any(real == r or real.startswith(r + os.sep) for r in roots):
            raise SystemExit(f"{where} is under a live root {roots}")
    os.makedirs(a.scratch, exist_ok=True)

    from metacog.defaults import SimpleEncoder, make_encoder, make_reranker
    from metacog.llm import ClaudeLLM
    control = control_call(ClaudeLLM())      # no real LLM, no bench
    encoder, reranker = make_encoder(), make_reranker()
    # Rule 6: the real mode never falls back on a default.
    if isinstance(encoder, SimpleEncoder) or reranker is None:
        raise SystemExit(f"real models unavailable: encoder={type(encoder).__name__} "
                         f"reranker={type(reranker).__name__}")

    run = {"questions_sha256": sha256_file(a.questions),
           "encoder": type(encoder).__name__, "reranker": type(reranker).__name__,
           "llm_control_call": control, "llm_cap": a.llm_cap,
           "sleep_build_cap_s": a.sleep_build_cap_s, "contexts": {}}
    records = []
    for ctx in ctxs:
        res = run_context(ctx, questions["contexts"][ctx], questions["offtopic"],
                          a.scratch, (encoder, reranker), ClaudeLLM,
                          llm_cap=a.llm_cap, sleep_cap_s=a.sleep_build_cap_s)
        records += res.pop("records")
        run["contexts"][ctx] = res
    run["scores"] = score(records, questions)
    run["decision"] = decide({c: {s: v for s, v in cells.items() if s in STRATEGIES}
                              for c, cells in run["scores"].items()},
                             {c: r["ineligible"] for c, r in run["contexts"].items()})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "d3_records.jsonl"), "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "d3_summary.json"), "w") as fh:
        json.dump(run, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"scores": run["scores"], "decision": run["decision"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
