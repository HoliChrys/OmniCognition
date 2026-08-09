"""
Step-by-step debugger for the SQL-driven journal workflow.

NOT a benchmark. Runs on SimpleEncoder + a fake LLM (no sentence-transformers,
no network), so it exercises the REAL code paths — record_hop, the walk's path
feed, SQL co-retrieval / lateral, the SQL Chasles trigger, its derived
refractory, and journal persistence — one stage at a time, the way you'd step
through a debugger. Each step prints [ok]/[FAIL] and the workflow exits non-zero
if any step fails, so it doubles as a smoke gate for the DB migration.

Usage
-----
  python -m benchmarks.debug_chasles_workflow                 # run every step
  python -m benchmarks.debug_chasles_workflow --steps ingest,walk,chasles
  python -m benchmarks.debug_chasles_workflow --list         # list step names
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory
from metacog.spike import record_hop


CHAIN = ["P0", "P1", "P2", "P3"]          # the fact path we make "often-taken"


class _FakeLLM:
    """Minimal LLM satisfying resolve_collision's contract (no network)."""

    def extract_common(self, texts):
        return "common"

    def remove_overlap(self, full, common):
        return full.replace(common, "").strip() or f"distinct {full[:6]}"


class _Probe:
    """Holds shared state across steps + a pass/fail ledger."""

    def __init__(self, store: str):
        self.store = store
        self.jpath = f"{store}.journal.db"
        self.mem = Memory(encoder=SimpleEncoder(), llm=_FakeLLM(),
                          storage_path=store, journal_path="auto")
        self.by_id = {}
        self.failures: List[str] = []

    def check(self, label: str, cond: bool, detail: str = "") -> bool:
        mark = "ok " if cond else "FAIL"
        print(f"    [{mark}] {label}" + (f"  — {detail}" if detail else ""))
        if not cond:
            self.failures.append(label)
        return cond


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

def step_ingest(p: _Probe):
    """Ingest a splittable fact chain + a couple of distractors."""
    for i, nid in enumerate(CHAIN):
        p.by_id[nid] = p.mem.ingest(f"shared common topic chunk {i}",
                                    kind="FACT", id=nid)
    for i in range(2):
        p.mem.ingest(f"unrelated distractor {i}", kind="FACT", id=f"D{i}")
    p.check("journal attached", p.mem.journal is not None, p.mem.journal.path)
    p.check("points ingested", len(p.mem.points) == 6, f"{len(p.mem.points)} pts")


def step_walk(p: _Probe):
    """Simulate the walk TWICE : record_hop feeds hops+n_spike (real), then the
    finalize hook's log_path(with_hops=False) feeds path_traversals (real)."""
    for _ in range(2):                                  # travelled twice
        for a, b in zip(CHAIN, CHAIN[1:]):
            record_hop(p.by_id[a], p.by_id[b], p.mem)  # hops + n_spike + refs
        p.mem.log_path(CHAIN, with_hops=False)         # path unit (walk feed)
    rows = p.mem.journal.frequent_paths(min_len=4, min_freq=2)
    p.check("path_traversals fed", [r["signature"] for r in rows] == ["P0>P1>P2>P3"],
            str([(r["signature"], r["freq"]) for r in rows]))


def step_spike(p: _Probe):
    """The per-node spike view (effective_spike) is consistent with the hops —
    NOT double-counted by the path feed (with_hops=False)."""
    es = p.mem.journal.effective_spike("P3")
    p.check("effective_spike(P3) == 2", es == 2, f"got {es}")
    top = p.mem.journal.hop_target_counts()
    p.check("hop targets ranked", top and top[0][1] >= 2, str(top[:3]))


def step_coretr(p: _Probe):
    """Log two overlapping retrievals -> SQL co-retrieval surfaces the pair."""
    p.mem.record_retrieval(["P0", "P1", "P2"], query_text="q1")
    p.mem.record_retrieval(["P0", "P1", "D0"], query_text="q2")
    co = p.mem.co_retrieved(["P0"])
    ids = {nid for nid, _ in co}
    p.check("co-retrieval finds P1", "P1" in ids, str(co))


def step_feedback(p: _Probe):
    """Concept A : auto-label retrievals by what was USED downstream, then
    fit_decay -> the L3 loop closes without a human."""
    # R1 : returned P0,P1,P2 ; two of them used -> useful=2 (positives)
    r1 = p.mem.record_retrieval(["P0", "P1", "P2"], query_text="fb-useful")
    # R2 : returned D0,D1 ; none used -> useless=0 (negatives, disjoint from R1)
    r2 = p.mem.record_retrieval(["D0", "D1"], query_text="fb-useless")
    scored = p.mem.score_retrievals([r1, r2], used_ids=["P0", "P1"])
    p.check("labels produced (2 and 0)",
            sorted(s for _, s in scored) == [0, 2], str(scored))
    res = p.mem.fit_decay()
    p.check("fit_decay now has both classes",
            res["n_pos"] > 0 and res["n_neg"] > 0,
            f"n_pos={res['n_pos']} n_neg={res['n_neg']} exponent={res['exponent']:.3f}")


def step_needodds(p: _Probe):
    """Concept B : need-odds (ACT-R base-level) blended into ranking behind
    recency_weight. OFF by default (order identical) ; ON re-ranks by access
    recency×frequency."""
    q = "shared common topic"
    base = [h["id"] for h in p.mem.retrieve(q, k=6)]           # recency_weight=0
    target = base[-1]                                          # the base-LAST node
    p.check("baseline captured (weight 0)", p.mem.recency_weight == 0.0,
            f"base={base[:4]} target={target}")
    # hammer accesses onto the base-last node so its need-odds dominates
    for _ in range(8):
        p.mem.record_retrieval([target], query_text="hammer")
    p.mem.recency_weight = 1.0                                 # pure need-odds
    ranked = [h["id"] for h in p.mem.retrieve(q, k=6)]
    p.mem.recency_weight = 0.0                                 # restore
    p.check("need-odds LIFTS the hammered base-last node",
            ranked.index(target) < base.index(target),
            f"{target}: base#{base.index(target)} -> need#{ranked.index(target)}")


def step_decayfit(p: _Probe):
    """Auto-maintenance : sleep() re-fits the decay exponent from accumulated
    feedback labels (concept A -> B), so the need-odds ranking stays calibrated
    without a manual fit_decay call."""
    before = p.mem.decay_exponent
    out = p.mem.sleep()
    p.check("sleep reports decay-fit", "decay_exponent" in out,
            f"n_pos={out.get('decay_fit_n_pos')} n_neg={out.get('decay_fit_n_neg')}")
    p.check("exponent maintained from feedback",
            out["decay_fit_n_pos"] > 0 and out["decay_fit_n_neg"] > 0,
            f"{before:.3f} -> {p.mem.decay_exponent:.3f}")


def step_lateral(p: _Probe):
    """lateral_collapse runs off the SQL ledger when a journal is present
    (informational on a tiny cloud — it is gated on a large tag-rich cloud)."""
    rep = p.mem.lateral_collapse()
    p.check("lateral ran (journal-driven)", "collided_groups" in rep,
            f"groups={rep.get('collided_groups')}")


def step_spreading(p: _Probe):
    """Concept C : associative spreading via the co-retrieval log. A node the
    cosine misses gets INJECTED because it was historically co-retrieved with a
    seed. OFF by default (candidate set identical)."""
    q = "shared common topic"
    base = [h["id"] for h in p.mem.retrieve(q, k=3)]
    seed = base[0]
    missed = "D1"                                      # far from q in embedding
    p.check("target is cosine-missed", missed not in base, f"base={base}")
    # forge an association : seed and D1 keep being co-retrieved together
    for _ in range(5):
        p.mem.record_retrieval([seed, missed], query_text="assoc")
    p.mem.spreading_weight = 1.0
    ranked = [h["id"] for h in p.mem.retrieve(q, k=3)]
    p.mem.spreading_weight = 0.0                       # restore
    p.check("association injects the missed node", missed in ranked,
            f"base={base} -> spread={ranked}")


def step_tags(p: _Probe):
    """Hierarchical tag index in SQL : ancestor query + glossary."""
    p.by_id["P0"].tags.append("health:condition:macrodactyly")
    p.by_id["P1"].tags.append("health:condition:fatigue")
    p.mem.reindex_tags()
    scoped = {q.id for q in p.mem.tag_scoped("health")}
    p.check("tag ancestor query", scoped == {"P0", "P1"}, str(scoped))
    p.check("glossary depth-ordered",
            "health:condition:fatigue" in p.mem.tag_glossary_sql())


def step_chasles(p: _Probe):
    """The SQL trigger : compress_chasles reads chasles_path_candidates and
    collapses the frequent path into a start->end shortcut."""
    cands = p.mem.chasles_path_candidates(min_freq=2)
    p.check("candidate surfaced by query", [c["signature"] for c in cands]
            == ["P0>P1>P2>P3"], str([c["signature"] for c in cands]))
    events = p.mem.compress_chasles()
    p.check("chasles fired (SQL-driven)", len(events) == 1, f"{len(events)} event(s)")
    if events:
        child = events[0]["child_id"]
        p.check("shortcut child appended", any(q.id == child for q in p.mem.points),
                child)


def step_refractory(p: _Probe):
    """The derived refractory : re-running is a no-op until re-travelled."""
    p.check("candidates now empty", p.mem.chasles_path_candidates(min_freq=2) == [])
    p.check("re-run is a no-op", p.mem.compress_chasles() == [])


def step_forget(p: _Probe):
    """Concept #2 : OFFLINE decay-forgetting. Query time only decays (ranking) ;
    here (sleep-phase, isolated memory) cold nodes are pruned — hot & tools kept,
    the untried untouched."""
    from metacog.epistemic import Point, PointKind
    from metacog.skills import TOOL_TAG, is_tool
    enc = SimpleEncoder()
    m = Memory(encoder=enc, journal=Journal())
    for i in range(6):
        m.ingest(f"hot fact {i}", kind="FACT", id=f"H{i}")
    for i in range(2):
        m.ingest(f"cold fact {i}", kind="FACT", id=f"C{i}")
    m.points.append(Point(id="tool::x", content="a tool",
                          embedding_orig=tuple(enc.encode("a tool")),
                          kind=PointKind.ACTION, tags=[TOOL_TAG]))
    m.ingest("never retrieved", kind="FACT", id="NEW")
    now = 10_000
    for i in range(6):
        m.journal.log_retrieval("q", [f"H{i}"], ts=now - 1)      # hot
    for i in range(2):
        m.journal.log_retrieval("q", [f"C{i}"], ts=now - 5_000)  # cold
    m.journal.log_retrieval("q", ["tool::x"], ts=now - 5_000)     # cold tool
    dry = m.forget(t=now, apply=False)
    p.check("dry-run flags only cold outliers", set(dry["candidates"]) == {"C0", "C1"},
            str(dry["candidates"]))
    out = m.forget(t=now, apply=True)
    ids = {q.id for q in m.points}
    p.check("cold nodes pruned", set(out["forgotten"]) == {"C0", "C1"})
    p.check("hot nodes kept", all(f"H{i}" in ids for i in range(6)))
    p.check("tools + untried protected",
            any(is_tool(q) for q in m.points) and "NEW" in ids)


def step_persist(p: _Probe):
    """Save + close, then reopen the SAME store -> journal state survives."""
    p.mem.save()
    p.mem.journal.conn.close()
    m2 = Memory(encoder=SimpleEncoder(), llm=_FakeLLM(),
                storage_path=p.store, journal_path="auto")
    scoped = {q.id for q in m2.tag_scoped("health")}
    p.check("tags survived restart", scoped == {"P0", "P1"}, str(scoped))
    hist = m2.collision_history("chasles")
    p.check("chasles audit survived restart", len(hist) >= 1, f"{len(hist)} event(s)")
    m2.journal.conn.close()


STEPS = [
    ("ingest", step_ingest),
    ("walk", step_walk),
    ("spike", step_spike),
    ("coretr", step_coretr),
    ("feedback", step_feedback),
    ("needodds", step_needodds),
    ("spreading", step_spreading),
    ("lateral", step_lateral),
    ("tags", step_tags),
    ("chasles", step_chasles),
    ("refractory", step_refractory),
    ("decayfit", step_decayfit),
    ("forget", step_forget),
    ("persist", step_persist),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", help="comma-separated subset (default: all)")
    ap.add_argument("--list", action="store_true", help="list step names")
    args = ap.parse_args(argv)
    if args.list:
        print("steps:", ", ".join(name for name, _ in STEPS))
        return 0

    wanted = ([s.strip() for s in args.steps.split(",")] if args.steps
              else [name for name, _ in STEPS])
    with tempfile.TemporaryDirectory() as d:
        p = _Probe(os.path.join(d, "mem.pkl"))
        for name, fn in STEPS:
            if name not in wanted:
                continue
            print(f"\n== step: {name} ==")
            try:
                fn(p)
            except Exception as exc:               # noqa: BLE001
                import traceback
                p.failures.append(f"{name} (exception)")
                print(f"    [FAIL] {name} raised {exc!r}")
                traceback.print_exc()
        print("\n" + "=" * 48)
        if p.failures:
            print(f"WORKFLOW INCOMPLETE — {len(p.failures)} failure(s): "
                  f"{p.failures}")
            return 1
        print("WORKFLOW COMPLETE — all steps passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
