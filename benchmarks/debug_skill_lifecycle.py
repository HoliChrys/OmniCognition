"""
Step-by-step debugger for the SKILL / TOOL lifecycle in the memory.

Validates the mnema agent-tool tier : the agent recognises a need and creates a
tool, the tool is stored AS A NODE in the memory (contextualized by the need),
and the CANONICAL agent later finds it ORGANICALLY (plain retrieve) to REUSE it —
no hardcoded @app.tool(). Runs on SimpleEncoder + a fake LLM (fallback synthesis,
no network). Each step prints [ok]/[FAIL] ; exits non-zero if the loop breaks.

    python -m benchmarks.debug_skill_lifecycle
    python -m benchmarks.debug_skill_lifecycle --steps need,create,reuse
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from metacog.defaults import SimpleEncoder
from metacog.epistemic import Point, PointKind
from metacog.memory import Memory
from metacog.skills import is_tool, tools_in


class _FakeLLM:
    """No `generate` -> synthesis uses the deterministic fallback path."""


NEED = "reverse a string end to end"
HOW = "iterate over the characters from last to first"


class _Probe:
    def __init__(self):
        self.mem = Memory(encoder=SimpleEncoder(), llm=_FakeLLM())
        self.mem.skills_enabled = True
        self.failures: List[str] = []
        self.tool_id = None

    def check(self, label, cond, detail=""):
        print(f"    [{'ok ' if cond else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
        if not cond:
            self.failures.append(label)
        return cond


def step_need(p: _Probe):
    """No tool covers the need yet — the agent would have to think from scratch."""
    p.check("no covering tool before creation", p.mem.match_tool(NEED) is None)


def step_create(p: _Probe):
    """The agent says 'to do NEED I need this tool' -> a TOOL node is created and
    stored IN the memory, contextualized by the need."""
    r = p.mem.ensure_tool(NEED, how=HOW)
    p.check("tool generated (not reused)", r["tool"] is not None and not r["reused"])
    if r["tool"]:
        p.tool_id = r["tool"]["id"]
    p.check("stored AS a tool node in memory",
            any(is_tool(x) for x in p.mem.points), f"{len(tools_in(p.mem.points))} tool(s)")
    tool = next((x for x in p.mem.points if x.id == p.tool_id), None)
    p.check("contextualized by the need (scope keywords lead)",
            bool(tool and tool.keywords), str(tool.keywords if tool else None))


def step_organic_retrieve(p: _Probe):
    """The CANONICAL agent finds the tool with plain retrieve — no special tool."""
    hits = [h["id"] for h in p.mem.retrieve(NEED, k=3)]
    p.check("canonical retrieve surfaces the tool organically",
            p.tool_id in hits, str(hits))


def step_reuse(p: _Probe):
    """Same need again -> capability-cache HIT : reuse, don't regenerate."""
    before = next((x.n_uses for x in p.mem.points if x.id == p.tool_id), None)
    r = p.mem.ensure_tool(NEED)
    after = next((x.n_uses for x in p.mem.points if x.id == p.tool_id), None)
    p.check("reused (no regeneration)", r["reused"] is True)
    p.check("use count incremented", after == (before or 0) + 1, f"{before} -> {after}")
    p.check("still a single tool node", len(tools_in(p.mem.points)) == 1)


def step_discriminate(p: _Probe):
    """An UNRELATED need must NOT falsely reuse the tool."""
    p.check("unrelated need does not match",
            p.mem.match_tool("bake a chocolate cake from scratch") is None)


def step_crystallize(p: _Probe):
    """The second creation trigger : an action re-derived across DIVERSE queries
    (a population of signatures, one dominant) crystallizes into a TOOL node."""
    enc = SimpleEncoder()
    pop = {"sort a list ascending": 12, "count the words": 2,
           "parse a date": 2, "trim whitespace": 2}
    for content, reps in pop.items():
        for i in range(reps):
            a = Point(id=f"{content[:5]}{i}", content=content,
                      embedding_orig=tuple(enc.encode(content)),
                      kind=PointKind.ACTION, keywords=content.split())
            p.mem.record_action_generation(a, None, f"{content} — approach {i}", [])
    before = len(tools_in(p.mem.points))
    out = p.mem.crystallize_skills()
    p.check("recurrence crystallizes a tool", out["crystallized"] >= 1,
            f"crystallized={out['crystallized']} ids={out['tool_ids'][:2]}")
    p.check("new tool node added to memory",
            len(tools_in(p.mem.points)) > before)


def step_lifecycle(p: _Probe):
    """Concept #3 : the retract / feedback / correct half. Isolated memory (one
    tool) so ensure_tool's regenerate-on-miss doesn't accumulate copies."""
    m = Memory(encoder=SimpleEncoder(), llm=_FakeLLM())
    tid = m.ensure_tool(NEED, how=HOW)["tool"]["id"]
    p.check("reusable before retire", m.ensure_tool(NEED)["reused"] is True)
    m.retire_tool(tid)
    # after a soft-retire the ONLY tool is deprecated -> a fresh one is generated
    r = m.ensure_tool(NEED)
    p.check("soft-retire stops reuse of the retired tool", r["reused"] is False)
    # correct + revive the ORIGINAL, retire the regenerated copy to isolate it
    m.retire_tool(r["tool"]["id"], hard=True)
    m.update_tool(tid, content="reverse via two-pointer swap")
    p.check("update revives + reuses again", m.ensure_tool(NEED)["reused"] is True)
    # feedback : two failures auto-retire the (now sole) tool
    m.report_tool(tid, ok=False)
    p.check("repeated failures auto-retire", m.report_tool(tid, ok=False)["retired"])
    p.check("auto-retired tool no longer reused",
            m.ensure_tool(NEED)["reused"] is False)


def step_persist(p: _Probe):
    """Tool nodes are normal nodes : they survive save/load and stay retrievable."""
    import os, tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pkl")
        p.mem.storage_path = path
        p.mem.save()
        m2 = Memory(encoder=SimpleEncoder(), llm=_FakeLLM(), storage_path=path)
        p.check("tool nodes survive save/load",
                len(tools_in(m2.points)) == len(tools_in(p.mem.points)))
        hits = [h["id"] for h in m2.retrieve(NEED, k=3)]
        p.check("reloaded tool still retrievable", p.tool_id in hits, str(hits))


STEPS = [
    ("need", step_need), ("create", step_create),
    ("organic_retrieve", step_organic_retrieve), ("reuse", step_reuse),
    ("discriminate", step_discriminate), ("crystallize", step_crystallize),
    ("lifecycle", step_lifecycle), ("persist", step_persist),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        print("steps:", ", ".join(n for n, _ in STEPS))
        return 0
    wanted = ([s.strip() for s in args.steps.split(",")] if args.steps
              else [n for n, _ in STEPS])
    p = _Probe()
    for name, fn in STEPS:
        if name not in wanted:
            continue
        print(f"\n== step: {name} ==")
        try:
            fn(p)
        except Exception as exc:  # noqa: BLE001
            import traceback
            p.failures.append(f"{name} (exception)")
            print(f"    [FAIL] {name} raised {exc!r}")
            traceback.print_exc()
    print("\n" + "=" * 48)
    if p.failures:
        print(f"SKILL LIFECYCLE BROKEN — {len(p.failures)} failure(s): {p.failures}")
        return 1
    print("SKILL LIFECYCLE OK — create -> store -> organic retrieve -> reuse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
