"""
TREE-navigable, PARALLEL-query debugger for OBLIQ-Bench.

Now that a query is answered by TWO PARALLEL channels — the plain RETRIEVAL/walk
and the CONTEXT search (events gravitating around a context = an event:type) —
debugging one query at a time, one channel at a time, hides the interaction.
This debugger indexes SEVERAL oblique queries into ONE shared mixed pool and
lets you NAVIGATE the whole thing as a TREE of branches :

    ROOT  (mixed pool : the union of every query's gold + shared distractors)
    ├── q0419                               ← a QUERY branch (carries its gold)
    │   ├── retrieval                        ← channel A : plain retrieve/walk
    │   └── context                          ← channel B : the event subsystem
    │       ├── [attack]  (context = type)   ← a CONTEXT (event:type) branch
    │       │   ├── south pars gas field…     ← an EVENT that gravitates to it
    │       │   │   ├── slot: attacker         ← schema slot fill (scoped+KB)
    │       │   │   └── slot: target
    │       │   └── …
    │       └── [war] …
    └── q0395 …

Every node knows its evidence ids, so `recall` at any node shows how much of the
ambient query's gold that branch surfaces — you can compare the retrieval branch
against the context branch, or drill into one event's slots, side by side across
queries. Expensive branches (the schema fill runs a scoped+KB walk per core
slot) expand LAZILY, only when you `cd` into them.

Usage
-----
  # interactive
  python -m benchmarks.obliq_bench.tree_nav --query-ids q0419,q0395 --bg 40
  # scripted (';'-separated commands then exit)
  python -m benchmarks.obliq_bench.tree_nav --query-ids q0419,q0433 --bg 30 \
      --script "tree; cd q0419; cd context; ls; recall"

Commands : ls · tree [depth] · cd <n|name> · up/.. · root · pwd · recall ·
           ids [n] · help · q
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Callable, List, Optional, Sequence

from benchmarks.obliq_bench.debug_qa import _load_queries, _load_qrels, _dl


# --------------------------------------------------------------------------- #
# the tree
# --------------------------------------------------------------------------- #
class Node:
    """One branch of the debug tree. `ids` is its evidence set ; `children` are
    produced lazily by `expander` (so an expensive context/slot branch is only
    materialised when you navigate into it)."""

    def __init__(self, name: str, kind: str, *, ids: Sequence[str] = (),
                 meta: str = "", gold: Optional[set] = None,
                 expensive: bool = False) -> None:
        self.name = name
        self.kind = kind
        self.ids = set(ids)
        self.meta = meta
        self.gold = gold                 # set on QUERY nodes ; ambient otherwise
        self.expensive = expensive
        self.children: List["Node"] = []
        self.expander: Optional[Callable[[], List["Node"]]] = None
        self._expanded = False

    def expand(self) -> List["Node"]:
        if not self._expanded and self.expander is not None:
            try:
                self.children = self.expander() or []
            except Exception as e:                       # never crash the REPL
                self.children = [Node(f"<expand error: {e}>", "error")]
            self._expanded = True
        return self.children

    def recall(self, gold: set) -> float:
        g = set(gold or ())
        return (len(self.ids & g) / len(g)) if g else 0.0


# --------------------------------------------------------------------------- #
# pool builder : several queries -> one shared, event-spawned, consolidated mem
# --------------------------------------------------------------------------- #
def build_pool(track: str, query_ids: List[str], bg: int, *, seed: int = 0,
               llm_merge: bool = True):
    from metacog.memory import Memory
    from metacog.llm import ClaudeLLM
    from metacog.event_extractor import LLMEventExtractor
    from benchmarks.locomo.encoders import SemanticEncoder

    queries, qrels = _load_queries(track), _load_qrels(track)
    query_ids = [q for q in query_ids if q in queries] or [sorted(queries)[0]]

    corpus = {}
    for ln in open(_dl(track, "corpus/corpus.jsonl")):
        o = json.loads(ln)
        corpus[str(o["_id"])] = (o.get("text") or o.get("title") or "")

    golds = {q: [g for g in sorted(qrels[q]) if g in corpus] for q in query_ids}
    rel = sorted({g for gs in golds.values() for g in gs})       # union of gold
    pool = [c for c in corpus if c not in set(rel)]
    random.Random(seed).shuffle(pool)
    keep = rel + pool[:max(0, bg)]

    llm = ClaudeLLM()
    mem = Memory(encoder=SemanticEncoder(), llm=llm,
                 event_extractor=LLMEventExtractor(llm))
    t0 = time.time()
    for cid in keep:
        mem.ingest(corpus[cid][:500], kind="FACT", id=cid)
    rep = mem.consolidate_events(use_llm=llm_merge)
    print(f"[pool] {len(query_ids)} queries, {len(rel)} gold(∪) + "
          f"{len(keep)-len(rel)} distractors = {len(keep)} docs ; spawn+"
          f"consolidate {time.time()-t0:.0f}s ; consolidate={rep}")
    return mem, {q: (queries[q], set(golds[q])) for q in query_ids}


# --------------------------------------------------------------------------- #
# tree construction (lazy)
# --------------------------------------------------------------------------- #
def _macros(mem):
    from metacog.epistemic import PointKind
    hubs = [p for p in mem.points if p.kind is PointKind.EVENT]
    ev_ids = {h.id for h in hubs}
    out = []
    for h in hubs:
        absorbed = any(t.startswith("event:in:")
                       and t.split(":", 2)[2] in ev_ids
                       and t.split(":", 2)[2] != h.id for t in h.tags)
        if not absorbed:
            out.append(h)
    out.sort(key=lambda h: len(mem.event_cluster(h.id)), reverse=True)
    return out


def _etype(hub) -> str:
    return next((t.split(":", 2)[2] for t in hub.tags
                 if t.startswith("event:type:")), "")


def build_tree(mem, qmap: dict) -> Node:
    root = Node("ROOT", "root", meta=f"{len(qmap)} queries")

    def query_children(question, gold):
        def make():
            # channel A : plain retrieval (the '0 clue' baseline)
            base = mem.retrieve(question, k=max(20, 2 * len(gold)))
            retr = Node("retrieval", "channel",
                        ids=[h["id"] for h in base],
                        meta=f"retrieve@{len(base)}", gold=gold)
            # channel B : the CONTEXT subsystem (lazy)
            ctx = Node("context", "channel", gold=gold, expensive=True)
            ctx.expander = context_children(question, gold)
            return [retr, ctx]
        return make

    def context_children(question, gold):
        def make():
            det = (mem.detect_event_type(question)
                   if hasattr(mem, "detect_event_type") else None)
            routed = det["etype"] if det else None
            # one CONTEXT branch per event:type present, routed one first/flagged
            seen, order = set(), []
            for h in _macros(mem):
                et = _etype(h)
                if et and et not in seen:
                    seen.add(et)
                    order.append(et)
            if routed in order:
                order.remove(routed)
                order.insert(0, routed)
            nodes = []
            for et in order:
                cids = [p.id for p in mem.context_members(et)]
                flag = "  <= ROUTED" if et == routed else ""
                n = Node(f"[{et}]", "context", ids=cids,
                         meta=f"{len(cids)} facts{flag}", gold=gold)
                n.expander = event_children(et, question, gold)
                nodes.append(n)
            return nodes or [Node("<no context>", "context", gold=gold)]
        return make

    def event_children(etype, question, gold):
        def make():
            tag = f"event:type:{etype}"
            from metacog.epistemic import PointKind
            hubs = [h for h in _macros(mem) if tag in h.tags]
            nodes = []
            for h in hubs:
                cl = mem.event_cluster(h.id)
                nm = h.content.split(" ", 1)[1] if " " in h.content else h.content
                n = Node(nm[:30], "event", ids=[p.id for p in cl],
                         meta=f"cluster={len(cl)}", gold=gold, expensive=True)
                n.expander = slot_children(nm, etype, question, gold)
                nodes.append(n)
            return nodes
        return make

    def slot_children(event_name, etype, question, gold):
        def make():
            from metacog.event_schema import fill_event_schema
            hub_id = mem._event_registry.get(
                f"{etype}::{event_name.strip().lower()}")
            scope = f"event:in:{hub_id}" if hub_id else None
            fill = fill_event_schema(mem, event_name, etype, k_per_slot=5,
                                     scope_tag=scope)
            nodes = []
            core = set(fill.get("core") or [])
            for slot, hits in (fill.get("filled") or {}).items():
                tag = "core" if slot in core else "peripheral"
                gap = " GAP" if slot in (fill.get("gaps") or []) else ""
                nodes.append(Node(
                    f"slot:{slot}", "slot", ids=[h["id"] for h in hits],
                    meta=f"{tag}{gap} ({len(hits)} hits)", gold=gold))
            return nodes
        return make

    children = []
    for qid, (question, gold) in qmap.items():
        qn = Node(qid, "query", gold=gold, meta=f"gold={len(gold)}",
                  expensive=True)
        qn.meta += f" | {question[:60]}…"
        qn.expander = query_children(question, gold)
        children.append(qn)
    root.children = children
    root._expanded = True
    return root


# --------------------------------------------------------------------------- #
# REPL
# --------------------------------------------------------------------------- #
class TreeREPL:
    def __init__(self, root: Node):
        self.path: List[Node] = [root]

    @property
    def cur(self) -> Node:
        return self.path[-1]

    def ambient_gold(self) -> set:
        for n in reversed(self.path):
            if n.gold is not None:
                return n.gold
        return set()

    def pwd(self) -> str:
        return " / ".join(n.name for n in self.path)

    def _fmt(self, n: Node, idx: int) -> str:
        g = self.ambient_gold()
        r = ("  recall=%.3f (%d/%d)" % (n.recall(g), len(n.ids & g), len(g))
             if g and n.ids else "")
        exp = " *" if n.expensive and not n._expanded else ""
        meta = f"  — {n.meta}" if n.meta else ""
        return f"  [{idx}] {n.name}{exp}  <{n.kind}>{meta}{r}"

    def ls(self) -> None:
        kids = self.cur.expand()
        print(f"@ {self.pwd()}")
        if not kids:
            print("  (leaf)")
        for i, k in enumerate(kids):
            print(self._fmt(k, i))

    def tree(self, depth: int = 2) -> None:
        def rec(n: Node, d: int, prefix: str):
            g = self.ambient_gold() or n.gold or set()
            r = ("  r=%.3f(%d/%d)" % (n.recall(g), len(n.ids & g), len(g))
                 if g and n.ids else "")
            print(f"{prefix}{n.name} <{n.kind}>{('  '+n.meta) if n.meta else ''}{r}")
            if d <= 0:
                if n.expensive and not n._expanded:
                    print(prefix + "  … (cd to expand)")
                return
            # don't trigger expensive expansion in a broad tree view
            kids = n.children if (n.expensive and not n._expanded) else n.expand()
            if n.expensive and not n._expanded:
                print(prefix + "  … (cd to expand)")
                return
            for k in kids:
                rec(k, d - 1, prefix + "  ")
        rec(self.cur, depth, "")

    def cd(self, token: str) -> None:
        if token in ("..", "up"):
            if len(self.path) > 1:
                self.path.pop()
            return
        if token == "root":
            self.path = self.path[:1]
            return
        kids = self.cur.expand()
        target = None
        if token.isdigit() and int(token) < len(kids):
            target = kids[int(token)]
        else:
            for k in kids:
                if k.name == token or k.name.strip("[]") == token:
                    target = k
                    break
            if target is None:
                for k in kids:                       # prefix match
                    if k.name.lower().startswith(token.lower()):
                        target = k
                        break
        if target is None:
            print(f"  no child '{token}' (use ls)")
            return
        if target.expensive and not target._expanded:
            print(f"  expanding {target.name} …")
        self.path.append(target)
        self.ls()

    def recall(self) -> None:
        g = self.ambient_gold()
        n = self.cur
        print("  @ %s : recall=%.3f  (%d/%d gold)  |ids|=%d" % (
            self.pwd(), n.recall(g), len(n.ids & g), len(g), len(n.ids)))

    def schema(self) -> None:
        """Verify the SCHEMA system recovers all its clues for the current event:
        per-slot fill, which CORE slots are filled vs GAPs, and the recall the
        slot-fills contribute to the ambient query's gold."""
        n = self.cur
        if n.kind != "event":
            print("  cd into an <event> node first (schema is per-event)")
            return
        kids = n.expand()                            # slot nodes (runs the fill)
        g = self.ambient_gold()
        core = [k for k in kids if "core" in k.meta]
        filled_core = [k for k in core if k.ids]
        gaps = [k for k in core if "GAP" in k.meta]
        allslot = set().union(*[k.ids for k in kids]) if kids else set()
        print(f"  SCHEMA @ {n.name}")
        for k in kids:
            rc = len(k.ids & g) if g else 0
            print("    %-22s %-22s hits=%-2d goldᶜ=%d" % (
                k.name, k.meta, len(k.ids), rc))
        print("  core filled %d/%d ; gaps=%s" % (
            len(filled_core), len(core), [k.name.replace('slot:', '')
                                          for k in gaps] or "none"))
        if g:
            print("  slot-fill recall=%.3f (%d/%d)  vs  cluster recall=%.3f" % (
                len(allslot & g) / len(g), len(allslot & g), len(g),
                n.recall(g)))

    def ids(self, k: int = 10) -> None:
        g = self.ambient_gold()
        by_id = {}
        # resolve content lazily from the root memory if attached
        for i, fid in enumerate(sorted(self.cur.ids)[:k]):
            mark = "*" if fid in g else " "
            print(f"  {mark} {fid}")
        if len(self.cur.ids) > k:
            print(f"  … (+{len(self.cur.ids)-k} more)")

    def run(self, cmd: str) -> bool:
        cmd = cmd.strip()
        if not cmd:
            return True
        op, *rest = cmd.split()
        arg = rest[0] if rest else ""
        if op in ("q", "quit", "exit"):
            return False
        elif op == "ls":
            self.ls()
        elif op == "tree":
            self.tree(int(arg) if arg.isdigit() else 2)
        elif op in ("cd",):
            self.cd(arg or "root")
        elif op in ("up", ".."):
            self.cd("..")
        elif op == "root":
            self.cd("root")
        elif op == "pwd":
            print("  " + self.pwd())
        elif op == "recall":
            self.recall()
        elif op == "schema":
            self.schema()
        elif op == "ids":
            self.ids(int(arg) if arg.isdigit() else 10)
        elif op in ("help", "?"):
            print("  ls · tree [d] · cd <n|name> · up · root · pwd · recall · "
                  "schema · ids [n] · q")
        else:
            print(f"  ? {op} (help)")
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description="OBLIQ tree-navigable debugger")
    ap.add_argument("--track", default="descriptive/twitter")
    ap.add_argument("--query-ids", default="q0419,q0395",
                    help="comma-separated OBLIQ query ids (the parallel branches)")
    ap.add_argument("--bg", type=int, default=40)
    ap.add_argument("--no-llm-merge", action="store_true",
                    help="skip LLM arbitration in consolidate_events (faster)")
    ap.add_argument("--script", help="';'-separated commands then exit")
    args = ap.parse_args()

    qids = [q.strip() for q in args.query_ids.split(",") if q.strip()]
    mem, qmap = build_pool(args.track, qids, args.bg,
                           llm_merge=not args.no_llm_merge)
    root = build_tree(mem, qmap)
    repl = TreeREPL(root)
    print("\nTREE debugger — type 'help'. Branches: query / channel / context / "
          "event / slot.\n")
    repl.ls()

    if args.script:
        for c in args.script.split(";"):
            print(f"\n>>> {c.strip()}")
            if not repl.run(c):
                break
        return
    while True:
        try:
            line = input(f"\n{repl.pwd()} > ")
        except (EOFError, KeyboardInterrupt):
            break
        if not repl.run(line):
            break


if __name__ == "__main__":
    main()
