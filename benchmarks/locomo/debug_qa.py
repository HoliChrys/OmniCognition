"""
Step-by-step QA debugger for the MetaCog-Mem LoCoMo path.

Drive the retrieval/answer tools ONE AT A TIME against a single LoCoMo
question and watch, live, which tool fires, what it returns, and whether
the gold evidence has surfaced yet — so you can see exactly where a hard
(e.g. cat3 inference) question goes wrong and what to optimise.

The conversation memory is built once and CACHED to /tmp (embeddings are
expensive); subsequent sessions load instantly.

Usage
-----
  # interactive REPL on a built-in probe
  python -m benchmarks.locomo.debug_qa --probe john
  python -m benchmarks.locomo.debug_qa --probe caroline

  # an arbitrary question over a sample
  python -m benchmarks.locomo.debug_qa --sample conv-41 \
      --question "What might John's financial status be?" --gold D5:5 --category 3

  # non-interactive: run a scripted sequence (commands separated by ';')
  python -m benchmarks.locomo.debug_qa --probe john --script \
      "gold; presearch John financial status | John kids have so much; walk John kids have so much; recall"

  # force a fresh rebuild of the memory cache
  python -m benchmarks.locomo.debug_qa --probe john --rebuild

REPL commands
-------------
  q                         show the question / gold answer / category
  gold                      show the gold evidence ids + their turn text
  tools                     list the MCP tools the agent can call
  presearch <q1> | <q2>     batch recon: top-k nearest per query (no walk)
  walk <query>              run a full walk_start; show stages, drift,
                            relevant_collected, neighbor_possibilities,
                            fact_ids_cumulative — gold ids flagged ✓GOLD
  scoped <query> :: t1,t2 [kb]   tag-scoped walk (kb = also knowledge_base)
  retrieve <query>          raw top-k hybrid retrieval
  show <id>                 a turn's text + its ±2 sequence neighbours
  recall                    cumulative gold recall across this session
  answer <text>             score token-F1 of <text> vs the gold answer
  auto                      run the full autonomous agent, trace each round

  --- stepping / branching (let the algo drive, or steer by hand) ---
  step                      let the ALGO trigger the next step : one
                            autonomous round (model picks the next tool from
                            the live conversation, executes it, pauses)
  say <text>                inject a user hint into the conversation
  checkpoint [name] / cp    snapshot the conversation state (to branch from)
  restore [name]            rewind to a checkpoint and try another path
  msgs                      list the live conversation messages
  reset                     clear the accumulated retrieved-id set

  --- SQL journal workflow (the DB-driven triggers) ---
  journal                   snapshot the append-only journal (row counts,
                            hops, frequent paths, collision kinds)
  paths [min_freq]          Chasles candidates BY QUERY : frequently-travelled
                            paths ready to collapse into a shortcut
  chasles                   fire compress_chasles (SQL-driven with a journal)
                            and show the resulting shortcut events
  tags [tag]                hierarchical tag index : ancestor query on <tag>,
                            or the depth-ordered glossary when omitted

  help                      this list
  quit / q!                 exit

Manual tool calls (walk/presearch/scoped/retrieve) are RECORDED into the
live conversation, so you can steer by hand and then `step` to let the algo
continue from where you left it. `checkpoint`/`restore` let you replay a
step and compare scenarios.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set

DATA = os.path.join(os.path.dirname(__file__), "data", "locomo10.json")

_PROBES = {
    "john": {
        "sample": "conv-41",
        "question": "What might John's financial status be?",
        "gold": ["D5:5"],
        "answer": "Middle-class or wealthy",
        "category": 3,
    },
    "caroline": {
        "sample": "conv-26",
        "question": "What fields would Caroline be likely to pursue in her career?",
        "gold": ["D1:9", "D1:11"],
        "answer": "Psychology, counseling certification",
        "category": 3,
    },
}


# ----------------------------------------------------------------------
# Memory build + cache
# ----------------------------------------------------------------------


def _slice(sample: str, nsess: int) -> Dict[str, Any]:
    data = json.load(open(DATA))
    conv = next(c for c in data if c.get("sample_id") == sample)
    C = conv["conversation"]
    out: Dict[str, Any] = {"speaker_a": C.get("speaker_a"),
                           "speaker_b": C.get("speaker_b")}
    hi = nsess if nsess > 0 else 999
    for i in range(1, hi + 1):
        for key in (f"session_{i}", f"session_{i}_date_time"):
            if key in C:
                out[key] = C[key]
    return out


def build_or_load_memory(sample: str, nsess: int, rebuild: bool):
    from metacog.memory import Memory
    from metacog.llm import ClaudeLLM
    from benchmarks.locomo.encoders import SemanticEncoder
    from benchmarks.locomo.eval import ingest_conversation

    cache = f"/tmp/locomo_qa_{sample}_{nsess}.pkl"
    enc = SemanticEncoder()
    llm = ClaudeLLM()
    # Same LLM keyword extractor as the bench (noun+condition phrases, no
    # frequency noise) ; fails closed to the default frequency extractor.
    try:
        from metacog.keywords import LLMKeywordExtractor
        kw_extractor = LLMKeywordExtractor()
    except Exception:
        kw_extractor = None
    if os.path.exists(cache) and not rebuild:
        mem = Memory(encoder=enc, llm=llm, storage_path=cache)
        print(f"[cache] loaded {len(mem.points)} points from {cache}")
    else:
        mem = Memory(encoder=enc, llm=llm,
                     **({"extractor": kw_extractor} if kw_extractor else {}))
        n = ingest_conversation(mem, _slice(sample, nsess), with_dates=True)
        mem.storage_path = cache
        mem.save()
        print(f"[build] ingested {n} turns -> {len(mem.points)} points "
              f"(cached to {cache})")
    # Keep storage_path unset for the live session so a stray save() can't
    # clobber the cache mid-debug.
    mem.storage_path = None
    return mem


# ----------------------------------------------------------------------
# The interactive session
# ----------------------------------------------------------------------


class QADebugger:
    def __init__(self, memory, question: str, gold: List[str],
                 answer: str, category: int, sample: str):
        self.memory = memory
        self.question = question
        self.gold = list(gold)
        self.gold_answer = answer
        self.category = category
        self.sample = sample
        self.seen: Set[str] = set()      # cumulative retrieved fact ids
        self._by_id = {p.id: p for p in memory.points}
        # Live agent conversation state — `step` advances it one round; the
        # manual tool commands ALSO append to it, so you can steer by hand
        # and then hand control back to the algo. Checkpoints snapshot it so
        # you can rewind to a step and try a different branch.
        self.messages: List[Dict[str, Any]] = [
            {"role": "user", "content": f"Question: {question}"}
        ]
        self.checkpoints: Dict[str, Any] = {}
        self.client = None               # set in _run
        self.atools: List[dict] = []     # anthropic tool schemas, set in _run
        self._tuid = 0                   # synthetic tool_use id counter

    # --- helpers ---
    def _mark(self, fid: str) -> str:
        return f"{fid} \033[92m✓GOLD\033[0m" if fid in self.gold else fid

    def _track(self, ids) -> None:
        for i in ids or []:
            if isinstance(i, str):
                self.seen.add(i)

    def _record(self, name: str, inp: dict, result_obj: Any) -> None:
        """Append a (synthetic) tool call + its result to the live agent
        conversation, so a manually-triggered tool becomes part of the
        context the algo sees on the next `step`. Lets you steer by hand
        then let the algo continue from there."""
        self._tuid += 1
        tid = f"manual_{self._tuid}"
        self.messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tid, "name": name, "input": inp}]})
        body = (json.dumps(result_obj) if not isinstance(result_obj, str)
                else result_obj)
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": body}]})

    def _recall(self) -> float:
        if not self.gold:
            return 0.0
        return len(self.seen & set(self.gold)) / len(self.gold)

    # --- MCP plumbing ---
    async def _call(self, session, name: str, **kw):
        r = await session.call_tool(name, kw)
        txt = r.content[0].text if r.content else "null"
        try:
            return json.loads(txt)
        except (ValueError, TypeError):
            return txt

    # --- command handlers ---
    def cmd_q(self, *_):
        print(f"\n  question : {self.question}")
        print(f"  gold ans : {self.gold_answer}")
        print(f"  category : {self.category}   sample : {self.sample}")
        print(f"  gold ids : {self.gold}")

    def cmd_gold(self, *_):
        print()
        for g in self.gold:
            p = self._by_id.get(g)
            txt = p.content if p else "<not in memory!>"
            print(f"  \033[92m{g}\033[0m: {txt}")

    def cmd_show(self, arg: str):
        fid = arg.strip()
        p = self._by_id.get(fid)
        if not p:
            print(f"  no such id: {fid}")
            return
        print(f"\n  {self._mark(fid)}: {p.content}")
        # walk ±2 on the sequence chain
        nxt = {q.sequence_prev: q.id for q in self.memory.points
               if q.sequence_prev}
        # backward
        cur, back = fid, []
        for _ in range(2):
            pp = self._by_id.get(cur)
            cur = pp.sequence_prev if pp else None
            if not cur:
                break
            back.append(cur)
        cur, fwd = fid, []
        for _ in range(2):
            cur = nxt.get(cur)
            if not cur:
                break
            fwd.append(cur)
        for nid in reversed(back):
            print(f"    prev {self._mark(nid)}: {self._by_id[nid].content[:80]}")
        for nid in fwd:
            print(f"    next {self._mark(nid)}: {self._by_id[nid].content[:80]}")

    def cmd_recall(self, *_):
        hits = sorted(self.seen & set(self.gold))
        print(f"\n  cumulative recall = {self._recall():.3f}  "
              f"hits={hits}  gold={self.gold}  (seen {len(self.seen)} ids)")

    def cmd_reset(self, *_):
        self.seen.clear()
        print("  retrieved-id set cleared.")

    def cmd_answer(self, arg: str):
        from benchmarks.locomo.eval import official_score
        f1 = official_score(arg.strip(), self.gold_answer, self.category)
        print(f"\n  pred : {arg.strip()!r}")
        print(f"  gold : {self.gold_answer!r}")
        print(f"  token-F1 = {f1:.3f}")

    async def cmd_tools(self, session, *_):
        listed = await session.list_tools()
        print("\n  " + "  ".join(sorted(t.name for t in listed.tools)))

    async def cmd_presearch(self, session, arg: str):
        queries = [q.strip() for q in arg.split("|") if q.strip()]
        if not queries:
            print("  usage: presearch <q1> | <q2> | ...")
            return
        out = await self._call(session, "presearch", queries=queries,
                               k_per_query=5)
        self._record("presearch", {"queries": queries, "k_per_query": 5}, out)
        for r in out.get("results", []):
            print(f"\n  Q: {r['query']!r}")
            for rank, h in enumerate(r.get("hits", []), 1):
                gid = self._mark(h["id"])
                print(f"    {rank}. {gid}  score={h['score']:.3f}  "
                      f"{h['content'][:70]}")
            self._track(h["id"] for h in r.get("hits", []))
        # presearch hits don't enter fact_ids_cumulative in the real agent,
        # but for analysis we track whether the gold is reachable by phrasing.
        self.cmd_recall()

    async def cmd_walk(self, session, arg: str):
        query = arg.strip()
        if not query:
            print("  usage: walk <query>")
            return
        out = await self._call(session, "walk_start", query=query)
        self._record("walk_start", {"query": query}, out)
        print(f"\n  walk_start({query!r})")
        print(f"    stages_run={out.get('stages_run')}  "
              f"sigma_path={out.get('sigma_path')}  "
              f"drifted={out.get('drifted')}  n_relevant={out.get('n_relevant')}")
        rc = out.get("relevant_collected") or []
        print(f"    relevant_collected ({len(rc)}):")
        for e in rc[:12]:
            print(f"      {self._mark(e.get('id',''))} "
                  f"[{e.get('relevance','')}] {e.get('content','')[:64]}")
        nb = out.get("neighbor_possibilities") or []
        if nb:
            print(f"    neighbor_possibilities ({len(nb)}):")
            for e in nb:
                print(f"      {self._mark(e.get('id',''))} "
                      f"{e.get('content','')[:64]}")
        cum = out.get("fact_ids_cumulative") or []
        gold_in_cum = [g for g in self.gold if g in cum]
        print(f"    fact_ids_cumulative: {len(cum)} ids  "
              f"gold∈cum={gold_in_cum or '∅'}")
        self._track(cum)
        self.cmd_recall()

    async def cmd_scoped(self, session, arg: str):
        # syntax: <query> :: tag1,tag2 [kb]
        kb = arg.strip().endswith(" kb")
        body = arg[:-3] if kb else arg
        if "::" in body:
            q, tagpart = body.split("::", 1)
        else:
            q, tagpart = body, ""
        tags = [t.strip() for t in tagpart.split(",") if t.strip()]
        out = await self._call(session, "scoped_answer", query=q.strip(),
                               tags=tags, knowledge_base=kb)
        self._record("scoped_answer",
                     {"query": q.strip(), "tags": tags, "knowledge_base": kb}, out)
        print(f"\n  scoped_answer(tags={tags}, kb={kb})")
        print(f"    scoped_answer: {out.get('scoped_answer','')[:160]}")
        for e in (out.get("scoped_evidence") or [])[:8]:
            print(f"      {self._mark(e.get('id',''))} {e.get('content','')[:64]}")
            self._track([e.get("id")])
        if kb:
            print(f"    global_answer: {out.get('global_answer','')[:160]}")
            for e in (out.get("global_evidence") or [])[:8]:
                print(f"      {self._mark(e.get('id',''))} "
                      f"{e.get('content','')[:64]}")
                self._track([e.get("id")])
        self.cmd_recall()

    async def cmd_clues(self, session, arg: str):
        q = arg.strip() or self.question
        out = await self._call(session, "clue_search", question=q,
                               k_per_clue=3, n_clues=6)
        self._record("clue_search", {"question": q}, out)
        print(f"\n  clue_search({q!r})")
        print("  generated clues (evidence-register, spanning answers):")
        for c in out.get("clues", []):
            print(f"    · {c}")
        print("  per-clue hits:")
        for r in out.get("results", []):
            for h in r.get("hits", []):
                print(f"      [{self._mark(h['id'])}] from clue {r['clue'][:40]!r}: "
                      f"{h['content'][:56]}")
            self._track(h["id"] for h in r.get("hits", []))
        merged = out.get("merged", [])
        self._track(h["id"] for h in merged)        # incl. lineage-bridge nbrs
        nb = out.get("neighbor_possibilities") or []
        if nb:
            print(f"  lineage-bridge neighbours ({len(nb)}): "
                  + ", ".join(self._mark(n['id']) for n in nb[:12]))
        gold_in = [g for g in self.gold if g in {h['id'] for h in merged}]
        print(f"  merged unique hits: {len(merged)}  gold∈merged={gold_in or '∅'}")
        self.cmd_recall()

    async def cmd_retrieve(self, session, arg: str):
        out = await self._call(session, "retrieve", query=arg.strip(), k=7)
        self._record("retrieve", {"query": arg.strip(), "k": 7}, out)
        for rank, h in enumerate(out or [], 1):
            print(f"    {rank}. {self._mark(h['id'])}  score={h['score']:.3f}  "
                  f"{h['content'][:70]}")
        self._track(h["id"] for h in (out or []))
        self.cmd_recall()

    async def cmd_auto(self, session, *_):
        from benchmarks.locomo.mcp_meta_agent import McpMetaAgent
        from benchmarks.locomo.eval import official_score
        print("\n  running the full autonomous agent ...")
        agent = McpMetaAgent()
        # agent.answer() calls asyncio.run() internally; run it in a worker
        # thread so it gets its own event loop (we're already inside one).
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None, agent.answer, self.memory, self.question)
        pred = res.get("answer", "")
        for t in res.get("trace") or []:
            act = t.get("action")
            if act == "tool":
                print(f"    round {t['round']}: TOOL {t['name']}({t.get('input')})")
            elif act and act.startswith("floor"):
                print(f"    round {t['round']}: FLOOR redirect "
                      f"(presearch={t.get('presearch_count')}, "
                      f"walks={t.get('distinct_walks')})")
            elif act in ("final", "final_tool"):
                print(f"    round {t['round']}: FINAL -> {t.get('text','')!r}")
        self._track(res.get("retrieved_ids"))
        f1 = official_score(pred, self.gold_answer, self.category)
        print(f"\n  PREDICTED: {pred!r}")
        print(f"  token-F1 = {f1:.3f}   ", end="")
        self.cmd_recall()

    async def cmd_step(self, session, *_):
        """Let the ALGO trigger the next step : one autonomous agent round.
        The model sees the live conversation (including any tools you ran by
        hand), picks the next tool(s) or answers. Executes them, appends to
        the conversation, and reports — paused for your inspection."""
        from benchmarks.locomo.mcp_meta_agent import (
            AGENT_SYSTEM, _create_with_retry, _FINAL_ANSWER_TOOL,
        )
        from benchmarks.locomo.eval import official_score
        resp = _create_with_retry(
            self.client, model=os.environ.get("CLAUDE_MODEL",
                                               "claude-haiku-4-5-20251001"),
            max_tokens=256, temperature=0, system=AGENT_SYSTEM,
            tools=self.atools, messages=self.messages,
        )
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        texts = [b.text for b in resp.content if b.type == "text"]
        if texts:
            print(f"  \033[95m[model]\033[0m {' '.join(texts)[:300]}")
        if not tool_uses:
            print("  → model produced NO tool call (would answer here).")
            return
        self.messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:
            print(f"  \033[93m→ ALGO picks:\033[0m {tu.name}({json.dumps(tu.input)[:160]})")
            if tu.name == "final_answer":
                val = str((tu.input or {}).get("value", "")).strip()
                f1 = official_score(val, self.gold_answer, self.category)
                print(f"    FINAL answer: {val!r}   token-F1={f1:.3f}")
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": "ok"})
                continue
            obj = await self._call(session, tu.name, **(tu.input or {}))
            # surface the salient bits + track ids / gold
            self._track(obj.get("fact_ids_cumulative") if isinstance(obj, dict) else None)
            if isinstance(obj, dict):
                for key in ("relevant_collected", "neighbor_possibilities"):
                    for e in (obj.get(key) or [])[:6]:
                        if isinstance(e, dict) and e.get("id"):
                            self.seen.add(e["id"])
                cum = obj.get("fact_ids_cumulative")
                if cum is not None:
                    gold_in = [g for g in self.gold if g in cum]
                    print(f"    result: {len(cum)} fact ids, "
                          f"drifted={obj.get('drifted')}, "
                          f"gold∈cum={gold_in or '∅'}")
                elif obj.get("results"):  # presearch
                    for r in obj["results"]:
                        hset = {h['id'] for h in r.get('hits', [])}
                        gold_in = [g for g in self.gold if g in hset]
                        print(f"    presearch {r['query']!r}: "
                              f"gold∈hits={gold_in or '∅'}")
                        self._track(h['id'] for h in r.get('hits', []))
            body = json.dumps(obj) if not isinstance(obj, str) else obj
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": body})
        self.messages.append({"role": "user", "content": results})
        self.cmd_recall()

    def cmd_checkpoint(self, arg: str):
        import copy
        name = arg.strip() or "default"
        self.checkpoints[name] = {
            "messages": copy.deepcopy(self.messages),
            "seen": set(self.seen),
            "tuid": self._tuid,
        }
        print(f"  checkpoint {name!r} saved "
              f"(at {len(self.messages)} messages, recall {self._recall():.3f}).")

    def cmd_restore(self, arg: str):
        import copy
        name = arg.strip() or "default"
        cp = self.checkpoints.get(name)
        if not cp:
            print(f"  no checkpoint {name!r} (have: {list(self.checkpoints)})")
            return
        self.messages = copy.deepcopy(cp["messages"])
        self.seen = set(cp["seen"])
        self._tuid = cp["tuid"]
        print(f"  restored {name!r} → {len(self.messages)} messages, "
              f"recall {self._recall():.3f}. Continue with `step` or manual tools.")

    def cmd_say(self, arg: str):
        """Inject a user nudge into the conversation (e.g. a hint to test)."""
        self.messages.append({"role": "user", "content": arg})
        print(f"  injected user message: {arg!r}")

    def cmd_msgs(self, *_):
        print(f"\n  {len(self.messages)} messages, checkpoints={list(self.checkpoints)}")
        for i, m in enumerate(self.messages):
            c = m["content"]
            if isinstance(c, str):
                tag = c[:80]
            else:
                tag = ", ".join(b.get("type", "?") if isinstance(b, dict)
                                else getattr(b, "type", "?") for b in c)
            print(f"    [{i}] {m['role']:9s} {tag}")

    # --- SQL journal workflow inspectors (no-op without a journal) ---
    def _need_journal(self) -> bool:
        if getattr(self.memory, "journal", None) is None:
            print("  (no journal on this memory — SQL triggers are off)")
            return False
        return True

    def cmd_journal(self, *_):
        """Snapshot the append-only journal : row counts per table."""
        if not self._need_journal():
            return
        j = self.memory.journal
        n_ret, n_acc = j.counts() if hasattr(j, "counts") else (None, None)
        print(f"\n  journal @ {j.path}")
        print(f"    retrievals={n_ret}  access_events={n_acc}")
        print(f"    hops -> {j.hop_target_counts()[:5]}")
        print(f"    frequent_paths(>=2) -> "
              f"{[(r['signature'], r['freq']) for r in j.frequent_paths(min_len=4, min_freq=2)][:5]}")
        print(f"    collisions -> {[e['kind'] for e in j.collision_events()][-5:]}")

    def cmd_paths(self, arg: str):
        """Frequently-travelled paths (the Chasles candidates by query).
        Optional arg: min_freq (default 2)."""
        if not self._need_journal():
            return
        mf = int(arg) if arg.strip().isdigit() else 2
        cands = self.memory.chasles_path_candidates(min_freq=mf)
        print(f"\n  chasles_path_candidates(min_freq={mf}) -> {len(cands)}")
        for c in cands[:10]:
            print(f"    {c['signature']}  freq={c['freq']}  "
                  f"absorb={c['intermediates']}")

    def cmd_chasles(self, *_):
        """Fire the Chasles compression (SQL-driven when a journal is present)
        and show the resulting shortcut events."""
        events = self.memory.compress_chasles()
        print(f"\n  compress_chasles -> {len(events)} shortcut(s)")
        for e in events:
            print(f"    child={e['child_id']}  absorbed={e['parent_ids']}  "
                  f"anchors={e['anchor_ids']}")

    def cmd_tags(self, arg: str):
        """Hierarchical tag index. Arg: a tag to run the ancestor query on;
        empty = print the depth-ordered glossary."""
        if not self._need_journal():
            return
        self.memory.reindex_tags()
        if arg.strip():
            hits = [p.id for p in self.memory.tag_scoped(arg.strip())]
            print(f"\n  tag_scoped({arg.strip()!r}) -> {hits[:20]}")
        else:
            gloss = self.memory.tag_glossary_sql()
            print(f"\n  glossary ({len(gloss)} tags, depth-ordered):")
            for t in gloss[:30]:
                print(f"    {t}")

    def cmd_help(self, *_):
        print(__doc__[__doc__.index("REPL commands"):])

    # --- dispatch ---
    async def dispatch(self, session, line: str) -> bool:
        line = line.strip()
        if not line:
            return True
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
        if cmd in ("quit", "exit", "q!"):
            return False
        sync = {
            "q": self.cmd_q, "gold": self.cmd_gold, "show": self.cmd_show,
            "recall": self.cmd_recall, "reset": self.cmd_reset,
            "answer": self.cmd_answer, "help": self.cmd_help,
            "checkpoint": self.cmd_checkpoint, "cp": self.cmd_checkpoint,
            "restore": self.cmd_restore, "say": self.cmd_say,
            "msgs": self.cmd_msgs,
            # SQL journal workflow inspectors (new actions)
            "journal": self.cmd_journal, "paths": self.cmd_paths,
            "chasles": self.cmd_chasles, "tags": self.cmd_tags,
        }
        asyncc = {
            "tools": self.cmd_tools, "presearch": self.cmd_presearch,
            "walk": self.cmd_walk, "scoped": self.cmd_scoped,
            "retrieve": self.cmd_retrieve, "auto": self.cmd_auto,
            "step": self.cmd_step, "clues": self.cmd_clues,
        }
        try:
            if cmd in sync:
                sync[cmd](arg)
            elif cmd in asyncc:
                await asyncc[cmd](session, arg)
            else:
                print(f"  unknown command: {cmd!r} (try 'help')")
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  [error] {exc}")
            traceback.print_exc()
        return True


async def _run(dbg: QADebugger, script: Optional[str]):
    from mcp.shared.memory import create_connected_server_and_client_session
    from metacog.mcp_server import build_app

    from benchmarks.locomo.mcp_meta_agent import (
        _resolve_client, _mcp_tools_to_anthropic, _FINAL_ANSWER_TOOL,
    )
    app = build_app(memory=dbg.memory)
    async with create_connected_server_and_client_session(app) as session:
        await session.initialize()
        # Wire the autonomous-`step` machinery : same client, system prompt
        # and full tool surface the real agent uses.
        dbg.client = _resolve_client(None)
        listed = await session.list_tools()
        dbg.atools = _mcp_tools_to_anthropic(listed.tools) + [_FINAL_ANSWER_TOOL]
        dbg.cmd_q()
        if script:
            # accept ';' and newline separated commands
            cmds = [c for chunk in script.split("\n") for c in chunk.split(";")]
            for line in cmds:
                line = line.strip()
                if not line:
                    continue
                print(f"\n\033[96mqa> {line}\033[0m")
                if not await dbg.dispatch(session, line):
                    break
            return
        # interactive
        loop = asyncio.get_event_loop()
        print("\n  (type 'help' for commands, 'quit' to exit)")
        while True:
            try:
                line = await loop.run_in_executor(None, input, "\nqa> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not await dbg.dispatch(session, line):
                break


def main() -> None:
    ap = argparse.ArgumentParser(description="Step-by-step LoCoMo QA debugger")
    ap.add_argument("--probe", choices=list(_PROBES))
    ap.add_argument("--sample", default="conv-41")
    ap.add_argument("--question")
    ap.add_argument("--gold", default="", help="comma-separated dia_ids")
    ap.add_argument("--answer", default="")
    ap.add_argument("--category", type=int, default=3)
    ap.add_argument("--nsess", type=int, default=0,
                    help="sessions to ingest (0 = whole conversation)")
    ap.add_argument("--script", help="run ';'-separated commands then exit")
    ap.add_argument("--script-file",
                    help="run commands from a file (one per line) then exit")
    ap.add_argument("--rebuild", action="store_true",
                    help="force re-embedding the memory cache")
    args = ap.parse_args()

    if args.probe:
        p = _PROBES[args.probe]
        sample, question = p["sample"], p["question"]
        gold, answer, cat = p["gold"], p["answer"], p["category"]
    else:
        if not args.question:
            ap.error("provide --probe or --question")
        sample, question = args.sample, args.question
        gold = [g.strip() for g in args.gold.split(",") if g.strip()]
        answer, cat = args.answer, args.category

    script = args.script
    if args.script_file:
        with open(args.script_file) as fh:
            script = fh.read()

    mem = build_or_load_memory(sample, args.nsess, args.rebuild)
    dbg = QADebugger(mem, question, gold, answer, cat, sample)
    asyncio.run(_run(dbg, script))


if __name__ == "__main__":
    main()
